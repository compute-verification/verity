"""vLLM's own switches, and the one place the integration writes them into the process environment.

The pins are ``setdefault``s: a value the operator exported wins, except a `FIXED` one's, which `pin` refuses. vLLM, torch and
``huggingface_hub`` read several of these once, at import or at engine construction, so each caller pins before it imports them.
A ``TargetProfile``'s ``env()`` and the compiled-engine cache switches are overrides, written with plain assignment.

Stdlib only, and nothing from ``verity_vllm``: every layer may call it (the config it needs comes in as arguments).
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping
from typing import Any

__all__ = ["ENGINE", "EXPORT", "INDUCTOR_DIR_MARK", "apply_compile_fresh", "apply_engine", "apply_target", "compile_dirs",
           "hide_devices", "own_compile_dirs", "pin", "require_inductor", "served_defaults", "snapshot", "step_timeout"]

#: The engine's execution profile, shared by production and replay: vLLM's defaults but for these, each
#: with its reason in `REASONS`.  Batch invariance is the workload's execution label's (`engine.served.EXECUTIONS`), not the profile's.
ENGINE = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
    # At torch's default a reduction kernel's tiling is a per-process runtime choice.  torch reads the variable once, when
    # torch._inductor.config is imported, so it is in the environment before that import in every process that compiles: this
    # one, and each TP rank, which inherits it when vLLM spawns it.
    "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK": "0",
}
#: Why `ENGINE` sets each switch (a served Commit's record carries it: `engine.served`).
REASONS = {
    "VLLM_USE_V2_MODEL_RUNNER": "trace replay requires Model Runner V2, which vLLM selects by default only for "
                                "DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES (vllm/config/vllm.py)",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "the engine core runs in-process, so the recorder reads the scheduler's step trace and replay "
                                      "intercepts the model runner; it changes the process topology, not the kernels",
    "VLLM_USE_FLASHINFER_SAMPLER": "the FlashInfer top-k/top-p sampler JIT-compiles at warm-up (ninja and an nvcc matching torch's CUDA; "
                                   "the pods have neither); the V2 sampler never uses it for greedy or explicitly seeded requests "
                                   "(sampler.py: any_greedy / any_explicit_seed), so the torch path W3 re-derives is the one that runs",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "a shared expert runs on the main stream, in the order a Build (EXPORT) gives it; otherwise vLLM "
                                          "moves it to an aux stream at <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD (256) tokens and the "
                                          "log's order is no longer the execution order operand_provenance (G5) takes it to be; costs the "
                                          "shared/routed overlap there, on shared-expert models only; value- and Program-neutral",
    "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK": "at 1, torch's default, Inductor adds a halved R0_BLOCK tiling to a reduction kernel at its "
                                          "first launch in each process, times both and keeps the faster, without caching the pick; the "
                                          "tilings sum in different orders, so two processes of one engine can differ in a bit.  At 0 "
                                          "every process runs the kernel's static tiling",
}
#: The `ENGINE` switches an operator's export can't change (`pin` refuses another inherited value), each with the
#: ``torch._inductor.config`` attribute it sets and the value that attribute reads under the pin (`require_inductor`).
FIXED = {"TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK": ("dynamic_scale_rblock", False)}

#: The meta-device instantiation of the pinned model class (``program/frontend/vllm_meta.py``).  Batch invariance is the Build
#: target's (``TargetProfile.env()``, applied before vLLM is imported), which is the workload's execution label's, not the profile's.
EXPORT = {
    "VLLM_USE_LAYERNAME": "0",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
    "HF_HUB_OFFLINE": "1",
    "VLLM_LOGGING_LEVEL": "WARNING",
    "TOKENIZERS_PARALLELISM": "false",
}

#: No compile cache of any layer (vLLM's, Inductor's FX graphs, AOTAutograd's): every compiled engine generates its wrappers
#: in-process, so hooks installed before the build are in them (``served_compiled_source``), and an engine compared with
#: one compiles the same way. Inductor's autotuning results (TORCHINDUCTOR_CACHE_DIR) stay shared.
COMPILE_FRESH = {"VLLM_DISABLE_COMPILE_CACHE": "1", "TORCHINDUCTOR_FX_GRAPH_CACHE": "0", "TORCHINDUCTOR_AUTOGRAD_CACHE": "0"}

#: Set by whoever gave this process its own TORCHINDUCTOR_CACHE_DIR (``prepare_compiled_process_cache``,
#: ``ops/served_commit.sh``).
INDUCTOR_DIR_MARK = "VERITY_INDUCTOR_DIR_PER_PROCESS"

#: The switches a served engine leaves at vLLM's defaults; a capture of record runs with none of them set, so the kernels it compares
#: are the ones vLLM selects by default whatever the process inherited.
SERVED_DEFAULTS = ("VLLM_BATCH_INVARIANT", "VLLM_USE_DEEP_GEMM", "VLLM_USE_DEEP_GEMM_E8M0", "VLLM_MOE_USE_DEEP_GEMM",
                   "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "VLLM_USE_FUSED_MOE_GROUPED_TOPK", "VLLM_TUNED_CONFIG_FOLDER", "NCCL_P2P_DISABLE")


def served_defaults() -> dict[str, str]:
    """Clear `SERVED_DEFAULTS` from the process environment (before vLLM is imported); returns the values cleared."""
    return {k: os.environ.pop(k) for k in SERVED_DEFAULTS if k in os.environ}


def pin(pins: Mapping[str, str]) -> dict[str, str]:
    for k, v in pins.items():
        if k in FIXED and os.environ.get(k, v) != v:
            raise RuntimeError(f"fail-closed: this process inherited {k}={os.environ[k]!r}, and an engine runs only at {k}={v} "
                               f"(engine.env.ENGINE: {REASONS[k]}); unset it")
    for k, v in pins.items():
        os.environ.setdefault(k, v)
    return {k: os.environ[k] for k in pins}


def require_inductor(engine_kwargs: Mapping[str, Any]) -> None:
    """Before ``LLM(**engine_kwargs)`` in this process: refuse an engine whose Inductor reads a `FIXED` switch otherwise than `ENGINE`
    pins it, because ``torch._inductor.config`` was imported before the pin (it read the variable then; a forked TP rank keeps
    what it read), or because the engine's ``compilation_config.inductor_compile_config`` sets the attribute over it."""
    cfg = sys.modules.get("torch._inductor.config")
    cc = engine_kwargs.get("compilation_config")
    over = cc.get("inductor_compile_config") if isinstance(cc, Mapping) else getattr(cc, "inductor_compile_config", None)
    for k, (attr, want) in FIXED.items():
        if cfg is not None and getattr(cfg, attr, want) != want:
            raise RuntimeError(f"fail-closed: torch._inductor.config was imported before {k}={ENGINE[k]} was pinned, and read "
                               f"{attr}={getattr(cfg, attr)!r}: pin the engine environment (engine_profile.apply_env) before "
                               "anything imports torch._dynamo, torch._inductor or vllm")
        if isinstance(over, Mapping) and attr in over and over[attr] != want:
            raise RuntimeError(f"fail-closed: compilation_config.inductor_compile_config sets {attr}={over[attr]!r}, over "
                               f"{k}={ENGINE[k]} (engine.env.ENGINE); leave it out")


def apply_engine(hf_home: str | None) -> dict[str, str]:
    """``ENGINE``, the checkpoint store's ``HF_HOME`` and ``HF_HUB_OFFLINE=1``; returns the ``ENGINE`` values in force, which an
    engine records as its ``env_flags``.  An inherited `FIXED` switch at another value is refused (`pin`)."""
    got = pin(ENGINE)
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return got


def step_timeout(seconds: float) -> dict[str, str]:
    """vLLM's multiprocess executor waits ``VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`` (300 by default) for each engine step's RPC and
    fails the run past it, while an observed or committed TP step of a long prefill (the observer, its snapshots, the Commit's
    taps) takes minutes: a stage gives a step its own RPC bound.  A liveness bound, not a served switch: nothing that runs
    changes, so it stays out of the profile's flags.  A pin: an operator's export wins."""
    return pin({"VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": str(int(seconds))})


def apply_target(env: Mapping[str, str]) -> dict[str, str]:
    """A ``TargetProfile.env()``: the profile's values replace whatever the process had."""
    os.environ.update(env)
    return {k: os.environ[k] for k in env}


def apply_compile_fresh() -> dict[str, str]:
    """``COMPILE_FRESH`` as overrides, before vLLM and torch read them."""
    return apply_target(COMPILE_FRESH)


def own_compile_dirs(root: str) -> dict[str, str]:
    """This process's own Inductor output code and vLLM compile dir under ``root``, with ``COMPILE_FRESH``, before vLLM and torch read
    them: a served Build reads the compile artifact of the process that serves (``pipeline.served_match``)."""
    return apply_target({"TORCHINDUCTOR_CACHE_DIR": f"{root}/inductor", "VLLM_CACHE_ROOT": f"{root}/vllm_cache", INDUCTOR_DIR_MARK: "1",
                         **COMPILE_FRESH})


def compile_dirs() -> dict[str, str] | None:
    """The Inductor output-code dir and vLLM cache root this process was given as its own (``own_compile_dirs``, or exported with
    ``INDUCTOR_DIR_MARK`` by ``ops/served_commit.sh``), else None."""
    ind, root = os.environ.get("TORCHINDUCTOR_CACHE_DIR"), os.environ.get("VLLM_CACHE_ROOT")
    return {"inductor": ind, "vllm_cache": root} if os.environ.get(INDUCTOR_DIR_MARK) == "1" and ind and root else None


def hide_devices() -> None:
    """No CUDA device is visible to this process (a meta-device build on a GPU host), whatever it inherited.  Takes effect
    only before torch initialises CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def snapshot() -> dict[str, str]:
    return dict(os.environ)


def disable_compile_cache() -> None:
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"


def prepare_compiled_process_cache() -> dict[str, Any]:
    """Before a COMPILED engine is built: make this process's Inductor output code identifiable.  (1) A fresh,
    empty TORCHINDUCTOR_CACHE_DIR unless the caller already exported one it marks as per-process (ops/served_commit.sh) -- torch's default
    /tmp/torchinductor_<user> is shared by every process on the pod, so a pin read from it would digest other workloads' kernels; (2)
    VLLM_DISABLE_COMPILE_CACHE=1 -- a vLLM compile-cache hit replays another process's artifacts and emits no output code here (the pin
    would be null: a compiled record without positive evidence of compilation).  Costs one Inductor compilation per process (~70 s for
    Llama-3.2-1B), recorded in the header as compilation.compile_cache = disabled."""
    out: dict[str, Any] = {}
    if os.environ.get(INDUCTOR_DIR_MARK) != "1":
        d = tempfile.mkdtemp(prefix="verity_inductor_")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = d
        os.environ[INDUCTOR_DIR_MARK] = "1"
        out["inductor_cache_dir"] = d
    else:
        out["inductor_cache_dir"] = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    disable_compile_cache()
    out["compile_cache"] = "disabled"
    return out

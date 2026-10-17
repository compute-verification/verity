"""vLLM's own switches, and the one place the integration writes them into the process environment.

The pins are ``setdefault``s: a value the operator exported wins. vLLM, torch and ``huggingface_hub`` read several of
these once, at import or at engine construction, so each caller pins before it imports them. A ``TargetProfile``'s
``env()`` and the compiled-engine cache switches are overrides, written with plain assignment.

Stdlib only, and nothing from ``verity_vllm``: every layer may call it (the config it needs comes in as arguments).
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from typing import Any

#: The engine's execution profile (CONTRACT.md §2 / §2.1), shared by production and replay: vLLM's defaults but for these, each
#: with its reason in `REASONS`.  Batch invariance is the row's execution label's (`engine.served.EXECUTIONS`), not the profile's.
ENGINE = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
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
}

#: The meta-device instantiation of the pinned model class (``program/frontend/vllm_meta.py``).
EXPORT = {
    "VLLM_BATCH_INVARIANT": "1",
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
#: ``ops/compiled_commit.sh``).
INDUCTOR_DIR_MARK = "VERITY_INDUCTOR_DIR_PER_PROCESS"


def pin(pins: Mapping[str, str]) -> dict[str, str]:
    for k, v in pins.items():
        os.environ.setdefault(k, v)
    return {k: os.environ[k] for k in pins}


def apply_engine(hf_home: str | None) -> dict[str, str]:
    """``ENGINE``, the checkpoint store's ``HF_HOME`` and ``HF_HUB_OFFLINE=1``; returns the ``ENGINE`` values in force."""
    got = pin(ENGINE)
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return got


def apply_target(env: Mapping[str, str]) -> dict[str, str]:
    """A ``TargetProfile.env()``: the profile's values replace whatever the process had."""
    os.environ.update(env)
    return {k: os.environ[k] for k in env}


def apply_compile_fresh() -> dict[str, str]:
    """``COMPILE_FRESH`` as overrides, before vLLM and torch read them."""
    return apply_target(COMPILE_FRESH)


def hide_devices() -> None:
    """No CUDA device is visible to this process (a meta-device build on a GPU host), whatever it inherited.  Takes effect
    only before torch initialises CUDA."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def snapshot() -> dict[str, str]:
    return dict(os.environ)


def set_since(before: Mapping[str, str]) -> dict[str, str]:
    """The keys set or changed since ``before`` (a `snapshot`). vLLM's batch invariance writes cuBLAS' and NCCL's switches while it
    builds an engine, so a process that resets its environment between Commits on one engine (`commit_group`, `hot.serve`) puts
    these back."""
    return {k: v for k, v in os.environ.items() if before.get(k) != v}


def disable_compile_cache() -> None:
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"


def prepare_compiled_process_cache() -> dict[str, Any]:
    """Before a COMPILED engine is built: make this process's Inductor output code identifiable.  (1) A fresh,
    empty TORCHINDUCTOR_CACHE_DIR unless the caller already exported one it marks as per-process (compiled_commit.sh) -- torch's default
    /tmp/torchinductor_<user> is shared by every process on the pod, so a pin read from it would digest other rows' kernels; (2)
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

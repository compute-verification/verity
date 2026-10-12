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

#: The engine's execution profile (CONTRACT.md §2 / §2.1), shared by production and replay.
ENGINE = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    # The FlashInfer top-k/top-p sampler JIT-compiles at warm-up (needs ninja + nvcc matching the
    # cu130 torch; the pod has neither). Inert for our workloads: the V2 sampler never uses it for
    # greedy or explicitly seeded requests (sampler.py: any_greedy / any_explicit_seed), so the
    # torch path (apply_top_k_top_p + gumbel_sample) that W3 re-derives is the one that runs.
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    # A shared expert runs on the main stream, in the order a Build (`EXPORT`) gives it. Otherwise vLLM moves it to an aux stream
    # at <= VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD (256) tokens, and the log's order is no longer the execution order that
    # operand_provenance (G5) takes it to be. Costs the shared/routed overlap there, on shared-expert models only.
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
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

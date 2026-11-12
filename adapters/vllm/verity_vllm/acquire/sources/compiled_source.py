"""What a compiled engine (vLLM torch.compile + CUDA graphs, enforce_eager=False) was built with, read back from its configuration for
the Commit's record."""
from __future__ import annotations

from typing import Any

import torch

from verity_vllm import settings
from verity_vllm.engine import handle


def compile_settings(llm: Any = None, runner: Any = None) -> dict[str, Any]:
    """What is actually compiled / graphed in this engine: CompilationMode, CUDAGraphMode, capture sizes, splitting ops, inductor graph
    partition, enforce_eager.  Recorded for every attempt (control and treatment must agree)."""
    out: dict[str, Any] = {}
    try:
        vc = getattr(handle.engine_of(llm), "vllm_config", None)
        if vc is None and runner is not None:
            vc = getattr(runner, "vllm_config", None)
        if vc is None and llm is not None:
            try:
                vc = getattr(handle.model_runner(llm), "vllm_config", None)
            except RuntimeError:
                vc = None
        if vc is not None:
            cfg = vc.compilation_config
            out["compilation_mode"] = str(getattr(cfg, "mode", getattr(cfg, "level", None)))
            out["cudagraph_mode"] = str(getattr(cfg, "cudagraph_mode", None))
            sizes = getattr(cfg, "cudagraph_capture_sizes", None) or []
            out["cudagraph_capture_sizes"] = sorted(int(s) for s in list(sizes)[:256])
            out["max_cudagraph_capture_size"] = getattr(cfg, "max_cudagraph_capture_size", None)
            out["splitting_ops"] = list(getattr(cfg, "splitting_ops", None) or [])
            out["use_inductor_graph_partition"] = bool(getattr(cfg, "use_inductor_graph_partition", False))
            out["backend"] = str(getattr(cfg, "backend", None))
            out["custom_ops"] = list(getattr(cfg, "custom_ops", None) or [])
            # the Inductor config of the pieces: emulate_precision_casts = the compiled kernels perform the Program's bf16
            # roundings at every op boundary (default Inductor carries fp32 through fused regions, so a materialised boundary value
            # changes the computation); recorded and compared across arms
            out["inductor_compile_config"] = dict(getattr(cfg, "inductor_compile_config", None) or {})
            out["emulate_precision_casts"] = bool(out["inductor_compile_config"].get("emulate_precision_casts", False))
            out["enforce_eager"] = bool(getattr(vc.model_config, "enforce_eager", False))
            out["torch"] = torch.__version__
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:200]
    out["compile_cache_disabled"] = settings.raw("VLLM_DISABLE_COMPILE_CACHE") == "1"
    return out


__all__ = ["compile_settings"]

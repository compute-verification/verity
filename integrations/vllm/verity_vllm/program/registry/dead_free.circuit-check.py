"""circuit-check's bindings of `verity_vllm.program.registry.dead_free`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """The RMSNorm at one warp's worth of columns and at Llama-3.2-1B's hidden size (8 warps' chunks), and the selects at V = 16."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dead_free as DF

    return [
        ("RMSNormTriton_v5", lambda: bind(DF.RMSNormTritonV5, N=64, EPS=1e-5)),
        ("RMSNormTriton_v5-wide", lambda: bind(DF.RMSNormTritonV5, N=2048, EPS=1e-5)),
        ("TokenSelect_v3", lambda: bind(DF.TokenSelectV3, V=16)),
        ("TopPMask_v3", lambda: bind(DF.TopPMaskV3, V=16)),
        ("GumbelSelectF32GivenGreedy_v2", lambda: bind(DF.GumbelSelectF32GivenGreedyV2, V=16)),
        ("GumbelTopPTokenSelectSharedGreedy_v2", lambda: bind(DF.GumbelTopPTokenSelectSharedGreedyV2, V=16)),
    ]

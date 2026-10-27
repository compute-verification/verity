"""circuit-check's bindings of `verity.ml.definitions.registry.rmsnorm_fused_split`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _rmsnorm_fused_split_roots() -> list[Any]:
    """The norm after Llama-3.2-1B's down_proj (N = 2048, EPS = 1e-5) as its four parts, each thread an FFMA past its first
    square, and their combine, whose r is in part order; and N = 64 as two parts of one stride."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import rmsnorm_fused_split as RS

    return [
        *(bind(RS.RMSNormFusedCudaPart, N=N, PARTS=k, PART=g) for N, k in ((2048, 4), (64, 2)) for g in range(k)),
        bind(RS.RMSNormFusedCudaCombine, N=2048, EPS=1e-5, PARTS=4),
        bind(RS.RMSNormFusedCudaCombine, N=64, EPS=1e-5, PARTS=2),
    ]

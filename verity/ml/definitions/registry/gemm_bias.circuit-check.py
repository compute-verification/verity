"""circuit-check's bindings of `verity.ml.definitions.registry.gemm_bias`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The biased linear at K = 32, N = 16 (the frontend's `gemm_bias` instance), and on the Ampere k16 step (`gemm_bias_dot`'s)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import gemm_bias as GB
    from verity.ml.definitions.registry import prims as P

    return [
        bind(GB.GemmBias, K=32, N=16),
        bind(GB.GemmBiasV2, K=32, N=16, DOT=P.AmpereBF16TcDot16),
    ]

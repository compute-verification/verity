"""circuit-check's bindings of `verity.ml.circuits.ops.moe`'s Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    """The select over 3 and 128 experts, the coordinates on both steps over 3 experts at K = 32, and the GEMMs at N = 3."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import moe as M
    from verity.ml.circuits.ops import prims as P

    A, H = P.AmpereBF16TcDot16, P.HopperBF16WgmmaDot16
    return [
        M.select_bf16(3),
        M.select_bf16(128),
        bind(M.MoePairCoordinate, E=3, K=32, DOT=A),
        bind(M.MoePairCoordinate, E=3, K=32, DOT=H),
        bind(M.MoePairCoordinateW, E=3, K=32, DOT=H),
        bind(M.MoePairGemm, E=3, K=16, N=3, DOT=H),
        bind(M.MoePairGemmW, E=3, K=16, N=3, DOT=H),
    ]

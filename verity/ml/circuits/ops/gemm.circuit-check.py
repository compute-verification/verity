"""circuit-check's bindings of `verity.ml.circuits.ops.gemm`'s Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import gemm as G
    from verity.ml.circuits.ops import prims as P

    A, H = P.AmpereBF16TcDot16, P.HopperBF16WgmmaDot16
    return [
        bind(G.DotBf16, K=64, DOT=A),
        bind(G.DotBf16, K=48, DOT=H),
        bind(G.GemmCoordinate, K=64, DOT=A),
        bind(G.GemmCoordinate, K=64, DOT=H),
        bind(G.Gemm, K=32, N=3, DOT=A),
        bind(G.GemmCoordinateE4m3, K=64, **G.GEMM_SM120_E4M3),
        bind(G.GemmCoordinateNvf4, K=128, **G.GEMM_SM120_NVF4),
        bind(G.GemmCoordinateMxf4, K=128, **G.GEMM_SM120_MXF4),
    ]

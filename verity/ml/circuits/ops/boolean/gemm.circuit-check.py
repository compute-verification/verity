"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.gemm`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The Boolean BF16 coordinate and dot on both steps at K = 64, the GEMM at K = 32, N = 3, the E4M3 coordinate on the three
    FP8 steps at K = 64, the NVFP4 and MXFP4 coordinates at K = 128, the s8 accumulator and scaled coordinate at K = 64, and the
    s8 linear row at K = 64, N = 3."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG

    return [
        bind(BG.GemmCoordinate, K=64, **BG.GEMM_AMPERE),
        bind(BG.GemmCoordinate, K=64, **BG.GEMM_HOPPER),
        bind(BG.DotBf16, K=64, **BG.GEMM_AMPERE),
        bind(BG.DotBf16, K=64, **BG.GEMM_HOPPER),
        bind(BG.Gemm, K=32, N=3, **BG.GEMM_AMPERE),
        bind(BG.Gemm, K=32, N=3, **BG.GEMM_HOPPER),
        bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_HOPPER_E4M3),
        bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_SM120_E4M3),
        bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_HOPPER_E4M3_MMAV2),
        bind(BG.GemmCoordinateNvf4, K=128, **BG.GEMM_SM120_NVF4),
        bind(BG.GemmCoordinateMxf4, K=128, **BG.GEMM_SM120_MXF4),
        bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8),
        bind(BG.ScaledMmS8Coordinate, K=64, **BG.GEMM_SM120_S8),
        bind(BG.ScaledMmS8, K=64, N=3, **BG.GEMM_SM120_S8),
    ]

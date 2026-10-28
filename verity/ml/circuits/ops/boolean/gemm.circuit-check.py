"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.gemm`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The Boolean BF16 coordinate and dot on both steps at K = 64, the padded coordinate on Hopper's step at T = 1 and 17 of
    K = 32 and the PV row on it at T = 1, 8 and 32 of K = 32 with D = 64 (and padded to T = 32 with its key count an input),
    the GEMM at K = 32, N = 3, the E4M3 coordinate on
    the three FP8 steps at K = 64, the NVFP4 and MXFP4 coordinates at K = 128, the s8 accumulator and scaled coordinate at
    K = 64, the scaled coordinate on two rows an operand at K = 64 (and with the residual add in its epilogue), and the
    scaled s8 linear at K = 64, N = 3."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG

    return [
        bind(BG.GemmCoordinate, K=64, **BG.GEMM_AMPERE),
        bind(BG.GemmCoordinate, K=64, **BG.GEMM_HOPPER),
        bind(BG.DotBf16, K=64, **BG.GEMM_AMPERE),
        bind(BG.DotBf16, K=64, **BG.GEMM_HOPPER),
        bind(BG.GemmCoordinatePad, T=1, K=32, **BG.GEMM_HOPPER),
        bind(BG.GemmCoordinatePad, T=17, K=32, **BG.GEMM_HOPPER),
        *(bind(BG.PvRow, T=T, K=32, D=64, **BG.GEMM_HOPPER) for T in (1, 8, 32)),
        bind(BG.PvRowPad, T=32, K=32, D=64, **BG.GEMM_HOPPER),
        bind(BG.Gemm, K=32, N=3, **BG.GEMM_AMPERE),
        bind(BG.Gemm, K=32, N=3, **BG.GEMM_HOPPER),
        bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_HOPPER_E4M3),
        bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_SM120_E4M3),
        bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_HOPPER_E4M3_MMAV2),
        bind(BG.GemmCoordinateNvf4, K=128, **BG.GEMM_SM120_NVF4),
        bind(BG.GemmCoordinateMxf4, K=128, **BG.GEMM_SM120_MXF4),
        bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8),
        bind(BG.ScaledMmS8Coordinate, K=64, **BG.GEMM_SM120_S8),
        bind(BG.ScaledMmS8CoordinateRows, K=64, R=2, **BG.GEMM_SM90_S8),
        bind(BG.ScaledMmS8ResidualCoordinateRows, K=64, R=2, **BG.GEMM_SM90_S8),
        bind(BG.ScaledMmS8, K=64, N=3, **BG.GEMM_SM120_S8),
    ]


def _population_roots() -> list[Any]:
    """`TemplateInstances_v1` of the scaled s8 coordinate (K = 64, sm_120) at N = 2, a served s8 linear's window program, and
    `TemplatePopulations_v1` of the s8 accumulator and that coordinate at counts 1 and 2."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.population import (
        mixed_population_program,
        population_program,
    )

    acc = bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8)
    coord = bind(BG.ScaledMmS8Coordinate, K=64, **BG.GEMM_SM120_S8)
    return [population_program(coord, 2).fn, mixed_population_program([(acc, 1), (coord, 2)]).fn]

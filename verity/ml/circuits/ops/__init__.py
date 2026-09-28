"""verity.ml.circuits.ops: exact, hardware-faithful semantics of ML primitives, and their registrations in the Verity IR.

    verity.ml.circuits.silicon      the tensor-core accumulator step (stdlib only; no IR dependency)
    verity.ml.circuits.ops.prims   IR primitives: AmpereBF16TcDot16, HopperBF16WgmmaDot16, HopperE4m3QgmmaDot32, BlackwellE4m3QmmaDot32,
                      Bf16ToF32, F32ToBf16Rn, F2fpBf16, F32ToE4m3Sat, const
    verity.ml.circuits.ops.scalar  IR primitives: integer, bit, select and comparison operations; FP32 abs / neg / max / min / cvt.sat
    verity.ml.circuits.ops.fp32    IR primitives: FP32 add / mul / fma / div (RN-even), the FTZ family, FA2 max and -inf guard, BF16
                      element-wise add / mul
    verity.ml.circuits.ops.mufu    IR primitives: the MUFU ex2 / rcp / sqrt / rsqrt / tanh approximations over their measured sm_89 tables
                      (package data), FA2's inverse sum, div.full's range scaling; numpy kernels for all but tanh
    verity.ml.circuits.ops.gemm    IR composites: DotBf16 / GemmCoordinate / Gemm parameterised by the step; GEMM_AMPERE, GEMM_HOPPER;
                      the sm_120 GemmCoordinateE4m3 / GemmCoordinateNvf4 / GemmCoordinateMxf4 and their GEMM_SM120_* bindings
    verity.ml.circuits.ops.boolean word arithmetic as Boolean circuits, built with `verity.ml.circuits.boolean`'s builders
                      (`forms`, `gather`, the multiplexer tree, and `trace`, Boolean Definitions from builders) and
                      `verity.ml.circuits.silicon`'s (`fp`, `tc_step`, `fp4`): `softmax`, attention's circuits over `forms`,
                      `mufu`: the MUFU operations on bits, `gemm`: the BF16 steps, cast,
                      DotBf16_v3, GemmCoordinate_v3 and Gemm_v3 on bits, the E4M3, E5M2, NVFP4 and MXFP4 steps and their
                      coordinates, `attention`: FA2's FTZ f32 operations, max, -inf guard and inverse sum on bits,
                      `elementwise`: FP32 add / mul / fma / div (no ftz), the BF16 residual add and the I32 add on bits, `cast`:
                      the bf16 -> f32 widening on bits, and `scalar`: the bf16 comparator, the selects, fabsf, PTX max / min
                      and the f32 -> E4M3 cast on bits
    verity.ml.kernels.tc numpy batch forms of the steps and GEMM coordinates (BF16, FP8 E4M3, NVFP4 / MXFP4), registered with
                      `verity.ml.circuits.evaluation` and self-checked against `verity.ml.circuits.silicon`
"""
from __future__ import annotations

__version__ = "0.1.0"

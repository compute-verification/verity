"""verity.ml: exact, hardware-faithful semantics of ML primitives, and their registrations in the Verity IR.

    verity.ml.tc      the tensor-core accumulator step (stdlib only; no IR dependency)
    verity.ml.prims   IR primitives: AmpereBF16TcDot16, HopperBF16WgmmaDot16, Bf16ToF32, F32ToBf16Rn, F2fpBf16, const
    verity.ml.gemm    IR composites: DotBf16 / GemmCoordinate / Gemm parameterised by the step; GEMM_AMPERE, GEMM_HOPPER
"""
from __future__ import annotations

__version__ = "0.1.0"

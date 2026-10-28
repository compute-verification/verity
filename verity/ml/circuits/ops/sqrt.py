"""Registered FP32 square root: `PouwF32SqrtRn_v1`, sqrt.rn.f32 as the GPU computes it (ρ's root in Pearl's statistics,
and the IEEE slow path of a served kernel's `sqrt.rn`).  Its Boolean version is `PouwF32SqrtRn_v2` (PoUW's
`verity.core.protocols.pouw.circuit.boolean`, from `ops.boolean.sqrt.f32_sqrt_rn`).

Provenance.  PoUW registered it in `verity.core.protocols.pouw.circuit.words` (same id, signature, conformance status and
evaluator), which re-exports it from here.
"""
from __future__ import annotations

from verity.ml.circuits.defs import primitive
from verity.ml.circuits.types import Value
from verity.ml.circuits.silicon import fp32 as _fp32

F32 = Value(32)
M32 = (1 << 32) - 1


@primitive("PouwF32SqrtRn", 1, [("a", F32)], F32, conformance="protocol-defined")
def PouwF32SqrtRn(a: int) -> int:
    """sqrt.rn.f32 (`verity.ml.circuits.silicon.fp32.sqrt`, rho's root in Pearl's statistics): nearest even, sqrt(-0) = -0,
    sqrt(+inf) = +inf; a NaN or a negative nonzero operand gives the GPU's NaN word 0x7FFFFFFF."""
    a &= M32
    if (a >> 23) & 0xFF == 0xFF:
        return a if a == 0x7F800000 else 0x7FFFFFFF
    if a >> 31 and a & 0x7FFFFFFF:
        return 0x7FFFFFFF
    return _fp32.sqrt(a)


__all__ = ["PouwF32SqrtRn"]

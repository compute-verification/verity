"""The bf16 <-> f32 casts as Boolean Definitions: the next version of each word primitive, the same function on bits.

    Bf16ToF32_v2    (a: Array<16, Value<1>>) -> Array<32, Value<1>>    word view: Bf16ToF32_v1
    F32ToBf16Rn_v2  (a: Array<32, Value<1>>) -> Array<16, Value<1>>    word view: F32ToBf16Rn_v1

Traced from `fp.bf16_to_f32`: no gates, its result's low 16 bits the constant 0 and its high 16 bits the operand's; and from
`fp.f32_to_bf16` with the software NaN 0x7FC0 (torch's round to nearest even). The other f32 -> bf16 cast, as F2FP rounds it
(NaN 0x7FFF), is `gemm.F2fpBf16_v2`.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.types import Array

from .. import prims as P
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.silicon.cast import BF16_NAN_SOFTWARE
from verity.ml.circuits.boolean.trace import definition

__all__ = ["Bf16ToF32", "F32ToBf16Rn"]

Bf16ToF32 = definition("Bf16ToF32", 2, lambda C, a: fp.bf16_to_f32(a), (("a", Array(16, BL.BIT)),), Array(32, BL.BIT),
                       word=P.Bf16ToF32, doc="the exact bf16 -> f32 widening on bits: Bf16ToF32_v1, the encoding << 16")
F32ToBf16Rn = definition("F32ToBf16Rn", 2, lambda C, a: fp.f32_to_bf16(C, a, BF16_NAN_SOFTWARE), (("a", Array(32, BL.BIT)),),
                         Array(16, BL.BIT), word=P.F32ToBf16Rn,
                         doc="torch's f32 -> bf16 on bits: F32ToBf16Rn_v1, round to nearest even, NaN -> 0x7FC0")

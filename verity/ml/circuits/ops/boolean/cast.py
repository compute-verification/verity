"""The bf16 -> f32 widening as a Boolean Definition: the next version of the word primitive, the same function on bits.

    Bf16ToF32_v2    (a: Array<16, Value<1>>) -> Array<32, Value<1>>    word view: Bf16ToF32_v1

Traced from `fp.bf16_to_f32`: no gates, its result's low 16 bits the constant 0 and its high 16 bits the operand's. The other
cast, f32 -> bf16 as F2FP rounds it, is `gemm.F2fpBf16_v2`.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.types import Array

from .. import prims as P
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.boolean.trace import definition

__all__ = ["Bf16ToF32"]

Bf16ToF32 = definition("Bf16ToF32", 2, lambda C, a: fp.bf16_to_f32(a), (("a", Array(16, BL.BIT)),), Array(32, BL.BIT),
                       word=P.Bf16ToF32, doc="the exact bf16 -> f32 widening on bits: Bf16ToF32_v1, the encoding << 16")

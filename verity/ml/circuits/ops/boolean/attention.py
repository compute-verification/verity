"""FA2 attention's softmax arithmetic as Boolean Definitions: the next version of each word primitive, the same function on bits.

    F32AddFtz_v3, F32SubFtz_v3, F32MulFtz_v3   (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>       word view: their _v1
    F32FmaFtz_v3, F32FmaSubFtz_v3              (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>    word view: their _v1
    F32Max_v2                                  (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>       word view: F32Max_v1
    GuardNegInfZero_v2                         (m: Array<32, Value<1>>) -> Array<32, Value<1>>          word view: GuardNegInfZero_v1
    Fa2InvSum_v2                               (sum: Array<32, Value<1>>) -> Array<32, Value<1>>        word view: Fa2InvSum_v1

Each is traced from `softmax`'s builder (`trace.definition`), the circuit C-Flock's lowering runs for its word primitive, so its
And gates are that builder's live ANDs. `Fa2InvSum_v2` is ``(sum == 0 || sum != sum) ? 1 : MUFU.RCP(sum)`` with `mufu`'s
reciprocal, whose table read is a plain circuit. The QK dot is `gemm.DotBf16_v3` and MUFU.EX2 is `mufu.MufuEx2Ftz_v2`.
`fa3_inv_sum` is FA3's epilogue (``? 0 : MUFU.RCP(sum)``); its Definition is in `verity_vllm`, beside its word primitive.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.types import Array

from .. import fp32 as W
from .. import mufu as M
from . import mufu as MU
from . import softmax as SM
from verity.ml.circuits.boolean.forms import NOT, OR, const_bits, mux, or_reduce
from verity.ml.circuits.boolean.trace import definition

__all__ = ["Fa2InvSum", "F32AddFtz", "F32FmaFtz", "F32FmaSubFtz", "F32Max", "F32MulFtz", "F32SubFtz", "GuardNegInfZero"]

F32 = Array(32, BL.BIT)
_AB = (("a", F32), ("b", F32))
_ABC = (("a", F32), ("b", F32), ("c", F32))

F32AddFtz = definition("F32AddFtz", 3, SM.f32_add_ftz, _AB, F32, word=W.F32AddFtz,
                       doc="FADD.FTZ on bits: F32AddFtz_v1, ftz(RN(ftz(a) + ftz(b)))")
F32SubFtz = definition("F32SubFtz", 3, SM.f32_sub_ftz, _AB, F32, word=W.F32SubFtz,
                       doc="FADD.FTZ with negated second operand on bits: F32SubFtz_v1, ftz(RN(ftz(a) - ftz(b)))")
F32MulFtz = definition("F32MulFtz", 3, SM.f32_mul_ftz, _AB, F32, word=W.F32MulFtz,
                       doc="FMUL.FTZ on bits: F32MulFtz_v1, ftz(RN(ftz(a) * ftz(b)))")
F32FmaFtz = definition("F32FmaFtz", 3, SM.f32_fma_ftz, _ABC, F32, word=W.F32FmaFtz,
                       doc="FFMA.FTZ on bits: F32FmaFtz_v1, ftz(RN(ftz(a) * ftz(b) + ftz(c)))")
F32FmaSubFtz = definition("F32FmaSubFtz", 3, SM.f32_fma_sub_ftz, _ABC, F32, word=W.F32FmaSubFtz,
                          doc="FFMA.FTZ with negated addend on bits: F32FmaSubFtz_v1, ftz(RN(ftz(a) * ftz(b) - ftz(c)))")
F32Max = definition("F32Max", 2, SM.f32_max, _AB, F32, word=W.F32Max,
                    doc="FA2 MaxOp on bits: F32Max_v1, x > y ? x : y on the flushed operands")
GuardNegInfZero = definition("GuardNegInfZero", 2, SM.guard_neg_inf_zero, (("m", F32),), F32, word=W.GuardNegInfZero,
                             doc="FA2 Check_inf on bits: GuardNegInfZero_v1, m == -inf ? 0 : m")


#: f32 1.0
ONE_F = 0x3F800000


def fa2_inv_sum(C, s):
    """``Fa2InvSum_v1``: ``(sum == 0 || sum != sum) ? 1 : MUFU.RCP(sum)``."""
    one = OR(C, NOT(or_reduce(C, s[:31])), SM._is_nan(C, s))
    return mux(C, one, const_bits(ONE_F, 32), MU.mufu_rcp(C, s))


def fa3_inv_sum(C, s):
    """FA3's ``Softmax::finalize`` (``Fa3InvSum_v1`` in `verity_vllm`): ``(sum == 0 || sum != sum) ? 0 : MUFU.RCP(sum)``."""
    zero = OR(C, NOT(or_reduce(C, s[:31])), SM._is_nan(C, s))
    return mux(C, zero, const_bits(0, 32), MU.mufu_rcp(C, s))


Fa2InvSum = definition("Fa2InvSum", 2, fa2_inv_sum, (("sum", F32),), F32, word=M.Fa2InvSum,
                       doc="FA2's softmax inverse sum on bits: Fa2InvSum_v1, (sum == 0 || NaN) ? 1 : MUFU.RCP(sum)")

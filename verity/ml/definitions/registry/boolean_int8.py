"""`int8_exact`'s linear and LUT attention on bits: the Boolean version of each of its word Definitions, so a served int8-exact
row builds Boolean-first (`verity.ml.definitions.boolean`).

    F32Rint_v2, F32ToS8Rzi_v2, S8ToF32_v2, F32MulWide_v2, F64Mul_v2, F64ToU32RniSat_v2, LutIndex_v2
                        traced from `verity.ml.circuits.ops.boolean.int8`'s builders; word views: their _v1
    Exp2Lut127_v3       (t: Array<8, Value<1>>) -> Array<32, Value<1>>; word view: Exp2Lut127_v1.  `boolean.table.read` (the
                        multiplexer tree of `TableRead{IB}x{EW}_v2`) over fattn's 256 entries as constant bits, which fold: only
                        an entry's 7 low bits are ever set
    U8ToU16_v2          (t: Array<8, Value<1>>) -> Array<16, Value<1>>; word view: U8ToU16_v1.  No gates: t's bits, then 8 zeros
    Int8QuantElement_v2, Int8RowScale_v2{K}, Int8StaticQuant_v2{K}, Int8RowQuant_v2{K}, Int8Linear_v2{K,N}, Int8AttnScale_v2{SCALE},
    Int8LutSoftmax_v3{S}, Int8AttnOut_v2, Int8LutAttentionRow_v3{D,S,SCALE}, Int8ColumnQuant_v2{S,D}, Int8KvQuant_v2{S,D}
                        σ of the word composite (`boolean.sigma_lift`): its body at each binding with each Call the Boolean
                        version of its Definition; word views: their _v1 at the same statics
    Int8RowQuant_v4{K,R}
                        σ of `Int8RowQuant_v3{K,R}` likewise, x as R rows of K/R values; word view: Int8RowQuant_v3

`Int8LutSoftmax_v2` and `Int8LutAttentionRow_v2`, the forms that read `exp2_lut127` as an operand, have no Boolean version here;
`_v3` of each is σ of `_v1`, whose LUT is the Definition's own.  The callees beyond this module's leaves are core's
(`ops.boolean`'s F32Div_v3, F32Mul_v3, F32Add_v3, Bf16ToF32_v2, F32ToBf16Rn_v2, F32Fmaxf_v2, F32Fminf_v2, F32Fabs_v2, SelectI32_v2,
I32Le_v2, I32Add_v2, S32LowByte_v2, S8Matmul_v2, ScaledMmS8_v2 and `Const<w>[0x..]_v2`), and `boolean_dsa`'s I32Sub_v2 and
I32ToF32_v2.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import table as BT
from verity.ml.circuits.boolean.forms import const_bits
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.ops.boolean import int8 as BI
from verity.ml.circuits.types import Array
from verity.ml.definitions.boolean import sigma_lift
from verity.ml.definitions.registry import int8_exact as IE

BIT = BL.BIT
F32, F64, I32, CODE = Array(32, BIT), Array(64, BIT), Array(32, BIT), Array(8, BIT)

F32Rint = definition("F32Rint", 2, BI.f32_rint, (("x", F32),), F32, word=IE.F32Rint,
                     doc="cvt.rni.f32.f32 on bits: F32Rint_v1, ties to even, the sign kept, an sNaN quieted")
F32ToS8Rzi = definition("F32ToS8Rzi", 2, BI.f32_to_s8_rzi, (("x", F32),), CODE, word=IE.F32ToS8Rzi,
                        doc="cvt.rzi.sat.s8.f32 on bits: F32ToS8Rzi_v1, truncated, saturated to [-128, 127], NaN 0")
S8ToF32 = definition("S8ToF32", 2, BI.s8_to_f32, (("c", CODE),), F32, word=IE.S8ToF32, doc="the s8 code as f32 on bits: S8ToF32_v1")
F32MulWide = definition("F32MulWide", 2, BI.f32_mul_wide, (("a", F32), ("b", F32)), F64, word=IE.F32MulWide,
                        doc="the exact f64 product of two f32 on bits: F32MulWide_v1, CPython's NaNs")
F64Mul = definition("F64Mul", 2, BI.f64_mul, (("a", F64), ("b", F64)), F64, word=IE.F64Mul,
                    doc="mul.rn.f64 on bits: F64Mul_v1, subnormals kept, CPython's NaNs")
F64ToU32RniSat = definition("F64ToU32RniSat", 2, BI.f64_to_u32_rni_sat, (("x", F64),), I32, word=IE.F64ToU32RniSat,
                            doc="min(rint(x), 2^32 - 1) for x > 0, else 0, on bits: F64ToU32RniSat_v1")
LutIndex = definition("LutIndex", 2, BI.lut_index, (("d", I32), ("a", I32)), CODE, word=IE.LutIndex,
                      doc="min(umulhi(max(d, 0), a), 255) on bits: LutIndex_v1")


def _exp2_lut127(C, t):
    return BT.read(C, t, [const_bits(w, BT.WORD_BITS) for w in IE.exp2_lut127_words()], 8, 2)


Exp2Lut127 = definition("Exp2Lut127", 3, _exp2_lut127, (("t", CODE),), I32, word=IE.Exp2Lut127,
                        doc="fattn's LUT[t] = rint(127 2^(-t/16)) on bits: Exp2Lut127_v1, the table read's multiplexer tree over "
                            "constant entries")
U8ToU16 = definition("U8ToU16", 2, lambda C, t: [*t, *const_bits(0, 8)], (("t", CODE),), Array(16, BIT), word=IE.U8ToU16,
                     doc="the byte zero-extended to a 16-bit word on bits: U8ToU16_v1, its high 8 bits the constant 0")

Int8QuantElementV2 = sigma_lift(IE.Int8QuantElement, 2)
Int8RowScaleV2 = sigma_lift(IE.Int8RowScale, 2)
Int8StaticQuantV2 = sigma_lift(IE.Int8StaticQuant, 2)
Int8RowQuantV2 = sigma_lift(IE.Int8RowQuant, 2)
Int8LinearV2 = sigma_lift(IE.Int8Linear, 2)
Int8AttnScaleV2 = sigma_lift(IE.Int8AttnScale, 2)
Int8LutSoftmaxV3 = sigma_lift(IE.Int8LutSoftmax, 3)
Int8AttnOutV2 = sigma_lift(IE.Int8AttnOut, 2)
Int8LutAttentionRowV3 = sigma_lift(IE.Int8LutAttentionRow, 3)
Int8ColumnQuantV2 = sigma_lift(IE.Int8ColumnQuant, 2)
Int8KvQuantV2 = sigma_lift(IE.Int8KvQuant, 2)
Int8RowQuantV4 = sigma_lift(IE.Int8RowQuantV3, 4)

INT8_LEAVES = [F32Rint, F32ToS8Rzi, S8ToF32, F32MulWide, F64Mul, F64ToU32RniSat, LutIndex, Exp2Lut127, U8ToU16]
INT8_SIGMA = [Int8QuantElementV2, Int8RowScaleV2, Int8StaticQuantV2, Int8RowQuantV2, Int8LinearV2, Int8AttnScaleV2,
              Int8LutSoftmaxV3, Int8AttnOutV2, Int8LutAttentionRowV3, Int8ColumnQuantV2, Int8KvQuantV2, Int8RowQuantV4]

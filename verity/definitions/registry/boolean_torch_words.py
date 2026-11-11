"""`torch_words`' word primitives on bits: the Boolean version of each, the same function (`verity.circuits.boolean.trace`).

    I64Add_v2         (a, b: Array<64, Value<1>>) -> Array<64, Value<1>>        word view: I64Add_v1, a ripple-carry adder
    I64Le_v2          (a, b: Array<64, Value<1>>) -> Value<1>                   word view: I64Le_v1, not b < a unsigned with the
                                                                                sign bits flipped
    I64Shl_v2         (a, b: Array<64, Value<1>>) -> Array<64, Value<1>>        word view: I64Shl_v1, a barrel shifter by b's low six
                                                                                bits, zero when a higher bit of b is set
    I64Shr_v2         (a, b: Array<64, Value<1>>) -> Array<64, Value<1>>        word view: I64Shr_v1, a logical barrel shifter between
                                                                                two XORs with the sign, by 63 when a higher bit of b
                                                                                is set
    I64And_v2         (a, b: Array<64, Value<1>>) -> Array<64, Value<1>>        word view: I64And_v1
    SelectI64_v2      (c: Value<1>, a, b: Array<64, Value<1>>) -> Array<64, Value<1>>    word view: SelectI64_v1
    I64ToU8_v2        (t: Array<64, Value<1>>) -> Array<8, Value<1>>            word view: I64ToU8_v1, no gates
    U8ToI64_v2        (t: Array<8, Value<1>>) -> Array<64, Value<1>>            word view: U8ToI64_v1, no gates
    I64ToF32Rn_v2     (t: Array<64, Value<1>>) -> Array<32, Value<1>>           word view: I64ToF32Rn_v1, |t| rounded by `fp.round_f32`
    F32ToI64Rz_v2     (x: Array<32, Value<1>>) -> Array<64, Value<1>>           word view: F32ToI64Rz_v1, the significand shifted
                                                                                and negated with the sign
    F32ToF16Rn_v2     (x: Array<32, Value<1>>) -> Array<16, Value<1>>           word view: F32ToF16Rn_v1, the significand rounded
                                                                                by `fp.round_fp` at f16
    F16ToF32_v2       (h: Array<16, Value<1>>) -> Array<32, Value<1>>           word view: F16ToF32_v1, normalized by `fp.round_f32`
    F32OrderKey_v2    (x: Array<32, Value<1>>) -> Array<32, Value<1>>           word view: F32OrderKey_v1
    OrderKeyF32_v2    (k: Array<32, Value<1>>) -> Array<32, Value<1>>           word view: OrderKeyF32_v1, no gates
"""
from __future__ import annotations

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, mux, or_reduce, shl_window, shr, sub
from verity.circuits.boolean.trace import definition
from verity.definitions.ops.boolean.scalar import gt_u, select
from verity.definitions.silicon import fp
from verity.circuits.types import Array

from . import torch_words as W

__all__ = ["F16ToF32", "F32OrderKey", "F32ToF16Rn", "F32ToI64Rz", "I64Add", "I64And", "I64Le", "I64Shl", "I64Shr", "I64ToF32Rn",
           "I64ToU8", "OrderKeyF32", "SelectI64", "U8ToI64"]

BIT = BL.BIT
I64, U8, F16, F32 = Array(64, BIT), Array(8, BIT), Array(16, BIT), Array(32, BIT)
_AB = (("a", I64), ("b", I64))


def i64_le(C, a, b):
    """``I64Le_v1``: signed a <= b, not a > b in the unsigned order of the words with their sign bits flipped."""
    def flip(x):
        return list(x[:63]) + [NOT(x[63])]
    return [NOT(gt_u(C, flip(a), flip(b)))]


def i64_shl(C, a, b):
    """``I64Shl_v1``: a shifted left by b's low six bits, as `shr` shifts the reversed word right, zero when a higher bit of b is
    set (b < 0 or b >= 64)."""
    return shr(C, list(a)[::-1], b[:6], zero_if=or_reduce(C, b[6:]))[::-1]


def i64_shr(C, a, b):
    """``I64Shr_v1``: a shifted right with its sign shifted in, a logical shift of a XOR its sign, XORed with the sign again; by
    63 (b's low six bits all ones) when a higher bit of b is set (b < 0 or b >= 64)."""
    big = or_reduce(C, b[6:])
    s = a[63]
    return [y ^ s for y in shr(C, [x ^ s for x in a], [OR(C, big, k) for k in b[:6]])]


def i64_to_f32(C, t):
    """``cvt.rn.f32.s64``: |t| (2^63 for INT64_MIN) rounded to f32 at exponent 190 by `fp.round_f32`, the sign t's."""
    return fp.round_f32(C, t[63], const_bits(190, 9), 0, fp.neg_if(C, list(t), t[63]), ZERO, underflow=False)


def f32_to_i64(C, x):
    """``cvt.rzi.s64.f32``: for 1 <= |x| < 2^63 (exponent field 127..189) bits [23, 86) of the significand shifted left by e - 127,
    negated with x's sign; from 2^63 up (the infinities too) INT64_MAX or INT64_MIN by the sign; below 1 zero; a NaN zero."""
    e, s = x[23:31], x[31]
    k, small = sub(C, e, const_bits(127, 8), 8)
    _, fits = sub(C, e, const_bits(190, 8), 8)
    mag = [C.AND(NOT(small), b) for b in shl_window(C, list(x[:23]) + [ONE], k[:6], 23, 86)]
    val = add(C, [b ^ s for b in mag + [ZERO]], [], cin=s, width=64)
    nan = C.AND(and_reduce(C, e), or_reduce(C, x[:23]))
    return [C.AND(NOT(nan), b) for b in mux(C, fits, val, [NOT(s)] * 63 + [s])]


def f32_to_f16(C, x):
    """``cvt.rn.f16.f32``: the value ``sig 2^(e_eff - 150)`` is ``sig 2^(E - 15 - 23)`` at E = e_eff - 112, rounded at f16 by
    `fp.round_fp` (subnormals kept, overflow to infinity); an infinity keeps its sign, a NaN is 0x7FFF."""
    X = fp.unpack(C, x)
    word = fp.round_fp(C, X.s, list(X.e_eff) + [ZERO], 112, X.sig, X.s, 5, 10, underflow=True)
    word = mux(C, X.inf, const_bits(0x7C00, 15) + [X.s], word)
    return mux(C, X.nan, const_bits(W.F16_NAN, 16), word)


def f16_to_f32(C, h):
    """``__half2float``: the value ``sig 2^(e_eff - 25)`` is ``sig 2^(E - 127 - 10)`` at E = e_eff + 112, which `fp.round_f32`
    normalizes exactly; an infinity keeps its sign, a NaN is 0x7FFFFFFF."""
    H = fp.unpack_fp(C, h, 5, 10)
    E = add(C, list(H.e_eff) + [ZERO] * 4, const_bits(112, 9), width=9)
    word = fp.round_f32(C, H.s, E, 0, H.sig, H.s, underflow=False)
    word = mux(C, H.inf, const_bits(0x7F800000, 31) + [H.s], word)
    return mux(C, H.nan, fp.f32_const(W.F32_NAN), word)


def f32_order_key(C, x):
    """``F32OrderKey_v1``: the low 31 bits XOR the sign, the top bit its complement; a NaN all ones."""
    nan = fp.unpack(C, x).nan
    key = [b ^ x[31] for b in x[:31]] + [NOT(x[31])]
    return [OR(C, nan, k) for k in key]


I64Add = definition("I64Add", 2, lambda C, a, b: add(C, a, b, width=64), _AB, I64, word=W.I64Add,
                    doc="add.s64 on bits: I64Add_v1, a + b modulo 2^64")
I64Le = definition("I64Le", 2, i64_le, _AB, BIT, word=W.I64Le, doc="signed 64-bit a <= b on bits: I64Le_v1")
I64Shl = definition("I64Shl", 2, i64_shl, _AB, I64, word=W.I64Shl, doc="a << b on 64-bit words, on bits: I64Shl_v1")
I64Shr = definition("I64Shr", 2, i64_shr, _AB, I64, word=W.I64Shr, doc="a >> b, the sign shifted in, on bits: I64Shr_v1")
I64And = definition("I64And", 2, lambda C, a, b: [C.AND(x, y) for x, y in zip(a, b)], _AB, I64, word=W.I64And,
                    doc="a & b on 64-bit words, on bits: I64And_v1")
SelectI64 = definition("SelectI64", 2, lambda C, c, a, b: select(C, c[0], a, b), (("c", BIT), ("a", I64), ("b", I64)), I64,
                       word=W.SelectI64, doc="c ? a : b on 64-bit words, on bits: SelectI64_v1")
I64ToU8 = definition("I64ToU8", 2, lambda C, t: list(t[:8]), (("t", I64),), U8, word=W.I64ToU8,
                     doc="the low byte of a 64-bit word, on bits: I64ToU8_v1")
U8ToI64 = definition("U8ToI64", 2, lambda C, t: [*t, *const_bits(0, 56)], (("t", U8),), I64, word=W.U8ToI64,
                     doc="a byte zero-extended to 64 bits, on bits: U8ToI64_v1")
I64ToF32Rn = definition("I64ToF32Rn", 2, i64_to_f32, (("t", I64),), F32, word=W.I64ToF32Rn,
                        doc="cvt.rn.f32.s64 on bits: I64ToF32Rn_v1")
F32ToI64Rz = definition("F32ToI64Rz", 2, f32_to_i64, (("x", F32),), I64, word=W.F32ToI64Rz,
                        doc="cvt.rzi.s64.f32 on bits: F32ToI64Rz_v1, saturated, a NaN 0")
F32ToF16Rn = definition("F32ToF16Rn", 2, f32_to_f16, (("x", F32),), F16, word=W.F32ToF16Rn,
                        doc="cvt.rn.f16.f32 on bits: F32ToF16Rn_v1, every NaN 0x7FFF")
F16ToF32 = definition("F16ToF32", 2, f16_to_f32, (("h", F16),), F32, word=W.F16ToF32,
                      doc="__half2float on bits: F16ToF32_v1, exact, every NaN 0x7FFFFFFF")
F32OrderKey = definition("F32OrderKey", 2, f32_order_key, (("x", F32),), F32, word=W.F32OrderKey,
                         doc="TopKTypeConfig<float>::convert on bits: F32OrderKey_v1")
OrderKeyF32 = definition("OrderKeyF32", 2, lambda C, k: [b ^ NOT(k[31]) for b in k[:31]] + [NOT(k[31])], (("k", F32),), F32,
                         word=W.OrderKeyF32, doc="TopKTypeConfig<float>::deconvert on bits: OrderKeyF32_v1")

"""The scalar comparison, selects and bit AND, IEEE f32 arithmetic, PTX max / min and the bf16, E4M3 and E2M1 casts as Boolean
Definitions: the next version of each word primitive, the same function on bits (the arithmetic and casts are the scalar
operations of the served FP8 linear, around its tensor-core steps, and of NVFP4 quantization).

    Bf16GtStrict_v2   (a, b: Array<16, Value<1>>) -> Value<1>                            word view: Bf16GtStrict_v1
    SelectBf16_v2     (c: Value<1>, a, b: Array<16, Value<1>>) -> Array<16, Value<1>>    word view: SelectBf16_v1
    SelectI32_v2      (c: Value<1>, a, b: Array<32, Value<1>>) -> Array<32, Value<1>>    word view: SelectI32_v1
    SelectF32_v2      (c: Value<1>, a, b: Array<32, Value<1>>) -> Array<32, Value<1>>    word view: SelectF32_v1
    F32Eq_v2          (a, b: Array<32, Value<1>>) -> Value<1>                            word view: F32Eq_v1
    BitAnd_v2         (a, b: Value<1>) -> Value<1>                                      word view: BitAnd_v1
    F32Add_v3, F32Mul_v3, F32Div_v3    (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>      word view: their _v1
    F32Fma_v3                          (a, b, c: Array<32, Value<1>>) -> Array<32, Value<1>>   word view: F32Fma_v1
    F32Fabs_v2                         (x: Array<32, Value<1>>) -> Array<32, Value<1>>         word view: F32Fabs_v1
    F32Fmaxf_v2, F32Fminf_v2           (a, b: Array<32, Value<1>>) -> Array<32, Value<1>>      word view: their _v1
    Bf16ToF32_v2                       (a: Array<16, Value<1>>) -> Array<32, Value<1>>         word view: Bf16ToF32_v1
    F32ToE4m3Sat_v2                    (x: Array<32, Value<1>>) -> Array<8, Value<1>>          word view: F32ToE4m3Sat_v1
    E4m3ToF32_v2                       (x: Array<8, Value<1>>) -> Array<32, Value<1>>          word view: E4m3ToF32_v1
    F32ToE2m1Sat_v2                    (x: Array<32, Value<1>>) -> Array<4, Value<1>>          word view: F32ToE2m1Sat_v1

Each is traced from a builder over `forms` (`trace.definition`), the circuit C-Flock's lowering runs for its word primitive. The
builders are also pieces of larger circuits, such as an argmax over a logits row: `bf16_key` maps a bf16 word to a NaN flag and a
16-bit key whose unsigned order is the value order (+0 and -0 one key), `f32_key` an f32 word likewise, `gt_u` compares two keys,
`select` is the bitwise multiplexer. The arithmetic is `elementwise`'s and the bf16 widening `cast`'s, named here too; max / min,
the E4M3 and E2M1 casts and the bit AND are written here (the first two formerly C-Flock's `tail_pieces`, which imports them).
"""
from __future__ import annotations

import functools
import operator

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.types import Array

from .. import prims as P
from .. import scalar as S
from verity.ml.circuits.silicon.cast import E2M1_UP
from verity.ml.circuits.silicon import fp
from .cast import Bf16ToF32
from .elementwise import F32Add, F32Div, F32Fma, F32Mul
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub
from .softmax import _eq_const, _is_nan, f32_gt
from verity.ml.circuits.boolean.trace import definition

__all__ = ["BitAnd", "Bf16GtStrict", "Bf16ToF32", "E4m3ToF32", "F32Add", "F32Div", "F32Eq", "F32Fabs", "F32Fma", "F32Fmaxf",
           "F32Fminf", "F32Mul", "F32ToE2m1Sat", "F32ToE4m3Sat", "SelectBf16", "SelectF32", "SelectI32", "bf16_gt_strict", "bf16_key",
           "e4m3_to_f32", "f32_eq", "f32_key", "f32_to_e2m1_sat", "gt_u", "select"]

BIT = BL.BIT
BF16 = Array(16, BIT)
I32 = Array(32, BIT)
F32 = Array(32, BIT)
E4M3 = Array(8, BIT)
E2M1 = Array(4, BIT)
_AB = (("a", F32), ("b", F32))


def bf16_key(C, x):
    """``(nan, key)`` of a bf16 word: ``key`` (16 bits, LSB first) orders the non-NaN values as unsigned integers, with one key
    for +0 and -0: a positive word with its sign bit set, a negative nonzero word with its magnitude complemented and its sign
    bit clear (-0 is +0's 0x8000). A NaN's key is arbitrary; ``nan`` says it is one."""
    m, e, s = x[:7], x[7:15], x[15]
    e_any, m_any = or_reduce(C, e), or_reduce(C, m)
    nan = C.AND(and_reduce(C, e), m_any)
    neg = C.AND(s, OR(C, e_any, m_any))
    return nan, [b ^ neg for b in x[:15]] + [NOT(neg)]


def f32_key(C, x):
    """`bf16_key` of an f32 word: ``(nan, key)``, ``key`` 32 bits whose unsigned order is the order of the non-NaN values
    (`F32GtStrict_v1`'s: subnormals as they are, +0 and -0 one key)."""
    m, e, s = x[:23], x[23:31], x[31]
    e_any, m_any = or_reduce(C, e), or_reduce(C, m)
    nan = C.AND(and_reduce(C, e), m_any)
    neg = C.AND(s, OR(C, e_any, m_any))
    return nan, [b ^ neg for b in x[:31]] + [NOT(neg)]


def gt_u(C, a, b):
    """``a > b`` for unsigned words of one width (LSB first): the borrow of ``b - a``."""
    return sub(C, b, a, len(a))[1]


def bf16_gt_strict(C, a, b):
    """``Bf16GtStrict_v1``: a > b on bf16 values (a NaN on either side compares false; +0 and -0 are equal)."""
    na, ka = bf16_key(C, a)
    nb, kb = bf16_key(C, b)
    return [C.AND(NOT(OR(C, na, nb)), gt_u(C, ka, kb))]


def select(C, c, a, b):
    """``c ? a : b`` bitwise, ``c`` one bit (one AND per bit)."""
    return mux(C, c, a, b)


def f32_eq(C, a, b):
    """``F32Eq_v1``: IEEE a == b, neither a NaN and the encodings equal or both zeros (+0 == -0)."""
    same = and_reduce(C, [NOT(x ^ y) for x, y in zip(a, b)])
    zeros = NOT(OR(C, or_reduce(C, a[:31]), or_reduce(C, b[:31])))
    return C.AND(C.AND(NOT(_is_nan(C, a)), NOT(_is_nan(C, b))), OR(C, same, zeros))


def _ult_const(C, w, k: int):
    """``w < k`` for an unsigned word and a constant."""
    return sub(C, w, const_bits(k, len(w)), len(w))[1]


def _fields(C, x):
    s, e, m = x[31], list(x[23:31]), list(x[:23])
    return s, e, m, _eq_const(C, e, 0xFF), NOT(or_reduce(C, e)), or_reduce(C, m)


def fmaxmin(C, a, b, want_max: bool):
    """PTX ``max.f32`` / ``min.f32``: a NaN operand yields the other, two NaNs 0x7FFFFFFF; ``max(+0, -0) = +0``,
    ``min(+0, -0) = -0``."""
    na, nb = _is_nan(C, a), _is_nan(C, b)
    r = mux(C, f32_gt(C, b, a)[0] if want_max else f32_gt(C, a, b)[0], b, a)
    both_zero = NOT(OR(C, or_reduce(C, a[:31]), or_reduce(C, b[:31])))
    zs = C.AND(a[31], b[31]) if want_max else OR(C, a[31], b[31])
    r = mux(C, both_zero, [ZERO] * 31 + [zs], r)
    r = mux(C, na, b, mux(C, nb, a, r))
    return mux(C, C.AND(na, nb), fp.f32_const(0x7FFFFFFF), r)


def f32_to_e4m3_sat(C, x):
    """``F32ToE4m3Sat_v1`` (``cvt.rn.satfinite.e4m3x2.f32``): RN-even onto the E4M3 grid (three fraction bits from 2^-6, steps of
    2^-9 below), ±448 (0x7E) past it and for the infinities, 0x7F for NaN."""
    s, e, m, e_max, e_zero, m_nz = _fields(C, x)
    sig = m + [NOT(e_zero)]
    normal = NOT(_ult_const(C, e, 121))                       # |x| >= 2^-6
    amt = mux(C, normal, const_bits(19, 5),
              mux(C, _ult_const(C, e, 116), const_bits(24, 5), sub(C, const_bits(140, 8), e, 8)[0][:5]))
    y, sticky = fp.shr_sticky(C, sig, amt)                    # the kept bits and a guard bit
    n = add(C, y[1:5] + [ZERO], [C.AND(y[0], OR(C, sticky, y[1]))], width=5)
    base = [C.AND(normal, b) for b in sub(C, e, const_bits(121, 8), 8)[0][:4]]
    enc = add(C, [ZERO] * 3 + base, n, width=8)
    sat = OR(C, NOT(_ult_const(C, e, 136)), NOT(_ult_const(C, enc, 0x7F)))
    out = mux(C, sat, const_bits(0x7E, 7), enc[:7]) + [s]
    return mux(C, C.AND(e_max, m_nz), const_bits(0x7F, 8), out)


def e4m3_to_f32(C, x):
    """``E4m3ToF32_v1``: an E4M3 byte widened to f32, exactly. A normal byte's exponent field is ``e + 120``, its fraction ``m``
    the top three bits; a subnormal ``m 2^-9`` is normalized by its leading bit (fields 118, 119, 120); NaN (0x7F / 0xFF) is
    0x7FFFE000, its sign dropped. The exponent's pairwise ANDs serve both its OR and the NaN test (C-Flock's lowering has no
    CSE)."""
    m, e, s = x[:3], x[3:7], x[7]
    e01, e23 = C.AND(e[0], e[1]), C.AND(e[2], e[3])
    h = OR(C, e[0] ^ e[1] ^ e01, e[2] ^ e[3] ^ e23)          # a normal (or NaN) byte
    nan = and_reduce(C, [C.AND(e01, e23), C.AND(m[0], m[1]), m[2]])
    t = C.AND(NOT(h), NOT(m[2]))                            # e = 0, m < 4: zero or a subnormal of field 118 / 119
    tm0, tm1 = C.AND(t, m[0]), C.AND(t, m[1])
    tm10 = C.AND(tm1, m[0])
    tnz = tm0 ^ tm1 ^ tm10                                  # fields 118, 119
    zero = t ^ tnz
    hm = [C.AND(h, b) for b in m]
    um0, um1 = m[0] ^ hm[0] ^ tm0, m[1] ^ hm[1] ^ tm1      # e = 0, m >= 4: field 120, its fraction m - 4 shifted up
    mant = [hm[0], hm[1] ^ um0, hm[2] ^ um1 ^ tm10]
    ex = NOT(e[3]) ^ nan
    field = [e[0] ^ tm1, e[1] ^ tnz, e[2] ^ tnz, ex ^ t, ex ^ zero, ex ^ zero, ex ^ zero, e[3]]
    return [ZERO] * 13 + [nan] * 7 + mant + field + [C.AND(s, NOT(nan))]


def _passes(C, w, bounds):
    """``w > bound`` (``w >= bound`` when ``up``) for each constant ``(bound, up)`` of an unsigned word: one comparator per
    bound over the bits above the bounds' common zero low bits, MSB first, each equal-prefix AND shared by every bound with
    that prefix (a trie), and the low bits one OR. ``w >= bound`` stops at the bound's lowest one bit: an AND below it would
    cancel out of ``gt ^ eq`` and leave a dead bit inside the forms `trace` writes."""
    low = min((b & -b).bit_length() - 1 for b, _ in bounds)
    lo, hi = or_reduce(C, w[:low]), w[low:]
    ands: dict = {}
    out = []
    for bound, up in bounds:
        k, eq, gt = bound >> low, ONE, ZERO
        for i in reversed(range((k & -k).bit_length() - 1 if up else 0, len(hi))):
            if (k >> i + 1, i) not in ands:
                ands[k >> i + 1, i] = C.AND(eq, hi[i])
            a = ands[k >> i + 1, i]
            gt, eq = (gt, a) if k >> i & 1 else (gt ^ a, eq ^ a)
        out.append(gt ^ (eq if up else C.AND(eq, lo)))
    return out


def f32_to_e2m1_sat(C, x):
    """``F32ToE2m1Sat_v1`` (``cvt.rn.satfinite.e2m1x2.f32``): the code is how many of the E2M1 boundaries (`cast.E2M1_UP`) |x|
    passes, the sign kept. The passes are a thermometer, so the count's bit j is the XOR of every 2^j-th; a NaN passes all
    seven, and the NaN test, ``|x| > inf``, drops its sign and turns its code 7 into ``E2M1_NAN_CODE``."""
    *levels, nan = _passes(C, x[:31], [*E2M1_UP, (0x7F800000, False)])
    code = [functools.reduce(operator.xor, levels[(1 << j) - 1::1 << j]) for j in range(3)]
    out = code + [C.AND(x[31], NOT(nan))]
    return [o ^ C.AND(nan, b) for o, b in zip(out, const_bits(P.E2M1_NAN_CODE ^ 0x7, 4))]


Bf16GtStrict = definition("Bf16GtStrict", 2, bf16_gt_strict, (("a", BF16), ("b", BF16)), BIT, word=S.Bf16GtStrict,
                          doc="a > b on bf16 bits: Bf16GtStrict_v1 (NaN compares false, +0 == -0)")
SelectBf16 = definition("SelectBf16", 2, lambda C, c, a, b: select(C, c[0], a, b), (("c", BIT), ("a", BF16), ("b", BF16)), BF16,
                        word=S.SelectBf16, doc="c ? a : b on bf16 bits: SelectBf16_v1")
SelectI32 = definition("SelectI32", 2, lambda C, c, a, b: select(C, c[0], a, b), (("c", BIT), ("a", I32), ("b", I32)), I32,
                       word=S.SelectI32, doc="c ? a : b on i32 bits: SelectI32_v1")
SelectF32 = definition("SelectF32", 2, lambda C, c, a, b: select(C, c[0], a, b), (("c", BIT), ("a", F32), ("b", F32)), F32,
                       word=S.SelectF32, doc="c ? a : b on f32 bits: SelectF32_v1")
F32Eq = definition("F32Eq", 2, lambda C, a, b: [f32_eq(C, a, b)], _AB, BIT, word=S.F32Eq,
                   doc="a == b on f32 bits: F32Eq_v1 (NaN unequal to everything, +0 == -0)")
F32Fabs = definition("F32Fabs", 2, lambda C, x: list(x[:31]) + [ZERO], (("x", F32),), F32, word=S.F32Fabs,
                     doc="fabsf on bits: F32Fabs_v1, the sign bit cleared")
F32Fmaxf = definition("F32Fmaxf", 2, lambda C, a, b: fmaxmin(C, a, b, True), _AB, F32, word=S.F32Fmaxf,
                      doc="max.f32 on bits: F32Fmaxf_v1, NaN yields the other operand, max(+0, -0) = +0")
F32Fminf = definition("F32Fminf", 2, lambda C, a, b: fmaxmin(C, a, b, False), _AB, F32, word=S.F32Fminf,
                      doc="min.f32 on bits: F32Fminf_v1, NaN yields the other operand, min(+0, -0) = -0")
F32ToE4m3Sat = definition("F32ToE4m3Sat", 2, f32_to_e4m3_sat, (("x", F32),), E4M3, word=P.F32ToE4m3Sat,
                          doc="cvt.rn.satfinite.e4m3x2.f32 on bits: F32ToE4m3Sat_v1")
E4m3ToF32 = definition("E4m3ToF32", 2, e4m3_to_f32, (("x", E4M3),), F32, word=P.E4m3ToF32,
                       doc="float(__nv_fp8_e4m3) on bits: E4m3ToF32_v1, exact, NaN 0x7FFFE000")
F32ToE2m1Sat = definition("F32ToE2m1Sat", 2, f32_to_e2m1_sat, (("x", F32),), E2M1, word=P.F32ToE2m1Sat,
                          doc="cvt.rn.satfinite.e2m1x2.f32 on bits: F32ToE2m1Sat_v1, NaN E2M1_NAN_CODE")
BitAnd = definition("BitAnd", 2, lambda C, a, b: [C.AND(a[0], b[0])], (("a", BIT), ("b", BIT)), BIT, word=S.BitAnd,
                    doc="a & b on one bit: BitAnd_v1")

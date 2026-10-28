"""The exact-int8 profile's numerics on bits: builders over `forms` for the f32 / f64 steps of vLLM's int8-exact linear and LUT
attention (`verity.ml.definitions.registry.int8_exact`).  Their Boolean Definitions are `verity.ml.definitions.registry.boolean_int8`'s,
since their word primitives are vLLM's and a Boolean Definition holds its word; `scalar`'s `F32ToS8RniSat_v2` is the first two
in a row.

    f32_rint(C, x)              `F32Rint_v1` (np.rint on f32): ties to even, the sign kept, an sNaN quieted, inf and a qNaN
                                themselves
    f32_to_s8_rzi(C, x)         `F32ToS8Rzi_v1` (cvt.rzi.sat.s8.f32): truncated toward zero, saturated to [-128, 127], NaN 0
    s8_to_f32(C, c)             `S8ToF32_v1`: the s8 code as f32, exact
    f32_mul_wide(C, a, b)       `F32MulWide_v1`: the f64 product of two f32, exact (never subnormal, never rounded)
    f64_mul(C, a, b)            `F64Mul_v1`: RN64(a b), subnormals kept
    f64_to_u32_rni_sat(C, x)    `F64ToU32RniSat_v1`: min(rint(x), 2^32 - 1) for x > 0, else 0 (a NaN 0)
    lut_index(C, d, a)          `LutIndex_v1`: min(floor(max(d, 0) a / 2^32), 255) as a byte

The two products are CPython's float multiply on x86-64 (after cvtss2sd for the f32 operands): a NaN operand's NaN quieted, the
first operand's when both are, and the default NaN 0xFFF8000000000000 for 0 * inf.  `f64_mul` rounds through `fp.round_fp`,
`round_f32` at f64's format.
"""
from __future__ import annotations

from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, multiply, mux, or_reduce, shr, sub
from verity.ml.circuits.silicon import fp

__all__ = ["f32_mul_wide", "f32_rint", "f32_to_s8_rzi", "f64_mul", "f64_to_u32_rni_sat", "lut_index", "s8_to_f32"]

#: f32 1.0's exponent field, bits 23..30
_ONE_EXP = 0x3F800000 >> 23


def _decode(C, bits, n: int) -> list:
    """[bits == k] for k < n, ``bits`` unsigned LSB first: one AND a pair of halves' decodes."""
    if len(bits) == 1:
        return [NOT(bits[0]), bits[0]][:n]
    h = len(bits) // 2
    lo, hi = _decode(C, bits[:h], 1 << h), _decode(C, bits[h:], 1 << (len(bits) - h))
    return [C.AND(lo[k & ((1 << h) - 1)], hi[k >> h]) for k in range(n)]


def f32_rint(C, x):
    """`F32Rint_v1`: for 1 <= |x| < 2^23 (exponent field 127..149, f = 150 - e fraction bits), the encoding plus 2^(f-1) - 1 plus
    its bit f, with its f low bits cleared (ties to even; a carry moves into the exponent); |x| in [0.5, 1) is 1.0 unless it is
    0.5, a smaller |x| zero, both with x's sign; from 2^23 up x itself, a NaN quieted."""
    e, s = x[23:31], x[31]
    d, small = sub(C, e, const_bits(127, 8), 8)            # e - 127; small iff |x| < 1
    _, below_half = sub(C, e, const_bits(126, 8), 8)
    _, frac = sub(C, e, const_bits(150, 8), 8)              # |x| < 2^23
    dec = _decode(C, d[:5], 23)                             # [e - 127 == k]: one-hot within 127 <= e < 150
    mask = [ZERO] * 23
    acc = ZERO
    for k in range(23):                                     # mask_j = [j < f] = [e - 127 <= 22 - j]; one-hot, so OR is XOR
        acc = acc ^ dec[k]
        mask[22 - k] = acc
    lsb = ZERO
    for k in range(23):
        lsb = lsb ^ C.AND(dec[k], x[23 - k])                # bit f = 23 - (e - 127)
    r = add(C, list(x[:31]), mask[1:] + [ZERO] * 8, cin=lsb, width=31)
    r = [C.AND(NOT(m), b) for b, m in zip(r, mask + [ZERO] * 8)]
    one = C.AND(NOT(below_half), or_reduce(C, x[:23]))      # 0.5 < |x| < 1
    tiny = [ZERO] * 23 + [one if _ONE_EXP >> i & 1 else ZERO for i in range(8)]
    nan = C.AND(and_reduce(C, e), or_reduce(C, x[:23]))
    big = list(x[:22]) + [OR(C, x[22], nan)] + list(x[23:31])
    return mux(C, frac, mux(C, small, tiny, r), big) + [s]


def f32_to_s8_rzi(C, x):
    """`F32ToS8Rzi_v1`: for 1 <= |x| < 128 (exponent field 127..133) the significand's top 7 bits shifted right by 133 - e and
    negated with x's sign; from 128 up (infinities too) 127 or -128 by the sign; below 1 zero; a NaN zero."""
    e, s = x[23:31], x[31]
    _, small = sub(C, e, const_bits(127, 8), 8)
    _, fits = sub(C, e, const_bits(134, 8), 8)
    amt, _ = sub(C, const_bits(133, 8), e, 8)
    mag = shr(C, list(x[17:23]) + [ONE], amt[:3], zero_if=small)
    val = add(C, [b ^ s for b in mag + [ZERO]], [], cin=s, width=8)
    sat = [NOT(s)] * 7 + [s]
    nan = C.AND(and_reduce(C, e), or_reduce(C, x[:23]))
    return [C.AND(NOT(nan), b) for b in mux(C, fits, val, sat)]


def s8_to_f32(C, c):
    """`S8ToF32_v1`: |c| (8 bits, 128 for -128) rounded exactly as an f32 at exponent 134, with c's sign; 0 is +0."""
    s = c[7]
    mag = add(C, [b ^ s for b in c], [], cin=s, width=8)
    return fp.round_f32(C, s, const_bits(134, 9), 0, mag, ZERO, underflow=False)


def _product_specials(C, r, A, B, s, nan_a, nan_b):
    """CPython's float multiply on x86-64 past the finite product ``r``: ``nan_a`` when a is a NaN, else ``nan_b`` when b is, else
    the default NaN for 0 * inf, else the infinity of sign ``s`` when an operand is infinite."""
    invalid = OR(C, C.AND(A.inf, B.zero), C.AND(A.zero, B.inf))
    r = mux(C, OR(C, A.inf, B.inf), [ZERO] * 51 + [invalid] + [ONE] * 11 + [OR(C, invalid, s)], r)
    return mux(C, OR(C, A.nan, B.nan), mux(C, A.nan, nan_a, nan_b), r)


def _f32_nan_as_f64(a):
    """cvtss2sd of an f32 NaN: its sign, the all-ones exponent, its payload at the top of the 52-bit field, quieted."""
    return [ZERO] * 29 + list(a[:22]) + [ONE] * 12 + [a[31]]


def _quiet64(a):
    return list(a[:51]) + [ONE] + list(a[52:])


def f32_mul_wide(C, a, b):
    """`F32MulWide_v1`: the 48-bit significand product at f64's exponent ea + eb + 770 (each f32 is sig 2^(e_eff - 150)), which
    is at least 725 even for subnormal operands, so the f64 word is exact and normal; a zero operand gives the zero of the
    product's sign."""
    A, B = fp.unpack(C, a), fp.unpack(C, b)
    s = A.s ^ B.s
    P = multiply(C, fp.trim_low(A.sig)[0], fp.trim_low(B.sig)[0])
    E_b = add(C, add(C, A.e_eff, B.e_eff, width=12), const_bits(770, 12), width=12)
    r = fp.round_fp(C, s, E_b, 0, P, s, 11, 52, underflow=False)
    return _product_specials(C, r, A, B, s, _f32_nan_as_f64(a), _f32_nan_as_f64(b))


def f64_mul(C, a, b):
    """`F64Mul_v1`: the 106-bit significand product rounded once (`fp.round_fp` at f64, E = ea + eb - 1022), subnormals kept,
    overflow to infinity."""
    A, B = fp.unpack_fp(C, a, 11, 52), fp.unpack_fp(C, b, 11, 52)
    s = A.s ^ B.s
    P = multiply(C, fp.trim_low(A.sig)[0], fp.trim_low(B.sig)[0])
    r = fp.round_fp(C, s, add(C, A.e_eff, B.e_eff, width=13), 1022, P, s, 11, 52)
    return _product_specials(C, r, A, B, s, _quiet64(a), _quiet64(b))


def f64_to_u32_rni_sat(C, x):
    """`F64ToU32RniSat_v1`: x = sig 2^(e - 1075), so the integer part is (2 sig) >> (1075 - e + 1) and its bit 0 below is the
    guard: 2 sig's top 33 bits shifted right by 1054 - e (`fp.shr_sticky`), rounded to even with the bits below as sticky,
    saturated at 2^32 - 1 from e > 1054 or a carry out; a sign bit set or a NaN gives 0."""
    m, e, s = list(x[:52]), list(x[52:63]), x[63]
    e_nz = or_reduce(C, e)
    w = [ZERO] + m + [e_nz]                                 # 2 sig, 54 bits
    d, big = sub(C, const_bits(1054, 11), e, 11)
    y, st = fp.shr_sticky(C, w[21:], d)
    sticky = OR(C, st, or_reduce(C, w[:21]))
    rnd = C.AND(y[0], OR(C, sticky, y[1]))
    R = add(C, y[1:], [rnd], width=32, carry_out=True)
    cap = OR(C, big, R[32])
    zero = OR(C, s, C.AND(and_reduce(C, e), or_reduce(C, m)))
    return [C.AND(NOT(zero), OR(C, cap, b)) for b in R[:32]]


def lut_index(C, d, a):
    """`LutIndex_v1`: max(d, 0) (d's low 31 bits unless its sign is set) times a, the product's bits 32..62, saturated at 255."""
    dpos = [C.AND(NOT(d[31]), b) for b in d[:31]]
    hi = multiply(C, dpos, list(a))[32:63]
    over = or_reduce(C, hi[8:])
    return [OR(C, over, b) for b in hi[:8]]

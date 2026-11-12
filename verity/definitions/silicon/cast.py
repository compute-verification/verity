"""Word casts between FP32, BF16, E4M3 and E2M1 encodings, as pure integer (and exact rational) functions."""
from __future__ import annotations

from fractions import Fraction

BF16_NAN_SOFTWARE = 0x7FC0   # c10::BFloat16::round_to_nearest_even, torch CPU: the software canonical NaN
BF16_NAN_HARDWARE = 0x7FFF   # SASS F2FP.BF16.F32.PACK_AB / cvt.rn.bf16.f32: measured on A100 and RTX 4090
E4M3_NAN = 0x7F              # OCP E4M3 (float8_e4m3fn): NaN 0x7F / 0xFF, max finite 0x7E = 448, no infinities

__all__ = ["BF16_NAN_HARDWARE", "BF16_NAN_SOFTWARE", "E2M1_UP", "E4M3_NAN", "bf16_is_nan", "bf16_to_f32_word",
           "e4m3_to_f32_word", "f32_is_nan", "f32_to_bf16_hw_word", "f32_to_bf16_rn_word", "f32_to_bf16_word", "f32_to_e2m1_sat_word",
           "f32_to_e4m3_sat_word", "s32_to_f32_rn_word"]


def f32_is_nan(u: int) -> bool:
    return (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF) != 0


def bf16_is_nan(w: int) -> bool:
    return (w & 0x7F80) == 0x7F80 and (w & 0x7F) != 0


def bf16_to_f32_word(w: int) -> int:
    """Exact widening: the BF16 word becomes the high half of the FP32 word."""
    return (w & 0xFFFF) << 16


def f32_to_bf16_rn_word(u: int, nan: int) -> int:
    """Round to nearest, ties to even, by integer arithmetic on the encoding; every NaN maps to ``nan``.
    A finite word that rounds past the largest finite BF16 carries into the exponent and becomes an infinity."""
    u &= 0xFFFFFFFF
    if f32_is_nan(u):
        return nan
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF


def f32_to_bf16_word(u: int) -> int:
    """The software conversion (torch CPU, ``gpu/gpt2/pinned_ops.cu``): NaN -> ``0x7FC0``."""
    return f32_to_bf16_rn_word(u, BF16_NAN_SOFTWARE)


def f32_to_bf16_hw_word(u: int) -> int:
    """``cvt.rn.bf16.f32`` as the A100 and RTX 4090 compute it: NaN -> ``0x7FFF`` (equal to ``total.cvt_rn_bf16_f32``)."""
    return f32_to_bf16_rn_word(u, BF16_NAN_HARDWARE)


def s32_to_f32_rn_word(w: int) -> int:
    """``cvt.rn.f32.s32`` (SASS ``I2FP.F32.S32``): the 32-bit two's-complement word ``w`` as the nearest FP32 word, ties to
    even. Exact below 2^24 in magnitude; never overflows (|w| <= 2^31); zero is +0."""
    w &= 0xFFFFFFFF
    v = w - (1 << 32) if w >> 31 else w
    if v == 0:
        return 0
    sign, m = (1 << 31 if v < 0 else 0), abs(v)
    e = m.bit_length() - 1
    if e <= 23:
        q = m << (23 - e)
    else:
        sh = e - 23
        q, r = m >> sh, m & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if r > half or (r == half and q & 1):
            q += 1
            if q >> 24:
                q >>= 1
                e += 1
    return sign | ((e + 127) << 23) | (q & 0x7FFFFF)


def f32_to_e4m3_sat_word(u: int) -> int:
    """``cvt.rn.satfinite.e4m3x2.f32``, one lane (``__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)``), as an H100
    computes it on every f32 word: NaN -> ``0x7F`` (the sign dropped); ±inf and every |x| that would round beyond 448 ->
    ``±0x7E`` (448, the sign kept); otherwise round-to-nearest-even onto the E4M3 grid (normal binades 2^-6..2^8 with three
    fraction bits, subnormals in multiples of 2^-9); |x| < 2^-10 and the 2^-10 tie round to a signed zero."""
    u &= 0xFFFFFFFF
    sign = 0x80 if u >> 31 else 0
    exp = (u >> 23) & 0xFF
    mant = u & 0x7FFFFF
    if exp == 0xFF:
        return E4M3_NAN if mant else (sign | 0x7E)
    if exp == 0 and mant == 0:
        return sign
    if exp == 0:
        mag = Fraction(mant, 1 << 149)
    else:
        mag = Fraction((1 << 23) | mant, 1) / Fraction(1 << (150 - exp)) if exp < 150 else Fraction(((1 << 23) | mant) << (exp - 150))
    if mag >= Fraction(464):  # >= the 448|480 tie: 480 is not encodable -> saturate
        return sign | 0x7E
    # grid step: 2^(e-3) for 2^e <= mag < 2^(e+1), e >= -6; 2^-9 below 2^-6
    e = mag.numerator.bit_length() - mag.denominator.bit_length()
    if Fraction(2) ** e > mag:
        e -= 1
    step_exp = e - 3 if e >= -6 else -9
    q = mag / (Fraction(2) ** step_exp)
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n & 1):
        n += 1
    val = n * (Fraction(2) ** step_exp)
    if val == 0:
        return sign
    if val >= Fraction(464):
        return sign | 0x7E
    if val < Fraction(2) ** -6:  # subnormal: mantissa n in units of 2^-9 (n <= 7)
        return sign | int(val / (Fraction(2) ** -9))
    e2 = val.numerator.bit_length() - val.denominator.bit_length()
    if Fraction(2) ** e2 > val:
        e2 -= 1
    m3 = int((val / (Fraction(2) ** e2) - 1) * 8)
    return sign | ((e2 + 7) << 3) | m3


def e4m3_to_f32_word(b: int) -> int:
    """An E4M3 byte widened to f32, exactly (``float(__nv_fp8_e4m3)``: ``cvt.rn.f16x2.e4m3x2`` then ``cvt.f32.f16``, both exact
    on finite bytes).  NaN (``0x7F`` / ``0xFF``) -> the f16 canonical NaN ``0x7FFF`` widened, ``0x7FFFE000``, sign dropped: the
    documented conversion chain, not a measurement."""
    b &= 0xFF
    if b & 0x7F == E4M3_NAN:
        return 0x7FFFE000
    sign = (b & 0x80) << 24
    e, m = (b >> 3) & 0xF, b & 0x7
    if e == 0:
        if m == 0:
            return sign
        p = m.bit_length() - 1                    # m * 2^-9 = (1 + (m - 2^p) / 2^p) * 2^(p - 9)
        return sign | ((p - 9 + 127) << 23) | ((m - (1 << p)) << (23 - p))
    return sign | ((e - 7 + 127) << 23) | (m << 20)


#: ``cvt.rn.satfinite.e2m1x2.f32``'s round-to-nearest-even onto the E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} as the seven
#: code boundaries (|x| as an f32 word, True when a tie on the boundary rounds up to the even code): the code is how many of them
#: |x| passes.  0.25, 1.25, 2.5 and 5 tie down, 0.75, 1.75 and 3.5 tie up; beyond 5 (inf included) saturates to 6.
E2M1_UP: tuple[tuple[int, bool], ...] = ((0x3E800000, False), (0x3F400000, True), (0x3FA00000, False), (0x3FE00000, True),
                                         (0x40200000, False), (0x40600000, True), (0x40A00000, False))


def f32_to_e2m1_sat_word(u: int, nan: int = 0x7) -> int:
    """``cvt.rn.satfinite.e2m1x2.f32``, one lane (SASS ``F2FP.SATFINITE.E2M1.F32.PACK_AB``): the sign bit kept (a signed zero
    included), the magnitude rounded to nearest even onto the E2M1 grid, ``|x| > 5`` and ±inf saturated to ±6 (code 7).  E2M1
    has no NaN: a NaN input gives the code ``nan``, its sign dropped (as the E4M3 cast drops it)."""
    u &= 0xFFFFFFFF
    sign = 0x8 if u >> 31 else 0
    mag = u & 0x7FFFFFFF
    if mag > 0x7F800000:
        return nan
    return sign | sum(1 for bound, up in E2M1_UP if mag > bound or (up and mag == bound))

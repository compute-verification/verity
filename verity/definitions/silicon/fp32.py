"""IEEE-754 binary32 arithmetic on words, rounded to nearest even with subnormals kept: the scalar FP32 pipe.

These are the words an H100 (and every sm_20+ part) writes for PTX ``add.rn.f32``, ``mul.rn.f32``, ``fma.rn.f32``,
``div.rn.f32`` and ``sqrt.rn.f32`` without ``.ftz`` (SASS ``FADD``, ``FMUL``, ``FFMA`` and the correctly rounded
division and square-root sequences).  Source negation (``-a`` as an operand modifier) is ``neg``; ``add(a, neg(b))`` is
``sub.rn.f32``.  Every function is exact integer arithmetic on the encoding and rounds once.

Inputs must be finite; a NaN or an infinity raises :class:`InvalidArtifact`.  Results follow IEEE 754-2019 in
roundTiesToEven: an exact zero sum of nonzero terms is ``+0``, ``(-0) + (-0)`` is ``-0``, a product's zero keeps the
product's sign, and a finite result beyond the binary32 range is the infinity of its sign.  ``div`` by zero and
``sqrt`` of a negative value raise rather than return an infinity or a NaN.

Used by the split-K FP8 schemes, whose checked running words are ``add`` of fresh tensor-core atoms, and by the
``sm90.mma.m16n8k32.e4m3`` lowering (two BF16 HMMA partials joined by FP32 ``FADD``), which ``instructions`` records.
"""
from __future__ import annotations

from math import isqrt

from .errors import InvalidArtifact

__all__ = ["NEG_ZERO", "add", "decode", "div", "fma", "mul", "neg", "round_to_f32", "sqrt"]

NEG_ZERO = 0x8000_0000
_MASK = 0xFFFF_FFFF
_MIN_EXP = -149              # weight of the least significant bit of a subnormal significand


def decode(word: int) -> tuple[bool, int, int]:
    """A finite FP32 word as ``(negative, significand, exponent)`` with value ``(-1)^negative * significand * 2^exponent``."""

    word &= _MASK
    field = (word >> 23) & 0xFF
    if field == 0xFF:
        raise InvalidArtifact(f"non-finite FP32 word {word:#010x}")
    frac = word & 0x7F_FFFF
    if field == 0:
        return bool(word >> 31), frac, _MIN_EXP
    return bool(word >> 31), frac | (1 << 23), field - 150


def round_to_f32(negative: bool, n: int, e: int, sticky: bool = False) -> int:
    """The binary32 word nearest to ``(-1)^negative * (n + t) * 2^e``, ties to even, where ``t = 0`` when ``sticky`` is
    false and ``0 < t < 1`` otherwise.  ``n >= 0``; a sticky value needs ``n`` at least 26 bits wide so that ``t`` lies
    strictly below the rounding bit."""

    sign = NEG_ZERO if negative else 0
    if n == 0:
        if sticky:
            raise InvalidArtifact("a sticky remainder needs the significand's leading bits")
        return sign
    top = n.bit_length() - 1 + e
    lsb = max(top - 23, _MIN_EXP)
    shift = lsb - e
    if shift <= 0:
        if sticky:
            raise InvalidArtifact("a sticky remainder needs at least two bits below the result's last place")
        q = n << -shift
    else:
        q, rem, half = n >> shift, n & ((1 << shift) - 1), 1 << (shift - 1)
        if rem > half or (rem == half and (sticky or q & 1)):
            q += 1
    if q >> 24:
        q, lsb = q >> 1, lsb + 1
    if q == 0:
        return sign
    if q >> 23:
        field = lsb + 23 + 127
        if field >= 0xFF:
            return sign | 0x7F80_0000
        return sign | (field << 23) | (q & 0x7F_FFFF)
    return sign | q


def neg(a: int) -> int:
    """The source-negation modifier: the sign bit flipped."""

    return (a ^ NEG_ZERO) & _MASK


def _sum(pa: bool, ma: int, ea: int, pb: bool, mb: int, eb: int, zero_negative: bool) -> int:
    e = min(ea, eb)
    s = (-ma if pa else ma) << (ea - e)
    s += (-mb if pb else mb) << (eb - e)
    if s == 0:
        return NEG_ZERO if zero_negative else 0
    return round_to_f32(s < 0, abs(s), e)


def add(a: int, b: int) -> int:
    """``add.rn.f32``."""

    na, ma, ea = decode(a)
    nb, mb, eb = decode(b)
    return _sum(na, ma, ea, nb, mb, eb, zero_negative=na and nb and not ma and not mb)


def mul(a: int, b: int) -> int:
    """``mul.rn.f32``."""

    na, ma, ea = decode(a)
    nb, mb, eb = decode(b)
    return round_to_f32(na != nb, ma * mb, ea + eb)


def fma(a: int, b: int, c: int) -> int:
    """``fma.rn.f32``: ``a * b + c`` rounded once."""

    na, ma, ea = decode(a)
    nb, mb, eb = decode(b)
    nc, mc, ec = decode(c)
    p = ma * mb
    return _sum(na != nb, p, ea + eb, nc, mc, ec, zero_negative=(na != nb) and nc and not p and not mc)


def div(a: int, b: int) -> int:
    """``div.rn.f32``."""

    na, ma, ea = decode(a)
    nb, mb, eb = decode(b)
    if mb == 0:
        raise InvalidArtifact("FP32 division by zero")
    if ma == 0:
        return NEG_ZERO if na != nb else 0
    k = 27 + mb.bit_length() - ma.bit_length()
    q, r = divmod(ma << k, mb)
    return round_to_f32(na != nb, q, ea - eb - k, sticky=r != 0)


def sqrt(a: int) -> int:
    """``sqrt.rn.f32``; ``sqrt(-0)`` is ``-0``."""

    na, ma, ea = decode(a)
    if ma == 0:
        return a & _MASK
    if na:
        raise InvalidArtifact("FP32 square root of a negative value")
    if ea & 1:
        ma, ea = ma << 1, ea - 1
    t = max(0, (54 - ma.bit_length()) // 2 + 1)
    m = ma << (2 * t)
    r = isqrt(m)
    return round_to_f32(False, r, ea // 2 - t, sticky=r * r != m)

"""IEEE-754 binary16 arithmetic on words, rounded to nearest even with subnormals kept: the half2 pipe, one lane.

These are the words an H100 writes in each 16-bit half for PTX ``add.rn.f16``, ``mul.rn.f16``, ``fma.rn.f16`` and
``max.f16`` (SASS ``HADD2``, ``HMUL2``, ``HFMA2``, ``HMNMX2``); f16 arithmetic has no flush-to-zero.  Every function is
exact integer arithmetic on the encoding and rounds once.  Inputs must be finite (NaN and infinities raise); a finite
result beyond the binary16 range is the infinity of its sign, and zero signs follow IEEE 754-2019 roundTiesToEven, as in
`verity.ml.tc.fp32`.  E4M3 is a subset of binary16, so `from_e4m3` is exact; `to_e4m3` is
``cvt.rn.satfinite.e4m3x2.f16x2``'s lane, which rounds the f16 value once (`verity.ml.tc.cast.f32_to_e4m3_sat_word` on
its exact FP32 widening).
"""
from __future__ import annotations

from .cast import f32_to_e4m3_sat_word
from .errors import InvalidArtifact

__all__ = ["NEG_ZERO", "add", "decode", "fma", "from_e4m3", "max_", "mul", "neg", "round_to_f16", "to_e4m3", "to_f32"]

NEG_ZERO = 0x8000
_MASK = 0xFFFF
_MIN_EXP = -24               # weight of the least significant bit of a subnormal significand


def decode(word: int) -> tuple[bool, int, int]:
    """A finite binary16 word as ``(negative, significand, exponent)``: value ``(-1)^negative * significand * 2^exponent``."""
    word &= _MASK
    field = (word >> 10) & 0x1F
    if field == 0x1F:
        raise InvalidArtifact(f"non-finite binary16 word {word:#06x}")
    frac = word & 0x3FF
    if field == 0:
        return bool(word >> 15), frac, _MIN_EXP
    return bool(word >> 15), frac | (1 << 10), field - 25


def round_to_f16(negative: bool, n: int, e: int) -> int:
    """The binary16 word nearest to ``(-1)^negative * n * 2^e`` (n >= 0), ties to even."""
    sign = NEG_ZERO if negative else 0
    if n == 0:
        return sign
    top = n.bit_length() - 1 + e
    lsb = max(top - 10, _MIN_EXP)
    shift = lsb - e
    if shift <= 0:
        q = n << -shift
    else:
        q, rem, half = n >> shift, n & ((1 << shift) - 1), 1 << (shift - 1)
        if rem > half or (rem == half and q & 1):
            q += 1
    if q >> 11:
        q, lsb = q >> 1, lsb + 1
    if q == 0:
        return sign
    if q >> 10:
        field = lsb + 10 + 15
        if field >= 0x1F:
            return sign | 0x7C00
        return sign | (field << 10) | (q & 0x3FF)
    return sign | q


def neg(a: int) -> int:
    return (a ^ NEG_ZERO) & _MASK


def _sum(pa: bool, ma: int, ea: int, pb: bool, mb: int, eb: int, zero_negative: bool) -> int:
    e = min(ea, eb)
    s = ((-ma if pa else ma) << (ea - e)) + ((-mb if pb else mb) << (eb - e))
    if s == 0:
        return NEG_ZERO if zero_negative else 0
    return round_to_f16(s < 0, abs(s), e)


def add(a: int, b: int) -> int:
    """``add.rn.f16``."""
    na, ma, ea = decode(a)
    nb, mb, eb = decode(b)
    return _sum(na, ma, ea, nb, mb, eb, zero_negative=na and nb and not ma and not mb)


def mul(a: int, b: int) -> int:
    """``mul.rn.f16``."""
    na, ma, ea = decode(a)
    nb, mb, eb = decode(b)
    return round_to_f16(na != nb, ma * mb, ea + eb)


def fma(a: int, b: int, c: int) -> int:
    """``fma.rn.f16``: ``a * b + c`` rounded once."""
    na, ma, ea = decode(a)
    nb, mb, eb = decode(b)
    nc, mc, ec = decode(c)
    p = ma * mb
    return _sum(na != nb, p, ea + eb, nc, mc, ec, zero_negative=(na != nb) and nc and not p and not mc)


def max_(a: int, b: int) -> int:
    """``max.f16`` on finite words: the larger value; of two zeros, +0."""
    na, ma, ea = decode(a)
    nb, mb, eb = decode(b)
    va = (-ma if na else ma) << (ea - _MIN_EXP)
    vb = (-mb if nb else mb) << (eb - _MIN_EXP)
    if va == vb:
        return a & b & _MASK if va == 0 else a & _MASK
    return a & _MASK if va > vb else b & _MASK


def to_f32(a: int) -> int:
    """The exact FP32 word of a binary16 word (``cvt.f32.f16``)."""
    na, m, e = decode(a)
    if m == 0:
        return 0x8000_0000 if na else 0
    top = m.bit_length() - 1 + e
    return (0x8000_0000 if na else 0) | ((top + 127) << 23) | ((m << (23 - (m.bit_length() - 1))) & 0x7F_FFFF)


def from_e4m3(code: int) -> int:
    """The binary16 word of an E4M3 code (exact; ``cvt.rn.f16x2.e4m3x2``'s lane).  NaN codes raise."""
    if code & 0x7F == 0x7F:
        raise InvalidArtifact("E4M3 NaN code")
    sign, e, m = code >> 7, (code >> 3) & 0xF, code & 7
    return round_to_f16(bool(sign), m, -9) if e == 0 else round_to_f16(bool(sign), 8 | m, e - 10)


def to_e4m3(a: int) -> int:
    """``cvt.rn.satfinite.e4m3x2.f16x2``'s lane: RN-even onto E4M3, saturating to ±448."""
    return f32_to_e4m3_sat_word(to_f32(a))

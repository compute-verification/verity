"""H-1T's IR primitives: its pinned forming map on E4M3 codes, and byte packing for its digests.

The map is Track H's `h1_tagged.h1t_ref` (docs/pouw/hardness-shaped-matrices.md §3.1) on one k32 slice of 29 real lanes and
3 tag lanes, in exact rational arithmetic over `h1_atom`'s E4M3 codec (`val`, and `enc`: round to nearest, ties to the
even code, saturating at ±448, a negative value that rounds to zero giving +0):

    F  = max(2^(e(M) − 6), 2^−2),  M = max |x| over the 29 real lanes          e(M) = floor(log2 M); F = 2^−2 at M = 0
    mu = max(p/8, F),  p = 2^floor(log2 |x|) (0 at x = 0)
    x1 = enc(x + s·mu),  x2 = enc(−s·mu)                                          real lanes
    v  = s_i·32F·(1 + m_i/8);  x1 = enc(v), x2 = enc(−v)                         tag lanes

The salt bits are the honest kernel's (`h1_kernel.h1t_form_xq`): per slice, raw words r0 … r17; real lane l's bit is bit 15
(l even) or 31 (l odd) of r_(l div 2), set when x2 < 0 (s = +1); tag lane 29 takes r17's bits 31 (the sign of v) and
25 … 23 (m), lane 30 r16's 15 and 9 … 7, lane 31 r16's 31 and 25 … 23.  The map is specified on legal inputs (finite codes,
|x| <= 416); each primitive is total, reading a NaN code as 0.  The slice's F travels as the byte e(F) + 8.
"""
from __future__ import annotations

from fractions import Fraction

from verity.ir.defs import primitive
from verity.ir.types import Array, Value

U32, U8 = Value(32), Value(8)
M32 = (1 << 32) - 1
REAL = 29
E_BIAS = 8

MAG: list[Fraction] = []
for _c in range(0x7F):
    _e, _m = _c >> 3, _c & 7
    MAG.append(Fraction(_m, 1 << 9) if _e == 0 else Fraction(8 + _m) * Fraction(2) ** (_e - 10))
E4M3_MAX = MAG[0x7E]


def val(code: int) -> Fraction:
    """An E4M3 code's value; a NaN code reads as 0."""
    mag = code & 0x7F
    if mag == 0x7F:
        return Fraction(0)
    return -MAG[mag] if code & 0x80 else MAG[mag]


def enc(x: Fraction) -> int:
    """`h1_atom.enc`: RNE onto E4M3, ties to the even code, saturating at ±448; +0 for a negative value that rounds to 0."""
    a = min(abs(x), E4M3_MAX)
    hi = 0
    while hi < 126 and MAG[hi] < a:
        hi += 1
    lo = max(hi - 1, 0)
    if a == MAG[hi]:
        pick = hi
    else:
        dlo, dhi = a - MAG[lo], MAG[hi] - a
        pick = lo if dlo < dhi else hi if dhi < dlo else (lo if lo % 2 == 0 else hi)
    return pick | (0x80 if x < 0 and pick > 0 else 0)


def _log2_floor(v: Fraction) -> int:
    """floor(log2 v) for v > 0."""
    e = v.numerator.bit_length() - v.denominator.bit_length()
    if Fraction(2) ** e > v:
        e -= 1
    elif Fraction(2) ** (e + 1) <= v:
        e += 1
    return e


def slice_exp(codes) -> int:
    """e(F) for one slice's real lanes."""
    m = max(abs(val(c)) for c in codes)
    return -2 if m == 0 else max(_log2_floor(m) - 6, -2)


def mu(code: int, ef: int) -> Fraction:
    x = abs(val(code))
    p = Fraction(0) if x == 0 else Fraction(2) ** _log2_floor(x)
    return max(p / 8, Fraction(2) ** ef)


def _bit(r: int, sel: int) -> int:
    return ((r & M32) >> (31 if sel & 1 else 15)) & 1


@primitive("PouwH1tSliceExp", 1, [("x", Array(REAL, U8))], U8, conformance="protocol-defined")
def PouwH1tSliceExp(*x: int) -> int:
    """A slice's floor exponent e(F) = max(e(M) − 6, −2), M the max |x| over its 29 real lanes, as the byte e(F) + 8."""
    return (slice_exp(x) + E_BIAS) & 0xFF


@primitive("PouwH1tX1", 1, [("x", U8), ("e", U8), ("r", U32), ("sel", U32)], U8, conformance="protocol-defined")
def PouwH1tX1(x: int, e: int, r: int, sel: int) -> int:
    """A real lane's block-1 code x1 = enc(x + s·mu): s = +1 iff bit 15 (``sel`` even) or 31 (odd) of ``r`` is set."""
    s = 1 if _bit(r, sel) else -1
    return enc(val(x) + s * mu(x, (e & 0xFF) - E_BIAS))


@primitive("PouwH1tX2", 1, [("x", U8), ("e", U8), ("r", U32), ("sel", U32)], U8, conformance="protocol-defined")
def PouwH1tX2(x: int, e: int, r: int, sel: int) -> int:
    """A real lane's block-2 code x2 = enc(−s·mu)."""
    s = 1 if _bit(r, sel) else -1
    return enc(-s * mu(x, (e & 0xFF) - E_BIAS))


@primitive("PouwH1tTag", 1, [("e", U8), ("r", U32), ("sel", U32)], U8, conformance="protocol-defined")
def PouwH1tTag(e: int, r: int, sel: int) -> int:
    """A tag lane's block-1 code enc(v), v = s·32F·(1 + m/8): the sign bit 15 and m = bits 9 … 7 of ``r`` (``sel`` even),
    or bit 31 and bits 25 … 23 (odd); the bit set gives v < 0."""
    half = ((r & M32) >> (16 if sel & 1 else 0)) & 0xFFFF
    m = (half >> 7) & 7
    v = 32 * Fraction(2) ** ((e & 0xFF) - E_BIAS) * (1 + Fraction(m, 8))
    return enc(-v if half & 0x8000 else v)


@primitive("PouwE4m3Neg", 1, [("c", U8)], U8, conformance="protocol-defined")
def PouwE4m3Neg(c: int) -> int:
    """An E4M3 code negated: its sign bit flipped (a tag lane's x2 = enc(−v), v never 0)."""
    return (c ^ 0x80) & 0xFF


@primitive("PouwPackBE4Bytes", 1, [("b0", U8), ("b1", U8), ("b2", U8), ("b3", U8)], U32, conformance="protocol-defined")
def PouwPackBE4Bytes(b0: int, b1: int, b2: int, b3: int) -> int:
    """Four bytes (E4M3 codes) as a big-endian word: half of a SHA-512 message word."""
    return ((b0 & 0xFF) << 24) | ((b1 & 0xFF) << 16) | ((b2 & 0xFF) << 8) | (b3 & 0xFF)


__all__ = ["E_BIAS", "MAG", "REAL", "U32", "U8", "enc", "mu", "slice_exp", "val",
           *sorted(n for n in list(globals()) if n.startswith("Pouw"))]

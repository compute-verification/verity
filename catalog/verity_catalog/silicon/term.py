"""Operand formats and exact products: the *terms* a tensor-core accumulator step sums.

Everything here is integer arithmetic on raw encodings.  A :class:`Term` is an unnormalised signed binary
number on the FP32 significand grid (its ``exponent`` is that of bit 23); the ``*_product`` functions form the
*exact* product of two operand words as a Term, and :func:`fp32_term` / :func:`pack_fp32` move the FP32
accumulator word to and from that grid.  No rounding happens anywhere in this module.

Operand formats (OCP / IEEE bit layouts, sign first):

    bf16   1-8-7    bias 127   exponent field 255 is inf (frac 0) or NaN            subnormals at 2^-133
    e4m3   1-4-3    bias 7     NO infinities; the two words with exp=15, frac=7 are NaN   max 448, subnormals at 2^-9
    e5m2   1-5-2    bias 15    exponent field 31 is inf (frac 0) or NaN (IEEE-style)  max 57344, subnormals at 2^-16
    e2m1   1-2-1    bias 1     no specials at all: the 16 codes are +-{0, .5, 1, 1.5, 2, 3, 4, 6}

The decoders ``decode_<fmt>(word)`` are exact: a finite word becomes ``(sign, exp, sig)`` with the value
``(-1)^sign * sig * 2^exp`` (``sig`` an integer with the hidden bit included for normals, ``exp`` the weight of its
least significant bit), and a non-finite word becomes one of the strings ``"nan"``, ``"+inf"``, ``"-inf"``.

The BF16 and E4M3 products are Hawkeye's (``utils.cpp::multiply_bfloat16_to_gfloat`` /
``multiply_fp8e4m3_to_gfloat``, commit ``30703fcb``); E5M2 and E2M1 follow the same rule -- subnormal inputs keep
their unnormalised significand at the minimum exponent, nothing is normalised or flushed -- for formats Hawkeye's
simulators do not have.  Moved here from ``silicon.py`` (which re-exports every name) in the silicon Phase-0
consolidation; the BF16/E4M3 functions are byte-for-byte the veritor ``core/silicon.py`` (eb865ce29) ones.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .errors import InvalidArtifact

__all__ = [
    "FORMATS",
    "FP32_MAX_EXPONENT",
    "FP32_MIN_NORMAL_EXPONENT",
    "FP32_SIGNIFICAND_WIDTH",
    "Decoded",
    "Format",
    "PRODUCT_WIDTH",
    "Special",
    "Term",
    "ZERO",
    "bf16_product",
    "decode",
    "decode_bf16",
    "decode_e2m1",
    "decode_e4m3",
    "decode_e5m2",
    "e2m1_product",
    "e4m3_product",
    "e5m2_product",
    "fp32_term",
    "pack_fp32",
    "product",
    "term_value",
]

FP32_SIGNIFICAND_WIDTH = 24
FP32_MIN_NORMAL_EXPONENT = -126
FP32_MAX_EXPONENT = 127

# Products are carried with this many significand bits before alignment, so
# that a 25-bit-wide adder (Ampere) sees them unshifted.  BF16: 8x8 -> 16 bits,
# shifted by 9.  E4M3: 4x4 -> 8 bits, shifted by 17.  E5M2: 3x3 -> 6 bits,
# shifted by 19.  E2M1: 2x2 -> 4 bits, shifted by 21.  This matches Hawkeye.
PRODUCT_WIDTH = 25


@dataclass(frozen=True, slots=True)
class Term:
    """An unnormalised signed binary term ``(-1)^negative * magnitude * 2^(exponent-23)``.

    ``magnitude == 0`` is the (unsigned) zero; its ``exponent`` is ignored.
    The exponent is that of bit 23, so a normal FP32 value with biased
    exponent ``E`` has ``exponent = E - 127`` and a 24-bit magnitude.
    """

    negative: bool
    exponent: int
    magnitude: int


ZERO = Term(False, 0, 0)


def term_value(term: Term) -> tuple[int, int]:
    """The exact value of a Term as ``(numerator, exponent)``: ``numerator * 2^exponent``, numerator signed."""

    n = -term.magnitude if term.negative else term.magnitude
    return n, term.exponent - (FP32_SIGNIFICAND_WIDTH - 1)


# --- formats -----------------------------------------------------------------------------------------------------

Special = Literal["nan", "+inf", "-inf"]
Decoded = tuple[int, int, int] | Special


@dataclass(frozen=True, slots=True)
class Format:
    """One operand encoding: ``bits = 1 + exp_bits + man_bits``; ``nan_rule`` says which words are not finite.

    ``"ieee"``: an all-ones exponent field is +-inf (fraction 0) or NaN.  ``"e4m3"``: only exponent AND fraction
    all ones is NaN, there are no infinities.  ``"none"``: every word is finite (E2M1).
    """

    name: str
    exp_bits: int
    man_bits: int
    nan_rule: Literal["ieee", "e4m3", "none"]

    @property
    def bits(self) -> int:
        return 1 + self.exp_bits + self.man_bits

    @property
    def bias(self) -> int:
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def sig_bits(self) -> int:
        """Significand bits, hidden bit included."""
        return self.man_bits + 1

    @property
    def product_shift(self) -> int:
        """Left shift that puts the ``2 * sig_bits``-bit exact product at the top of :data:`PRODUCT_WIDTH`."""
        return PRODUCT_WIDTH - 2 * self.sig_bits

    @property
    def product_exponent_offset(self) -> int:
        """``exponent = (ea or 1) + (eb or 1) + offset`` for a product carried at ``product_shift``."""
        return FP32_SIGNIFICAND_WIDTH - 1 - self.product_shift - 2 * self.bias - 2 * self.man_bits

    @property
    def min_exponent(self) -> int:
        """Weight of the least significant bit of a subnormal (and of the minimum normal) significand."""
        return 1 - self.bias - self.man_bits

    def decode(self, word: int) -> Decoded:
        if not 0 <= word < (1 << self.bits):
            raise InvalidArtifact(f"{self.name} operand outside {self.bits} bits")
        sign = word >> (self.bits - 1)
        e = (word >> self.man_bits) & ((1 << self.exp_bits) - 1)
        m = word & ((1 << self.man_bits) - 1)
        all_ones = (1 << self.exp_bits) - 1
        if self.nan_rule == "ieee" and e == all_ones:
            if m:
                return "nan"
            return "-inf" if sign else "+inf"
        if self.nan_rule == "e4m3" and e == all_ones and m == (1 << self.man_bits) - 1:
            return "nan"
        if e == 0:
            return sign, self.min_exponent, m
        return sign, e - self.bias - self.man_bits, (1 << self.man_bits) | m

    def is_finite(self, word: int) -> bool:
        return not isinstance(self.decode(word), str)

    def product(self, a: int, b: int) -> Term:
        """Exact product of two words as a :data:`PRODUCT_WIDTH`-scaled Term; non-finite operands raise."""

        for word in (a, b):
            if isinstance(self.decode(word), str):
                raise InvalidArtifact(f"non-finite {self.name} operand")
        mask_m = (1 << self.man_bits) - 1
        mask_e = (1 << self.exp_bits) - 1
        ea, eb = (a >> self.man_bits) & mask_e, (b >> self.man_bits) & mask_e
        hidden = 1 << self.man_bits
        sa = (a & mask_m) | (hidden if ea else 0)
        sb = (b & mask_m) | (hidden if eb else 0)
        magnitude = (sa * sb) << self.product_shift
        if magnitude == 0:
            return ZERO
        negative = ((a ^ b) >> (self.bits - 1)) & 1 == 1
        return Term(negative, (ea or 1) + (eb or 1) + self.product_exponent_offset, magnitude)


BF16 = Format("bf16", 8, 7, "ieee")
E4M3 = Format("e4m3", 4, 3, "e4m3")
E5M2 = Format("e5m2", 5, 2, "ieee")
E2M1 = Format("e2m1", 2, 1, "none")

FORMATS: dict[str, Format] = {f.name: f for f in (BF16, E4M3, E5M2, E2M1)}


# --- products (the BF16 / E4M3 bodies are Hawkeye's, kept verbatim) -----------------------------------------------

def bf16_product(a: int, b: int) -> Term:
    """Exact product of two BF16 words as a 25-bit-scaled :class:`Term`.

    Subnormal inputs keep their unnormalised significand at exponent ``-126``
    (``utils.cpp::multiply_bfloat16_to_gfloat``).  Non-finite operands are
    outside the modelled domain and raise.
    """

    ea, eb = (a >> 7) & 0xFF, (b >> 7) & 0xFF
    if ea == 0xFF or eb == 0xFF:
        raise InvalidArtifact("non-finite BF16 operand")
    sa = (a & 0x7F) | (0x80 if ea else 0)
    sb = (b & 0x7F) | (0x80 if eb else 0)
    magnitude = (sa * sb) << 9
    if magnitude == 0:
        return ZERO
    return Term(((a ^ b) >> 15) & 1 == 1, (ea or 1) + (eb or 1) - 254, magnitude)


def e4m3_product(a: int, b: int) -> Term:
    """Exact product of two OCP E4M3 bytes as a 25-bit-scaled :class:`Term`.

    E4M3 has no infinities; ``0x7f``/``0xff`` are NaN and raise.  Subnormals
    keep their unnormalised significand at exponent ``-6``
    (``utils.cpp::multiply_fp8e4m3_to_gfloat``).
    """

    if (a & 0x7F) == 0x7F or (b & 0x7F) == 0x7F:
        raise InvalidArtifact("E4M3 NaN operand")
    ea, eb = (a >> 3) & 0xF, (b >> 3) & 0xF
    sa = (a & 0x7) | (0x8 if ea else 0)
    sb = (b & 0x7) | (0x8 if eb else 0)
    magnitude = (sa * sb) << 17
    if magnitude == 0:
        return ZERO
    return Term(((a ^ b) >> 7) & 1 == 1, (ea or 1) + (eb or 1) - 14, magnitude)


def e5m2_product(a: int, b: int) -> Term:
    """Exact product of two OCP E5M2 bytes as a 25-bit-scaled :class:`Term`.

    E5M2 is IEEE-like: exponent field 31 is an infinity (fraction 0) or a NaN; both raise.  Subnormals keep
    their unnormalised 2-bit significand at exponent ``-14``.  The 6-bit product is carried at bits 19..24
    (``PRODUCT_WIDTH``), so ``exponent = (ea or 1) + (eb or 1) - 30``.  No silicon has been measured with this
    product yet: it exists so the E5M2 *hypotheses* of ``instructions.py`` are executable.
    """

    return E5M2.product(a, b)


def e2m1_product(a: int, b: int) -> Term:
    """Exact product of two FP4 E2M1 nibbles as a 25-bit-scaled :class:`Term`.

    The sixteen codes are ``+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}`` (OCP MX; no NaN, no infinity), so every product is
    a multiple of ``1/4`` below ``36``.  The 4-bit product is carried at bits 21..24, ``exponent = (ea or 1) +
    (eb or 1) - 2``.  Block scales are *not* applied here: they belong to the model (``BlockScaledAlignAdd``).
    """

    return E2M1.product(a, b)


_PRODUCTS: dict[str, Callable[[int, int], Term]] = {
    "bf16": bf16_product,
    "e4m3": e4m3_product,
    "e5m2": e5m2_product,
    "e2m1": e2m1_product,
}


def product(dtype: str, a: int, b: int) -> Term:
    """The exact product function of ``dtype`` (``"bf16"``, ``"e4m3"``, ``"e5m2"``, ``"e2m1"``)."""

    try:
        return _PRODUCTS[dtype](a, b)
    except KeyError:
        raise InvalidArtifact(f"no product for dtype {dtype!r}") from None


# --- decoders --------------------------------------------------------------------------------------------------------

def decode_bf16(word: int) -> Decoded:
    """``(sign, exp, sig)`` with value ``(-1)^sign * sig * 2^exp``, or ``"nan"`` / ``"+inf"`` / ``"-inf"``."""

    return BF16.decode(word)


def decode_e4m3(word: int) -> Decoded:
    """As :func:`decode_bf16` for E4M3: only ``0x7F``/``0xFF`` are ``"nan"``; ``0x7E`` is 448."""

    return E4M3.decode(word)


def decode_e5m2(word: int) -> Decoded:
    """As :func:`decode_bf16` for E5M2: ``0x7C``/``0xFC`` are the infinities, ``0x7D..0x7F`` NaN."""

    return E5M2.decode(word)


def decode_e2m1(word: int) -> Decoded:
    """As :func:`decode_bf16` for E2M1: every nibble is finite."""

    return E2M1.decode(word)


def decode(dtype: str, word: int) -> Decoded:
    try:
        return FORMATS[dtype].decode(word)
    except KeyError:
        raise InvalidArtifact(f"no decoder for dtype {dtype!r}") from None


# --- the FP32 accumulator word ---------------------------------------------------------------------------------------

def fp32_term(bits: int) -> Term:
    """Decode FP32 bits into a 24-bit :class:`Term`; infinities and NaN raise."""

    exponent_field = (bits >> 23) & 0xFF
    if exponent_field == 0xFF:
        raise InvalidArtifact("non-finite FP32 accumulator")
    mantissa = bits & 0x7F_FFFF
    negative = bits >> 31 == 1
    if exponent_field == 0:
        if mantissa == 0:
            return ZERO
        return Term(negative, FP32_MIN_NORMAL_EXPONENT, mantissa)
    return Term(negative, exponent_field - 127, (1 << 23) | mantissa)


def pack_fp32(term: Term) -> int:
    """Encode a 24-bit :class:`Term` as FP32 bits (zero is ``+0``; overflow is ±inf).

    Mirrors ``Gfloat::operator float``: a magnitude without bit 23 set at
    exponent ``-126`` is a subnormal.
    """

    if term.magnitude == 0:
        return 0
    if term.exponent > FP32_MAX_EXPONENT:
        return (int(term.negative) << 31) | 0x7F80_0000
    exponent_field = term.exponent + 127
    if term.magnitude & (1 << 23) == 0:
        exponent_field -= 1
    return (
        (int(term.negative) << 31)
        | (exponent_field << 23)
        | (term.magnitude & 0x7F_FFFF)
    )

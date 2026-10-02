"""PoUW's IR primitives: the word operations `ncp-v2`'s circuit is built from.

Every primitive returns one value of at most 32 bits, so every unit of any `Q_word` cut of these Definitions meets the width
rule, and a 64-bit hash word is carried as two 32-bit halves: SHA-512's words as (hi, lo), Keccak's lanes as (lo, hi).
Encodings: `U32` = `Value<32>`; `I7` = `Value<7>`, a two's-complement int7 (A's and B's entries: the type is the int7 domain
check); `U8` = `Value<8>`, a byte read as u8 (X + β) or s8 (the noise, Y) as each primitive says.  Every evaluator is total on
encodings.
"""
from __future__ import annotations

from verity.ir.defs import primitive
from verity.ir.types import Array, Value

U32, U8, I7, F32 = Value(32), Value(8), Value(7), Value(32)
M32 = (1 << 32) - 1
M64 = (1 << 64) - 1
#: route U's bias: X + β is u8 on the int7 domain (Lean `biasU8Iff`); `schemes.ncp.BETA`
BETA = 128


def sx(v: int, bits: int) -> int:
    """The two's-complement value of a ``bits``-bit encoding."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def _rotr64(x: int, c: int) -> int:
    c &= 63
    return ((x >> c) | (x << (64 - c))) & M64 if c else x


def _halves(x: int) -> tuple[int, int]:
    return x >> 32, x & M32


# ---- 32-bit words ------------------------------------------------------------------------------------------------------

@primitive("PouwXor32", 1, [("a", U32), ("b", U32)], U32, conformance="protocol-defined")
def PouwXor32(a: int, b: int) -> int:
    """a xor b."""
    return (a ^ b) & M32


@primitive("PouwChi32", 1, [("a", U32), ("b", U32), ("c", U32)], U32, conformance="protocol-defined")
def PouwChi32(a: int, b: int, c: int) -> int:
    """Keccak's χ on one half-lane: a xor (not b and c)."""
    return (a ^ (~b & c)) & M32


@primitive("PouwRotl64Lo", 1, [("lo", U32), ("hi", U32), ("c", U32)], U32, conformance="protocol-defined")
def PouwRotl64Lo(lo: int, hi: int, c: int) -> int:
    """The low half of the 64-bit word ``hi:lo`` rotated left by ``c mod 64``."""
    x = ((hi & M32) << 32) | (lo & M32)
    return _rotr64(x, (64 - (c & 63)) & 63) & M32


@primitive("PouwRotl64Hi", 1, [("lo", U32), ("hi", U32), ("c", U32)], U32, conformance="protocol-defined")
def PouwRotl64Hi(lo: int, hi: int, c: int) -> int:
    """The high half of the 64-bit word ``hi:lo`` rotated left by ``c mod 64``."""
    x = ((hi & M32) << 32) | (lo & M32)
    return _rotr64(x, (64 - (c & 63)) & 63) >> 32


@primitive("PouwAdd64Lo", 1, [("alo", U32), ("blo", U32)], U32, conformance="protocol-defined")
def PouwAdd64Lo(alo: int, blo: int) -> int:
    """The low half of a 64-bit sum."""
    return (alo + blo) & M32


@primitive("PouwAdd64Hi", 1, [("ahi", U32), ("alo", U32), ("bhi", U32), ("blo", U32)], U32, conformance="protocol-defined")
def PouwAdd64Hi(ahi: int, alo: int, bhi: int, blo: int) -> int:
    """The high half of a 64-bit sum: the halves' sum plus the low halves' carry."""
    return (ahi + bhi + (((alo & M32) + (blo & M32)) >> 32)) & M32


def _sha(name: str, f):
    """SHA-512's Σ0, Σ1, σ0 or σ1 on the word ``hi:lo``, one half per primitive."""

    @primitive(f"PouwSha512{name}Hi", 1, [("hi", U32), ("lo", U32)], U32, conformance="protocol-defined")
    def hi_half(hi: int, lo: int) -> int:
        return f(((hi & M32) << 32) | (lo & M32)) >> 32

    @primitive(f"PouwSha512{name}Lo", 1, [("hi", U32), ("lo", U32)], U32, conformance="protocol-defined")
    def lo_half(hi: int, lo: int) -> int:
        return f(((hi & M32) << 32) | (lo & M32)) & M32

    return hi_half, lo_half


PouwSha512BigSigma0Hi, PouwSha512BigSigma0Lo = _sha("BigSigma0", lambda x: _rotr64(x, 28) ^ _rotr64(x, 34) ^ _rotr64(x, 39))
PouwSha512BigSigma1Hi, PouwSha512BigSigma1Lo = _sha("BigSigma1", lambda x: _rotr64(x, 14) ^ _rotr64(x, 18) ^ _rotr64(x, 41))
PouwSha512SmallSigma0Hi, PouwSha512SmallSigma0Lo = _sha("SmallSigma0", lambda x: _rotr64(x, 1) ^ _rotr64(x, 8) ^ (x >> 7))
PouwSha512SmallSigma1Hi, PouwSha512SmallSigma1Lo = _sha("SmallSigma1", lambda x: _rotr64(x, 19) ^ _rotr64(x, 61) ^ (x >> 6))


@primitive("PouwCh32", 1, [("e", U32), ("f", U32), ("g", U32)], U32, conformance="protocol-defined")
def PouwCh32(e: int, f: int, g: int) -> int:
    """SHA-512's Ch on one half: (e and f) xor (not e and g)."""
    return ((e & f) ^ (~e & g)) & M32


@primitive("PouwMaj32", 1, [("a", U32), ("b", U32), ("c", U32)], U32, conformance="protocol-defined")
def PouwMaj32(a: int, b: int, c: int) -> int:
    """SHA-512's Maj on one half."""
    return ((a & b) ^ (a & c) ^ (b & c)) & M32


# ---- bytes and int7 entries --------------------------------------------------------------------------------------------

@primitive("PouwPackBE4", 1, [("b0", I7), ("b1", I7), ("b2", I7), ("b3", I7)], U32, conformance="protocol-defined")
def PouwPackBE4(b0: int, b1: int, b2: int, b3: int) -> int:
    """Four int7 entries as their two's-complement bytes, big-endian: SHA-512's message word half."""
    return sum((sx(b, 7) & 0xFF) << (24 - 8 * i) for i, b in enumerate((b0, b1, b2, b3)))


@primitive("PouwNoiseByte", 1, [("w", U32), ("c", U32)], U8, conformance="protocol-defined")
def PouwNoiseByte(w: int, c: int) -> int:
    """A noise entry from XOF byte ``c mod 4`` of the half-lane ``w`` (little-endian): its low six bits minus 31, exactly
    uniform on [−31, 32], as an s8 byte (`schemes.ncp.noise_row`)."""
    return (((w >> (8 * (c & 3))) & 63) - 31) & 0xFF


@primitive("PouwFormX", 1, [("a", I7), ("e", U8)], U8, conformance="protocol-defined")
def PouwFormX(a: int, e: int) -> int:
    """An entry of X + β from an int7 activation and an s8 noise entry: a + e + 128, as u8."""
    return (sx(a, 7) + sx(e, 8) + BETA) & 0xFF


@primitive("PouwFormX3", 1, [("a", I7), ("e1", U8), ("e2", U8)], U8, conformance="protocol-defined")
def PouwFormX3(a: int, e1: int, e2: int) -> int:
    """An entry of the third block of X + β: 128 − (a + e1 + e2), as u8."""
    return (BETA - (sx(a, 7) + sx(e1, 8) + sx(e2, 8))) & 0xFF


@primitive("PouwAdd8", 1, [("b", I7), ("f", U8)], U8, conformance="protocol-defined")
def PouwAdd8(b: int, f: int) -> int:
    """An entry of Y's first block: b + f, as s8."""
    return (sx(b, 7) + sx(f, 8)) & 0xFF


@primitive("PouwSub8", 1, [("b", I7), ("f", U8)], U8, conformance="protocol-defined")
def PouwSub8(b: int, f: int) -> int:
    """An entry of Y's second block: b − f, as s8 (F₂ = −R·F₁ with every sign +1)."""
    return (sx(b, 7) - sx(f, 8)) & 0xFF


@primitive("PouwAddSub8", 1, [("b", I7), ("f", U8), ("g", U8)], U8, conformance="protocol-defined")
def PouwAddSub8(b: int, f: int, g: int) -> int:
    """An entry of Y's third block: b + f − g, as s8."""
    return (sx(b, 7) + sx(f, 8) - sx(g, 8)) & 0xFF


# ---- the accumulator ---------------------------------------------------------------------------------------------------

A16 = Array(16, U8)


@primitive("PouwNegBetaSum16", 1, [("acc", U32), ("y", A16)], U32, conformance="protocol-defined")
def PouwNegBetaSum16(acc: int, *y: int) -> int:
    """acc − β·Σ y over 16 s8 entries, mod 2^32: the route-U initial value c₀ = −β·Σ_l Y[l][j], 16 entries at a time."""
    return (acc - BETA * sum(sx(v, 8) for v in y)) & M32


@primitive("PouwDot16U8S8", 1, [("acc", U32), ("x", A16), ("y", A16)], U32, conformance="protocol-defined")
def PouwDot16U8S8(acc: int, *xy: int) -> int:
    """One 16-deep route-U step: acc + Σ (X + β)·Y over 16 u8 × s8 products, mod 2^32 (Lean `mmWU`)."""
    x, y = xy[:16], xy[16:]
    return (acc + sum((a & 0xFF) * sx(b, 8) for a, b in zip(x, y))) & M32


# ---- dequantization ----------------------------------------------------------------------------------------------------

def i32_to_f32_rn(z: int) -> int:
    """cvt.rn.f32.s32: an int32 as the nearest FP32, ties to even, as its encoding."""
    v = sx(z, 32)
    if v == 0:
        return 0
    sign = 0x80000000 if v < 0 else 0
    m = -v if v < 0 else v
    e = m.bit_length() - 1                                  # m = 1.f × 2^e, e <= 31
    if e > 23:
        shift = e - 23
        q, r = m >> shift, m & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if r > half or (r == half and q & 1):
            q += 1
        if q >> 24:                                         # rounding carried into a new binade
            q >>= 1
            e += 1
        mant = q & 0x7FFFFF
    else:
        mant = (m << (23 - e)) & 0x7FFFFF
    return sign | ((e + 127) << 23) | mant


@primitive("PouwI32ToF32Rn", 1, [("z", U32)], F32, conformance="protocol-defined")
def PouwI32ToF32Rn(z: int) -> int:
    """cvt.rn.f32.s32: Z as FP32, round to nearest even."""
    return i32_to_f32_rn(z)


__all__ = ["A16", "BETA", "F32", "I7", "M32", "U32", "U8", "sx", "i32_to_f32_rn",
           *sorted(n for n in list(globals()) if n.startswith("Pouw"))]

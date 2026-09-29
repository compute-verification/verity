"""The IR primitives NCP-FP8's circuit adds to `words`: binary16 arithmetic, FP32 sqrt, SHA-256's word functions and byte
placement.

The arithmetic is #295's reference, `verity.ml.tc.fp16` and `verity.ml.tc.fp32`, one operation per primitive, so the circuit
computes what `schemes.fp8_is` computes by construction.  Those functions are specified on finite words and raise on the
others; each primitive here extends its function to every encoding with one fixed NaN word (`F16_NAN`, `F32_NAN`), which the
D-3 forming never produces (its inputs are E4M3 codes and finite f16 values).  Every output is at most 32 bits.
"""
from __future__ import annotations

from verity.ir.defs import primitive
from verity.ir.types import Value
from verity.ml.tc import fp16, fp32
from verity.ml.tc.errors import InvalidArtifact

U32, U16, U8 = Value(32), Value(16), Value(8)
M32 = (1 << 32) - 1
F16_NAN, F32_NAN = 0x7E00, 0x7FC0_0000


def _total(nan: int, f, *args: int) -> int:
    try:
        return f(*args)
    except (InvalidArtifact, ValueError, ZeroDivisionError):
        return nan


def _rotr32(x: int, c: int) -> int:
    return ((x >> c) | (x << (32 - c))) & M32


# ---- SHA-256 (FIPS 180-4) --------------------------------------------------------------------------------------------

@primitive("PouwAdd32", 1, [("a", U32), ("b", U32)], U32, conformance="protocol-defined")
def PouwAdd32(a: int, b: int) -> int:
    """a + b mod 2^32."""
    return (a + b) & M32


@primitive("PouwSha256BigSigma0", 1, [("x", U32)], U32, conformance="protocol-defined")
def PouwSha256BigSigma0(x: int) -> int:
    """SHA-256's Σ0: ROTR 2 xor ROTR 13 xor ROTR 22."""
    return _rotr32(x, 2) ^ _rotr32(x, 13) ^ _rotr32(x, 22)


@primitive("PouwSha256BigSigma1", 1, [("x", U32)], U32, conformance="protocol-defined")
def PouwSha256BigSigma1(x: int) -> int:
    """SHA-256's Σ1: ROTR 6 xor ROTR 11 xor ROTR 25."""
    return _rotr32(x, 6) ^ _rotr32(x, 11) ^ _rotr32(x, 25)


@primitive("PouwSha256SmallSigma0", 1, [("x", U32)], U32, conformance="protocol-defined")
def PouwSha256SmallSigma0(x: int) -> int:
    """SHA-256's σ0: ROTR 7 xor ROTR 18 xor SHR 3."""
    return _rotr32(x, 7) ^ _rotr32(x, 18) ^ ((x & M32) >> 3)


@primitive("PouwSha256SmallSigma1", 1, [("x", U32)], U32, conformance="protocol-defined")
def PouwSha256SmallSigma1(x: int) -> int:
    """SHA-256's σ1: ROTR 17 xor ROTR 19 xor SHR 10."""
    return _rotr32(x, 17) ^ _rotr32(x, 19) ^ ((x & M32) >> 10)


# ---- bytes into words ---------------------------------------------------------------------------------------------------

@primitive("PouwMove32", 1, [("x", U32), ("s", U32), ("n", U32), ("d", U32)], U32, conformance="protocol-defined")
def PouwMove32(x: int, s: int, n: int, d: int) -> int:
    """Bits s … s + n − 1 of ``x`` placed at bit ``d``: ((x >> s) mod 2^n) << d, mod 2^32 (s, n, d read mod 64)."""
    s, n, d = s & 63, n & 63, d & 63
    return ((((x & M32) >> s) & ((1 << n) - 1)) << d) & M32


@primitive("PouwPlace8", 1, [("b", U8), ("d", U32)], U32, conformance="protocol-defined")
def PouwPlace8(b: int, d: int) -> int:
    """A byte placed at bit ``d``: b << d, mod 2^32 (d read mod 64)."""
    return ((b & 0xFF) << (d & 63)) & M32


@primitive("PouwPackBE4U8", 1, [("b0", U8), ("b1", U8), ("b2", U8), ("b3", U8)], U32, conformance="protocol-defined")
def PouwPackBE4U8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Four bytes as a big-endian word: SHA-256's message word."""
    return ((b0 & 0xFF) << 24) | ((b1 & 0xFF) << 16) | ((b2 & 0xFF) << 8) | (b3 & 0xFF)


@primitive("PouwF16Field", 1, [("h", U32), ("sel", U32)], U16, conformance="protocol-defined")
def PouwF16Field(h: int, sel: int) -> int:
    """D-3's noise field from XOF bytes: t = 0x3C00 | (w & 0x3FF), w the little-endian 16-bit half ``sel mod 2`` of the
    half-lane ``h`` (an f16 in [1, 2))."""
    return 0x3C00 | (((h & M32) >> (16 * (sel & 1))) & 0x3FF)


# ---- binary16 (`verity.ml.tc.fp16`, #295) --------------------------------------------------------------------------------

@primitive("PouwF16FromE4m3", 1, [("c", U8)], U16, conformance="protocol-defined")
def PouwF16FromE4m3(c: int) -> int:
    """``cvt.rn.f16x2.e4m3x2``'s lane: an E4M3 code as its binary16 word (exact)."""
    return _total(F16_NAN, fp16.from_e4m3, c & 0xFF)


@primitive("PouwF16Abs", 1, [("a", U16)], U16, conformance="protocol-defined")
def PouwF16Abs(a: int) -> int:
    """|a|: the sign bit cleared."""
    return a & 0x7FFF


@primitive("PouwF16Neg", 1, [("a", U16)], U16, conformance="protocol-defined")
def PouwF16Neg(a: int) -> int:
    """``neg.f16``: the sign bit flipped (`fp16.neg`)."""
    return _total(F16_NAN, fp16.neg, a & 0xFFFF)


@primitive("PouwF16Add", 1, [("a", U16), ("b", U16)], U16, conformance="protocol-defined")
def PouwF16Add(a: int, b: int) -> int:
    """``add.rn.f16`` (`fp16.add`)."""
    return _total(F16_NAN, fp16.add, a & 0xFFFF, b & 0xFFFF)


@primitive("PouwF16Mul", 1, [("a", U16), ("b", U16)], U16, conformance="protocol-defined")
def PouwF16Mul(a: int, b: int) -> int:
    """``mul.rn.f16`` (`fp16.mul`)."""
    return _total(F16_NAN, fp16.mul, a & 0xFFFF, b & 0xFFFF)


@primitive("PouwF16Fma", 1, [("a", U16), ("b", U16), ("c", U16)], U16, conformance="protocol-defined")
def PouwF16Fma(a: int, b: int, c: int) -> int:
    """``fma.rn.f16``: a·b + c rounded once (`fp16.fma`)."""
    return _total(F16_NAN, fp16.fma, a & 0xFFFF, b & 0xFFFF, c & 0xFFFF)


@primitive("PouwF16Max", 1, [("a", U16), ("b", U16)], U16, conformance="protocol-defined")
def PouwF16Max(a: int, b: int) -> int:
    """``max.f16`` on finite words (`fp16.max_`)."""
    return _total(F16_NAN, fp16.max_, a & 0xFFFF, b & 0xFFFF)


@primitive("PouwF16ToF32", 1, [("a", U16)], U32, conformance="protocol-defined")
def PouwF16ToF32(a: int) -> int:
    """``cvt.f32.f16``: the exact FP32 word (`fp16.to_f32`)."""
    return _total(F32_NAN, fp16.to_f32, a & 0xFFFF)


def _f32_to_f16(x: int) -> int:
    return fp16.round_to_f16(*fp32.decode(x))


@primitive("PouwF32ToF16Rn", 1, [("x", U32)], U16, conformance="protocol-defined")
def PouwF32ToF16Rn(x: int) -> int:
    """An FP32 word rounded to the nearest binary16, ties to even (`fp16.round_to_f16` of `fp32.decode`)."""
    return _total(F16_NAN, _f32_to_f16, x & M32)


@primitive("PouwF16ToE4m3", 1, [("a", U16)], U8, conformance="protocol-defined")
def PouwF16ToE4m3(a: int) -> int:
    """``cvt.rn.satfinite.e4m3x2.f16x2``'s lane: RN-even onto E4M3, saturating to ±448 (`fp16.to_e4m3`)."""
    return _total(0x7F, fp16.to_e4m3, a & 0xFFFF)


@primitive("PouwF32Sqrt", 1, [("x", U32)], U32, conformance="protocol-defined")
def PouwF32Sqrt(x: int) -> int:
    """``sqrt.rn.f32`` (`verity.ml.tc.fp32.sqrt`)."""
    return _total(F32_NAN, fp32.sqrt, x & M32)


__all__ = ["F16_NAN", "F32_NAN", "U16", "U32", "U8", *sorted(n for n in list(globals()) if n.startswith("Pouw"))]

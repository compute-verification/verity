"""torch 2.14's element-wise ops on int64, uint8 and f16 words, and the radix order of f32 words, as word primitives: what a
training step's quantizer computes beside its f32 and bf16 arithmetic (a count, a clamp, a code, a table index, an f16 scale, a
k-th smallest value), each as torch's CUDA kernels compute it, which is torch CPU's too wherever a docstring says nothing else.

    I64Add_v1       (a, b: I64) -> I64            a + b modulo 2^64 (torch's int64 add wraps)
    I64Le_v1        (a, b: I64) -> BIT            signed a <= b
    I64Shl_v1       (a, b: I64) -> I64            a << b modulo 2^64 (``bitwise_left_shift``), 0 for b < 0 or b >= 64
    I64Shr_v1       (a, b: I64) -> I64            a >> b, the sign shifted in (``bitwise_right_shift``), by 63 for b < 0 or b >= 63
    I64And_v1       (a, b: I64) -> I64            a & b
    SelectI64_v1    (c: BIT, a, b: I64) -> I64    c ? a : b
    I64ToU8_v1      (t: I64) -> U8                the low byte: C++'s conversion to uint8_t, modulo 2^8
    U8ToI64_v1      (t: U8) -> I64                zero-extended
    I64ToF32Rn_v1   (t: I64) -> F32               ``cvt.rn.f32.s64`` (and x86's ``cvtsi2ss``): round to nearest even
    F32ToI64Rz_v1   (x: F32) -> I64               ``cvt.rzi.s64.f32``: truncated toward zero, from 2^63 up INT64_MAX, below -2^63
                                                  INT64_MIN, a NaN 0
    F32ToF16Rn_v1   (x: F32) -> F16               ``__float2half``, ``cvt.rn.f16.f32``: ties to even, subnormals kept, overflow to
                                                  infinity, every NaN 0x7FFF
    F16ToF32_v1     (h: F16) -> F32               ``__half2float``: exact, every NaN 0x7FFFFFFF
    F32OrderKey_v1  (x: F32) -> U32               the radix key torch's CUDA topk and kthvalue select on
                                                  (``TopKTypeConfig<float>::convert``): a nonnegative word with its sign bit set, a
                                                  negative one complemented, a NaN 0xFFFFFFFF; -0 sorts before +0
    OrderKeyF32_v1  (k: U32) -> F32               its inverse (``deconvert``), so the key 0xFFFFFFFF is 0x7FFFFFFF

(I64 is ``Value<64>``, U8 ``Value<8>``, F16 ``Value<16>``, F32 and U32 ``Value<32>``, BIT ``Value<1>``.)

The f16 conversions' NaN words are the GPU's: torch CPU's f32 -> f16 keeps a NaN's payload in its F16C vector body and makes it
0x7E00 in its scalar tail, and its f16 -> f32 keeps the payload, quieting a signaling NaN in the vector body only.  So is the
truncation's saturation: torch CPU's (x86's ``cvttss2si``) is INT64_MIN for a NaN and from 2^63 up.
"""
from __future__ import annotations

import numpy as np

from verity.circuits.defs import primitive
from verity.circuits.types import Value

__all__ = ["F16ToF32", "F32OrderKey", "F32ToF16Rn", "F32ToI64Rz", "I64Add", "I64And", "I64Le", "I64Shl", "I64Shr", "I64ToF32Rn",
           "I64ToU8", "OrderKeyF32", "SelectI64", "U8ToI64", "TORCH_WORDS", "f32_rn_of_int"]

I64, U8, F16, F32, BIT = Value(64), Value(8), Value(16), Value(32), Value(1)
M64, M32 = (1 << 64) - 1, 0xFFFFFFFF
F16_NAN, F32_NAN = 0x7FFF, 0x7FFFFFFF
_EXACT = ("exact-model-tested: integer arithmetic on the words, torch's on CPU and GPU alike "
          "(verity/definitions/tests/test_torch_words.py)")
_PTX = ("model-untested on the GPU: as the PTX ISA and torch 2.14's CUDA source define it; torch CPU agrees on every word but a "
        "NaN (verity/definitions/tests/test_torch_words.py)")
_PTX_SAT = ("model-untested on the GPU: as the PTX ISA and torch 2.14's CUDA source define it; torch CPU agrees on every word but a "
            "NaN and from 2^63 up (verity/definitions/tests/test_torch_words.py)")


def _s64(t: int) -> int:
    t &= M64
    return t - (1 << 64) if t >> 63 else t


def f32_rn_of_int(v: int) -> int:
    """The f32 word nearest to the integer v, ties to even (exact integer arithmetic: no double rounding through f64)."""
    m = abs(v)
    sign = 0x80000000 if v < 0 else 0
    if m < 1 << 24:
        return sign | int(np.float32(m).view(np.uint32))
    shift = m.bit_length() - 24
    q, r = divmod(m, 1 << shift)
    half = 1 << (shift - 1)
    if r > half or (r == half and q & 1):
        q += 1
    return sign | int(np.float32(float(q) * 2.0 ** shift).view(np.uint32))


@primitive("I64Add", 1, [("a", I64), ("b", I64)], I64, conformance=_EXACT)
def I64Add(a: int, b: int) -> int:
    """a + b modulo 2^64."""
    return (a + b) & M64


@primitive("I64Le", 1, [("a", I64), ("b", I64)], BIT, conformance=_EXACT)
def I64Le(a: int, b: int) -> int:
    """a <= b as signed 64-bit integers."""
    return 1 if _s64(a) <= _s64(b) else 0


@primitive("I64Shl", 1, [("a", I64), ("b", I64)], I64, conformance=_EXACT)
def I64Shl(a: int, b: int) -> int:
    """a shifted left by b modulo 2^64, as unsigned; 0 for b < 0 or b >= 64 (torch's kernels test the amount first)."""
    b = _s64(b)
    return 0 if b < 0 or b >= 64 else (a << b) & M64


@primitive("I64Shr", 1, [("a", I64), ("b", I64)], I64, conformance=_EXACT)
def I64Shr(a: int, b: int) -> int:
    """a shifted right by b with its sign shifted in; by 63 for b < 0 or b >= 63, every bit the sign."""
    b = _s64(b)
    return (_s64(a) >> (63 if b < 0 or b >= 63 else b)) & M64


@primitive("I64And", 1, [("a", I64), ("b", I64)], I64, conformance=_EXACT)
def I64And(a: int, b: int) -> int:
    """a & b on 64-bit words."""
    return a & b & M64


@primitive("SelectI64", 1, [("c", BIT), ("a", I64), ("b", I64)], I64, conformance=_EXACT)
def SelectI64(c: int, a: int, b: int) -> int:
    """c ? a : b on 64-bit words."""
    return (a if c & 1 else b) & M64


@primitive("I64ToU8", 1, [("t", I64)], U8, conformance=_EXACT)
def I64ToU8(t: int) -> int:
    """The low byte of t: ``static_cast<uint8_t>``, modulo 2^8."""
    return t & 0xFF


@primitive("U8ToI64", 1, [("t", U8)], I64, conformance=_EXACT)
def U8ToI64(t: int) -> int:
    """The byte t zero-extended to 64 bits."""
    return t & 0xFF


@primitive("I64ToF32Rn", 1, [("t", I64)], F32, conformance=_EXACT)
def I64ToF32Rn(t: int) -> int:
    """The signed 64-bit integer t rounded to f32, ties to even (``cvt.rn.f32.s64``; 0 is +0)."""
    return f32_rn_of_int(_s64(t))


def _f32_nan(x: int) -> bool:
    return (x & 0x7F800000) == 0x7F800000 and (x & 0x7FFFFF) != 0


@primitive("F32ToI64Rz", 1, [("x", F32)], I64, conformance=_PTX_SAT)
def F32ToI64Rz(x: int) -> int:
    """``cvt.rzi.s64.f32`` (torch's CUDA conversion to int64, and to uint8 through it): x truncated toward zero, saturated
    (from 2^63 up, +inf too, INT64_MAX; below -2^63, -inf too, INT64_MIN); a NaN 0."""
    x &= M32
    if _f32_nan(x):
        return 0
    v = float(np.array([x], dtype=np.uint32).view(np.float32)[0])
    if v >= 2.0 ** 63:
        return (1 << 63) - 1
    if v < -2.0 ** 63:
        return 1 << 63
    return int(v) & M64


@primitive("F32ToF16Rn", 1, [("x", F32)], F16, conformance=_PTX)
def F32ToF16Rn(x: int) -> int:
    """``cvt.rn.f16.f32``: x rounded to f16, ties to even, f16 subnormals kept, past 65504 (by the rounding) infinity; every NaN
    0x7FFF."""
    x &= M32
    if _f32_nan(x):
        return F16_NAN
    return int(np.array([x], dtype=np.uint32).view(np.float32).astype(np.float16).view(np.uint16)[0])


@primitive("F16ToF32", 1, [("h", F16)], F32, conformance=_PTX)
def F16ToF32(h: int) -> int:
    """``__half2float``: the f16 word h widened to f32, exactly; every NaN 0x7FFFFFFF."""
    h &= 0xFFFF
    if (h & 0x7C00) == 0x7C00 and h & 0x3FF:
        return F32_NAN
    return int(np.array([h], dtype=np.uint16).view(np.float16).astype(np.float32).view(np.uint32)[0])


@primitive("F32OrderKey", 1, [("x", F32)], F32, conformance=_EXACT)
def F32OrderKey(x: int) -> int:
    """``TopKTypeConfig<float>::convert``: a NaN 0xFFFFFFFF; a word with its sign set complemented; any other with its sign bit set.
    Unsigned order of keys is the order of the values, -0 before +0 and every NaN last."""
    x &= M32
    if _f32_nan(x):
        return M32
    return x ^ M32 if x >> 31 else x ^ 0x80000000


@primitive("OrderKeyF32", 1, [("k", F32)], F32, conformance=_EXACT)
def OrderKeyF32(k: int) -> int:
    """``TopKTypeConfig<float>::deconvert``: a key with its top bit set has it cleared, any other is complemented."""
    k &= M32
    return k ^ 0x80000000 if k >> 31 else k ^ M32


TORCH_WORDS = [I64Add, I64Le, I64Shl, I64Shr, I64And, SelectI64, I64ToU8, U8ToI64, I64ToF32Rn, F32ToI64Rz, F32ToF16Rn, F16ToF32,
               F32OrderKey, OrderKeyF32]

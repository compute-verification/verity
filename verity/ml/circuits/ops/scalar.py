"""Registered scalar primitives: the basic integer, bit, select and comparison operations, the FP32 instructions
whose semantics is a bit operation or an exact comparison on the encoding (`abs`, `neg`, `max`, `min`, `cvt.sat`), and the
f32 -> s8 rounding cast (`cvt.rni.sat.s8.f32`).

Encodings: f32 = `Value<32>`, bf16 = `Value<16>`, i32 = `Value<32>` (two's complement), s8 = `Value<8>` (two's complement),
bit = `Value<1>`.  Every evaluator is integer arithmetic on the encoding, so no host floating-point state (flush-to-zero,
denormals-are-zero) reaches it.  An id that appears in stored Programs registered by the vLLM integration denotes the
function it denoted there.
"""
from __future__ import annotations

from verity.ml.circuits.defs import primitive
from verity.ml.circuits.types import Value

from verity.ml.circuits.silicon.cast import bf16_is_nan, f32_is_nan

F32, BF16, I32, S8, U16, BIT = Value(32), Value(16), Value(32), Value(8), Value(16), Value(1)
#: the GPU's canonical FP32 NaN: max.f32 / min.f32 of two NaNs, and every NaN result of add, sub, mul, div and fma (`fp32`'s `_v2`), an
#: invalid operation and a NaN operand alike, no operand's payload, sign or signalling bit kept (measured on the L40S and H100:
#: every build, PTX and C++ operators, ftz on and off)
F32_NAN_PTX = 0x7FFFFFFF

__all__ = ["BIT", "BF16", "F32", "I32", "S8", "U16", "Bf16GtStrict", "BitAnd", "BitNot", "BitOr", "F32BitsShl23", "F32Eq",
           "F32Fabs", "F32Fmaxf", "F32Fminf", "F32IsFinite", "F32Neg", "F32Sat", "F32ToS8RniSat", "I32Add", "I32Eq", "I32Le",
           "SelectBf16", "SelectF32", "SelectI32", "U32Hi16"]


def _order(u: int, width: int) -> int:
    """An integer whose order is the value order of the non-NaN float encodings of ``width`` bits; ``+0`` and ``-0`` share 0."""
    sign = 1 << (width - 1)
    u &= (sign << 1) - 1
    m = u & (sign - 1)
    return -m if u & sign else m


# ---- FP32 instructions that act on the encoding ------------------------------------------------------------------------


@primitive("F32Fabs", 1, [("x", F32)], F32, conformance="exact-model-tested")
def F32Fabs(x: int) -> int:
    """`fabsf`: clear the sign bit (NaN payloads kept: documented, not measured)."""
    return x & 0x7FFFFFFF


@primitive("F32Neg", 1, [("a", F32)], F32, conformance="exact-model-tested")
def F32Neg(a: int) -> int:
    """`neg.f32`: flip the sign bit (exact; -0 for +0, a NaN keeps its payload: documented, not measured)."""
    return (a ^ 0x80000000) & 0xFFFFFFFF


def _fmaxmin(a: int, b: int, want_max: bool) -> int:
    """CUDA `fmaxf`/`fminf` (IEEE maxNum/minNum, PTX `max.f32`/`min.f32`): a NaN operand yields the other operand,
    two NaNs the canonical NaN; `+0` is treated as greater than `-0`."""
    na, nb = f32_is_nan(a), f32_is_nan(b)
    if na and nb:
        return F32_NAN_PTX
    if na:
        return b
    if nb:
        return a
    ka, kb = _order(a, 32), _order(b, 32)
    if ka == kb:  # equal values: only ±0 can differ in encoding
        if (a & 0x7FFFFFFF) == 0 and (b & 0x7FFFFFFF) == 0:
            pos = 0 if (a == 0 or b == 0) else 0x80000000
            neg = 0x80000000 if (a == 0x80000000 or b == 0x80000000) else 0
            return pos if want_max else neg
        return a
    if want_max:
        return a if ka > kb else b
    return a if ka < kb else b


@primitive("F32Fmaxf", 1, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32Fmaxf(a: int, b: int) -> int:
    """`fmaxf(a, b)` (PTX `max.f32`): the larger operand; NaN yields the other operand; `max(+0, -0) = +0`."""
    return _fmaxmin(a, b, True)


@primitive("F32Fminf", 1, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32Fminf(a: int, b: int) -> int:
    """`fminf(a, b)` (PTX `min.f32`): the smaller operand; NaN yields the other operand; `min(+0, -0) = -0`."""
    return _fmaxmin(a, b, False)


@primitive("F32Sat", 1, [("a", F32)], F32, conformance="exact-model-tested")
def F32Sat(a: int) -> int:
    """`cvt.sat.f32.f32`: clamp to [+0.0, 1.0]; NaN -> +0.0 (PTX ISA: "NaN results are flushed to positive zero");
    -0.0 -> +0.0 (the saturated range's lower bound is +0)."""
    u = a & 0xFFFFFFFF
    if f32_is_nan(u) or u >> 31:
        return 0
    return 0x3F800000 if u >= 0x3F800000 else u


@primitive("F32ToS8RniSat", 1, [("x", F32)], S8, conformance="exact-model-tested")
def F32ToS8RniSat(x: int) -> int:
    """`cvt.rni.sat.s8.f32`: x rounded to the nearest integer, ties to even, saturated to [-128, 127] (the infinities too),
    NaN 0; the s8 code. The same as torch's ``round(x).clamp(-128, 127).to(torch.int8)`` on CPU, NaN included."""
    u = x & 0xFFFFFFFF
    if f32_is_nan(u):
        return 0
    e, neg = (u >> 23) & 0xFF, u >> 31
    sig = (u & 0x7FFFFF) | (0x800000 if e else 0)
    sh = 150 - max(e, 1)                 # |x| = sig 2^-sh
    if sh <= 0:
        v = sig << -sh
    else:
        q, r, half = sig >> sh, sig & ((1 << sh) - 1), 1 << (sh - 1)
        v = q + (r > half or (r == half and q & 1))
    v = min(v, 128 if neg else 127)
    return (-v if neg else v) & 0xFF


@primitive("U32Hi16", 1, [("a", I32)], U16, conformance="protocol-defined")
def U32Hi16(a: int) -> int:
    """A 32-bit word's high 16 bits."""
    return (a >> 16) & 0xFFFF


@primitive("F32BitsShl23", 1, [("a", F32)], F32, conformance="exact-model-tested")
def F32BitsShl23(a: int) -> int:
    """`mov.b32 r, a; shl.b32 r, r, 23; mov.b32 f, r`: the encoding shifted left by 23 (mod 2^32) read back as f32.  For
    `a = 12582912 + n` (`0x4B400000 | n`, n in [1, 253]) this is the normal f32 `2^(n - 127)` (the exponent-scaling step
    of CUDA's `expf`: the magic constant's own bits shift out)."""
    return ((a & 0xFFFFFFFF) << 23) & 0xFFFFFFFF


# ---- comparisons -------------------------------------------------------------------------------------------------------


@primitive("F32IsFinite", 1, [("a", F32)], BIT, conformance="model-untested")
def F32IsFinite(a: int) -> int:
    """1 iff the f32 value is neither NaN nor +/-inf."""
    return 0 if (a & 0x7F800000) == 0x7F800000 else 1


@primitive("F32Eq", 1, [("a", F32), ("b", F32)], BIT, conformance="exact-model-tested")
def F32Eq(a: int, b: int) -> int:
    """IEEE fp32 `a == b` (+0 == -0; NaN unequal to everything)."""
    if f32_is_nan(a & 0xFFFFFFFF) or f32_is_nan(b & 0xFFFFFFFF):
        return 0
    return 1 if _order(a, 32) == _order(b, 32) else 0


@primitive("Bf16GtStrict", 1, [("a", BF16), ("b", BF16)], BIT, conformance="exact-model-tested")
def Bf16GtStrict(a: int, b: int) -> int:
    """a > b on bf16 values (NaN compares false; +0 and -0 are equal)."""
    if bf16_is_nan(a & 0xFFFF) or bf16_is_nan(b & 0xFFFF):
        return 0
    return 1 if _order(a, 16) > _order(b, 16) else 0


@primitive("I32Le", 1, [("a", I32), ("b", I32)], BIT, conformance="exact-model-tested")
def I32Le(a: int, b: int) -> int:
    """Signed 32-bit a <= b."""
    sa = a - (1 << 32) if a & 0x80000000 else a
    sb = b - (1 << 32) if b & 0x80000000 else b
    return 1 if sa <= sb else 0


@primitive("I32Eq", 1, [("a", I32), ("b", I32)], BIT, conformance="exact-model-tested")
def I32Eq(a: int, b: int) -> int:
    return 1 if (a & 0xFFFFFFFF) == (b & 0xFFFFFFFF) else 0


# ---- integer, bit and selection ----------------------------------------------------------------------------------------


@primitive("I32Add", 1, [("a", I32), ("b", I32)], I32, conformance="exact-model-tested")
def I32Add(a: int, b: int) -> int:
    """Wrapping 32-bit add."""
    return (a + b) & 0xFFFFFFFF


@primitive("BitAnd", 1, [("a", BIT), ("b", BIT)], BIT, conformance="protocol-defined")
def BitAnd(a: int, b: int) -> int:
    """Logical and on 1-bit encodings: `(a & b) & 1`.  Absorbing: `BitAnd(0, x) = 0` for every `x`."""
    return a & b & 1


@primitive("BitNot", 1, [("a", BIT)], BIT, conformance="protocol-defined")
def BitNot(a: int) -> int:
    """Logical not on 1-bit encodings: `(a ^ 1) & 1`."""
    return (a ^ 1) & 1


@primitive("BitOr", 1, [("a", BIT), ("b", BIT)], BIT, conformance="exact-model-tested")
def BitOr(a: int, b: int) -> int:
    return (a | b) & 1


@primitive("SelectF32", 1, [("c", BIT), ("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def SelectF32(c: int, a: int, b: int) -> int:
    """c ? a : b."""
    return a if c & 1 else b


@primitive("SelectBf16", 1, [("c", BIT), ("a", BF16), ("b", BF16)], BF16, conformance="exact-model-tested")
def SelectBf16(c: int, a: int, b: int) -> int:
    return a if c & 1 else b


@primitive("SelectI32", 1, [("c", BIT), ("a", I32), ("b", I32)], I32, conformance="exact-model-tested")
def SelectI32(c: int, a: int, b: int) -> int:
    return a if c & 1 else b

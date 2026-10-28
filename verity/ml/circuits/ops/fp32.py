"""Registered FP32 and BF16 arithmetic primitives: FP32 add/mul/fma/div rounded to nearest even (no ftz), the FTZ family
of the fast-math FlashAttention-2 build, `max`, the `-inf` guard, and the BF16 element-wise operations the eager kernels
compute in FP32 and round once to BF16.

Encodings: f32 = `Value<32>`, bf16 = `Value<16>`.  Every evaluator is total on encodings: NaN, infinities, signed zeros and
subnormals map to the outputs each docstring states.  Add, sub, mul, div and max run on numpy `float32` scalars; FMA is exact
rational arithmetic with one rounding.

NaN words.  The `_v1` arithmetic gives the host's NaN, not the hardware's: add, sub, mul and div return numpy float32's word
on x86-64 (an operand's NaN quieted, `0xFFC00000` for an invalid operation) and the FMAs `F32_NAN`.  Each `_v2` is its `_v1`
with every NaN result the GPU's `F32_NAN_PTX`, and equal to it on every other word.  A registered id denotes one function, so
the GPU's NaN came as new ids (as `verity.ml.circuits.ops.prims.AmpereBF16TcDot16_v2` did); Programs move to `_v2` at a re-baseline.

Provenance.  The `_v1` primitives are the definitions the vLLM integration registered in `verity.ml.definitions.registry.prims`
(same ids, signatures, conformance statuses and evaluators), which re-exports them from here.  Its stored Programs cite these
ids.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from verity.ml.circuits.defs import primitive
from verity.ml.circuits.types import Value

from .scalar import F32_NAN_PTX

BF16, F32 = Value(16), Value(32)

__all__ = ["f32_decode", "f32_round", "BF16", "BF16_NAN_F2FP", "BF16_NAN_TORCH", "F32", "F32_NAN", "F32_NAN_PTX", "Bf16Add", "Bf16AddF2fp",
           "Bf16MulBf16", "Bf16MulF32", "F32Add", "F32AddFtz", "F32Div", "F32Fma", "F32FmaFtz", "F32FmaSubFtz", "F32Max", "F32Mul",
           "F32MulFtz", "F32SubFtz", "F32AddV2", "F32AddFtzV2", "F32DivV2", "F32FmaV2", "F32FmaFtzV2", "F32FmaSubFtzV2", "F32MulV2",
           "F32MulFtzV2", "F32SubFtzV2", "GuardNegInfZero", "bf16_to_f32_bits", "bits_of", "f32_add_bits", "f32_div_bits",
           "f32_fma_bits", "f32_gpu_nan", "f32_is_nan", "f32_max_bits", "f32_mul_bits", "f32_of", "f32_sub_bits",
           "f32_to_bf16_rn_bits", "ftz_bits"]

# The float32 evaluators must return the IEEE result on overflow, invalid and division by zero, never warn or raise.
np.seterr(all="ignore")

# ---------------------------------------------------------------------------------------------------------
# exact float helpers on encodings
# ---------------------------------------------------------------------------------------------------------

#: the `_v1` FMAs' NaN, and the canonical NaN of the stand-in and reference evaluators that cite it; `_v2` returns `F32_NAN_PTX`
F32_NAN = 0x7FC00000
BF16_NAN_TORCH = 0x7FC0  # aten / torch conversion canonical NaN
BF16_NAN_F2FP = 0x7FFF  # SASS F2FP.BF16.F32 canonical NaN


def f32_of(bits: int) -> np.float32:
    return np.array([bits & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]


def bits_of(x) -> int:
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def bf16_to_f32_bits(b: int) -> int:
    return (b & 0xFFFF) << 16


def f32_to_bf16_rn_bits(u: int, nan: int) -> int:
    u &= 0xFFFFFFFF
    if (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF):
        return nan
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16) & 0xFFFF


def ftz_bits(u: int) -> int:
    """Flush a subnormal f32 encoding to signed zero."""
    u &= 0xFFFFFFFF
    if (u & 0x7F800000) == 0 and (u & 0x7FFFFF):
        return u & 0x80000000
    return u


def _decode(u: int) -> tuple[int, Fraction | None]:
    """f32 encoding -> (class, exact value). class 0 finite, 1 +inf, -1 -inf, 2 NaN."""
    u &= 0xFFFFFFFF
    s = -1 if u >> 31 else 1
    e = (u >> 23) & 0xFF
    m = u & 0x7FFFFF
    if e == 0xFF:
        return (2, None) if m else (s, None)
    if e == 0:
        return 0, Fraction(s * m, 1 << 149)
    return 0, Fraction(s * (m | 0x800000), 1) * Fraction(1, 1 << (150 - e)) if e < 150 else Fraction(s * (m | 0x800000) << (e - 150))


def _round_f32(x: Fraction, neg_zero: bool = False) -> int:
    """Round an exact rational to the nearest f32 (ties to even); overflow -> inf; exact zero keeps sign flag."""
    if x == 0:
        return 0x80000000 if neg_zero else 0
    s = 0x80000000 if x < 0 else 0
    a = -x if x < 0 else x
    # find exponent: 2^e <= a < 2^(e+1)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(1 << e, 1) > a if e >= 0 else Fraction(1, 1 << -e) > a:
        e -= 1
    if e < -126:
        e = -126  # subnormal scale
    # significand with 23 fraction bits: a / 2^(e-23)
    scale = e - 23
    q = a * (1 << -scale) if scale < 0 else a / (1 << scale)
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n & 1):
        n += 1
    if n >= 1 << 24:
        n >>= 1
        e += 1
    if e > 127:
        return s | 0x7F800000
    if n < 1 << 23:  # subnormal (e == -126)
        return s | n
    return s | ((e + 127) << 23) | (n & 0x7FFFFF)


#: the public names of the two helpers above, for code outside core
f32_decode = _decode
f32_round = _round_f32

def f32_fma_bits(a: int, b: int, c: int) -> int:
    """IEEE fused multiply-add on encodings, RN-even, one rounding (exact rational arithmetic)."""
    ca, va = _decode(a)
    cb, vb = _decode(b)
    cc, vc = _decode(c)
    if 2 in (ca, cb, cc):
        return F32_NAN
    if ca == 0 and cb == 0:
        prod = va * vb
        prod_zero_neg = (prod == 0) and ((a >> 31) ^ (b >> 31)) == 1
        if cc == 0:
            s = prod + vc
            if s == 0:
                # IEEE RN: an exact zero sum is -0 only when both addends are (signed) zeros with negative
                # sign; exact cancellation of nonzero addends gives +0.
                if prod == 0 and vc == 0:
                    return 0x80000000 if (prod_zero_neg and (c >> 31) & 1) else 0
                return 0
            return _round_f32(s)
        return 0x7F800000 if cc == 1 else 0xFF800000
    # a product involving an infinity
    if (ca == 0 and va == 0) or (cb == 0 and vb == 0):
        return F32_NAN  # inf * 0
    sign = ((a >> 31) ^ (b >> 31)) & 1
    if cc == 0:
        return 0xFF800000 if sign else 0x7F800000
    if (cc == -1) != bool(sign):
        return F32_NAN  # inf - inf
    return 0xFF800000 if sign else 0x7F800000


def f32_add_bits(a: int, b: int) -> int:
    return bits_of(np.float32(f32_of(a)) + np.float32(f32_of(b)))


def f32_sub_bits(a: int, b: int) -> int:
    return bits_of(np.float32(f32_of(a)) - np.float32(f32_of(b)))


def f32_mul_bits(a: int, b: int) -> int:
    return bits_of(np.float32(f32_of(a)) * np.float32(f32_of(b)))


def f32_div_bits(a: int, b: int) -> int:
    return bits_of(np.float32(f32_of(a)) / np.float32(f32_of(b)))


def f32_max_bits(a: int, b: int) -> int:
    """`FMNMX` / `max(x, y)` as the FA2 kernel uses it: `x > y ? x : y` (NaN handling of `MaxOp`)."""
    fa, fb = f32_of(a), f32_of(b)
    return a if fa > fb else b


def f32_is_nan(u: int) -> bool:
    return (u & 0x7F800000) == 0x7F800000 and (u & 0x7FFFFF) != 0


def f32_gpu_nan(u: int) -> int:
    """``u``, or `F32_NAN_PTX` when ``u`` is a NaN: the GPU's word for every NaN result of the FP32 arithmetic."""
    return F32_NAN_PTX if f32_is_nan(u) else u


# ---------------------------------------------------------------------------------------------------------
# f32 arithmetic (IEEE RN-even, no ftz) and the FTZ family used by the fast-math FA2 kernel
# ---------------------------------------------------------------------------------------------------------


@primitive("F32Add", 1, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32Add(a: int, b: int) -> int:
    """FADD: RN(a + b)."""
    return f32_add_bits(a, b)


@primitive("F32Mul", 1, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32Mul(a: int, b: int) -> int:
    """FMUL: RN(a * b)."""
    return f32_mul_bits(a, b)


@primitive("F32Fma", 1, [("a", F32), ("b", F32), ("c", F32)], F32, conformance="exact-model-tested")
def F32Fma(a: int, b: int, c: int) -> int:
    """FFMA: RN(a * b + c), one rounding."""
    return f32_fma_bits(a, b, c)


@primitive("F32Div", 1, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32Div(a: int, b: int) -> int:
    """IEEE `div.rn.f32`: RN(a / b)."""
    return f32_div_bits(a, b)


@primitive("F32AddFtz", 1, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32AddFtz(a: int, b: int) -> int:
    """FADD.FTZ: ftz(RN(ftz(a) + ftz(b)))."""
    return ftz_bits(f32_add_bits(ftz_bits(a), ftz_bits(b)))


@primitive("F32SubFtz", 1, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32SubFtz(a: int, b: int) -> int:
    """FADD.FTZ with negated second operand: ftz(RN(ftz(a) - ftz(b)))."""
    return ftz_bits(f32_sub_bits(ftz_bits(a), ftz_bits(b)))


@primitive("F32MulFtz", 1, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32MulFtz(a: int, b: int) -> int:
    """FMUL.FTZ: ftz(RN(ftz(a) * ftz(b)))."""
    return ftz_bits(f32_mul_bits(ftz_bits(a), ftz_bits(b)))


@primitive("F32FmaFtz", 1, [("a", F32), ("b", F32), ("c", F32)], F32, conformance="exact-model-tested")
def F32FmaFtz(a: int, b: int, c: int) -> int:
    """FFMA.FTZ: ftz(RN(ftz(a) * ftz(b) + ftz(c)))."""
    return ftz_bits(f32_fma_bits(ftz_bits(a), ftz_bits(b), ftz_bits(c)))


@primitive("F32FmaSubFtz", 1, [("a", F32), ("b", F32), ("c", F32)], F32, conformance="exact-model-tested")
def F32FmaSubFtz(a: int, b: int, c: int) -> int:
    """FFMA.FTZ with negated addend: ftz(RN(ftz(a) * ftz(b) - ftz(c)))."""
    return ftz_bits(f32_fma_bits(ftz_bits(a), ftz_bits(b), ftz_bits(c) ^ 0x80000000))


# --- version 2: every NaN result the GPU's `F32_NAN_PTX` (measured on the L40S and H100, every build, ftz on and off);
# equal to `_v1` on every other word ------------------------------------------------------------------------------------------


@primitive("F32Add", 2, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32AddV2(a: int, b: int) -> int:
    """FADD: RN(a + b); every NaN result `F32_NAN_PTX`."""
    return f32_gpu_nan(F32Add.evaluate(a, b))


@primitive("F32Mul", 2, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32MulV2(a: int, b: int) -> int:
    """FMUL: RN(a * b); every NaN result `F32_NAN_PTX`."""
    return f32_gpu_nan(F32Mul.evaluate(a, b))


@primitive("F32Fma", 2, [("a", F32), ("b", F32), ("c", F32)], F32, conformance="exact-model-tested")
def F32FmaV2(a: int, b: int, c: int) -> int:
    """FFMA: RN(a * b + c), one rounding; every NaN result `F32_NAN_PTX`."""
    return f32_gpu_nan(F32Fma.evaluate(a, b, c))


@primitive("F32Div", 2, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32DivV2(a: int, b: int) -> int:
    """IEEE `div.rn.f32`: RN(a / b); every NaN result `F32_NAN_PTX`."""
    return f32_gpu_nan(F32Div.evaluate(a, b))


@primitive("F32AddFtz", 2, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32AddFtzV2(a: int, b: int) -> int:
    """FADD.FTZ: ftz(RN(ftz(a) + ftz(b))); every NaN result `F32_NAN_PTX`."""
    return f32_gpu_nan(F32AddFtz.evaluate(a, b))


@primitive("F32SubFtz", 2, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32SubFtzV2(a: int, b: int) -> int:
    """ftz(RN(ftz(a) - ftz(b))); every NaN result `F32_NAN_PTX`."""
    return f32_gpu_nan(F32SubFtz.evaluate(a, b))


@primitive("F32MulFtz", 2, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32MulFtzV2(a: int, b: int) -> int:
    """FMUL.FTZ: ftz(RN(ftz(a) * ftz(b))); every NaN result `F32_NAN_PTX`."""
    return f32_gpu_nan(F32MulFtz.evaluate(a, b))


@primitive("F32FmaFtz", 2, [("a", F32), ("b", F32), ("c", F32)], F32, conformance="exact-model-tested")
def F32FmaFtzV2(a: int, b: int, c: int) -> int:
    """FFMA.FTZ: ftz(RN(ftz(a) * ftz(b) + ftz(c))); every NaN result `F32_NAN_PTX`."""
    return f32_gpu_nan(F32FmaFtz.evaluate(a, b, c))


@primitive("F32FmaSubFtz", 2, [("a", F32), ("b", F32), ("c", F32)], F32, conformance="exact-model-tested")
def F32FmaSubFtzV2(a: int, b: int, c: int) -> int:
    """FFMA.FTZ with negated addend: ftz(RN(ftz(a) * ftz(b) - ftz(c))); every NaN result `F32_NAN_PTX`."""
    return f32_gpu_nan(F32FmaSubFtz.evaluate(a, b, c))


@primitive("F32Max", 1, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def F32Max(a: int, b: int) -> int:
    """`max(x, y)` = `x > y ? x : y` (FA2 `MaxOp`); ftz on both operands as the fast-math build does."""
    return f32_max_bits(ftz_bits(a), ftz_bits(b))


@primitive("GuardNegInfZero", 1, [("m", F32)], F32, conformance="exact-model-tested")
def GuardNegInfZero(m: int) -> int:
    """FA2 `Check_inf`: m == -inf ? 0 : m."""
    return 0 if (m & 0xFFFFFFFF) == 0xFF800000 else m


# ---------------------------------------------------------------------------------------------------------
# bf16 element-wise (aten eager kernels: compute in f32, round once to bf16)
# ---------------------------------------------------------------------------------------------------------


@primitive("Bf16Add", 1, [("a", BF16), ("b", BF16)], BF16, conformance="exact-model-tested")
def Bf16Add(a: int, b: int) -> int:
    """aten.add(bf16, bf16): bf16_rn(f32(a) + f32(b)); NaN -> 0x7FC0."""
    return f32_to_bf16_rn_bits(f32_add_bits(bf16_to_f32_bits(a), bf16_to_f32_bits(b)), BF16_NAN_TORCH)


@primitive("Bf16AddF2fp", 1, [("a", BF16), ("b", BF16)], BF16, conformance="exact-model-tested")
def Bf16AddF2fp(a: int, b: int) -> int:
    """CUDA `_f16Vec += _f16Vec` (fused_add_rms_norm residual add): bf16_rn(f32(a) + f32(b)); NaN -> 0x7FFF."""
    return f32_to_bf16_rn_bits(f32_add_bits(bf16_to_f32_bits(a), bf16_to_f32_bits(b)), BF16_NAN_F2FP)


@primitive("Bf16MulF32", 1, [("a", BF16), ("s", F32)], BF16, conformance="model-untested")
def Bf16MulF32(a: int, s: int) -> int:
    """`_f16Vec *= float` (vLLM type_convert): bf16_rn(f32(a) * s); NaN -> 0x7FFF (F2FP)."""
    return f32_to_bf16_rn_bits(f32_mul_bits(bf16_to_f32_bits(a), s), BF16_NAN_F2FP)


@primitive("Bf16MulBf16", 1, [("a", BF16), ("b", BF16)], BF16, conformance="model-untested")
def Bf16MulBf16(a: int, b: int) -> int:
    """`_f16Vec *= _f16Vec`: bf16_rn(f32(a) * f32(b)); NaN -> 0x7FFF."""
    return f32_to_bf16_rn_bits(f32_mul_bits(bf16_to_f32_bits(a), bf16_to_f32_bits(b)), BF16_NAN_F2FP)

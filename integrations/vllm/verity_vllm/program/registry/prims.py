"""One-output primitives for the B1 eager batch-invariant profile (registered `Name_vN`, total on encodings).

Encodings: bf16 = `Value<16>` (IEEE binary16-exponent bfloat16 bit pattern), f32 = `Value<32>`, i32 =
`Value<32>` (two's complement), bit = `Value<1>`. Every evaluator is total: NaN/inf/signed-zero/subnormal
encodings map to ordinary outputs following the registered semantics stated in each docstring.

Hardware-shaped primitives wrap the existing exact models, they do not re-derive them:
  * `TcDot16` — `veritor.core.silicon_total` (`mma.sync.m16n8k16` k-step, total on every encoding), evaluated through
    `registry.tc_total` (the pure-Python 3.11 port the pods run; `veritor` / `verity_vllm.program.kernels.mma` are the test-only references);
  * MUFU `Ex2`, `Rcp` (FA2, `--use_fast_math`, ftz) — `fa2_model.MufuTables` (sm_89 measured tables);
  * MUFU `Sqrt`, `Rcp` (Triton RMSNorm) and `Rsqrt` (CUDA RMSNorm, `rsqrt.approx.f32`) —
    `rms_triton_model.RmsTables`.
Conformance statuses are per definition and are reported by `conformance.py`; the tables are conformance
evidence measured on sm_89 — this pod is sm_80 (A100), so every table-backed primitive is at most
`model-untested` on this architecture until the correspondence run says otherwise.
"""

from __future__ import annotations

import math
import os
from fractions import Fraction
from functools import lru_cache
from importlib import resources

import numpy as np

from verity.ir.defs import PrimitiveDefinition, primitive
from verity.ir.types import Array, Value
from verity.ml.tc.silicon import AMPERE_BF16_M16N8K16 as _AMPERE_BF16_M16N8K16
from verity.ml.tc.total import tc_dot_total as _tc_dot_total

BF16, F32, I32, BIT = Value(16), Value(32), Value(32), Value(1)

np.seterr(all="ignore")

# ---------------------------------------------------------------------------------------------------------
# exact float helpers on encodings
# ---------------------------------------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------------------------------------


@primitive("Bf16ToF32", 1, [("a", BF16)], F32, conformance="exact-model-tested")
def Bf16ToF32(a: int) -> int:
    """Exact widening: bf16 encoding << 16."""
    return bf16_to_f32_bits(a)


@primitive("F32ToBf16Rn", 1, [("a", F32)], BF16,
           conformance="exact-model-tested; not bound by B0/B1 (superseded in production by F2fpBf16_v1, the hardware "
                       "NaN word; L8) -- the software canonicalisation of c10::BFloat16::round_to_nearest_even and of "
                       "veritor.tc-ampere-bf16@1 / veritor.pinned.*@1 f32_to_bf16; equal to F2fpBf16_v1 on every non-NaN "
                       "word (lean/VeritorSpec/TCBHw.lean castF32ToBf16Hw_eq_of_not_nan; exhaustive over 2^32 in Rust/Python)")
def F32ToBf16Rn(a: int) -> int:
    """aten / torch round-to-nearest-even f32 -> bf16; NaN -> 0x7FC0 (the software canonical word)."""
    return f32_to_bf16_rn_bits(a, BF16_NAN_TORCH)


@primitive("F2fpBf16", 1, [("a", F32)], BF16,
           conformance="exact-model-tested; the production f32->bf16 conversion of B0/B1 (GemmCoordinate, RMSNorm* stores): "
                       "NaN -> 0x7FFF is the word cvt.rn.bf16.f32 / F2FP.BF16.F32 produced on the A100 and the RTX 4090 for all "
                       "84 measured NaN inputs (quiet/signalling, both signs, single-bit and random payloads; L6 "
                       "docs/data/tc-total-2026-09-07/*/cvt_special.json, 4403 words per device, 0 model mismatches); the same "
                       "function as veritor.tc-ampere-bf16@2 f32_to_bf16 (L8)")
def F2fpBf16(a: int) -> int:
    """SASS `F2FP.BF16.F32.PACK_AB` (`cvt.rn.bf16.f32`): RNE f32 -> bf16; every NaN -> 0x7FFF (measured, L6)."""
    return f32_to_bf16_rn_bits(a, BF16_NAN_F2FP)


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


def _silu_f32_bits(g_f32_bits: int) -> int:
    """W4b's registered stand-in for the CUDA `silu` (expf + division): RN32 of the exact real value
    g / (1 + e^-g). Total: NaN -> NaN, +inf -> +inf, -inf -> -0 (exact limit)."""
    c, v = _decode(g_f32_bits)
    if c == 2:
        return F32_NAN
    if c == 1:
        return 0x7F800000
    if c == -1:
        return 0x80000000
    g = float(v)
    if g < -80.0:
        return bits_of(np.float32(g * math.exp(g)))  # underflows to -0 well before this
    return bits_of(np.float32(g / (1.0 + math.exp(-g))))


@primitive("SiluMulBf16", 1, [("g", BF16), ("u", BF16)], BF16, conformance="exact-model-tested")
def SiluMulBf16(g: int, u: int) -> int:
    """vLLM CUDA `act_and_mul_kernel<silu>` (eager): bf16(f32(bf16(silu_f32(g))) * u); W4b's model, exact on
    every captured element in numerics.md; `silu_f32` is the float64-exact stand-in for expf+div."""
    s = f32_to_bf16_rn_bits(_silu_f32_bits(bf16_to_f32_bits(g)), BF16_NAN_TORCH)
    return f32_to_bf16_rn_bits(f32_mul_bits(bf16_to_f32_bits(s), bf16_to_f32_bits(u)), BF16_NAN_TORCH)


@primitive("RopeOut", 1, [("x", BF16), ("y", BF16), ("c", BF16), ("s", BF16)], BF16, conformance="exact-model-tested")
def RopeOut(x: int, y: int, c: int, s: int) -> int:
    """vLLM CUDA `rotary_embedding_kernel` (NeoX): bf16_rn(RN32(x*c - y*s)) with the bf16 cos/sin cache
    words; products of two bf16 are exact in f32, so this equals fma(x, c, -(y*s)) (W11/numerics.md model)."""
    t = f32_mul_bits(bf16_to_f32_bits(y), bf16_to_f32_bits(s))  # exact
    o = f32_fma_bits(bf16_to_f32_bits(x), bf16_to_f32_bits(c), t ^ 0x80000000)
    return f32_to_bf16_rn_bits(o, BF16_NAN_TORCH)


@primitive("RopeOutAdd", 1, [("x", BF16), ("y", BF16), ("c", BF16), ("s", BF16)], BF16, conformance="exact-model-tested")
def RopeOutAdd(x: int, y: int, c: int, s: int) -> int:
    """Second half of the NeoX rotation: bf16_rn(RN32(y*c + x*s))."""
    t = f32_mul_bits(bf16_to_f32_bits(x), bf16_to_f32_bits(s))
    o = f32_fma_bits(bf16_to_f32_bits(y), bf16_to_f32_bits(c), t)
    return f32_to_bf16_rn_bits(o, BF16_NAN_TORCH)


# ---------------------------------------------------------------------------------------------------------
# tensor core k-step
# ---------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _mma():
    """The finite reference (`verity_vllm.program.kernels.mma` -> `verity.ml.tc.silicon`): used by the tests as one of the oracles
    `_tc_dot16_total` is checked against; NOT used at evaluation time."""
    from verity_vllm.program.kernels import mma  # noqa: WPS433
    return mma


def _tc_dot16_total(acc: int, a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """`AmpereBF16TcDot16`: acc + sum_{k<16} a[k]*b[k] as `mma.sync.m16n8k16` computes one accumulator coordinate --
    `tc_total.tc_dot_total` on `AMPERE_BF16_M16N8K16` (groups (8, 8), 25-bit truncating adder, exponent floor -132), the
    pure-Python (3.11) gate-for-gate port of `veritor.core.silicon_total.tc_dot_total(AMPERE_BF16_DOT16_TOTAL)` (TOTAL family
    version 3, measured on the A100 and the RTX 4090): on finite operands = `silicon.tc_dot(AMPERE_BF16_M16N8K16)` (the R15/R16
    relation, unchanged); on every other encoding the MEASURED total rules -- any NaN operand / acc, inf * 0, or infinities of
    both signs (an overflowed group counts as its infinity) -> the canonical 0x7FFFFFFF; a one-signed infinity -> that infinity.
    [R17 dense, int-3] Before this the prim evaluated through `verity_vllm.program.kernels.mma` (unimportable on the pods' 3.11) with a
    documented-not-measured extension path that gave 0x7FC00000 for NaN and ignored a group's finite overflow beside an infinite
    operand; both differences are on non-finite outcomes only, which the vectorised twin flags and the sampled replay never
    compares (tests/program/test_derived_rows.py states them)."""
    return _tc_dot_total(_AMPERE_BF16_M16N8K16, acc, a, b)


A16 = Array(16, BF16)


@primitive("AmpereBF16TcDot16", 1, [("acc", F32), ("a", A16), ("b", A16)], F32, conformance="exact-model-tested")
def AmpereBF16TcDot16(acc: int, *ab: int) -> int:
    """One accumulator-coordinate transition of `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
    (16 products into `acc`): `verity.ml.tc.total.tc_dot_total` on `AMPERE_BF16_M16N8K16` (see `_tc_dot16_total`)."""
    return _tc_dot16_total(acc, tuple(ab[:16]), tuple(ab[16:32]))


# ---------------------------------------------------------------------------------------------------------
# MUFU tables (measured on sm_89; conformance evidence, see module docstring)
# ---------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _fa2():
    from verity_vllm.program.kernels import fa2_model, fa2_relation
    return fa2_model, fa2_relation.tables()


@lru_cache(maxsize=None)
def _rms():
    from verity_vllm.program.kernels import rms_relation, rms_triton_model
    return rms_triton_model, rms_relation.tables("cuda")


def _scalar(fn, tables, x: int) -> int:
    y = fn(np.array([x & 0xFFFFFFFF], dtype=np.uint32).view(np.float32), tables)
    return int(np.asarray(y, dtype=np.float32).view(np.uint32)[0])


@primitive("MufuEx2Ftz", 1, [("x", F32)], F32, conformance="exact-model-tested")
def MufuEx2Ftz(x: int) -> int:
    """`ex2.approx.ftz.f32` (MUFU.EX2), measured table + reduction rule (attention-model.md §3.1)."""
    fa2, t = _fa2()
    return _scalar(fa2.mufu_ex2, t, x)


@primitive("MufuRcpFtz", 1, [("x", F32)], F32, conformance="exact-model-tested")
def MufuRcpFtz(x: int) -> int:
    """`rcp.approx.ftz.f32` (MUFU.RCP), measured 2^23 mantissa table (attention-model.md §3.1)."""
    fa2, t = _fa2()
    return _scalar(fa2.mufu_rcp, t, x)


@primitive("Fa2InvSum", 1, [("sum", F32)], F32, conformance="exact-model-tested")
def Fa2InvSum(s: int) -> int:
    """FA2 `normalize_softmax_lse`: (sum == 0 || sum != sum) ? 1 : MUFU.RCP(sum)."""
    if (s & 0x7FFFFFFF) == 0 or f32_is_nan(s):
        return 0x3F800000
    return MufuRcpFtz.evaluate(s)


@primitive("MufuSqrtFtz", 1, [("x", F32)], F32, conformance="exact-model-tested")
def MufuSqrtFtz(x: int) -> int:
    """`sqrt.approx.ftz.f32` (MUFU.SQRT) as the Triton RMSNorm kernel uses it (rmsnorm-model.md)."""
    rm, t = _rms()
    return _scalar(rm.mufu_sqrt, t, x)


@primitive("DivFullRcp", 1, [("b", F32)], F32, conformance="exact-model-tested")
def DivFullRcp(b: int) -> int:
    """`div.full.f32` reciprocal part as ptxas emits it: range-scale b (|b| > 2^126: *0.25; |b| < 2^-126:
    *2^24) then MUFU.RCP; `DivFullScaleA` produces the matching scaled numerator (rms_triton_model.cpp)."""
    rm, t = _rms()
    fb = f32_of(b)
    ab = abs(fb)
    if ab > np.float32(8.507059173023462e37):
        fb = np.float32(fb * np.float32(0.25))
    elif not (ab < np.float32(1.1754943508222875e-38)):
        pass
    else:
        fb = np.float32(fb * np.float32(16777216.0))
    y = rm.lib()
    xb = np.array([bits_of(fb)], dtype=np.uint32)
    out = np.empty_like(xb)
    # mufu_rcp (non-ftz variant used by div.full) is exposed through rsq/rcp tables via div_full_words
    a1 = np.array([np.float32(1.0)], dtype=np.float32)
    res = rm.div_full(a1, np.array([fb], dtype=np.float32), t)  # 1.0 * rcp(scaled b) == rcp(scaled b) exactly
    return int(np.asarray(res, dtype=np.float32).view(np.uint32)[0])


@primitive("DivFullScaleA", 1, [("a", F32), ("b", F32)], F32, conformance="exact-model-tested")
def DivFullScaleA(a: int, b: int) -> int:
    """`div.full.f32` numerator range scaling: a*0.25 when |b| > 2^126, a*2^24 when |b| < 2^-126, else a."""
    fb = f32_of(b)
    ab = abs(fb)
    fa = f32_of(a)
    if ab > np.float32(8.507059173023462e37):
        return bits_of(np.float32(fa * np.float32(0.25)))
    if ab < np.float32(1.1754943508222875e-38):
        return bits_of(np.float32(fa * np.float32(16777216.0)))
    return a & 0xFFFFFFFF


@primitive("RsqrtApprox", 1, [("x", F32)], F32, conformance="exact-model-tested")
def RsqrtApprox(x: int) -> int:
    """`rsqrt.approx.f32` (rsqrtf without fast-math): MUFU.RSQ with the subnormal range scaling
    (rms_triton_model.rsqrt_f32; CUDA fused_add_rms_norm / rms_norm kernels)."""
    rm, t = _rms()
    return _scalar(rm.rsqrt_f32, t, x)


# ---------------------------------------------------------------------------------------------------------
# MUFU.TANH (`tanh.approx.f32`, FA2 `apply_softcap` = `cutlass::fast_tanh`) — measured-shard pilot 2026-09-17 (lane gemma)
#
# The hardware tanh is NOT RN32(tanh) (98 % of words differ on [0.5, 0.75), up to 126 ulp there, 133 ulp in the ov-dense samples),
# is not monotone (1081 adjacent inversions per 2^22 words, drops of up to 38 ulp at mantissa-segment boundaries), and no closed form
# is known — so, like `MufuEx2Ftz` / `MufuRcpFtz`, it is a measured word table.  Unlike ex2/rcp it has no exponent reduction: the
# table spans every f32 word between the small-|x| identity region and the saturation point, hence SHARDS of contiguous word ranges
# (`verity_vllm/ops/measure_tanh.py`, format `verity/mufu-tanh-shard/xz-blocks/v1`: xz blocks of int8 first differences of
# an injective ordered image of the output words, ~0.005 B/word).  The evaluator answers
#   (a) NaN inputs and the closed regions in `MUFU_TANH_RULES` (each established by an EXHAUSTIVE scan of the whole region),
#   (b) every other word from the table: the in-tree `quarantine/dense/tables/mufu_tanh_sm89.{json,xzblocks}` (manifest with
#       sha256, hardware/software, rules, evidence job ids, a 512-word raw sample; table [0x31000000, 0x41000000) = [2^-29, 8), which
#       covers the table-only region [2^-8, 8) with margin; the manifest is pinned by `MUFU_TANH_TABLE_MANIFEST_SHA256`); negative
#       words via the table's odd symmetry, which is used only where the measurement shows 0 violations over the table's whole range.
# `MufuTanhUnmeasured` is raised only when the table does not cover the word or does not load: manifest missing or not the pinned
# one, table file sha256 != manifest.  Never an approximation.
# Hardware/software of the table: RTX 4090 (sm_89), driver 580.178.04, torch 2.13.0+cu129 / triton 3.7.1
# (`inline_asm_elementwise("tanh.approx.f32 $0, $1;")`); the sub-ranges re-measured on driver 580.126.20 (cp-cov-a) hashed identical.
# The table is conformance evidence for that stack only.
# ---------------------------------------------------------------------------------------------------------

MUFU_TANH_SHARD_FORMAT = "verity/mufu-tanh-shard/xz-blocks/v1"
MUFU_TANH_TABLE_FORMAT = "verity/mufu-tanh-table/v1"
MUFU_TANH_TABLE_DIR = str(resources.files(__package__) / "quarantine" / "dense" / "tables")
MUFU_TANH_TABLE_MANIFEST = os.path.join(MUFU_TANH_TABLE_DIR, "mufu_tanh_sm89.json")
MUFU_TANH_TABLE_MANIFEST_SHA256 = "674b3663b5737e9d6b87a1a6a249e5a2a17cea279f89bae5d42fe8d0b123bb60"  # another table is another MufuTanh version
_F32_SIGN = 0x80000000

# Closed regions of the domain, each established ONLY by an exhaustive measurement of the whole region; None = not established, the
# evaluator raises there.  Evidence (2026-09-17, RTX 4090 sm_89, driver 580.178.04, `measure_tanh.py`, summaries in
# out/gen/lanes/gemma/ of lane/w-gemma; every job also ran the sign-flipped words and found 0 odd-symmetry violations):
#   identity_below 0x3B800001: job 0917T215010-79fd04 scanned [0x00000000, 0x31000000): 822,083,584 / 822,083,584 words hw(x) == x
#                              (word sha 125078cf…); job 0917T214646-3b1c9a's shard [0x31000000, 0x41000000) has its first hw != x word
#                              at 0x3B800001 (= 2^-8 + ulp) -> hw(x) == x for every |x| <= 2^-8, both signs (±0 included).
#   saturate_from  0x41000000: job 0917T215017-e0d178 scanned [0x41000000, 0x80000000): 1,048,576,001 / 1,048,576,001 non-NaN words
#                              (8.0 .. max finite, +inf) -> 0x3F800000 exactly (word sha aa255aa2…); negatives -> 0xBF800000 by the
#                              0-violation symmetry check.  Below 8.0 no word saturates (last non-one word 0x40FFFFFF, shard 3b1c9a).
#   nan_out        0x7FFFFFFF: job e0d178: all 8,388,607 positive NaN payloads -> 0x7FFFFFFF; job 0917T215355-08991c scanned
#                              [0xFF800001, 0x100000000): all 8,388,607 negative payloads -> 0x7FFFFFFF (word sha 2e0c30da…).
# Between the two rules, [0x3B800001, 0x41000000) is TABLE ONLY (shard 3b1c9a = the in-tree `mufu_tanh_sm89.xzblocks`, 449,932 B,
# sha256 5e261b88…, pulled from cp-cov-d by job 0917T220405-876a84 and re-hashed on the laptop).
MUFU_TANH_RULES: dict = {
    "identity_below": 0x3B800001,   # word w: hw(x) == x (bitwise) for every word 0 <= |x| < w
    "saturate_from": 0x41000000,    # word w: hw(x) == ±1.0 for every word |x| >= w, inf included
    "nan_out": 0x7FFFFFFF,          # the one word every NaN input (any payload, either sign) maps to
}


class MufuTanhUnmeasured(LookupError):
    """`MufuTanh.evaluate` refuses inputs outside the measured shards / established regions (never approximates silently)."""


class _TanhShard:
    """One measured word range: `header` = block index (`verity/mufu-tanh-shard/xz-blocks/v1`), `sym` = the odd-symmetry record of
    the measurement over exactly this range (`{"checked", "violations"}`); the file sha256 is verified against the header at load
    (mismatch -> `MufuTanhUnmeasured`); blocks are decoded on demand."""

    def __init__(self, path: str, header: dict, sym: dict | None, sha_words: str | None = None, job: str | None = None) -> None:
        import hashlib as _hl
        self.path, self.header = path, header
        if header.get("format") != MUFU_TANH_SHARD_FORMAT:
            raise MufuTanhUnmeasured(f"MufuTanh table {path}: unknown format {header.get('format')!r}")
        self.lo, self.hi = int(header["lo"], 16), int(header["hi"], 16)
        self.block_words = int(header["block_words"])
        if not os.path.exists(path):
            raise MufuTanhUnmeasured(f"MufuTanh table file missing: {path}")
        h = _hl.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        if h.hexdigest() != header["sha256_file"]:
            raise MufuTanhUnmeasured(f"MufuTanh table {path}: file sha256 {h.hexdigest()} != recorded {header['sha256_file']} (corrupt)")
        sym = sym or {}
        self.sym_ok = sym.get("violations") == 0 and sym.get("checked") == self.hi - self.lo
        self.sha_words, self.job = sha_words, job
        self._cache: dict[int, np.ndarray] = {}

    def block(self, k: int) -> np.ndarray:
        import lzma as _lzma
        b = self._cache.get(k)
        if b is None:
            hb = self.header["blocks"][k]
            with open(self.path, "rb") as f:
                f.seek(hb["offset"])
                raw = f.read(hb["bytes"])
            d = np.frombuffer(_lzma.decompress(raw), dtype=np.dtype(hb["dtype"])).astype(np.int64)
            if d.shape[0] != hb["n"] - 1:
                raise ValueError(f"MufuTanh shard {self.path} block {k}: {d.shape[0]} diffs for {hb['n']} words")
            o = np.empty(hb["n"], dtype=np.int64)
            o[0] = hb["first_ordered"]
            np.cumsum(d, out=o[1:])
            o[1:] += hb["first_ordered"]
            b = np.where(o < 0, (-(o + 1)) | _F32_SIGN, o).astype(np.uint32)  # inverse of measure_tanh.ordered_inj
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[k] = b
        return b

    def lookup(self, w: int) -> int | None:
        if not self.lo <= w < self.hi:
            return None
        i = w - self.lo
        return int(self.block(i // self.block_words)[i % self.block_words])


def mufu_tanh_table_manifest(path: str = MUFU_TANH_TABLE_MANIFEST) -> dict:
    """The in-tree table's manifest (format `verity/mufu-tanh-table/v1`): sha256s, hardware/software, rules, evidence job ids, the
    512-word raw sample (`sample.pairs` = [[x_hex, hw_hex], ...]) and the xz-block index (`table`)."""
    import json as _json
    with open(path) as f:
        m = _json.load(f)
    if m.get("format") != MUFU_TANH_TABLE_FORMAT:
        raise MufuTanhUnmeasured(f"MufuTanh manifest {path}: unknown format {m.get('format')!r}")
    return m


@lru_cache(maxsize=None)
def _tanh_shards() -> tuple:
    """The loaded table: the in-tree manifest, which must be the pinned one (`MUFU_TANH_TABLE_MANIFEST_SHA256`).  Raises
    `MufuTanhUnmeasured` when it is missing / not the pinned one / corrupt or does not cover the table-only region
    [identity_below, saturate_from) (a configuration error, not a measurement gap)."""
    import hashlib as _hl
    if not os.path.exists(MUFU_TANH_TABLE_MANIFEST):
        raise MufuTanhUnmeasured(f"MufuTanh table manifest missing: {MUFU_TANH_TABLE_MANIFEST}")
    with open(MUFU_TANH_TABLE_MANIFEST, "rb") as f:
        digest = _hl.sha256(f.read()).hexdigest()
    if digest != MUFU_TANH_TABLE_MANIFEST_SHA256:
        raise MufuTanhUnmeasured(f"MufuTanh table manifest {MUFU_TANH_TABLE_MANIFEST}: sha256 {digest} != pinned {MUFU_TANH_TABLE_MANIFEST_SHA256}")
    man = mufu_tanh_table_manifest()
    t = man["table"]
    s = _TanhShard(os.path.join(MUFU_TANH_TABLE_DIR, t["file"]), t, t.get("odd_symmetry"), t.get("sha256_words_le_u32"),
                   (man.get("evidence") or {}).get("table_job", {}).get("id"))
    lo_rule, hi_rule = MUFU_TANH_RULES["identity_below"], MUFU_TANH_RULES["saturate_from"]
    if lo_rule is not None and hi_rule is not None and not (s.lo <= lo_rule and s.hi >= hi_rule and s.sym_ok):
        raise MufuTanhUnmeasured(f"MufuTanh table [0x{s.lo:08X}, 0x{s.hi:08X}) sym_ok={s.sym_ok} does not close the table-only "
                                 f"region [0x{lo_rule:08X}, 0x{hi_rule:08X}) for both signs")
    return (s,)


def mufu_tanh_coverage() -> dict:
    """What the evaluator can answer right now: table word ranges (with their odd-symmetry verdict) and the established rules."""
    try:
        shards = _tanh_shards()
    except MufuTanhUnmeasured as e:
        return {"shards": [], "rules": {k: (None if v is None else f"0x{v:08X}") for k, v in MUFU_TANH_RULES.items()}, "error": str(e)}
    return {"shards": [{"lo": f"0x{s.lo:08X}", "hi": f"0x{s.hi:08X}", "n_words": s.hi - s.lo, "odd_symmetry_verified": s.sym_ok,
                        "sha256_words": s.sha_words, "path": s.path} for s in shards],
            "rules": {k: (None if v is None else f"0x{v:08X}") for k, v in MUFU_TANH_RULES.items()}}


@primitive("MufuTanh", 1, [("x", F32)], F32,
           conformance="quarantine: measured exact table 2026-09-17 (RTX 4090 sm_89, driver 580.178.04, triton 3.7.1) covering all 2^32 "
                       "words — in-tree quarantine/dense/tables/mufu_tanh_sm89.xzblocks (sha256 5e261b88…) + three exhaustively-scanned "
                       "rules; on-worker conformance job 0917T215619-1af45f 400,077 words 0 mismatches; exercised end-to-end by the "
                       "dense/AttentionSoftcap@v1 difftest vs FA2 (job 0917T223335-9ad747, 25/25 bitwise, driver 595.91.07)")
def MufuTanh(x: int) -> int:
    """`tanh.approx.f32` (MUFU.TANH) as FA2 `apply_softcap` executes it on sm_89: NaN and the exhaustively scanned regions from
    `MUFU_TANH_RULES`, every other word from the measured table (see the section comment).  Raises `MufuTanhUnmeasured` only when
    the table does not cover the word or does not load (missing, not the pinned one, corrupt)."""
    u = x & 0xFFFFFFFF
    sign, a = u & _F32_SIGN, u & 0x7FFFFFFF
    if a > 0x7F800000:  # NaN input
        r = MUFU_TANH_RULES["nan_out"]
        if r is None:
            raise MufuTanhUnmeasured(f"MufuTanh(0x{u:08X}): NaN inputs not exhaustively measured (only sampled payloads)")
        return r
    w = MUFU_TANH_RULES["saturate_from"]
    if w is not None and a >= w:
        return 0x3F800000 | sign
    w = MUFU_TANH_RULES["identity_below"]
    if w is not None and a < w:
        return u
    shards = _tanh_shards()  # raises MufuTanhUnmeasured itself when the table does not load
    for s in shards:
        y = s.lookup(u)
        if y is not None:
            return y
    if sign:
        for s in shards:
            if s.sym_ok:
                y = s.lookup(a)
                if y is not None:
                    return y ^ _F32_SIGN
    cov = mufu_tanh_coverage()
    raise MufuTanhUnmeasured(f"MufuTanh(0x{u:08X}) = tanh.approx.f32({f32_of(u)!r}): word outside the loaded tables "
                             f"{[(s['lo'], s['hi']) for s in cov['shards']]} and the established regions {cov['rules']}")


# ---------------------------------------------------------------------------------------------------------
# integer / selection
# ---------------------------------------------------------------------------------------------------------


@primitive("I32Le", 1, [("a", I32), ("b", I32)], BIT, conformance="exact-model-tested")
def I32Le(a: int, b: int) -> int:
    """Signed 32-bit a <= b."""
    sa = a - (1 << 32) if a & 0x80000000 else a
    sb = b - (1 << 32) if b & 0x80000000 else b
    return 1 if sa <= sb else 0


@primitive("I32Add", 1, [("a", I32), ("b", I32)], I32, conformance="exact-model-tested")
def I32Add(a: int, b: int) -> int:
    """Wrapping 32-bit add."""
    return (a + b) & 0xFFFFFFFF


@primitive("I32Eq", 1, [("a", I32), ("b", I32)], BIT, conformance="exact-model-tested")
def I32Eq(a: int, b: int) -> int:
    return 1 if (a & 0xFFFFFFFF) == (b & 0xFFFFFFFF) else 0


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


@primitive("Bf16GtStrict", 1, [("a", BF16), ("b", BF16)], BIT, conformance="exact-model-tested")
def Bf16GtStrict(a: int, b: int) -> int:
    """a > b on bf16 values (NaN compares false), used by greedy argmax (first index wins ties)."""
    fa, fb = f32_of(bf16_to_f32_bits(a)), f32_of(bf16_to_f32_bits(b))
    return 1 if fa > fb else 0


# ---------------------------------------------------------------------------------------------------------
# gather (wide fan-in: the whole vocabulary column is an operand collection)
# ---------------------------------------------------------------------------------------------------------

_GATHERS: dict[int, PrimitiveDefinition] = {}


def gather_bf16(vocab: int) -> PrimitiveDefinition:
    """`GatherBf16x{V}`: (token: i32, column: Array<V, bf16>) -> bf16 = column[token]; out-of-range or negative
    token -> 0x7FFF (a total, documented extension; production would fault)."""
    p = _GATHERS.get(vocab)
    if p is None:
        def ev(tok: int, *col: int, _V=vocab) -> int:
            t = tok - (1 << 32) if tok & 0x80000000 else tok
            return col[t] if 0 <= t < _V else 0x7FFF

        def ev_lazy(get, _V=vocab) -> int:
            tok = get(0)
            t = tok - (1 << 32) if tok & 0x80000000 else tok
            return get(1 + t) if 0 <= t < _V else 0x7FFF

        p = PrimitiveDefinition(f"GatherBf16x{vocab}", 1, (("token", I32), ("column", Array(vocab, BF16))), BF16, ev,
                                doc=gather_bf16.__doc__ or "", conformance="exact-model-tested")
        p.evaluate_lazy = ev_lazy  # type: ignore[attr-defined]
        _GATHERS[vocab] = p
    return p


ALL_SCALAR = [Bf16ToF32, F32ToBf16Rn, F2fpBf16, F32Add, F32Mul, F32Fma, F32Div, F32AddFtz, F32SubFtz, F32MulFtz,
              F32FmaFtz, F32FmaSubFtz, F32Max, GuardNegInfZero, Bf16Add, Bf16AddF2fp, Bf16MulF32, Bf16MulBf16, SiluMulBf16,
              RopeOut, RopeOutAdd, AmpereBF16TcDot16, MufuEx2Ftz, MufuRcpFtz, Fa2InvSum, MufuSqrtFtz, DivFullRcp,
              DivFullScaleA, RsqrtApprox, I32Le, I32Add, I32Eq, SelectF32, SelectBf16, SelectI32, Bf16GtStrict]

"""Vectorized exact evaluators for **derived boundaries** — interior gate values of the registered B0/B1
serving semantics that the commit store does not materialise.

"Any gate output can be a proof-block boundary": proof-block placement is defined over the circuit (core rule 4), so a plan
may select values that live *inside* the kernel-aligned leaves the adapter commits.  Replay can only serve
such a plan if the interior value can be produced fast and bit-exactly from committed neighbours.  This
module provides that production path for four derived classes, as numpy twins of the registered gate
semantics (`verity.ml.definitions.registry.b1` / `prims`) in the `rmsnorm_fused` pattern:

  * ``rmsnorm_triton_rows``   — the batch-invariant Triton `_rms_norm_kernel` reduction tree; emits the
    per-row scalars ``(sum, mean_eps, rms, inv)`` — `inv` is the gate
    ``F32Mul(DivFullScaleA(1, rms), DivFullRcp(rms))`` of `RMSNormTriton_v1` (b1_authored.py line ~81).
  * ``rmsnorm_fused_v2_rows`` — the generic CUDA `fused_add_rms_norm_kernel` under batch-invariance
    (`RMSNormFusedCuda_v2`); emits ``(var, s)`` — `s` is the `RsqrtApprox` gate of the epilogue.
    (Thin wrapper over the committed twin in `rmsnorm_fused`.)
  * ``silu_mul_rows``         — `SiluMul_v1`; emits the per-element bf16 silu intermediate
    ``mid = bf16_rn(silu_f32(gate))`` (the value between the gate_up GEMM row and the act_mul row; interior
    to the `SiluMulBf16_v1` primitive, materialised through the registered `_silu_f32_bits` semantics as a
    total 2^16 table).
  * ``attention_head_states`` — the FA2 online-softmax chain of `AttentionHead_v1` (blocks last -> first);
    emits the `AttnBlock_v1` interface states ``(m, l[4], o[D])`` after every 64-key block, then the
    `Fa2InvSum` epilogue and the bf16 head row.

Bit-exactness contract: same reduction trees, same MUFU tables (`fa2_relation.tables()` /
`rms_relation.tables("cuda")` — the sm_89-measured tables the registered primitives evaluate through), same
FTZ behaviour, same tensor-core group-sum (Ampere `mma.sync.m16n8k16`: groups (8,8), 25-bit adder, exponent
floor -132 — `veritor.core.silicon.AMPERE_BF16_M16N8K16`, ported here to vectorised int64), true fused
multiply-adds (Boldo–Melquiond round-to-odd through float64, validated against `prims.f32_fma_bits`).
Non-finite bf16 operands in a tensor-core dot follow the *documented extension* of
`prims.AmpereBF16TcDot16` only in the scalar registry; here such rows are flagged ``nonfinite`` and are
never claimed exact (the `fa2_model` convention).

Validation: `tests/program/test_derived_rows.py` (gate-by-gate circuit evaluation on small shapes) and
`verity_vllm.check.poc_derived_b0_check` (every norm/attention/silu row of a committed B0 request).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P

# ---------------------------------------------------------------------------------------------------------
# bit helpers (numpy on uint32 / int64 encodings)
# ---------------------------------------------------------------------------------------------------------

U32 = np.uint32
I64 = np.int64


def as_u32(x: np.ndarray) -> np.ndarray:
    """f32 array -> u32 encodings (bitcast)."""
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def as_f32(u: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(u, dtype=np.uint32).view(np.float32)


def bf16_to_f32_words(w: np.ndarray) -> np.ndarray:
    """bf16 words -> f32 words (exact widening, `Bf16ToF32_v1`)."""
    return np.asarray(w, dtype=np.uint16).astype(np.uint32) << 16


def f2fp_bf16(u: np.ndarray) -> np.ndarray:
    """`F2fpBf16_v1` on f32 words: RNE, canonical NaN 0x7FFF."""
    u = np.asarray(u, dtype=np.uint32)
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r[nan] = 0x7FFF
    return r


def bf16_rn_torch(u: np.ndarray) -> np.ndarray:
    """aten round-to-nearest-even f32 words -> bf16 words, NaN -> 0x7FC0 (`F32ToBf16Rn_v1`)."""
    u = np.asarray(u, dtype=np.uint32)
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r[nan] = 0x7FC0
    return r


def ftz(u: np.ndarray) -> np.ndarray:
    """Flush subnormal f32 encodings to signed zero."""
    u = np.asarray(u, dtype=np.uint32)
    sub = ((u & 0x7F800000) == 0) & ((u & 0x007FFFFF) != 0)
    return np.where(sub, u & 0x80000000, u)


def _f32_binop(a: np.ndarray, b: np.ndarray, op) -> np.ndarray:
    return as_u32(op(as_f32(a), as_f32(b)))


def add_rn(a, b):
    return _f32_binop(a, b, np.add)


def mul_rn(a, b):
    return _f32_binop(a, b, np.multiply)


def div_rn(a, b):
    return _f32_binop(a, b, np.divide)


def add_ftz(a, b):
    return ftz(add_rn(ftz(a), ftz(b)))


def sub_ftz(a, b):
    return ftz(_f32_binop(ftz(a), ftz(b), np.subtract))


def mul_ftz(a, b):
    return ftz(mul_rn(ftz(a), ftz(b)))


def max_fmnmx(a, b):
    """`F32Max_v1`: ftz both operands, then `x > y ? x : y` (NaN in x compares false -> y; +0/-0 -> y)."""
    a, b = ftz(a), ftz(b)
    return np.where(as_f32(a) > as_f32(b), a, b)


def guard_neg_inf_zero(m):
    """`GuardNegInfZero_v1`: m == -inf ? 0 : m."""
    m = np.asarray(m, dtype=np.uint32)
    return np.where(m == np.uint32(0xFF800000), np.uint32(0), m)


# ---------------------------------------------------------------------------------------------------------
# true fused multiply-add, vectorised (Boldo–Melquiond round-to-odd through float64)
# ---------------------------------------------------------------------------------------------------------
#
# fma_rn(a, b, c) = RN32(a*b + c), one rounding, on f32 words.  The product of two binary32 values is exact
# in binary64 (24+24 <= 53); TwoSum gives s = RN64(p + c) and the exact error e = (p + c) - s; rounding the
# 64-bit sum to odd (force the last kept bit when inexact) then RN64->RN32 equals the single RN32 of the
# exact value (Boldo & Melquiond 2008, 53 >= 24 + 2).  Validated against `prims.f32_fma_bits` (exact
# rational arithmetic) in test_derived_rows.py.


def _round_to_odd64(s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """s (f64) already = RN64(x), e = x - s exactly (f64).  Returns the f64 round-to-odd of x."""
    si = s.view(np.int64)
    inexact = (e != 0) & np.isfinite(s)
    even = (si & 1) == 0
    move = inexact & even
    # step s one ulp toward x (the direction of e): on the sign-magnitude float grid, +inf-ward is
    # magnitude+1 for positives and magnitude-1 for negatives (s == +-0 with e != 0 cannot happen:
    # the rounded sum of a 48-bit f32 product and an f32 addend is never below the f64 subnormal range).
    toward_pos = e > 0
    neg_s = (si >> 63) != 0
    delta = np.where(toward_pos != neg_s, np.int64(1), np.int64(-1))  # in magnitude space
    sign = si & np.int64(-(1 << 63))
    mag = (si & np.int64((1 << 63) - 1)) + np.where(move, delta, np.int64(0))
    out = np.where(move, sign | mag, si)
    return out.view(np.float64)


def fma_rn(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """`F32Fma_v1` vectorised: RN32(a * b + c) on f32 word arrays."""
    a, b, c = np.broadcast_arrays(np.asarray(a, U32), np.asarray(b, U32), np.asarray(c, U32))
    af, bf, cf = (as_f32(x).astype(np.float64) for x in (a, b, c))
    p = af * bf  # exact
    # TwoSum (Knuth): s + e == p + c exactly, for any magnitudes
    s = p + cf
    bp = s - p
    e = (p - (s - bp)) + (cf - bp)
    v = _round_to_odd64(s, e)
    out = as_u32(v.astype(np.float32))
    # IEEE exact-zero sign: RN64/RN32 of f64 keeps the correct signed zero except the exact-cancellation
    # case, where f64 addition already yields +0 (matches f32_fma_bits: exact cancellation -> +0), and the
    # all-zero-addends case, where the f64 sum sign is the IEEE one.  NaN canonicalisation:
    nan = ((out & 0x7F800000) == 0x7F800000) & ((out & 0x007FFFFF) != 0)
    out = np.where(nan, np.uint32(P.F32_NAN), out)
    return out


def fma_ftz(a, b, c):
    """`F32FmaFtz_v1`: ftz(RN(ftz(a)*ftz(b) + ftz(c)))."""
    return ftz(fma_rn(ftz(a), ftz(b), ftz(c)))


def fma_sub_ftz(a, b, c):
    """`F32FmaSubFtz_v1`: ftz(RN(ftz(a)*ftz(b) - ftz(c)))."""
    return ftz(fma_rn(ftz(a), ftz(b), np.asarray(ftz(c), U32) ^ np.uint32(0x80000000)))


# ---------------------------------------------------------------------------------------------------------
# MUFU tables, vectorised ports of the measured bit rules (fa2_model.cpp / rms_triton_model.cpp)
# ---------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _fa2_tables():
    from verity.ml.definitions.numerics import fa2_relation
    return fa2_relation.tables()


@lru_cache(maxsize=None)
def _rms_tables():
    from verity.ml.definitions.numerics import rms_relation
    return rms_relation.tables("cuda")  # sqrt + rcp + rsq banks


def mufu_ex2_ftz(x: np.ndarray, table: np.ndarray | None = None) -> np.ndarray:
    """`MufuEx2Ftz_v1` (`ex2.approx.ftz.f32`) on f32 words: table T[j] with the fixed-point reduction rule."""
    T = _fa2_tables().ex2 if table is None else table
    x = np.asarray(x, dtype=np.uint32)
    ax = x & 0x7FFFFFFF
    e = (ax >> 23).astype(np.int64)
    neg = (x >> 31).astype(bool)
    mant = ((ax & 0x7FFFFF) | 0x800000).astype(np.int64)
    unb = e - 127
    sh_l = np.clip(unb, 0, 63)
    # K = floor(|x| 2^23): mant < 2^24, so every right shift of 24 or more is 0 and |x| < 2^-23 reads T[0] = 1.0
    # at n = 0 for either sign.  The exhaustive sm_89 verification (attention-model.md §3.1, all 2^32 inputs,
    # 0 mismatches) ran the model on the device, where the shift clamps; a host C shift by >= 64 wraps mod 64
    sh_r = np.clip(-unb, 0, 24)
    K = np.where(unb >= 0, mant << sh_l, mant >> sh_r)
    frac_zero = (K & 0x7FFFFF) == 0
    fixed = np.where(~neg, K, np.where(frac_zero, -K, -K - 1))
    n = fixed >> 23  # arithmetic
    j = (fixed & 0x7FFFFF).astype(np.int64)
    t = T[j].astype(np.int64)
    re = ((t >> 23) & 0xFF) + n
    out = np.where(re >= 255, np.uint32(0x7F800000),
                   np.where(re <= 0, np.uint32(0),
                            ((re.astype(np.int64) << 23) | (t & 0x7FFFFF)).astype(np.uint32)))
    # specials, in the C order
    out = np.where(e == 0, np.uint32(0x3F800000), out)  # ftz: +-0 / subnormal -> 1.0
    out = np.where((unb >= 30) & (e != 255), np.where(neg, np.uint32(0), np.uint32(0x7F800000)), out)
    inf = (e == 255) & ((ax & 0x7FFFFF) == 0)
    nan = (e == 255) & ((ax & 0x7FFFFF) != 0)
    out = np.where(inf, np.where(neg, np.uint32(0), np.uint32(0x7F800000)), out)
    out = np.where(nan, np.uint32(0x7FFFFFFF), out)
    return out


def _mufu_rcp_core(x: np.ndarray, R: np.ndarray) -> np.ndarray:
    """`rcp.approx.ftz.f32` bit rule (identical in fa2_model.cpp and rms_triton_model.cpp)."""
    x = np.asarray(x, dtype=np.uint32)
    s = x & 0x80000000
    ax = x & 0x7FFFFFFF
    e = (ax >> 23).astype(np.int64)
    r = R[(ax & 0x7FFFFF).astype(np.int64)].astype(np.int64)
    re = ((r >> 23) & 0xFF) - (e - 127)
    out = np.where(re <= 0, s,
                   np.where(re >= 0xFF, s | 0x7F800000,
                            (s.astype(np.int64) | (re << 23) | (r & 0x7FFFFF)).astype(np.uint32)))
    out = np.where(e == 0, s | np.uint32(0x7F800000), out)
    inf = (e == 255) & ((ax & 0x7FFFFF) == 0)
    nan = (e == 255) & ((ax & 0x7FFFFF) != 0)
    out = np.where(inf, s, out)
    out = np.where(nan, np.uint32(0x7FFFFFFF), out)
    return out.astype(np.uint32)


def mufu_rcp_ftz(x: np.ndarray) -> np.ndarray:
    """`MufuRcpFtz_v1` (FA2 epilogue rcp, fa2 tables)."""
    return _mufu_rcp_core(x, _fa2_tables().rcp)


def fa2_inv_sum(s: np.ndarray) -> np.ndarray:
    """`Fa2InvSum_v1`: (sum == +-0 || NaN) ? 1.0 : MUFU.RCP(sum)."""
    s = np.asarray(s, dtype=np.uint32)
    degen = ((s & 0x7FFFFFFF) == 0) | (((s & 0x7F800000) == 0x7F800000) & ((s & 0x7FFFFF) != 0))
    return np.where(degen, np.uint32(0x3F800000), mufu_rcp_ftz(s))


def mufu_sqrt_ftz(x: np.ndarray) -> np.ndarray:
    """`MufuSqrtFtz_v1` (`sqrt.approx.ftz.f32`, Triton RMSNorm): two-bank table by exponent parity."""
    T = _rms_tables().sqrt
    x = np.asarray(x, dtype=np.uint32)
    s = x & 0x80000000
    ax = x & 0x7FFFFFFF
    e = (ax >> 23).astype(np.int64)
    k = e - 127
    par = k & 1
    t = T[((par << 23) | (ax & 0x7FFFFF)).astype(np.int64)].astype(np.int64)
    re = ((t >> 23) & 0xFF) + (k - par) // 2
    out = ((re << 23) | (t & 0x7FFFFF)).astype(np.uint32)
    out = np.where(s != 0, np.uint32(0x7FFFFFFF), out)  # negative -> NaN
    out = np.where(e == 0, s, out)  # ftz: +-0 / subnormal -> +-0
    inf = (e == 255) & ((ax & 0x7FFFFF) == 0)
    nan = (e == 255) & ((ax & 0x7FFFFF) != 0)
    out = np.where(inf, np.where(s != 0, np.uint32(0x7FFFFFFF), np.uint32(0x7F800000)), out)
    out = np.where(nan, np.uint32(0x7FFFFFFF), out)
    return out


def _mufu_rsq(x: np.ndarray) -> np.ndarray:
    """Bare `MUFU.RSQ` (`rsqrt.approx.ftz.f32`)."""
    T = _rms_tables().rsq
    x = np.asarray(x, dtype=np.uint32)
    s = x & 0x80000000
    ax = x & 0x7FFFFFFF
    e = (ax >> 23).astype(np.int64)
    k = e - 127
    par = k & 1
    t = T[((par << 23) | (ax & 0x7FFFFF)).astype(np.int64)].astype(np.int64)
    re = ((t >> 23) & 0xFF) - (k - par) // 2
    out = ((re << 23) | (t & 0x7FFFFF)).astype(np.uint32)
    out = np.where(s != 0, np.uint32(0x7FFFFFFF), out)
    out = np.where(e == 0, s | np.uint32(0x7F800000), out)  # +-0/subnormal -> +-inf
    inf = (e == 255) & ((ax & 0x7FFFFF) == 0)
    nan = (e == 255) & ((ax & 0x7FFFFF) != 0)
    out = np.where(inf, np.where(s != 0, np.uint32(0x7FFFFFFF), np.uint32(0)), out)
    out = np.where(nan, np.uint32(0x7FFFFFFF), out)
    return out


def rsqrt_approx(x: np.ndarray) -> np.ndarray:
    """`RsqrtApprox_v1` (`rsqrt.approx.f32`, CUDA fused/rms norm): MUFU.RSQ with subnormal range scaling."""
    x = np.asarray(x, dtype=np.uint32)
    xf = as_f32(x)
    scale = ~(np.abs(xf) >= np.float32(1.1754943508222875e-38))  # |x| < 2^-126 (NaN -> False)
    xin = np.where(scale, as_u32((xf * np.float32(16777216.0)).astype(np.float32)), x)
    r = _mufu_rsq(xin)
    rf = as_f32(r)
    return np.where(scale, as_u32((rf * np.float32(4096.0)).astype(np.float32)), r)


def div_full_rcp(bw: np.ndarray) -> np.ndarray:
    """`DivFullRcp_v1`: `div.full.f32` reciprocal part — range-scale b, then MUFU.RCP (rms rcp table)."""
    bw = np.asarray(bw, dtype=np.uint32)
    bf = as_f32(bw)
    ab = np.abs(bf)
    big = ab > np.float32(8.507059173023462e37)
    small = ab < np.float32(1.1754943508222875e-38)  # NaN -> False on both
    scaled = np.where(big, (bf * np.float32(0.25)).astype(np.float32),
                      np.where(small, (bf * np.float32(16777216.0)).astype(np.float32), bf))
    return _mufu_rcp_core(as_u32(scaled), _rms_tables().rcp)


def div_full_scale_a(aw: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """`DivFullScaleA_v1`: numerator range scaling of `div.full.f32`."""
    aw = np.asarray(aw, dtype=np.uint32)
    bf = as_f32(np.asarray(bw, dtype=np.uint32))
    af = as_f32(aw)
    ab = np.abs(bf)
    big = ab > np.float32(8.507059173023462e37)
    small = ab < np.float32(1.1754943508222875e-38)
    return np.where(big, as_u32((af * np.float32(0.25)).astype(np.float32)),
                    np.where(small, as_u32((af * np.float32(16777216.0)).astype(np.float32)), aw))


# ---------------------------------------------------------------------------------------------------------
# Ampere tensor-core k16 step, vectorised (veritor.core.silicon AMPERE_BF16_M16N8K16: groups (8,8),
# 25-bit adder, exponent floor -132)
# ---------------------------------------------------------------------------------------------------------

_TC_WIDTH = 25
_TC_FLOOR = -132
_RESCALE = _TC_WIDTH - 24  # 1


def _bitlen(m: np.ndarray) -> np.ndarray:
    """bit_length of non-negative int64 arrays."""
    m = np.asarray(m, dtype=np.int64)
    bl = np.zeros(m.shape, dtype=np.int64)
    x = m.copy()
    for sh in (32, 16, 8, 4, 2, 1):
        big = x >= (np.int64(1) << sh)
        bl[big] += sh
        x[big] >>= sh
    bl[m > 0] += 1
    return bl


def _tc_dot16_pipeline(acc: np.ndarray, a: np.ndarray, b: np.ndarray, *, groups: tuple[int, ...], width: int,
                       floor: int) -> tuple[np.ndarray, np.ndarray]:
    """One bf16 k16 tensor-core step under a `veritor.core.silicon.Pipeline`'s parameters, vectorised: acc (...,) f32 words,
    a/b (..., 16) bf16 words -> (f32 words, nonfinite mask).  `groups` lists, in k order, how many products each grouped sum
    takes (the incoming accumulator joins the first group and each group's result joins the next); `width` is the adder's
    significand width and `floor` its exponent floor -- exactly `silicon.group_sum` / `silicon.tc_dot`, gate for gate.

    Lanes with a non-finite bf16 operand or a non-finite incoming accumulator are flagged (the scalar registry's documented
    total extension is not replicated here; flagged lanes report the propagated word but must be treated as `unresolved`,
    the fa2_model convention).  Finite-input overflow saturates to a sticky infinity exactly as the silicon model does."""
    if sum(groups) != 16:
        raise ValueError(f"a k16 step takes 16 products, groups {groups} sum to {sum(groups)}")
    rescale = width - 24                                      # the 24-bit FP32 grid -> the adder's width (>= 0 for the bf16 pipelines)
    if rescale < 0:
        raise ValueError(f"bf16 pipeline adder width {width} < 24 (truncating rescale is the FP8 twin's case)")
    acc = np.asarray(acc, dtype=np.uint32)
    a = np.asarray(a, dtype=np.uint16)
    b = np.asarray(b, dtype=np.uint16)
    ea = ((a.astype(np.int64) >> 7) & 0xFF)
    eb = ((b.astype(np.int64) >> 7) & 0xFF)
    nonfin = (ea == 0xFF).any(axis=-1) | (eb == 0xFF).any(axis=-1) | ((acc & 0x7F800000) == 0x7F800000)
    sa = (a.astype(np.int64) & 0x7F) | ((ea != 0).astype(np.int64) << 7)
    sb = (b.astype(np.int64) & 0x7F) | ((eb != 0).astype(np.int64) << 7)
    p_mag = (sa * sb) << 9  # 24-bit-grid magnitudes (bit 23 at p_exp)
    p_exp = np.maximum(ea, 1) + np.maximum(eb, 1) - 254
    p_neg = (((a ^ b).astype(np.int64) >> 15) & 1).astype(bool)
    # incoming accumulator as a Term
    acc_i = acc.astype(np.int64)
    a_ef = (acc_i >> 23) & 0xFF
    a_mant = acc_i & 0x7FFFFF
    acc_neg = (acc_i >> 31) == 1
    acc_mag = np.where(a_ef == 0, a_mant, a_mant | 0x800000)
    acc_exp = np.where(a_ef == 0, np.int64(-126), a_ef - 127)
    sat = np.zeros(acc.shape, dtype=bool)
    sat_word = np.zeros(acc.shape, dtype=np.uint32)
    g0 = 0
    for size in groups:
        sl = slice(g0, g0 + size)
        g0 += size
        mags = np.concatenate([acc_mag[..., None], p_mag[..., sl]], axis=-1)
        exps = np.concatenate([acc_exp[..., None], p_exp[..., sl]], axis=-1)
        negs = np.concatenate([acc_neg[..., None], p_neg[..., sl]], axis=-1)
        live = mags != 0
        max_exp = np.max(np.where(live, exps, np.int64(-(1 << 40))), axis=-1)
        max_exp = np.maximum(max_exp, np.int64(floor))
        scaled = mags << rescale
        shift = np.clip(max_exp[..., None] - exps, 0, 63)
        aligned = scaled >> shift
        total = np.sum(np.where(negs, -aligned, aligned) * live, axis=-1)
        neg = total < 0
        mag = np.abs(total)
        bl = _bitlen(mag)
        exp = max_exp + bl - width
        down = np.clip(bl - width, 0, 63)
        up = np.clip(width - bl, 0, 63)
        mag = np.where(bl > width, mag >> down, mag << up)
        sub = np.clip(np.int64(-126) - exp, 0, 63)
        mag = mag >> sub
        exp = np.maximum(exp, np.int64(-126))
        mag = mag >> rescale
        zero = mag == 0
        # saturation between groups: exp > 127 with a live magnitude -> sticky inf of that sign
        over = (~zero) & (exp > 127)
        sat_word = np.where(over & ~sat, (neg.astype(np.uint32) << 31) | np.uint32(0x7F800000), sat_word)
        sat = sat | over
        acc_mag = np.where(zero, np.int64(0), mag)
        acc_exp = exp
        acc_neg = neg & ~zero
    # pack
    exp_field = acc_exp + 127
    exp_field = exp_field - ((acc_mag & (1 << 23)) == 0)  # subnormal at exp -126 -> field 0
    word = ((acc_neg.astype(np.int64) << 31) | (exp_field << 23) | (acc_mag & 0x7FFFFF)).astype(np.uint32)
    word = np.where(acc_mag == 0, np.uint32(0), word)
    word = np.where(sat, sat_word, word)
    return word, nonfin


def tc_dot16(acc: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`AmpereBF16TcDot16_v2` vectorised: acc (...,) f32 words, a/b (..., 16) bf16 words -> (f32 words, nonfinite mask).
    `mma.sync.m16n8k16` bf16 -> f32 on sm_80 / sm_89: two groups of eight products, 25-bit adder, exponent floor -132
    (`veritor.core.silicon.AMPERE_BF16_M16N8K16`).  See `_tc_dot16_pipeline` for the flagging / saturation conventions."""
    return _tc_dot16_pipeline(acc, a, b, groups=(8, 8), width=_TC_WIDTH, floor=_TC_FLOOR)


# ---------------------------------------------------------------------------------------------------------
# Hopper bf16 tensor-core k16 step, vectorised: ONE accumulator coordinate of
# `wgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16` (SASS HGMMA.64xNx16.F32.BF16) as `veritor.core.silicon_total.
# HOPPER_BF16_WGMMA_K16` defines it -- one group of all sixteen products plus the accumulator, 26-bit adder, exponent floor
# -133 (measured on the H100: 307,281,920 output words, 0 mismatches; `scale-d = 0` measured equal to acc = +0).  The registered scalar primitive is `verity.ml.circuits.ops.prims.HopperBF16WgmmaDot16`; this twin
# is checked word for word against it in tests/program/test_derived_rows.py.
#
# The k-CHAIN of the Hopper GEMM of record (`Gemm_v2{K,N,DOT=HopperBF16WgmmaDot16_v1}`, gemm_targets.REGISTERED['hopper']) is
# NOT the Triton kernel's: vLLM d9105ea80 `enable_batch_invariant_mode` installs the Triton `matmul_kernel_persistent` overrides
# only on the cc 8.x family; on cc 9.0 the bf16 Linear stays cuBLASLt with split-K disabled by configuration
# (CUBLAS_WORKSPACE_CONFIG=:16:8, CUBLASLT_WORKSPACE_SIZE=1, allow_bf16_reduced_precision_reduction=(False, False),
# preferred_blas_library=cublaslt).  Under EXACTLY that configuration the H100 (sm 132, torch 2.13.0+cu129) output equals the
# ascending k16 wgmma chain from a zero accumulator + RN to bf16, word for word, on every Qwen2.5-1.5B linear shape
# (K,N in {(1536,2048),(1536,1536),(1536,17920),(8960,1536),(1536,151936)}) x step lengths M in {1..8,12,16,...,8192}:
# 0 mismatching words of 27,240,000;
# the Ampere chain differs on every case.  Under torch DEFAULTS (split-K allowed) the same GPU splits K=8960 three ways at
# M<=64 (tile-64 split-K, f32 (p0+p1)+p2) -- so the relation below is the relation of a VLLM_BATCH_INVARIANT=1 record only,
# which is what every row of record declares (sweep/record.py engine_args), the same premise Gemm_v1 already rests on.
# `dot_bf16_hopper` is therefore `dot_bf16` with the step swapped.
# ---------------------------------------------------------------------------------------------------------

_HOPPER_TC_GROUPS = (16,)
_HOPPER_TC_WIDTH = 26
_HOPPER_TC_FLOOR = -133


def tc_dot16_hopper(acc: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`HopperBF16WgmmaDot16_v1` vectorised: acc (...,) f32 words, a/b (..., 16) bf16 words -> (f32 words, nonfinite mask).
    One group of sixteen products + the accumulator, 26-bit adder, exponent floor -133 (`silicon_total.HOPPER_BF16_WGMMA_K16`
    == `silicon.HOPPER_BF16_M16N8K16`).  Flagging / saturation as `tc_dot16`."""
    return _tc_dot16_pipeline(acc, a, b, groups=_HOPPER_TC_GROUPS, width=_HOPPER_TC_WIDTH, floor=_HOPPER_TC_FLOOR)


# ---------------------------------------------------------------------------------------------------------
# RoPE_v1 vectorised -- `b1.RoPE{NHEADS, D}` = NHEADS x `RoPEHead{D}` over ONE bf16 cos/sin
# cache row `cs = [cos[0:R] | sin[0:R]]` (R = D/2, the `weights` field `...rotary_emb.cos_sin_cache[pos]`), the vLLM CUDA
# `rotary_embedding_kernel` (NeoX):  out[i] = bf16_rn(RN32(x[i]*c[i] - x[i+R]*s[i])),  out[i+R] = bf16_rn(RN32(x[i+R]*c[i] + x[i]*s[i])).
# A product of two bf16 words is exact in f32, so RN32(x*c - y*s) == fma(x, c, -(y*s)) exactly as `prims.RopeOut` / `RopeOutAdd`
# (exact-model-tested) compute it; bf16_rn is the aten conversion (NaN 0x7FC0).  Checked word for word against the scalar prims in
# tests/program/test_derived_rows.py.
# ---------------------------------------------------------------------------------------------------------


def rope_rows(x: np.ndarray, cs: np.ndarray, store=None) -> np.ndarray:
    """`RoPE_v1{NHEADS, D}` vectorised: x (..., NHEADS, D) bf16 words, cs (..., D) bf16 words (one cache row per leading index,
    broadcast over heads) -> (..., NHEADS, D) bf16 words.  ``store`` rounds the f32 results (default `bf16_rn_torch`; `f2fp_bf16`
    is `inductor.InductorRoPE_v1`'s)."""
    return rope_f32_rows(bf16_to_f32_words(np.asarray(x, dtype=np.uint16)), cs, store=store)


def rope_f32_rows(xf: np.ndarray, cs: np.ndarray, store=None) -> np.ndarray:
    """`rope_rows` of f32 words xf (..., NHEADS, D), a rotation whose input no bf16 store rounded (`inductor.InductorQKNormRoPE_v1`)."""
    store = bf16_rn_torch if store is None else store
    xf = np.asarray(xf, dtype=np.uint32)
    cs = np.asarray(cs, dtype=np.uint16)
    D = xf.shape[-1]
    if D % 2 or cs.shape[-1] != D:
        raise ValueError(f"RoPE_v1: head dim {D} / cache row width {cs.shape[-1]} (need even D, cache row == D)")
    R = D // 2
    c = bf16_to_f32_words(cs[..., None, :R])                                    # (..., 1, R) broadcast over heads
    s = bf16_to_f32_words(cs[..., None, R:])
    x1, x2 = xf[..., :R], xf[..., R:]
    lo = fma_rn(x1, c, mul_rn(x2, s) ^ np.uint32(0x80000000))                    # x*c - y*s  (y*s exact for bf16 x, so one rounding)
    hi = fma_rn(x2, c, mul_rn(x1, s))                                            # y*c + x*s
    return np.concatenate([store(lo), store(hi)], axis=-1)


#: the k16 step twin per registered `DOT` primitive id (`b1.DotBf16V2{K, DOT}` / `Gemm_v2{K,N,DOT}` statics; the spec text carries
#: `DOT={"fn":"<id>"}`).  A DOT outside this table has no twin: the consumer refuses by name, never falls back to another chain.
TC_DOT16_BY_PRIM = {"AmpereBF16TcDot16_v2": tc_dot16, "HopperBF16WgmmaDot16_v1": tc_dot16_hopper}


def _dot_bf16_chain(q: np.ndarray, k: np.ndarray, step) -> tuple[np.ndarray, np.ndarray]:
    D = q.shape[-1]
    if D % 16:
        raise ValueError(f"a k16 chain needs K a multiple of 16 (K = {D})")
    acc = np.zeros(np.broadcast_shapes(q.shape[:-1], k.shape[:-1]), dtype=np.uint32)
    nf = np.zeros(acc.shape, dtype=bool)
    for s0 in range(0, D, 16):
        acc, f = step(acc, np.broadcast_to(q[..., s0:s0 + 16], nf.shape + (16,)),
                      np.broadcast_to(k[..., s0:s0 + 16], nf.shape + (16,)))
        nf |= f
    return acc, nf


def dot_bf16(q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`DotBf16_v1{K=D}` vectorised: q, k (..., D) bf16 words -> (f32 words, nonfinite).  Ascending k-chain
    of D/16 `tc_dot16` steps from a zero accumulator."""
    return _dot_bf16_chain(q, k, tc_dot16)


def dot_bf16_hopper(q: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`DotBf16_v2{K=D, DOT=HopperBF16WgmmaDot16_v1}` vectorised: q, k (..., D) bf16 words -> (f32 words, nonfinite).
    Ascending k-chain of D/16 `tc_dot16_hopper` steps from a zero accumulator (cuBLASLt with split-K disabled, the bf16 Linear
    of a VLLM_BATCH_INVARIANT=1 step on cc 9.0; pinned word for word on the H100, see the section comment above)."""
    return _dot_bf16_chain(q, k, tc_dot16_hopper)


def dot_bf16_for(dot_prim_id: str):
    """The vectorised `DotBf16_v2{K, DOT}` chain for a registered DOT primitive id, or a `KeyError` naming the id (the Gemm_v2
    evaluator's dispatch; an unregistered step is a refusal by name, never a silent fallback to another architecture's adder)."""
    if dot_prim_id not in TC_DOT16_BY_PRIM:
        raise KeyError(f"no vectorised k16 twin for DOT primitive {dot_prim_id!r} (registered: {sorted(TC_DOT16_BY_PRIM)})")
    step = TC_DOT16_BY_PRIM[dot_prim_id]
    return lambda q, k: _dot_bf16_chain(q, k, step)


# ---------------------------------------------------------------------------------------------------------
# Hopper FP8 tensor-core k32 step, vectorised (veritor.core.silicon HOPPER_E4M3_K32 / silicon_total_fp8
# HOPPER_E4M3_WGMMA_K32: ONE group of 32 exact e4m3 products + the accumulator, 14-bit truncating adder, exponent
# floor -139; measured on an H100, 144,543,744 words, 0 mismatches) and the two block-scaled FP8 linear
# families of the Qwen3-4B-FP8 Program (`verity/ml/definitions/registry/fp8.py`: Fp8GroupQuant_v1{K,G}, ScaledMmFp8Block_v1{K,N,G}).
# The scalar Definitions are the reference; these twins are checked word-for-word against them and against the saved
# H100 kernel words (`tests/program/data/fp8_block_operands.npz`) in tests/program/test_derived_rows_fp8.py.
# ---------------------------------------------------------------------------------------------------------

_FP8_TC_WIDTH = 14
_FP8_TC_FLOOR = -139
_FP8_RESCALE = 24 - _FP8_TC_WIDTH     # 10: the 24-bit-grid magnitudes are TRUNCATED to the adder (>> 10) and widened back (<< 10)
_E4M3_NAN_MASK = 0x7F                 # 0x7F / 0xFF: exponent and mantissa all ones (OCP E4M3 has no infinities)


def tc_dot32_e4m3(acc: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`HopperE4m3QgmmaDot32_v1` vectorised: acc (...,) f32 words, a/b (..., 32) e4m3 bytes -> (f32 words, nonfinite mask).

    One accumulator-coordinate transition of `wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3`: the 32 exact products
    (`silicon.e4m3_product`: significand (sa*sb) << 17 on the 24-bit grid, exponent (ea or 1) + (eb or 1) - 14, subnormals
    unnormalised at -6) AND the accumulator are rescaled to the 14-bit adder by truncation (the accumulator loses its low 10
    bits -- `acc + 0 != acc`, the measured `accid` family), aligned to the largest exponent (floor -139) by truncating right
    shifts, summed exactly, normalised to 14 bits by truncation, denormalised below 2^-126, widened back to 24 bits.  Lanes
    with an e4m3 NaN byte or a non-finite accumulator are flagged (the total semantics' NaN / sticky-infinity rules are not
    replicated here: flagged lanes are `unresolved` for the replay, the tc_dot16 convention).  A finite overflow above the
    FP32 range saturates to the infinity of its sign, as the silicon model does."""
    acc = np.asarray(acc, dtype=np.uint32)
    a = np.asarray(a, dtype=np.uint8).astype(np.int64)
    b = np.asarray(b, dtype=np.uint8).astype(np.int64)
    ea, eb = (a >> 3) & 0xF, (b >> 3) & 0xF
    nonfin = ((a & _E4M3_NAN_MASK) == _E4M3_NAN_MASK).any(axis=-1) | ((b & _E4M3_NAN_MASK) == _E4M3_NAN_MASK).any(axis=-1) \
        | ((acc & 0x7F800000) == 0x7F800000)
    sa = (a & 0x7) | ((ea != 0).astype(np.int64) << 3)
    sb = (b & 0x7) | ((eb != 0).astype(np.int64) << 3)
    p_mag = (sa * sb) << 17                                   # 24-bit-grid magnitudes (bit 23 at p_exp; <= 225 << 17 < 2^25)
    p_exp = np.maximum(ea, 1) + np.maximum(eb, 1) - 14
    p_neg = (((a ^ b) >> 7) & 1).astype(bool)
    acc_i = acc.astype(np.int64)
    a_ef = (acc_i >> 23) & 0xFF
    a_mant = acc_i & 0x7FFFFF
    acc_neg = (acc_i >> 31) == 1
    acc_mag = np.where(a_ef == 0, a_mant, a_mant | 0x800000)
    acc_exp = np.where(a_ef == 0, np.int64(-126), a_ef - 127)
    mags = np.concatenate([acc_mag[..., None], p_mag], axis=-1)
    exps = np.concatenate([acc_exp[..., None], p_exp], axis=-1)
    negs = np.concatenate([acc_neg[..., None], p_neg], axis=-1)
    live = mags != 0
    max_exp = np.max(np.where(live, exps, np.int64(-(1 << 40))), axis=-1)
    max_exp = np.maximum(max_exp, np.int64(_FP8_TC_FLOOR))
    scaled = mags >> _FP8_RESCALE                              # rescale = width - 24 < 0: truncating right shift
    shift = np.clip(max_exp[..., None] - exps, 0, 63)
    aligned = scaled >> shift
    total = np.sum(np.where(negs, -aligned, aligned) * live, axis=-1)
    neg = total < 0
    mag = np.abs(total)
    bl = _bitlen(mag)
    exp = max_exp + bl - _FP8_TC_WIDTH
    down = np.clip(bl - _FP8_TC_WIDTH, 0, 63)
    up = np.clip(_FP8_TC_WIDTH - bl, 0, 63)
    mag = np.where(bl > _FP8_TC_WIDTH, mag >> down, mag << up)
    sub = np.clip(np.int64(-126) - exp, 0, 63)
    mag = mag >> sub
    exp = np.maximum(exp, np.int64(-126))
    mag = mag << _FP8_RESCALE                                  # back to the 24-bit grid
    zero = mag == 0
    over = (~zero) & (exp > 127)
    exp_field = exp + 127
    exp_field = exp_field - ((mag & (1 << 23)) == 0)           # subnormal at exp -126 -> field 0
    word = (((neg & ~zero).astype(np.int64) << 31) | (exp_field << 23) | (mag & 0x7FFFFF)).astype(np.uint32)
    word = np.where(zero, np.uint32(0), word)
    word = np.where(over, (neg.astype(np.uint32) << 31) | np.uint32(0x7F800000), word)
    return word, nonfin


def scaled_mm_fp8_block_row(xq: np.ndarray, sx: np.ndarray, w: np.ndarray, ws: np.ndarray, *, G: int = 128,
                            chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """`ScaledMmFp8Block_v1{K,N,G}` on ONE token row (`_C.cutlass_scaled_mm` blockwise, CutlassFp8BlockScaledMMKernel): xq (K,) e4m3
    bytes, sx (K/G,) f32 words, w (N, K) e4m3 bytes, ws (cdiv(N, G), K/G) f32 words -> (N bf16 words, nonfinite mask).  Per output
    column n (`ScaledMmFp8BlockCoordinate`): for kb ascending, `temp` = G/32 `tc_dot32_e4m3` steps from +0 over the tile's bytes, `s` =
    FMUL(sx[kb], ws[n // G][kb]), `acc` = FFMA(temp, s, acc) from +0 (FFMA, not mul+add; ascending k), then `cvt.rn.bf16.f32`.  The
    last weight block may be partial (`fp8.weight_blocks`).  Vectorised over <= `chunk` columns (bounds the int64 temporaries)."""
    xq = np.asarray(xq, dtype=np.uint8).reshape(-1)
    K = xq.size
    w = np.asarray(w, dtype=np.uint8).reshape(-1, K)
    N = w.shape[0]
    if K % G or G % 32:
        raise ValueError(f"ScaledMmFp8Block: K={K} must be a multiple of G={G} (G a multiple of 32)")
    sx = np.asarray(sx, dtype=np.uint32).reshape(K // G)
    ws = np.asarray(ws, dtype=np.uint32).reshape(-(-N // G), K // G)
    out = np.empty(N, dtype=np.uint16)
    nonfin = np.zeros(N, dtype=bool)
    for n0 in range(0, N, chunk):
        n1 = min(N, n0 + chunk)
        wc = w[n0:n1]
        rows = np.arange(n0, n1) // G
        acc = np.zeros(n1 - n0, dtype=np.uint32)
        nf = np.zeros(n1 - n0, dtype=bool)
        for kb in range(K // G):
            temp = np.zeros(n1 - n0, dtype=np.uint32)
            for st in range(G // 32):
                lo = kb * G + 32 * st
                temp, f = tc_dot32_e4m3(temp, np.broadcast_to(xq[lo:lo + 32], (n1 - n0, 32)), wc[:, lo:lo + 32])
                nf |= f
            s = mul_rn(np.broadcast_to(sx[kb], (n1 - n0,)), ws[rows, kb])
            acc = fma_rn(temp, s, acc)
        out[n0:n1] = f2fp_bf16(acc)
        nonfin[n0:n1] = nf | ((acc & 0x7F800000) == 0x7F800000)
    return out, nonfin


def fp8_group_quant_row(x: np.ndarray, *, G: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """`Fp8GroupQuant_v1{K,G}` on ONE bf16 row (`_C.per_token_group_fp8_quant`, GroupShape(1, G)): x (K,) bf16 words -> (xq (K,) e4m3
    bytes, sx (K/G,) f32 words).  Per group: `s = fmaxf-fold(eps=1e-10f, |f32(x_e)|) / 448.0f` (an exact max -- fmaxf drops a NaN
    operand -- then an IEEE div.rn.f32), `xq[e] = cvt.rn.satfinite.e4m3(fminf(fmaxf(f32(x[e]) / s, -448), 448))` (a true division; a NaN
    quotient is dropped by fmaxf -> -448 -> 0xFE, the kernel's word)."""
    from verity.ml.definitions.numerics.fp8_twins import np_f32_to_e4m3_sat     # the Definition's own vectorised `cvt.rn.satfinite.e4m3` twin
    x = np.asarray(x, dtype=np.uint16).reshape(-1)
    K = x.size
    if K % G:
        raise ValueError(f"Fp8GroupQuant: K={K} is not a multiple of G={G}")
    xf = as_f32(bf16_to_f32_words(x)).reshape(K // G, G)
    ax = np.abs(xf)
    m = np.max(np.where(np.isnan(ax), np.float32(-np.inf), ax), axis=1, initial=np.float32(1e-10)).astype(np.float32)   # fmaxf fold from eps
    s = div_rn(as_u32(m), as_u32(np.full(K // G, np.float32(448.0))))                                                 # (K/G,) f32 words
    q = as_f32(div_rn(as_u32(xf), np.broadcast_to(s[:, None], (K // G, G))))
    q = np.where(np.isnan(q), np.float32(-448.0), np.maximum(q, np.float32(-448.0)))                                   # fmaxf(q, -448)
    q = np.minimum(q, np.float32(448.0)).astype(np.float32)                                                             # fminf(., 448)
    return np_f32_to_e4m3_sat(q).reshape(K), s


# ---------------------------------------------------------------------------------------------------------
# RMSNorm (Triton batch-invariant `_rms_norm_kernel`) — emits (sum, mean_eps, rms, inv)
# ---------------------------------------------------------------------------------------------------------


def rmsnorm_triton_rows(x: np.ndarray, w: np.ndarray, eps: float):
    """`RMSNormTriton_v1{N, EPS}` over rows.  x (R, N) bf16 words, w (N,) or (R, N) bf16 words.

    Returns dict with bf16 ``out`` (R, N) and the derived f32 words ``sum``, ``mean_eps``, ``rms``,
    ``inv`` (each (R,)).  Tree: chunks of 1024 columns; 128 threads x 8 columns
    (`acc = x1*x1; fma(x0, x0, acc); fma(x2..x7)` — the bf16 squares are exact in f32, so the fmas are
    RN adds of exact products); warp xor-butterfly 16, 8, 4, 2, 1; CTA `(w0 + w2) + (w1 + w3)`; chunks
    added in order from +0.  Epilogue: `mean_eps = fma(DivFullRcp(N), sum, eps)` (true fused),
    `rms = MufuSqrtFtz(mean_eps)`, `inv = DivFullScaleA(1, rms) * DivFullRcp(rms)`,
    `out_j = bf16_rne((inv * x_j) * w_j)`."""
    x = np.ascontiguousarray(x, dtype=np.uint16)
    R, N = x.shape
    w2 = np.broadcast_to(np.asarray(w, dtype=np.uint16), (R, N))
    xf = as_f32(bf16_to_f32_words(x))
    total = np.zeros(R, dtype=np.uint32)  # Const32[0]
    for c0 in range(0, N, 1024):
        lanes = []  # 128 per-thread partial sums, f32 words
        for t in range(128):
            base = c0 + 8 * t
            if base >= N:
                lanes.append(np.zeros(R, dtype=np.uint32))
                continue
            xs = xf[:, base:base + 8]  # exact: base + 8 <= N (N % 8 == 0 for every B0/B1 width)
            sq = as_u32((xs * xs).astype(np.float32))  # exact products (bf16-derived inputs)
            # acc = x1*x1 ; fma(x0, x0, acc) == RN(x0^2 + acc) ; fma(x2..x7) likewise (products exact)
            acc = sq[:, 1]
            acc = add_rn(sq[:, 0], acc)
            for j in range(2, 8):
                acc = add_rn(sq[:, j], acc)
            lanes.append(acc)
        # xor-butterfly per warp (masks 16, 8, 4, 2, 1), all 32 lanes valid
        warps = []
        for wi in range(4):
            v = lanes[32 * wi:32 * wi + 32]
            for msk in (16, 8, 4, 2, 1):
                v = [add_rn(v[i], v[i ^ msk]) for i in range(32)]
            warps.append(v[0])
        chunk = add_rn(add_rn(warps[0], warps[2]), add_rn(warps[1], warps[3]))
        total = add_rn(total, chunk)
    rcp_n = div_full_rcp(np.full(R, P.bits_of(np.float32(float(N))), dtype=np.uint32))
    eps_w = np.full(R, P.bits_of(np.float32(eps)), dtype=np.uint32)
    mean_eps = fma_rn(rcp_n, total, eps_w)
    rms = mufu_sqrt_ftz(mean_eps)
    one = np.full(R, 0x3F800000, dtype=np.uint32)
    inv = mul_rn(div_full_scale_a(one, rms), div_full_rcp(rms))
    y = mul_rn(mul_rn(inv[:, None], as_u32(xf)), bf16_to_f32_words(w2))
    return {"out": f2fp_bf16(y), "sum": total, "mean_eps": mean_eps, "rms": rms, "inv": inv}


# ---------------------------------------------------------------------------------------------------------
# RMSNorm fused CUDA v2 — thin wrapper over the committed twin; emits (var, s)
# ---------------------------------------------------------------------------------------------------------


def rmsnorm_fused_v2_rows(x: np.ndarray, res: np.ndarray, w: np.ndarray, eps: float):
    """`RMSNormFusedCuda_v2{N, EPS}` over rows (x, res, w: (R, N) bf16 words).  Returns dict with bf16
    ``normed``/``resid`` and the derived f32 words ``var`` (cub BlockReduce output) and ``s``
    (`RsqrtApprox(var / N + eps)`).  Delegates to `rmsnorm_fused.fused_rows` (the twin that
    reproduced 2,340/2,340 fused-norm rows of the B0 store) with the v2 variance tree."""
    from verity.ml.definitions.numerics.rmsnorm_fused import fused_rows, variance_v2
    R, N = np.asarray(x).shape
    w2 = np.ascontiguousarray(np.broadcast_to(np.asarray(w, dtype=np.uint16), (R, N)))
    out, rp, var, s = fused_rows(np.ascontiguousarray(x, dtype=np.uint16),
                                 np.ascontiguousarray(res, dtype=np.uint16), w2, eps, variance_v2)
    return {"normed": out, "resid": rp, "var": as_u32(var), "s": as_u32(s)}


# ---------------------------------------------------------------------------------------------------------
# SiLU-and-mul — emits the bf16 silu intermediate through the registered total table
# ---------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def silu_mid_table() -> np.ndarray:
    """The 2^16 bf16 -> bf16 silu table of `SiluMulBf16_v1`'s interior: `bf16_rn(silu_f32(f32(g)))`,
    NaN -> 0x7FC0 — built from the registered `_silu_f32_bits` (float64-exact stand-in), not measured."""
    out = np.empty(65536, dtype=np.uint16)
    for gw in range(65536):
        out[gw] = P.f32_to_bf16_rn_bits(P._silu_f32_bits(gw << 16), P.BF16_NAN_TORCH)
    return out


def silu_mul_rows(gu: np.ndarray):
    """`SiluMul_v1{I}` over rows: gu (R, 2I) bf16 words.  Returns dict with the derived bf16 ``mid``
    (R, I) = silu(gate) and the leaf ``out`` (R, I) = bf16(f32(mid) * f32(up))."""
    gu = np.ascontiguousarray(gu, dtype=np.uint16)
    R, GU = gu.shape
    half = GU // 2
    g, u = gu[:, :half], gu[:, half:]
    mid = silu_mid_table()[g]
    prod = mul_rn(bf16_to_f32_words(mid), bf16_to_f32_words(u))
    return {"mid": mid, "out": bf16_rn_torch(prod)}


# ---------------------------------------------------------------------------------------------------------
# Attention (FA2 online softmax, one (query, head) row) — emits per-block (m, l[4], o[D])
# ---------------------------------------------------------------------------------------------------------

BN = b1.BN  # `AttnBlock_v1`'s block (64).  `attn_block` reads the block size from the K block it is given; `attention_head_states` takes ``BN``


def attn_block(q, kb, vb, m_prev, l_prev, o_prev, *, D: int, NVIS: int, FIRST: bool, dot16=None, check: bool = True, rescale_only: bool = False, scale_bits: int | None = None):
    """`AttnBlock_v1{D, NVIS, FIRST}` vectorised over rows (`AttnBlock_v2{..., BN, DOT}` with ``dot16`` = the DOT step's twin,
    `TC_DOT16_BY_PRIM[DOT]`; default the Ampere `tc_dot16`; `AttnBlock_v4{..., CHECK}` with ``check``: False reads m unguarded; `AttnBlock_v5` also ``rescale_only``; ``scale_bits`` the f32 word of `scale_log2` when the block reads it as an input (`AttnBlock_v10`/`_v11`), else `b1.scale_log2_bits(D)`).

    q (R, D), kb/vb (R, BN, D) bf16 words (rows past NVIS must be zero — `Clear_OOB_MN` padding);
    m_prev (R,), l_prev (R, 4), o_prev (R, D) f32 words (ignored when FIRST).
    Returns (m_new, l (R,4), o (R,D), nonfinite (R,))."""
    BN = kb.shape[1]  # the block size is the block's: 64 (`AttnBlock_v1`) or `b1.fa2_kblock_n(D)` (`AttnBlock_v3`)
    step = tc_dot16 if dot16 is None else dot16
    R = q.shape[0]
    nf = np.zeros(R, dtype=bool)
    s, f = _dot_bf16_chain(q[:, None, :], kb[:, :NVIS, :], step)  # (R, NVIS): the D/16-step k-chains, vectorised

    nf |= f.any(axis=-1)
    bmax = s[:, 0]
    for c in range(1, NVIS):
        bmax = max_fmnmx(bmax, s[:, c])
    m_new = bmax if FIRST else max_fmnmx(m_prev, bmax)
    m_use = guard_neg_inf_zero(m_new) if check else m_new
    scale = np.full(R, b1.scale_log2_bits(D) if scale_bits is None else int(scale_bits), dtype=np.uint32)
    if not FIRST:
        scores_scale = mufu_ex2_ftz(mul_ftz(sub_ftz(m_prev, m_use), scale))
    max_scaled = mul_ftz(guard_neg_inf_zero(m_new) if rescale_only else m_use, scale)
    p = mufu_ex2_ftz(fma_sub_ftz(s, scale[:, None], max_scaled[:, None]))  # (R, NVIS)
    lanes = []
    for ln in range(4):
        cols = [c for c in range(NVIS) if b1._lane(c) == ln]
        zero = np.zeros(R, dtype=np.uint32)
        first_p = p[:, cols[0]] if cols else zero
        if FIRST:
            acc = first_p
        else:
            acc = fma_ftz(scores_scale, l_prev[:, ln], first_p)
        for c in cols[1:]:
            acc = add_ftz(acc, p[:, c])
        lanes.append(acc)
    l_new = np.stack(lanes, axis=1)
    pb = f2fp_bf16(p)  # (R, NVIS)
    if NVIS < BN:
        pb = np.concatenate([pb, np.zeros((R, BN - NVIS), dtype=np.uint16)], axis=1)
    if FIRST:
        o = np.zeros((R, D), dtype=np.uint32)
    else:
        o = mul_ftz(o_prev, scores_scale[:, None])
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break
        prow = pb[:, k0:k0 + 16]  # (R, 16)
        vt = np.swapaxes(vb[:, k0:k0 + 16, :], 1, 2)  # (R, D, 16)
        o, f = step(o, np.broadcast_to(prow[:, None, :], (R, D, 16)), vt)
        nf |= f.any(axis=-1)
    return m_new, l_new, o, nf


def fa3_inv_sum(s: np.ndarray) -> np.ndarray:
    """`Fa3InvSum_v1` (hopper/softmax.h finalize): (sum == +-0 || NaN) ? 0.0 : MUFU.RCP(sum)."""
    s = np.asarray(s, dtype=np.uint32)
    degen = ((s & 0x7FFFFFFF) == 0) | (((s & 0x7F800000) == 0x7F800000) & ((s & 0x7FFFFF) != 0))
    return np.where(degen, np.uint32(0), mufu_rcp_ftz(s))


#: the softmax-epilogue inverse-sum twin per registered INV primitive id (`Attention_v2{..., INV}`)
INV_SUM_BY_PRIM = {"Fa2InvSum_v1": fa2_inv_sum, "Fa3InvSum_v1": fa3_inv_sum}


def attention_head_states(q: np.ndarray, k: np.ndarray, v: np.ndarray, BN: int | None = None, *, dot16=None, inv=None, masked_from: int | None = None, rescale_only: bool = False, scale_bits: int | None = None):
    """`AttentionHead_v1{T, D}` (``BN`` None / 64) or `AttentionHead_v3{T, D}` (``BN = b1.fa2_kblock_n(D)``) vectorised over rows,
    emitting every block state; `AttentionHead_v2{T, D, BN, DOT, INV}` with ``dot16`` = `TC_DOT16_BY_PRIM[DOT]` and ``inv`` =
    `INV_SUM_BY_PRIM[INV]` (defaults: the Ampere step, `Fa2InvSum_v1`); `AttentionHead_v4{..., MASKED_FROM}` with ``masked_from``, `_v5` also ``rescale_only``.

    q (R, D); k, v (R, T, D) bf16 words (per-row KV, GQA-expanded by the caller).  Returns dict:
    ``m`` (R, nb), ``l`` (R, nb, 4), ``o`` (R, nb, D) f32 words in *processing order* (block nb-1 first),
    ``blocks`` — the block indices in processing order, ``out`` (R, D) bf16 words, ``inv``/``total`` (R,),
    ``nonfinite`` (R,)."""
    q = np.ascontiguousarray(q, dtype=np.uint16)
    k = np.ascontiguousarray(k, dtype=np.uint16)
    v = np.ascontiguousarray(v, dtype=np.uint16)
    inv_fn = fa2_inv_sum if inv is None else inv
    R, T, D = k.shape
    if BN is None:
        BN = b1.BN  # the v1 chain
    nb = (T + BN - 1) // BN
    ms, ls, os = [], [], []
    nf = np.zeros(R, dtype=bool)
    m = l = o = None
    order = list(range(nb - 1, -1, -1))
    for b in order:
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[:, lo:hi], v[:, lo:hi]
        if nvis < BN:
            pad = np.zeros((R, BN - nvis, D), dtype=np.uint16)
            kb = np.concatenate([kb, pad], axis=1)
            vb = np.concatenate([vb, pad], axis=1)
        first = b == nb - 1
        zero = np.zeros(R, dtype=np.uint32)
        m, l, o, f = attn_block(q, kb, vb,
                                zero if first else m,
                                np.zeros((R, 4), np.uint32) if first else l,
                                np.zeros((R, D), np.uint32) if first else o,
                                D=D, NVIS=nvis, FIRST=first, dot16=dot16, check=masked_from is None or b >= masked_from, rescale_only=rescale_only, scale_bits=scale_bits)
        nf |= f
        ms.append(m)
        ls.append(l)
        os.append(o)
    total = add_ftz(add_ftz(l[:, 0], l[:, 2]), add_ftz(l[:, 1], l[:, 3]))
    inv = inv_fn(total)
    y = mul_ftz(o, inv[:, None])
    return {"m": np.stack(ms, 1), "l": np.stack(ls, 1), "o": np.stack(os, 1), "blocks": order,
            "out": f2fp_bf16(y), "total": total, "inv": inv, "nonfinite": nf}


def attention_rows(q: np.ndarray, kc: np.ndarray, vc: np.ndarray, BN: int, *, dot_prim: str = "AmpereBF16TcDot16_v2",
                   inv_prim: str = "Fa2InvSum_v1", masked_from: int | None = None, rescale_only: bool = False, scale_bits: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """one attention ROW of the Program -- `Attention_v3{T, NH, KVH, D, BN}` (FA2, the defaults) or
    `Attention_v2{T, NH, KVH, D, BN, DOT, INV}` (FA3 on sm_90: DOT = HopperBF16WgmmaDot16_v1, INV = Fa3InvSum_v1; `Attention_v4` adds ``masked_from``, `Attention_v5` ``rescale_only`` as well) -- exactly as the
    registered Definitions compose it: head h reads KV head h // (NH/KVH); one query at position T-1 over keys 0..T-1, ceil(T/BN) blocks
    visited last -> first (`AttentionHead_v2/v3`).  q (NH, D), kc / vc (T, KVH, D) bf16 words -> (out (NH, D) bf16 words, nonfinite (NH,)).
    This is the composition of the row's proof blocks on its committed boundary (q, the K/V history, out); the block INTERIORS Q(P) selects live in
    the M1 export stream and are not read here -- the caller names that mechanism."""
    q = np.asarray(q, dtype=np.uint16)
    kc = np.asarray(kc, dtype=np.uint16)
    vc = np.asarray(vc, dtype=np.uint16)
    NH, D = q.shape
    _T, KVH, D2 = kc.shape
    if D2 != D or vc.shape != kc.shape or NH % KVH:
        raise ValueError(f"attention row shapes: q {q.shape} kc {kc.shape} vc {vc.shape}")
    if int(BN) <= 0 or int(BN) % 16 or D % 16:
        # the P.V product is a k16 chain over the key block (attn_block): a block that is not a multiple of 16 keys, or a head dim that is
        # not a multiple of 16, is outside the registered relation -- refused, never evaluated with a truncated chain
        raise ValueError(f"attention row BN = {BN}, D = {D}: the k16 chains need multiples of 16")
    if dot_prim not in TC_DOT16_BY_PRIM:
        raise KeyError(f"no vectorised k16 twin for DOT primitive {dot_prim!r} (registered: {sorted(TC_DOT16_BY_PRIM)})")
    if inv_prim not in INV_SUM_BY_PRIM:
        raise KeyError(f"no vectorised inverse-sum twin for INV primitive {inv_prim!r} (registered: {sorted(INV_SUM_BY_PRIM)})")
    G = NH // KVH
    kvh = np.arange(NH) // G                                                     # (NH,): the KV head of each query head
    k = np.ascontiguousarray(np.transpose(kc, (1, 0, 2))[kvh])                   # (NH, T, D) GQA-expanded views
    v = np.ascontiguousarray(np.transpose(vc, (1, 0, 2))[kvh])
    st = attention_head_states(q, k, v, int(BN), dot16=TC_DOT16_BY_PRIM[dot_prim], inv=INV_SUM_BY_PRIM[inv_prim], masked_from=masked_from, rescale_only=rescale_only, scale_bits=scale_bits)
    return st["out"], st["nonfinite"]


__all__ = ["rmsnorm_triton_rows", "rmsnorm_fused_v2_rows", "silu_mul_rows", "attn_block",
           "attention_head_states", "attention_rows", "INV_SUM_BY_PRIM", "tc_dot16", "dot_bf16", "fma_rn", "fma_ftz", "fma_sub_ftz",
           "mufu_ex2_ftz", "mufu_rcp_ftz", "fa2_inv_sum", "fa3_inv_sum", "mufu_sqrt_ftz", "rsqrt_approx",
           "div_full_rcp", "div_full_scale_a", "silu_mid_table", "f2fp_bf16", "bf16_rn_torch", "ftz",
           "add_ftz", "sub_ftz", "mul_ftz", "max_fmnmx", "guard_neg_inf_zero", "as_u32", "as_f32",
           "bf16_to_f32_words"]

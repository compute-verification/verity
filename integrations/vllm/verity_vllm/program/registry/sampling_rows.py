"""Vectorised exact twins of the sampler Definition — `GumbelTopPTokenSelect_v1{V}` over a whole vocabulary row at once
(the `derived_rows` pattern: numpy twins of registered gate semantics, held word-equal to the lane-walk evaluators).

[R17 stoch, COORD R17-34 (3) / R17-28 (2) lever (a)]  `stoch_recompute` evaluates every sampled row of a record through the
Definition of record, `GumbelTopPTokenSelect_v1 = TemperatureScale_v1 -> TopPMask_v1 -> GumbelSelectF32_v1`, by the generic
gate interpreter: 17 gates x V lanes per row (2.18 M gates at V = 128,256), ~44 s/row of which 60 % is the interpreter's
per-gate address resolution and 29 % the per-lane fp32 emulation through `struct.pack` (profiles/README.md).  This module
computes the SAME words over the row with numpy — the same Philox4x32-10, the same libdevice `__nv_logf` / `__nv_log1pf`
transcriptions, the same FTZ / RN / RZ rules, the same `div.full.f32` MUFU table, the same `topp_keep_row` — so a row costs
milliseconds instead of tens of seconds.

WHAT THIS IS NOT: not a new Definition (nothing in `sampling.py` changes; no Program digest moves) and not a weaker check —
the Definition stays the reference; `tests/program/test_sampling_rows.py` holds every function here word-equal to the scalar
registry functions (`sampling.noise_lane` / `nv_logf` / `nv_log1pf` / `fma_ftz` / `add_rz_ftz` / `stream_key`) and the whole row
equal to `evaluate_spec(GumbelTopPTokenSelect_v1)` on the R16 GPU fixture rows and random rows at every S, and `stoch_recompute`
keeps a per-run sample of rows through the lane walk beside the vectorised path (`--reference-rows`), naming both on the record.

Bit-exactness notes (each mirrors the scalar function it twins, by name):
  * `fma_ftz`  — `derived_rows.fma_ftz` (Boldo–Melquiond round-to-odd through float64; validated against `prims.f32_fma_bits`).
    `sampling.fma_ftz` rounds the same exact value the same way (TwoSum + exact tie rule); both are RN32 of a*b + c.  The
    only difference is NaN canonicalisation: derived_rows returns `P.F32_NAN`, sampling returns the struct NaN — no finite
    logit row reaches a NaN inside the noise chain (u in [2^-31', 1)), and a NaN logit lane is `y = NaN`, which never wins.
  * `add_rz_ftz` — equals the REGISTERED scalar, whose toward-zero step is a no-op (see the function: F-stoch-r17-01, revb's call).
  * `nv_logf` / `nv_log1pf` — the int32 exponent arithmetic in int64 with explicit wraparound; the branches as `np.where`.
  * `philox` — uint64 products of uint32 operands (exact), high/low words as in `philox4x32_10_word0`.
  * the select — `GumbelSelectStep`'s scan: best = y[0], i = 0; a lane replaces the best iff `y > best` (IEEE strict, NaN false):
    the FIRST lane holding the row maximum; a NaN y[0] pins the result at 0 (nothing compares greater than NaN).
"""

from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import sampling as SP

U32 = np.uint32
I64 = np.int64
M32 = 0xFFFFFFFF


def _u32(x) -> np.ndarray:
    return np.asarray(x, dtype=U32)


def _i32_of_u32(u: np.ndarray) -> np.ndarray:
    """int64 array holding the int32 reading of uint32 words (two's complement)."""
    u = _u32(u).astype(I64)
    return np.where(u & 0x80000000, u - (1 << 32), u)


def _u32_of_i(i: np.ndarray) -> np.ndarray:
    """uint32 words of int64 values reduced mod 2^32."""
    return (np.asarray(i, dtype=I64) & M32).astype(U32)


def f32_of_int(i: np.ndarray) -> np.ndarray:
    """`_u(float(i))`: the RN-even f32 word of an integer (through float64, exact for |i| < 2^53)."""
    return DR.as_u32(np.asarray(i, dtype=I64).astype(np.float64).astype(np.float32))


# ---------------------------------------------------------------------------------------------------------
# fp32 word arithmetic — the scalar helpers of `sampling.py`, vectorised
# ---------------------------------------------------------------------------------------------------------

fma_ftz = DR.fma_ftz
mul_ftz = DR.mul_ftz
add_ftz = DR.add_ftz
ftz = DR.ftz


def add_rz_ftz_v1_model(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[F-stoch-r17-01 WITNESS, test-only] `sampling_twins.add_rz_ftz_v1_model` lane for lane: the `add.rz.ftz.f32` model `nv_log1pf` used up
    to R17, which is ftz(RN32(RN64(ftz(a) + ftz(b)))) -- round to NEAREST (its toward-zero correction was dead code: a double-ulp step
    that f32 rounding undoes).  Kept so the exhaustive comparison with the true RZ (`nv_log1pf` vs `nv_log1pf_v1_model`) stays runnable."""
    a, b = np.broadcast_arrays(ftz(_u32(a)), ftz(_u32(b)))
    s = DR.as_f32(a).astype(np.float64) + DR.as_f32(b).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(DR.as_u32(s.astype(np.float32)))


def add_rz_ftz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`add.rz.ftz.f32`, TRUE toward zero — `sampling.add_rz_ftz` lane for lane: the exact sum s + e (TwoSum in float64, exact for f32
    operands); RN32(s) stepped one f32 ulp toward zero when its magnitude exceeds the exact value; a finite sum never overflows to inf
    (saturates at FLT_MAX); a zero sum keeps its sign; FTZ on the result."""
    a, b = np.broadcast_arrays(ftz(_u32(a)), ftz(_u32(b)))
    fa, fb = DR.as_f32(a).astype(np.float64), DR.as_f32(b).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        s = fa + fb
        bb = s - fa
        e = (fa - (s - bb)) + (fb - bb)
        r = DR.as_u32(s.astype(np.float32))
        rf = DR.as_f32(r).astype(np.float64)
        finite = np.isfinite(s) & (s != 0.0)
        sat = finite & np.isinf(rf)                                                # RN32 overflow of a finite sum: FLT_MAX with s's sign
        step = finite & ~np.isinf(rf) & ((np.abs(rf) > np.abs(s)) | ((rf == s) & (e != 0.0) & ((e < 0) == (s > 0)))) & ((r & 0x7FFFFFFF) != 0)
        r = np.where(step, (r & np.uint32(0x80000000)) | ((r & np.uint32(0x7FFFFFFF)) - np.uint32(1)), r)
        r = np.where(sat, (r & np.uint32(0x80000000)) | np.uint32(0x7F7FFFFF), r)
    return ftz(_u32(r))


def nv_logf(a: np.ndarray) -> np.ndarray:
    """libdevice `__nv_logf` (FTZ) on f32 word arrays — `sampling.nv_logf` lane for lane."""
    a = _u32(a)
    small = DR.as_f32(a) < DR.as_f32(_u32(SP.FLT_MIN_BITS))
    a2 = np.where(small, mul_ftz(a, _u32(SP.TWO_P23_BITS)), a)
    i0 = np.where(small, _u32(0xC1B80000), _u32(0))
    ai = _i32_of_u32(a2)
    e = (ai - 1059760811) & -8388608                       # int64 arithmetic; the mask keeps it a multiple of 2^23
    e_i32 = _i32_of_u32(_u32_of_i(e))
    m = _u32_of_i(ai - e_i32)
    i2 = fma_ftz(f32_of_int(e_i32), _u32(SP.TWO_M23_BITS), i0)
    f = add_ftz(m, _u32(0xBF800000))
    r = fma_ftz(_u32(SP.LOGF_POLY[0]), f, _u32(SP.LOGF_POLY[1]))
    for k in SP.LOGF_POLY[2:]:
        r = fma_ftz(r, f, _u32(k))
    r = fma_ftz(r, f, _u32(0xBF000000))
    r = mul_ftz(r, f)
    r = fma_ftz(r, f, f)
    r = fma_ftz(i2, _u32(SP.LN2_BITS), r)
    r = np.where(a2 >= 0x7F800000, np.where(_neg_not_nan(a2), _u32(SP.LOG_INVALID_NAN), fma_ftz(a2, _u32(0x7F800000), _u32(0x7F800000))), r)
    r = np.where((a2 & 0x7FFFFFFF) == 0, _u32(0xFF800000), r)
    return _u32(r)


def _neg_not_nan(a: np.ndarray) -> np.ndarray:
    return (a >= 0x80000000) & ((a & 0x7FFFFFFF) <= 0x7F800000)


def nv_log1pf(a: np.ndarray) -> np.ndarray:
    """libdevice `__nv_log1pf` (FTZ) on f32 word arrays — `sampling.nv_log1pf` lane for lane (`__fadd_rz(a, 1)` a true toward-zero add,
    F-stoch-r17-01)."""
    return _log1pf_from_sum(_u32(a), add_rz_ftz(_u32(a), _u32(SP.ONE_BITS)))


def nv_log1pf_v1_model(a: np.ndarray) -> np.ndarray:
    """[F-stoch-r17-01 WITNESS, test-only] `sampling_twins.nv_log1pf_v1_model` lane for lane: log1pf with the RN model of `__fadd_rz`."""
    return _log1pf_from_sum(_u32(a), add_rz_ftz_v1_model(_u32(a), _u32(SP.ONE_BITS)))


def _log1pf_from_sum(a: np.ndarray, s: np.ndarray) -> np.ndarray:
    """`sampling._log1pf_from_sum` lane for lane: the body of `__nv_log1pf` after its first instruction (s = the `__fadd_rz(a, 1)` word)."""
    si = _i32_of_u32(s)
    e = (si - 1061158912) & -8388608
    e_i32 = _i32_of_u32(_u32_of_i(e))
    m1 = _u32_of_i(_i32_of_u32(a) - e_i32)
    t = _u32_of_i(1082130432 - e_i32)
    r0 = fma_ftz(_u32(0x3E800000), t, _u32(0xBF800000))
    u = add_ftz(m1, r0)
    ef = mul_ftz(f32_of_int(e_i32), _u32(SP.TWO_M23_BITS))
    r = fma_ftz(_u32(SP.LOG1PF_POLY[0]), u, _u32(SP.LOG1PF_POLY[1]))
    for k in SP.LOG1PF_POLY[2:]:
        r = fma_ftz(r, u, _u32(k))
    r = fma_ftz(r, u, _u32(0xBF000000))
    r = mul_ftz(r, u)
    r = fma_ftz(r, u, u)
    r = fma_ftz(ef, _u32(SP.LN2_BITS), r)
    big = a >= 0x7F800000
    ai = _i32_of_u32(a)
    r = np.where(big & (ai > -1082130432),
                 np.where(_neg_not_nan(a), _u32(SP.LOG_INVALID_NAN), fma_ftz(a, _u32(0x7F800000), _u32(0x7F800000))), r)
    r = np.where(big & ((a & 0x7FFFFFFF) == 0), _u32(0x80000000), r)
    return _u32(r)


# ---------------------------------------------------------------------------------------------------------
# Philox4x32-10 and the Gumbel noise row
# ---------------------------------------------------------------------------------------------------------


def philox_word0(c0: np.ndarray, c1, c2, c3, k0, k1) -> np.ndarray:
    """`sampling.philox4x32_10_word0` over arrays of uint32 counters (keys broadcast)."""
    c0, c1, c2, c3, k0, k1 = np.broadcast_arrays(*(np.asarray(x, dtype=np.uint64) for x in (c0, c1, c2, c3, k0, k1)))
    c0, c1, c2, c3, k0, k1 = (x.copy() for x in (c0, c1, c2, c3, k0, k1))
    A, B_ = np.uint64(SP.PHILOX_ROUND_A), np.uint64(SP.PHILOX_ROUND_B)
    KA, KB = np.uint64(SP.PHILOX_KEY_A), np.uint64(SP.PHILOX_KEY_B)
    m32 = np.uint64(M32)
    for _ in range(10):
        pb = c2 * B_                                     # uint32 x uint32 fits uint64 exactly
        pa = c0 * A
        c0, c1, c2, c3 = ((pb >> np.uint64(32)) & m32) ^ c1 ^ k0, pb & m32, ((pa >> np.uint64(32)) & m32) ^ c3 ^ k1, pa & m32
        k0 = (k0 + KA) & m32
        k1 = (k1 + KB) & m32
    return c0.astype(U32)


def rand_row(key: int, V: int) -> np.ndarray:
    """f32 words of max(tl.rand(key, v), 2^-31') for v = 0..V-1 (`sampling._rand_lane` for the whole row)."""
    v = np.arange(int(V), dtype=np.uint64)
    w = philox_word0(v, 0, 0, 0, int(key) & M32, 0)
    xi = _i32_of_u32(w)
    xi = np.where(xi < 0, -xi - 1, xi)
    u = mul_ftz(f32_of_int(xi), _u32(SP.TL_RAND_SCALE_BITS))
    return _u32(np.where(DR.as_f32(u) < DR.as_f32(_u32(SP.TL_RAND_MIN_BITS)), _u32(SP.TL_RAND_MIN_BITS), u))


def noise_row(key: int, V: int) -> np.ndarray:
    """`GumbelNoiseLane_v1(key, v)` for v = 0..V-1 at once: f32 words of g(key)[v] = -logf(-log1pf(-max(rand(key, v), 2^-31')))."""
    u = rand_row(key, V)
    l1 = nv_log1pf(_u32(u) ^ np.uint32(0x80000000))
    return _u32(nv_logf(_u32(l1) ^ np.uint32(0x80000000)) ^ np.uint32(0x80000000))


# ---------------------------------------------------------------------------------------------------------
# the three stages of the Definition, per row
# ---------------------------------------------------------------------------------------------------------


def f32_eq(a: int, b: int) -> bool:
    """`F32Eq_v1`: IEEE equality of two f32 words (+0 == -0; NaN unequal)."""
    return bool(SP._f(int(a)) == SP._f(int(b)))


def temperature_scale_row(logits_bf16: np.ndarray, temp: int) -> np.ndarray:
    """`TemperatureScale_v1{V}`: f32 words of the widened bf16 row, divided by T through `div.full.f32`
    (`F32Mul(DivFullScaleA(x, T), DivFullRcp(T))`) unless T in {0, 1} (the kernel's early return)."""
    xf = DR.bf16_to_f32_words(np.asarray(logits_bf16, dtype=np.uint16))
    if f32_eq(temp, 0) or f32_eq(temp, SP.ONE_BITS):
        return _u32(xf)
    t = _u32([int(temp) & M32])
    rcp = DR.div_full_rcp(t)                                     # one word, the same MUFU.RCP table as `prims.DivFullRcp`
    return _u32(DR.mul_rn(DR.div_full_scale_a(xf, t), rcp))


def topp_mask_row(x_f32: np.ndarray, top_p: int, splits: int) -> np.ndarray:
    """`TopPMask_v1{V}`: the row with every lane outside the top-p nucleus set to -inf (`TopPMaskWordx{V}` keep bits, then
    `SelectF32(BitAt(keep, i), x, -inf)` per lane), over every i32 `splits` as the primitive (`sampling.topp_keep`)."""
    x = _u32(x_f32)
    keep = SP.topp_keep(DR.as_f32(x), SP._f(int(top_p)), int(splits))
    if keep.shape[0] != x.shape[0]:
        raise ValueError(f"topp_keep_row returned {keep.shape[0]} keep bits for a {x.shape[0]}-lane row")
    return np.where(keep, x, _u32(SP.NEG_INF_BITS))


def gumbel_select_row(x_f32: np.ndarray, seed: int, pos: int, temp: int) -> tuple[int, np.ndarray]:
    """`GumbelSelectF32_v1{V}`: (token, y) — the first index of max_v (x[v] + g(seed, pos)[v]) (`F32AddFtz`), or of max_v x[v]
    when temp == 0 (no noise); y is the noisy row the select ran over (f32 words), returned for the record."""
    x = _u32(x_f32).reshape(-1)
    V = x.shape[0]
    key = SP.stream_key(int(seed), int(pos))
    noisy = not f32_eq(temp, 0)
    y = add_ftz(x, noise_row(key, V)) if noisy else x
    yf = DR.as_f32(_u32(y))
    if np.isnan(yf[0]):
        return 0, _u32(y)                              # nothing compares strictly greater than a NaN best
    cand = np.where(np.isnan(yf), np.float32(-np.inf), yf)   # a NaN lane never wins; -inf lanes tie with the -inf default
    return int(np.argmax(cand)), _u32(y)               # argmax = FIRST index of the maximum, as the strict scan


def gumbel_topp_token_select_row(logits_bf16: np.ndarray, top_p: int, seed: int, pos: int, temp: int, splits: int) -> dict:
    """`GumbelTopPTokenSelect_v1{V}` over one row: the served token and the intermediate rows (f32 words) by stage name."""
    scaled = temperature_scale_row(logits_bf16, temp)
    masked = topp_mask_row(scaled, top_p, splits)
    tok, y = gumbel_select_row(masked, seed, pos, temp)
    return {"token": tok, "scaled": scaled, "masked": masked, "noisy": y}


__all__ = ["add_rz_ftz", "add_rz_ftz_v1_model", "nv_logf", "nv_log1pf", "nv_log1pf_v1_model", "philox_word0", "rand_row", "noise_row", "temperature_scale_row",
           "topp_mask_row", "gumbel_select_row", "gumbel_topp_token_select_row", "f32_of_int", "fma_ftz", "mul_ftz", "add_ftz", "ftz"]

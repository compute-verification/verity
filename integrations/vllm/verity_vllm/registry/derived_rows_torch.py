"""Torch twin of :mod:`verity_vllm.registry.derived_rows` — the same bit-exact derived-boundary evaluators
on ``torch`` tensors (CUDA or CPU), for the GPU leg of the acquisition-cost measurement.

Representation: f32 words and bf16 words are carried as ``int64`` tensors (values in [0, 2^32) / [0, 2^16));
float steps bitcast through ``int32.view(float32)``. Elementwise float32/float64 torch ops are IEEE RN-even
on both CUDA and CPU (TF32 only affects matmuls, which this module never issues); everything else is integer
bit manipulation, identical to the numpy module.  Equivalence with `derived_rows` (and hence with the
registered gate semantics) is asserted in `tests/ir/test_derived_rows_torch.py` and end-to-end by
`verity_vllm_adapter.derived_b0_check --backend torch`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from verity_vllm.registry import b1
from verity_vllm.registry import derived_rows as DR
from verity_vllm.registry import prims as P

M32 = 0xFFFFFFFF


def words_from_numpy(a: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(device)


def as_f32(w: torch.Tensor) -> torch.Tensor:
    """int64 f32 words -> float32 (bitcast)."""
    w32 = torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)
    return w32.view(torch.float32)


def as_u32(f: torch.Tensor) -> torch.Tensor:
    """float32 -> int64 f32 words."""
    return f.view(torch.int32).to(torch.int64) & M32


def bf16_to_f32_words(w: torch.Tensor) -> torch.Tensor:
    return w << 16


def f2fp_bf16(u: torch.Tensor) -> torch.Tensor:
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return torch.where(nan, torch.tensor(0x7FFF, dtype=torch.int64, device=u.device), r & 0xFFFF)


def bf16_rn_torch(u: torch.Tensor) -> torch.Tensor:
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return torch.where(nan, torch.tensor(0x7FC0, dtype=torch.int64, device=u.device), r & 0xFFFF)


def ftz(u: torch.Tensor) -> torch.Tensor:
    sub = ((u & 0x7F800000) == 0) & ((u & 0x007FFFFF) != 0)
    return torch.where(sub, u & 0x80000000, u)


def add_rn(a, b):
    return as_u32(as_f32(a) + as_f32(b))


def mul_rn(a, b):
    return as_u32(as_f32(a) * as_f32(b))


def add_ftz(a, b):
    return ftz(add_rn(ftz(a), ftz(b)))


def sub_ftz(a, b):
    return ftz(as_u32(as_f32(ftz(a)) - as_f32(ftz(b))))


def mul_ftz(a, b):
    return ftz(mul_rn(ftz(a), ftz(b)))


def max_fmnmx(a, b):
    a, b = ftz(a), ftz(b)
    return torch.where(as_f32(a) > as_f32(b), a, b)


def guard_neg_inf_zero(m):
    return torch.where(m == 0xFF800000, torch.zeros_like(m), m)


# -- true fma (float64 TwoSum + round-to-odd, as in derived_rows) -------------------------------------------


def _round_to_odd64(s: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    si = s.view(torch.int64)
    inexact = (e != 0) & torch.isfinite(s)
    move = inexact & ((si & 1) == 0)
    toward_pos = e > 0
    neg_s = (si >> 63) != 0
    delta = torch.where(toward_pos != neg_s, torch.tensor(1, dtype=torch.int64, device=s.device),
                        torch.tensor(-1, dtype=torch.int64, device=s.device))
    sign = si & -(1 << 63)
    mag = (si & ((1 << 63) - 1)) + torch.where(move, delta, torch.zeros_like(delta))
    return torch.where(move, sign | mag, si).view(torch.float64)


def fma_rn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    a, b, c = torch.broadcast_tensors(a, b, c)
    af, bf, cf = (as_f32(x).to(torch.float64) for x in (a, b, c))
    p = af * bf
    s = p + cf
    bp = s - p
    e = (p - (s - bp)) + (cf - bp)
    out = as_u32(_round_to_odd64(s, e).to(torch.float32))
    nan = ((out & 0x7F800000) == 0x7F800000) & ((out & 0x007FFFFF) != 0)
    return torch.where(nan, torch.tensor(P.F32_NAN, dtype=torch.int64, device=out.device), out)


def fma_ftz(a, b, c):
    return ftz(fma_rn(ftz(a), ftz(b), ftz(c)))


def fma_sub_ftz(a, b, c):
    return ftz(fma_rn(ftz(a), ftz(b), ftz(c) ^ 0x80000000))


# -- MUFU tables --------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _fa2_tables(device_str: str):
    t = DR._fa2_tables()
    dev = torch.device(device_str)
    return {"ex2": words_from_numpy(t.ex2, dev), "rcp": words_from_numpy(t.rcp, dev)}


@lru_cache(maxsize=None)
def _rms_tables(device_str: str):
    t = DR._rms_tables()
    dev = torch.device(device_str)
    return {"sqrt": words_from_numpy(t.sqrt, dev), "rcp": words_from_numpy(t.rcp, dev),
            "rsq": words_from_numpy(t.rsq, dev)}


def mufu_ex2_ftz(x: torch.Tensor) -> torch.Tensor:
    T = _fa2_tables(str(x.device))["ex2"]
    ax = x & 0x7FFFFFFF
    e = ax >> 23
    neg = (x >> 31) != 0
    mant = (ax & 0x7FFFFF) | 0x800000
    unb = e - 127
    sh_l = torch.clamp(unb, 0, 63)
    sh_r = (-unb) & 63  # the compiled C rule's mod-64 shift (see derived_rows.mufu_ex2_ftz)
    K = torch.where(unb >= 0, mant << sh_l, mant >> torch.where(unb < 0, sh_r, torch.zeros_like(sh_r)))
    frac_zero = (K & 0x7FFFFF) == 0
    fixed = torch.where(~neg, K, torch.where(frac_zero, -K, -K - 1))
    n = fixed >> 23
    j = fixed & 0x7FFFFF
    t = T[j]
    re = ((t >> 23) & 0xFF) + n
    out = torch.where(re >= 255, torch.tensor(0x7F800000, dtype=torch.int64, device=x.device),
                      torch.where(re <= 0, torch.zeros_like(re), (re << 23) | (t & 0x7FFFFF)))
    one = torch.tensor(0x3F800000, dtype=torch.int64, device=x.device)
    out = torch.where(e == 0, one, out)
    pinf = torch.tensor(0x7F800000, dtype=torch.int64, device=x.device)
    zero = torch.zeros_like(out)
    out = torch.where((unb >= 30) & (e != 255), torch.where(neg, zero, pinf), out)
    inf = (e == 255) & ((ax & 0x7FFFFF) == 0)
    nan = (e == 255) & ((ax & 0x7FFFFF) != 0)
    out = torch.where(inf, torch.where(neg, zero, pinf), out)
    return torch.where(nan, torch.tensor(0x7FFFFFFF, dtype=torch.int64, device=x.device), out)


def _mufu_rcp_core(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    s = x & 0x80000000
    ax = x & 0x7FFFFFFF
    e = ax >> 23
    r = R[ax & 0x7FFFFF]
    re = ((r >> 23) & 0xFF) - (e - 127)
    out = torch.where(re <= 0, s, torch.where(re >= 0xFF, s | 0x7F800000, s | (re << 23) | (r & 0x7FFFFF)))
    out = torch.where(e == 0, s | 0x7F800000, out)
    inf = (e == 255) & ((ax & 0x7FFFFF) == 0)
    nan = (e == 255) & ((ax & 0x7FFFFF) != 0)
    out = torch.where(inf, s, out)
    return torch.where(nan, torch.tensor(0x7FFFFFFF, dtype=torch.int64, device=x.device), out)


def mufu_rcp_ftz(x: torch.Tensor) -> torch.Tensor:
    return _mufu_rcp_core(x, _fa2_tables(str(x.device))["rcp"])


def fa2_inv_sum(s: torch.Tensor) -> torch.Tensor:
    degen = ((s & 0x7FFFFFFF) == 0) | (((s & 0x7F800000) == 0x7F800000) & ((s & 0x7FFFFF) != 0))
    return torch.where(degen, torch.tensor(0x3F800000, dtype=torch.int64, device=s.device), mufu_rcp_ftz(s))


def mufu_sqrt_ftz(x: torch.Tensor) -> torch.Tensor:
    T = _rms_tables(str(x.device))["sqrt"]
    s = x & 0x80000000
    ax = x & 0x7FFFFFFF
    e = ax >> 23
    k = e - 127
    par = k & 1
    t = T[(par << 23) | (ax & 0x7FFFFF)]
    re = ((t >> 23) & 0xFF) + torch.div(k - par, 2, rounding_mode="floor")
    out = (re << 23) | (t & 0x7FFFFF)
    nan_w = torch.tensor(0x7FFFFFFF, dtype=torch.int64, device=x.device)
    out = torch.where(s != 0, nan_w, out)
    out = torch.where(e == 0, s, out)
    inf = (e == 255) & ((ax & 0x7FFFFF) == 0)
    nan = (e == 255) & ((ax & 0x7FFFFF) != 0)
    out = torch.where(inf, torch.where(s != 0, nan_w, torch.tensor(0x7F800000, dtype=torch.int64, device=x.device)), out)
    return torch.where(nan, nan_w, out)


def _mufu_rsq(x: torch.Tensor) -> torch.Tensor:
    T = _rms_tables(str(x.device))["rsq"]
    s = x & 0x80000000
    ax = x & 0x7FFFFFFF
    e = ax >> 23
    k = e - 127
    par = k & 1
    t = T[(par << 23) | (ax & 0x7FFFFF)]
    re = ((t >> 23) & 0xFF) - torch.div(k - par, 2, rounding_mode="floor")
    out = (re << 23) | (t & 0x7FFFFF)
    nan_w = torch.tensor(0x7FFFFFFF, dtype=torch.int64, device=x.device)
    out = torch.where(s != 0, nan_w, out)
    out = torch.where(e == 0, s | 0x7F800000, out)
    inf = (e == 255) & ((ax & 0x7FFFFF) == 0)
    nan = (e == 255) & ((ax & 0x7FFFFF) != 0)
    out = torch.where(inf, torch.where(s != 0, nan_w, torch.zeros_like(out)), out)
    return torch.where(nan, nan_w, out)


def rsqrt_approx(x: torch.Tensor) -> torch.Tensor:
    xf = as_f32(x)
    scale = ~(torch.abs(xf) >= 1.1754943508222875e-38)
    xin = torch.where(scale, as_u32(xf * 16777216.0), x)
    r = _mufu_rsq(xin)
    return torch.where(scale, as_u32(as_f32(r) * 4096.0), r)


def div_full_rcp(bw: torch.Tensor) -> torch.Tensor:
    bf = as_f32(bw)
    ab = torch.abs(bf)
    big = ab > 8.507059173023462e37
    small = ab < 1.1754943508222875e-38
    scaled = torch.where(big, bf * 0.25, torch.where(small, bf * 16777216.0, bf))
    return _mufu_rcp_core(as_u32(scaled), _rms_tables(str(bw.device))["rcp"])


def div_full_scale_a(aw: torch.Tensor, bw: torch.Tensor) -> torch.Tensor:
    bf = as_f32(bw)
    af = as_f32(aw)
    ab = torch.abs(bf)
    big = ab > 8.507059173023462e37
    small = ab < 1.1754943508222875e-38
    return torch.where(big, as_u32(af * 0.25), torch.where(small, as_u32(af * 16777216.0), aw))


# -- Ampere tensor-core k16 step ----------------------------------------------------------------------------

_TC_WIDTH = DR._TC_WIDTH
_TC_FLOOR = DR._TC_FLOOR
_RESCALE = DR._RESCALE


def _bitlen(m: torch.Tensor) -> torch.Tensor:
    bl = torch.zeros_like(m)
    x = m.clone()
    for sh in (32, 16, 8, 4, 2, 1):
        big = x >= (1 << sh)
        bl = bl + big * sh
        x = torch.where(big, x >> sh, x)
    return bl + (m > 0)


def tc_dot16(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """int64 words: acc (...,), a/b (..., 16).  Returns (words, nonfinite mask)."""
    dev = acc.device
    ea = (a >> 7) & 0xFF
    eb = (b >> 7) & 0xFF
    nonfin = (ea == 0xFF).any(-1) | (eb == 0xFF).any(-1) | ((acc & 0x7F800000) == 0x7F800000)
    sa = (a & 0x7F) | ((ea != 0).to(torch.int64) << 7)
    sb = (b & 0x7F) | ((eb != 0).to(torch.int64) << 7)
    p_mag = (sa * sb) << 9
    p_exp = torch.maximum(ea, torch.ones_like(ea)) + torch.maximum(eb, torch.ones_like(eb)) - 254
    p_neg = (((a ^ b) >> 15) & 1) != 0
    a_ef = (acc >> 23) & 0xFF
    a_mant = acc & 0x7FFFFF
    acc_neg = (acc >> 31) == 1
    acc_mag = torch.where(a_ef == 0, a_mant, a_mant | 0x800000)
    acc_exp = torch.where(a_ef == 0, torch.tensor(-126, dtype=torch.int64, device=dev), a_ef - 127)
    sat = torch.zeros_like(acc, dtype=torch.bool)
    sat_word = torch.zeros_like(acc)
    for g in (0, 1):
        sl = slice(8 * g, 8 * g + 8)
        mags = torch.cat([acc_mag.unsqueeze(-1), p_mag[..., sl]], dim=-1)
        exps = torch.cat([acc_exp.unsqueeze(-1), p_exp[..., sl]], dim=-1)
        negs = torch.cat([acc_neg.unsqueeze(-1), p_neg[..., sl]], dim=-1)
        live = mags != 0
        max_exp = torch.where(live, exps, torch.tensor(-(1 << 40), dtype=torch.int64, device=dev)).max(dim=-1).values
        max_exp = torch.clamp(max_exp, min=_TC_FLOOR)
        scaled = mags << _RESCALE
        shift = torch.clamp(max_exp.unsqueeze(-1) - exps, 0, 63)
        aligned = scaled >> shift
        total = (torch.where(negs, -aligned, aligned) * live).sum(dim=-1)
        neg = total < 0
        mag = torch.abs(total)
        bl = _bitlen(mag)
        exp = max_exp + bl - _TC_WIDTH
        down = torch.clamp(bl - _TC_WIDTH, 0, 63)
        up = torch.clamp(_TC_WIDTH - bl, 0, 63)
        mag = torch.where(bl > _TC_WIDTH, mag >> down, mag << up)
        sub = torch.clamp(-126 - exp, 0, 63)
        mag = mag >> sub
        exp = torch.clamp(exp, min=-126)
        mag = mag >> _RESCALE
        zero = mag == 0
        over = (~zero) & (exp > 127)
        inf_w = (neg.to(torch.int64) << 31) | 0x7F800000
        sat_word = torch.where(over & ~sat, inf_w, sat_word)
        sat = sat | over
        acc_mag = torch.where(zero, torch.zeros_like(mag), mag)
        acc_exp = exp
        acc_neg = neg & ~zero
    exp_field = acc_exp + 127 - ((acc_mag & (1 << 23)) == 0).to(torch.int64)
    word = (acc_neg.to(torch.int64) << 31) | (exp_field << 23) | (acc_mag & 0x7FFFFF)
    word = torch.where(acc_mag == 0, torch.zeros_like(word), word)
    word = torch.where(sat, sat_word, word)
    return word, nonfin


def dot_bf16(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    D = q.shape[-1]
    shape = torch.broadcast_shapes(q.shape[:-1], k.shape[:-1])
    acc = torch.zeros(shape, dtype=torch.int64, device=q.device)
    nf = torch.zeros(shape, dtype=torch.bool, device=q.device)
    for s0 in range(0, D, 16):
        acc, f = tc_dot16(acc, q[..., s0:s0 + 16].expand(*shape, 16), k[..., s0:s0 + 16].expand(*shape, 16))
        nf |= f
    return acc, nf


# -- row evaluators (same trees as derived_rows; see that module for the gate citations) --------------------


def rmsnorm_triton_rows(x: torch.Tensor, w: torch.Tensor, eps: float):
    R, N = x.shape
    dev = x.device
    w2 = w.expand(R, N) if w.dim() == 1 else w
    xw = bf16_to_f32_words(x)
    total = torch.zeros(R, dtype=torch.int64, device=dev)
    for c0 in range(0, N, 1024):
        lanes = []
        for t in range(128):
            base = c0 + 8 * t
            if base >= N:
                lanes.append(torch.zeros(R, dtype=torch.int64, device=dev))
                continue
            xs = as_f32(xw[:, base:base + 8])
            sq = as_u32(xs * xs)  # exact products
            acc = sq[:, 1]
            acc = add_rn(sq[:, 0], acc)
            for j in range(2, 8):
                acc = add_rn(sq[:, j], acc)
            lanes.append(acc)
        warps = []
        for wi in range(4):
            v = lanes[32 * wi:32 * wi + 32]
            for msk in (16, 8, 4, 2, 1):
                v = [add_rn(v[i], v[i ^ msk]) for i in range(32)]
            warps.append(v[0])
        chunk = add_rn(add_rn(warps[0], warps[2]), add_rn(warps[1], warps[3]))
        total = add_rn(total, chunk)
    rcp_n = div_full_rcp(torch.full((R,), P.bits_of(np.float32(float(N))), dtype=torch.int64, device=dev))
    eps_w = torch.full((R,), P.bits_of(np.float32(eps)), dtype=torch.int64, device=dev)
    mean_eps = fma_rn(rcp_n, total, eps_w)
    rms = mufu_sqrt_ftz(mean_eps)
    one = torch.full((R,), 0x3F800000, dtype=torch.int64, device=dev)
    inv = mul_rn(div_full_scale_a(one, rms), div_full_rcp(rms))
    y = mul_rn(mul_rn(inv.unsqueeze(1), xw), bf16_to_f32_words(w2))
    return {"out": f2fp_bf16(y), "sum": total, "mean_eps": mean_eps, "rms": rms, "inv": inv}


def _cub_block_reduce_sum(partials: torch.Tensor) -> torch.Tensor:
    """partials (R, nthreads) f32 words."""
    n = partials.shape[1]
    warps = []
    for w in range((n + 31) // 32):
        lanes = {i: partials[:, 32 * w + i] for i in range(min(32, n - 32 * w))}
        for off in (1, 2, 4, 8, 16):
            lanes = {i: (add_rn(v, lanes[i + off]) if i + off in lanes else v)
                     for i, v in lanes.items() if i % (2 * off) == 0}
        warps.append(lanes[0])
    var = warps[0]
    for wv in warps[1:]:
        var = add_rn(var, wv)
    return var


def rmsnorm_fused_v2_rows(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor, eps: float):
    R, N = x.shape
    w2 = w.expand(R, N) if w.dim() == 1 else w
    rp = f2fp_bf16(as_u32(as_f32(bf16_to_f32_words(x)) + as_f32(bf16_to_f32_words(res))))
    z = bf16_to_f32_words(rp)
    zf = as_f32(z)
    nthreads = min(N, b1.FUSED_RMS_BLOCK)
    acc = torch.zeros((R, nthreads), dtype=torch.int64, device=x.device)
    for base in range(0, N, nthreads):
        cnt = min(nthreads, N - base)
        sq = as_u32(zf[:, base:base + cnt] * zf[:, base:base + cnt])  # exact for bf16 values
        acc = torch.cat([add_rn(acc[:, :cnt], sq), acc[:, cnt:]], dim=1) if cnt < nthreads else add_rn(acc, sq)
    var = _cub_block_reduce_sum(acc)
    # tensor / tensor: true `div.rn.f32`.  A python-float divisor would let torch lower the division to a
    # multiply-by-reciprocal on CUDA (1-ulp off true division on ~6% of the B0 variance words).
    n_t = torch.full((), float(N), dtype=torch.float32, device=x.device)
    mean = as_u32(as_f32(var) / n_t)
    s = rsqrt_approx(as_u32(as_f32(mean) + np.float32(eps).item()))
    y = mul_rn(mul_rn(z, s.unsqueeze(1)), bf16_to_f32_words(w2))
    return {"normed": f2fp_bf16(y), "resid": rp, "var": var, "s": s}


@lru_cache(maxsize=None)
def _silu_mid_table(device_str: str) -> torch.Tensor:
    return words_from_numpy(DR.silu_mid_table(), torch.device(device_str))


def silu_mul_rows(gu: torch.Tensor):
    R, GU = gu.shape
    half = GU // 2
    g, u = gu[:, :half], gu[:, half:]
    mid = _silu_mid_table(str(gu.device))[g]
    prod = mul_rn(bf16_to_f32_words(mid), bf16_to_f32_words(u))
    return {"mid": mid, "out": bf16_rn_torch(prod)}


BN = b1.BN  # `AttnBlock_v1`'s block (64).  [E7 D54] `attn_block` reads the block size from the K block; `attention_head_states` takes ``BN``


def attn_block(q, kb, vb, m_prev, l_prev, o_prev, *, D: int, NVIS: int, FIRST: bool):
    BN = kb.shape[1]  # [E7 D54] 64 (`AttnBlock_v1`) or `b1.fa2_kblock_n(D)` (`AttnBlock_v3`)
    R = q.shape[0]
    dev = q.device
    nf = torch.zeros(R, dtype=torch.bool, device=dev)
    s, f = dot_bf16(q.unsqueeze(1), kb[:, :NVIS, :])
    nf |= f.any(-1)
    bmax = s[:, 0]
    for c in range(1, NVIS):
        bmax = max_fmnmx(bmax, s[:, c])
    m_new = bmax if FIRST else max_fmnmx(m_prev, bmax)
    m_use = guard_neg_inf_zero(m_new)
    scale = torch.full((R,), b1._scale_log2_bits(D), dtype=torch.int64, device=dev)
    if not FIRST:
        scores_scale = mufu_ex2_ftz(mul_ftz(sub_ftz(m_prev, m_use), scale))
    max_scaled = mul_ftz(m_use, scale)
    p = mufu_ex2_ftz(fma_sub_ftz(s, scale.unsqueeze(1), max_scaled.unsqueeze(1)))
    lanes = []
    for ln in range(4):
        cols = [c for c in range(NVIS) if b1._lane(c) == ln]
        zero = torch.zeros(R, dtype=torch.int64, device=dev)
        first_p = p[:, cols[0]] if cols else zero
        acc = first_p if FIRST else fma_ftz(scores_scale, l_prev[:, ln], first_p)
        for c in cols[1:]:
            acc = add_ftz(acc, p[:, c])
        lanes.append(acc)
    l_new = torch.stack(lanes, dim=1)
    pb = f2fp_bf16(p)
    if NVIS < BN:
        pb = torch.cat([pb, torch.zeros((R, BN - NVIS), dtype=torch.int64, device=dev)], dim=1)
    o = torch.zeros((R, D), dtype=torch.int64, device=dev) if FIRST else mul_ftz(o_prev, scores_scale.unsqueeze(1))
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break
        prow = pb[:, k0:k0 + 16]
        vt = vb[:, k0:k0 + 16, :].transpose(1, 2)
        o, f = tc_dot16(o, prow.unsqueeze(1).expand(R, D, 16), vt)
        nf |= f.any(-1)
    return m_new, l_new, o, nf


def attention_head_states(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, BN: int | None = None):
    R, T, D = k.shape
    if BN is None:
        BN = b1.BN  # the v1 chain; `b1.fa2_kblock_n(D)` for AttentionHead_v3 [E7 D54]
    dev = q.device
    nb = (T + BN - 1) // BN
    ms, ls, os = [], [], []
    nf = torch.zeros(R, dtype=torch.bool, device=dev)
    m = l = o = None
    order = list(range(nb - 1, -1, -1))
    for b in order:
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[:, lo:hi], v[:, lo:hi]
        if nvis < BN:
            pad = torch.zeros((R, BN - nvis, D), dtype=torch.int64, device=dev)
            kb = torch.cat([kb, pad], dim=1)
            vb = torch.cat([vb, pad], dim=1)
        first = b == nb - 1
        m, l, o, f = attn_block(q, kb, vb,
                                torch.zeros(R, dtype=torch.int64, device=dev) if first else m,
                                torch.zeros((R, 4), dtype=torch.int64, device=dev) if first else l,
                                torch.zeros((R, D), dtype=torch.int64, device=dev) if first else o,
                                D=D, NVIS=nvis, FIRST=first)
        nf |= f
        ms.append(m)
        ls.append(l)
        os.append(o)
    total = add_ftz(add_ftz(l[:, 0], l[:, 2]), add_ftz(l[:, 1], l[:, 3]))
    inv = fa2_inv_sum(total)
    y = mul_ftz(o, inv.unsqueeze(1))
    return {"m": torch.stack(ms, 1), "l": torch.stack(ls, 1), "o": torch.stack(os, 1), "blocks": order,
            "out": f2fp_bf16(y), "total": total, "inv": inv, "nonfinite": nf}

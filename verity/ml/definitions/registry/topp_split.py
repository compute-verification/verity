"""Exact-fp32 model of vLLM d9105ea80 `topk_topp_triton._apply_topp_split` — the p-only split top-p pipeline
(`_topp_sb_stats_kernel` -> `_topp_sb_step_kernel` x 5 -> `_topp_sb_mask_kernel`; batch <= 64, no top-k) for ONE logits row,
as Triton 3.7.1 + ptxas lower it on sm_89 (num_warps=8, F=8 pivots, 5 rounds, BLOCK 2048 / 8192).

Every fp32 op is one IEEE RN-even step (numpy float32).  `tl.exp` = MUFU.EX2(x * log2e) with ptxas' non-ftz `t < -126` path
(measured table, `verity.ml.definitions.numerics.fa2_model`); `/` = div.full.f32 = MUFU.RCP(range-scaled b) * a (`rms_triton_model`);
`tl.log` = libdevice `__nv_logf` (`sampling.nv_logf`).  Reduction trees are explicit (`Cfg`, defaults = the fitted lowering):

  * stats `tl.sum` over the [8192] block: sizePerThread spt = `spt_for(V, S)` (4 / 2 / 1 -- Triton's coalescing pass decides it
    from the alignment of the row stride (V) and the slice start; it is a function of the constexpr VOCAB_SIZE and S) ->
    per-thread sequential over its (rep, j) registers, xor-butterfly 16..1, cross-warp tree;
  * `Z = tl.sum(exp_s * w)` over the [S] slice partials: the lane product is FMA-CONTRACTED into the first (xor S/2) butterfly
    stage (Triton emits mul/add without .rn; ptxas fuses), so Z is a PER-WARP-HALF value: lanes 0..S/2-1 and S/2..S-1 can hold
    values 1 ulp apart.  Each element's pr = exp(x - M) / Z uses the Z of its own lane in the 1-D load layout
    (lane = (j // spt) % 32 of the block offset j) -- in the step AND the mask kernel (the mask kernel's `x > pivot_logit` and
    `exp(x - M) / Z == dup` are per-lane too);
  * the step kernel compares `pr2 > pivs[:, None]` in the 2-D layout blocked<{[1,1], threadsPerWarp [1,32], warpsPerCTA [1,8]}>:
    column j is held by lane j % 32, and every thread holds the ladder computed from ITS OWN half's lo/hi (the [F] ladder is a
    slice of that 2-D layout: fully replicated, no shared-memory broadcast), so column j meets the ladder of half (j % 32) >= 16;
    the [F, BLOCK] `tl.sum(axis=1)`: per-thread sequential over stride-256 columns, butterfly, cross-warp tree; the [S, F]
    axis-0 sums in combine: rows in warps of 4.

Validated bit-exact (stats, the partials of every executed round, masked output) on 63 captured rows (55 real Llama-3.2-1B T=0.8
rows + 8 edge rows at V=128256) and on 40/40/40/24/24/40/40/24/40/16/40 random bf16-valued rows at V = 3000, 1025, 1040, 1088,
2112, 1024, 4096, 32000, 49152, 151936, 128256.  The registry primitive
`TopPMaskWordx{V}s{S}` (sampling.py) calls `topp_keep_row`.

Public: `topp_split_row(x_in_f32, p, S, F, R, cfg) -> (x_out_f32, stats[S,4], parts[R,S,F,3], trace)`; `topp_keep_row(x_in_f32, p, S)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from verity.ml.definitions.registry.topp_geometry import spt_for, z_group

SPLIT_FANOUT = 8          # `_SPLIT_FANOUT` (F)
SPLIT_ROUNDS = 5          # `_SPLIT_ROUNDS` (R)
STATS_BLOCK = 8192
STEP_BLOCK = 2048
MASK_BLOCK = 8192


@lru_cache(maxsize=None)
def _fa():
    from verity.ml.definitions.numerics import fa2_model as FA
    from verity.ml.definitions.numerics import fa2_relation as FAR
    return FA, FAR.tables()


@lru_cache(maxsize=None)
def _rm():
    from verity.ml.definitions.numerics import rms_relation as RR
    from verity.ml.definitions.numerics import rms_triton_model as RM
    return RM, RR.tables("cuda")

f32 = np.float32
LOG2E = f32(1.4426950408889634)
INF = f32(np.inf)
NEG_INF = f32(-np.inf)


@dataclass
class Cfg:
    spt_stats: int = 4          # sizePerThread of the [8192] load in the stats kernel (= spt_for(V, S); see Cfg.for_vocab)
    spt_step: int = 1           # sizePerThread of the [F, 2048] broadcast tensor the step kernel reduces (fitted: 1)
    num_warps: int = 8
    fma_exp_sum: bool = True    # `exp_sum * rescale + tl.sum(e)` contracted to fma (only observable when a slice spans > 1 BLOCK, i.e.
                                #   V / S > 8192: S <= 8 at V = 128256; fitted on the GPU fixture, 7/7 multi-block rows; the single-block
                                #   case is unchanged: exp_sum = 0, rescale = 0 -> fma(0, 0, se) = se)
    fma_z: bool = True          # `tl.sum(exp_s * w)`: the lane product fused into the first butterfly stage (fitted: True)
    fma_ok: bool = False        # `mass - gmin * nmin` contracted to fma(-gmin, nmin, mass)
    exp_impl: str = "ex2"       # "ex2" (mul log2e + MUFU.EX2) | "np" (numpy expf, diagnostic only)
    div_impl: str = "full"      # "full" (div.full.f32 model) | "np"
    log_impl: str = "nv"        # "nv" (__nv_logf) | "np"
    parts_tree: str = "w4x8"    # [S, F] axis-0 tree: "w4x8" (rows in warps of 4: (r0+r2)+(r1+r3); warps butterfly 4,2,1) | "seq" | "bfly32"
    z_tree: str = "bfly32"      # [S] sum: "bfly32" | "seq"
    step_layout: str = "rows-replicated"   # [F, BLOCK] axis-1 sum: "rows-replicated" (warps along BLOCK; = 1-D tree per row) | "warp-per-row"
    spt_x: int = 4              # sizePerThread of the 1-D [BLOCK] x / pr tensors in the step and mask kernels (lane = (j // spt_x) % 32)
    pivs_half: object = "lane2d"  # "lane2d" (the kernel): the step kernel compares `pr2 > pivs[:, None]` in the 2-D layout
                                #   blocked<{sizePerThread=[1,1], threadsPerWarp=[1,32], warpsPerCTA=[1,8]}> where column j sits in
                                #   lane j % 32 and every thread holds its own warp-half's ladder (lo/hi from that half's Z);
                                #   0 | 1 | "lane" (1-D lane-half) are diagnostic alternatives.

    @classmethod
    def for_vocab(cls, V: int, S: int) -> "Cfg":
        spt = spt_for(V, S)
        return cls(spt_stats=spt, spt_x=spt)


def tl_exp(x: np.ndarray, cfg: Cfg) -> np.ndarray:
    x = np.asarray(x, dtype=f32)
    if cfg.exp_impl == "np":
        return np.exp(x).astype(f32)
    t = (x * LOG2E).astype(f32)
    # ex2.approx.f32 (non-ftz) as ptxas expands it: t < -126 -> MUFU.EX2(t * 0.5) squared (denormal-range outputs)
    small = t < f32(-126.0)
    tt = np.where(small, (t * f32(0.5)).astype(f32), t).astype(f32)
    FA, tab = _fa()
    e = FA.mufu_ex2(np.ascontiguousarray(tt.ravel()), tab).reshape(t.shape)
    return np.where(small, (e * e).astype(f32), e).astype(f32)


def tl_div(a: np.ndarray, b: np.ndarray, cfg: Cfg) -> np.ndarray:
    a = np.asarray(a, dtype=f32)
    if cfg.div_impl == "np":
        return (a / np.asarray(b, dtype=f32)).astype(f32)
    shape = a.shape
    RM, tab = _rm()
    return RM.div_full(np.ascontiguousarray(a.ravel()), np.asarray(b, dtype=f32), tab).reshape(shape)


def tl_log(x: np.ndarray, cfg: Cfg) -> np.ndarray:
    x = np.asarray(x, dtype=f32)
    if cfg.log_impl == "np":
        return np.log(x).astype(f32)
    from verity.ml.definitions.registry.sampling import nv_logf as _nv_logf_word   # (lazy: sampling.py imports this module)
    w = x.ravel().view(np.uint32)
    out = np.array([_nv_logf_word(int(v)) for v in w], dtype=np.uint32).view(f32)
    return out.reshape(x.shape)


def fma32(a, b, c) -> np.ndarray:
    """RN-even fp32 fma on arrays (exact product in double, then correctly rounded a*b+c)."""
    shape = np.broadcast(np.asarray(a), np.asarray(b), np.asarray(c)).shape
    a = np.atleast_1d(np.asarray(a, dtype=f32)).astype(np.float64); b = np.atleast_1d(np.asarray(b, dtype=f32)).astype(np.float64)
    c = np.atleast_1d(np.asarray(c, dtype=f32)).astype(np.float64)
    p = a * b                       # exact (24+24 bits < 53)
    s = p + c                       # double RN; error term via TwoSum
    bb = s - p
    e = (p - (s - bb)) + (c - bb)
    r = s.astype(f32)               # first rounding (double -> single)
    # double rounding fix: if s is exactly a single-precision midpoint and e != 0, e decides
    rd = r.astype(np.float64)
    lo = np.nextafter(rd, -np.inf).astype(f32).astype(np.float64); hi = np.nextafter(rd, np.inf).astype(f32).astype(np.float64)
    # s lies strictly between two adjacent singles iff rd != s; the midpoint case is (lo+rd)/2 == s or (rd+hi)/2 == s
    mid_lo = (lo + rd) * 0.5; mid_hi = (rd + hi) * 0.5
    fix_up = (s == mid_hi) & (e > 0)      # exact value above the midpoint -> round up
    fix_dn = (s == mid_lo) & (e < 0)
    out = r.copy()
    out[fix_up] = hi[fix_up].astype(f32); out[fix_dn] = lo[fix_dn].astype(f32)
    return out.reshape(shape)


# ------------------------------------------------------------------------------------------------ reduction trees


def _bfly(vals: np.ndarray, axis_len: int, masks) -> np.ndarray:
    """xor-butterfly along the LAST axis of `vals` (length axis_len): v[i] = v[i] + v[i ^ m] for m in masks; returns [..., 0]."""
    cur = vals.astype(f32)
    idx = np.arange(axis_len)
    for m in masks:
        cur = (cur + cur[..., idx ^ m]).astype(f32)
    return cur[..., 0]


def tree_sum_1d(vals: np.ndarray, spt: int, num_warps: int) -> np.float32:
    """`tl.sum` of a 1-D [BLOCK] fp32 tensor, blocked layout sizePerThread=spt, threadsPerWarp=32, warpsPerCTA=num_warps:
    per-thread sequential over its registers in (rep, j) order, warp xor-butterfly 16,8,4,2,1, cross-warp xor-butterfly
    (num_warps/2 .. 1)."""
    BLOCK = vals.shape[-1]
    per_thread = BLOCK // (32 * num_warps)
    reps = per_thread // spt
    v = vals.reshape(vals.shape[:-1] + (reps, num_warps, 32, spt)).astype(f32)  # [..., rep, w, t, j]
    acc = v[..., 0, :, :, 0].copy()
    for rep in range(reps):
        for j in range(spt):
            if rep == 0 and j == 0:
                continue
            acc = (acc + v[..., rep, :, :, j]).astype(f32)
    warp = _bfly(acc, 32, (16, 8, 4, 2, 1))          # [..., w]
    masks = []
    m = num_warps // 2
    while m >= 1:
        masks.append(m); m //= 2
    return _bfly(warp, num_warps, masks) if num_warps > 1 else warp[..., 0]


def tree_sum_rows(vals: np.ndarray, spt: int, num_warps: int, layout: str) -> np.ndarray:
    """`tl.sum(t, axis=1)` of a [F, BLOCK] fp32 tensor.  "rows-replicated": every thread holds all F rows x its BLOCK/(32*num_warps)
    columns (= the 1-D tree per row).  "warp-per-row": warp f owns row f (32 threads x BLOCK/32 columns, no cross-warp stage)."""
    if layout == "rows-replicated":
        return tree_sum_1d(vals, spt, num_warps)
    F, BLOCK = vals.shape
    per_thread = BLOCK // 32
    reps = per_thread // spt
    v = vals.reshape(F, reps, 32, spt).astype(f32)   # [f, rep, t, j]
    acc = v[:, 0, :, 0].copy()
    for rep in range(reps):
        for j in range(spt):
            if rep == 0 and j == 0:
                continue
            acc = (acc + v[:, rep, :, j]).astype(f32)
    return _bfly(acc, 32, (16, 8, 4, 2, 1))


def tree_sum_S(vals: np.ndarray, S: int, cfg: Cfg, prod_terms=None) -> np.ndarray:
    """`tl.sum(exp_s * w)` over a [S] tensor (S <= 32: one warp, sizePerThread 1, replicated across warps): butterfly over S lanes
    (masks S/2 .. 1).  With `prod_terms=(a, b)` and cfg.fma_z the lane product is fused into each butterfly add (fma(a_i, b_i, shfl))."""
    if cfg.z_tree == "seq":
        acc = f32(0)
        for i in range(S):
            acc = f32(acc + vals[i])
        return acc
    masks = []
    m = S // 2
    while m >= 1:
        masks.append(m); m //= 2
    if prod_terms is not None and cfg.fma_z:
        a, b = prod_terms
        idx = np.arange(S)
        cur = vals.astype(f32)
        first = True
        for mm in masks:
            if first:
                cur = fma32(a, b, cur[idx ^ mm]); first = False
            else:
                cur = (cur + cur[idx ^ mm]).astype(f32)
        return cur          # per-lane results: lanes [0, S/2) and [S/2, S) can differ (the xor-S/2 fma stage is asymmetric)
    return _bfly(vals.astype(f32), S, masks)


def tree_sum_parts_axis0(m: np.ndarray, S: int, F: int, cfg: Cfg) -> np.ndarray:
    """`tl.sum(mass_s, axis=0)` of the [S, F] partials (blocked layout sizePerThread [1,1], threadsPerWarp [4,8], warpsPerCTA [S/4,1])."""
    m = m.astype(f32)
    if cfg.parts_tree == "seq":
        acc = m[0].copy()
        for i in range(1, S):
            acc = (acc + m[i]).astype(f32)
        return acc
    if cfg.parts_tree == "bfly32":
        masks = []
        k = S // 2
        while k >= 1:
            masks.append(k); k //= 2
        return _bfly(m.T, S, masks)
    # w4x8: rows grouped in warps of 4 (rows w*4..w*4+3), intra-warp butterfly over the 4 rows (xor 2, 1), inter-warp butterfly
    rows_per_warp = min(4, S)
    nw = S // rows_per_warp
    v = m.reshape(nw, rows_per_warp, F).transpose(2, 0, 1)   # [F, w, r]
    masks_r = [2, 1] if rows_per_warp == 4 else ([1] if rows_per_warp == 2 else [])
    wv = _bfly(v, rows_per_warp, masks_r) if masks_r else v[..., 0]  # [F, w]
    masks_w = []
    k = nw // 2
    while k >= 1:
        masks_w.append(k); k //= 2
    return _bfly(wv, nw, masks_w) if masks_w else wv[..., 0]


# ------------------------------------------------------------------------------------------------ the kernels


def stats_kernel(x: np.ndarray, p: float, S: int, BLOCK: int, cfg: Cfg) -> np.ndarray:
    V = x.shape[0]
    out = np.zeros((S, 4), f32)
    if f32(p) >= f32(1.0):
        return out  # kernel returns without storing (buffer content undefined); caller compares only when p < 1
    slice_len = (V + S - 1) // S
    for s in range(S):
        start = s * slice_len; end = min(start + slice_len, V)
        m = NEG_INF; exp_sum = f32(0.0); mn = INF; cnt = f32(0.0)
        for i in range(start, end, BLOCK):
            xb = np.full(BLOCK, NEG_INF, f32)
            n = min(BLOCK, end - i)
            xb[:n] = x[i:i + n]
            finite = xb > NEG_INF
            nm = f32(max(m, xb.max()))
            with np.errstate(invalid="ignore"):
                rescale = f32(0.0) if m == NEG_INF else tl_exp(f32(m - nm), cfg)
                e = np.where(finite, tl_exp((xb - nm).astype(f32), cfg), f32(0.0)).astype(f32)
            se = tree_sum_1d(e, cfg.spt_stats, cfg.num_warps)
            exp_sum = fma32(exp_sum, rescale, se)[()] if cfg.fma_exp_sum else f32(f32(exp_sum * rescale) + se)
            m = nm
            mn = f32(min(mn, np.where(finite, xb, INF).min()))
            cnt = f32(cnt + tree_sum_1d(finite.astype(f32), cfg.spt_stats, cfg.num_warps))
        out[s] = (m, exp_sum, mn, cnt)
    return out


def row_stats(stats: np.ndarray, S: int, cfg: Cfg):
    m_s, exp_s, mn_s, c_s = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    M = f32(m_s.max())
    with np.errstate(invalid="ignore"):
        w = np.where(m_s == NEG_INF, f32(0.0), tl_exp((m_s - M).astype(f32), cfg)).astype(f32)
    prod = (exp_s * w).astype(f32)
    Z = tree_sum_S(prod, S, cfg, prod_terms=(exp_s, w))
    if np.ndim(Z) == 0:
        Zh = np.array([Z, Z], f32)
    else:
        Zh = np.array([Z[0], Z[S // 2]], f32)      # (Z seen by lanes 0..S/2-1, Z seen by lanes S/2..S-1)
    return M, Zh, f32(mn_s.min()), f32(tree_sum_S(c_s, S, cfg))


def lane_half(j: np.ndarray, cfg: Cfg, S: int) -> np.ndarray:
    """Z-group (0/1, `z_group`) of the thread holding element offset `j` of a 1-D [BLOCK] tensor (blocked layout, sizePerThread=cfg.spt_x,
    32 lanes: lane = (j // spt_x) % 32)."""
    return z_group((j // cfg.spt_x) % 32, S)


def ladder(lo: np.float32, hi: np.float32, F: int, cfg: Cfg) -> np.ndarray:
    fidx = np.arange(F, dtype=f32)
    q = tl_div(np.array([hi], f32), lo, cfg)[0]
    l = tl_log(np.array([q], f32), cfg)[0]
    t = (l * (fidx + f32(1.0)).astype(f32)).astype(f32)
    u = tl_div(t, f32(F + 1.0), cfg)
    return (lo * tl_exp(u, cfg)).astype(f32)


def combine(part: np.ndarray, Z, p, lo, hi, best, best_mass, best_minl, best_nmin, S, F, is_last, cfg: Cfg):
    """`_topp_sb_combine` on one round's [S, F, 3] partials.  Returns the 14-tuple in kernel order."""
    mass_s, minl_s, nmin_s = part[:, :, 0], part[:, :, 1], part[:, :, 2]
    mass = tree_sum_parts_axis0(mass_s, S, F, cfg)                                 # [F]
    gmin = minl_s.min(axis=0).astype(f32)
    nmin = np.where(np.abs((minl_s - gmin[None, :]).astype(f32)) < f32(1e-9), nmin_s, f32(0.0)).astype(f32).sum(axis=0).astype(f32)  # exact ints
    pivs = ladder(lo, hi, F, cfg)
    p = f32(p)
    if cfg.fma_ok:
        lhs = fma32(-gmin, nmin, mass)
    else:
        lhs = (mass - (gmin * nmin).astype(f32)).astype(f32)
    ok = (mass >= p) & (lhs < p)
    fidx = np.arange(F)
    ok_idx = int(np.max(np.where(ok, fidx, -1)))
    ge = mass >= p
    ge_idx = int(np.max(np.where(ge, fidx, -1)))
    best_sel = f32(-1.0)
    if ge_idx >= 0:
        bp = pivs[ge_idx]
        if bp > best:
            best, best_mass, best_minl, best_nmin = bp, mass[ge_idx], gmin[ge_idx], nmin[ge_idx]
            best_sel = f32(ge_idx)
    done = nomask = pivot = sel_mass = f32(0.0); dup = numdup = numkeep = f32(1.0); sel_fidx = f32(-1.0)
    if ok_idx >= 0:
        pivot, sel_mass, dup, numdup = pivs[ok_idx], mass[ok_idx], gmin[ok_idx], nmin[ok_idx]
        sel_fidx = f32(ok_idx); done = f32(1.0)
    else:
        new_lo = f32(max(lo, np.where(ge, pivs, NEG_INF).max()))
        new_hi = f32(min(hi, np.where(~ge, pivs, INF).min()))
        if is_last or not (new_hi > f32(new_lo * f32(1.0 + 1e-7))):
            if best > f32(0.0):
                pivot, sel_mass, dup, numdup = best, best_mass, best_minl, best_nmin
            else:
                nomask = f32(1.0)
            done = f32(1.0)
        lo, hi = new_lo, new_hi
    if done != 0.0 and nomask == 0.0:
        nomask = f32(1.0) if f32(pivot * Z) >= f32(1.0) else f32(0.0)
        q = tl_div(np.array([f32(max(f32(sel_mass - p), f32(0.0)))], f32), dup, cfg)[0]
        with np.errstate(invalid="ignore"):
            qi = np.uint32(0) if not np.isfinite(q) or q <= 0 else (np.uint32(0xFFFFFFFF) if q >= 4294967296.0 else np.uint32(np.trunc(q)))
        numkeep = f32(numdup - f32(qi))
        numkeep = f32(min(max(numkeep, f32(1.0)), numdup))
    return (done, nomask, f32(pivot), f32(dup), f32(numdup), f32(numkeep), f32(lo), f32(hi), f32(best), f32(best_mass), f32(best_minl), f32(best_nmin), sel_fidx, best_sel)


def _row_prelude(stats, S, cfg, half: int = 0):
    M, Zh, mn, num_finite = row_stats(stats, S, cfg)
    Z = f32(Zh[half])
    if Z <= 0.0 or num_finite < 1.0:
        return None
    max_prob = tl_div(np.array([f32(1.0)], f32), Z, cfg)[0]
    min_prob = f32(max(tl_div(tl_exp(np.array([f32(mn - M)], f32), cfg), Z, cfg)[0], f32(1e-30)))
    if max_prob <= f32(min_prob * f32(1.0 + 1e-6)):
        return None
    return M, Z, mn, num_finite, max_prob, min_prob


def _z_for_block(xb_len: int, Zh, cfg: Cfg, S: int) -> np.ndarray:
    return np.asarray(Zh, f32)[lane_half(np.arange(xb_len), cfg, S)]


def step_kernel(x: np.ndarray, stats: np.ndarray, parts: np.ndarray, p: float, ROUND: int, S: int, F: int, R: int, BLOCK: int, cfg: Cfg) -> None:
    """Writes parts[ROUND] (all S slices) in place, replaying the combine chain for rounds < ROUND exactly as the kernel does."""
    V = x.shape[0]
    pres = [_row_prelude(stats, S, cfg, h) for h in (0, 1)]
    if pres[0] is None or pres[1] is None:
        return   # (if only one half returns early the kernel diverges; the model follows the early return)
    M = pres[0][0]
    Zh = np.array([pres[0][1], pres[1][1]], f32)
    pivs_h = []
    for h in (0, 1):
        _, Z, mn, num_finite, max_prob, min_prob = pres[h]
        lo, hi = min_prob, max_prob
        best, best_mass, best_minl, best_nmin = f32(-1.0), f32(0.0), f32(1.0), f32(0.0)
        done = f32(0.0)
        for j in range(ROUND):
            if done == 0.0:
                (done, nomask, pivot, dup, numdup, numkeep, lo, hi, best, best_mass, best_minl, best_nmin, sel_fidx, best_sel) = \
                    combine(parts[j], Z, p, lo, hi, best, best_mass, best_minl, best_nmin, S, F, False, cfg)
        pivs_h.append(None if done != 0.0 else ladder(lo, hi, F, cfg))
    if pivs_h[0] is None or pivs_h[1] is None:
        return
    Zb = _z_for_block(BLOCK, Zh, cfg, S)
    if cfg.pivs_half == "lane2d":      # column j is held by lane j % 32 of the 2-D compare layout; that thread's own Z-group's ladder
        hb2 = z_group(np.arange(BLOCK) % 32, S)
        pivs_b = np.where(hb2[None, :] == 0, pivs_h[0][:, None], pivs_h[1][:, None]).astype(f32)  # [F, BLOCK]
    elif cfg.pivs_half == "lane":      # diagnostic: the 1-D load layout's Z-group
        hb = lane_half(np.arange(BLOCK), cfg, S)
        pivs_b = np.where(hb[None, :] == 0, pivs_h[0][:, None], pivs_h[1][:, None]).astype(f32)   # [F, BLOCK]
    else:
        pivs_b = np.broadcast_to(pivs_h[int(cfg.pivs_half)][:, None], (F, BLOCK))
    slice_len = (V + S - 1) // S
    for s in range(S):
        start = s * slice_len; end = min(start + slice_len, V)
        mass = np.zeros(F, f32); minl = np.ones(F, f32); nmin = np.zeros(F, f32)
        for i in range(start, end, BLOCK):
            xb = np.full(BLOCK, NEG_INF, f32)
            n = min(BLOCK, end - i)
            xb[:n] = x[i:i + n]
            with np.errstate(invalid="ignore"):
                pr = tl_div(tl_exp((xb - M).astype(f32), cfg), Zb, cfg)                   # [BLOCK], each lane-half with its own Z
            pr2 = np.broadcast_to(pr[None, :], (F, BLOCK))
            above = pr2 > pivs_b
            contrib = np.where(above, pr2, f32(0.0)).astype(f32)                          # [F, BLOCK]
            mass = (mass + tree_sum_rows(contrib, cfg.spt_step, cfg.num_warps, cfg.step_layout)).astype(f32)
            cand = np.where(above, pr2, INF).astype(f32)
            tmin = cand.min(axis=1).astype(f32)
            with np.errstate(invalid="ignore"):
                teq = above & (np.abs((pr2 - tmin[:, None]).astype(f32)) < f32(1e-9))
            tcnt = teq.sum(axis=1).astype(f32)
            is_new = tmin < minl
            with np.errstate(invalid="ignore"):
                is_same = np.abs((tmin - minl).astype(f32)) < f32(1e-9)
            nmin = np.where(is_new, tcnt, (nmin + np.where(is_same, tcnt, f32(0.0))).astype(f32)).astype(f32)
            minl = np.minimum(minl, tmin).astype(f32)
        parts[ROUND, s, :, 0] = mass; parts[ROUND, s, :, 1] = minl; parts[ROUND, s, :, 2] = nmin


def _mask_chain(stats, parts, p, S, F, R, cfg, half):
    """The mask kernel's scalar chain for one warp-half: returns None (early return) or the dict of scalars."""
    pre = _row_prelude(stats, S, cfg, half)
    if pre is None:
        return {"early": "uniform-or-empty"}
    M, Z, mn, num_finite, max_prob, min_prob = pre
    lo, hi = min_prob, max_prob
    best, best_mass, best_minl, best_nmin = f32(-1.0), f32(0.0), f32(1.0), f32(0.0)
    done = nomask = pivot = f32(0.0); dup = numdup = numkeep = f32(1.0); fin = best_fin = f32(-1.0)
    for j in range(R):
        if done == 0.0:
            (done, nomask, pivot, dup, numdup, numkeep, lo, hi, best, best_mass, best_minl, best_nmin, sel_fidx, best_sel) = \
                combine(parts[j], Z, p, lo, hi, best, best_mass, best_minl, best_nmin, S, F, j == R - 1, cfg)
            if sel_fidx >= 0.0:
                fin = f32(j * F + sel_fidx)
            if best_sel >= 0.0:
                best_fin = f32(j * F + best_sel)
    d = {"M": M, "Z": Z, "done": done, "nomask": nomask, "pivot": pivot, "dup": dup, "numdup": numdup, "numkeep": numkeep,
         "fin": fin, "best_fin": best_fin, "early": None}
    if nomask != 0.0:
        d["early"] = "nomask"; return d
    if fin < 0.0:
        fin = best_fin
    d["fin"] = fin
    logZ = tl_log(np.array([Z], f32), cfg)[0]
    d["pivot_logit"] = f32(f32(tl_log(np.array([pivot], f32), cfg)[0] + logZ) + M)
    d["ties"] = bool(numkeep < numdup)
    return d


def mask_kernel(x: np.ndarray, stats: np.ndarray, parts: np.ndarray, p: float, S: int, F: int, R: int, BLOCK: int, cfg: Cfg, trace: dict) -> np.ndarray:
    V = x.shape[0]
    out = x.copy()
    ch = [_mask_chain(stats, parts, p, S, F, R, cfg, h) for h in (0, 1)]
    trace.update({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in ch[0].items()})
    trace["z_split"] = bool(ch[0].get("Z", 0) != ch[1].get("Z", 0))
    trace["half_divergence"] = (ch[0]["early"] != ch[1]["early"]) or (ch[0]["early"] is None and ch[0]["ties"] != ch[1]["ties"])
    keep_all = np.ones(V, dtype=bool)
    # every lane follows its own warp half's early return (`sampling.topp_keep`); a stopped lane is kept and not counted as a dup
    stopped = [ch[h]["early"] is not None for h in (0, 1)]
    trace["stopped"] = stopped
    if all(stopped):
        trace["keep"] = keep_all
        return out
    h0 = stopped.index(False)                               # half 0 when it goes on; its scalars stand in for a stopped half's
    val = lambda key: np.array([ch[h0 if stopped[h] else h][key] for h in (0, 1)], f32)   # noqa: E731
    M = ch[h0]["M"]
    Zb = _z_for_block(BLOCK, val("Z"), cfg, S)
    hb = lane_half(np.arange(BLOCK), cfg, S)
    stop_b = np.array(stopped)[hb]
    pivot_logit_b = val("pivot_logit")[hb]
    dup_b = val("dup")[hb]
    numkeep_b = val("numkeep")[hb]
    ties = ch[h0]["ties"]
    trace["ties"] = ties
    slice_len = (V + S - 1) // S
    for s in range(S):
        num_preceding_h = np.zeros(2, f32)
        if ties:
            for h in (0, 1):
                if stopped[h]:
                    continue
                fin, dup = ch[h]["fin"], ch[h]["dup"]
                fin_round = int(np.trunc(f32(fin / f32(F))))
                fin_slot = int(f32(fin - f32(f32(fin_round) * f32(F))))
                minl_s = parts[fin_round, :, fin_slot, 1]; nmin_s = parts[fin_round, :, fin_slot, 2]
                dupc_s = np.where(np.abs((minl_s - dup).astype(f32)) < f32(1e-9), nmin_s, f32(0.0)).astype(f32)
                num_preceding_h[h] = f32(np.where(np.arange(S) < s, dupc_s, f32(0.0)).sum())
        num_preceding_b = num_preceding_h[hb]
        start = s * slice_len; end = min(start + slice_len, V)
        for i in range(start, end, BLOCK):
            n = min(BLOCK, end - i)
            xb = np.full(BLOCK, NEG_INF, f32); xb[:n] = x[i:i + n]
            keep = xb > pivot_logit_b
            if ties:
                with np.errstate(invalid="ignore"):
                    dmask = (tl_div(tl_exp((xb - M).astype(f32), cfg), Zb, cfg) == dup_b) & ~stop_b
                cum = (np.cumsum(dmask.astype(np.int32)).astype(f32) + num_preceding_b).astype(f32)
                keep_dup = dmask & (cum <= numkeep_b)
                keep = keep & (~dmask | keep_dup)
            keep = keep | stop_b
            out[i:i + n] = np.where(keep, xb, NEG_INF)[:n]
            keep_all[i:i + n] = keep[:n]
    trace["keep"] = keep_all
    return out


def topp_split_row(x_in: np.ndarray, p: float, S: int, F: int = SPLIT_FANOUT, R: int = SPLIT_ROUNDS, cfg: Cfg | None = None):
    x = np.asarray(x_in, dtype=f32).copy()
    cfg = cfg or Cfg.for_vocab(int(x.shape[0]), S)
    trace: dict = {}
    if f32(p) >= f32(1.0):
        trace["early"] = "p>=1"; trace["keep"] = np.ones(x.shape[0], dtype=bool)
        return x, np.zeros((S, 4), f32), np.zeros((R, S, F, 3), f32), trace
    stats = stats_kernel(x, p, S, STATS_BLOCK, cfg)
    parts = np.zeros((R, S, F, 3), f32)
    for r in range(R):
        step_kernel(x, stats, parts, p, r, S, F, R, STEP_BLOCK, cfg)
    out = mask_kernel(x, stats, parts, p, S, F, R, MASK_BLOCK, cfg, trace)
    return out, stats, parts, trace


def topp_keep_row(x_in: np.ndarray, p: float, S: int) -> np.ndarray:
    """The mask kernel's `keep` predicate per vocabulary lane (all True when the pipeline leaves the row untouched: p >= 1,
    (nearly) uniform / empty row, or a `nomask` search outcome; True on a warp half's lanes when only that half stops,
    `sampling.topp_keep`)."""
    return topp_split_row(x_in, p, S)[3]["keep"]

"""Exact-fp32 model of vLLM d9105ea80 `topk_topp_triton._topk_topp_kernel`'s standalone top-p path for ONE logits row, as
Triton 3.7.1 + ptxas lower it for sm_90 (num_warps = 8, BLOCK_SIZE 8192, BLOCK_SIZE_TRUNC 4096, the CUDA launch).

`apply_top_k_top_p_triton` runs it for a row whose top-k is off (k >= V, or no top-k in the batch) whenever the split pipeline
(`topp_split`) does not: a sampler batch of more than `topp_geometry.SPLIT_MAX_BATCH` rows.  A row's result depends on the row
and its p alone: each program handles its rows one at a time and the program's BUFFER row is rewritten before it is read.

The kernel, pass by pass (`standalone_row`; ``trace`` keeps every intermediate):

  * zeroth pass, the first [8192] block: n = #(lanes > -inf), avg = sum / n, sq = sum(x * x) / n, std = sqrt(max(sq - avg * avg,
    0)), max_sample = avg + std * 10;
  * first pass, every [8192] tile: the row max, the min of the lanes > -inf, sum_exp += tl.sum(exp(x - max_sample));
  * the Gaussian cut: sigma = `NORMAL_CDF_TO_SIGMA[clamp(int(p * 200), 0, 199)]`, sigma += |sigma| * -0.25, outlier_prob =
    exp(avg + std * sigma - max_sample) / sum_exp;
  * second pass: probs = exp(x - max_sample) / sum_exp, the outliers (probs > outlier_prob) compacted in lane order into the
    buffer, sum_outlier += tl.sum(outlier * probs);
  * the search: when sum_outlier > p, over the compacted outliers in [4096] tiles from min = outlier_prob, else over every lane's
    probability in [8192] tiles from min = exp(min - max_sample) / sum_exp; to max = exp(max - max_sample) / sum_exp.  Each
    round's pivot is (max - min) * 0.5 + min; a tile's mass above it and `_update_min_larger_stats` (the least probability above
    the pivot and the lanes within 1e-9 of it, merged tile by tile); a round ends the search when mass >= p and mass - min * count
    < p, or when the bracket is under 1e-9 wide, or at round 18 (then the pivot is (max + min) / 2);
  * the mask: the pivot logit log(pivot * sum_exp) + max_sample; nothing masked unless it is below the row max; else a lane is
    kept when above it, and of the lanes within 1e-9 of log(min * sum_exp) + max_sample only the first num_keep in lane order,
    num_keep = count - u32((mass - p) / min) (u32 arithmetic), when that is below the count.

Lowering (read off the compiled kernel's TTGIR and PTX):

  * every [N] load is `blocked<{sizePerThread [4], threadsPerWarp [32], warpsPerCTA [8]}>`, the outlier buffer's [4096] tile
    included, and every `tl.sum` / `tl.max` / `tl.min` reshapes it (allow_reorder) to sizePerThread 1 without moving a register:
    per thread sequential over its (rep, j) registers, the warp's xor butterfly 16 .. 1, the 8 warps' butterfly 4, 2, 1
    (`topp_split.tree_sum_1d(·, 4, 8)`);
  * `tl.sum(x * x)` is one fma chain a thread: fma(x0, x0, x1 * x1), then fma(xk, xk, acc);
  * `sq - avg * avg`, `avg + std * 10`, `avg + std * sigma`, `(max - min) * 0.5 + min` and `mass - min * count` are fma.rn.f32
    (the count by cvt.rn.f32.u32); `sigma + |sigma| * -0.25` too, its product exact;
  * `tl.sqrt` is sqrt.approx.ftz.f32 (MUFU.SQRT, `mufu.MufuSqrtFtz`); `tl.exp`, `/` and `tl.log` are `topp_split`'s (MUFU.EX2 of
    x * log2e with ptxas' non-ftz path, div.full.f32, libdevice `__nv_logf`); `(max + min) / 2.0` is a div.full.f32;
  * `int(p * 200)` is cvt.rzi.s32.f32 (NaN -> 0), the u32 cast cvt.rzi.u32.f32 (saturating, NaN -> 0);
  * a masked product `outlier * probs` or `probs * above` is fused into the tree's adds as fma(probs, 0 or 1, acc): the product
    is exact, so it is the tree of the gated values.

Domain: every lane finite or -inf, no intermediate NaN, and no `tl.log` operand below FLT_MIN.  On a NaN the model's max / min
are numpy's where the GPU's max.f32 / min.f32 return the other operand; `sampling.nv_logf` flushes a subnormal operand where the
inlined `__nv_logf` first scales it by 2^23 with a non-ftz mul.
"""
from __future__ import annotations

import numpy as np

from verity.definitions.registry.topp_split import Cfg, fma32, tl_div, tl_exp, tl_log, tree_sum_1d, _bfly

f32 = np.float32
BLOCK = 8192              # BLOCK_SIZE on CUDA
BLOCK_TRUNC = 4096        # BLOCK_SIZE_TRUNC on CUDA
NUM_WARPS = 8
SPT = 4                   # sizePerThread of every [BLOCK] / [BLOCK_TRUNC] load
MAX_ROUNDS = 18
EPS = f32(1e-9)
NEG_INF, INF = f32(-np.inf), f32(np.inf)

#: vLLM d9105ea80 `topk_topp_triton._NORMAL_CDF_TO_SIGMA_TABLE`, the fp32 table the kernel indexes by int(p * 200)
NORMAL_CDF_TO_SIGMA = (
    3.656, 3.650, 3.650, 3.650, 3.626, 3.626, 3.626, 3.514, 3.514, 3.503,
    3.503, 3.434, 3.434, 3.428, 3.428, 3.387, 3.380, 3.380, 3.376, 3.373,
    3.373, 3.356, 3.354, 3.354, 3.291, 3.249, 3.234, 3.214, 3.198, 3.198,
    3.185, 3.177, 3.177, 3.165, 3.164, 3.161, 3.138, 3.120, 3.115, 3.113,
    3.093, 3.066, 3.054, 3.043, 3.037, 3.023, 2.993, 2.991, 2.976, 2.970,
    2.952, 2.946, 2.932, 2.908, 2.902, 2.895, 2.886, 2.874, 2.861, 2.844,
    2.836, 2.810, 2.801, 2.790, 2.784, 2.779, 2.767, 2.757, 2.745, 2.733,
    2.723, 2.716, 2.693, 2.678, 2.671, 2.656, 2.649, 2.629, 2.611, 2.595,
    2.592, 2.585, 2.574, 2.550, 2.543, 2.534, 2.521, 2.518, 2.497, 2.485,
    2.468, 2.450, 2.441, 2.430, 2.412, 2.402, 2.389, 2.383, 2.377, 2.364,
    2.349, 2.338, 2.332, 2.319, 2.310, 2.301, 2.282, 2.274, 2.266, 2.250,
    2.242, 2.236, 2.226, 2.215, 2.207, 2.196, 2.179, 2.171, 2.162, 2.147,
    2.135, 2.121, 2.109, 2.095, 2.085, 2.073, 2.063, 2.045, 2.030, 2.016,
    2.003, 1.992, 1.983, 1.972, 1.960, 1.949, 1.940, 1.928, 1.912, 1.897,
    1.881, 1.869, 1.854, 1.838, 1.824, 1.807, 1.792, 1.779, 1.764, 1.751,
    1.739, 1.726, 1.711, 1.697, 1.685, 1.668, 1.652, 1.636, 1.622, 1.603,
    1.585, 1.568, 1.551, 1.534, 1.513, 1.499, 1.480, 1.464, 1.441, 1.422,
    1.394, 1.373, 1.347, 1.320, 1.296, 1.270, 1.246, 1.219, 1.190, 1.163,
    1.135, 1.104, 1.073, 1.041, 1.006, 0.969, 0.931, 0.894, 0.851, 0.806,
    0.757, 0.702, 0.643, 0.574, 0.498, 0.405, 0.288, 0.134, -0.110, -3.813,
)


def _cfg() -> Cfg:
    return Cfg()


def sqrt_approx(x: np.float32) -> np.float32:
    """sqrt.approx.ftz.f32: MUFU.SQRT by its measured table."""
    from verity.definitions.ops.mufu import mufu_sqrt_words, table
    w = np.array([np.float32(x)], f32).view(np.uint32)
    return mufu_sqrt_words(w, table("sqrt")).astype(np.uint32).view(f32)[0]


def tree_sumsq(vals: np.ndarray) -> np.float32:
    """`tl.sum(x * x)` of an [8192] block: a thread's fma chain fma(x0, x0, x1 * x1), fma(xk, xk, acc) over its (rep, j) registers,
    then the warp and cross-warp butterflies."""
    per_thread = len(vals) // (32 * NUM_WARPS)
    v = vals.reshape(per_thread // SPT, NUM_WARPS, 32, SPT).astype(f32)      # [rep, w, t, j]
    regs = [v[r, :, :, j] for r in range(per_thread // SPT) for j in range(SPT)]
    acc = fma32(regs[0], regs[0], (regs[1] * regs[1]).astype(f32))
    for r in regs[2:]:
        acc = fma32(r, r, acc)
    return _bfly(_bfly(acc, 32, (16, 8, 4, 2, 1)), NUM_WARPS, (4, 2, 1))


def tile_sums(vals: np.ndarray, B: int) -> np.ndarray:
    """`tl.sum` of each [B] tile of ``vals`` (a multiple of B long)."""
    return tree_sum_1d(vals.reshape(-1, B), SPT, NUM_WARPS)


def _pad(v: np.ndarray, B: int, fill) -> np.ndarray:
    n = -(-len(v) // B) * B
    out = np.full(n, fill, f32)
    out[:len(v)] = v
    return out


def _u32(q: np.float32) -> int:
    """cvt.rzi.u32.f32: truncate, saturating to [0, 2^32 - 1], NaN -> 0."""
    q = float(q)
    if not q > 0:
        return 0
    return 0xFFFFFFFF if q >= 2.0 ** 32 else int(q)


def _s32(q: np.float32) -> int:
    """cvt.rzi.s32.f32: truncate, saturating to the i32 range, NaN -> 0."""
    q = float(q)
    if q != q:
        return 0
    return max(-2 ** 31, min(2 ** 31 - 1, int(q))) if abs(q) < 2.0 ** 31 else (2 ** 31 - 1 if q > 0 else -2 ** 31)


def _log(x: np.float32) -> np.float32:
    return tl_log(np.array([x], f32), _cfg())[0]


def _div(a, b) -> np.ndarray:
    return tl_div(np.atleast_1d(np.asarray(a, f32)), f32(b), _cfg())


def search(data: np.ndarray, B: int, p: np.float32, lo: np.float32, hi: np.float32, trace: dict | None = None):
    """The kernel's pivot search over ``data`` in [B] tiles (zero-padded): (pivot, min_larger, count, mass)."""
    tiles = _pad(data, B, 0.0).reshape(-1, B)
    rounds = []
    for it in range(1, MAX_ROUNDS + 1):
        piv = fma32(f32(hi - lo), f32(0.5), lo)[()]
        above = tiles > piv
        mass_t = tree_sum_1d(np.where(above, tiles, f32(0)), SPT, NUM_WARPS)
        tmin = np.where(above, tiles, f32(1.0)).min(axis=1)
        tcnt = (above & (np.abs((tiles - tmin[:, None]).astype(f32)) < EPS)).sum(axis=1)
        mass, ml, n = f32(0), f32(1.0), 0
        for t in range(len(tiles)):
            mass = f32(mass + mass_t[t])
            is_new = tmin[t] < ml
            is_same = abs(f32(tmin[t] - ml)) < EPS
            n = int(tcnt[t]) if is_new else (n + int(tcnt[t]) * int(is_same)) & 0xFFFFFFFF
            ml = min(ml, tmin[t])
        found = mass >= p and fma32(-ml, f32(n), mass)[()] < p
        out = (piv, ml, n, mass) if found else None
        if mass > p:
            lo = piv
        elif mass < p:
            hi = piv
        rounds.append({"pivot": piv, "mass": mass, "min": ml, "count": n, "lo": lo, "hi": hi})
        if f32(hi - lo) < EPS or it >= MAX_ROUNDS:
            out = (_div(f32(hi + lo), 2.0)[0], ml, n, mass)
        if out is not None:
            if trace is not None:
                trace["rounds"] = rounds
            return out
    raise AssertionError("unreachable: round 18 always ends the search")


def standalone_row(x_in: np.ndarray, p: float, trace: dict | None = None) -> np.ndarray:
    """The keep bits the kernel's standalone top-p path leaves on one row (True: the logit is kept, False: set to -inf)."""
    x = np.asarray(x_in, dtype=f32)
    V = len(x)
    p = f32(p)
    keep_all = np.ones(V, bool)
    if not p < f32(1.0):
        return keep_all
    blk0 = _pad(x[:BLOCK], BLOCK, NEG_INF)[:BLOCK]
    fin = blk0 > NEG_INF
    nf = int(fin.sum())
    fl = np.where(fin, blk0, f32(0))
    nff = f32(nf)
    avg = _div(tile_sums(fl, BLOCK)[0], nff)[0] if nf > 0 else f32(0)
    sq = _div(tree_sumsq(fl), nff)[0] if nf > 0 else f32(0)
    var = fma32(-avg, avg, sq)[()]
    std = sqrt_approx(max(var, f32(0)))
    ms = fma32(std, f32(10.0), avg)[()]

    xp = _pad(x, BLOCK, NEG_INF)
    tiles = xp.reshape(-1, BLOCK)
    mx = tiles.max(axis=1).max()
    mn = min(np.where(tiles > NEG_INF, tiles, INF).min(), mx)
    e = np.where(np.arange(len(xp)) < V, tl_exp((xp - ms).astype(f32), _cfg()), f32(0))
    se = f32(0)
    for s in tile_sums(e, BLOCK):
        se = f32(se + s)

    idx = max(0, min(_s32(f32(p * f32(200))), 199))
    sig0 = f32(NORMAL_CDF_TO_SIGMA[idx])
    sig = fma32(abs(sig0), f32(-0.25), sig0)[()]
    opiv = fma32(std, sig, avg)[()]
    oprob = _div(tl_exp(np.array([opiv - ms], f32), _cfg()), se)[0]

    probs = _div(tl_exp((x - ms).astype(f32), _cfg()), se)
    out = probs > oprob
    so = f32(0)
    for s in tile_sums(_pad(np.where(out, probs, f32(0)), BLOCK, 0.0), BLOCK):
        so = f32(so + s)
    hi = _div(tl_exp(np.array([mx - ms], f32), _cfg()), se)[0]
    lo = _div(tl_exp(np.array([mn - ms], f32), _cfg()), se)[0]
    outliers = so > p
    if outliers:
        piv, ml, n, mass = search(probs[out], BLOCK_TRUNC, p, oprob, hi, trace)
    else:
        piv, ml, n, mass = search(probs, BLOCK, p, lo, hi, trace)

    t = f32(ml * se)
    dup = f32((NEG_INF if t == 0 else _log(t)) + ms)
    nk = (n - _u32(_div(f32(mass - p), ml)[0])) & 0xFFFFFFFF
    t = f32(piv * se)
    fpiv = f32((NEG_INF if t == 0 else _log(t)) + ms)
    if trace is not None:
        trace.update(n_finite=nf, avg=avg, sq=sq, std=std, max_sample=ms, max=mx, min=mn, sum_exp=se, sigma=sig,
                     outlier_prob=oprob, n_outliers=int(out.sum()), sum_outliers=so, outliers=bool(outliers), pivot=piv,
                     min_larger=ml, count=n, mass=mass, dup=dup, num_keep=nk, pivot_logit=fpiv)
    if not fpiv < mx or fpiv == NEG_INF:
        return keep_all
    keep = x > fpiv
    if nk < n:
        d = np.abs((x - dup).astype(f32)) < EPS
        rank = np.cumsum(d)
        keep &= ~(d & (rank > nk))
    return keep


def standalone_mask_row(x_in: np.ndarray, p: float) -> np.ndarray:
    """The row the kernel stores: each logit kept or set to -inf."""
    x = np.asarray(x_in, dtype=f32)
    return np.where(standalone_row(x, p), x, NEG_INF).astype(f32)


__all__ = ["BLOCK", "BLOCK_TRUNC", "MAX_ROUNDS", "NORMAL_CDF_TO_SIGMA", "search", "standalone_mask_row", "standalone_row",
           "sqrt_approx", "tile_sums", "tree_sumsq"]

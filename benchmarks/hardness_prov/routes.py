"""The cheapest exact route through a served int8 code matrix q (n outputs x K inputs), against a 2-bit Z (x = 0 leaves
E's bits), in W1 units (dense int8 = 1).

A route splits q = P + S: a cheap part P and a residual S, both exact integers. S is priced by the cheapest dense route
that holds its range [-a, a], plus its entries beyond a on the ALU (IDP4A, 8 a MAC), or by 2:4 sparsity. The dense routes
are `hardness_public/prices.py`'s (b1 planes on the H200, FP4 pieces on sm_120) plus exact Strassen in s8: a depth-d
Strassen sums at most 2^d of S's entries an operand, so it fits s8 while 2^d·a ≤ 127. Its prices are HT-CLAIM v2's (§5,
HT-PREMISE's executor, fast reading, MAC-weighted at Llama-3.2-1B's shapes, read): 0.766 at d = 3, 0.814 at d = 2,
0.896 at d = 1. They are applied to sm_120 as well (HT-PREMISE v2 executed 0.761 there at K 2,048).

The parts P searched:

- low rank: the real rank-r SVD of q, priced r/n + 3r/K (the K-side factor in int8, the s32 intermediate as three int8
  passes on the n side). A real L is generous to the cheater by at most one code of S's range; an exact split needs an
  integer low-rank part;
- a per-row scaled ternary a·t (the plant's t, or t refit from q at half the row's max), one ternary pass;
- a grid g·M (q mostly divisible by g), M priced as a residual and the off-grid entries on the ALU;
- nothing (P = 0): S = q, which covers small codes with a pinned absmax (the pin is an ALU outlier);
- clustered rows (`vq`): q's rows k-means clustered into m integer centers, P = C[assign], priced m/n (each output
  is its center's output plus its residual's); clustered columns likewise at m/K (the inputs summed per cluster
  first). High rank, so the top-r energy share doesn't see it.

Every price is a route the cheater can run; none is a lower bound.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hardness_public"))
from prices import CHIPS, dense_part

B = 2
ALPHA = (0, 1, 2, 3, 4, 7, 15, 31, 63, 127)
STRASSEN = {3: 0.766, 2: 0.814, 1: 0.896}
GRIDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16)
N_SIDE_PASSES = 3


def strassen(a):
    if a == 0:
        return 0.0
    for d in (3, 2, 1):
        if (1 << d) * a <= 127:
            return STRASSEN[d]
    return 1.0


def small(chip, a, b=B):
    """Cheapest dense route for entries in [-a, a]."""
    if a == 0:
        return 0.0
    return min(dense_part(chip, a, b), strassen(a))


def residual(S, chip, b=B):
    """(cost, a) of the cheapest route for an integer residual S: a dense pass on [-a, a] and the rest on the ALU, or
    2:4 on S's nonzeros."""
    S = np.asarray(S)
    ab = np.minimum(np.abs(S), 255).astype(np.int64).ravel()
    gt = 1.0 - np.cumsum(np.bincount(ab, minlength=256)) / ab.size
    alu = CHIPS[chip]["alu_mac"]
    best, arg = (2.0, 255) if gt[127] > 0 else (1.0, 127)
    for a in ALPHA:
        c = small(chip, a, b) + gt[a] * alu
        if c < best:
            best, arg = c, a
    n, k = S.shape
    if k % 4 == 0 and gt[0] <= 0.5:
        cnt = (S != 0).reshape(n, k // 4, 4).sum(axis=2)
        excess = float(np.maximum(cnt - 2, 0).sum()) / S.size
        for a in ALPHA[1:]:
            c = CHIPS[chip]["sparse24"] * dense_part(chip, a, b) + (excess + gt[a]) * alu
            if c < best:
                best, arg = c, -a
    return float(best), int(arg)


def lowrank(r, n, k):
    return r / n + N_SIDE_PASSES * r / k


def ranks(m):
    return sorted({r for r in (0, 1, 2, 4, 8, 16, 32, m // 8, m // 4) if r < m})


def ternary_refit(q):
    thr = 0.5 * np.abs(q).max(axis=1, keepdims=True)
    return (np.sign(q) * (np.abs(q) > thr)).astype(np.int32)


def ternary_split(q, t):
    """q = a·t + S, a per row the mode of |q| on t's support."""
    a = np.array([np.bincount(np.abs(q[i][t[i] != 0])).argmax() if (t[i] != 0).any() else 0 for i in range(len(q))])
    return q - a[:, None] * t


VQ_FRACTIONS = (16, 8, 5)
VQ_RESTARTS = 3


def kmeans(x, m, iters=8, seed=0):
    """Lloyd's k-means on the rows of x (float32), greedy k-means++ seeded (each new center the best of 2 + ln m
    sampled candidates); returns (centers, assign)."""
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    tries = 2 + int(np.log(max(m, 2)))
    xx = (x * x).sum(axis=1, keepdims=True)
    idx = [int(rng.integers(len(x)))]
    d2 = ((x - x[idx[0]]) ** 2).sum(axis=1)
    for _ in range(m - 1):
        tot = float(d2.sum())
        if tot <= 0:
            idx.append(int(rng.integers(len(x))))
            continue
        cand = rng.choice(len(x), size=tries, p=d2 / tot)
        dc = np.maximum(xx - 2 * x @ x[cand].T + xx[cand, 0], 0)
        nd = np.minimum(d2[:, None], dc)
        j = int(np.argmin(nd.sum(axis=0)))
        idx.append(int(cand[j]))
        d2 = nd[:, j]
    c = x[idx].copy()
    for _ in range(iters):
        a = np.argmin(xx - 2 * x @ c.T + (c * c).sum(axis=1), axis=1)
        cnt = np.bincount(a, minlength=m)
        s = np.zeros_like(c)
        np.add.at(s, a, x)
        live = cnt > 0
        c[live] = s[live] / cnt[live, None]
    return c, np.argmin(xx - 2 * x @ c.T + (c * c).sum(axis=1), axis=1)


def vq_routes(q, chip, cache):
    """Cheapest clustered-rows or clustered-columns route: (cost, side, m, a)."""
    best = (1.0, None, 0, 0)
    for side, mat, full in (("rows", q, q.shape[0]), ("cols", q.T, q.shape[1])):
        for f in VQ_FRACTIONS:
            m = full // f
            if m < 2:
                continue
            key = (side, m)
            if key not in cache:
                fits = []
                for seed in range(VQ_RESTARTS):
                    c, a = kmeans(mat, m, seed=seed)
                    S = mat - np.rint(c).astype(np.int64)[a]
                    fits.append((float((np.abs(S) > 15).mean()), seed, S))
                cache[key] = min(fits, key=lambda f: f[:2])[2]
            c, a_ = residual(cache[key], chip)
            cost = m / full + c
            if cost < best[0]:
                best = (cost, side, m, a_)
    return best


def search(q, t=None, vq=True):
    """Every family's cheapest route for q, per chip, and the best."""
    q = np.asarray(q, dtype=np.int64)
    n, k = q.shape
    vq_cache = {}
    U, s, Vt = np.linalg.svd(q.astype(np.float64), full_matrices=False)
    out = {}
    lr_S = {}
    for r in ranks(min(n, k)):
        L = (U[:, :r] * s[:r]) @ Vt[:r] if r else 0.0
        lr_S[r] = np.rint(q - L).astype(np.int64)
    terns = {"refit": ternary_refit(q)}
    if t is not None:
        terns["plant"] = np.asarray(t, dtype=np.int64)
    grids = {g: (q % g == 0) for g in GRIDS}
    for chip in CHIPS:
        fam = {}
        lr = {}
        for r, S in lr_S.items():
            c, a = residual(S, chip)
            lr[r] = (lowrank(r, n, k) + c, a)
        r_best = min(lr, key=lambda r: lr[r][0])
        fam["lowrank"] = {"cost": min(1.0, lr[r_best][0]), "r": r_best, "a": lr[r_best][1],
                          "by_r": {str(r): round(v[0], 4) for r, v in lr.items()}}
        fam["plain"] = {"cost": min(1.0, lr[0][0]), "a": lr[0][1]}
        for name, tt in terns.items():
            c, a = residual(ternary_split(q, tt), chip)
            fam[f"ternary_{name}"] = {"cost": min(1.0, small(chip, 1) + c), "a": a}
        gbest = (1.0, 0)
        for g, on in grids.items():
            f_on = float(on.mean())
            if f_on < 0.9:
                continue
            c, _ = residual(np.where(on, q // g, 0), chip)
            c += (1 - f_on) * CHIPS[chip]["alu_mac"]
            if c < gbest[0]:
                gbest = (c, g)
        fam["grid"] = {"cost": min(1.0, gbest[0]), "g": gbest[1]}
        if vq:
            c, side, m, a = vq_routes(q, chip, vq_cache)
            fam["vq"] = {"cost": min(1.0, c), "side": side, "m": m, "a": a}
        best = min(fam, key=lambda f: fam[f]["cost"])
        out[chip] = {"cost": fam[best]["cost"], "family": best, "families": fam}
    e = np.cumsum(s.astype(np.float64) ** 2) / float((s.astype(np.float64) ** 2).sum())
    out["energy_top"] = {str(r): float(e[r - 1]) for r in (1, 4, 8, 16, 32, 64, 128, min(n, k) // 8) if 0 < r <= len(e)}
    return out


def stats(q):
    """Public code statistics, with `tail{a}`: the share of the codes' energy beyond ±a, Σ(|q| − a)₊² / Σq², which the
    cheap part of a route whose residual stays in [−a, a] must carry."""
    q = np.asarray(q, dtype=np.int64)
    p = np.bincount((q + 127).ravel(), minlength=255) / q.size
    nz = p[p > 0]
    e = float((q.astype(np.float64) ** 2).sum()) or 1.0
    tails = {f"tail{a}": float((np.maximum(np.abs(q) - a, 0).astype(np.float64) ** 2).sum() / e) for a in (15, 31, 63)}
    return {**tails, "sigma_q": float(q.std()), "entropy_bits": float(-(nz * np.log2(nz)).sum()),
            "frac_abs_le15": float((np.abs(q) <= 15).mean()), "frac_abs_gt31": float((np.abs(q) > 31).mean()),
            "frac_abs_gt63": float((np.abs(q) > 63).mean()),
            "frac_zero": float((q == 0).mean()), "p_odd": float((q % 2 != 0).mean())}

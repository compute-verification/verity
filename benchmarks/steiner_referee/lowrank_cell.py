#!/usr/bin/env python3
"""Exact rank-metric Steiner lengths of one cell of the low-rank-mask prefix targets (RS-D2F, CMP-13).

With masks E = U V^T and F = W Z^T of rank r and A = B = 0, cell (i, j) of the committed prefixes is
u_i^T G_s z_j, where G_s = sum_{t <= s} H_t and H_t = V_t^T W_t (block t of V and W) has rank min(r, L). Any Steiner
tree in F_q^{r x r} (rank metric) spanning 0 and G_1..G_S is an attack: each edge c d^T becomes one product
(u_i . c)(d . z_j) in every cell, so the n^2 cells cost n^2 times the tree's length. The naive attack is the chain,
S * rank(H). This computes the minimum exactly by Dreyfus-Wagner on the Cayley graph of F_q^{r x r} whose generators
are the nonzero rank-1 matrices (its graph metric is the rank metric).

Usage: lowrank_cell.py R Q RHO S TRIALS SEED   (prints one JSON line per instance and a summary line)
"""

import itertools
import json
import sys

import numpy as np


def setup(r, q):
    nd = r * r
    size = q**nd
    pw = q ** np.arange(nd, dtype=np.int64)
    digits = (np.arange(size, dtype=np.int64)[:, None] // pw[None, :]) % q
    gens = set()
    for c in itertools.product(range(q), repeat=r):
        if not any(c):
            continue
        for d in itertools.product(range(q), repeat=r):
            if not any(d):
                continue
            m = (np.outer(c, d) % q).reshape(-1)
            gens.add(int((m * pw).sum()))
    gens = np.array(sorted(gens), dtype=np.int64)
    nb = ((digits[:, None, :] + digits[gens][None, :, :]) % q) @ pw
    nb = nb.astype(np.int32)
    big = np.int16(10000)
    rank = np.full(size, big, dtype=np.int16)
    rank[0] = 0
    rank = relax(rank, nb)
    return digits, pw, nb, rank


def relax(f, nb):
    """g(v) = min_u f(u) + rank(v - u), by Dial's algorithm on the unit-weight Cayley graph."""
    g = f.copy()
    big = 10000
    d = int(g.min())
    while True:
        frontier = np.nonzero(g == d)[0]
        if frontier.size == 0:
            if not np.any((g > d) & (g < big)):
                break
            d += 1
            continue
        nbrs = nb[frontier].ravel()
        better = nbrs[g[nbrs] > d + 1]
        g[better] = d + 1
        d += 1
    return g


def steiner(targets, digits, pw, nb, rank, q):
    k = len(targets)
    dp = {}
    for i, t in enumerate(targets):
        dp[1 << i] = rank[((digits - digits[t]) % q) @ pw]
    full = (1 << k) - 1
    for mask in sorted(range(1, full + 1), key=lambda m: bin(m).count("1")):
        if mask in dp:
            continue
        low = mask & -mask
        best = None
        sub = (mask - 1) & mask
        while sub:
            if sub & low:
                cand = dp[sub] + dp[mask ^ sub]
                best = cand if best is None else np.minimum(best, cand)
            sub = (sub - 1) & mask
        dp[mask] = relax(best.astype(np.int16), nb)
    return int(dp[full][0])


def random_rank(r, q, rho, rng, pw, rank):
    while True:
        a = rng.integers(0, q, size=(r, rho))
        b = rng.integers(0, q, size=(rho, r))
        m = (a @ b) % q
        code = int((m.reshape(-1) * pw).sum())
        if int(rank[code]) == rho:
            return m


def main():
    r, q, rho, s, trials, seed = (int(x) for x in sys.argv[1:7])
    digits, pw, nb, rank = setup(r, q)
    rng = np.random.default_rng(seed)
    results = []
    while len(results) < trials:
        hs = [random_rank(r, q, rho, rng, pw, rank) for _ in range(s)]
        gs = np.cumsum(np.array(hs), axis=0) % q
        codes = [int((g.reshape(-1) * pw).sum()) for g in gs]
        if 0 in codes or len(set(codes)) < s:
            continue
        st = steiner(codes, digits, pw, nb, rank, q)
        star = int(sum(int(rank[c]) for c in codes))
        rec = {"r": r, "q": q, "rho": rho, "S": s, "steiner": st, "chain": s * rho, "star": star,
               "per_target": st / s, "G": [g.tolist() for g in gs]}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    sts = [x["steiner"] for x in results]
    print(json.dumps({"summary": True, "r": r, "q": q, "rho": rho, "S": s, "trials": trials, "chain": s * rho,
                      "steiner_min": min(sts), "steiner_mean": sum(sts) / len(sts), "steiner_max": max(sts),
                      "below_chain": sum(1 for x in sts if x < s * rho)}), flush=True)


if __name__ == "__main__":
    main()

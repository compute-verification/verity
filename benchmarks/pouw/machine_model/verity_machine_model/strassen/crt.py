"""CRT schedules: C = AB from C mod m_1, ..., C mod m_s, each residue product computed by any schedule over Z_m.

Reduction mod m is free (no additions are counted), so inside a residue every combination is reduced again and an
operand never ranges over more than m values: the sched DP with every child range clamped at m (Tables(clamp=m)).
w_l(m) is the least cost of C mod m at depth budget l from int8 operands, reduced to min(256, m) values. A CRT
schedule is a set of pairwise coprime moduli whose product is at least the number of values an entry of C can take,
|S_K| for an inner dimension K = 2^l b; it costs sum w_l(m_i).

Upper bound: the best pairwise coprime set among the moduli whose w_l is computed (an explicit schedule). Lower bound:
w_l is non-decreasing in m (a schedule for a larger clamp is one for a smaller), so every integer m costs at least
w_l's lower table at the largest computed point at or below it; pairwise coprime moduli are distinct, so the least
cost of distinct moduli with product at least |S_K| (a 0/1 knapsack in bits, item bits rounded up, the need rounded
down) is a lower bound. Above M_INDIV the moduli are cells with unbounded multiplicity (a further relaxation).
"""
from __future__ import annotations

import math
import sys

import numpy as np

from .sched import BIG, STRASSEN, TRIVIAL, WINOGRAD, Tables, at

M_INDIV = 4097
UNIT = 4096  # knapsack capacity unit: 1/4096 bit


def points(mmax: int) -> list[int]:
    """The moduli whose w_l is computed: every integer up to 300, then a geometric grid to mmax."""
    pts = set(range(2, 301))
    for x in range(int(32 * math.log2(300)), 32 * 12 + 1):
        pts.add(int(math.ceil(2 ** (x / 32))))
    x = 8 * 12
    while 2 ** (x / 8) <= mmax:
        pts.add(int(math.ceil(2 ** (x / 8))))
        x += 1 if x < 8 * 24 else 2
    pts.add(mmax)
    return sorted(p for p in pts if 2 <= p <= mmax)


def residue_costs(name: str, m: int, lmax: int):
    """[(w_lo, w_up, has_lo)] for l = 0..lmax: C mod m from int8 operands, in units of 8^-(l+1) s8 MACs per block MAC."""
    T = Tables(name, lmax, [STRASSEN, WINOGRAD, TRIVIAL], clamp=m)
    v = min(256, m)
    out = []
    tabs = T.level(None, None, None, None, 0)
    for l in range(lmax + 1):
        if l:
            tabs = T.level(*tabs, l)
        lo, up, hlo, _ = tabs
        out.append((at(T, lo, v, v), at(T, up, v, v), at(T, hlo, v, v)))
    return out


def outputs_lower(K: int) -> int:
    """A lower bound on |S_K|, the number of values sum_{i<K} x_i y_i takes over x, y in [-128, 127].

    Take y = -128 in K - 1 terms and y = -127 in one: the sum is -128 Z + X with Z = X_1 + X, X_1 any of the
    255 (K - 1) + 1 values of the first K - 1 x's and X = x_K. For each Z in [-128 (K - 1) - 1, 127 (K - 1) - 128] every
    X in [-128, -1] is reachable, giving the 128 values [-128 Z - 128, -128 Z - 1]; these blocks are disjoint, so
    |S_K| >= 128 (255 K - 381)."""
    return max(128 * (255 * K - 381), 1)


def outputs_upper(K: int) -> int:
    """Every entry of C lies in [-16256 K, 16384 K]: moduli with product >= 32640 K + 1 determine it."""
    return 32640 * K + 1


def best_coprime(costs: dict[int, int], need: int, min_size: int = 1):
    """Least sum of costs over pairwise coprime moduli, at least min_size of them, with product >= need (exhaustive
    with a rate bound)."""
    ms = sorted((m for m in costs if costs[m] < BIG), reverse=True)
    rate = [costs[m] / math.log2(m) for m in ms]
    suff = rate[:]
    for i in range(len(ms) - 2, -1, -1):
        suff[i] = min(suff[i], suff[i + 1])
    best = [BIG, ()]

    def dfs(i, prod, cost, chosen):
        if prod >= need and len(chosen) >= min_size:
            if cost < best[0]:
                best[0], best[1] = cost, tuple(chosen)
            return
        if i >= len(ms) or cost + suff[i] * math.log2(max(need / prod, 1)) >= best[0]:
            return
        for j in range(i, len(ms)):
            m = ms[j]
            if cost + costs[m] >= best[0] or cost + suff[j] * math.log2(max(need / prod, 1)) >= best[0]:
                continue
            if all(math.gcd(m, c) == 1 for c in chosen):
                chosen.append(m)
                dfs(j + 1, prod * m, cost + costs[m], chosen)
                chosen.pop()

    dfs(0, 1, 0, [])
    return best[0], best[1]


def knapsack_lower(lo_at, need: int, cells: list[tuple[int, int, int]], min_items: int = 1, ceiling: int = BIG) -> int:
    """Least cost of distinct integers m in [2, M_INDIV] (cost lo_at(m)) plus any number of cell items (cost, bits_hi)
    with total bits >= log2(need), using at least min_items items (1 or 2), or ceiling's least excluded item: an item
    costing at least ceiling is left out, and a set containing it costs at least that item."""
    cap = int(math.floor(math.log2(need) * UNIT))
    excluded = BIG
    items = [(lo_at(m), math.log2(m)) for m in range(2, M_INDIV + 1)] + [c for c in cells for _ in range(6)]
    kept = []
    for cost, bits in items:
        if cost >= ceiling:
            excluded = min(excluded, cost)
        else:
            kept.append((cost, bits))
    INF = np.int64(BIG)
    dp = np.full((3, cap + 1), INF, dtype=np.int64)  # dp[c][b]: at least min(c, 2) items, b bits (capped at cap)
    dp[0, 0] = 0

    def add(cost, bits):
        b = min(int(math.ceil(bits * UNIT)), cap)
        new = dp.copy()
        for c in range(3):
            src = dp[c]
            tgt = min(c + 1, 2)
            shifted = np.full(cap + 1, INF, dtype=np.int64)
            shifted[b:] = src[:cap + 1 - b] + cost
            shifted[cap] = min(shifted[cap], int(src[cap - b:].min()) + cost)
            new[tgt] = np.minimum(new[tgt], shifted)
        dp[:] = np.minimum(new, INF)

    for cost, bits in kept:
        add(cost, bits)
    cheapest = min(cost for cost, _ in items) if min_items >= 2 else 0
    return int(min(excluded + cheapest, *(dp[c, cap] for c in range(min_items, 3))))


def run(name: str, lmax: int, bs=(1,), mmax: int = 2 ** 24, dump: str | None = None):
    """mmax: w_l(mmax) is naive's cost at every l <= 12 on every instance, so every larger modulus costs naive's.
    dump: a JSON file of every modulus's costs, written if absent and read if present."""
    import json
    import os

    pts = points(mmax)
    if dump and os.path.exists(dump):
        with open(dump) as fh:
            w = {int(m): v for m, v in json.load(fh).items()}
        assert sorted(w) == pts and all(len(v) > lmax for v in w.values())
    else:
        w = {}
        for i, m in enumerate(pts):
            w[m] = residue_costs(name, m, lmax)
            if i % 50 == 0:
                print(f"  ... {i}/{len(pts)} moduli (m = {m})", file=sys.stderr, flush=True)
        if dump:
            with open(dump, "w") as fh:
                json.dump({str(m): v for m, v in w.items()}, fh)
    for l in range(1, lmax + 1):
        n8 = 8 ** (l + 1)
        lo = {m: w[m][l][0] for m in pts}
        up = {m: w[m][l][1] for m in pts}
        hlo = {m: w[m][l][2] for m in pts}
        floor_pt = lambda m: pts[np.searchsorted(pts, m, side="right") - 1]
        cells = [(lo[pts[k]], math.log2(pts[k + 1])) for k in range(len(pts) - 1) if pts[k] > M_INDIV]
        cells.append((lo[pts[-1]], 1024.0))  # every m >= mmax costs at least w_l(mmax)
        for b in bs:
            K = 2 ** l * b
            need_up = outputs_upper(K)
            need_lo = outputs_lower(K)
            ub, S = best_coprime(up, need_up, min_size=2)
            lb2 = knapsack_lower(lambda m: lo[floor_pt(m)], need_lo, cells, min_items=2, ceiling=max(ub, n8))
            single = hlo[floor_pt(min(need_lo, mmax))]
            lb = min(lb2, single)
            print(f"{name} l={l:2d} b={b:3d} n={K:7d}: CRT with a scheme in [{lb / n8:.6f}, {ub / n8:.6f}]"
                  f"  (>= 2 moduli >= {lb2 / n8:.6f}; one modulus with a scheme >= {single / n8:.6f})"
                  f"  best set {S}  lb-naive {lb - n8}", flush=True)


def crossover(name: str, dump: str, b_min: int, lmax: int = 12, mmax: int = 2 ** 24):
    """The least n at which a CRT schedule with a scheme beats naive tiled on n x n, padded to N = 2^l b' (b' >= b_min,
    leaves b' x b'; the moduli need only cover the n real terms), bracketed: an explicit set wins at n_up, and below
    n_lo every set's lower bound loses. Depths above lmax: a tree of depth l costs at least 7^l b'^3 / 8 s8 MACs
    (every scheme saves at most 1/8, every format costs at least 1/8 a MAC), checked to exceed n_up^3."""
    import json

    pts = points(mmax)
    with open(dump) as fh:
        w = {int(m): v for m, v in json.load(fh).items()}
    floor_pt = lambda m: pts[np.searchsorted(pts, m, side="right") - 1]
    best_up, best_lo = (BIG, None), (BIG, None)
    for l in range(1, lmax + 1):
        n8 = 8 ** (l + 1)
        lo = {m: w[m][l][0] for m in pts}
        up = {m: w[m][l][1] for m in pts}
        hlo = {m: w[m][l][2] for m in pts}
        cells = [(lo[pts[k]], math.log2(pts[k + 1])) for k in range(len(pts) - 1) if pts[k] > M_INDIV]
        cells.append((lo[pts[-1]], 1024.0))

        def cost_up(n, thr):
            return best_coprime(up, outputs_upper(n), min_size=2)

        def cost_lo(n, thr):
            """A lower bound, exact below thr: a modulus costing thr or more can't be in a set that costs less."""
            need = outputs_lower(n)
            lb2 = knapsack_lower(lambda m: lo[floor_pt(m)], need, cells, min_items=2, ceiling=thr)
            return min(lb2, hlo[floor_pt(min(need, mmax))]), None

        for bp in range(b_min, 3 * b_min + 1):
            N = 2 ** l * bp
            for which, f in (("up", cost_up), ("lo", cost_lo)):
                cur = best_up if which == "up" else best_lo
                n = 2
                while n <= min(N, cur[0] - 1):
                    c, S = f(n, n ** 3 * n8 // N ** 3 + 1)
                    if c >= n8:
                        break  # costs are non-decreasing in n: no n >= this one wins at (l, N)
                    if c * N ** 3 < n ** 3 * n8:
                        if n < cur[0]:
                            cur = (n, (l, bp, N, c / n8, S))
                        break
                    n = max(n + 1, math.ceil(N * (c / n8) ** (1 / 3)))
                if which == "up":
                    best_up = cur
                else:
                    best_lo = cur
    n_up = best_up[0]
    for l in range(lmax + 1, 64):
        assert 7 ** l * b_min ** 3 > 8 * n_up ** 3, (l, n_up)
        break
    print(f"{name} b_min={b_min}: CRT beats naive from n = {n_up} (explicit: depth, b', N, cost, moduli = {best_up[1]});"
          f" every CRT schedule loses below n = {best_lo[0]} ({best_lo[1]})", flush=True)
    return best_lo, best_up


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "crossover":
    for b in sys.argv[4].split(","):
        crossover(sys.argv[2], sys.argv[3], int(b))
    sys.exit(0)

if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (1,),
        dump=sys.argv[4] if len(sys.argv) > 4 else None)

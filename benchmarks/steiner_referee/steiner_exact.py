#!/usr/bin/env python3
"""Exact Steiner lengths over F2 on 4x4 bilinear forms (<2,2,2>, L = 1), for every subset of the 8 prefix targets
T_{pi,s} (s = 1, 2) plus 0, by Dreyfus-Wagner on all 2^16 forms with the 225 rank-1 steps.

Checks, for every set T of targets:
  * St({0} u T) = L * sum_pi hi_pi                    (Theorem 1 with Corollary 2: prefixes are geodesic chains);
  * St(T) >= L * (sum_pi hi_pi - min_pi lo_pi)         (Theorem B's per-tree bound: a tree rooted at a free head value,
                                                         however far, spanning T; St(T) is the shortest tree on T alone).

    python3 steiner_exact.py           # prints one JSON line
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np

N = 16
INF = 100


def bit(row, col):
    return 1 << (row * 4 + col)


def rank1_masks():
    out = set()
    for u in range(1, 16):
        for v in range(1, 16):
            m = 0
            for i in range(4):
                if u >> i & 1:
                    for j in range(4):
                        if v >> j & 1:
                            m |= bit(i, j)
            out.add(m)
    return sorted(out)


def target_mask(r, c, s):
    m = 0
    for k in range(s):
        m |= bit(r * 2 + k, c * 2 + k)
    return m


def relax(d, steps, idx):
    while True:
        best = d.copy()
        for m in steps:
            np.minimum(best, d[idx ^ m] + 1, out=best)
        if (best == d).all():
            return d
        d = best


def main():
    steps = rank1_masks()
    assert len(steps) == 225
    idx = np.arange(1 << N, dtype=np.int64)
    targets = [((r, c), s) for r in range(2) for c in range(2) for s in (1, 2)]
    terms = [target_mask(r, c, s) for (r, c), s in targets] + [0]  # terminal 8 is 0
    k = len(terms)
    dp = {}
    for i, t in enumerate(terms):
        d = np.full(1 << N, INF, dtype=np.int16)
        d[t] = 0
        dp[1 << i] = relax(d, steps, idx)
    for size in range(2, k + 1):
        for comb in itertools.combinations(range(k), size):
            S = sum(1 << i for i in comb)
            low = S & -S
            best = np.full(1 << N, INF, dtype=np.int16)
            sub = (S - 1) & S
            while sub:
                if sub & low:
                    np.minimum(best, dp[sub] + dp[S ^ sub], out=best)
                sub = (sub - 1) & S
            dp[S] = relax(best, steps, idx)
    rows, fails, tight_b = 0, [], 0
    for size in range(1, 9):
        for comb in itertools.combinations(range(8), size):
            S = sum(1 << i for i in comb)
            st_T = int(dp[S][terms[comb[0]]])
            st_0T = int(dp[S | (1 << 8)][0])
            hi, lo = {}, {}
            for i in comb:
                pi, s = targets[i]
                hi[pi] = max(hi.get(pi, 0), s)
                lo[pi] = min(lo.get(pi, 99), s)
            want0 = sum(hi.values())
            wantB = sum(hi.values()) - min(lo.values())
            rows += 1
            tight_b += st_T == wantB
            if st_0T != want0 or st_T < wantB:
                fails.append({"targets": [targets[i] for i in comb], "St_T": st_T, "St_0T": st_0T, "want0": want0,
                              "wantB": wantB})
    out = {"field": "F2", "shape": "<2,2,2>", "L": 1, "subsets": rows, "per_tree_bound_tight": tight_b,
           "failures": len(fails), "first_failures": fails[:3]}
    print(json.dumps(out))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

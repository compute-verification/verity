"""Group-split label certificates for Lemma C_d at any d (note 20261015T0436Z-draft-merges-and-reuse).

Links sit one per cell of a P x Q D-grid, link t adding W_t = sum_j beta_tj Z_j. A group's common-label certificate
takes labels g_t, the entries at t of a basis of ker B_group, on the X side (one column of P per D-column, the
labels of the group's links in it stacked by D-row) or on the Y side (one row per D-row). The words cancel, so the
group saves at most L * gap, gap = |group| - rank P. Part 16 of merges_check.py proves that at d = 2 the links of two
D-rows always split into two groups with total gap <= 2. This script searches, at any d, for a split into at most d
groups with total gap <= rank B, on patterns whose disjoint links are independent (others can't all be helped).

    python merges_split_search.py P Q d p TRIALS SEED [--realized L] [--out DIR]
"""
import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import nullspace, rank


def certificate(links, betas, idx, p, side):
    """(gap, labels U with columns indexed by idx, P) for the group idx with labels on the given side."""
    k = len(idx)
    if not k:
        return 0, None, None
    B = np.array([betas[t] for t in idx], dtype=np.int64).T % p
    U = nullspace(B, k, p)
    assert not (B @ U.T % p).any(), "labels do not cancel the words"
    a = len(U)
    if not a:
        return k, U, None
    nR = max(links[t][0] for t in idx) + 1
    nC = max(links[t][1] for t in idx) + 1
    P = np.zeros((nR * a, nC) if side == "row" else (nR, nC * a), dtype=np.int64)
    for n, t in enumerate(idx):
        R, C = links[t]
        if side == "row":
            P[R * a:(R + 1) * a, C] += U[:, n]
        else:
            P[R, C * a:(C + 1) * a] += U[:, n]
    return k - rank(P % p, p), U, P % p


def disjoint_independent(links, betas, p, d):
    for n in range(2, d + 2):
        for S in itertools.combinations(range(len(links)), n):
            if (len({links[t][0] for t in S}) == n and len({links[t][1] for t in S}) == n
                    and rank(np.array([betas[t] for t in S]), p) < n):
                return False
    return True


def best_split(links, betas, p, groups, target):
    """The smallest total gap over splits into <= groups groups (each on its better side), stopping at target:
    (total, [(group, side)])."""
    k = len(links)
    best = (None, None)
    for assign in itertools.product(range(groups), repeat=k - 1):
        assign = (0,) + assign
        if any(a > max(assign[:n]) + 1 for n, a in enumerate(assign) if n):
            continue
        parts, tot = [], 0
        for g in range(groups):
            idx = [t for t in range(k) if assign[t] == g]
            if not idx:
                continue
            gap, side = min((certificate(links, betas, idx, p, s)[0], s) for s in ("row", "col"))
            parts.append((idx, side))
            tot += gap
            if best[0] is not None and tot >= best[0]:
                break
        else:
            if best[0] is None or tot < best[0]:
                best = (tot, parts)
                if tot <= target:
                    return best
    return best


def label_search(links, betas, p, need, rng, tries):
    """Rank-1 labels M_t = g_t h_t^T with sum_t beta_tj M_t = 0, by alternating solves from shared starts (per
    D-column, per D-row, per random pool, all distinct): the first (g, h) with rank P >= need, or None."""
    k = len(links)
    B = np.array(betas, dtype=np.int64).T % p
    nR, nC = 1 + max(r for r, _ in links), 1 + max(c for _, c in links)

    def solve(fixed):
        A = [[B[j, t] * fixed[t][c] % p for t in range(k)] for j in range(len(B)) for c in range(len(fixed[0]))]
        N = nullspace(np.array(A, dtype=np.int64), k, p)
        return [N[:, t] for t in range(k)] if len(N) else None

    def P_of(g, h):
        a, b = len(g[0]), len(h[0])
        M = np.zeros((a * nR, b * nC), dtype=np.int64)
        for t, (r, c) in enumerate(links):
            M[r * a:(r + 1) * a, c * b:(c + 1) * b] = np.outer(g[t], h[t]) % p
        return M

    for _ in range(tries):
        side = rng.random() < 0.5
        key = rng.choice([[c for _, c in links], [r for r, _ in links],
                          [rng.randrange(rng.randint(1, k)) for _ in range(k)], list(range(k))])
        b = rng.randint(1, k)
        pool, fix = {}, []
        for x in key:
            pool.setdefault(x, np.array([rng.randrange(p) for _ in range(b)], dtype=np.int64))
            fix.append(pool[x])
        for _ in range(4):
            other = solve(fix)
            if other is None:
                break
            g, h = (other, fix) if side else (fix, other)
            if rank(P_of(g, h), p) >= need:
                return g, h, P_of(g, h)
            fix, side = other, not side
    return None


def random_pattern(rng, P, Q, d, p):
    pts = [v for v in itertools.product(range(p), repeat=d) if any(v) and v[next(i for i in range(d) if v[i])] == 1]
    cells = [(R, C) for R in range(P) for C in range(Q) if rng.random() < 0.8]
    pool = rng.sample(pts, rng.randint(d, d + 3))
    betas = []
    for _ in cells:
        s, v = rng.randrange(1, p), rng.choice(pool)
        betas.append(tuple(s * x % p for x in v))
    return cells, betas


def realized(rng, P, Q, L, d, p):
    """Helped links of a span of d random structured words, each cell with its best beta."""
    import merges_check as mc
    shape = mc.Shape(P, Q, L, L)
    cells = [((i, j), 0) for i in range(P) for j in range(Q)]
    Zs = [mc.random_word(shape, cells, rng, p) for _ in range(d)]
    combos = [bs for bs in itertools.product(range(p), repeat=d) if any(bs)]
    links, betas, ranks = [], [], []
    for (c, _) in cells:
        D = shape.D(c, 0)
        r, bs = min((rank((D - sum(b * Z for b, Z in zip(bs, Zs))) % p, p), bs) for bs in combos)
        if r < L:
            links.append(c)
            betas.append(bs)
            ranks.append(r)
    return shape, Zs, links, betas, ranks


def main():
    ap = argparse.ArgumentParser()
    for name in ("P", "Q", "d", "p", "trials", "seed"):
        ap.add_argument(name, type=int)
    ap.add_argument("--realized", type=int, default=0, help="L: realized spans of random words instead of patterns")
    ap.add_argument("--out", default=os.environ.get("OUT", "."))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    hist, tested, over, top = {}, 0, [], 0
    for _ in range(a.trials * 50):
        if tested >= a.trials:
            break
        if a.realized:
            _, _, links, betas, ranks = realized(rng, a.P, a.Q, a.realized, a.d, a.p)
            if len(links) < 2:
                continue
        else:
            links, betas = random_pattern(rng, a.P, a.Q, a.d, a.p)
            if len(links) <= a.d or not disjoint_independent(links, betas, a.p, a.d):
                continue
        rB = rank(np.array(betas), a.p)
        if rB < 2:
            continue
        tested += 1
        tot, _ = best_split(links, betas, a.p, rB, rB)
        hist[f"{rB}:{tot}"] = hist.get(f"{rB}:{tot}", 0) + 1
        if a.realized:
            saved = sum(a.realized - r for r in ranks)
            assert saved <= a.realized * tot, "a certificate is violated"
            top = max(top, saved)
        if tot > rB:
            found = label_search(links, betas, a.p, len(links) - rB, rng, 3000)
            over.append({"links": links, "betas": [list(b) for b in betas], "gap": tot, "labels_reach": bool(found)})
            print("total gap above rank B:", links, betas, tot, "general labels reach k - rank B:", bool(found),
                  flush=True)
        if tested % 20 == 0:
            print(f"  {tested} tested, histogram (rank B: best gap) {hist}", flush=True)
    res = {"P": a.P, "Q": a.Q, "d": a.d, "p": a.p, "realized_L": a.realized, "tested": tested, "histogram": hist,
           "over_rank": over, "top_saving": top}
    tag = f"split_{a.P}x{a.Q}_d{a.d}_p{a.p}" + (f"_L{a.realized}" if a.realized else "")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{tag}.json").write_text(json.dumps(res, indent=1))
    unresolved = sum(not o["labels_reach"] for o in over)
    print(f"{tag}: {tested} tested, histogram (rank B: best gap) {hist}, {len(over)} above rank B "
          f"({unresolved} not reached by general labels either), top saving {top}")


if __name__ == "__main__":
    main()

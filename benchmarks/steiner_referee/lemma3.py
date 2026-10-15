#!/usr/bin/env python3
"""Referee's independent check of Lemma 3 (note:20261015T0416Z-draft-steiner-rigidity-laminar), written from the
statement, not from the author's code.

For a set P of grid cells and a full binary hierarchy on P, type every internal node by the *combinatorial* handle
recursion (leaf: ({r}, {c}); x-node: rows union, columns intersection; y-node: the reverse; type x iff the children's
row handles are disjoint), build the functionals xi, eta, the spaces V_x, V_y, and labels a, b as bases of their
annihilators over GF(p). Then check directly, by rank over GF(p):
  (L) rank Lambda(S) <= 1 for every node S, Lambda(S) = sum_{t in S} a_t b_t^T;
  (I) the a's of each grid row and the b's of each grid column are linearly independent;
and that the typing never sticks (both handle pairs meeting). The fast path checks (I) through
V_x cap F^row = 0 (rank of V_x's generators with that row's coordinates deleted); --full builds the labels.

    python3 lemma3.py random  --rows 4 --cols 4 --n 20000 --primes 2,3,5 --gen all --seed 0 [--full-every 50]
    python3 lemma3.py exhaust --rows 2 --cols 4 [--subsets] --primes 2,3
    python3 lemma3.py mutate  --rows 3 --cols 3 --n 300

Output: one JSON line per job on stdout (counts, failures with their hierarchy), and exit status 1 on any failure.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys

import numpy as np


def rank_mod(m: np.ndarray, p: int) -> int:
    a = np.array(m, dtype=np.int64) % p
    rows, cols = a.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        piv = np.nonzero(a[r:, c])[0]
        if piv.size == 0:
            continue
        i = r + piv[0]
        if i != r:
            a[[r, i]] = a[[i, r]]
        inv = pow(int(a[r, c]), p - 2, p)
        a[r] = (a[r] * inv) % p
        nz = np.nonzero(a[:, c])[0]
        for j in nz:
            if j != r:
                a[j] = (a[j] - a[j, c] * a[r]) % p
        r += 1
    return r


def nullspace_mod(m: np.ndarray, ncols: int, p: int) -> np.ndarray:
    """Basis (as rows) of {s : m s = 0} over GF(p)."""
    if m.shape[0] == 0:
        return np.eye(ncols, dtype=np.int64)
    a = np.array(m, dtype=np.int64) % p
    rows = a.shape[0]
    pivots = []
    r = 0
    for c in range(ncols):
        if r == rows:
            break
        piv = np.nonzero(a[r:, c])[0]
        if piv.size == 0:
            continue
        i = r + piv[0]
        if i != r:
            a[[r, i]] = a[[i, r]]
        inv = pow(int(a[r, c]), p - 2, p)
        a[r] = (a[r] * inv) % p
        for j in np.nonzero(a[:, c])[0]:
            if j != r:
                a[j] = (a[j] - a[j, c] * a[r]) % p
        pivots.append(c)
        r += 1
    free = [c for c in range(ncols) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(ncols, dtype=np.int64)
        v[f] = 1
        for i, pc in enumerate(pivots):
            v[pc] = (-a[i, f]) % p
        basis.append(v)
    return np.array(basis, dtype=np.int64).reshape(len(basis), ncols)


# ---------- hierarchies: list of merges (i, j) over node ids; leaves 0..k-1, merge number m creates node k+m ----------

def gen_uniform(cells, rng):
    k = len(cells)
    roots = list(range(k))
    merges = []
    nxt = k
    while len(roots) > 1:
        i, j = rng.sample(range(len(roots)), 2)
        a, b = roots[i], roots[j]
        merges.append((a, b))
        roots = [x for idx, x in enumerate(roots) if idx not in (i, j)] + [nxt]
        nxt += 1
    return merges


def _rows_cols(cells, members):
    return {cells[t][0] for t in members}, {cells[t][1] for t in members}


def gen_share(cells, rng, greedy=0.8):
    """Prefer merging two current sets that share both rows and columns (the case the typing rule must survive)."""
    k = len(cells)
    roots = {i: frozenset([i]) for i in range(k)}
    merges = []
    nxt = k
    while len(roots) > 1:
        keys = list(roots)
        if rng.random() < greedy:
            best, bs = None, -1
            cand = [(a, b) for a, b in itertools.combinations(keys, 2)]
            rng.shuffle(cand)
            for a, b in cand[:200]:
                ra, ca = _rows_cols(cells, roots[a])
                rb, cb = _rows_cols(cells, roots[b])
                s = min(len(ra & rb), len(ca & cb)) * 4 + len(ra & rb) + len(ca & cb) + rng.random()
                if s > bs:
                    bs, best = s, (a, b)
            a, b = best
        else:
            a, b = rng.sample(keys, 2)
        merges.append((a, b))
        roots[nxt] = roots.pop(a) | roots.pop(b)
        nxt += 1
    return merges


def gen_perm(cells, rng):
    """First group cells into partial permutations (distinct rows and columns: diagonals, Latin-square classes),
    build each group's sub-hierarchy, then merge groups pairwise (groups covering the same rows and columns)."""
    k = len(cells)
    idx = list(range(k))
    rng.shuffle(idx)
    groups = []
    for t in idx:
        r, c = cells[t]
        for g in groups:
            if all(cells[u][0] != r and cells[u][1] != c for u in g) and rng.random() < 0.9:
                g.append(t)
                break
        else:
            groups.append([t])
    merges = []
    nxt = k
    gro = []
    for g in groups:
        roots = list(g)
        while len(roots) > 1:
            i, j = rng.sample(range(len(roots)), 2)
            merges.append((roots[i], roots[j]))
            roots = [x for q, x in enumerate(roots) if q not in (i, j)] + [nxt]
            nxt += 1
        gro.append(roots[0])
    while len(gro) > 1:
        i, j = rng.sample(range(len(gro)), 2)
        merges.append((gro[i], gro[j]))
        gro = [x for q, x in enumerate(gro) if q not in (i, j)] + [nxt]
        nxt += 1
    return merges


def gen_caterpillar(cells, rng):
    k = len(cells)
    order = list(range(k))
    rng.shuffle(order)
    merges = []
    cur = order[0]
    nxt = k
    for t in order[1:]:
        merges.append((cur, t) if rng.random() < 0.5 else (t, cur))
        cur = nxt
        nxt += 1
    return merges


def gen_balanced(cells, rng):
    k = len(cells)
    level = list(range(k))
    rng.shuffle(level)
    merges = []
    nxt = k
    while len(level) > 1:
        new = []
        for i in range(0, len(level) - 1, 2):
            merges.append((level[i], level[i + 1]))
            new.append(nxt)
            nxt += 1
        if len(level) % 2:
            new.append(level[-1])
        rng.shuffle(new)
        level = new
    return merges


GENS = {"uniform": gen_uniform, "share": gen_share, "perm": gen_perm, "caterpillar": gen_caterpillar,
        "balanced": gen_balanced}


# ---------- Lemma 3 ----------

def type_hierarchy(cells, merges, rule="comb"):
    """Returns (types, D, E, members) or raises Stuck. D, E: per node, the sets xi and eta sum over."""
    k = len(cells)
    rho = [frozenset([cells[t][0]]) for t in range(k)]
    gam = [frozenset([cells[t][1]]) for t in range(k)]
    D = [frozenset([t]) for t in range(k)]
    E = [frozenset([t]) for t in range(k)]
    mem = [frozenset([t]) for t in range(k)]
    types = []
    for (a, b) in merges:
        rd = rho[a] & rho[b]
        gd = gam[a] & gam[b]
        if rule == "comb":
            if not rd:
                ty = "x"
            elif not gd:
                ty = "y"
            else:
                raise RuntimeError("stuck")
        elif rule == "always_x":
            ty = "x"
        elif rule == "inverted":
            ty = "y" if not rd else "x"
        else:
            raise ValueError(rule)
        types.append(ty)
        if ty == "x":
            rho.append(rho[a] | rho[b]); gam.append(gam[a] & gam[b])
            D.append(D[a]); E.append(E[a] | E[b])
        else:
            rho.append(rho[a] & rho[b]); gam.append(gam[a] | gam[b])
            D.append(D[a] | D[b]); E.append(E[a])
        mem.append(mem[a] | mem[b])
    return types, D, E, mem


def generators(k, merges, types, D, E):
    gx, gy = [], []
    for m, (a, b) in enumerate(merges):
        v = np.zeros(k, dtype=np.int64)
        if types[m] == "x":
            for t in D[a]:
                v[t] += 1
            for t in D[b]:
                v[t] -= 1
            gx.append(v)
        else:
            for t in E[a]:
                v[t] += 1
            for t in E[b]:
                v[t] -= 1
            gy.append(v)
    return (np.array(gx, dtype=np.int64).reshape(len(gx), k), np.array(gy, dtype=np.int64).reshape(len(gy), k))


def check_fast(cells, merges, p, rule="comb"):
    """(I) through V cap F^line = 0, i.e. deleting a line's coordinates keeps V's rank. Returns None or a reason."""
    k = len(cells)
    try:
        types, D, E, _ = type_hierarchy(cells, merges, rule)
    except RuntimeError:
        return "stuck"
    gx, gy = generators(k, merges, types, D, E)
    for g, axis in ((gx, 0), (gy, 1)):
        if g.shape[0] == 0:
            continue
        rk = rank_mod(g, p)
        for line in sorted({c[axis] for c in cells}):
            keep = [t for t in range(k) if cells[t][axis] != line]
            if rank_mod(g[:, keep], p) != rk:
                return f"(I) fails on {'row' if axis == 0 else 'column'} {line}"
    return None


def check_full(cells, merges, p, rule="comb"):
    """Build a, b and check (L) on every node and (I) on every row and column directly."""
    k = len(cells)
    try:
        types, D, E, mem = type_hierarchy(cells, merges, rule)
    except RuntimeError:
        return "stuck"
    gx, gy = generators(k, merges, types, D, E)
    A = nullspace_mod(gx, k, p).T  # k x d: row t is a_t
    B = nullspace_mod(gy, k, p).T
    for axis, lab in ((0, A), (1, B)):
        for line in sorted({c[axis] for c in cells}):
            ts = [t for t in range(k) if cells[t][axis] == line]
            if lab.shape[1] == 0 or rank_mod(lab[ts, :], p) != len(ts):
                return f"(I) fails on {'row' if axis == 0 else 'column'} {line}"
    for S in mem:
        ts = sorted(S)
        lam = (A[ts, :].T @ B[ts, :]) % p
        if rank_mod(lam, p) > 1:
            return f"(L) fails on node {ts}"
    return None


def grid_cells(rows, cols):
    return [(r, c) for r in range(rows) for c in range(cols)]


def all_hierarchies(k):
    """Every full binary hierarchy on leaves 0..k-1, as merge lists, each unordered tree once: (2k-3)!! of them, by
    inserting leaf n-1 above any node of a tree on n-1 leaves."""
    def trees(n):
        if n == 1:
            yield {0: None}
            return
        for par in trees(n - 1):
            nodes = list(par)
            for e in nodes:  # insert above node e
                q = dict(par)
                new_int = ("i", n)
                q[new_int] = par[e]
                q[e] = new_int
                q[n - 1] = new_int
                yield q
    for par in trees(k):
        children = {}
        for v, pa in par.items():
            if pa is not None:
                children.setdefault(pa, []).append(v)
        ids = {i: i for i in range(k)}
        merges = []
        nxt = [k]

        def build(v):
            if v in ids:
                return ids[v]
            a, b = children[v]
            ia, ib = build(a), build(b)
            merges.append((ia, ib))
            ids[v] = nxt[0]
            nxt[0] += 1
            return ids[v]
        root = [v for v, pa in par.items() if pa is None][0]
        build(root)
        yield merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["random", "exhaust", "mutate"])
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--primes", default="2,3,5")
    ap.add_argument("--gen", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset-prob", type=float, default=0.0, help="random mode: drop each cell with this probability")
    ap.add_argument("--full-every", type=int, default=50)
    ap.add_argument("--subsets", action="store_true", help="exhaust mode: every subset of size >= 2, not just the grid")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args()
    primes = [int(x) for x in args.primes.split(",")]
    rng = random.Random(args.seed)
    grid = grid_cells(args.rows, args.cols)
    out = {"mode": args.mode, "grid": [args.rows, args.cols], "primes": primes, "seed": args.seed}
    fails = []
    if args.mode == "random":
        gens = list(GENS) if args.gen == "all" else args.gen.split(",")
        counts = {g: 0 for g in gens}
        full = 0
        for it in range(args.n):
            cells = [c for c in grid if rng.random() >= args.subset_prob]
            if len(cells) < 2:
                continue
            g = gens[it % len(gens)]
            merges = GENS[g](cells, rng)
            for p in primes:
                why = check_fast(cells, merges, p)
                if why is None and args.full_every and it % args.full_every == 0:
                    why = check_full(cells, merges, p)
                    full += 1
                if why:
                    fails.append({"gen": g, "p": p, "cells": cells, "merges": merges, "why": why})
            counts[g] += 1
        out.update(counts=counts, full_checks=full)
    elif args.mode == "exhaust":
        sets = [grid]
        if args.subsets:
            sets = [list(s) for k in range(2, len(grid) + 1) for s in itertools.combinations(grid, k)]
        total = 0
        idx = -1
        for cells in sets:
            for merges in all_hierarchies(len(cells)):
                idx += 1
                if idx % args.shards != args.shard:
                    continue
                for p in primes:
                    why = check_fast(cells, merges, p)
                    if why:
                        fails.append({"p": p, "cells": cells, "merges": merges, "why": why})
                total += 1
        out.update(hierarchies=total, cell_sets=len(sets), shard=[args.shard, args.shards], enumerated=idx + 1)
    else:  # mutate: the checker must catch broken typings
        caught = {"always_x": 0, "inverted": 0}
        for it in range(args.n):
            merges = gen_share(grid, rng) if it % 2 else gen_perm(grid, rng)
            for rule in caught:
                if check_full(grid, merges, primes[0], rule) is not None:
                    caught[rule] += 1
            why = check_full(grid, merges, primes[0])
            if why:
                fails.append({"cells": grid, "merges": merges, "why": why})
        out.update(n=args.n, mutations_caught=caught)
        if min(caught.values()) == 0:
            fails.append({"why": "a mutation was never caught: the checker has no teeth"})
    out["failures"] = len(fails)
    out["first_failures"] = fails[:3]
    print(json.dumps(out))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

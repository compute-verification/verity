"""Lemma 3 on every binary hierarchy of the full 3x3 grid (the n = 3 case of Conjecture A): 2,027,025 hierarchies.

For each one: the typing rule never gets stuck, and (I) holds over GF(2), GF(3) and GF(10007), i.e. no nonzero
combination of the x-constraints is supported on one grid row and none of the y-constraints on one grid column. (L)
holds for every typing by Claim 1, and check.py verifies it, with the handle recursion, on smaller grids.
Pure integer arithmetic, so the whole enumeration is quick on a few dozen cores.

Usage: exhaustive_3x3.py [WORKERS] [LIMIT] [--full] [--sample N]. LIMIT checks only the first LIMIT hierarchies of
each worker's share; --full also runs check.py's complete per-tree check (labels, (L), (I), handles) over GF(2) and
GF(10007); --sample N checks N random hierarchies (check.random_tree) instead of all of them.
"""

import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check import all_binary_trees, check_tree, random_tree  # noqa: E402

CELLS = [(r, c) for r in range(3) for c in range(3)]
INDEX = {t: i for i, t in enumerate(CELLS)}
ROW = [sum(1 << INDEX[(r, c)] for c in range(3)) for r in range(3)]
COL = [sum(1 << INDEX[(r, c)] for r in range(3)) for c in range(3)]
PRIMES = (3, 10007)


def typing(tree):
    """Bottom-up typing; returns (x-constraints, y-constraints) as (plus mask, minus mask) pairs."""
    xc, yc = [], []

    def go(t):
        if isinstance(t[0], int):
            bit = 1 << INDEX[t]
            return bit, bit, {t[0]}, {t[1]}
        A, B = go(t[0]), go(t[1])
        if not (A[2] & B[2]):
            xc.append((A[0], B[0]))
            return A[0], A[1] | B[1], A[2] | B[2], A[3] & B[3]
        assert not (A[3] & B[3]), "both merges blocked"
        yc.append((A[1], B[1]))
        return A[0] | B[0], A[1], A[2] & B[2], A[3] | B[3]

    go(tree)
    return xc, yc


def rank2(vecs):
    basis = {}
    for v in vecs:
        while v:
            h = v.bit_length() - 1
            if h not in basis:
                basis[h] = v
                break
            v ^= basis[h]
    return len(basis)


def rankp(vecs, p):
    rows = [list(v) for v in vecs]
    rk = 0
    for col in range(9):
        piv = next((i for i in range(rk, len(rows)) if rows[i][col] % p), None)
        if piv is None:
            continue
        rows[rk], rows[piv] = rows[piv], rows[rk]
        inv = pow(rows[rk][col], p - 2, p)
        rows[rk] = [x * inv % p for x in rows[rk]]
        for i in range(len(rows)):
            if i != rk and rows[i][col] % p:
                f = rows[i][col]
                rows[i] = [(x - f * y) % p for x, y in zip(rows[i], rows[rk])]
        rk += 1
    return rk


def signed(pair):
    plus, minus = pair
    return [(1 if plus >> i & 1 else 0) - (1 if minus >> i & 1 else 0) for i in range(9)]


def independent(cons, masks):
    """No nonzero combination of the constraints is supported inside one of the masks."""
    g2 = [a | b for a, b in cons]  # over GF(2), xi_A - xi_B is the indicator of the disjoint union
    r2 = rank2(g2)
    for m in masks:
        if rank2([v & ~m for v in g2]) != r2:
            return False
    sv = [signed(c) for c in cons]
    for p in PRIMES:
        rp = rankp(sv, p)
        for m in masks:
            if rankp([[0 if m >> i & 1 else x for i, x in enumerate(v)] for v in sv], p) != rp:
                return False
    return True


def share(wid, workers, sample):
    if sample:
        rng = random.Random(20261015 + wid)
        return (random_tree(CELLS, rng) for _ in range(sample // workers))
    return (tree for i, tree in enumerate(all_binary_trees(CELLS)) if i % workers == wid)


def work(args):
    wid, workers, limit, full, sample = args
    done = 0
    for tree in share(wid, workers, sample):
        xc, yc = typing(tree)
        assert independent(xc, ROW), f"row independence fails: {tree}"
        assert independent(yc, COL), f"column independence fails: {tree}"
        if full:
            for p in (2, 10007):
                check_tree(tree, p)
        done += 1
        if limit and done >= limit:
            break
    return done


if __name__ == "__main__":
    argv = sys.argv[1:]
    full = "--full" in argv
    sample = int(argv[argv.index("--sample") + 1]) if "--sample" in argv else 0
    args = [a for i, a in enumerate(argv) if a != "--full" and a != "--sample" and (i == 0 or argv[i - 1] != "--sample")]
    workers = int(args[0]) if args else 32
    limit = int(args[1]) if len(args) > 1 else 0
    t0 = time.time()
    with mp.Pool(workers) as pool:
        counts = pool.map(work, [(w, workers, limit, full, sample) for w in range(workers)])
    print(f"[labeling] {'random' if sample else 'exhaustive'} 3x3: {sum(counts)} binary hierarchies on the full 3x3 "
          f"grid: typing never stuck, (I) over GF(2), GF(3), GF(10007){' plus the full per-tree check' if full else ''}: "
          f"ok ({time.time() - t0:.0f} s, {workers} workers)")

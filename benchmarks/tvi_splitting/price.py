"""What a split value costs at its consumer, by brute force on small vectors: the locality lemma's two-party prices.

A value v in Z_q^n (q = 2^b) is split into pieces a (Alice's side of a covering) and b (Bob's), v = a + b, and the
consumer g(v) is computed exactly. Every wire between the two sides is a counted input, so the split costs at least the
communication complexity of g on the pieces' supports S_A x S_B. For each split type and consumer this prints:

- `H_A`, `H_B`: log2 of the supports (the pieces' entropies, uniform on their supports);
- `localize`: min(H_A, H_B), what shipping the smaller piece costs, after which the value is whole in one block;
- `one-way`: log2 of the number of distinct rows of g's matrix on S_A x S_B, the exact cost of Alice -> Bob;
- `log-rank`: log2 of the rank of g's matrix on S_A x S_B, a lower bound on any deterministic two-way protocol
  (every monochromatic rectangle is a rank-one block). Any real encoding of g's outputs gives a valid bound; each
  distinct output is mapped to a random real, which attains the best one generically (packing an output tuple into
  one integer would cap the rank at about n q, a sum of per-coordinate circulants).

    python price.py [--n 4] [--b 2]
"""

from __future__ import annotations

import argparse
import itertools
import math
import random

import numpy as np


def vectors(n, q):
    return list(itertools.product(range(q), repeat=n))


def table(n=4, b=2, seed=3):
    q = 1 << b
    rng = random.Random(seed)
    full = vectors(n, q)
    half = n // 2
    sig = [rng.randrange(q) for _ in range(q)]  # an elementwise nonlinearity: SwiGLU's silu in miniature

    def g_norm(a, bb):  # RMSNorm's inner product, exactly: ||a + b||^2 (its low 2b bits)
        return sum(((x + y) % q) ** 2 for x, y in zip(a, bb)) % (q * q)

    def g_silu(a, bb):  # an elementwise nonlinearity of every coordinate
        return tuple(sig[(x + y) % q] for x, y in zip(a, bb))

    def g_lin(a, bb):  # a linear consumer: Bob's share of W v needs nothing from Alice's
        return 0

    tokens = [tuple(rng.randrange(q) for _ in range(n)) for _ in range(4)]
    disj_a = [v for v in full if all(x == 0 for x in v[half:])]
    disj_b = [v for v in full if all(x == 0 for x in v[:half])]
    lo = [v for v in full if all(x < q // 2 for x in v)]
    hi = [v for v in full if all(x % (q // 2) == 0 for x in v)]
    sub = rng.sample(full, 1 << (n * b // 2))
    splits = [
        ("additive shares, both full range", full, full),
        ("shares, Alice's on 2^(nb/2) points, Bob's full", sub, full),
        ("bit-slice: Alice low planes, Bob high planes", lo, hi),
        ("coordinate-disjoint (tensor-parallel)", disj_a, disj_b),
        ("Bob's piece theta's part of a 2-bit token", full, tokens),
    ]
    enc_rng = np.random.default_rng(seed)
    rows = []
    for consumer, g in (("norm", g_norm), ("silu", g_silu), ("linear", g_lin)):
        for name, SA, SB in splits:
            G = [[g(a, bb) for bb in SB] for a in SA]
            codes = {}
            M = np.array([[codes.setdefault(x, enc_rng.standard_normal()) for x in r] for r in G])
            HA, HB = math.log2(len(SA)), math.log2(len(SB))
            ow = math.log2(len({tuple(r) for r in G}))
            rk = np.linalg.matrix_rank(M)
            rows.append((name, consumer, HA, HB, min(HA, HB), ow, math.log2(rk) if rk else 0.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--b", type=int, default=2)
    args = ap.parse_args()
    print(f"n = {args.n} coordinates of b = {args.b} bits ({args.n * args.b} bits per piece at full range)\n")
    print("| split | consumer | H_A | H_B | localize | one-way | log-rank |")
    print("|---|---|---|---|---|---|---|")
    for name, consumer, HA, HB, loc, ow, lr in table(args.n, args.b):
        print(f"| {name} | {consumer} | {HA:.1f} | {HB:.1f} | {loc:.1f} | {ow:.1f} | {lr:.1f} |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A check of checked_strip.py's P6: pv's cross ranks over F₂.

P6 states that the rows of (word, served coin) have rank (q + 1) + r at every instance, and checks it at shapes whose query
sits at q ≥ r − 1. `PVBlock` puts the query in the diagonal block (k − L ≤ q < k), so q < r − 1 happens only in a
sequence's first block (k = L). There the causal mask leaves q + 1 coins in the diagonal block, and the rank is
(q + 1) + min(q + 1, r). This script computes the ranks with checked_strip's own `PVBlock`, `instance` and `cross_ranks`
at every q of the first block, for r = 8, 16, 32, and at later blocks, where the two formulas agree.

  uv run python benchmarks/pouw/theory/pv_rank_check.py [--quick] [--mutations] [--verdict PATH]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checked_strip as cs
from exact_kit.checks import Checks

KINDS = ("zero", "onehot", "random", "balanced")


def ranks(k, r, L, q, rng):
    B = cs.PVBlock(1, k, 1, r, L, q=[q])
    return B, [cs.cross_ranks(B, *cs.instance(B, kind, rng), mod=1) for kind in KINDS]


def corrected(B, q):
    return q + 1 + B.r * (B.m > 1), q + 1 + min(q + 1, B.r)


def p6(B, q):
    return q + 1 + B.r * (B.m > 1), q + 1 + B.r


def main() -> int:
    checks = Checks()
    quick = "--quick" in checks.args
    rng = np.random.default_rng(20261017)
    first = [(32, r, 32, q) for r in (8, 16, 32) for q in (range(0, 32, 5) if quick else range(32))]
    later = [(64, 32, 32, 32), (64, 32, 32, 63), (64, 16, 32, 40)] + ([] if quick else [(128, 32, 32, 96)])
    rows = []
    for k, r, L, q in first + later:
        B, res = ranks(k, r, L, q, rng)
        rows.append((k, r, L, q, B, res))

    bad = [(k, r, q, [x[:2] for x in res]) for k, r, L, q, B, res in rows
           if not all(x[:2] == corrected(B, q) and x[2] == 0 for x in res)]
    checks.check(f"pv's cross ranks are (q + 1) + r·[m > 1] on the peeled rows and (q + 1) + min(q + 1, r) on the coin "
                 f"rows, with same-side blocks 0, at {len(rows)} (k, r, q) points × {len(KINDS)} instances", not bad,
                 f"mismatches {bad[:4]}")
    p6_fails = sorted((k, r, q) for k, r, L, q, B, res in rows if any(x[:2] != p6(B, q) for x in res))
    below = sorted((k, r, q) for k, r, L, q, B, res in rows if q < r - 1)
    checks.check("P6's (q + 1) + r fails at exactly the points with q < r − 1 (the first block, k = L)",
                 p6_fails == below and all(k == 32 for k, _, _ in below),
                 f"{len(p6_fails)} failing points, e.g. {p6_fails[:3]}")
    k0 = next(res for k, r, L, q, B, res in rows if (k, r, q) == (32, 32, 0))
    checks.check("counterexample: (k, r, L) = (32, 32, 32), q = 0 has ranks (1, 2) at every instance, not P6's (1, 33)",
                 all(x[:2] == (1, 2) for x in k0), str([x[:2] for x in k0]))
    over = {r: sum(r - 1 - q for q in range(r - 1)) for r in (8, 16, 32)}
    checks.check("over-credit of P6 a (head, channel) a sequence is r(r − 1)/2 products (496 at r = 32), about 0.02% of "
                 "pv's ≈ n²/2 at n = 2048", all(v == r * (r - 1) // 2 for r, v in over.items()),
                 f"{over}; {over[32] / (2048 ** 2 / 2):.4%} at n = 2048")
    if checks.mutations:
        Bm = cs.PVBlock(1, 32, 1, 32, 32, two=False, q=[5])
        rm = cs.cross_ranks(Bm, *cs.instance(Bm, "onehot", rng), mod=1)
        checks.mutation("pv without c at q = 5 loses the coin rows' extra rank (corrected formula's 12 not reached)",
                        rm[1] != corrected(Bm, 5)[1], f"ranks {rm[:2]}")
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())

"""Uniform-depth relative tensor-core cost of Strassen and Strassen-Winograd: kappa_r = 8^-r sum_products c(V_A, V_B).

A depth-r product's operand is a +-1 combination of N s8 entries (N the product of the per-level l1 norms), so it ranges
over exactly 255 N + 1 consecutive integers. c is bracketed by Costs.lower (proved) and Costs.exact (an explicit slicing).
"""
from __future__ import annotations

import sys
from collections import Counter
from fractions import Fraction as F

from .costs import Costs, inst

STRASSEN = [(2, 2), (2, 1), (1, 2), (1, 2), (2, 1), (2, 2), (2, 2)]
WINOGRAD = [(1, 1), (1, 1), (4, 1), (1, 4), (2, 2), (3, 3), (2, 2)]


def profile(scheme, r):
    """Counter (N_A, N_B) -> number of depth-r products."""
    prof = Counter({(1, 1): 1})
    for _ in range(r):
        nxt = Counter()
        for (na, nb), cnt in prof.items():
            for ua, ub in scheme:
                nxt[(na * ua, nb * ub)] += cnt
        prof = nxt
    return prof


def kappa(C, scheme, r, bound):
    tot = F(0)
    for (na, nb), cnt in profile(scheme, r).items():
        tot += cnt * bound(255 * na + 1, 255 * nb + 1)
    return tot / 8 ** r


if __name__ == "__main__":
    name = sys.argv[1]
    rmax = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    C = Costs(inst(name))
    lo = C.lower
    up = lambda a, b: C.exact(a, b)[0]
    for sname, sch in (("strassen", STRASSEN), ("winograd", WINOGRAD)):
        for r in range(1, rmax + 1):
            kl, ku = kappa(C, sch, r, lo), kappa(C, sch, r, up)
            print(f"{name} {sname} r={r:2d} lower={float(kl):.6f} upper={float(ku):.6f}"
                  + ("" if kl == ku else "  GAP"), flush=True)

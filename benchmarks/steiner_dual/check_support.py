"""Independent check of a cut-LP solution written by bcr.py (BCR_SUPPORT=FILE): every arc is a rank-1 step over F2,
and for every terminal the arc capacities x carry a unit flow from 0 (integer max-flow after scaling by the common
denominator). Prints sum x, which is then an upper bound on the bidirected cut LP. Usage: check_support.py FILE
"""

import json
import sys
from fractions import Fraction
from math import lcm

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_flow


def f2rank_rows(mask, R, C):
    basis = []
    for row in range(R):
        x = mask >> (row * C) & ((1 << C) - 1)
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
    return len(basis)


def main():
    d = json.load(open(sys.argv[1]))
    n, k, m = d["instance"]
    R, C = n * k, k * m
    arcs = [(a, b, Fraction(v).limit_denominator(10 ** 6)) for a, b, v in d["arcs"]]
    for a, b, _ in arcs:
        assert f2rank_rows(a ^ b, R, C) == 1, (a, b)
    D = lcm(*[x.denominator for _, _, x in arcs])
    nodes = sorted({0, *d["terminals"], *[a for a, _, _ in arcs], *[b for _, b, _ in arcs]})
    ix = {v: i for i, v in enumerate(nodes)}
    cap = {}
    for a, b, x in arcs:
        cap[(ix[a], ix[b])] = cap.get((ix[a], ix[b]), 0) + int(x * D)
    rows, cols = zip(*cap)
    G = sp.csr_matrix((np.array(list(cap.values()), dtype=np.int32), (rows, cols)), shape=(len(nodes), len(nodes)))
    flows = [maximum_flow(G, ix[0], ix[t]).flow_value for t in d["terminals"]]
    total = sum(x for _, _, x in arcs)
    print(f"{len(arcs)} arcs, all rank-1 steps; denominator {D}; max-flow 0 -> each terminal: "
          f"{[Fraction(f, D) for f in flows]}; sum x = {total} (LP reported {d['value']:.9f})")
    assert all(f >= D for f in flows)
    print(f"feasible: the bidirected cut LP of <{n},{k},{m}> over F2 is at most {total}")


if __name__ == "__main__":
    main()

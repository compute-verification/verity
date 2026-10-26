"""ACCEL ALGO-COST item 2 (RT-MODEL M5): block adds a segment for one Strassen level when an MMA may start a chain from
any existing accumulator (D != C), exhaustive over fork forests.

Steady state with each output block's accumulator carried across segments (roots O11..O22). An MMA adds one product to
one existing node, so without adds every node lies on one root path, and the four outputs have four roots. Products 6
and 7 feed one output each and go into its root chain. Products 1..5 hang in a forest rooted at zero (each is MMA'd
once; its node's path vector is the signed sum of it and its ancestors); an output's shared part is an integer
combination of path vectors, each term one block add (an IADD3 takes two). This enumerates every forest on the five
shared products with every sign, and for each output the cheapest combination with coefficients in {-1, 0, 1}; then,
for the best forests, adds shared sub-sums by a breadth-first search over two-term sums used by two outputs.
"""

from __future__ import annotations

import itertools

import numpy as np

SHARED = [1, 2, 3, 4, 5]
NEED = {"O11": {1: 1, 4: 1, 5: -1}, "O12": {3: 1, 5: 1}, "O21": {2: 1, 4: 1}, "O22": {1: 1, 2: -1, 3: 1}}
TARGETS = np.array([[NEED[o].get(p, 0) for p in SHARED] for o in NEED])
COEF = np.array(list(itertools.product((-1, 0, 1), repeat=5)))  # 243 x 5


def forests(n: int):
    """Parent arrays (−1 = root) of every rooted forest on n labelled nodes."""
    for par in itertools.product(range(-1, n), repeat=n):
        ok = True
        for v in range(n):
            seen, u = set(), v
            while u != -1:
                if u in seen or par[u] == v and u == v:
                    ok = False
                    break
                seen.add(u)
                u = par[u]
            if not ok:
                break
        if ok and all(par[v] != v for v in range(n)):
            yield par


def paths(par, sign) -> np.ndarray:
    P = np.zeros((5, 5), dtype=int)
    for v in range(5):
        u = v
        while u != -1:
            P[v, u] = sign[u]
            u = par[u]
    return P


def main() -> None:
    nterms = np.abs(COEF).sum(1)
    best = None
    count = 0
    for par in forests(5):
        for sign in itertools.product((1, -1), repeat=5):
            count += 1
            P = paths(par, sign)
            combos = COEF @ P  # 243 x 5 reachable vectors
            adds = iadd3 = 0
            for t in TARGETS:
                hit = np.all(combos == t, axis=1)
                if not hit.any():
                    adds = None
                    break
                k = int(nterms[hit].min())
                adds += k
                iadd3 += -(-k // 2)
            if adds is None:
                continue
            key = (adds, iadd3)
            if best is None or key < best[0]:
                best = (key, [(par, sign)])
            elif key == best[0] and len(best[1]) < 5:
                best[1].append((par, sign))
    print(f"{count} signed forests on products 1..5")
    (adds, iadd3), ex = best
    print(f"min block adds a segment without shared sub-sums: {adds} two-input adds, {iadd3} IADD3 (per output, separately)")
    for par, sign in ex[:3]:
        print(f"   parents {par} signs {sign}")
    best3 = None
    for par in forests(5):
        for sign in itertools.product((1, -1), repeat=5):
            P = paths(par, sign)
            combos = COEF @ P
            tot = 0
            for t in TARGETS:
                hit = np.all(combos == t, axis=1)
                if not hit.any():
                    tot = None
                    break
                tot += -(-int(nterms[hit].min()) // 2)
            if tot is not None and (best3 is None or tot < best3[0]):
                best3 = (tot, par, sign)
    print(f"min IADD3 a segment (per output, separately): {best3[0]} at parents {best3[1]} signs {best3[2]}")


if __name__ == "__main__":
    main()

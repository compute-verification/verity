"""Deterministic witnesses for claims in note 20261015T0437Z-draft-steiner-rigidity-dual (CMP-8 worker A1b).

1. The single-cell exchange inequality does not sum over cells: a 9x9 Z over F2 (3x3 grid of 3x3 cells) with
   sum_t max(0, rank Z - rank Z_t^o) = rank Z + 1, where Z_t^o is Z with cell t set to 0.
2. Strassen mod 2 is a set decomposition of <2,2,2> with 7 terms (the obstruction to laminarity-blind certificates).
"""

import numpy as np

import fq

Z = np.array([
    [1, 1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0]])


def cellsum():
    rz = int(fq.rank(Z, 2)[0])
    tot = 0
    for r in range(3):
        for c in range(3):
            Zo = Z.copy()
            Zo[3 * r:3 * r + 3, 3 * c:3 * c + 3] = 0
            tot += max(0, rz - int(fq.rank(Zo, 2)[0]))
    return rz, tot


def strassen_mod2():
    T = fq.matmul_targets(2)
    # x index r*2+k (A_rk), y index k*2+c (B_kc)
    def form(a, b):
        u = np.zeros(4, dtype=np.int64)
        v = np.zeros(4, dtype=np.int64)
        for (r, k) in a:
            u[r * 2 + k] ^= 1
        for (k, c) in b:
            v[k * 2 + c] ^= 1
        return np.outer(u, v) % 2
    M = [form([(0, 0), (1, 1)], [(0, 0), (1, 1)]), form([(1, 0), (1, 1)], [(0, 0)]),
         form([(0, 0)], [(0, 1), (1, 1)]), form([(1, 1)], [(1, 0), (0, 0)]),
         form([(0, 0), (0, 1)], [(1, 1)]), form([(1, 0), (0, 0)], [(0, 0), (0, 1)]),
         form([(0, 1), (1, 1)], [(1, 0), (1, 1)])]
    sets = {(0, 0): [0, 3, 4, 6], (0, 1): [2, 4], (1, 0): [1, 3], (1, 1): [0, 1, 2, 5]}
    for pos, idx in sets.items():
        assert np.array_equal(sum(M[i] for i in idx) % 2, T[pos]), pos
    return len(M)


if __name__ == "__main__":
    rz, tot = cellsum()
    print("cellwise exchange sum:", tot, "> rank Z =", rz)
    assert tot == rz + 1
    print("Strassen mod 2 set decomposition of <2,2,2>:", strassen_mod2(), "terms")

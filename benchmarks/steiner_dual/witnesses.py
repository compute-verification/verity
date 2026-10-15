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


def strassen_products_any_field():
    """[(R_i, S_i)]: the 7 integer rank-1 forms and their 0/1 output sets of strassen_sets_any_field."""
    out = []
    strassen_sets_any_field(0, out)
    return out


def strassen_blocks_any_field():
    """[(a, b, S_i)]: product i is (sum_rk a[r,k] X_rk)(sum_kj b[k,j] Y_kj), integer 2x2 coefficient arrays with the
    scalar absorbed into a, feeding the output cells S_i with coefficient 1. Block form of
    strassen_products_any_field (its R_i is outer(a.ravel(), b.ravel()))."""
    out = []
    for R, S in strassen_products_any_field():
        rows = [i for i in range(4) if R[i].any()]
        cols = [j for j in range(4) if R[:, j].any()]
        b = R[rows[0]] // np.gcd.reduce(R[rows[0]][R[rows[0]] != 0])
        a = R[:, cols[0]] // b[cols[0]]
        assert np.array_equal(np.outer(a, b), R)
        out.append((a.reshape(2, 2), b.reshape(2, 2), S))
    return out


def strassen_sets_any_field(p, out=None):
    """Strassen transformed by W -> W R (R = [[1,0],[1,1]]) on the output side: every output pattern is a scalar
    multiple of a 0/1 set, so <2,2,2> = sum_i R_i (x) 1_{S_i} with 7 rank-1 R_i over every field (checked over the
    integers mod p, p = 0 meaning exact integers). The sets are not laminar ({11,21} and {21,22})."""
    Rm = np.array([[1, 0], [1, 1]])
    Rinv = np.array([[1, 0], [-1, 1]])
    W = [np.array(w) for w in ([[1, 0], [0, 1]], [[0, 0], [1, -1]], [[0, 1], [0, 1]], [[1, 0], [1, 0]],
                               [[-1, 1], [0, 0]], [[0, 0], [0, 1]], [[1, 0], [0, 0]])]
    # Strassen's products as (u over A_rk, v over B_kc) with signs; A = X, B = Y R^{-1}
    prods = [({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}), ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
             ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}), ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}),
             ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}), ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}),
             ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1})]
    T = fq.matmul_targets(2)
    total = {pos: np.zeros((4, 4), dtype=np.int64) for pos in T}
    sets = []
    for (ua, vb), w in zip(prods, W):
        u = np.zeros(4, dtype=np.int64)
        for (r, k), s in ua.items():
            u[r * 2 + k] += s
        # v over B_kc with B = Y R^{-1}: B_kc = sum_j Y_kj Rinv[j, c]  ->  coefficient of y_kj
        v = np.zeros(4, dtype=np.int64)
        for (k, c), s in vb.items():
            for j in range(2):
                v[k * 2 + j] += s * Rinv[j, c]
        wp = w @ Rm
        nz = wp[wp != 0]
        assert len(set(nz.tolist())) == 1, wp
        lam = int(nz[0])
        S = {(r, c) for r in range(2) for c in range(2) if wp[r, c]}
        sets.append(sorted(S))
        Rk = lam * np.outer(u, v)
        if out is not None:
            out.append((Rk, S))
        for pos in S:
            total[pos] = total[pos] + Rk
    for pos in T:
        d = total[pos] - T[pos]
        assert not (d % p if p else d).any(), pos
    return sets


if __name__ == "__main__":
    rz, tot = cellsum()
    print("cellwise exchange sum:", tot, "> rank Z =", rz)
    assert tot == rz + 1
    print("Strassen mod 2 set decomposition of <2,2,2>:", strassen_mod2(), "terms")
    for p in (0, 2, 3, 5):
        sets = strassen_sets_any_field(p)
    print("7-term set decomposition of <2,2,2> over Z (hence every field):", sets)

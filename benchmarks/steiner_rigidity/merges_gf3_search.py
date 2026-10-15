"""Exhaustive search over GF(3) at L = 2 for d*L at d = 2 (CMP-8, A3; note 20261015T0436Z-draft-merges-and-reuse).

`python merges_gf3_search.py P Q [--out DIR]`: every 2-dimensional space V of words on the P x Q D-grid (2P x 2Q
matrices) in which all four directions of the pencil help some cell, with the best element of V per cell; prints the
largest total saving and writes `gf3_P_Q.json` to DIR (default $OUT, else the working directory). d*L says it is at
most 2L = 4.

At L = 2 a word w helps cell t iff w = D_t + R with rank R <= 1, saving 2 - rank R (a word may help two cells of one
line). So a space's total is read off the membership of its 8 nonzero elements in the helpful set H.
The first basis word is a helpful word of cell (0, 0) (every space that helps anything has one, up to permuting block
rows and columns), taken up to conjugation by GL_2(3) on every block and the permutations that fix (0, 0), all of
which fix every D_t; the second ranges over H. Spaces where at most three directions help save at most 2L by Lemma C
and the three-direction case split (merges_check.py part 14), so only four-direction spaces are scored. On 2 x Q grids
it also counts the open case of part 14: helped cells sharing a D-column (else part 8 applies), generic rank below 2L
(over GF(9)), and no three directions reaching every helped cell's best saving.
"""

import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

P3 = 3


def add3(a1, a2, b1, b2):
    """GF(3) addition on bit planes (bit e of plane 1 / plane 2: entry e is 1 / 2)."""
    t = (a1 | b2) ^ (a2 | b1)
    return (a2 | b2) ^ t, (a1 | b1) ^ t


def planes(M):
    """(plane 1, plane 2) of a batch of matrices (n, R, C) over GF(3), as uint64."""
    flat = M.reshape(len(M), -1)
    w = (np.uint64(1) << np.arange(flat.shape[1], dtype=np.uint64))
    return (((flat == 1) * w).sum(axis=1, dtype=np.uint64), ((flat == 2) * w).sum(axis=1, dtype=np.uint64))


def unit(P, Q, t):
    D = np.zeros((2 * P, 2 * Q), dtype=np.int8)
    for k in range(2):
        D[2 * t[0] + k, 2 * t[1] + k] = 1
    return D


def rank_le1(nr, nc):
    """All matrices of rank <= 1 over GF(3), nr x nc."""
    us = [np.array(u) for u in itertools.product(range(3), repeat=nr) if next((x for x in u if x), 0) == 1]
    vs = [np.array(v) for v in itertools.product(range(3), repeat=nc) if any(v)]
    out = [np.zeros((nr, nc), dtype=np.int8)] + [(np.outer(u, v) % 3).astype(np.int8) for u in us for v in vs]
    return np.array(out)


def gl2():
    out = []
    for a, b, c, d in itertools.product(range(3), repeat=4):
        det = (a * d - b * c) % 3
        if det:
            inv = pow(det, 1, 3)  # det^-1 = det over GF(3)
            out.append((np.array([[a, b], [c, d]]), np.array([[d, -b], [-c, a]]) * inv % 3))
    return out


def act(M, G, Gi, rp, cp, P, Q):
    """(I_P (x) G) M (I_Q (x) G^-1), then block rows by rp and block columns by cp, on a batch (n, 2P, 2Q)."""
    B = M.reshape(len(M), P, 2, Q, 2)
    B = np.einsum("ik,npkql,lj->npiqj", G, B, Gi) % 3
    B = B[:, list(rp)][:, :, :, list(cp)]
    return B.reshape(len(M), 2 * P, 2 * Q)


class Helpful:
    """The helpful set H with each word's cell and saving, searchable by a 64-bit hash of its two planes."""

    def __init__(self, P, Q):
        R1 = rank_le1(2 * P, 2 * Q)
        rk = np.array([0] + [1] * (len(R1) - 1))
        cells = [(i, j) for i in range(P) for j in range(Q)]
        p1, p2 = (np.concatenate(x) for x in zip(*(planes((unit(P, Q, t) + R1) % 3) for t in cells)))
        keys, inv = np.unique(np.stack([p1, p2], axis=1), axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        self.sav = np.zeros((len(keys), len(cells)), dtype=np.int8)
        self.sav[inv, np.repeat(np.arange(len(cells)), len(R1))] = np.tile(2 - rk, len(cells))
        self.p1, self.p2 = keys[:, 0], keys[:, 1]
        h = self.hash(self.p1, self.p2)
        order = np.argsort(h)
        self.h, self.p1, self.p2, self.sav = h[order], self.p1[order], self.p2[order], self.sav[order]
        assert len(np.unique(self.h)) == len(self.h), "hash collision inside H"
        self.n_cells = len(cells)

    @staticmethod
    def hash(p1, p2):
        with np.errstate(over="ignore"):
            return p1 * np.uint64(0x9E3779B97F4A7C15) ^ (p2 * np.uint64(0xC2B2AE3D27D4EB4F) + np.uint64(0x165667B1))

    def find(self, p1, p2):
        """Index into H of each (p1, p2), or -1."""
        h = self.hash(p1, p2)
        i = np.minimum(np.searchsorted(self.h, h), len(self.h) - 1)
        ok = (self.h[i] == h) & (self.p1[i] == p1) & (self.p2[i] == p2)
        return np.where(ok, i, -1)


def canonical_first(P, Q):
    """Representatives of the helpful words of cell (0, 0) up to the symmetries that fix every D_t and (0, 0)."""
    R1 = rank_le1(2 * P, 2 * Q)
    W = (unit(P, Q, (0, 0)) + R1) % 3
    best = None
    for (G, Gi), rp, cp in itertools.product(gl2(), itertools.permutations(range(1, P)),
                                             itertools.permutations(range(1, Q))):
        img = act(W, G, Gi, (0,) + rp, (0,) + cp, P, Q)
        a, b = planes(img)
        key = np.stack([a, b], axis=1)
        if best is None:
            best = key
        else:
            swap = (key[:, 0] < best[:, 0]) | ((key[:, 0] == best[:, 0]) & (key[:, 1] < best[:, 1]))
            best[swap] = key[swap]
    reps = np.unique(best, axis=0)
    return reps[:, 0], reps[:, 1]


def decode(p1, p2, P, Q):
    e, one = np.arange(4 * P * Q, dtype=np.uint64), np.uint64(1)
    bits = ((np.uint64(p1) >> e) & one) + 2 * ((np.uint64(p2) >> e) & one)
    return bits.astype(np.int64).reshape(2 * P, 2 * Q)


def _mul9(x, y):
    return (x[0] * y[0] - x[1] * y[1]) % 3, (x[0] * y[1] + x[1] * y[0]) % 3


def rank_gf9(M):
    """Rank over GF(9) = GF(3)[i]/(i^2 + 1) of a matrix of pairs (a, b) = a + b i."""
    M, r = [list(row) for row in M], 0
    units = {x: y for x in itertools.product(range(3), repeat=2) for y in itertools.product(range(3), repeat=2)
             if _mul9(x, y) == (1, 0)}
    for c in range(len(M[0])):
        piv = next((i for i in range(r, len(M)) if M[i][c] != (0, 0)), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        M[r] = [_mul9(units[M[r][c]], x) for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != (0, 0):
                f = M[i][c]
                M[i] = [tuple((u - v) % 3 for u, v in zip(x, _mul9(f, y))) for x, y in zip(M[i], M[r])]
        r += 1
    return r


def generic_rank(Z1, Z2):
    """Rank of Z1 + t Z2 over GF(3)(t): its largest rank at the six points of GF(9) outside GF(3), enough for a
    nonzero minor of degree <= 4 = 2L."""
    return max(rank_gf9([[((int(x) + a * int(y)) % 3, b * int(y) % 3) for x, y in zip(r1, r2)] for r1, r2 in zip(Z1, Z2)])
               for a in range(3) for b in (1, 2))


def needs_four(dsav, best):
    """Whether no three directions reach every helped cell's best saving (dsav: direction x cell)."""
    sets = [set(np.nonzero(dsav[:, c] == best[c])[0]) for c in range(len(best)) if best[c] > 0]
    return not any(all(s & set(T) for s in sets) for T in itertools.combinations(range(4), 3))


def total_of(H, idx):
    """Total saving of spaces given, per space, the H-index (or -1) of each of its 8 nonzero elements."""
    best = np.zeros((idx.shape[0], H.n_cells), dtype=np.int64)
    for e in range(idx.shape[1]):
        j = idx[:, e]
        best = np.maximum(best, np.where((j >= 0)[:, None], H.sav[np.maximum(j, 0)].astype(np.int64), 0))
    return best.sum(axis=1)


def search(P, Q, verify=20, log=print):
    """The search on the P x Q grid; re-checks the totals of `verify` spaces by ranks (merges_check.pencil_profile)."""
    t0 = time.time()
    H = Helpful(P, Q)
    a1s, a2s = canonical_first(P, Q)
    log(f"{P}x{Q}: |H| = {len(H.h)}, {len(a1s)} first words up to symmetry ({time.time() - t0:.1f} s)")
    hist, top, examples, n_four, verified = {}, 0, [], 0, 0
    opened = {"shared_column": 0, "rank_below_2L": 0, "needs_four_directions": 0}
    open_top = dict.fromkeys(opened, 0)
    for n, (a1, a2) in enumerate(zip(a1s, a2s)):
        b1, b2 = H.p1, H.p2
        s1, s2 = add3(a1, a2, b1, b2)
        d1, d2 = add3(a1, a2, b2, b1)
        iS, iSn, iD, iDn = H.find(s1, s2), H.find(s2, s1), H.find(d1, d2), H.find(d2, d1)
        dup = ((b1 == a1) & (b2 == a2)) | ((b1 == a2) & (b2 == a1))
        m = ((iS >= 0) | (iSn >= 0)) & ((iD >= 0) | (iDn >= 0)) & ~dup
        if not m.any():
            continue
        k = np.nonzero(m)[0]
        iA, iAn = H.find(np.array([a1]), np.array([a2]))[0], H.find(np.array([a2]), np.array([a1]))[0]
        idx = np.stack([np.full(len(k), iA), np.full(len(k), iAn), k, H.find(b2[k], b1[k]),
                        iS[k], iSn[k], iD[k], iDn[k]], axis=1)
        tot = total_of(H, idx)
        for j in (0, int(np.argmax(tot)))[: max(0, verify - verified)]:
            from merges_check import Shape, pencil_profile
            Z1, Z2 = decode(a1, a2, P, Q), decode(b1[k[j]], b2[k[j]], P, Q)
            _, best = pencil_profile(Shape(P, Q, 2, 2), Z1, Z2, 3)
            assert sum(s for s, _ in best.values()) == tot[j], "bit-plane total disagrees with ranks"
            verified += 1
        n_four += len(k)
        if P == 2:
            sav = np.where((idx >= 0)[:, :, None], H.sav[np.maximum(idx, 0)].astype(np.int64), 0)
            dsav = sav.reshape(len(k), 4, 2, -1).max(axis=2)
            best = dsav.max(axis=1).reshape(len(k), P, Q)
            for j in np.nonzero(((best > 0).sum(axis=1) == 2).any(axis=1))[0]:
                Z1, Z2 = decode(a1, a2, P, Q), decode(b1[k[j]], b2[k[j]], P, Q)
                steps = ("shared_column", "rank_below_2L", "needs_four_directions")
                ok = (True, generic_rank(Z1, Z2) < 4, needs_four(dsav[j], best[j].reshape(-1)))
                for key, _ in itertools.takewhile(lambda x: x[1], zip(steps, ok)):
                    opened[key] += 1
                    open_top[key] = max(open_top[key], int(tot[j]))
        for v, c in zip(*np.unique(tot, return_counts=True)):
            hist[int(v)] = hist.get(int(v), 0) + int(c)
        if tot.max() > top:
            top = int(tot.max())
        for j in np.nonzero(tot > 4)[0][:5 - len(examples)]:
            examples.append({"a": [int(a1), int(a2)], "b": [int(b1[k[j]]), int(b2[k[j]])], "total": int(tot[j])})
        if n % 50 == 0:
            log(f"  {n}/{len(a1s)} first words, {n_four} four-direction pairs, top {top}, {time.time() - t0:.0f} s")
    assert top <= 4 or examples, "inconsistent"
    return {"P": P, "Q": Q, "L": 2, "field": 3, "helpful": len(H.h), "first_words": len(a1s),
            "four_direction_pairs": n_four, "histogram": hist, "max_total": top, "bound_2L": 4,
            "counterexamples": examples, "verified_by_ranks": verified, "seconds": round(time.time() - t0, 1),
            "two_rows_open_case": {"counts": opened, "max_total": open_top} if P == 2 else None}


def main():
    P, Q = int(sys.argv[1]), int(sys.argv[2])
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path(os.environ.get("OUT", "."))
    res = search(P, Q, log=lambda s: print(s, flush=True))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"gf3_{P}_{Q}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()

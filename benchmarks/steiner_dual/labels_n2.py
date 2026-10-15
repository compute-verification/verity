"""Labelled flattening with optimised labels on <2,2,2> over F2 (note 20261015T0437Z-draft-steiner-rigidity-dual).

For output sets S_1..S_m (subsets of the 4 cells) and labels a_t, b_t in F2^d with a's independent in each grid row
and b's in each grid column, rank Phi = 8 <= sum_e rank Lambda(S_e), Lambda(S) = sum_{t in S} a_t b_t^T. This finds
min over labels of sum_e rank Lambda(S_e) by exhaustion (d = 2 and 3), for
  - the 7-term set decompositions of <2,2,2> over F2 (the 6 output sides of Strassen's orbit), and
  - every output side of a one-merge DAG with 7 products (merges.structures), every labelling onto the grid,
    also with labels zero off a sub-family P' of the targets (the bound is then 2|P'|).
Usage: labels_n2.py [d] [--three]   (d = 2, 3, 4 give the same counts; --three adds the 2-minute rank check of the
three outputs C12, C21, C22)
"""

import itertools
import sys

import numpy as np

import merges

ROWS = [(0, 1), (2, 3)]
COLS = [(0, 2), (1, 3)]


def label_table(d):
    """(label choices) x (16 subsets): rank over F2 of Lambda(S). GL_d on the a's and on the b's keeps every rank
    and (I), so a_11, a_12 = e1, e2 and b_11, b_21 = e1, e2 lose nothing; over F2 two nonzero vectors are
    independent iff distinct."""
    vecs = np.array([[v >> i & 1 for i in range(d)] for v in range(1, 1 << d)], dtype=np.int64)
    nv = len(vecs)
    pairs = [(i, j) for i in range(nv) for j in range(nv) if i != j]
    e1, e2 = 0, 1
    ca = np.array([(e1, e2, i, j) for (i, j) in pairs], dtype=np.int64)
    cb = np.array([(e1, i, e2, j) for (i, j) in pairs], dtype=np.int64)
    ia, ib = np.meshgrid(np.arange(len(ca)), np.arange(len(cb)), indexing="ij")
    ia, ib = ia.ravel(), ib.ravel()
    A = vecs[ca[ia]]
    B = vecs[cb[ib]]
    outer = A[:, :, :, None] * B[:, :, None, :]
    table = np.zeros((len(ia), 16), dtype=np.int8)
    for S in range(1, 16):
        L = sum(outer[:, t] for t in range(4) if S >> t & 1) % 2
        table[:, S] = merges.fq.rank(L, 2) if len(L) else 0
    return table


def best(table, sets):
    return int(table[:, list(sets)].sum(axis=1, dtype=np.int64).min())


def subfamily_tables(d):
    """For every nonempty sub-family P' of the cells: labels nonzero exactly on P', (I) inside P' (Theorem 1 holds
    for any set of targets, so rank Phi' = 2|P'|). Returns [(P', table)], table as in label_table."""
    vecs = np.array([[v >> i & 1 for i in range(d)] for v in range(1 << d)], dtype=np.int64)
    nv = len(vecs)
    out = []
    for Pm in range(1, 16):
        cells = [t for t in range(4) if Pm >> t & 1]

        def choices(pairs):
            # GL_d: the first cell of P' gets e1 (index 1), its mate in a pair inside P' gets e2 (index 2)
            first = cells[0]
            mate = next((t for (s, t) in pairs if s == first and t in cells), None)
            mate = mate if mate is not None else next((s for (s, t) in pairs if t == first and s in cells), None)
            opts = []
            for t in range(4):
                if t not in cells:
                    opts.append([0])
                elif t == first:
                    opts.append([1])
                elif t == mate:
                    opts.append([2])
                else:
                    opts.append(range(1, nv))
            c = np.array(list(itertools.product(*opts)), dtype=np.int64)
            ok = np.ones(len(c), dtype=bool)
            for (s, t) in pairs:
                if s in cells and t in cells:
                    ok &= c[:, s] != c[:, t]
            return c[ok]

        ca = choices(ROWS)
        cb = choices(COLS)
        ia, ib = np.meshgrid(np.arange(len(ca)), np.arange(len(cb)), indexing="ij")
        ia, ib = ia.ravel(), ib.ravel()
        outer = vecs[ca[ia]][:, :, :, None] * vecs[cb[ib]][:, :, None, :]
        table = np.zeros((len(ia), 16), dtype=np.int8)
        for S in range(1, 16):
            L = sum(outer[:, t] for t in range(4) if S >> t & 1) % 2
            table[:, S] = merges.fq.rank(L, 2)
        out.append((Pm, table))
    return out


def excluded(subtables, sets):
    """Smallest slack sum_e rank Lambda(S_e) - 2|P'| over sub-families and labels (< 0: the side is excluded)."""
    return min(best(t, sets) - 2 * bin(Pm).count("1") for Pm, t in subtables)


def main():
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    table = label_table(d)
    print(f"d = {d}: {len(table)} label choices")
    outs = {tuple(sorted(w for (_, _, w) in dec)) for dec in merges.orbit()}
    for o in sorted(outs):
        print("decomposition output side", [bin(w)[2:].zfill(4)[::-1] for w in o], "min sum rank:", best(table, o))
    perms = list(itertools.permutations(range(4)))
    fams = set()
    for par, N, B, tg, lam in merges.structures():
        if 0 in lam or merges.f2rank(lam) < 4:
            continue
        for s in perms:
            fams.add(tuple(sorted(merges.relabel(w, s) for w in lam)))
    scores = {f: best(table, f) for f in fams}
    hist = {}
    for v in scores.values():
        hist[v] = hist.get(v, 0) + 1
    print(f"one-merge output sides, labelled onto the grid: {len(fams)}; min sum rank -> count: {dict(sorted(hist.items()))}"
          " (<= 7: the flattening alone excludes the side)")
    sub = subfamily_tables(d)
    slack = {f: excluded(sub, f) for f in fams}
    left = sorted(f for f, v in slack.items() if v >= 0)
    print(f"with sub-families P' (labels in F2^{d}): {len(fams) - len(left)} excluded, {len(left)} not")
    for f in left[:3]:
        print("  not excluded:", [bin(w)[2:].zfill(4)[::-1] for w in f], "slack", slack[f])
    print("decomposition output sides, slack with sub-families:", sorted(excluded(sub, o) for o in outs))
    # tensor ranks over F2 of <2,2,2> restricted to 1, 2, 3, 4 cells: 2, 4, 6 (three_cell_rank), 7
    rk = {1: 2, 2: 4, 3: 6, 4: 7}
    counted = sum(any(sum(1 for w in f if w & P) < rk[bin(P).count("1")] for P in range(1, 16)) for f in left)
    print(f"of those {len(left)}, {counted} touch some set of cells with fewer products than its restricted rank; "
          f"the other {len(left) - counted} only the direct solve (merges.solvable) excludes")
    if "--three" in sys.argv:
        print("3-output restriction (C12, C21, C22) over F2: 5-term output sides that work:", three_cell(5),
              "; 6-term:", three_cell(6))


def three_cell(r):
    """Number of r-term output sides (vectors in F2^3) of a decomposition of (C12, C21, C22) over F2 (2 min for r=6)."""
    T = [merges.TGT[1], merges.TGT[2], merges.TGT[3]]
    R1 = np.array(merges.RANK1, dtype=np.int64)
    found = 0
    for lam in itertools.combinations_with_replacement(range(1, 8), r):
        L = np.array([[w >> t & 1 for w in lam] for t in range(3)], dtype=np.int64)
        P = next((P for P in itertools.combinations(range(r), 3) if merges.f2rank([lam[i] for i in P]) == 3), None)
        if P is None:
            continue
        F = [i for i in range(r) if i not in P]
        aug = np.concatenate([L[:, list(P)], np.eye(3, dtype=np.int64)], axis=1)
        for col in range(3):
            piv = next(q for q in range(col, 3) if aug[q, col])
            aug[[col, piv]] = aug[[piv, col]]
            for q in range(3):
                if q != col and aug[q, col]:
                    aug[q] ^= aug[col]
        Bi = aug[:, 3:]
        c = Bi @ L[:, F] % 2
        grids = [g.ravel() for g in np.meshgrid(*([R1] * len(F)), indexing="ij")]
        ok = np.ones(grids[0].shape, dtype=bool)
        for j in range(3):
            v = np.full(ok.shape, 0, dtype=np.int64)
            for t in range(3):
                if Bi[j, t]:
                    v ^= T[t]
            for f in range(len(F)):
                if c[j, f]:
                    v = v ^ grids[f]
            ok &= merges.IS_RANK1[v]
        found += bool(ok.any())
    return found


if __name__ == "__main__":
    main()

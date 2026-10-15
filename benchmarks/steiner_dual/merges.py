"""Merges on <2,2,2> (note 20261015T0437Z-draft-steiner-rigidity-dual, CMP-8 worker A1b).

1. two_merge_schedule(p): 7 products and 2 merges compute <2,2,2>, exactly over the integers (so over every field):
   the products of witnesses.strassen_products_any_field on the DAG
     0 -M3-> N3 -M5-> C12,  N3 -M1-> N13 -M4-> N134 -M7-> C11,  N13 -M2-> N132,  0 -M6-> N6,
     C21 = N134 + N6,  C22 = N132 + N6.
   strassen_dag(n, depth) builds it on n/2-blocks (chained products as naive chains on their offsets, recursing into
   the two products that start from 0) and counts products, merges and the sum over merges of the smaller summand's
   rank, the loss A1a's Corollary 4 allows: always exactly twice the saving.
2. one_merge_search(): over F2, no DAG with 7 product steps and one merge computes <2,2,2>. Every DAG with one merge
   is a rooted tree H plus one extra arc B -> N; a product's output set is then its H-subtree's targets, XOR N's
   targets when the product lies on H's path to B. Its 7 output sets must be the output side of a rank-7
   decomposition. Two independent ways of knowing those: the orbit of Strassen under the symmetries of <2,2,2>
   (de Groote: that is all of them), and a direct solve, for each candidate output side, for 7 rank-1 forms.
"""

import itertools
import sys

import numpy as np

import fq
import witnesses

CELLS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def two_merge_schedule(p):
    prods = witnesses.strassen_products_any_field()
    M = {i + 1: R for i, (R, _) in enumerate(prods)}
    N = {}
    N[3] = M[3]
    C12 = N[3] + M[5]
    N[13] = N[3] + M[1]
    N[134] = N[13] + M[4]
    C11 = N[134] + M[7]
    N[132] = N[13] + M[2]
    N[6] = M[6]
    C21 = N[134] + N[6]
    C22 = N[132] + N[6]
    T = fq.matmul_targets(2)
    for pos, val in ((0, 0), C11), ((0, 1), C12), ((1, 0), C21), ((1, 1), C22):
        d = val - T[pos]
        assert not (d % p if p else d).any(), pos
    return 7, 2


class Builder:
    """Accumulate-and-merge DAGs for <n,n,n> over Z: node values are integer forms (x index r*n+j, y index j*n+c)."""

    def __init__(self, n):
        self.n = n
        self.products = 0
        self.merges = 0
        self.weight = 0

    def naive(self, P, Q, O):
        """Cells O[i][j] + sum_k P[i][k] Q[k][j], each a chain of rank-1 steps from its offset node."""
        b = len(P)
        out = [[None] * b for _ in range(b)]
        for i in range(b):
            for j in range(b):
                v = np.zeros((self.n ** 2, self.n ** 2), dtype=np.int64) if O is None else O[i][j]
                for k in range(b):
                    v = v + np.outer(P[i][k], Q[k][j])
                    self.products += 1
                out[i][j] = v
        return out

    def merge(self, A, B):
        out = []
        for ra, rb in zip(A, B):
            row = []
            for a, b in zip(ra, rb):
                ka, kb = fq.rank(np.stack([a, b]), 10007)
                self.merges += 1
                self.weight += int(min(ka, kb))
                row.append(a + b)
            out.append(row)
        return out

    def strassen(self, P, Q, depth):
        """P Q from 0: one level of the two-merge schedule, recursing only into M3 and M6 (the products that start
        from 0); the chained products M5, M1, M4, M7, M2 are naive chains on their offsets."""
        b = len(P)
        if depth == 0 or b == 1:
            return self.naive(P, Q, None)
        h = b // 2

        def blk(M, r, c):
            return [row[c * h:(c + 1) * h] for row in M[r * h:(r + 1) * h]]

        ops = []
        for a, bb, S in witnesses.strassen_blocks_any_field():
            L = [[sum(a[r, k] * blk(P, r, k)[i][j] for r in range(2) for k in range(2)) for j in range(h)]
                 for i in range(h)]
            R = [[sum(bb[k, c] * blk(Q, k, c)[i][j] for k in range(2) for c in range(2)) for j in range(h)]
                 for i in range(h)]
            ops.append((L, R))
        M = {i + 1: op for i, op in enumerate(ops)}
        N3 = self.strassen(*M[3], depth - 1)
        C12 = self.naive(*M[5], N3)
        N13 = self.naive(*M[1], N3)
        N134 = self.naive(*M[4], N13)
        C11 = self.naive(*M[7], N134)
        N132 = self.naive(*M[2], N13)
        N6 = self.strassen(*M[6], depth - 1)
        C21 = self.merge(N134, N6)
        C22 = self.merge(N132, N6)
        top = [r1 + r2 for r1, r2 in zip(C11, C12)]
        bot = [r1 + r2 for r1, r2 in zip(C21, C22)]
        return top + bot


def strassen_dag(n, depth):
    """Products, merges and sum of min-ranks of the schedule for <n,n,n>; asserts every output is exact over Z."""
    bd = Builder(n)
    ex = np.eye(n * n, dtype=np.int64)
    P = [[ex[i * n + k] for k in range(n)] for i in range(n)]
    Q = [[ex[k * n + j] for j in range(n)] for k in range(n)]
    C = bd.strassen(P, Q, depth)
    T = fq.matmul_targets(n)
    for (r, c), t in T.items():
        assert np.array_equal(C[r][c], t), (r, c)
    return bd.products, bd.merges, bd.weight


# ---- F2 encodings: a 4x4 form is a 16-bit int, bit 4*x + y; x = r*2+j (x_rj), y = j*2+c (y_jc)

def outer_mask(u, v):
    m = 0
    for a in range(4):
        if u >> a & 1:
            for b in range(4):
                if v >> b & 1:
                    m |= 1 << (4 * a + b)
    return m


RANK1 = sorted({outer_mask(u, v) for u in range(1, 16) for v in range(1, 16)})
IS_RANK1 = np.zeros(1 << 16, dtype=bool)
IS_RANK1[RANK1] = True
TGT = []
for (r, c) in CELLS:
    m = 0
    for j in range(2):
        m |= 1 << (4 * (r * 2 + j) + (j * 2 + c))
    TGT.append(m)


def m2(x):
    return np.array([[x & 1, x >> 1 & 1], [x >> 2 & 1, x >> 3 & 1]], dtype=np.int64)


def i2(a):
    a = np.asarray(a) % 2
    return int(a[0, 0] | a[0, 1] << 1 | a[1, 0] << 2 | a[1, 1] << 3)


def inv2(a):
    for x in range(16):
        b = m2(x)
        if np.array_equal(a @ b % 2, np.eye(2, dtype=np.int64)):
            return b
    raise ValueError


def valid(dec):
    acc = [0, 0, 0, 0]
    for (u, v, w) in dec:
        pm = outer_mask(u, v)
        for t in range(4):
            if w >> t & 1:
                acc[t] ^= pm
    return acc == TGT


def strassen_f2():
    a_sets = [[(0, 0), (1, 1)], [(1, 0), (1, 1)], [(0, 0)], [(1, 1)], [(0, 0), (0, 1)], [(1, 0), (0, 0)],
              [(0, 1), (1, 1)]]
    b_sets = [[(0, 0), (1, 1)], [(0, 0)], [(0, 1), (1, 1)], [(1, 0), (0, 0)], [(1, 1)], [(0, 0), (0, 1)],
              [(1, 0), (1, 1)]]
    cells = {(0, 0): [0, 3, 4, 6], (0, 1): [2, 4], (1, 0): [1, 3], (1, 1): [0, 1, 2, 5]}
    dec = []
    for i in range(7):
        u = sum(1 << (r * 2 + k) for (r, k) in a_sets[i])
        v = sum(1 << (k * 2 + c) for (k, c) in b_sets[i])
        w = sum(1 << t for t, pos in enumerate(CELLS) if i in cells[pos])
        dec.append((u, v, w))
    assert valid(dec)
    return frozenset(dec)


def orbit():
    g = [np.array([[1, 1], [0, 1]]), np.array([[0, 1], [1, 0]])]
    I = np.eye(2, dtype=np.int64)
    gens = []
    for h in g:
        for (P, Q, R) in ((h, I, I), (I, h, I), (I, I, h)):
            Pit, Qit, Ri = inv2(P).T, inv2(Q).T, inv2(R)
            gens.append(lambda t, P=P, Q=Q, R=R, Pit=Pit, Qit=Qit, Ri=Ri: (
                i2(Pit @ m2(t[0]) @ Q.T), i2(Qit @ m2(t[1]) @ R.T), i2(P @ m2(t[2]) @ Ri)))
    gens.append(lambda t: (i2(m2(t[1]).T), i2(m2(t[0]).T), i2(m2(t[2]).T)))
    gens.append(lambda t: (t[1], i2(m2(t[2]).T), i2(m2(t[0]).T)))
    start = strassen_f2()
    seen = {start}
    todo = [start]
    while todo:
        d = todo.pop()
        for f in gens:
            e = frozenset(f(t) for t in d)
            if e not in seen:
                assert valid(e)
                seen.add(e)
                todo.append(e)
    return seen


def relabel(w, perm):
    return sum(1 << perm[t] for t in range(4) if w >> t & 1)


def rooted_trees(n):
    """One parent array (parent[k] < k) per unlabelled rooted tree on n nodes."""
    reps = {}
    for par in itertools.product(*[range(k) for k in range(1, n)]):
        par = (None,) + par
        kids = [[] for _ in range(n)]
        for k in range(1, n):
            kids[par[k]].append(k)

        def enc(v):
            return "(" + "".join(sorted(enc(c) for c in kids[v])) + ")"
        reps.setdefault(enc(0), par)
    return list(reps.values())


def structures():
    """Yield (parent, N, B, targets, lambdas) for every DAG with 7 product steps and one merge N = A + B."""
    n = 9
    for par in rooted_trees(n):
        kids = [[] for _ in range(n)]
        for k in range(1, n):
            kids[par[k]].append(k)
        leaves = {v for v in range(1, n) if not kids[v]}
        desc = [set() for _ in range(n)]
        for v in range(n - 1, -1, -1):
            desc[v] = {v}.union(*[desc[c] for c in kids[v]])
        anc = [set() for _ in range(n)]
        for v in range(1, n):
            anc[v] = anc[par[v]] | {par[v]}
        for N in range(1, n):
            A = par[N]
            if A == 0:
                continue
            for B in range(1, n):
                if B == A or B in desc[N]:
                    continue
                onpath = anc[B] | {B}
                for tg in itertools.combinations(range(1, n), 4):
                    if not leaves - {B} <= set(tg):
                        continue
                    slot = {v: i for i, v in enumerate(tg)}
                    sub = [sum(1 << slot[x] for x in desc[v] if x in slot) for v in range(n)]
                    lam = []
                    for h in range(1, n):
                        if h == N:
                            continue
                        lam.append(sub[h] ^ (sub[N] if h in onpath else 0))
                    yield par, N, B, tg, tuple(sorted(lam))


def check_formula(trials, seed=1):
    """The output-set formula of structures() against the node values a random DAG actually computes over F2."""
    rng = np.random.default_rng(seed)
    allst = list(structures())
    for i in rng.choice(len(allst), size=trials, replace=False):
        par, N, B, tg, _ = allst[i]
        n = len(par)
        R = {h: RANK1[rng.integers(len(RANK1))] for h in range(1, n) if h != N}
        val = [0] * n
        order = sorted(range(1, n), key=lambda v: (v in desc_of(par, N), v))
        for v in order:
            val[v] = val[par[v]] ^ val[B] if v == N else val[par[v]] ^ R[v]
        kids = [[] for _ in range(n)]
        for k in range(1, n):
            kids[par[k]].append(k)
        onpath = {B}
        x = B
        while x:
            x = par[x]
            onpath.add(x)
        dn = desc_of(par, N)
        for t, node in enumerate(tg):
            want = 0
            for h in R:
                dh = desc_of(par, h)
                if (node in dh) ^ (h in onpath and node in dn):
                    want ^= R[h]
            assert val[node] == want, (par, N, B, tg)
    return trials


def desc_of(par, v):
    out = {v}
    changed = True
    while changed:
        changed = False
        for k in range(1, len(par)):
            if par[k] in out and k not in out:
                out.add(k)
                changed = True
    return out


def f2rank(vecs):
    basis = []
    for x in vecs:
        for b in basis:
            x = min(x, x ^ b)
        if x:
            basis.append(x)
    return len(basis)


def solvable(lam):
    """Is there a rank-7 decomposition over F2 with output side lam (7 cell sets, labelled)?"""
    L = np.array([[w >> t & 1 for w in lam] for t in range(4)], dtype=np.int64)
    for P in itertools.combinations(range(7), 4):
        if f2rank([lam[i] for i in P]) == 4:
            break
    F = [i for i in range(7) if i not in P]
    aug = np.concatenate([L[:, list(P)], np.eye(4, dtype=np.int64)], axis=1)
    for col in range(4):
        piv = next(r for r in range(col, 4) if aug[r, col])
        aug[[col, piv]] = aug[[piv, col]]
        for r in range(4):
            if r != col and aug[r, col]:
                aug[r] ^= aug[col]
    Binv = aug[:, 4:]
    c = Binv @ L[:, F] % 2
    K = []
    for j in range(4):
        k = 0
        for t in range(4):
            if Binv[j, t]:
                k ^= TGT[t]
        K.append(k)
    R1 = np.array(RANK1, dtype=np.int64)
    a = R1[:, None]
    b = R1[None, :]
    for f0 in RANK1:
        ok = np.ones((len(RANK1), len(RANK1)), dtype=bool)
        for j in range(4):
            val = np.full(ok.shape, K[j], dtype=np.int64)
            if c[j, 0]:
                val ^= f0
            if c[j, 1]:
                val = val ^ a
            if c[j, 2]:
                val = val ^ b
            ok &= IS_RANK1[val]
            if not ok.any():
                break
        if ok.any():
            return True
    return False


def one_merge_search(check_complete):
    orb = orbit()
    outputs = {tuple(sorted(w for (_, _, w) in d)) for d in orb}
    perms = list(itertools.permutations(range(4)))
    out24 = {tuple(sorted(relabel(w, s) for w in o)) for o in outputs for s in perms}
    hits = []
    cand = set()
    count = 0
    for par, N, B, tg, lam in structures():
        count += 1
        if 0 in lam or f2rank(lam) < 4:
            continue
        if lam in out24:
            hits.append((par, N, B, tg, lam))
        cand.add(min(tuple(sorted(relabel(w, s) for w in lam)) for s in perms))
    prop51 = tuple(sorted(sum(1 << CELLS.index(pos) for pos in S)
                          for _, S in witnesses.strassen_products_any_field()))
    assert prop51 in outputs and solvable(prop51)
    print("rank-7 decompositions in Strassen's orbit over F2:", len(orb), "; distinct output sides:", len(outputs))
    print("one-merge DAGs (7 products):", count, "; distinct output sides up to relabelling:", len(cand),
          "; realising a decomposition in the orbit:", len(hits))
    if check_complete:
        # the dihedral symmetries of the 2x2 grid preserve solvability; S4 = D4 * 3 cosets
        d4 = [(0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0), (0, 2, 1, 3), (2, 0, 3, 1), (1, 3, 0, 2),
              (3, 1, 2, 0)]
        labelled = set()
        for lam in cand:
            for s in perms:
                x = tuple(sorted(relabel(w, s) for w in lam))
                labelled.add(min(tuple(sorted(relabel(w, d) for w in x)) for d in d4))
        solved = [x for x in labelled if solvable(x)]
        print("labelled output sides checked by direct solve:", len(labelled), "; solvable:", len(solved))
        assert not solved
    return hits


if __name__ == "__main__":
    for p in (0, 2, 3, 5):
        assert two_merge_schedule(p) == (7, 2)
    print("<2,2,2> with 7 products and 2 merges: exact over Z (so over every field)")
    for n, depth in ((2, 1), (4, 1), (4, 2), (8, 1), (8, 2), (8, 3)):
        p, k, w = strassen_dag(n, depth)
        assert w == 2 * (n ** 3 - p)
        print(f"<{n},{n},{n}> depth {depth}: {p} products (saving {n ** 3 - p}), {k} merges, "
              f"sum of min-ranks {w} = 2 x saving; exact over Z")
    print("output-set formula matches propagated node values on", check_formula(2000), "random one-merge DAGs")
    hits = one_merge_search("--complete" in sys.argv)
    assert not hits
    print("no DAG with 7 products and one merge computes <2,2,2> over F2")

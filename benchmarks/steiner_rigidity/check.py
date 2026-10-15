"""Checks for the laminar-flattening proof of Steiner rigidity (CMP-8, worker A1a).

Part 1: exact Steiner lengths over F2 on 4x4 matrices (2^16 states), the starter's dynamic program.
Part 2: the labeling lemma, exhaustively over every binary hierarchy on every subset of a 2x3 grid and on random
hierarchies of larger grids, over GF(2) and GF(10007): the typing rule always succeeds, every node's label sum has
rank <= 1, labels are independent within each grid row (a) and column (b), and the representability sets the proof
tracks agree with their linear-algebra definition.
Part 3: the flattening identity on random Steiner trees with dense Steiner points.
Part 4: the merge bound (Corollary 4): its identity on Strassen as an accumulate-and-merge DAG and on random DAGs.
"""

import itertools
import random
import sys

import numpy as np

# ---------- Part 1: exact Steiner lengths over F2, 4x4 matrices ----------

N = 1 << 16
IDX = np.arange(N, dtype=np.int64)
INF = 99


def mat(es):
    m = 0
    for r, c in es:
        m |= 1 << (4 * r + c)
    return m


GENS = np.array(sorted({sum(1 << (4 * r + c) for r in range(4) if u >> r & 1 for c in range(4) if v >> c & 1)
                        for u in range(1, 16) for v in range(1, 16)}), dtype=np.int64)


def relax(f):
    while True:
        g = f.copy()
        for r in GENS:
            np.minimum(g, f[IDX ^ r] + 1, out=g)
        if np.array_equal(g, f):
            return f
        f = g


def steiner(terms):
    k = len(terms)
    D = {}
    for i, t in enumerate(terms):
        f = np.full(N, INF, dtype=np.int16)
        f[t] = 0
        D[1 << i] = relax(f)
    for mask in range(1, 1 << k):
        if mask & (mask - 1) == 0:
            continue
        low = mask & -mask
        g = np.full(N, INF, dtype=np.int16)
        sub = (mask - 1) & mask
        while sub:
            if sub & low:
                np.minimum(g, D[sub] + D[mask ^ sub], out=g)
            sub = (sub - 1) & mask
        D[mask] = relax(g)
    return int(D[(1 << k) - 1][0])


def rank2(m):
    rows = [(m >> (4 * r)) & 15 for r in range(4)]
    rk = 0
    for bit in range(4):
        piv = next((i for i in range(rk, 4) if rows[i] >> bit & 1), None)
        if piv is None:
            continue
        rows[rk], rows[piv] = rows[piv], rows[rk]
        for i in range(4):
            if i != rk and rows[i] >> bit & 1:
                rows[i] ^= rows[rk]
        rk += 1
    return rk


def block_grid(blocks):
    """2x2 grid of 2x2 blocks: blocks[(r, c)] is a 2x2 0/1 matrix placed at rows 2r.., cols 2c.."""
    out = []
    for (r, c), A in blocks.items():
        out.append(mat([(2 * r + i, 2 * c + j) for i in range(2) for j in range(2) if A[i][j]]))
    return out


def part1():
    I2 = [[1, 0], [0, 1]]
    cases = {
        "matmul 2x2x2 (starter layout)": [mat([(0, 0), (1, 2)]), mat([(0, 1), (1, 3)]),
                                          mat([(2, 0), (3, 2)]), mat([(2, 1), (3, 3)])],
        "grid, all blocks I": block_grid({(0, 0): I2, (0, 1): I2, (1, 0): I2, (1, 1): I2}),
        "grid, holonomy transvection": block_grid({(0, 0): I2, (0, 1): I2, (1, 0): I2, (1, 1): [[1, 1], [0, 1]]}),
        "grid, holonomy order 3": block_grid({(0, 0): I2, (0, 1): I2, (1, 0): I2, (1, 1): [[0, 1], [1, 1]]}),
        "grid, mixed ranks": block_grid({(0, 0): [[1, 1], [1, 1]], (0, 1): I2, (1, 0): [[0, 1], [0, 0]],
                                         (1, 1): [[1, 1], [0, 1]]}),
    }
    for name, terms in cases.items():
        st = steiner(terms)
        naive = sum(rank2(t) for t in terms)
        print(f"[F2 exact] {name}: Steiner {st}, sum of ranks {naive}")
        assert st == naive, name
    # Pairwise-trivial row spaces are not enough: Z = E11 + E22, T_i = Z + rank one.
    Z = [(0, 0), (1, 1)]
    T1 = mat(Z + [(0, 2)])
    T2 = mat(Z + [(1, 3)])
    T3 = mat(Z + [(0, 2), (0, 3), (1, 2), (1, 3)])
    st = steiner([T1, T2, T3])
    print(f"[F2 exact] pairwise counterexample: ranks {[rank2(t) for t in (T1, T2, T3)]}, "
          f"Steiner {st}, sum of ranks {sum(rank2(t) for t in (T1, T2, T3))}")
    assert st == 5


# ---------- linear algebra mod p ----------

def rref(M, p):
    M = np.array(M, dtype=np.int64) % p
    rows, cols = M.shape
    piv = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0:
            continue
        i = r + nz[0]
        M[[r, i]] = M[[i, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        for j in range(rows):
            if j != r and M[j, c]:
                M[j] = (M[j] - M[j, c] * M[r]) % p
        piv.append(c)
        r += 1
    return M[:r], piv


def rank(M, p):
    M = np.array(M, dtype=np.int64)
    return 0 if M.size == 0 else len(rref(M, p)[1])


def nullspace(C, k, p):
    """Rows form a basis of {s in F_p^k : C s = 0}."""
    if len(C) == 0:
        return np.eye(k, dtype=np.int64)
    R, piv = rref(np.array(C, dtype=np.int64), p)
    basis = []
    for f in (j for j in range(k) if j not in piv):
        v = np.zeros(k, dtype=np.int64)
        v[f] = 1
        for i, pc in enumerate(piv):
            v[pc] = (-R[i, f]) % p
        basis.append(v)
    return np.array(basis, dtype=np.int64).reshape(len(basis), k)


# ---------- Part 2: the labeling lemma ----------

def leaves_of(tree):
    return [tree] if isinstance(tree[0], int) else leaves_of(tree[0]) + leaves_of(tree[1])


def type_tree(tree, index):
    """Bottom-up typing. Returns the nodes (post-order) and the x- and y-constraints (pairs of index lists)."""
    nodes, xcons, ycons = [], [], []

    def go(t):
        if isinstance(t[0], int):
            i = index[t]
            node = dict(S=[i], D=[i], E=[i], rho={t[0]}, gam={t[1]}, kind="leaf", nx=0, ny=0)
        else:
            A, B = go(t[0]), go(t[1])
            if not (A["rho"] & B["rho"]):
                xcons.append((A["D"], B["D"]))
                node = dict(S=A["S"] + B["S"], D=A["D"], E=A["E"] + B["E"], rho=A["rho"] | B["rho"],
                            gam=A["gam"] & B["gam"], kind="x")
            else:
                assert not (A["gam"] & B["gam"]), "both merges blocked"
                ycons.append((A["E"], B["E"]))
                node = dict(S=A["S"] + B["S"], D=A["D"] + B["D"], E=A["E"], rho=A["rho"] & B["rho"],
                            gam=A["gam"] | B["gam"], kind="y")
            node["nx"], node["ny"] = len(xcons), len(ycons)
            node["xc"] = A.get("xc", []) + B.get("xc", []) + ([xcons[-1]] if node["kind"] == "x" else [])
            node["yc"] = A.get("yc", []) + B.get("yc", []) + ([ycons[-1]] if node["kind"] == "y" else [])
        nodes.append(node)
        return node

    go(tree)
    return nodes, xcons, ycons


def functional(pair, k):
    v = np.zeros(k, dtype=np.int64)
    for i in pair[0]:
        v[i] += 1
    for i in pair[1]:
        v[i] -= 1
    return v


def representable(ind, cons, coords, k, p):
    """Is the indicator vector ind in span(cons) + F^coords?"""
    base = [functional(c, k) for c in cons] + [np.eye(k, dtype=np.int64)[j] for j in coords]
    if not base:
        return not np.any(ind % p)
    return rank(base + [ind], p) == rank(base, p)


def check_tree(tree, p, check_rep=True):
    lv = leaves_of(tree)
    index = {t: i for i, t in enumerate(lv)}
    k = len(lv)
    nodes, xcons, ycons = type_tree(tree, index)
    a = nullspace([functional(c, k) for c in xcons], k, p)  # a_t = a[:, t]
    b = nullspace([functional(c, k) for c in ycons], k, p)
    for node in nodes:  # (L): every node's label sum has rank <= 1
        Msum = sum(np.outer(a[:, i], b[:, i]) for i in node["S"]) % p
        assert rank(Msum, p) <= 1, "rank-1 condition fails"
    rows = {r for r, _ in lv}
    cols = {c for _, c in lv}
    for r in rows:  # (I): independence within each grid row / column
        ids = [index[t] for t in lv if t[0] == r]
        assert rank(a[:, ids].T, p) == len(ids), "row independence fails"
    for c in cols:
        ids = [index[t] for t in lv if t[1] == c]
        assert rank(b[:, ids].T, p) == len(ids), "column independence fails"
    if check_rep:  # the tracked sets rho, gamma equal their linear-algebra definitions
        for node in nodes:
            S = node["S"]
            ind_D = np.zeros(k, dtype=np.int64)
            ind_D[node["D"]] = 1
            ind_E = np.zeros(k, dtype=np.int64)
            ind_E[node["E"]] = 1
            rho = {r for r in rows if representable(ind_D, node.get("xc", []),
                                                    [i for i in S if lv[i][0] == r], k, p)}
            gam = {c for c in cols if representable(ind_E, node.get("yc", []),
                                                    [i for i in S if lv[i][1] == c], k, p)}
            assert rho == node["rho"] and gam == node["gam"], "rho/gamma recursion fails"
            for r in node["rho"]:
                for c in node["gam"]:
                    assert index.get((r, c), -1) in S, "containment claim fails"
    return a, b, nodes


def all_binary_trees(items):
    if len(items) == 1:
        yield items[0]
        return
    first, rest = items[0], items[1:]
    for mask in range(1 << len(rest)):
        left = [first] + [rest[i] for i in range(len(rest)) if mask >> i & 1]
        right = [rest[i] for i in range(len(rest)) if not mask >> i & 1]
        if not right:
            continue
        for L in all_binary_trees(left):
            for R in all_binary_trees(right):
                yield (L, R)


def random_tree(items, rng):
    clusters = list(items)
    while len(clusters) > 1:
        i, j = rng.sample(range(len(clusters)), 2)
        new = (clusters[i], clusters[j])
        clusters = [c for t, c in enumerate(clusters) if t not in (i, j)] + [new]
    return clusters[0]


def part2(rng):
    grid = [(r, c) for r in range(2) for c in range(3)]
    count = 0
    for size in range(2, 7):
        for subset in itertools.combinations(grid, size):
            for tree in all_binary_trees(list(subset)):
                for p in (2, 10007):
                    check_tree(tree, p)
                count += 1
    print(f"[labeling] exhaustive: all {count} binary hierarchies on all subsets of the 2x3 grid, GF(2) and GF(10007): ok")
    count = 0
    for (pp, qq, trials) in ((3, 3, 300), (2, 5, 200), (4, 4, 100), (3, 5, 60)):
        grid = [(r, c) for r in range(pp) for c in range(qq)]
        for _ in range(trials):
            subset = rng.sample(grid, rng.randint(2, len(grid)))
            tree = random_tree(subset, rng)
            for p in (2, 10007):
                check_tree(tree, p, check_rep=(len(subset) <= 10))
            count += 1
    print(f"[labeling] random: {count} hierarchies on 3x3, 2x5, 4x4, 3x5 grids, GF(2) and GF(10007): ok")


# ---------- Part 3: flattening identity on random Steiner trees ----------

def part3(rng, p=10007, m=2):
    for trial in range(40):
        pp, qq = rng.choice([(2, 2), (2, 3), (3, 3)])
        grid = [(r, c) for r in range(pp) for c in range(qq)]
        subset = rng.sample(grid, rng.randint(2, len(grid)))
        tree = random_tree(subset, rng)
        a, b, _ = check_tree(tree, p, check_rep=False)
        lv = leaves_of(tree)
        index = {t: i for i, t in enumerate(lv)}

        def target(t):
            T = np.zeros((pp * m, qq * m), dtype=np.int64)
            T[t[0] * m:(t[0] + 1) * m, t[1] * m:(t[1] + 1) * m] = np.eye(m, dtype=np.int64)
            return T

        # Node values: leaves are targets, internal nodes are dense random Steiner points; root parent is 0.
        edges = []  # (set of leaf indices below, W = child value - parent value)

        def go(t, parent):
            if isinstance(t[0], int):
                edges.append(([index[t]], (target(t) - parent) % p))
                return
            val = np.array([[rng.randrange(p) for _ in range(qq * m)] for _ in range(pp * m)], dtype=np.int64)
            if rng.random() < 0.5:  # low-rank shift of the parent, the kind of point a cheater would use
                u = np.array([rng.randrange(p) for _ in range(pp * m)], dtype=np.int64)
                v = np.array([rng.randrange(p) for _ in range(qq * m)], dtype=np.int64)
                val = (parent + np.outer(u, v)) % p
            edges.append((leaves_of(t) and [index[x] for x in leaves_of(t)], (val - parent) % p))
            go(t[0], val)
            go(t[1], val)

        go(tree, np.zeros((pp * m, qq * m), dtype=np.int64))
        lhs = sum(np.kron(np.outer(a[:, index[t]], b[:, index[t]]), target(t)) for t in lv) % p
        rhs = sum(np.kron(sum(np.outer(a[:, i], b[:, i]) for i in S) % p, W) for S, W in edges) % p
        assert np.array_equal(lhs, rhs), "flattening identity fails"
        assert rank(lhs, p) == m * len(lv), "rank of the flattening is not m|P|"
        assert sum(rank(W, p) for _, W in edges) >= m * len(lv)
    print(f"[flattening] 40 random Steiner trees with dense Steiner points: identity holds, rank = m|P|: ok")


# ---------- Part 4: computations with merges (Corollary 4) ----------

def laminar_to_tree(sets, universe):
    """A binary hierarchy (nested tuples of cells) refining a laminar family of sets of cells."""
    fam = {frozenset(s) for s in sets if s} | {frozenset([u]) for u in universe} | {frozenset(universe)}
    fam = sorted(fam, key=len)

    def build(S):
        kids = [T for T in fam if T < S and not any(T < U < S for U in fam)]
        trees = [build(T) for T in kids] if len(S) > 1 else [next(iter(S))]
        while len(trees) > 1:
            trees = [(trees[0], trees[1])] + trees[2:]
        return trees[0]

    return build(frozenset(universe))


def run_dag(nodes, targets, p, shape):
    """nodes: list of ("root",), ("prod", parent, R), ("merge", a, alpha, b, beta); H keeps arc a, deletes arc b.
    targets: {cell: node index}. Returns (lhs, rhs, labels) of the merge identity, with Lemma 3 labels for H."""
    val, omega, hpar = [], [], []
    for nd in nodes:
        if nd[0] == "root":
            val.append(np.zeros(shape, dtype=np.int64)); omega.append(1); hpar.append(None)
        elif nd[0] == "prod":
            val.append((val[nd[1]] + nd[2]) % p); omega.append(omega[nd[1]]); hpar.append(nd[1])
        else:
            _, a, al, b, be = nd
            val.append((al * val[a] + be * val[b]) % p); omega.append(al * omega[a] % p); hpar.append(a)

    def below(w):
        out = set()
        for cell, x in targets.items():
            while x is not None:
                if x == w:
                    out.add(cell)
                    break
                x = hpar[x]
        return out

    cells = list(targets)
    tree = laminar_to_tree([below(w) for w in range(len(nodes))], cells)
    a, b, _ = check_tree(tree, p, check_rep=False)
    index = {t: i for i, t in enumerate(leaves_of(tree))}
    inv = lambda z: pow(int(z) % p, p - 2, p)
    lam = lambda S: sum((np.outer(a[:, index[t]], b[:, index[t]]) for t in S), np.zeros((a.shape[0], b.shape[0]),
                                                                                          dtype=np.int64)) % p
    lhs = sum(np.kron(lam([t]), val[w] * inv(omega[w]) % p) for t, w in targets.items()) % p
    rhs = np.zeros_like(lhs)
    for w, nd in enumerate(nodes):
        S = below(w)
        assert rank(lam(S), p) <= 1
        if nd[0] == "prod":
            rhs = (rhs + np.kron(lam(S), nd[2] * inv(omega[w]) % p)) % p
        elif nd[0] == "merge":
            rhs = (rhs + np.kron(lam(S), val[nd[3]] * (nd[4] * inv(omega[w]) % p) % p)) % p
    return lhs, rhs, val


def part4(rng, p=10007):
    # Strassen 2x2 as an accumulate-and-merge DAG. Rows (i,k) -> 2i+k, columns (j,k') -> 2j+k'.
    def x(i, k):
        v = np.zeros(4, dtype=np.int64); v[2 * i + k] = 1; return v

    def y(k, j):
        v = np.zeros(4, dtype=np.int64); v[2 * j + k] = 1; return v

    prods = [(x(0, 0) + x(1, 1), y(0, 0) + y(1, 1)), (x(1, 0) + x(1, 1), y(0, 0)), (x(0, 0), y(0, 1) - y(1, 1)),
             (x(1, 1), y(1, 0) - y(0, 0)), (x(0, 0) + x(0, 1), y(1, 1)), (x(1, 0) - x(0, 0), y(0, 0) + y(0, 1)),
             (x(0, 1) - x(1, 1), y(1, 0) + y(1, 1))]
    nodes = [("root",)] + [("prod", 0, np.outer(u, v) % p) for u, v in prods]  # P1..P7 are nodes 1..7
    P = lambda i: i
    nodes += [("merge", P(1), 1, P(4), 1), ("merge", 8, 1, P(5), -1), ("merge", 9, 1, P(7), 1),   # 10 = C11
              ("merge", P(3), 1, P(5), 1),                                                       # 11 = C12
              ("merge", P(2), 1, P(4), 1),                                                       # 12 = C21
              ("merge", P(1), 1, P(2), -1), ("merge", 13, 1, P(3), 1), ("merge", 14, 1, P(6), 1)]  # 15 = C22
    targets = {(0, 0): 10, (0, 1): 11, (1, 0): 12, (1, 1): 15}
    lhs, rhs, val = run_dag(nodes, targets, p, (4, 4))
    for (r, c), w in targets.items():
        T = np.zeros((4, 4), dtype=np.int64)
        for k in range(2):
            T[2 * r + k, 2 * c + k] = 1
        assert np.array_equal(val[w], T), "Strassen DAG does not compute the product"
    assert np.array_equal(lhs, rhs), "merge identity fails on Strassen"
    loss = sum(rank(val[nd[3]], p) for nd in nodes if nd[0] == "merge")
    print(f"[merges] Strassen DAG: 7 products, 8 merges, rank of the flattening {rank(lhs, p)}, "
          f"bound 7 >= {rank(lhs, p)} - {loss}: ok")
    for trial in range(200):
        shape = (4, 4)
        nodes = [("root",)]
        for _ in range(rng.randint(3, 9)):
            if len(nodes) > 2 and rng.random() < 0.35:
                a, b = rng.sample(range(len(nodes)), 2)
                nodes.append(("merge", a, rng.randrange(1, p), b, rng.randrange(1, p)))
            else:
                u = np.array([rng.randrange(p) for _ in range(4)], dtype=np.int64)
                v = np.array([rng.randrange(p) for _ in range(4)], dtype=np.int64)
                nodes.append(("prod", rng.randrange(len(nodes)), np.outer(u, v) % p))
        picks = rng.sample(range(1, len(nodes)), min(4, len(nodes) - 1))
        cells = rng.sample([(r, c) for r in range(2) for c in range(2)], len(picks))
        lhs, rhs, _ = run_dag(nodes, dict(zip(cells, picks)), p, shape)
        assert np.array_equal(lhs, rhs), "merge identity fails"
    print("[merges] 200 random DAGs with weighted merges: identity of Corollary 4 holds: ok")


if __name__ == "__main__":
    rng = random.Random(20261015)
    parts = sys.argv[1:] or ["1", "2", "3", "4"]
    if "4" in parts:
        part4(rng)
    if "2" in parts:
        part2(rng)
    if "3" in parts:
        part3(rng)
    if "1" in parts:
        part1()

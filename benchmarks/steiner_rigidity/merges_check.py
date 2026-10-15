"""Checks for the chain-form merge bound (CMP-8, worker A3): note 20261015T0436Z-draft-merges-and-reuse.

Part 1: the labeling lemma for set systems (Lemma 3'). Targets carry sets of "row sets" and "column sets"; if every
row set meets every column set in at most one target, A1a's typing rule still succeeds, every node of the hierarchy
has a rank <= 1 label sum, and the labels are independent within each row set and each column set. Checked on the
hybrid systems the proof uses: per position one merge-free prefix (a target spanning the D-cells of levels 1..sigma)
and single D-cells above it.
Part 2: Lemma 2(b) for those systems: rank of the flattening equals the sum of the targets' ranks.
Part 3: the chain-form bound on random valid chain-form computations: the flattening identity holds edge by edge,
every edge's label sum has rank <= 1, and #products >= L * (sum of sigma + #kept links).
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import nullspace, rank, rref  # noqa: E402


def binary_hierarchy(sets, universe):
    """Nested pairs of target ids refining a laminar family (the sets plus singletons plus the universe)."""
    fam = {frozenset(s) for s in sets if s} | {frozenset([u]) for u in universe} | {frozenset(universe)}
    fam = sorted(fam, key=len)
    for A in fam:
        for B in fam:
            assert A <= B or B <= A or not (A & B), "family is not laminar"

    def build(S):
        if len(S) == 1:
            return next(iter(S))
        kids = [T for T in fam if T < S and not any(T < U < S for U in fam)]
        trees = [build(T) for T in kids]
        while len(trees) > 1:
            trees = [(trees[0], trees[1])] + trees[2:]
        return trees[0]

    return build(frozenset(universe))


def type_sets(tree, rows_of, cols_of):
    """A1a's bottom-up typing with handles drawn from arbitrary row sets and column sets."""
    nodes, xcons, ycons = [], [], []

    def go(t):
        if isinstance(t, int):
            node = dict(S=[t], D=[t], E=[t], rho=set(rows_of[t]), gam=set(cols_of[t]))
        else:
            A, B = go(t[0]), go(t[1])
            if not (A["rho"] & B["rho"]):
                xcons.append((A["D"], B["D"]))
                node = dict(S=A["S"] + B["S"], D=A["D"], E=A["E"] + B["E"], rho=A["rho"] | B["rho"],
                            gam=A["gam"] & B["gam"])
            else:
                assert not (A["gam"] & B["gam"]), "both merges blocked"
                ycons.append((A["E"], B["E"]))
                node = dict(S=A["S"] + B["S"], D=A["D"] + B["D"], E=A["E"], rho=A["rho"] & B["rho"],
                            gam=A["gam"] | B["gam"])
        nodes.append(node)
        return node

    go(tree)
    return nodes, xcons, ycons


def functional(pair, k):
    v = np.zeros(k, dtype=np.int64)
    v[pair[0]] += 1
    v[pair[1]] -= 1
    return v


def set_labels(tree, k, rows_of, cols_of, p):
    nodes, xcons, ycons = type_sets(tree, rows_of, cols_of)
    a = nullspace([functional(c, k) for c in xcons], k, p)
    b = nullspace([functional(c, k) for c in ycons], k, p)
    for node in nodes:
        M = sum(np.outer(a[:, i], b[:, i]) for i in node["S"]) % p
        assert rank(M, p) <= 1, "rank-1 condition fails"
    for lab, sets_of in ((a, rows_of), (b, cols_of)):
        for s in set().union(*sets_of):
            ids = [t for t in range(k) if s in sets_of[t]]
            assert rank(lab[:, ids].T, p) == len(ids), "independence fails"
    return a, b, [frozenset(n["S"]) for n in nodes]


class Shape:
    """P x m by m x Q matmul with prefix stride L; X index (i, k) -> i*m + k, Y index (j, k) -> j*m + k."""

    def __init__(self, P, Q, m, L):
        self.P, self.Q, self.m, self.L, self.S = P, Q, m, L, m // L

    def D(self, pos, s):
        i, j = pos
        M = np.zeros((self.P * self.m, self.Q * self.m), dtype=np.int64)
        for k in range(s * self.L, (s + 1) * self.L):
            M[i * self.m + k, j * self.m + k] = 1
        return M

    def T(self, pos, s):
        return sum((self.D(pos, t) for t in range(s)), np.zeros((self.P * self.m, self.Q * self.m), dtype=np.int64))


def hybrid_system(shape, rng):
    """Targets: per position a prefix of sigma levels (if sigma >= 1) and a random subset of the D-cells above it."""
    targets = []  # (pos, kind, sigma or level)
    for i in range(shape.P):
        for j in range(shape.Q):
            sig = rng.randint(0, shape.S)
            if sig:
                targets.append(((i, j), "prefix", sig))
            for s in range(sig, shape.S):
                if rng.random() < 0.7:
                    targets.append(((i, j), "cell", s))
    rows_of, cols_of = [], []
    for (i, j), kind, v in targets:
        levels = range(v) if kind == "prefix" else [v]
        rows_of.append(frozenset((i, s) for s in levels))
        cols_of.append(frozenset((j, s) for s in levels))
    for R in set().union(*rows_of):
        for C in set().union(*cols_of):
            assert sum(1 for t in range(len(targets)) if R in rows_of[t] and C in cols_of[t]) <= 1
    return targets, rows_of, cols_of


def target_matrix(shape, t):
    pos, kind, v = t
    return shape.T(pos, v) if kind == "prefix" else shape.D(pos, v)


def random_laminar(k, rng):
    """A random laminar family on range(k), from a random binary hierarchy with some sets removed."""
    clusters = [frozenset([i]) for i in range(k)]
    fam = []
    while len(clusters) > 1:
        x, y = rng.sample(range(len(clusters)), 2)
        new = clusters[x] | clusters[y]
        fam.append(new)
        clusters = [c for t, c in enumerate(clusters) if t not in (x, y)] + [new]
    return [S for S in fam if rng.random() < 0.6]


def flatten(labels_a, labels_b, mats, coeff, p):
    """sum over targets t of coeff[t] * (a_t b_t^T) kron mats[t]."""
    out = None
    for t, M in mats.items():
        term = coeff.get(t, 1) * np.kron(np.outer(labels_a[:, t], labels_b[:, t]), M)
        out = term if out is None else out + term
    return out % p


def part1_2(rng):
    count = 0
    for (P, Q, m, L, trials) in ((2, 2, 4, 2, 150), (2, 3, 4, 2, 100), (3, 3, 3, 1, 60), (2, 2, 6, 2, 60)):
        shape = Shape(P, Q, m, L)
        for _ in range(trials):
            targets, rows_of, cols_of = hybrid_system(shape, rng)
            k = len(targets)
            if k < 2:
                continue
            tree = binary_hierarchy(random_laminar(k, rng), range(k))
            for p in (2, 10007):
                a, b, _ = set_labels(tree, k, rows_of, cols_of, p)
                mats = {t: target_matrix(shape, targets[t]) for t in range(k)}
                Phi = flatten(a, b, mats, {}, p)
                assert rank(Phi, p) == sum(rank(M, p) for M in mats.values()), "Lemma 2(b) fails"
            count += 1
    print(f"[labels] {count} hybrid systems x random laminar families, GF(2) and GF(10007): "
          "typing succeeds, rank <= 1 on every set, independence in every row and column set, "
          "rank of the flattening = sum of ranks: ok")


def rank_one_terms(M, p):
    """M = sum of rank(M) rank-1 terms u v^T over GF(p)."""
    M = np.array(M, dtype=np.int64) % p
    if not M.any():
        return []
    R, piv = rref(M, p)
    C = M[:, piv]
    return [np.outer(C[:, i], R[i]) % p for i in range(len(piv))]


def chain_form(shape, rng, p):
    """A random valid chain-form computation. Returns the bound's ingredients and checks the flattening identity.

    Accumulator forest F: nodes with values; edges (parent, child, step). Prefix targets are F nodes (reached from
    random shared Steiner points). Each link above sigma either has no merge, one merge of one accumulator (kept), or
    two merges (dropped). Local steps are a rank decomposition of what remains."""
    PmQm = (shape.P * shape.m, shape.Q * shape.m)
    zero = np.zeros(PmQm, dtype=np.int64)
    F_val, F_par, F_step = [zero], [None], [None]

    def add_path(parent, target):
        node = parent
        for R in rank_one_terms(target - F_val[parent], p):
            F_val.append((F_val[node] + R) % p)
            F_par.append(node)
            F_step.append(R)
            node = len(F_val) - 1
        if node == parent and not np.array_equal(F_val[parent] % p, target % p):
            raise AssertionError
        return node

    positions = [(i, j) for i in range(shape.P) for j in range(shape.Q)]
    sigma = {pos: rng.randint(0, shape.S) for pos in positions}
    # Shared Steiner points: random sums of pieces of several positions' prefixes.
    for _ in range(rng.randint(1, 4)):
        piece = zero.copy()
        for pos in rng.sample(positions, rng.randint(1, len(positions))):
            s = rng.randint(0, shape.S - 1)
            piece = piece + rng.randint(1, p - 1) * shape.D(pos, s)
        add_path(rng.randrange(len(F_val)), piece % p)
    prefix_node = {}
    for pos in positions:
        if sigma[pos]:
            prefix_node[pos] = add_path(rng.randrange(len(F_val)), shape.T(pos, sigma[pos]))
    # Accumulators meant for merges: pieces of D-cells, maybe shared, built from random F nodes.
    for _ in range(rng.randint(2, 6)):
        piece = zero.copy()
        for pos in rng.sample(positions, rng.randint(1, 2)):
            s = rng.randint(0, shape.S - 1)
            sub = shape.D(pos, s)
            keep = [k for k in range(s * shape.L, (s + 1) * shape.L) if rng.random() < 0.6]
            mask = np.zeros(shape.m, dtype=bool)
            mask[keep] = True
            rowmask = np.tile(mask, shape.P)
            piece = piece + (sub * rowmask[:, None])
        add_path(rng.randrange(len(F_val)), piece % p)

    links = []  # dict(pos, s, kind, Z, beta, local)
    for pos in positions:
        for s in range(sigma[pos], shape.S):
            D = shape.D(pos, s)
            kind = rng.choice(["none", "one", "one", "two"])
            if kind == "none":
                links.append(dict(pos=pos, s=s, kind=kind, Z=None, beta=1, local=rank_one_terms(D, p)))
            elif kind == "one":
                Z, beta = rng.randrange(1, len(F_val)), rng.randint(1, p - 1)
                links.append(dict(pos=pos, s=s, kind=kind, Z=Z, beta=beta,
                                  local=rank_one_terms(D - beta * F_val[Z], p)))
            else:
                Z1, Z2 = rng.randrange(1, len(F_val)), rng.randrange(1, len(F_val))
                rest = D - F_val[Z1] - F_val[Z2]
                links.append(dict(pos=pos, s=s, kind=kind, Z=None, beta=1, local=rank_one_terms(rest, p)))

    n_products = (len(F_val) - 1) + sum(len(l["local"]) for l in links)
    n_merges = sum({"none": 0, "one": 1, "two": 2}[l["kind"]] for l in links)
    # The kept family: prefix targets and kept links.
    fam = [("prefix", pos) for pos in positions if sigma[pos]] + \
          [("link", idx) for idx, l in enumerate(links) if l["kind"] != "two"]
    k = len(fam)
    if k < 2:
        return None
    rows_of, cols_of, mats, anchor, beta = [], [], {}, {}, {}
    for t, (kind, x) in enumerate(fam):
        if kind == "prefix":
            (i, j), levels = x, range(sigma[x])
            mats[t], anchor[t], beta[t] = shape.T(x, sigma[x]), prefix_node[x], 1
        else:
            l = links[x]
            (i, j), levels = l["pos"], [l["s"]]
            mats[t], anchor[t], beta[t] = shape.D(l["pos"], l["s"]), l["Z"], l["beta"]
        rows_of.append(frozenset((i, s) for s in levels))
        cols_of.append(frozenset((j, s) for s in levels))

    def ancestors(node):
        out = set()
        while node is not None and node != 0:
            out.add(node)
            node = F_par[node]
        return out

    below = {node: set() for node in range(1, len(F_val))}  # F edge (into node) -> targets anchored below it
    for t in range(k):
        if anchor[t] is not None:
            for node in ancestors(anchor[t]):
                below[node].add(t)
    tree = binary_hierarchy([frozenset(S) for S in below.values()], range(k))
    a1, b, _ = set_labels(tree, k, rows_of, cols_of, p)
    inv = {t: pow(int(beta[t]), p - 2, p) for t in range(k)}
    a = np.array([[a1[r, t] * inv[t] % p for t in range(k)] for r in range(a1.shape[0])], dtype=np.int64)
    lhs = flatten(a, b, mats, {}, p)
    rhs = np.zeros_like(lhs)
    for node, S in below.items():
        Lam = sum((beta[t] * np.outer(a[:, t], b[:, t]) for t in S), np.zeros((a.shape[0], b.shape[0]),
                                                                                dtype=np.int64)) % p
        assert rank(Lam, p) <= 1, "edge label sum has rank > 1"
        rhs = rhs + np.kron(Lam, F_step[node])
    for t, (kind, x) in enumerate(fam):
        if kind == "link":
            for R in links[x]["local"]:
                rhs = rhs + np.kron(np.outer(a[:, t], b[:, t]), R)
    assert np.array_equal(lhs % p, rhs % p), "flattening identity fails"
    total_rank = sum(rank(M, p) for M in mats.values())
    assert rank(lhs, p) == total_rank, "Lemma 2(b) fails"
    bound = shape.L * (sum(sigma.values()) + sum(1 for l in links if l["kind"] != "two"))
    assert total_rank == bound
    assert n_products >= bound, "chain-form bound fails"
    return n_products, bound, n_merges, shape.P * shape.Q * shape.m


def part3(rng):
    count, tight = 0, 0
    for (P, Q, m, L, trials) in ((2, 2, 4, 2, 80), (2, 3, 4, 2, 40), (2, 2, 6, 3, 40), (3, 3, 2, 1, 30)):
        shape = Shape(P, Q, m, L)
        for _ in range(trials):
            for p in (3, 10007):
                res = chain_form(shape, rng, p)
                if res is None:
                    continue
                count += 1
                tight += res[0] == res[1]
    print(f"[chain form] {count} random valid chain-form computations, GF(3) and GF(10007): flattening identity, "
          f"rank <= 1 per edge, rank = sum of ranks, #products >= L*(sum sigma + #kept): ok ({tight} tight)")


if __name__ == "__main__":
    rng = random.Random(20261015)
    parts = sys.argv[1:] or ["1", "3"]
    if "1" in parts:
        part1_2(rng)
    if "3" in parts:
        part3(rng)

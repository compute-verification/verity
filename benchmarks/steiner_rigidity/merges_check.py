"""Checks for the chain-form merge bound (CMP-8, worker A3): note 20261015T0436Z-draft-merges-and-reuse.

Part 1: the labeling lemma for set systems (Lemma 3'). Targets carry sets of "row sets" and "column sets"; if every
row set meets every column set in at most one target, A1a's typing rule still succeeds, every node of the hierarchy
has a rank <= 1 label sum, and the labels are independent within each row set and each column set. Checked on the
hybrid systems the proof uses: per position one merge-free prefix (a target spanning the D-cells of levels 1..sigma)
and single D-cells above it.
Part 2: Lemma 2(b) for those systems: rank of the flattening equals the sum of the targets' ranks.
Part 3: the chain-form bound on random valid chain-form computations: the flattening identity holds edge by edge,
every edge's label sum has rank <= 1, and #products >= L * (sum of sigma + #kept links).
Part 4: products of two mixed operands, each split into two pieces charged to the word giving the Y-factor: the
pieces of a word w have rank nu_w <= #distinct label sets of its uses, and #products >= rank - 1/2 sum_w nu_w (bilinear
count). Winograd's pairs meet it with equality.
Part 5: Theorem B in general. The covering lemma exhaustively (per-tree bounds L*(sum hi - min lo), or L*sum hi for
the tree at 0, sum to at least L*(|G_T| - H)); then random computations with merges split into merge-free trees, each
tree re-rooted at 0 through its lowest correct target and checked with A1a's flattening on its top targets.
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


def mixed_chain_form(shape, rng, p, winograd_only=False):
    """Chain form with products of two mixed operands (words w = w_X + w_Y). Checks the charge of 1/2 per
    (word, label set) pair: #products >= rank Phi - 1/2 sum_w nu_w, with nu_w <= #distinct label sets of w's uses."""
    Xd, Yd = shape.P * shape.m, shape.Q * shape.m
    zero = np.zeros((Xd, Yd), dtype=np.int64)
    words = []  # (wX, wY)
    products = []  # dict(where=("F", node) or ("local", link), beta, pieces=[(word, matrix)])

    def new_word(wX, wY):
        words.append((np.array(wX, dtype=np.int64) % p, np.array(wY, dtype=np.int64) % p))
        return len(words) - 1

    def mixed(g, h):
        gX, gY = words[g]
        hX, hY = words[h]
        pieces = [(h, np.outer(gX, hY) % p), (g, np.outer(hX, gY) % p)]
        return (pieces[0][1] + pieces[1][1]) % p, pieces

    def xv(i, k):
        v = np.zeros(Xd, dtype=np.int64)
        v[i * shape.m + k] = 1
        return v

    def yv(j, k):
        v = np.zeros(Yd, dtype=np.int64)
        v[j * shape.m + k] = 1
        return v

    dense = [new_word([rng.randrange(p) for _ in range(Xd)], [rng.randrange(p) for _ in range(Yd)])
             for _ in range(0 if winograd_only else 3)]
    F_val, F_par = [zero], [None]
    if not winograd_only:
        for _ in range(rng.randint(2, 6)):
            parent = rng.randrange(len(F_val))
            if rng.random() < 0.5:
                g, h = rng.choice(dense), rng.choice(dense)
                beta, pieces = mixed(g, h)
            else:
                beta = np.outer([rng.randrange(p) for _ in range(Xd)], [rng.randrange(p) for _ in range(Yd)]) % p
                pieces = None
            F_val.append((F_val[parent] + beta) % p)
            F_par.append(parent)
            products.append(dict(where=("F", len(F_val) - 1), beta=beta, pieces=pieces))

    positions = [(i, j) for i in range(shape.P) for j in range(shape.Q)]
    links = []
    for (i, j) in positions:
        for s in range(shape.S):
            D = shape.D((i, j), s)
            kind = "none" if winograd_only else rng.choice(["none", "one", "two"])
            Z = rng.randrange(1, len(F_val)) if kind != "none" and len(F_val) > 1 else None
            kind = kind if Z is not None else "none"
            beta_t = rng.randint(1, p - 1) if kind == "one" else 1
            rest = (D - beta_t * F_val[Z]) % p if kind == "one" else D.copy()
            if kind == "two":
                rest = (rest - F_val[rng.randrange(1, len(F_val))]) % p
            t = len(links)
            local = []
            ks = list(range(s * shape.L, (s + 1) * shape.L))
            for a in range(0, len(ks) - 1, 2):  # Winograd pairs on this cell, two fresh words per product
                if winograd_only or rng.random() < 0.5:
                    k1, k2 = ks[a], ks[a + 1]
                    g, h = new_word(xv(i, k1), yv(j, k2)), new_word(xv(i, k2), yv(j, k1))
                    beta, pieces = mixed(g, h)
                    rest = (rest - beta) % p
                    local.append(dict(where=("local", t), beta=beta, pieces=pieces))
            if not winograd_only:
                uses = [rng.choice(dense)] if rng.random() < 0.5 else []
                g = new_word(xv(i, ks[0]), yv(j, ks[-1]))
                uses += [g] * rng.randint(0, 3)
                for g in uses:  # a link-specific or dense word reused against fresh partners
                    h = new_word([rng.randrange(p) for _ in range(Xd)], [rng.randrange(p) for _ in range(Yd)])
                    beta, pieces = mixed(g, h)
                    rest = (rest - beta) % p
                    local.append(dict(where=("local", t), beta=beta, pieces=pieces))
            for R in rank_one_terms(rest, p):
                local.append(dict(where=("local", t), beta=R, pieces=None))
            links.append(dict(pos=(i, j), s=s, kind=kind, Z=Z, beta=beta_t, local=local))
            products.extend(local)

    kept = [t for t, l in enumerate(links) if l["kind"] != "two"]
    if len(kept) < 2:
        return None
    k = len(kept)
    idx = {t: n for n, t in enumerate(kept)}
    rows_of = [frozenset([(links[t]["pos"][0], links[t]["s"])]) for t in kept]
    cols_of = [frozenset([(links[t]["pos"][1], links[t]["s"])]) for t in kept]

    def ancestors(node):
        out = set()
        while node:
            out.add(node)
            node = F_par[node]
        return out

    below = {node: set() for node in range(1, len(F_val))}
    for t in kept:
        if links[t]["Z"] is not None:
            for node in ancestors(links[t]["Z"]):
                below[node].add(idx[t])
    tree = binary_hierarchy([frozenset(S) for S in below.values()], range(k))
    a1, b, _ = set_labels(tree, k, rows_of, cols_of, p)
    inv = [pow(int(links[t]["beta"]), p - 2, p) for t in kept]
    a = (a1 * np.array(inv, dtype=np.int64)[None, :]) % p

    def label_set(where):
        kind, x = where
        if kind == "F":
            return frozenset(below[x])
        return frozenset([idx[x]]) if x in idx else frozenset()

    def Lam(S):
        out = np.zeros((a.shape[0], b.shape[0]), dtype=np.int64)
        for n in S:
            out = out + links[kept[n]]["beta"] * np.outer(a[:, n], b[:, n])
        return out % p

    lhs = flatten(a, b, {n: shape.D(links[t]["pos"], links[t]["s"]) for n, t in enumerate(kept)}, {}, p)
    rhs = np.zeros_like(lhs)
    other, by_word = 0, {}
    for e in products:
        S = label_set(e["where"])
        if not S:
            continue
        La = Lam(S) if e["where"][0] == "F" else np.outer(a[:, next(iter(S))], b[:, next(iter(S))]) % p
        assert rank(La, p) <= 1
        rhs = rhs + np.kron(La, e["beta"])
        if e["pieces"] is None:
            other += 1
        else:
            for w, piece in e["pieces"]:
                by_word.setdefault(w, []).append((S, np.kron(La, piece)))
    assert np.array_equal(lhs % p, rhs % p), "flattening identity fails"
    rank_phi = rank(lhs, p)
    assert rank_phi == shape.L * k
    nu_total = 0
    for w, items in by_word.items():
        nu = rank(sum(m for _, m in items) % p, p)
        assert nu <= len({S for S, _ in items}), "nu_w exceeds the number of label sets"
        nu_total += nu
    assert rank_phi <= other + nu_total, "rank bound fails"
    n_products = len(products)
    assert 2 * n_products >= 2 * rank_phi - nu_total, "mixed charge fails"
    return n_products, rank_phi, nu_total


def tree_bounds(trees, L):
    """trees: {tree: {pos: [correct levels]}}, tree 0 rooted at 0. The per-tree bounds of Theorem B."""
    out = {}
    for tau, levels in trees.items():
        hi = sum(max(c) for c in levels.values())
        lo = min((min(c) for c in levels.values()), default=0)
        out[tau] = L * hi if tau == 0 else L * (hi - lo)
    return out


def covering_holds(trees, L):
    G = sum(len(c) for levels in trees.values() for c in levels.values())
    H = sum(1 for tau, levels in trees.items() if tau != 0 and levels)
    return sum(tree_bounds(trees, L).values()) >= L * (G - H), sum(tree_bounds(trees, L).values()) == L * (G - H)


def part5a():
    """The covering lemma, exhaustively: every assignment of every set of correct targets (2 positions, 3 levels) to
    tau_0 and up to 3 head trees."""
    import itertools
    count, tight = 0, 0
    slots = [(pos, c) for pos in range(2) for c in range(1, 4)]
    for mask in range(1, 1 << len(slots)):
        G = [slots[i] for i in range(len(slots)) if mask >> i & 1]
        for assign in itertools.product(range(4), repeat=len(G)):
            trees = {}
            for (pos, c), tau in zip(G, assign):
                trees.setdefault(tau, {}).setdefault(pos, []).append(c)
            ok, eq = covering_holds(trees, 1)
            assert ok, f"covering lemma fails: {trees}"
            count += 1
            tight += eq
    print(f"[covering] {count} assignments of correct targets to trees (2 positions x 3 levels, tau_0 + 3 heads): "
          f"sum of per-tree bounds >= L*(|G_T| - H): ok ({tight} tight)")


def merge_dag(shape, rng, p):
    """A random computation of prefix targets with merges: continued and restarted running sums, cheap merges of
    the running sum with a partial accumulator, heads shared by several positions, wrong commits. Splits it into
    merge-free trees and checks each tree's bound through A1a's flattening on the tree re-rooted at 0 via its lowest
    correct target, then Theorem B's total."""
    Xd, Yd = shape.P * shape.m, shape.Q * shape.m
    zero = np.zeros((Xd, Yd), dtype=np.int64)
    val, par, step = [zero], [None], [None]  # par None: root (node 0) or a merge (head)

    def prod_path(node, target):
        for R in rank_one_terms(target - val[node], p):
            val.append((val[node] + R) % p)
            par.append(node)
            step.append(R)
            node = len(val) - 1
        return node

    def merge(a, b, alpha=None, beta=None):
        alpha = alpha or rng.randint(1, p - 1)
        beta = beta or rng.randint(1, p - 1)
        val.append((alpha * val[a] + beta * val[b]) % p)
        par.append(None)
        step.append(None)
        return len(val) - 1

    def partial_cell(pos, s):
        keep = [k for k in range(s * shape.L, (s + 1) * shape.L) if rng.random() < 0.6]
        mask = np.zeros(shape.m, dtype=np.int64)
        mask[keep] = 1
        return shape.D(pos, s) * np.tile(mask, shape.P)[:, None]

    positions = [(i, j) for i in range(shape.P) for j in range(shape.Q)]
    for _ in range(rng.randint(1, 3)):
        piece = sum((partial_cell(pos, rng.randrange(shape.S)) for pos in rng.sample(positions, 2)), zero)
        prod_path(rng.randrange(len(val)), piece % p)
    heads, committed, cur = [], {}, {pos: 0 for pos in positions}
    for c in range(1, shape.S + 1):
        for pos in rng.sample(positions, len(positions)):
            target, correct = shape.T(pos, c), True
            action = rng.choice(["continue", "restart", "merge", "cheap", "cheap", "host", "wrong"])
            if action == "host" and not heads:
                action = "merge"
            if action == "continue":
                start = cur[pos]
            elif action == "restart":
                start = rng.randrange(len(val))
            elif action == "merge":
                start = merge(rng.randrange(len(val)), rng.randrange(len(val)))
                heads.append(start)
            elif action == "cheap":
                Z = prod_path(rng.randrange(len(val)), partial_cell(pos, c - 1))
                start = merge(cur[pos], Z, 1, 1)
                heads.append(start)
            elif action == "host":
                start = rng.choice(heads)
            else:
                start, correct = cur[pos], False
                R = np.outer([rng.randrange(p) for _ in range(Xd)], [rng.randrange(p) for _ in range(Yd)])
                target = (target + R) % p
            node = prod_path(start, target)
            committed[(pos, c)] = (node, correct and np.array_equal(val[node] % p, shape.T(pos, c) % p))
            cur[pos] = node

    def root_of(x):
        while par[x] is not None:
            x = par[x]
        return x

    def chain(x):
        out = []
        while par[x] is not None:
            out.append(x)
            x = par[x]
        return out

    trees = {}
    for (pos, c), (node, ok) in committed.items():
        if ok:
            trees.setdefault(root_of(node), {}).setdefault(pos, []).append((c, node))
    n_products = {}
    for x in range(1, len(val)):
        if par[x] is not None:
            n_products[root_of(x)] = n_products.get(root_of(x), 0) + 1
    bounds = tree_bounds({tau: {pos: [c for c, _ in cs] for pos, cs in lv.items()} for tau, lv in trees.items()},
                         shape.L)
    for tau, levels in trees.items():
        tops = [(pos, *max(cs)) for pos, cs in levels.items()]  # (pos, hi, node)
        k = len(tops)
        star = None if tau == 0 else min(((c, node) for cs in levels.values() for c, node in cs))
        virtual = [] if star is None else rank_one_terms(val[star[1]], p)
        rev = set() if star is None else set(chain(star[1]))
        edges = [(frozenset(range(k)), R) for R in virtual]
        for x in range(1, len(val)):
            if par[x] is None or root_of(x) != tau:
                continue
            below = frozenset(t for t, (_, _, node) in enumerate(tops) if x in chain(node))
            edges.append((frozenset(range(k)) - below, -step[x]) if x in rev else (below, step[x]))
        assert len(edges) == n_products.get(tau, 0) + len(virtual)
        rows_of = [frozenset((pos[0], s) for s in range(hi)) for pos, hi, _ in tops]
        cols_of = [frozenset((pos[1], s) for s in range(hi)) for pos, hi, _ in tops]
        if k == 1:
            a = b = np.ones((1, 1), dtype=np.int64)
        else:
            a, b, _ = set_labels(binary_hierarchy([S for S, _ in edges], range(k)), k, rows_of, cols_of, p)
        lhs = flatten(a, b, {t: shape.T(pos, hi) for t, (pos, hi, _) in enumerate(tops)}, {}, p)
        rhs = np.zeros_like(lhs)
        for S, R in edges:
            Lam = sum((np.outer(a[:, t], b[:, t]) for t in S), np.zeros((a.shape[0], b.shape[0]), dtype=np.int64)) % p
            assert rank(Lam, p) <= 1, "edge label sum has rank > 1"
            rhs = rhs + np.kron(Lam, R)
        assert np.array_equal(lhs % p, rhs % p), "flattening identity fails on the re-rooted tree"
        assert rank(lhs, p) == shape.L * sum(hi for _, hi, _ in tops), "Lemma 2(b) fails"
        assert n_products.get(tau, 0) >= bounds[tau], "per-tree bound fails"
    G = sum(len(cs) for lv in trees.values() for cs in lv.values())
    H = sum(1 for tau in trees if tau != 0)
    M = sum(1 for x in range(1, len(val)) if par[x] is None)
    assert H <= M
    assert sum(bounds.values()) >= shape.L * (G - H), "covering fails"
    total = sum(n_products.values())
    assert total >= shape.L * (G - H), "Theorem B fails"
    multi = sum(1 for tau, lv in trees.items() if tau != 0 and len(lv) >= 2)
    return total, shape.L * (G - H), multi


def part5(rng):
    part5a()
    count, tight, multi = 0, 0, 0
    for (P, Q, m, L, trials) in ((2, 2, 4, 2, 60), (2, 3, 4, 2, 30), (2, 2, 6, 3, 30), (3, 3, 2, 1, 20)):
        shape = Shape(P, Q, m, L)
        for _ in range(trials):
            for p in (3, 10007):
                total, bound, n_multi = merge_dag(shape, rng, p)
                count += 1
                tight += total == bound
                multi += n_multi
    print(f"[merges] {count} random computations with continued and restarted running sums, cheap merges, shared "
          "heads and wrong commits, GF(3) and GF(10007): per-tree flattening on the re-rooted tree, per-tree bound "
          f"L*(sum hi - min lo), #products >= L*(|G_T| - H): ok ({tight} tight, {multi} head trees hosting "
          "2+ positions)")


def part4(rng):
    count = 0
    for (P, Q, m, L, trials) in ((2, 2, 4, 2, 60), (2, 3, 4, 4, 30), (2, 2, 6, 3, 30)):
        shape = Shape(P, Q, m, L)
        for _ in range(trials):
            for p in (3, 10007):
                if mixed_chain_form(shape, rng, p) is not None:
                    count += 1
    shape = Shape(2, 2, 4, 4)
    n_products, rank_phi, nu_total = mixed_chain_form(shape, rng, 10007, winograd_only=True)
    assert 2 * n_products == 2 * rank_phi - nu_total == shape.P * shape.Q * shape.m
    print(f"[mixed] {count} random chain-form computations with reused mixed words, GF(3) and GF(10007): identity, "
          "nu_w <= #label sets, #products >= rank - nu/2: ok; Winograd on 2x2x4, L=4: "
          f"{n_products} products = {rank_phi} - {nu_total}/2 (tight)")


if __name__ == "__main__":
    rng = random.Random(20261015)
    parts = sys.argv[1:] or ["1", "3", "4", "5"]
    if "1" in parts:
        part1_2(rng)
    if "3" in parts:
        part3(rng)
    if "4" in parts:
        part4(rng)
    if "5" in parts:
        part5(rng)

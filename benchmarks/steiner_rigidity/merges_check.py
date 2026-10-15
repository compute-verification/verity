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
Part 6: one word added to the running sums of single-level links saves at most L over all its uses, and the links
it helps lie in one D-row or one D-column.
Part 7: Theorem C end to end on random computations that reuse accumulators and merge-built words across positions,
and Strassen inside one block, which shows the per-word charge W' cannot be dropped.
Part 8: links adding combinations of d words: the one-sided certificate and the d*L bound when no D-column holds two
links, and the d^2*L bound on any set of cells.
Part 9: the label certificate past the shared labels: rank-1 labels from a quadric through the points of ker B, on the
instance where row-, column- and pool-shared labels fall short.
Part 10: mixed products charged by junk: on a segment between special nodes, #products >= rank(bilinear part) - j/2,
with j the smaller junk rank of the segment's net change (0 when its ends share a junk-free side); Winograd is tight.
Part 11: recursive Strassen inside one block saves L - L^(log2 7 - 2) per merge-built word, so Theorem C's charge of
L per distinct word is asymptotically tight.
Part 12: symmetric Lemma C: links adding one word save at most L in total with mixed products counted once, whatever
junk the word carries (on a row star, sum_t rank >= 2L(k-1) + (k-2) rank Z_YY for any word, step by step).
Part 13: d*L for two words on a cross (a D-row and a D-column): labels on the two sides coupled through one direction
lam cancel both words and reach rank P >= k - rank B, unless each side adds one word.
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check import nullspace, rank, rref


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
            node = {"S": [t], "D": [t], "E": [t], "rho": set(rows_of[t]), "gam": set(cols_of[t])}
        else:
            A, B = go(t[0]), go(t[1])
            if not (A["rho"] & B["rho"]):
                xcons.append((A["D"], B["D"]))
                node = {"S": A["S"] + B["S"], "D": A["D"], "E": A["E"] + B["E"], "rho": A["rho"] | B["rho"],
                            "gam": A["gam"] & B["gam"]}
            else:
                assert not (A["gam"] & B["gam"]), "both merges blocked"
                ycons.append((A["E"], B["E"]))
                node = {"S": A["S"] + B["S"], "D": A["D"] + B["D"], "E": A["E"], "rho": A["rho"] & B["rho"],
                            "gam": A["gam"] | B["gam"]}
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
                links.append({"pos": pos, "s": s, "kind": kind, "Z": None, "beta": 1, "local": rank_one_terms(D, p)})
            elif kind == "one":
                Z, beta = rng.randrange(1, len(F_val)), rng.randint(1, p - 1)
                links.append({"pos": pos, "s": s, "kind": kind, "Z": Z, "beta": beta,
                                  "local": rank_one_terms(D - beta * F_val[Z], p)})
            else:
                Z1, Z2 = rng.randrange(1, len(F_val)), rng.randrange(1, len(F_val))
                rest = D - F_val[Z1] - F_val[Z2]
                links.append({"pos": pos, "s": s, "kind": kind, "Z": None, "beta": 1, "local": rank_one_terms(rest, p)})

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
            products.append({"where": ("F", len(F_val) - 1), "beta": beta, "pieces": pieces})

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
                    local.append({"where": ("local", t), "beta": beta, "pieces": pieces})
            if not winograd_only:
                uses = [rng.choice(dense)] if rng.random() < 0.5 else []
                g = new_word(xv(i, ks[0]), yv(j, ks[-1]))
                uses += [g] * rng.randint(0, 3)
                for g in uses:  # a link-specific or dense word reused against fresh partners
                    h = new_word([rng.randrange(p) for _ in range(Xd)], [rng.randrange(p) for _ in range(Yd)])
                    beta, pieces = mixed(g, h)
                    rest = (rest - beta) % p
                    local.append({"where": ("local", t), "beta": beta, "pieces": pieces})
            for R in rank_one_terms(rest, p):
                local.append({"where": ("local", t), "beta": R, "pieces": None})
            links.append({"pos": (i, j), "s": s, "kind": kind, "Z": Z, "beta": beta_t, "local": local})
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


def random_word(shape, cells, rng, p):
    """A word built to help several D-cells: partials of a row or column star whose inner indices partition the block
    (the tight family), partials of random cells, sums of whole cells, or a random low-rank matrix; maybe plus noise."""
    Xd, Yd = shape.P * shape.m, shape.Q * shape.m
    Z = np.zeros((Xd, Yd), dtype=np.int64)

    def partial(pos, s, ks):
        out = np.zeros_like(Z)
        for k in ks:
            out[pos[0] * shape.m + k, pos[1] * shape.m + k] = 1
        return out

    kind = rng.choice(["row", "col", "cells", "whole", "lowrank"])
    if kind in ("row", "col"):
        i, j, s = rng.randrange(shape.P), rng.randrange(shape.Q), rng.randrange(shape.S)
        star = [((i, jj), s) for jj in range(shape.Q)] if kind == "row" else [((ii, j), s) for ii in range(shape.P)]
        star = rng.sample(star, rng.randint(1, len(star)))
        ks = list(range(s * shape.L, (s + 1) * shape.L))
        rng.shuffle(ks)
        owner = [rng.randrange(len(star)) for _ in ks]
        for n, (pos, _) in enumerate(star):
            Z = Z + rng.randint(1, p - 1) * partial(pos, s, [k for k, o in zip(ks, owner) if o == n])
    elif kind == "cells":
        for pos, s in rng.sample(cells, rng.randint(1, 4)):
            ks = [k for k in range(s * shape.L, (s + 1) * shape.L) if rng.random() < 0.6]
            Z = Z + rng.randint(1, p - 1) * partial(pos, s, ks)
    elif kind == "whole":
        for pos, s in rng.sample(cells, rng.randint(1, 3)):
            Z = Z + rng.randint(1, p - 1) * shape.D(pos, s)
    else:
        for _ in range(rng.randint(1, 3)):
            Z = Z + np.outer([rng.randrange(p) for _ in range(Xd)], [rng.randrange(p) for _ in range(Yd)])
    if rng.random() < 0.3:
        Z = Z + np.outer([rng.randrange(p) for _ in range(Xd)], [rng.randrange(p) for _ in range(Yd)])
    return Z % p


def part6(rng):
    """One word Z added (times any beta_t) to the running sums of any set of single-level links: link t needs at least
    rank(D_t - beta_t Z) more products, so it saves s_t = L - that. Over every D-cell, with each beta_t chosen to help:
    the cells with s_t > 0 form a star (one D-row or one D-column) and sum s_t <= L."""
    count, tight, multi = 0, 0, 0
    for (P, Q, m, L, p, trials) in ((2, 2, 4, 2, 3, 300), (2, 3, 4, 2, 3, 200), (3, 3, 4, 2, 5, 80),
                                    (2, 3, 6, 3, 3, 120), (2, 2, 4, 4, 3, 100)):
        shape = Shape(P, Q, m, L)
        cells = [((i, j), s) for i in range(P) for j in range(Q) for s in range(shape.S)]
        for _ in range(trials):
            Z = random_word(shape, cells, rng, p)
            savings = {}
            for pos, s in cells:
                r = min(rank((shape.D(pos, s) - b * Z) % p, p) for b in range(1, p))
                if r < L:
                    savings[(pos, s)] = L - r
            rows = {(pos[0], s) for pos, s in savings}
            cols = {(pos[1], s) for pos, s in savings}
            assert len(rows) <= 1 or len(cols) <= 1, "positive savers are not a star"
            assert sum(savings.values()) <= L, "one word saves more than L"
            count += 1
            tight += sum(savings.values()) == L
            multi += sum(savings.values()) == L and len(savings) >= 2
    print(f"[per word] {count} words (row and column star splits, partial and whole cells, low rank, noise) against "
          "every D-cell with the best beta per cell: positive savers form a star and save <= L in total: ok "
          f"({tight} save exactly L, {multi} of them across 2+ cells)")


def reuse_dag(shape, rng, p):
    """A random computation with words reused across positions. Classifies one-addend links, runs the flattening on
    tau_0 with the links that add a tau_0 word, checks Lemma C per added word outside tau_0, Theorem B's per-tree bound
    on the other heads, and #products >= L*(|G_T| - H' - W')."""
    Xd, Yd = shape.P * shape.m, shape.Q * shape.m
    zero = np.zeros((Xd, Yd), dtype=np.int64)
    val, par, step, mrg = [zero], [None], [None], [None]

    def prod_path(node, target):
        for R in rank_one_terms(target - val[node], p):
            val.append((val[node] + R) % p)
            par.append(node)
            step.append(R)
            mrg.append(None)
            node = len(val) - 1
        return node

    def merge(a, z, alpha, beta):
        val.append((alpha * val[a] + beta * val[z]) % p)
        par.append(None)
        step.append(None)
        mrg.append((a, alpha, z, beta))
        return len(val) - 1

    def partial(pos, s, ks):
        out = np.zeros_like(zero)
        for k in ks:
            out[pos[0] * shape.m + k, pos[1] * shape.m + k] = 1
        return out

    positions = [(i, j) for i in range(shape.P) for j in range(shape.Q)]
    acc = []  # tau_0 accumulators: (node, cell it was built for)
    for _ in range(rng.randint(2, 5)):
        pos, s = rng.choice(positions), rng.randrange(shape.S)
        ks = [k for k in range(s * shape.L, (s + 1) * shape.L) if rng.random() < 0.6]
        acc.append((prod_path(rng.choice([0] + [a for a, _ in acc]), partial(pos, s, ks)), (pos, s)))
    words = []  # merge-built words: (node, {cell: coefficient})
    for _ in range(rng.randint(1, 3)):
        i, s = rng.randrange(shape.P), rng.randrange(shape.S)
        star = rng.sample([((i, j), s) for j in range(shape.Q)], rng.randint(2, shape.Q))
        ks = list(range(s * shape.L, (s + 1) * shape.L))
        owner = [rng.randrange(len(star)) for _ in ks]
        coef, node = {}, None
        for n, cell in enumerate(star):
            g = rng.randint(1, p - 1)
            Pn = prod_path(0, partial(cell[0], s, [k for k, o in zip(ks, owner) if o == n]) * g % p)
            node = Pn if node is None else merge(node, Pn, 1, 1)
            coef[cell] = g
        words.append((node, coef))
    committed, cur = {}, {pos: 0 for pos in positions}
    for c in range(1, shape.S + 1):
        for pos in rng.sample(positions, len(positions)):
            target, s = shape.T(pos, c), c - 1
            action = rng.choice(["continue", "acc", "word", "word", "two", "restart", "wrong"])
            if action == "continue":
                start = cur[pos]
            elif action == "acc":
                helpful = [a for a, cell in acc if cell == (pos, s)]
                Z = rng.choice(helpful) if helpful and rng.random() < 0.7 else rng.choice(acc)[0]
                start = merge(cur[pos], Z, 1, rng.randint(1, p - 1) if Z not in helpful else 1)
            elif action == "word":
                Z, coef = rng.choice(words)
                beta = pow(coef[(pos, s)], p - 2, p) if (pos, s) in coef else rng.randint(1, p - 1)
                start = merge(cur[pos], Z, 1, beta)
            elif action == "two":
                start = merge(merge(cur[pos], rng.choice(acc)[0], 1, 1), rng.choice(words)[0], 1, 1)
            elif action == "restart":
                start = rng.randrange(len(val))
            else:
                start = cur[pos]
                target = (target + np.outer([rng.randrange(p) for _ in range(Xd)],
                                            [rng.randrange(p) for _ in range(Yd)])) % p
            node = prod_path(start, target)
            committed[(pos, c)] = (node, np.array_equal(val[node] % p, shape.T(pos, c) % p))
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
    links = {}  # head -> (pos, c, node, Z, beta, path length)
    for v, lv in trees.items():
        if v == 0 or mrg[v][1] != 1 or sum(len(cs) for cs in lv.values()) != 1:
            continue
        (pos, [(c, node)]), = lv.items()
        if np.array_equal(val[mrg[v][0]] % p, shape.T(pos, c - 1) % p):
            links[v] = (pos, c, node, mrg[v][2], mrg[v][3], len(chain(node)))
    hi0 = {pos: max(c for c, _ in cs) for pos, cs in trees.get(0, {}).items()}
    tau0_links = {v: l for v, l in links.items() if root_of(l[3]) == 0 and l[1] > hi0.get(l[0], 0)}
    # Flattening on tau_0: tops (anchored at their nodes) and tau_0 links (anchored at Z, scaled by beta).
    fam = [(pos, hi0[pos], node, 1, None) for pos, cs in trees.get(0, {}).items() for c, node in cs if c == hi0[pos]]
    fam += [(l[0], l[1], l[3], l[4], v) for v, l in tau0_links.items()]
    k = len(fam)
    if k:
        rows_of = [frozenset((pos[0], s) for s in (range(c) if v is None else [c - 1])) for pos, c, _, _, v in fam]
        cols_of = [frozenset((pos[1], s) for s in (range(c) if v is None else [c - 1])) for pos, c, _, _, v in fam]
        mats = {t: shape.T(pos, c) if v is None else shape.D(pos, c - 1) for t, (pos, c, _, _, v) in enumerate(fam)}
        edges = []
        for x in range(1, len(val)):
            if par[x] is not None and root_of(x) == 0:
                edges.append((frozenset(t for t, f in enumerate(fam) if x in chain(f[2])), step[x]))
        if k == 1:
            a1 = b = np.ones((1, 1), dtype=np.int64)
        else:
            a1, b, _ = set_labels(binary_hierarchy([S for S, _ in edges], range(k)), k, rows_of, cols_of, p)
        a = (a1 * np.array([pow(int(f[3]), p - 2, p) for f in fam], dtype=np.int64)[None, :]) % p
        lhs = flatten(a, b, mats, {}, p)
        rhs = np.zeros_like(lhs)
        for S, R in edges:
            Lam = sum((fam[t][3] * np.outer(a[:, t], b[:, t]) for t in S),
                      np.zeros((a.shape[0], b.shape[0]), dtype=np.int64)) % p
            assert rank(Lam, p) <= 1, "edge label sum has rank > 1"
            rhs = rhs + np.kron(Lam, R)
        for t, (_, _, _, _, v) in enumerate(fam):
            if v is not None:
                for x in chain(links[v][2]):
                    rhs = rhs + np.kron(np.outer(a[:, t], b[:, t]), step[x])
        assert np.array_equal(lhs % p, rhs % p), "flattening identity fails on tau_0 with its links"
        assert rank(lhs, p) == sum(rank(M, p) for M in mats.values()), "Lemma 2(b) fails"
        assert n_products.get(0, 0) + sum(l[5] for l in tau0_links.values()) >= rank(lhs, p)
    by_word = {}
    for v, l in links.items():
        if root_of(l[3]) != 0:
            by_word.setdefault(l[3], []).append(l[5])
    for Z, lens in by_word.items():
        assert sum(lens) >= shape.L * (len(lens) - 1), "Lemma C fails"
    bounds = tree_bounds({tau: {pos: [c for c, _ in cs] for pos, cs in lv.items()} for tau, lv in trees.items()},
                         shape.L)
    for tau in trees:
        if tau not in links:
            assert n_products.get(tau, 0) >= bounds[tau], "per-tree bound fails"
    G = sum(len(cs) for lv in trees.values() for cs in lv.values())
    H = sum(1 for tau in trees if tau != 0)
    H1, W1 = sum(1 for tau in trees if tau != 0 and tau not in links), len(by_word)
    total = sum(n_products.values())
    assert total >= shape.L * (G - H1 - W1), "Theorem C fails"
    saved = sum(max(0, shape.L - l[5]) for l in links.values() if root_of(l[3]) != 0)
    return H - H1 - W1, sum(1 for lens in by_word.values() if len(lens) >= 2), saved


def strassen_words(L, p=10007):
    """Strassen inside one k-block of a 2x2 grid (halves of L/2 inner indices as the 2x2 split): seven tau_0
    accumulators of L/2 products each, merged into the four outputs, each output a one-addend link from 0."""
    shape, h2 = Shape(2, 2, L, L), L // 2
    Xd = Yd = 2 * L

    def side(coefs, k, d):
        v = np.zeros(d, dtype=np.int64)
        for (a, h), c in coefs.items():
            v[a * L + h * h2 + k] += c
        return v

    A = lambda i, h: {(i, h): 1}
    B = lambda h, j: {(j, h): 1}

    def add(*ds):
        out = {}
        for sgn, d in ds:
            for key, c in d.items():
                out[key] = out.get(key, 0) + sgn * c
        return out

    Ms = [(add((1, A(0, 0)), (1, A(1, 1))), add((1, B(0, 0)), (1, B(1, 1)))),
          (add((1, A(1, 0)), (1, A(1, 1))), B(0, 0)),
          (A(0, 0), add((1, B(0, 1)), (-1, B(1, 1)))),
          (A(1, 1), add((1, B(1, 0)), (-1, B(0, 0)))),
          (add((1, A(0, 0)), (1, A(0, 1))), B(1, 1)),
          (add((1, A(1, 0)), (-1, A(0, 0))), add((1, B(0, 0)), (1, B(0, 1)))),
          (add((1, A(0, 1)), (-1, A(1, 1))), add((1, B(1, 0)), (1, B(1, 1))))]
    Mv = [sum(np.outer(side(a, k, Xd), side(b, k, Yd)) for k in range(h2)) % p for a, b in Ms]
    outputs = {(0, 0): [(1, 0), (1, 3), (-1, 4), (1, 6)], (0, 1): [(1, 2), (1, 4)], (1, 0): [(1, 1), (1, 3)],
               (1, 1): [(1, 0), (-1, 1), (1, 2), (1, 5)]}
    for pos, terms in outputs.items():
        W = sum(s * Mv[i] for s, i in terms) % p
        assert np.array_equal(W, shape.D(pos, 0) % p), "Strassen output wrong"
    n_products, G, H1, W1 = 7 * h2, 4, 0, 4
    assert n_products >= L * (G - H1 - W1), "Theorem C fails on Strassen"
    assert n_products < L * G, "Strassen should beat L*|G_T|"
    return n_products, L * G


def part7(rng):
    count, gain, reused, saved = 0, 0, 0, 0
    for (P, Q, m, L, trials) in ((2, 2, 4, 2, 60), (2, 3, 4, 2, 40), (2, 3, 6, 3, 30), (3, 3, 4, 4, 15)):
        shape = Shape(P, Q, m, L)
        for _ in range(trials):
            for p in (3, 10007):
                g, r, s = reuse_dag(shape, rng, p)
                count, gain, reused, saved = count + 1, gain + g, reused + r, saved + s
    print(f"[reuse] {count} random computations reusing tau_0 accumulators and merge-built words across positions, "
          "GF(3) and GF(10007): flattening on tau_0 with its one-addend links, Lemma C per added word, per-tree "
          f"bound on other heads, #products >= L*(|G_T| - H' - W'): ok (H - H' - W' = {gain} heads no longer charged; "
          f"{reused} words added to 2+ running sums; {saved} products saved by links adding merge-built words)")
    for L in (2, 4, 6):
        n_products, full = strassen_words(L)
        print(f"[reuse] Strassen in one block, L={L}: four merge-built words, one link each, {n_products} products "
              f"< L*|G_T| = {full}, so W' cannot be dropped; Theorem C holds")


def part8(rng):
    """Links adding totals from the span of d words (best combination per cell). With at most one link per D-column
    the one-sided certificate (labels a basis of ker B on the X side) gives rank Psi >= L*(k - rank B), hence
    sum_t rank(D_t - W_t) >= L*(k - rank B); on any set of cells, sum_t rank(D_t - W_t) >= L*(k - d^2)."""
    import itertools
    count, saved_total, general = 0, 0, 0
    for (P, Q, m, L, p, trials) in ((2, 2, 4, 2, 3, 60), (3, 2, 4, 2, 3, 60), (3, 3, 2, 2, 3, 40), (2, 3, 4, 4, 3, 30)):
        shape = Shape(P, Q, m, L)
        cells = [((i, j), s) for i in range(P) for j in range(Q) for s in range(shape.S)]
        for _ in range(trials):
            d = rng.choice([2, 3])
            Zs = [random_word(shape, cells, rng, p) for _ in range(d)]
            combos = [bs for bs in itertools.product(range(p), repeat=d) if any(bs)]

            def best(cell, shape=shape, Zs=Zs, p=p, combos=combos):
                D = shape.D(*cell)
                return min((rank((D - sum(b * Z for b, Z in zip(bs, Zs))) % p, p), bs) for bs in combos)

            chosen = {}
            for cell in rng.sample(cells, len(cells)):
                col = (cell[0][1], cell[1])
                if col not in chosen:
                    chosen[col] = cell
            links = list(chosen.values())
            res = [best(c) for c in links]
            B = np.array([bs for _, bs in res], dtype=np.int64).T % p
            k, rB = len(links), rank(B, p)
            U = nullspace(B, k, p)  # rows: a basis of ker B; u_t = U[:, t]
            Ws = [sum(b * Z for b, Z in zip(bs, Zs)) % p for _, bs in res]
            Psi_D = sum(np.kron(U[:, t][:, None], shape.D(*links[t])) for t in range(k)) % p
            Psi_DW = sum(np.kron(U[:, t][:, None], (shape.D(*links[t]) - Ws[t]) % p) for t in range(k)) % p
            assert np.array_equal(Psi_D, Psi_DW), "W-terms do not cancel"
            assert rank(Psi_D, p) >= L * (k - rB), "one-sided certificate too weak"
            assert sum(r for r, _ in res) >= L * (k - rB), "d*L bound fails (one link per D-column)"
            saved_total += sum(max(0, L - r) for r, _ in res)
            res_all = [best(c) for c in cells]
            assert sum(r for r, _ in res_all) >= L * (len(cells) - d * d), "d^2*L bound fails"
            general = max(general, sum(max(0, L - r) for r, _ in res_all) / (d * L))
            count += 1
    print(f"[d words] {count} spans of 2-3 words against cell sets with <= 1 link per D-column (best combination per "
          "cell): W-terms cancel in the one-sided certificate, rank Psi >= L*(k - rank B), savings <= L*rank B: ok "
          f"({saved_total} products saved in total); on all cells savings <= d^2*L: ok (max seen {general:.2f}*d*L)")


def quadric_labels(cells, B, rng, p, tries=200):
    """Rank-1 2x2 labels M_t = phi(p_t) with sum_t B[j, t] M_t = 0, where p_t = U[:, t] are the points of ker B
    (dim 4) and det phi is a split nondegenerate quadric through them; the labels with the largest rank of P (the
    D-grid matrix with block M_t at cell t), or None."""
    import itertools
    k = len(cells)
    U = nullspace(B, k, p)
    if U.shape[0] != 4:
        return None
    pts = [U[:, t] for t in range(k)]
    inv = lambda a: pow(int(a) % p, p - 2, p)
    roots = {x * x % p: x for x in range(p)}
    mons = list(itertools.combinations_with_replacement(range(4), 2))
    Qs = nullspace(np.array([[x[i] * x[j] % p for i, j in mons] for x in pts]), len(mons), p)
    nR, nC = 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)
    best = None
    for _ in range(tries):
        q = sum(rng.randrange(p) * Qs[i] for i in range(len(Qs))) % p
        S = np.zeros((4, 4), dtype=np.int64)
        for (i, j), c in zip(mons, q):
            S[i, j] = S[j, i] = c if i == j else c * inv(2) % p
        if rank(S, p) < 4:
            continue
        f = lambda a, b, S=S: int(a @ S @ b) % p
        u = pts[0]
        v = next(e for e in np.eye(4, dtype=np.int64) if f(u, e))
        v = v * inv(f(u, v)) % p
        v = (v - f(v, v) * inv(2) * u) % p
        a, b = nullspace(np.array([u @ S % p, v @ S % p]), 4, p)
        A, Bc, C = f(a, a), f(a, b), f(b, b)
        if A and (Bc * Bc - A * C) % p not in roots:
            continue
        w1 = a if A == 0 else ((-Bc + roots[(Bc * Bc - A * C) % p]) * inv(A) * a + b) % p
        w2 = a if f(w1, a) else b
        w2 = w2 * inv(f(w1, w2)) % p
        w2 = (w2 - f(w2, w2) * inv(2) * w1) % p
        basis = np.stack([u, v, w1, w2], axis=1)
        for transpose in (False, True):
            labels = []
            for x in pts:
                N = nullspace(np.concatenate([basis, x.reshape(4, 1)], axis=1) % p, 5, p)[0]
                y = -N[:4] * inv(N[4]) % p
                M = np.array([[y[0], y[2]], [-y[3], y[1]]]) % p
                labels.append(M.T if transpose else M)
            Pm = np.zeros((2 * nR, 2 * nC), dtype=np.int64)
            for (r, c), M in zip(cells, labels):
                Pm[2 * r:2 * r + 2, 2 * c:2 * c + 2] = M
            if best is None or rank(Pm, p) > best[0]:
                best = (rank(Pm, p), labels)
    return best


def part9(rng):
    """Quadric labels for the label certificate. On 7 cells of a 3x3 D-grid with a generic rank-3 B, row-, column-
    and pool-shared labels reach rank P = 3 < k - rank B = 4; labels from a quadric through the points of ker B have
    rank 1, cancel every word, and reach rank P = 6, so these links save at most L for any words. Then random
    instances with dim ker B = 4: how often quadric labels reach k - rank B."""
    p = 101
    pos = [(0, 0), (1, 1), (2, 1), (2, 2), (1, 2), (0, 2), (1, 0)]
    B = np.array([[35, 14, 20, 75, 22, 7, 44], [47, 14, 32, 63, 72, 4, 63], [28, 82, 39, 94, 70, 85, 23]])
    k, rB = len(pos), rank(B, p)
    rP, labels = quadric_labels(pos, B, rng, p)
    for M in labels:
        assert rank(M, p) == 1, "label not rank 1"
    for j in range(B.shape[0]):
        assert not (sum(B[j, t] * labels[t] for t in range(k)) % p).any(), "labels do not cancel the words"
    assert rP >= k - rB, "quadric labels too weak"
    for L in (2, 4):
        shape = Shape(3, 3, L, L)
        cells = [(ij, 0) for ij in pos]
        Ds = [shape.D(*c) for c in cells]
        Psi_D = sum(np.kron(labels[t], Ds[t]) for t in range(k)) % p
        assert rank(Psi_D, p) == L * rP, "rank Psi != L * rank P"
        for _ in range(10):
            Zs = [random_word(shape, cells, rng, p) for _ in range(B.shape[0])]
            Ws = [sum(B[j, t] * Zs[j] for j in range(B.shape[0])) % p for t in range(k)]
            Psi_DW = sum(np.kron(labels[t], (Ds[t] - Ws[t]) % p) for t in range(k)) % p
            assert np.array_equal(Psi_D, Psi_DW), "W-terms do not cancel"
            assert sum(rank((D - W) % p, p) for D, W in zip(Ds, Ws)) >= L * rP, "certificate bound fails"
    reached, total = 0, 0
    for _ in range(40):
        nR, nC = rng.choice([(2, 3), (3, 2), (2, 4), (3, 3), (3, 4)])
        allc = [(r, c) for r in range(nR) for c in range(nC)]
        d = rng.randint(1, min(4, len(allc) - 4))
        cells = rng.sample(allc, 4 + d)
        Bx = np.array([[rng.randrange(1, p) for _ in range(4 + d)] for _ in range(d)])
        got = quadric_labels(cells, Bx, rng, p, tries=60)
        if got is None:
            continue
        total += 1
        reached += got[0] >= len(cells) - rank(Bx, p)
    print(f"[quadric labels] 7 cells of a 3x3 D-grid, generic rank-3 B: rank-1 labels from a quadric through ker B "
          f"cancel every word and give rank P = {rP} >= k - rank B = {k - rB} (shared labels reach 3); Psi identity "
          f"and bound at L = 2, 4 against random words: ok; random instances with dim ker B = 4: {reached}/{total} "
          "reach k - rank B")


def sym_product(g, h, p):
    """Symmetric matrix of the quadratic form (g.v)(h.v) on V = X + Y."""
    return (np.outer(g, h) + np.outer(h, g)) * pow(2, p - 2, p) % p


def segment_bound(R, nX, p):
    """(rank of the bilinear part, j = min of the XX- and YY-junk ranks) of a segment's net change R."""
    return rank(R[:nX, nX:], p), min(rank(R[:nX, :nX], p), rank(R[nX:, nX:], p))


def part10(rng):
    """Mixed products charged by junk. A segment (a maximal path between special nodes of a merge-free tree) with net
    change R, a symmetric form on V = X + Y with bilinear block Q and junk blocks P (XX), S (YY), has
    rank R >= 2 rank Q - min(rank P, rank S); every product has symmetric rank <= 2, so
    #products >= rank(bilinear part) - j/2 with j = min(rank P, rank S). Random segments with mixed, pure and
    junk-only products from starts with no junk, one-sided or two-sided junk; then Winograd from a head carrying
    -Xi_i - H_j, which meets the bound with equality."""
    count, tight, one_sided = 0, 0, 0
    for p in (3, 10007):
        for _ in range(300):
            nX, nY = rng.randint(1, 4), rng.randint(1, 4)
            n = nX + nY
            cancel = rng.choice([(), ("Y",), ("X",), ("X", "Y")])
            steps, mixed = [], 0
            for _ in range(rng.randint(1, 6)):
                ops = []
                for _ in range(2):
                    v = np.array([rng.randrange(p) for _ in range(n)])
                    t = rng.choice(["mixed", "mixed", "X", "Y"])
                    if t == "X":
                        v[nX:] = 0
                    elif t == "Y":
                        v[:nX] = 0
                    ops.append(v)
                steps.append(sym_product(ops[0], ops[1], p))
                mixed += all(v[:nX].any() and v[nX:].any() for v in ops)
                for side in cancel:
                    g, h = ops[0].copy(), ops[1].copy()
                    for v in (g, h):
                        v[nX:] = 0 if side == "X" else v[nX:]
                        v[:nX] = 0 if side == "Y" else v[:nX]
                    if g.any() and h.any():
                        steps.append((-sym_product(g, h, p)) % p)
            R = sum(steps) % p
            rQ, j = segment_bound(R, nX, p)
            P, S = R[:nX, :nX], R[nX:, nX:]
            assert rank(R, p) >= 2 * rQ - min(rank(P, p), rank(S, p)), "block inequality fails"
            assert all(rank(s, p) <= 2 for s in steps), "a product has symmetric rank > 2"
            assert 2 * len(steps) >= 2 * rQ - j, "segment bound fails"
            assert len(steps) + mixed >= rQ, "bilinear bound fails"
            if cancel:
                assert j == 0, "cancelled side still carries junk"
                assert len(steps) >= rQ, "one-sided junk but mixed products gained"
                one_sided += 1
            tight += 2 * len(steps) == 2 * rQ - j
            count += 1
        for _ in range(100):
            nX, nY = rng.randint(1, 4), rng.randint(1, 4)
            n = nX + nY
            steps = []
            for _ in range(rng.randint(1, 3)):
                a, a2, c, c2 = ([rng.randrange(p) for _ in range(nX)] + [0] * nY for _ in range(4))
                b, d = ([0] * nX + [rng.randrange(p) for _ in range(nY)] for _ in range(2))
                a, a2, c, c2, b, d = map(np.array, (a, a2, c, c2, b, d))
                steps += [sym_product(a + b, c + d, p), sym_product(a2 + b, (c2 - d) % p, p)]
            R = sum(steps) % p
            rQ, j = segment_bound(R, nX, p)
            assert not R[nX:, nX:].any() and j == 0, "paired products should cancel their YY junk"
            assert len(steps) >= rQ, "YY junk cancelled between mixed products, but they gained"
            one_sided += 1
            count += 1
    for L in (2, 4, 6):
        p = 10007
        n = 2 * L
        eye = np.eye(n, dtype=np.int64)
        x = lambda k, eye=eye: eye[k]
        y = lambda k, eye=eye, L=L: eye[L + k]
        Xi = sum(sym_product(x(2 * k), x(2 * k + 1), p) for k in range(L // 2)) % p
        H = sum(sym_product(y(2 * k + 1), y(2 * k), p) for k in range(L // 2)) % p
        head = (-Xi - H) % p
        steps = [sym_product(x(2 * k) + y(2 * k + 1), x(2 * k + 1) + y(2 * k), p) for k in range(L // 2)]
        end = (head + sum(steps)) % p
        target = sum(sym_product(x(k), y(k), p) for k in range(L)) % p
        assert np.array_equal(end, target), "Winograd does not reach the dot product"
        rQ, j = segment_bound((end - head) % p, L, p)
        assert (rQ, j) == (L, L) and 2 * len(steps) == 2 * rQ - j, "Winograd should meet the bound"
    print(f"[junk] {count} random segments over GF(3) and GF(10007) (mixed and pure products, some followed by the "
          "pure products that cancel their XX or YY junk): rank R >= 2 rank Q - min junk rank, #products >= "
          f"rank B - j/2: ok ({tight} tight); {one_sided} with one-sided net junk gain nothing; Winograd from a head "
          "carrying -Xi - H at L = 2, 4, 6: #products = L - L/2 (tight)")


def recursive_strassen(L, p=10007):
    """Strassen applied recursively to an L x L group of positions inside one block (L = 2^k): 7^k tensor-core
    products of pure operands (CUDA-prepared combinations of x's and of y's), merged into one word per position.
    Returns (number of products, number of words); asserts every word equals its target D_(i,j) exactly."""
    nX = nY = L * L
    products = []

    def mul(A, B):
        s = len(A)
        if s == 1:
            products.append((A[0][0] % p, B[0][0] % p))
            out = np.zeros(4 ** 8, dtype=np.int64)
            out[len(products) - 1] = 1
            return [[out]]
        h = s // 2
        blk = lambda M, r, c: [row[c * h:(c + 1) * h] for row in M[r * h:(r + 1) * h]]
        add = lambda P, Q, sgn=1: [[P[i][j] + sgn * Q[i][j] for j in range(h)] for i in range(h)]
        A11, A12, A21, A22 = blk(A, 0, 0), blk(A, 0, 1), blk(A, 1, 0), blk(A, 1, 1)
        B11, B12, B21, B22 = blk(B, 0, 0), blk(B, 0, 1), blk(B, 1, 0), blk(B, 1, 1)
        M1 = mul(add(A11, A22), add(B11, B22))
        M2 = mul(add(A21, A22), B11)
        M3 = mul(A11, add(B12, B22, -1))
        M4 = mul(A22, add(B21, B11, -1))
        M5 = mul(add(A11, A12), B22)
        M6 = mul(add(A21, A11, -1), add(B11, B12))
        M7 = mul(add(A12, A22, -1), add(B21, B22))
        C11 = add(add(M1, M4), add(M7, M5, -1))
        C12 = add(M3, M5)
        C21 = add(M2, M4)
        C22 = add(add(M1, M2, -1), add(M3, M6))
        return [C11[i] + C12[i] for i in range(h)] + [C21[i] + C22[i] for i in range(h)]

    eX, eY = np.eye(nX, dtype=np.int64), np.eye(nY, dtype=np.int64)
    A = [[eX[i * L + k] for k in range(L)] for i in range(L)]
    B = [[eY[j * L + k] for j in range(L)] for k in range(L)]
    C = mul(A, B)
    for i in range(L):
        for j in range(L):
            word = np.zeros((nX, nY), dtype=np.int64)
            for m, c in enumerate(C[i][j][:len(products)]):
                if c % p:
                    word = (word + c * np.outer(*products[m])) % p
            D = np.zeros((nX, nY), dtype=np.int64)
            for k in range(L):
                D[i * L + k, j * L + k] = 1
            assert np.array_equal(word, D), "recursive Strassen word differs from its target"
    return len(products), L * L


def part11(rng):
    """The per-word charge L of Theorem C is asymptotically tight: recursive Strassen on an L x L group of positions
    inside one block uses 7^k products of pure operands for L^2 targets of rank L (L = 2^k), with one merge-built word
    per target, each added to an empty running sum (H' = 0, W' = L^2): it saves (8^k - 7^k)/4^k = L - L^(log2 7 - 2)
    per distinct word, against the L that Theorem C allows."""
    rows = []
    for L in (2, 4, 8):
        n_products, words = recursive_strassen(L)
        k = L.bit_length() - 1
        targets, H_prime, W_prime = words, 0, words
        assert n_products == 7 ** k and n_products >= L * (targets - H_prime - W_prime), "Theorem C fails"
        saved = L * targets - n_products
        assert saved * 4 ** k == (8 ** k - 7 ** k) * words
        rows.append(f"L={L}: {n_products} products for {words} targets, "
                    f"{saved / words:.3f} = L - L^0.807 saved per word")
    print("[per-word charge] recursive Strassen in one block, every word checked exactly: " + "; ".join(rows)
          + "; so Theorem C's charge of L per distinct word is asymptotically tight")


def star_targets(L, k):
    """D-hat_t of a row star of k cells on X_r + Y_1 + ... + Y_k (blocks of size L): I_L at blocks (0, t) and (t, 0)."""
    n = L * (k + 1)
    Ds = []
    for t in range(k):
        D = np.zeros((n, n), dtype=np.int64)
        for i in range(L):
            D[i, L * (t + 1) + i] = D[L * (t + 1) + i, i] = 1
        Ds.append(D)
    return Ds


def rank_mod(M, Q, p):
    """Rank of M modulo the column space of Q."""
    return rank(np.concatenate([M, Q], axis=1) % p, p) - rank(Q, p)


def sym_star_bounds(L, k, Z, gammas, p):
    """The proof of symmetric Lemma C on a row star, step by step, for one word Z (any square matrix) and links
    F_t = gamma_t D-hat_t - Z. Returns (sum_t rank F_t, rank Z_YY)."""
    Ds = star_targets(L, k)
    F = [(g * D - Z) % p for g, D in zip(gammas, Ds)]
    Dyy = (-Z[L:, L:]) % p
    w = rank(Dyy, p)
    U = nullspace(Dyy, k * L, p).T
    betas = [pow(g, p - 2, p) for g in gammas]
    K = np.concatenate([b * np.eye(L, dtype=np.int64) for b in betas], axis=0) % p
    K2 = nullspace(np.concatenate([b * np.eye(L, dtype=np.int64) for b in betas], axis=1) % p, k * L, p).T
    dim_UK = (k * L - w) + L - rank(np.concatenate([U, K], axis=1), p)
    dim_KQ = (k - 1) * L + w - rank(np.concatenate([K2, Dyy], axis=1), p)
    total, sum_B, sum_C = 0, 0, 0
    for Ft in F:
        B_U = rank(Ft[:L, L:] @ U % p, p) if U.shape[1] else 0
        C_Q = rank_mod(Ft[L:, :L], Dyy, p)
        r = rank(Ft, p)
        assert r >= B_U + w + C_Q, "block-triangular step fails"
        total, sum_B, sum_C = total + r, sum_B + B_U, sum_C + C_Q
    assert sum_B >= k * L - w - dim_UK, "certificate on ker Z_YY fails"
    assert sum_C >= (k - 1) * L - dim_KQ, "certificate modulo col Z_YY fails"
    assert total >= 2 * L * (k - 1) + (k - 2) * w, "symmetric Lemma C fails on a row star"
    a = rank(Z[:L, L:] @ U % p, p) if U.shape[1] else 0
    b = rank_mod(Z[L:, :L], Dyy, p)
    rZ = rank(Z, p)
    assert rZ >= w + a + b, "rank Z >= rank Z_YY + a + b fails"
    assert sum_B >= k * L - w - a and sum_C >= k * L - w - b, "independent-label certificates fail"
    assert total >= 2 * k * L - rZ + (k - 1) * w, "per-word cap fails on a row star"
    return total, w, rZ


def star_word(L, k, gammas, rng, p, sym):
    """A word aimed at the star: partials of the cells (inner indices split among them), plus junk on X_r and on the
    Y blocks of a few cells (Winograd-style pairs or random low rank), plus low-rank noise."""
    n = L * (k + 1)
    Z = np.zeros((n, n), dtype=np.int64)
    owner = [rng.randrange(k) for _ in range(L)]
    for i, t in enumerate(owner):
        if rng.random() < 0.8:
            Z[i, L * (t + 1) + i] = Z[L * (t + 1) + i, i] = gammas[t]
    for _ in range(rng.randint(0, 3)):
        kind = rng.choice(["xx", "yy", "cross", "winograd", "noise"])
        if kind == "winograd":
            t = rng.randrange(k)
            for i in range(0, L - 1, 2):
                for u, v in ((i, i + 1), (L * (t + 1) + i + 1, L * (t + 1) + i)):
                    Z[u, v] = (Z[u, v] - gammas[t]) % p
                    Z[v, u] = (Z[v, u] - gammas[t]) % p
            continue
        a = np.array([rng.randrange(p) for _ in range(n)], dtype=np.int64)
        b = np.array([rng.randrange(p) for _ in range(n)], dtype=np.int64)
        if kind == "xx":
            a[L:], b[L:] = 0, 0
        elif kind == "yy":
            t = rng.randrange(k)
            for v in (a, b):
                v[: L * (t + 1)], v[L * (t + 2) :] = 0, 0
        elif kind == "cross":
            t = rng.randrange(k)
            a[L:], b[:L] = 0, 0
            b[: L * (t + 1)], b[L * (t + 2) :] = 0, 0
        Z = (Z + np.outer(a, b) + (np.outer(b, a) if sym else 0)) % p
    return Z


def part12(rng):
    """Symmetric Lemma C: links adding one word Z save at most L in total with mixed products counted once, whatever
    junk Z carries. Path changes are quadratic forms R_t = M(D_t) - beta_t M(Z), every product has symmetric rank
    <= 2, so it suffices that sum_t rank R_t >= 2L(k - 1). Disjoint pairs give 4L; on a row star, for any Z,
    sum_t rank(gamma_t D-hat_t - Z) >= 2L(k - 1) + (k - 2) rank Z_YY, by a block-triangular split on ker Z_YY and
    one label certificate on each side. With independent labels instead, the same split gives the per-word cap
    sum_t rank >= 2kL - rank Z + (k - 1) rank Z_YY, since rank Z >= rank Z_YY + a + b (a, b: the word's two
    off-diagonal blocks on ker Z_YY and modulo col Z_YY). Each step is checked on star words with junk, random
    words and the local search's minimisers; then random cell patterns in a P x Q block, where the links the word
    helps form a star."""
    count, tight, tight_w, tight_cap = 0, 0, 0, 0
    for p in (3, 5, 10007):
        for _ in range(250):
            L, k = rng.randint(1, 4), rng.randint(1, 4)
            gammas = [rng.randrange(1, p) for _ in range(k)]
            sym = rng.random() < 0.7
            if rng.random() < 0.8:
                Z = star_word(L, k, gammas, rng, p, sym)
            else:
                n = L * (k + 1)
                Z = np.array([[rng.randrange(p) for _ in range(n)] for _ in range(n)], dtype=np.int64)
                Z = (np.triu(Z) + np.triu(Z, 1).T) % p if sym else Z
            total, w, rZ = sym_star_bounds(L, k, Z, gammas, p)
            tight += total == 2 * L * (k - 1)
            tight_w += w > 0 and total == 2 * L * (k - 1) + (k - 2) * w
            tight_cap += rZ < 2 * L and total == 2 * k * L - rZ
            count += 1
    for L, k in ((2, 3), (2, 4), (3, 3)):
        p = 3
        Ds = star_targets(L, k)
        for _ in range(4):
            gammas = [rng.randrange(1, p) for _ in range(k)]
            Z = star_word(L, k, gammas, rng, p, True)
            best = sum(rank((g * D - Z) % p, p) for g, D in zip(gammas, Ds))
            for _ in range(400):
                Z2 = Z.copy()
                i, j = rng.randrange(len(Z)), rng.randrange(len(Z))
                Z2[i, j] = Z2[j, i] = rng.randrange(p)
                s = sum(rank((g * D - Z2) % p, p) for g, D in zip(gammas, Ds))
                if s <= best:
                    Z, best = Z2, s
            total, _, _ = sym_star_bounds(L, k, Z, gammas, p)
            tight += total == 2 * L * (k - 1)
            count += 1
    patterns = 0
    for p in (3, 10007):
        for _ in range(120):
            P, Q, L = rng.randint(2, 3), rng.randint(2, 3), rng.randint(1, 3)
            nX, n = P * L, (P + Q) * L
            cells = rng.sample([(r, c) for r in range(P) for c in range(Q)], rng.randint(2, P * Q))
            Dh = []
            for r, c in cells:
                D = np.zeros((n, n), dtype=np.int64)
                for i in range(L):
                    D[r * L + i, nX + c * L + i] = D[nX + c * L + i, r * L + i] = 1
                Dh.append(D)
            betas = [rng.randrange(1, p) for _ in cells]
            Z = np.zeros((n, n), dtype=np.int64)
            for (r, c), b in zip(cells, betas):
                if rng.random() < 0.4:
                    Z = (Z + Dh[cells.index((r, c))] * pow(b, p - 2, p) * rng.randrange(p)) % p
            for _ in range(rng.randint(0, 3)):
                a = np.array([rng.randrange(p) for _ in range(n)], dtype=np.int64)
                Z = (Z + np.outer(a, a)) % p
            ranks = [rank((D - b * Z) % p, p) for D, b in zip(Dh, betas)]
            assert sum(ranks) >= 2 * L * (len(cells) - 1), "symmetric Lemma C fails on a cell pattern"
            assert sum(ranks) >= 2 * L * len(cells) - rank(Z, p), "per-word cap fails on a cell pattern"
            helped = [c for c, r in zip(cells, ranks) if r < 2 * L]
            assert len({r for r, _ in helped}) <= 1 or len({c for _, c in helped}) <= 1, "helped links not a star"
            patterns += 1
    print(f"[sym Lemma C] {count} row stars (L <= 4, k <= 4; star words with XX, YY and Winograd-style junk, random "
          "words, local-search minimisers; GF(3), GF(5), GF(10007)): block-triangular split, both certificates, "
          f"sum rank >= 2L(k-1) + (k-2) rank Z_YY: ok ({tight} at 2L(k-1), {tight_w} tight with Z_YY != 0); "
          f"rank Z >= rank Z_YY + a + b and sum rank >= 2kL - rank Z + (k-1) rank Z_YY: ok ({tight_cap} tight with "
          f"rank Z < 2L); {patterns} cell patterns: sum rank >= max(2L(k-1), 2Lk - rank Z), helped links form a "
          "star: ok")


def cross_labels(B, S, T, lam, p):
    """Rank-1 labels u_t h0^T on S (the row side) and g0 v_t^T on T (the column side), coupled through lam in F^2:
    the columns of U (and V) are a basis of {x : B_S x in span lam}, with B_S U = lam g0^T and B_T V = -lam h0^T.
    None if g0 or h0 vanishes (lam outside im B_S or im B_T)."""
    def side(idx):
        Bs = B[:, idx]
        X = nullspace(((lam[1] * Bs[0] - lam[0] * Bs[1]) % p).reshape(1, -1), len(idx), p).T % p
        i = 0 if lam[0] % p else 1
        return X, Bs @ X % p * pow(int(lam[i]), p - 2, p) % p
    U, BU = side(S)
    V, BV = side(T)
    i = 0 if lam[0] % p else 1
    g0, h0 = BU[i], -BV[i] % p
    if not g0.any() or not h0.any():
        return None
    labels = {}
    for n, t in enumerate(S):
        labels[t] = np.outer(U[n], h0) % p
    for n, t in enumerate(T):
        labels[t] = np.outer(g0, V[n]) % p
    return labels, U.shape[1], V.shape[1]


def part13(rng):
    """d*L for two words on a cross (a D-row and a D-column, with or without the cell where they meet): the coupled
    labels cancel both words, rank P = k + 2 - rank B_S - rank B_T - eps (eps = 1 iff the center is present with
    beta_c on span lam, whichever side holds it), and some choice of side and lam reaches k - rank B unless each
    side adds one word (Lemma C per side). Then the Psi identity and the bound at L = 2 against random words."""
    counts = {"label": 0, "one word per side": 0, "rank B = 1": 0}
    psi = 0
    for p in (3, 5, 101):
        for _ in range(150):
            a, b, center = rng.randint(1, 4), rng.randint(1, 4), rng.random() < 0.5
            cells = [(0, c) for c in range(1, a + 1)] + [(r, 0) for r in range(1, b + 1)] + [(0, 0)] * center
            k = len(cells)
            pool = [v for v in (np.array([rng.randrange(p), rng.randrange(p)]) for _ in range(rng.randint(1, 4)))
                    if v.any()] or [np.array([1, 0])]
            B = np.array([pool[rng.randrange(len(pool))] if rng.random() < 0.6 else
                          np.array([rng.randrange(1, p), rng.randrange(p)]) for _ in range(k)]).T % p
            rB = rank(B, p)
            if rB < 2:
                counts["rank B = 1"] += 1
                continue
            rows, cols, c = list(range(a)), list(range(a, a + b)), [k - 1] * center
            best, single = -1, False
            for S, T in ((rows + c, cols), (rows, cols + c)):
                single |= rank(B[:, S], p) == 1 and rank(B[:, T], p) == 1
                for lam in [B[:, t] for t in range(k)] + [np.array([1, 0]), np.array([0, 1])]:
                    got = cross_labels(B, S, T, lam, p)
                    if got is None:
                        continue
                    labels, dU, dV = got
                    assert dU == len(S) - rank(B[:, S], p) + 1 and dV == len(T) - rank(B[:, T], p) + 1
                    for j in range(2):
                        assert not (sum(B[j, t] * labels[t] for t in range(k)) % p).any(), "labels do not cancel"
                    Pm = np.zeros((dU * (b + 1), dV * (a + 1)), dtype=np.int64)
                    for t, (r, col) in enumerate(cells):
                        Pm[r * dU:(r + 1) * dU, col * dV:(col + 1) * dV] = labels[t]
                    eps = int(bool(c) and rank(np.stack([B[:, c[0]], lam]), p) == 1)
                    rP = rank(Pm, p)
                    assert rP == k + 2 - rank(B[:, S], p) - rank(B[:, T], p) - eps, "rank P formula fails"
                    best = max(best, rP)
                    if psi < 40 and p == 101 and rP >= k - rB:
                        L = 2
                        shape = Shape(b + 1, a + 1, L, L)
                        lc = [((r, col), 0) for r, col in cells]
                        Ds = [shape.D(*x) for x in lc]
                        Zs = [random_word(shape, lc, rng, p) for _ in range(2)]
                        Ws = [(B[0, t] * Zs[0] + B[1, t] * Zs[1]) % p for t in range(k)]
                        Psi_D = sum(np.kron(labels[t], Ds[t]) for t in range(k)) % p
                        Psi_DW = sum(np.kron(labels[t], (Ds[t] - Ws[t]) % p) for t in range(k)) % p
                        assert np.array_equal(Psi_D, Psi_DW) and rank(Psi_D, p) == L * rP, "Psi identity fails"
                        assert sum(rank((D - W) % p, p) for D, W in zip(Ds, Ws)) >= L * (k - rB), "d*L fails"
                        psi += 1
            if best >= k - rB:
                counts["label"] += 1
            else:
                assert single, (cells, B.tolist(), best, k - rB)
                counts["one word per side"] += 1
    print(f"[crosses, d = 2] GF(3), GF(5), GF(101): coupled labels cancel both words and rank P = k + 2 - rank B_S - "
          f"rank B_T - eps: ok; k - rank B reached ({counts['label']}) or one word per side "
          f"({counts['one word per side']}), rank B = 1 ({counts['rank B = 1']}); Psi identity and d*L bound at L = 2 "
          f"against random words ({psi}): ok")


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
    parts = sys.argv[1:] or ["1", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
    if "1" in parts:
        part1_2(rng)
    if "3" in parts:
        part3(rng)
    if "4" in parts:
        part4(rng)
    if "5" in parts:
        part5(rng)
    if "6" in parts:
        part6(rng)
    if "7" in parts:
        part7(rng)
    if "8" in parts:
        part8(rng)
    if "9" in parts:
        part9(rng)
    if "10" in parts:
        part10(rng)
    if "11" in parts:
        part11(rng)
    if "12" in parts:
        part12(rng)
    if "13" in parts:
        part13(rng)

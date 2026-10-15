"""Checks for Conjecture C-mix and pouw-core's `MixedWords` premise (worker A7, CMP-8).

Note: `note:20261015T0822Z-draft-conjecture-c-mix`. Builds on A3's `note:20261015T0436Z-draft-merges-and-reuse`
(Theorems C and M, Lemma C-sym; `merges_check.py` on branch `cursor/merges-reuse-ea24`).

Quadratic forms on V = X + Y are symmetric matrices mod p. A product g.h adds sym(g h^T), of symmetric rank <= 2, and
a committed target T = sum x_rk y_kc is M(T) = [[0, T], [T^T, 0]], of rank 2 rank T. The symmetric Steiner number
St_sym is the least total rank of a tree's edges; #products >= St_sym / 2, mixed products counted once.

Parts:
  1  `MixedWords` as stated (T_mix <= tau_x * distinct mixed words) is not a theorem: a cancelling gadget on a
     correct computation raises T_mix past any tau_x * X, while the combined floor k*own + tau_x*X + L*HW holds.
  2  Exact St_sym (Dreyfus-Wagner over every tree, `symsteiner.c`) on small instances over GF(3) and GF(5):
     rooted at 0, St_sym = 2L * sum_pi hi_pi; with a free root (a head tree), 2L (sum hi - min lo).
  3  The depth-3 inequality behind Proposition 3 (two Steiner points, three targets with one pair in disjoint rows
     and columns), on random Steiner points.
  4  Local search for symmetric gains in deeper trees at L = 2.
  5  The span bound: every computation costs at least the tensor rank of T = sum_t S_t (x) e_t, and R(T) >= 2Lk when
     no two cells share a D-column (or a D-row), for trees of any depth and for DAGs with merges.
  6  The L-shape at L = 2, exactly over GF(3): the proved inequalities leave five cost-11 rank profiles for the open
     topology, each pins its Steiner points to rank summands, and enumerating them finds no tree below 12.
  7  The L-shape at every L: the gate lemma (a nearby rank summand), the exact case (both Steiner points on their
     gates gives rank(V - U) >= 2L) step by step, and the surgery that moves a tree onto its gates without raising
     its cost, on random trees.
  8  Symmetric rigidity fails at depth 3: a merge-free tree with mixed words computes 2 x 2 matrix multiplication
     in 7 products, exactly over the integers, and its P x Q family saves (LQ/4)(P - 1) products with PLQ distinct
     mixed words. With --big, the exhaustive normal-form search at L = 2 over GF(3) that found it (least cost 14).

Run: python cmix_check.py [part ...] [--big]   (--big adds the 5-dimensional GF(3) instances, about 14 M states)
"""

import itertools
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def rank(M, p):
    """Rank of an integer matrix mod a prime p."""
    A = np.array(M, dtype=np.int64) % p
    r, (rows, cols) = 0, A.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        A[r] = A[r] * pow(int(A[r, c]), p - 2, p) % p
        nz = A[:, c].copy()
        nz[r] = 0
        A = (A - np.outer(nz, A[r])) % p
        r += 1
        if r == rows:
            break
    return r


def sym(g, h, p):
    """The symmetric matrix of the quadratic form (g.v)(h.v)."""
    return (np.outer(g, h) + np.outer(h, g)) * pow(2, p - 2, p) % p


class Grid:
    """One block of P x Q positions, `lev` levels of L inner indices. X has coordinates (r, k), Y has (k, c)."""

    def __init__(self, P, Q, L, lev=1):
        self.P, self.Q, self.L, self.lev = P, Q, L, lev
        self.K = L * lev
        self.nX, self.nY = P * self.K, Q * self.K
        self.n = self.nX + self.nY

    def x(self, r, k):
        return r * self.K + k

    def y(self, k, c):
        return self.nX + k * self.Q + c

    def target(self, r, c, s, scale=1):
        """M(T_{(r,c),s}) with T = sum_{k < sL} x_rk y_kc, entries `scale` in both off-diagonal blocks."""
        M = np.zeros((self.n, self.n), dtype=np.int64)
        for k in range(s * self.L):
            M[self.x(r, k), self.y(k, c)] = M[self.y(k, c), self.x(r, k)] = scale
        return M


# ---------------------------------------------------------------------------------------------------------------
# Part 1: the bare premise is false; the combined floor is what the bound uses.


def gadget(G, a, b, rng, p):
    """Mixed words w_1..w_a, h_1..h_b and the products w_i h_j, plus (-sum w)(sum h): their forms sum to zero."""
    def mixed():
        while True:
            v = np.array([rng.randrange(p) for _ in range(G.n)], dtype=np.int64)
            if v[: G.nX].any() and v[G.nX :].any():
                return v
    W = [mixed() for _ in range(a)]
    H = [mixed() for _ in range(b)]
    prods = [(w, h) for w in W for h in H]
    prods.append(((-sum(W)) % p, sum(H) % p))
    words = {tuple(v) for pair in prods for v in pair}
    return prods, words


def part1(rng):
    """A correct computation (every target by L pure products) with the gadget on one target's link. Its forms
    cancel exactly, so every committed value stays correct; T_mix = ab + 1 and X = a + b + 2 distinct mixed words,
    so T_mix > tau_x * X once ab is large. Theorem C's floor (products + T_mix >= L |G_T|) and the combined floor with
    tau_x = 0 (products >= L |G_T|) both hold."""
    p = 10007
    for (P, Q, L), (a, b), tau in (((2, 2, 4), (20, 20), 8), ((2, 3, 2), (80, 80), 32), ((1, 2, 8), (80, 80), 32)):
        G = Grid(P, Q, L)
        cells = [(r, c) for r in range(P) for c in range(Q)]
        prods, words = gadget(G, a, b, rng, p)
        for drop_last in (False, True):
            ok = True
            pure = 0
            for (r, c) in cells:
                val = np.zeros((G.n, G.n), dtype=np.int64)
                for k in range(L):
                    g = np.zeros(G.n, dtype=np.int64)
                    h = np.zeros(G.n, dtype=np.int64)
                    g[G.x(r, k)], h[G.y(k, c)] = 1, 1
                    val = (val + sym(g, h, p)) % p
                    pure += 1
                if (r, c) == cells[0]:
                    for w, h in (prods[:-1] if drop_last else prods):
                        val = (val + sym(w, h, p)) % p
                want = G.target(r, c, 1) * pow(2, p - 2, p) % p
                ok &= bool((val == want).all())
            if drop_last:
                assert not ok, "mutation not caught: a gadget without its cancelling product left the targets correct"
                continue
            assert ok, "the gadget changed a committed value"
            k_targets, T_mix, X = len(cells), len(prods), len(words)
            products = pure + T_mix
            assert X == a + b + 2
            assert products + T_mix >= L * k_targets, "Theorem C's floor"
            assert products >= L * k_targets, "the combined floor at tau_x = 0"
            assert T_mix > tau * X, "the gadget should break T_mix <= tau_x X"
            print(f"  part 1: {P}x{Q}, L={L}: T_mix = {T_mix} > {tau} * X = {tau * X}; products {products} >= L|G_T| "
                  f"= {L * k_targets}; the gadget's forms sum to 0 (mutation caught)")


# ---------------------------------------------------------------------------------------------------------------
# Part 2: exact symmetric Steiner numbers.


def encode(M, q):
    n, r, pw = len(M), 0, 1
    for i in range(n):
        for j in range(i, n):
            r += int(M[i][j] % q) * pw
            pw *= q
    return r


def solver():
    exe = Path(tempfile.gettempdir()) / f"symsteiner_{os.getpid()}"
    if not exe.exists():
        subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-o", str(exe), str(HERE / "symsteiner.c")],
                       check=True)
    return exe


def st_sym(mats, q, rooted):
    n = len(mats[0])
    inp = f"{q} {n} {len(mats)} {1 if rooted else 0} " + " ".join(str(encode(M, q)) for M in mats)
    out = subprocess.run([str(solver())], input=inp, capture_output=True, text=True, check=True).stdout
    return int(out.splitlines()[0].split()[1])


def instances(big):
    """(name, grid, targets as ((r, c), s), q). Every cell or prefix pattern of each grid, up to the grid's row and
    column permutations."""
    out = []
    small = [((2, 2, 1, 1), (3, 5)), ((1, 3, 1, 1), (3, 5)), ((3, 1, 1, 1), (3,)), ((1, 1, 1, 2), (3, 5)),
             ((1, 1, 2, 1), (3, 5))]
    large = [((2, 3, 1, 1), (3,)), ((1, 4, 1, 1), (3,)), ((1, 2, 1, 2), ()), ((2, 1, 1, 2), ())]
    for (P, Q, L, lev), qs in small + (large if big else []):
        G = Grid(P, Q, L, lev)
        if G.n > (5 if 3 in qs else 4):
            continue
        cells = [(r, c) for r in range(P) for c in range(Q)]
        seen = set()
        for size in range(1, len(cells) + 1):
            for sub in itertools.combinations(cells, size):
                for levels in itertools.product(range(1, lev + 1), repeat=size):
                    tg = tuple(sorted(zip(sub, levels)))
                    canon = min(tuple(sorted(((pr[r], pc[c]), s) for (r, c), s in tg))
                                for pr in itertools.permutations(range(P)) for pc in itertools.permutations(range(Q)))
                    if canon in seen:
                        continue
                    seen.add(canon)
                    for q in qs:
                        out.append((f"{P}x{Q} L={L} lev={lev}", G, list(tg), q))
        if lev == 2:
            for q in qs:
                out.append((f"{P}x{Q} L={L} prefix chain", G, [((0, 0), 1), ((0, 0), 2)], q))
    return out


def part2(rng, big=False):
    """Every tree, by Dreyfus-Wagner. Rooted at 0, St_sym({0} U T) = 2L * sum_pi hi_pi (one top per position, A1a's
    Corollary 2 in the symmetric metric); with a free root, St_sym(T) >= 2L (sum hi - min lo), Theorem B's per-tree
    bound for a head tree. Mutations: targets carrying Winograd junk (the solver must find the cheaper trees), and
    each bound claimed one unit stronger."""
    count = 0
    for name, G, tg, q in instances(big):
        mats = [G.target(r, c, s) for (r, c), s in tg]
        hi = {}
        lo = {}
        for (pos, s) in tg:
            hi[pos] = max(hi.get(pos, 0), s)
            lo[pos] = min(lo.get(pos, 99), s)
        want_root = 2 * G.L * sum(hi.values())
        got = st_sym(mats, q, True)
        assert got == want_root, f"{name} {tg} q={q}: rooted St_sym {got} != {want_root}"
        assert not got >= want_root + 1, "mutation not caught: rooted bound + 1"
        if len(mats) >= 2:
            want_free = 2 * G.L * (sum(hi.values()) - min(lo.values()))
            got_f = st_sym(mats, q, False)
            assert got_f >= want_free, f"{name} {tg} q={q}: free-root St_sym {got_f} < {want_free}"
        count += 1
    print(f"  part 2: {count} instances, St_sym = 2L * sum hi exactly when rooted at 0, and >= 2L (sum hi - min lo) "
          "with a free root")
    # Mutation: Winograd junk on the targets. A single cell at L = 2 with junk x1 x2 + y1 y2 is one product: rank 2.
    G = Grid(1, 1, 2)
    for q in (3, 5):
        T = G.target(0, 0, 1)
        T[G.x(0, 0), G.x(0, 1)] = T[G.x(0, 1), G.x(0, 0)] = 1
        T[G.y(0, 0), G.y(1, 0)] = T[G.y(1, 0), G.y(0, 0)] = 1
        got = st_sym([T], q, True)
        assert got == 2 < 2 * G.L, "mutation not caught: Winograd junk should halve the rank"
    G = Grid(2, 2, 1)
    for q in (3, 5):
        mats = [G.target(r, c, 1) for r in range(2) for c in range(2)]
        J = np.zeros((G.n, G.n), dtype=np.int64)
        J[G.x(0, 0), G.x(1, 0)] = J[G.x(1, 0), G.x(0, 0)] = 1
        J[G.y(0, 0), G.y(0, 1)] = J[G.y(0, 1), G.y(0, 0)] = 1
        J = (J + G.target(0, 0, 1) + G.target(1, 1, 1)) % q
        shared = [(M + J) % q for M in mats]
        got = st_sym(shared, q, True)
        naive = sum(rank(M, q) for M in shared)
        assert got < naive, "mutation not caught: a shared Steiner point carrying the common junk should help"
    print("  part 2: mutations caught (Winograd junk on a target, shared junk on four targets, bounds + 1)")


# ---------------------------------------------------------------------------------------------------------------
# Part 3: two Steiner points.


def random_point(G, below, rng, p):
    """A Steiner point aimed at the targets below it: partials of their cells, Winograd and random junk, noise."""
    M = np.zeros((G.n, G.n), dtype=np.int64)
    for (r, c) in below:
        for k in range(G.L):
            if rng.random() < 0.6:
                M[G.x(r, k), G.y(k, c)] = M[G.y(k, c), G.x(r, k)] = (M[G.x(r, k), G.y(k, c)] + 1) % p
    for _ in range(rng.randint(0, 4)):
        kind = rng.choice(["winograd", "xx", "yy", "mixed", "noise"])
        r, c = rng.choice(below)
        g = np.zeros(G.n, dtype=np.int64)
        h = np.zeros(G.n, dtype=np.int64)
        if kind == "winograd":
            for k in range(0, G.L - 1, 2):
                g[G.x(r, k)], g[G.y(k + 1, c)] = 1, 1
                h[G.x(r, k + 1)], h[G.y(k, c)] = 1, 1
        else:
            for v in (g, h):
                for i in range(G.n):
                    if rng.random() < 0.3:
                        v[i] = rng.randrange(p)
                if kind == "xx":
                    v[G.nX:] = 0
                elif kind == "yy":
                    v[: G.nX] = 0
        M = (M + rng.randrange(1, p) * sym(g, h, p)) % p
    return M


def part3(rng):
    """Proposition 3 (two Steiner points): for the tree 0 - u - {t_a, v - {t_b, t_c}} with t_b, t_c in disjoint
    D-rows and D-columns of one block, rank u + rank(S_a - u) + rank(v - u) + rank(S_b - v) + rank(S_c - v) >= 6L.
    Checked step by step: A3's cap at u (rank u + sum_t rank(S_t - u) >= 6L), the triangle at t_a, the median bound
    at v, and the total. Mutations: the median bound with S_b, S_c in one D-row (it needs 4L), and the total + 1 on
    the tight instance."""
    count, tight = 0, 0
    for p in (3, 5, 10007):
        for _ in range(250):
            L = rng.choice([1, 2, 3])
            G = Grid(2, 2, L)
            a = rng.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
            b, c = rng.choice([((0, 0), (1, 1)), ((0, 1), (1, 0))])
            if a in (b, c):
                a = next(t for t in [(0, 0), (0, 1), (1, 0), (1, 1)] if t not in (b, c))
            Sa, Sb, Sc = (G.target(*t, 1) for t in (a, b, c))
            u = random_point(G, [a, b, c], rng, p)
            v = (u + random_point(G, [b, c], rng, p)) % p if rng.random() < 0.5 else random_point(G, [b, c], rng, p)
            ru, rau, rbu, rcu = rank(u, p), rank((Sa - u) % p, p), rank((Sb - u) % p, p), rank((Sc - u) % p, p)
            rvu, rbv, rcv = rank((v - u) % p, p), rank((Sb - v) % p, p), rank((Sc - v) % p, p)
            assert ru + rau + rbu + rcu >= 6 * L, "A3's cap at u"
            assert ru + rau >= 2 * L, "triangle"
            assert 2 * (rvu + rbv + rcv) >= rbu + rcu + 4 * L, "median bound at v"
            total = ru + rau + rvu + rbv + rcv
            assert total >= 6 * L, "Proposition 3 fails"
            count += 1
            tight += total == 6 * L
    G = Grid(2, 2, 2)
    Sa, Sb, Sc = G.target(0, 1, 1), G.target(0, 0, 1), G.target(1, 1, 1)
    total = rank(Sa, 3) + rank(Sb, 3) + rank(Sc, 3)
    assert total == 6 * G.L and not total >= 6 * G.L + 1, "mutation not caught: total + 1 on u = v = 0"
    # Mutation: the median bound's 4L needs disjoint rows and columns; in one D-row rank(S_b - S_c) is 2L.
    S1, S2 = G.target(0, 0, 1), G.target(0, 1, 1)
    assert rank((S1 - S2) % 3, 3) == 2 * G.L < 4 * G.L, "mutation not caught: same-row targets at distance 4L"
    print(f"  part 3: Proposition 3 on {count} random (u, v), {tight} tight; mutations caught")


# ---------------------------------------------------------------------------------------------------------------
# Part 4: local search for symmetric gains in deeper trees.


def topologies(k, rooted):
    """Full binary topologies on k leaves, as nested tuples of leaf indices (rooted: the top node hangs from 0)."""
    def build(items):
        if len(items) == 1:
            yield items[0]
            return
        first, rest = items[0], items[1:]
        for size in range(0, len(rest)):
            for comb in itertools.combinations(rest, size):
                left = (first,) + comb
                right = tuple(x for x in rest if x not in comb)
                for a in build(left):
                    for b in build(right):
                        yield (a, b)
    return list(build(tuple(range(k))))


def flatten_tree(top):
    """Nodes of a nested-tuple topology: a list of (node id, parent id, leaf index or None); node 0 is the top."""
    nodes = []

    def walk(t, parent):
        nid = len(nodes)
        nodes.append([nid, parent, t if isinstance(t, int) else None])
        if not isinstance(t, int):
            for ch in t:
                walk(ch, nid)
    walk(top, None)
    return nodes


def tree_cost(nodes, vals, rooted, p, cache):
    total = 0
    for nid, par, _ in nodes:
        if par is None:
            if rooted:
                total += cache(vals[nid], p)
            continue
        total += cache((vals[nid] - vals[par]) % p, p)
    return total


def search(G, cells, top, rooted, p, rng, steps, mats=None):
    """Simulated annealing over the Steiner points of one topology; returns (best cost, best values)."""
    mats = mats if mats is not None else [G.target(r, c, 1) for (r, c) in cells]
    nodes = flatten_tree(top)
    below = {}
    for nid, par, leaf in reversed(nodes):
        below.setdefault(nid, [])
        if leaf is not None:
            below[nid] = [leaf]
        if par is not None:
            below.setdefault(par, []).extend(below[nid])
    steiner = [nid for nid, _, leaf in nodes if leaf is None]
    vals = {}
    for nid, _, leaf in nodes:
        vals[nid] = mats[leaf] if leaf is not None else random_point(G, [cells[i] for i in below[nid]], rng, p)
    memo = {}

    def rk(M, p):
        key = M.tobytes()
        if key not in memo:
            if len(memo) > 200000:
                memo.clear()
            memo[key] = rank(M, p)
        return memo[key]

    cur = tree_cost(nodes, vals, rooted, p, rk)
    best, best_vals = cur, {k: v.copy() for k, v in vals.items()}
    for step in range(steps):
        T = 1.2 * (1 - step / steps) + 0.05
        s = rng.choice(steiner)
        old = vals[s]
        move = rng.random()
        nbrs = [nid for nid, par, _ in nodes if par == s] + [par for nid, par, _ in nodes if nid == s and par is not None]
        if move < 0.5:
            # Wedderburn step toward a neighbor (or toward 0 at a rooted top): with D = nb - s and z^T D z != 0,
            # s + (Dz)(Dz)^T / z^T D z lies one rank closer to nb.
            target = vals[rng.choice(nbrs)] if nbrs and not (rooted and s == 0 and rng.random() < 0.3) else 0 * old
            D = (target - old) % p
            new = old
            for _ in range(6):
                z = np.array([rng.randrange(p) for _ in range(G.n)], dtype=np.int64)
                Dz = D @ z % p
                alpha = int(z @ Dz % p)
                if alpha:
                    new = (old + np.outer(Dz, Dz) * pow(alpha, p - 2, p)) % p
                    break
        elif move < 0.7:
            g = np.zeros(G.n, dtype=np.int64)
            h = np.zeros(G.n, dtype=np.int64)
            for v in (g, h):
                for i in rng.sample(range(G.n), rng.randint(1, 3)):
                    v[i] = rng.randrange(1, p)
            new = (old + rng.randrange(1, p) * sym(g, h, p)) % p
        elif move < 0.8:
            g = np.zeros(G.n, dtype=np.int64)
            for i in rng.sample(range(G.n), rng.randint(1, 3)):
                g[i] = rng.randrange(1, p)
            new = (old + rng.randrange(1, p) * np.outer(g, g)) % p
        elif move < 0.9:
            new = vals[rng.choice(nbrs)].copy() if nbrs else old
            if rng.random() < 0.5:
                g = np.zeros(G.n, dtype=np.int64)
                g[rng.randrange(G.n)] = 1
                h = np.zeros(G.n, dtype=np.int64)
                h[rng.randrange(G.n)] = 1
                new = (new + sym(g, h, p)) % p
        else:
            new = random_point(G, [cells[i] for i in below[s]], rng, p)
        vals[s] = new
        c = tree_cost(nodes, vals, rooted, p, rk)
        if c <= cur or rng.random() < np.exp(-(c - cur) / T):
            cur = c
            if c < best:
                best, best_vals = c, {k: v.copy() for k, v in vals.items()}
        else:
            vals[s] = old
    return best, best_vals


SEARCH = [  # (name, grid (P, Q, L), cells)
    ("row of 3, L=2", (1, 3, 2), [(0, 0), (0, 1), (0, 2)]),
    ("L-shape, L=2", (2, 2, 2), [(0, 0), (0, 1), (1, 1)]),
    ("2x2, L=2", (2, 2, 2), [(0, 0), (0, 1), (1, 0), (1, 1)]),
    ("row of 4, L=2", (1, 4, 2), [(0, 0), (0, 1), (0, 2), (0, 3)]),
    ("column of 3, L=2", (3, 1, 2), [(0, 0), (1, 0), (2, 0)]),
]


def search_jobs(seeds):
    jobs = []
    for name, (P, Q, L), cells in SEARCH:
        for rooted in (True, False):
            for ti, top in enumerate(topologies(len(cells), rooted)):
                for p in (3, 5):
                    for seed in seeds:
                        jobs.append((name, P, Q, L, cells, rooted, ti, top, p, seed))
    return jobs


def run_job(job, steps=6000):
    name, P, Q, L, cells, rooted, ti, top, p, seed = job
    G = Grid(P, Q, L)
    rng = random.Random(hash((name, rooted, ti, p, seed)) & 0xFFFFFFFF)
    best, vals = search(G, cells, top, rooted, p, rng, steps)
    bound = 2 * L * len(cells) - (0 if rooted else 2 * L)
    return {"name": name, "rooted": rooted, "topology": str(top), "p": p, "seed": seed, "best": int(best),
            "bound": bound, "violation": bool(best < bound),
            "points": {str(k): v.tolist() for k, v in vals.items()} if best < bound else None}


def part4(rng, seeds=range(2), procs=1, out=None, steps=6000):
    """Simulated annealing over Steiner points for every full topology of the SEARCH instances, rooted at 0 (bound
    2Lk) and with a free root (bound 2L(k - 1)), over GF(3) and GF(5). Calibration (mutation): with a junk word J of
    rank 4 added to the two targets under the deepest Steiner point, the search must find the tree through v = J,
    cheaper than the star."""
    import json
    from multiprocessing import Pool
    G = Grid(2, 2, 2)
    cells = [(0, 1), (0, 0), (1, 0)]
    J = np.zeros((G.n, G.n), dtype=np.int64)
    for (i, j) in ((G.x(0, 0), G.x(1, 1)), (G.y(0, 0), G.y(1, 1)), (G.x(1, 0), G.y(0, 1)), (G.x(0, 1), G.y(1, 0))):
        J[i, j] = J[j, i] = 1
    mats = [G.target(r, c, 1) for (r, c) in cells]
    mats[1], mats[2] = (mats[1] + J) % 3, (mats[2] + J) % 3
    star = sum(rank(M, 3) for M in mats)
    found, _ = search(G, cells, (0, (1, 2)), True, 3, random.Random(1), 4000, mats=mats)
    assert found < star, f"mutation not caught: the search missed the shared junk point ({found} vs star {star})"
    print(f"  part 4: calibration: shared junk found ({found} < star {star})")
    # Planted trees: edges made of products of 2-sparse words on 0 - u - {t_a, v - {t_b, t_c}}; the search, started from
    # points unrelated to the plant, must get down to the planted length on most of them.
    hits, trials = 0, 8
    for i in range(trials):
        prng = random.Random(100 + i)
        def low(r):
            M = np.zeros((G.n, G.n), dtype=np.int64)
            for _ in range(r // 2):
                g = np.zeros(G.n, dtype=np.int64)
                h = np.zeros(G.n, dtype=np.int64)
                for w in (g, h):
                    for j in prng.sample(range(G.n), 2):
                        w[j] = prng.randrange(1, 3)
                M = (M + sym(g, h, 3)) % 3
            return M
        u = low(2)
        v = (u + low(2)) % 3
        leaves = [(u + low(4)) % 3, (v + low(4)) % 3, (v + low(4)) % 3]
        planted = rank(u, 3) + rank((v - u) % 3, 3) + sum(rank((t - b) % 3, 3) for t, b in zip(leaves, (u, v, v)))
        got, _ = search(G, cells, (0, (1, 2)), True, 3, random.Random(i), 6000, mats=leaves)
        hits += got <= planted
    assert hits >= trials // 2, f"mutation not caught: the search reaches planted trees on only {hits}/{trials}"
    print(f"  part 4: calibration: planted depth-3 trees recovered on {hits}/{trials}")
    jobs = search_jobs(seeds)
    results = []
    with Pool(procs) as pool:
        for res in pool.imap_unordered(run_job_default, [(j, steps) for j in jobs]):
            results.append(res)
            if out:
                with open(out, "a") as f:
                    f.write(json.dumps(res) + "\n")
    viol = [r for r in results if r["violation"]]
    tight = sum(r["best"] == r["bound"] for r in results)
    assert not viol, f"symmetric gain found: {viol[0]}"
    print(f"  part 4: {len(results)} searches, none below the bound, {tight} reach it")


def run_job_default(arg):
    job, steps = arg
    return run_job(job, steps)


# ---------------------------------------------------------------------------------------------------------------
# Part 5: the span bound. Any computation from 0 (a tree, or a DAG of products and free linear combinations) writes
# T = sum_t S_t (x) e_t as a sum of at most sum_e rank(Delta_e) (2 per product) rank-1 tensors, so its cost is at
# least the tensor rank R(T). Substitution: R(T) >= dim B + R(A) >= rank_Y(T') + R(T'), where B and A are T's
# mode-1 slices from Y- and X-functionals (pure X (x) C and pure Y (x) C) and T' = sum_t D_t (x) e_t.


def rref(M, p):
    """Reduced row echelon form mod p and its rank."""
    A = np.array(M, dtype=np.int64) % p
    r, (rows, cols) = 0, A.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        A[r] = A[r] * pow(int(A[r, c]), p - 2, p) % p
        nz = A[:, c].copy()
        nz[r] = 0
        A = (A - np.outer(nz, A[r])) % p
        r += 1
        if r == rows:
            break
    return A, r


def rowspace(M, p):
    A, r = rref(M, p)
    return list(A[:r])


def lemma_steps(n_kill, mats, cover, p, kill="X"):
    """The substitution lemma on one computation, step by step. `mats` are the target slices S_t (n x n), `cover`
    the rank-1 matrices (u, w) in V (x) C whose span must contain T's mode-1 slices. kill="X" projects V onto Y
    (the first n_kill coordinates are X). Returns (m, bound)."""
    n, k = mats[0].shape[0], len(mats)
    T = np.stack(mats, axis=2) % p
    slices = [T[i].reshape(-1) for i in range(n)]
    vec = [np.outer(u, w).reshape(-1) % p for u, w in cover]
    m = len(cover)
    assert rank(vec + slices, p) == rank(vec, p), "the cover's span misses a slice of T"
    killed = list(range(n_kill)) if kill == "X" else list(range(n_kill, n))
    kept = [i for i in range(n) if i not in killed]
    I0 = [i for i, (u, _) in enumerate(cover) if not u[kept].any()]
    rest = [i for i in range(m) if i not in I0]
    proj = []
    for i in rest:
        u = cover[i][0].copy()
        u[killed] = 0
        proj.append(np.outer(u, cover[i][1]).reshape(-1) % p)
    d = rank(proj, p) if proj else 0
    dim_rel = len(rest) - d
    B = [slices[i] for i in kept]          # contractions with kept-side functionals land in killed (x) C
    A = [slices[i] for i in killed]
    dim_B = rank(B, p)
    assert dim_B <= len(I0) + dim_rel, "substitution: B outside span(I0) + rho(Rel)"
    flat_A = max(rank(A, p), rank([T[np.ix_(killed, kept, [t])].reshape(-1) for t in range(k)], p),
                 rank([T[np.ix_(killed, [j], range(k))].reshape(-1) for j in kept], p))
    assert d >= flat_A, "substitution: the projected cover cannot hold A"
    bound = dim_B + flat_A
    assert m >= bound
    return m, bound


def tree_cover(G, cells, top, rng, p, rooted=True):
    """A random tree over the cells (Steiner points from `random_point`) as (edges' rank-1 cover, cost)."""
    nodes = flatten_tree(top)
    children = {nid: [] for nid, _, _ in nodes}
    for nid, par, _ in nodes:
        if par is not None:
            children[par].append(nid)

    def leaves(nid):
        leaf = nodes[nid][2]
        return [leaf] if leaf is not None else [x for ch in children[nid] for x in leaves(ch)]

    vals = {}
    for nid, par, leaf in nodes:
        vals[nid] = G.target(*cells[leaf], 1) if leaf is not None else random_point(G, [cells[x] for x in leaves(nid)], rng, p)
    cover, cost = [], 0
    for nid, par, leaf in nodes:
        if par is None and not rooted:
            continue
        delta = vals[nid] if par is None else (vals[nid] - vals[par]) % p
        chi = np.zeros(len(cells), dtype=np.int64)
        chi[leaves(nid)] = 1
        if not rooted:
            chi = (chi[:-1] - chi[-1]) % p
        for q in rowspace(delta, p):
            cover.append((q, chi))
        cost += rank(delta, p)
    return cover, cost


def strassen_cover(p):
    """Strassen's 7 products on the full 2 x 2 grid at L = 2: a DAG with 14 < 16 = 2Lk rank-1 terms."""
    G = Grid(2, 2, 2)
    a = lambda r, k: np.eye(G.n, dtype=np.int64)[G.x(r, k)]
    b = lambda k, c: np.eye(G.n, dtype=np.int64)[G.y(k, c)]
    M = [(a(0, 0) + a(1, 1), b(0, 0) + b(1, 1)), (a(1, 0) + a(1, 1), b(0, 0)), (a(0, 0), b(0, 1) - b(1, 1)),
         (a(1, 1), b(1, 0) - b(0, 0)), (a(0, 0) + a(0, 1), b(1, 1)), (a(1, 0) - a(0, 0), b(0, 0) + b(0, 1)),
         (a(0, 1) - a(1, 1), b(1, 0) + b(1, 1))]
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    use = {(0, 0): [1, 0, 0, 1, -1, 0, 1], (0, 1): [0, 0, 1, 0, 1, 0, 0], (1, 0): [0, 1, 0, 1, 0, 0, 0],
           (1, 1): [1, -1, 1, 0, 0, 1, 0]}
    mats = [G.target(r, c, 1) for (r, c) in cells]
    for t, cell in enumerate(cells):
        got = sum(2 * c * sym(g % p, h % p, p) for c, (g, h) in zip(use[cell], M)) % p
        assert (got == mats[t] % p).all(), "Strassen's products miss a target"
    cover = []
    for i, (g, h) in enumerate(M):
        w = np.array([use[cell][i] for cell in cells], dtype=np.int64) % p
        cover += [(g % p, w), (h % p, w)]
    return G, mats, cover


def rank_at_most(mats, p, extra):
    """Exact test: is R(T) <= dim W + extra (extra in {0, 1}), W = T's mode-1 slice space, over GF(p)?"""
    n, k = mats[0].shape[0], len(mats)
    T = np.stack(mats, axis=2) % p
    Wb = rowspace([T[i].reshape(-1) for i in range(n)], p)
    dW = len(Wb)
    full = list(Wb)
    for i in range(n * k):
        e = np.zeros(n * k, dtype=np.int64)
        e[i] = 1
        if rank(full + [e], p) > len(full):
            full.append(e)
    Bm = np.array(full, dtype=np.int64)
    aug, _ = rref(np.hstack([Bm.T, np.eye(n * k, dtype=np.int64)]), p)
    inv = aug[:, n * k:]                    # coordinates of v in the basis `full`: inv @ v
    projv = lambda d: [np.array(v) for v in itertools.product(range(p), repeat=d)
                       if any(v) and v[next(i for i in range(d) if v[i])] == 1]
    ones = [np.outer(u, w).reshape(-1) % p for u in projv(n) for w in projv(k)]
    R1 = np.array(ones, dtype=np.int64)
    Q = (R1 @ inv.T % p)[:, dW:]
    inside = [i for i in range(len(ones)) if not Q[i].any()]
    if extra == 0:
        return rank([ones[i] for i in inside], p) == dW if inside else dW == 0
    for z in projv(Q.shape[1]):
        lead = Q[:, next(i for i in range(len(z)) if z[i])]
        members = [i for i in range(len(ones)) if (Q[i] == lead[i] * z % p).all()]
        if len(members) > dW and rank([ones[i] for i in members], p) == dW + 1:
            return True
    return False


def part5(rng):
    """The span bound, checked on random trees and on Strassen. (a) Column-injective patterns (no two cells in one
    D-column) and row-injective ones: on random trees rooted at 0 the substitution's steps hold, and the bound is
    2Lk; with a free root (differences S_t - S_k), 2L(k - 1). (b) The hypothesis is needed: Strassen's DAG on the full
    2 x 2 grid at L = 2 covers T with 14 < 2Lk = 16 rank-1 terms, and the lemma's bound there is only 8. (c) Exact
    tensor rank at L = 1 over GF(3): a row pair has R = 4 = 2Lk; the L-shape has R = 6 = 2Lk although the bound gives 5.
    Mutation: the bound + 1 on the star."""
    count = 0
    pats = [((1, 3), [(0, 0), (0, 1), (0, 2)], "X"), ((2, 3), [(0, 0), (0, 1), (1, 2)], "X"),
            ((3, 1), [(0, 0), (1, 0), (2, 0)], "Y"), ((2, 2), [(0, 0), (1, 1)], "X"),
            ((1, 4), [(0, 0), (0, 1), (0, 2), (0, 3)], "X")]
    for p in (3, 5, 10007):
        for (P, Q), cells, kill in pats:
            for L in (1, 2):
                G = Grid(P, Q, L)
                k = len(cells)
                tops = topologies(k, rooted=True)
                for _ in range(6):
                    top = rng.choice(tops)
                    cover, cost = tree_cover(G, cells, top, rng, p)
                    mats = [G.target(r, c, 1) for (r, c) in cells]
                    m, bound = lemma_steps(G.nX, mats, cover, p, kill)
                    assert m == cost and bound == 2 * L * k, (m, bound)
                    cover, cost = tree_cover(G, cells, top, rng, p, rooted=False)
                    diff = [(mats[t] - mats[-1]) % p for t in range(k - 1)]
                    m, bound = lemma_steps(G.nX, diff, cover, p, kill)
                    assert m == cost and bound >= 2 * L * (k - 1), (m, bound)
                    count += 2
    G = Grid(1, 3, 2)
    mats = [G.target(0, c, 1) for c in range(3)]
    star = [(q, np.eye(3, dtype=np.int64)[t]) for t in range(3) for q in rowspace(mats[t], 3)]
    m, bound = lemma_steps(G.nX, mats, star, 3)
    assert m == bound == 12 and not m >= bound + 1, "mutation not caught: the bound + 1 on the star"
    G, mats, cover = strassen_cover(3)
    m, bound = lemma_steps(G.nX, mats, cover, 3)
    assert m == 14 < 2 * G.L * 4 and bound == 8, "mutation not caught: R(T) >= 2Lk without injectivity"
    G = Grid(1, 2, 1)
    row2 = [G.target(0, c, 1) for c in range(2)]
    assert not rank_at_most(row2, 3, 0) and rank_at_most(row2, 3, 1)
    G = Grid(2, 2, 1)
    lshape = [G.target(r, c, 1) for (r, c) in [(0, 0), (0, 1), (1, 0)]]
    cover = [(q, np.eye(3, dtype=np.int64)[t]) for t in range(3) for q in rowspace(lshape[t], 3)]
    assert lemma_steps(G.nX, lshape, cover, 3)[1] == 5
    assert not rank_at_most(lshape, 3, 1), "the L-shape at L = 1 has tensor rank <= 5 over GF(3)"
    print(f"  part 5: substitution steps on {count} random trees (rooted 2Lk, free root 2L(k-1)); Strassen 14 < 16 "
          f"with bound 8; exact GF(3), L = 1: row pair R = 4, L-shape R = 6 (> bound 5); mutations caught")


# ---------------------------------------------------------------------------------------------------------------
# Part 6: the L-shape at L = 2, exactly, over every tree. Cells a = (1, 0), b = (0, 0), c = (0, 1). With the line
# pair {b, c} at v (the column pair is its transpose; a disjoint pair at v is Proposition 3), a tree is
# 0 - U - {a, V - {b, c}} with edge ranks (p, q, s, d, e). Every proved inequality below cuts the cost-(6L - 1)
# profiles down to a few, and in each the Steiner points are rank summands of known matrices.


def nullspace(M, p):
    A, r = rref(M, p)
    pivots = [next(j for j in range(A.shape[1]) if A[i, j]) for i in range(r)]
    out = []
    for f in (j for j in range(A.shape[1]) if j not in pivots):
        v = np.zeros(A.shape[1], dtype=np.int64)
        v[f] = 1
        for i, pc in enumerate(pivots):
            v[pc] = -A[i, f] % p
        out.append(v)
    return out


def subspaces(r, g, p):
    """Every g-dimensional subspace of F_p^r, as a g x r matrix in reduced row echelon form."""
    for piv in itertools.combinations(range(r), g):
        free = [(i, j) for i in range(g) for j in range(piv[i] + 1, r) if j not in piv]
        for vals in itertools.product(range(p), repeat=len(free)):
            S = np.zeros((g, r), dtype=np.int64)
            S[range(g), piv] = 1
            for (i, j), v in zip(free, vals):
                S[i, j] = v
            yield S


def summand_factors(N, g, p):
    """The rank-g summands of N as pairs (NY, (Y^T N Y)^-1), one per g-dimensional subspace of col N on which
    Y^T N Y is invertible."""
    n = N.shape[0]
    if g == 0:
        yield np.zeros((n, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.int64)
        return
    basis, C = nullspace(N, p), []
    for i in range(n):
        e = np.eye(n, dtype=np.int64)[i]
        if rank(basis + [e], p) > len(basis):
            basis.append(e)
            C.append(e)
    C = np.array(C, dtype=np.int64).T
    for S in subspaces(C.shape[1], g, p):
        NY = N @ (C @ S.T) % p
        A, r = rref(np.hstack([NY.T @ C @ S.T % p, np.eye(g, dtype=np.int64)]), p)
        if (A[:, :g] == np.eye(g, dtype=np.int64)).all():
            yield NY, A[:, g:]


def summands(N, g, p):
    """Every symmetric A with rank A = g and rank(N - A) = rank N - g. Rank additivity makes every g-inverse of N one
    of A, so A = N Y (Y^T N Y)^-1 Y^T N with col A = col NY: one A for each g-dimensional subspace of col N on which
    Y^T N Y is invertible (any field)."""
    return [NY @ H @ NY.T % p for NY, H in summand_factors(N, g, p)]


def lshape_profiles(L):
    """Edge ranks (p, q, s, d, e) of 0 - U - {a, V - {b, c}} with cost < 6L that pass every proved inequality."""
    cons = [((1, 1, 0, 0, 0), 2 * L), ((1, 0, 1, 1, 0), 2 * L), ((1, 0, 1, 0, 1), 2 * L),  # rank S_t
            ((0, 0, 0, 1, 1), 2 * L), ((0, 1, 1, 1, 0), 2 * L), ((0, 1, 1, 0, 1), 4 * L),  # r(b,c), r(a,b), r(a,c)
            ((1, 1, 1, 1, 0), 4 * L), ((1, 1, 1, 0, 1), 4 * L), ((1, 0, 1, 1, 1), 4 * L),  # pairs from 0: Theorem S
            ((1, 1, 1, 1, 1), 5 * L),                                                      # Theorem S: 2L + R(T') = 5L
            ((1, 1, 2, 1, 1), 6 * L)]                                                      # A3's cap at U
    return [r for r in itertools.product(range(4 * L + 1), repeat=5)
            if sum(r) < 6 * L and all(np.dot(c, r) >= h for c, h in cons)]


def lshape_pairs(mats, prof, p, item, strategy, Vs):
    Sa, Sb, Sc = mats
    pp, q, s, d, e = prof
    rk = lambda M: rank(M % p, p)
    if strategy == "UV":
        pairs = ((item, V) for V in Vs)
    elif strategy == "U":
        pairs = ((item, (item + A) % p) for A in (summands((Sc - item) % p, s, p) if rk(Sc - item) == s + e else []))
    else:
        pairs = (((Sa - A) % p, item) for A in (summands((Sa - item) % p, q, p) if rk(Sa - item) == q + s else []))
    tried, best, found = 0, None, []
    for U, V in pairs:
        r = (rk(U), rk(Sa - U), rk(V - U), rk(Sb - V), rk(Sc - V))
        tried += 1
        best = sum(r) if best is None else min(best, sum(r))
        if all(x <= y for x, y in zip(r, prof)):
            found.append((U, V))
    return tried, best, found


def lshape_pairs_star(args):
    return lshape_pairs(*args)


def lshape_search(mats, prof, p, procs=1):
    """Every tree of profile `prof` (exact ranks): pin U or V as rank summands, pair them, check the rest. Returns
    (pairs tried, least cost among them, trees within the profile)."""
    Sa, Sb, Sc = mats
    pp, q, s, d, e = prof
    rk = lambda M: rank(M % p, p)
    Us = summands(Sa, pp, p) if pp + q == rk(Sa) else None
    Vs = [(Sb + A) % p for A in summands((Sc - Sb) % p, d, p)] if d + e == rk(Sc - Sb) else None
    tight = q + s + e == rk(Sa - Sc)
    if Us is not None and Vs is not None:
        outer, strategy = Us, "UV"
    elif Us is not None and tight:
        outer, strategy = Us, "U"
    elif Vs is not None and tight:
        outer, strategy = Vs, "V"
    else:
        raise AssertionError(f"no summand strategy for profile {prof}")
    jobs = [(mats, prof, p, item, strategy, Vs) for item in outer]
    if procs > 1:
        import multiprocessing
        with multiprocessing.Pool(procs) as pool:
            parts = pool.map(lshape_pairs_star, jobs, chunksize=1)
    else:
        parts = [lshape_pairs(*job) for job in jobs]
    bests = [b for _, b, _ in parts if b is not None]
    return (sum(t for t, _, _ in parts), min(bests) if bests else None, [f for _, _, fs in parts for f in fs])


def part6(rng, p=3, procs=1):
    """The L-shape at L = 2 over GF(p): no tree costs less than 2Lk = 12. Checks `summands` against brute force at
    n = 3 and 4; mutation: a planted cost-11 tree (b moved next to V) is found."""
    for n, trials in ((3, 6), (4, 2)):
        allsym = []
        for vals in itertools.product(range(p), repeat=n * (n + 1) // 2):
            A = np.zeros((n, n), dtype=np.int64)
            A[np.triu_indices(n)] = vals
            A = (A + np.triu(A, 1).T) % p
            allsym.append((A, rank(A, p)))
        for _ in range(trials):
            N = sum(rng.choice([1, p - 1]) * np.outer(z, z) for z in
                    (np.array([rng.randrange(p) for _ in range(n)]) for _ in range(rng.randrange(2, n + 1)))) % p
            r = rank(N, p)
            for g in range(1, r + 1):
                brute = {A.tobytes() for A, ra in allsym if ra == g and rank((N - A) % p, p) == r - g}
                got = {A.astype(np.int64).tobytes() for A in summands(N, g, p)}
                assert got == brute, f"summands({g}) wrong at n = {n}"
    L = 2
    G = Grid(2, 2, L)
    mats = [G.target(1, 0, 1), G.target(0, 0, 1), G.target(0, 1, 1)]
    profs = lshape_profiles(L)
    assert profs and all(sum(r) == 6 * L - 1 for r in profs), profs
    report = []
    for prof in profs:
        tried, best, found = lshape_search(mats, prof, p, procs)
        assert not found, f"a tree of cost {sum(prof)} < 12 on the L-shape: profile {prof}"
        assert best >= 6 * L
        report.append(f"{prof}: {tried} pairs, least {best}")
    prof = next(r for r in profs if r[0] + r[1] == 2 * L and r[1] + r[2] + r[4] == 4 * L)
    pp, q, s, d, e = prof
    U, V = next((U, V) for U in summands(mats[0], pp, p) for A in summands((mats[2] - U) % p, s, p)
                for V in [(U + A) % p] if rank((mats[2] - V) % p, p) == e)
    while True:
        H = sum(rng.choice([1, p - 1]) * np.outer(z, z) for z in
                (np.array([rng.randrange(p) for _ in range(G.n)]) for _ in range(d))) % p
        if rank(H, p) == d:
            break
    planted = [mats[0], (V + H) % p, mats[2]]
    assert lshape_search(planted, prof, p)[2], "mutation not caught: a planted cost-11 tree was missed"
    print(f"  part 6: L-shape, L = 2, GF({p}): {len(profs)} profiles of cost 11 pass the inequalities; none is "
          f"realized ({'; '.join(report)}); summands match brute force; planted tree found")


# ---------------------------------------------------------------------------------------------------------------
# Part 7: the L-shape at every L. Gate lemma: a symmetric A has a rank summand A0 of N with rank(A - A0) <= alpha =
# rank A + rank(N - A) - rank N. Exact case: U a summand of S_a and V - S_b one of S_c - S_b give rank(V - U) >= 2L.
# Surgery: moving U and V to their gates doesn't raise the cost, which is then 4L + rank(V0 - U0) >= 6L.


def inverse(M, p):
    g = M.shape[0]
    A, _ = rref(np.hstack([M % p, np.eye(g, dtype=np.int64)]), p)
    assert (A[:, :g] == np.eye(g, dtype=np.int64)).all(), "singular"
    return A[:, g:]


def gate(N, A, p, rng, broken=False):
    """A summand A0 of N with rank(A - A0) <= alpha, by the gate lemma's steps. Each step takes w = A x = B y in
    col A and col B (B = N - A; alpha > 0 forces one) and changes A by a matrix of rank r that lowers alpha by >= r:
    (1) x^T A x != 0: A - w w^T / x^T A x; (2) y^T B y != 0: A + w w^T / y^T B y; (3) both 0: the rank-2 reduction
    of A along (x, e_k), w_k != 0, which keeps rank(B + E) <= rank B. `broken` drops a term of (3). Returns (A0,
    options used)."""
    n, rN = N.shape[0], rank(N, p)
    A0 = A % p
    alpha0 = alpha = rank(A0, p) + rank((N - A0) % p, p) - rN
    used = []
    while alpha:
        B = (N - A0) % p
        ker = nullspace(np.hstack([A0, (-B) % p]), p)
        assert any((A0 @ z[:n] % p).any() for z in ker), "alpha > 0 but col A meets col B only in 0"
        while True:
            v = sum(rng.randrange(p) * z for z in ker) % p
            x, y = v[:n], v[n:]
            w = A0 @ x % p
            if w.any():
                break
        cA, cB = int(x @ A0 @ x % p), int(y @ B @ y % p)
        if cA:
            E, opt = np.outer(w, w) * pow(cA, p - 2, p) % p, 1
        elif cB:
            E, opt = -np.outer(w, w) * pow(cB, p - 2, p) % p, 2
        else:
            k = next(i for i in range(n) if w[i])
            gi, w2, d = pow(int(w[k]), p - 2, p), A0[:, k], int(A0[k, k])
            c = 0 if broken else -d * gi * gi % p
            E = (c * np.outer(w, w) % p + gi * ((np.outer(w, w2) + np.outer(w2, w)) % p)) % p
            opt = 3 if d else 4
        A1 = (A0 - E) % p
        r = rank(E, p)
        a1 = rank(A1, p) + rank((N - A1) % p, p) - rN
        assert r >= 1 and a1 <= alpha - r, f"gate option {min(opt, 3)}: a change of rank {r} took alpha {alpha} to {a1}"
        A0, alpha = A1, a1
        used.append(opt)
    assert rank((A - A0) % p, p) <= alpha0
    return A0, used


def rsym(n, r, rng, p):
    M = np.zeros((n, n), dtype=np.int64)
    for _ in range(r):
        z = np.array([rng.randrange(p) for _ in range(n)])
        M = (M + rng.randrange(1, p) * np.outer(z, z)) % p
    return M


def random_summand(N, g, rng, p):
    """A random rank-g summand of N, as (NY, (Y^T N Y)^-1)."""
    n = N.shape[0]
    if g == 0:
        return np.zeros((n, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.int64)
    while True:
        Y = np.array([[rng.randrange(p) for _ in range(g)] for _ in range(n)], dtype=np.int64)
        if rank(Y.T @ N @ Y % p, p) == g:
            return N @ Y % p, inverse(Y.T @ N @ Y % p, p)


def lshape_z(L, p):
    """The L-shape in coordinates (x0, x1, y0, z = y1 - y0): S_a = x1.y0, S_b = x0.y0, S_c - S_b = x0.z."""
    G = Grid(2, 2, L)
    T = np.eye(G.n, dtype=np.int64)
    for k in range(L):
        T[G.y(k, 1), G.y(k, 0)] = 1
    mats = [T.T @ G.target(r, c, 1) @ T % p for r, c in ((1, 0), (0, 0), (0, 1))]
    blocks = {"x0": [G.x(0, k) for k in range(L)], "x1": [G.x(1, k) for k in range(L)],
              "y0": [G.y(k, 0) for k in range(L)], "z": [G.y(k, 1) for k in range(L)]}
    return G, mats, blocks


def uv_steps(L, p, mats, blocks, mu, K, be, Gm):
    """The exact case's proof on U = mu K mu^T (a summand of S_a) and V = S_b + be Gm be^T (S_b plus a summand of
    S_c - S_b), step by step. Returns rank(V - U)."""
    Sa, Sb, Sc = mats
    n, pp, d = Sa.shape[0], mu.shape[1], be.shape[1]
    U, V = mu @ K % p @ mu.T % p, (Sb + be @ Gm % p @ be.T) % p
    assert rank(U, p) == pp and rank((Sa - U) % p, p) == 2 * L - pp, "U is not a summand of S_a"
    assert rank((V - Sb) % p, p) == d and rank((Sc - V) % p, p) == 2 * L - d, "V - S_b is not a summand"
    s = rank((V - U) % p, p)
    P = np.hstack([be, mu])
    Dinv = np.zeros((d + pp, d + pp), dtype=np.int64)
    Dinv[:d, :d], Dinv[d:, d:] = inverse(Gm, p), -inverse(K, p) % p
    Big = np.block([[Sb, P], [P.T, -Dinv % p]]) % p
    assert s == rank(Big, p) - (d + pp), "Schur complement of -D^-1"
    xy = blocks["x0"] + blocks["y0"]
    top = blocks["x1"] + blocks["z"]
    rest = top + list(range(n, n + d + pp))
    R = (Big[np.ix_(rest, rest)] - Big[np.ix_(rest, xy)] @ inverse(Big[np.ix_(xy, xy)], p) % p
         @ Big[np.ix_(xy, rest)]) % p
    assert rank(Big, p) == 2 * L + rank(R, p), "eliminating the (x0, y0) block"
    t = len(top)
    assert not R[:t, :t].any(), "R's (x1, z) block is 0"
    Phi, Psi = R[:t, t:], R[t:, t:]
    ker = nullspace(Phi, p)
    Kb = np.array(ker, dtype=np.int64).T if ker else np.zeros((d + pp, 0), dtype=np.int64)
    assert rank(R, p) == 2 * rank(Phi, p) + rank(Kb.T @ Psi @ Kb % p, p), "rank of [[0, Phi], [Phi^T, Psi]]"
    bz, ux1 = be[blocks["z"]], mu[blocks["x1"]]
    assert rank(Phi, p) == rank(bz, p) + rank(ux1, p)
    assert 2 * rank(bz, p) >= d and 2 * rank(ux1, p) >= pp, "ker b_z, ker u_x1 totally isotropic"
    assert s >= 2 * L, f"exact case: rank(V - U) = {s} < 2L"
    return s


def part7(rng):
    """The L-shape at every L (odd characteristic). Checks the gate lemma on random pairs (all three options, and
    the bound attained), the exact case step by step (exhaustively at L = 1, at random for L = 2..4) and the surgery
    on random trees of the open topology. Mutations: option (3) without its w w^T term; alpha - 1 on a tight
    instance; 2L + 1 in the exact case on U = 0, V = S_b; 6L + 1 on the star."""
    opts, gates, tight_gate = {1: 0, 2: 0, 3: 0, 4: 0}, 0, 0
    for p in (3, 5, 10007):
        for _ in range(400):
            n = rng.choice([3, 4, 6, 8])
            N = rsym(n, rng.randrange(0, n + 1), rng, p)
            if rng.random() < 0.4:
                A = rsym(n, rng.randrange(0, n + 1), rng, p)
            else:
                NY, H = random_summand(N, rng.randrange(rank(N, p) + 1), rng, p)
                A = (NY @ H % p @ NY.T + rsym(n, rng.randrange(1, 3), rng, p)) % p
            A0, used = gate(N, A, p, rng)
            alpha = rank(A % p, p) + rank((N - A) % p, p) - rank(N, p)
            tight_gate += alpha > 0 and rank((A - A0) % p, p) == alpha
            for o in used:
                opts[o] += 1
            gates += 1
    assert opts[1] and opts[2] and opts[3], f"gate options not all exercised: {opts}"
    N, A = np.diag([1, 0, 0]), np.diag([2, 0, 0])
    alpha = rank(A, 3) + rank((N - A) % 3, 3) - rank(N, 3)
    assert alpha == 1 and min(rank((A - S) % 3, 3) for g in range(2) for S in summands(N, g, 3)) == alpha, \
        "mutation not caught: the gate bound alpha - 1"
    caught = 0
    for _ in range(200):
        n = rng.choice([4, 6])
        N = rsym(n, rng.randrange(1, n + 1), rng, 3)
        try:
            gate(N, rsym(n, rng.randrange(1, n + 1), rng, 3), 3, rng, broken=True)
        except AssertionError:
            caught += 1
    assert caught, "mutation not caught: option (3) without its w w^T term"

    exact, least = 0, {}
    for p in (3, 5):
        G, mats, blocks = lshape_z(1, p)
        for pp in range(3):
            for mu, K in summand_factors(mats[0], pp, p):
                for d in range(3):
                    for be, Gm in summand_factors((mats[2] - mats[1]) % p, d, p):
                        s = uv_steps(1, p, mats, blocks, mu, K, be, Gm)
                        least[1] = min(least.get(1, s), s)
                        exact += 1
    for L, p, trials in ((2, 3, 400), (2, 5, 300), (3, 3, 300), (3, 10007, 100), (4, 3, 200)):
        G, mats, blocks = lshape_z(L, p)
        for _ in range(trials):
            mu, K = random_summand(mats[0], rng.randrange(2 * L + 1), rng, p)
            be, Gm = random_summand((mats[2] - mats[1]) % p, rng.randrange(2 * L + 1), rng, p)
            s = uv_steps(L, p, mats, blocks, mu, K, be, Gm)
            least[L] = min(least.get(L, s), s)
            exact += 1
    G, mats, blocks = lshape_z(2, 3)
    s = uv_steps(2, 3, mats, blocks, *next(summand_factors(mats[0], 0, 3)),
                 *next(summand_factors((mats[2] - mats[1]) % 3, 0, 3)))
    assert s == 4 and not s >= 5, "mutation not caught: 2L + 1 in the exact case at U = 0, V = S_b"

    trees, tight = 0, 0
    for p in (3, 5, 10007):
        for _ in range(300):
            L = rng.choice([1, 2, 3])
            G = Grid(2, 2, L)
            a, b, c = (1, 0), (0, 0), (0, 1)
            Sa, Sb, Sc = (G.target(*t, 1) for t in (a, b, c))
            kind = rng.randrange(3)
            if kind == 0:
                U = random_point(G, [a, b, c], rng, p)
                V = (U + random_point(G, [b, c], rng, p)) % p if rng.random() < 0.5 else random_point(G, [b, c], rng, p)
            else:
                mu, K = random_summand(Sa, rng.randrange(2 * L + 1), rng, p)
                be, Gm = random_summand((Sc - Sb) % p, rng.randrange(2 * L + 1), rng, p)
                U = (mu @ K @ mu.T + (kind == 1) * rsym(G.n, rng.randrange(1, 3), rng, p)) % p
                V = (Sb + be @ Gm @ be.T + rsym(G.n, rng.randrange(0, 3), rng, p)) % p
            rk = lambda M: rank(M % p, p)
            pp, q, s, d, e = rk(U), rk(Sa - U), rk(V - U), rk(Sb - V), rk(Sc - V)
            cost = pp + q + s + d + e
            U0, _ = gate(Sa, U, p, rng)
            A0, _ = gate((Sc - Sb) % p, (V - Sb) % p, p, rng)
            V0 = (Sb + A0) % p
            assert rk(U - U0) <= pp + q - 2 * L and rk(V - V0) <= d + e - 2 * L, "gate bound"
            assert rk(U0) + rk(Sa - U0) == 2 * L and rk(Sb - V0) + rk(Sc - V0) == 2 * L, "gates are geodesic"
            new = rk(U0) + rk(Sa - U0) + rk(V0 - U0) + rk(Sb - V0) + rk(Sc - V0)
            assert new <= cost, f"surgery raised the cost: {cost} -> {new}"
            assert new >= 6 * L and cost >= 6 * L, f"L-shape tree of cost {cost} < 6L"
            trees += 1
            tight += cost == 6 * L
    G = Grid(2, 2, 2)
    star = sum(rank(G.target(*t, 1), 3) for t in ((1, 0), (0, 0), (0, 1)))
    assert star == 12 and not star >= 13, "mutation not caught: 6L + 1 on the star"
    print(f"  part 7: gate lemma on {gates} pairs (options {opts[1]}, {opts[2]}, {opts[3] + opts[4]}; bound attained "
          f"{tight_gate} times); exact case on {exact} summand pairs (least rank(V - U): "
          f"{', '.join(f'L={L}: {v}' for L, v in sorted(least.items()))}); surgery on {trees} trees ({tight} at 6L); "
          f"{caught} broken-gate runs caught; mutations caught")


# ---------------------------------------------------------------------------------------------------------------
# Part 8: symmetric rigidity fails at depth 3. Moving each Steiner point, from the leaves up, onto a geodesic between
# its two children (the gate lemma) never raises the cost, so the balanced tree 0 - W - {V1 - {00, 01},
# V2 - {10, 11}} on the full 2 x 2 block is the finite family V_r = S_r0 + (a summand of S_r1 - S_r0), W a summand
# of V1 or V2, of cost 4L + rank V_i + rank(V_other - W). At L = 2 over GF(3) its least cost is 14 < 16, and one
# least tree is a merge-free 7-product tree for 2 x 2 matrix multiplication that holds over the integers.


def winograd_tree(P, Q, L):
    """The merge-free tree for the P x Q block at inner length L (Q, L even). For each inner pair (k, k') = (2j, 2j+1)
    and column pair (c, c') = (2m, 2m+1):
      W_m   += y_kc' y_k'c
      V_rm  += -(x_rk - y_k'c)(x_rk' - y_kc')              (V_rm hangs from W_m)
      t_rc  += x_rk (y_kc + x_rk' - y_kc')                 (t_rc, t_rc' hang from V_rm)
      t_rc' += x_rk' (y_k'c' + x_rk - y_k'c)
    Returns the grid and the edges (parent, child, products (coefficient, g, h))."""
    G = Grid(P, Q, L)
    e = lambda i: np.eye(G.n, dtype=np.int64)[i]
    x = lambda r, k: e(G.x(r, k))
    y = lambda k, c: e(G.y(k, c))
    edges = {}
    for m in range(Q // 2):
        c, c2 = 2 * m, 2 * m + 1
        for j in range(L // 2):
            k, k2 = 2 * j, 2 * j + 1
            edges.setdefault(("0", ("W", m)), []).append((1, y(k, c2), y(k2, c)))
            for r in range(P):
                edges.setdefault((("W", m), ("V", r, m)), []).append((-1, x(r, k) - y(k2, c), x(r, k2) - y(k, c2)))
                edges.setdefault((("V", r, m), ("t", r, c)), []).append((1, x(r, k), y(k, c) + x(r, k2) - y(k, c2)))
                edges.setdefault((("V", r, m), ("t", r, c2)), []).append((1, x(r, k2), y(k2, c2) + x(r, k) - y(k2, c)))
    return G, edges


def tree_values(G, edges):
    """Each node's doubled quadratic form over the integers: a product a*g*h adds a(g h^T + h g^T), and a target
    sum_k x_rk y_kc is G.target(r, c, 1)."""
    vals = {"0": np.zeros((G.n, G.n), dtype=np.int64)}
    pending = dict(edges)
    while pending:
        for (u, v), prods in list(pending.items()):
            if u in vals:
                vals[v] = vals[u] + sum(a * (np.outer(g, h) + np.outer(h, g)) for a, g, h in prods)
                del pending[u, v]
    return vals


def winograd_check(P, Q, L, edges=None):
    """Exact over the integers: every target is its sum_k x_rk y_kc, the tree has (LQ/4)(3P + 1) products against
    LPQ pure ones, and it uses PLQ distinct mixed words. Returns (products, mixed words)."""
    G, tree = winograd_tree(P, Q, L)
    edges = edges or tree
    vals = tree_values(G, edges)
    for r in range(P):
        for c in range(Q):
            assert (vals["t", r, c] == G.target(r, c, 1)).all(), f"target ({r}, {c}) is not sum_k x_rk y_kc"
    prods = [(g, h) for ps in edges.values() for _, g, h in ps]
    mixed = {tuple(w) for g, h in prods for w in (g, h) if w[:G.nX].any() and w[G.nX:].any()}
    assert len(prods) == L * Q * (3 * P + 1) // 4 < L * P * Q or P == 1
    assert len(mixed) == P * L * Q
    return len(prods), len(mixed)


BALANCED = None


def set_balanced(data):
    global BALANCED
    BALANCED = data


def balanced_row(i):
    """Least balanced cost over every V2 and every W, for one V1, from BALANCED."""
    V1s, V2s, sig, L, p = BALANCED
    least = None
    for j, V2 in enumerate(V2s):
        r1, r2 = rank(V1s[i], p), rank(V2, p)
        if max(r1, r2) >= 3 * L:
            continue
        mine, other, ws = (V1s[i], V2, sig[0][i]) if r1 <= r2 else (V2, V1s[i], sig[1][j])
        cost = 4 * L + rank(mine, p) + min(rank((other - W) % p, p) for W in ws)
        least = cost if least is None else min(least, cost)
    return least


def balanced_exhaustive(L, p, procs):
    """Every balanced normal-form tree of the full 2 x 2 block: V1 = S_00 + a summand of S_01 - S_00, V2 likewise in
    row 1, W a summand of whichever of V1, V2 has the lower rank. A pair with max(rank V1, rank V2) >= 3L needs no
    search: rank(V2 - W) >= L from V2's x1 rows (W has none), so the cost is >= 4L + 3L + L. Returns the least cost
    among the rest, how many pairs needed a search, and how many pairs there are."""
    G = Grid(2, 2, L)
    S = {t: G.target(*t, 1) for t in ((0, 0), (0, 1), (1, 0), (1, 1))}
    V1s = [(S[0, 0] + A) % p for g in range(2 * L + 1) for A in summands((S[0, 1] - S[0, 0]) % p, g, p)]
    V2s = [(S[1, 0] + A) % p for g in range(2 * L + 1) for A in summands((S[1, 1] - S[1, 0]) % p, g, p)]
    low = lambda Vs: [[W for g in range(rank(V, p) + 1) for W in summands(V, g, p)] if rank(V, p) < 3 * L else []
                      for V in Vs]
    data = (V1s, V2s, (low(V1s), low(V2s)), L, p)
    searched = sum(max(rank(a, p), rank(b, p)) < 3 * L for a in V1s for b in V2s)
    set_balanced(data)
    if procs > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(procs, set_balanced, (data,)) as pool:
            parts = pool.map(balanced_row, range(len(V1s)), chunksize=1)
    else:
        parts = [balanced_row(i) for i in range(len(V1s))]
    return min(c for c in parts if c is not None), searched, len(V1s) * len(V2s)


def part8(rng, exhaustive=False, procs=1):
    """The counterexample to symmetric rigidity. Checks the tree exactly over the integers for P x Q blocks at
    inner length L (it is a polynomial identity, so it holds over every commutative ring), its symmetric cost on the
    2 x 2 block (14, every edge one product), and with `exhaustive` that 14 is the least balanced normal-form cost
    at L = 2 over GF(3). Mutation: a sign flipped in one correction product breaks a target."""
    report = []
    for P, Q, L in ((2, 2, 2), (3, 2, 2), (4, 2, 2), (2, 4, 2), (3, 2, 4), (8, 4, 4)):
        prods, words = winograd_check(P, Q, L)
        report.append(f"{P}x{Q}, L={L}: {prods} products < {L * P * Q}, {words} mixed words")
    G, edges = winograd_tree(2, 2, 2)
    vals = tree_values(G, edges)
    for p in (3, 5, 10007):
        cost = sum(rank((vals[v] - vals[u]) % p, p) for u, v in edges)
        assert cost == 14 and all(rank((vals[v] - vals[u]) % p, p) == 2 for u, v in edges), "symmetric cost 14"
    broken = dict(edges)
    (a, g, h), = broken[("V", 1, 0), ("t", 1, 1)]
    broken[("V", 1, 0), ("t", 1, 1)] = [(a, g, h - 2 * np.eye(G.n, dtype=np.int64)[G.x(1, 0)])]
    try:
        winograd_check(2, 2, 2, broken)
        raise RuntimeError("mutation not caught: a sign flipped in a correction product")
    except AssertionError:
        pass
    print("  part 8: the merge-free tree computes every target exactly over the integers (" + "; ".join(report)
          + "); on the 2 x 2 block at L = 2 its symmetric cost is 14 < 16 over GF(3), GF(5), GF(10007); mutation caught")
    if exhaustive:
        best, searched, pairs = balanced_exhaustive(2, 3, procs)
        assert best == 14, f"least balanced normal-form cost {best}, expected 14"
        print(f"  part 8: every balanced normal-form tree at L = 2 over GF(3): {pairs} pairs (V1, V2), {searched} "
              f"searched over W; least cost {best}")


PARTS = {"1": part1, "2": part2, "3": part3, "4": part4, "5": part5, "6": part6, "7": part7, "8": part8}


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*")
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--seeds", default="0:2", help="part 4's seeds, A:B")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default=None, help="part 4's JSON lines")
    ap.add_argument("--field", type=int, default=3, help="part 6's prime")
    args = ap.parse_args(argv)
    rng = random.Random(20261015)
    for name in args.parts or list(PARTS):
        print(f"part {name}:")
        if name == "2":
            part2(rng, args.big)
        elif name == "4":
            a, b = (int(s) for s in args.seeds.split(":"))
            part4(rng, range(a, b), args.procs, args.out, args.steps)
        elif name == "6":
            part6(rng, args.field, args.procs)
        elif name == "8":
            part8(rng, exhaustive=args.big, procs=args.procs)
        else:
            PARTS[name](rng)
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1:])

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
     and columns), on random Steiner points; and local search for symmetric gains in deeper trees at L = 2.

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


PARTS = {"1": part1, "2": part2, "3": part3, "4": part4}


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*")
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--seeds", default="0:2", help="part 4's seeds, A:B")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default=None, help="part 4's JSON lines")
    args = ap.parse_args(argv)
    rng = random.Random(20261015)
    for name in args.parts or list(PARTS):
        print(f"part {name}:")
        if name == "2":
            part2(rng, args.big)
        elif name == "4":
            a, b = (int(s) for s in args.seeds.split(":"))
            part4(rng, range(a, b), args.procs, args.out, args.steps)
        else:
            PARTS[name](rng)
    print("ok")


if __name__ == "__main__":
    main(sys.argv[1:])

"""Forks in the ring fragment, mod 2: the cheapest accumulator tree for a set of committed words.

With free forks (an out-of-place MMA leaves its C live) the accumulators form a tree rooted at 0, and a word's value
is the sum of the edge segments on its root path. Mod 2 a quadratic polynomial on the cube, up to affine terms, is an
alternating form; a product of two affine forms has rank at most 2, and every alternating form of rank 2r is a sum of
r such products. So an edge from form Y to form Y' costs rank(Y' - Y)/2 products, and the cheapest fork program for
words Q_1..Q_k is a Steiner tree on {0, Q_1, ..., Q_k} in that metric. Fork-free chains are its paths.

Forms are bitmasks over the pairs (a, b), a < b, of the variables.
"""

from __future__ import annotations

import itertools
import sys


def pairs(nv: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(nv) for b in range(a + 1, nv)]


def rank_alt(mask: int, nv: int, pr: list[tuple[int, int]]) -> int:
    rows = [0] * nv
    for bit, (a, b) in enumerate(pr):
        if mask >> bit & 1:
            rows[a] ^= 1 << b
            rows[b] ^= 1 << a
    r = 0
    for col in range(nv):
        piv = next((i for i in range(r, nv) if rows[i] >> col & 1), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(nv):
            if i != r and rows[i] >> col & 1:
                rows[i] ^= rows[r]
        r += 1
    return r


def form(terms: list[tuple[int, int]], pr: list[tuple[int, int]]) -> int:
    idx = {p: i for i, p in enumerate(pr)}
    m = 0
    for a, b in terms:
        m ^= 1 << idx[(min(a, b), max(a, b))]
    return m


def row_words(L: int, k: int) -> tuple[int, list[tuple[int, int]], list[int]]:
    """One row's words: u_j (j < L) against k independent coin blocks c^(x), Q_x = sum_j u_j c^(x)_j.

    A row's prefix words T_t(i, l) = sum_j u_ij S_jlt, with S_jlt the coin sums of blocks t' <= t, are these up to an
    invertible change of the coin variables, which keeps every rank."""
    nv = L * (k + 1)
    pr = pairs(nv)
    qs = [form([(j, L * (x + 1) + j) for j in range(L)], pr) for x in range(k)]
    return nv, pr, qs


def median(L: int, k: int) -> tuple[int, int]:
    """Cheapest star with one fork point Y: d(0, Y) + sum_x d(Y, Q_x), over every form Y (exhaustive)."""
    nv, pr, qs = row_words(L, k)
    best = (10**9, -1)
    for y in range(1 << len(pr)):
        c = rank_alt(y, nv, pr) + sum(rank_alt(y ^ q, nv, pr) for q in qs)
        if c < best[0]:
            best = (c, y)
    return best[0] // 2, best[1]


def steiner_exact(nv: int, pr: list[tuple[int, int]], terms: list[int]) -> int:
    """Dreyfus–Wagner over all 2^|pr| forms, terminals `terms` (0 among them)."""
    n = 1 << len(pr)
    rk = [rank_alt(m, nv, pr) // 2 for m in range(n)]
    k = len(terms)
    inf = 10**9
    dp = [[inf] * n for _ in range(1 << k)]
    for i, t in enumerate(terms):
        for v in range(n):
            dp[1 << i][v] = rk[v ^ t]
    for s in range(1, 1 << k):
        if s & (s - 1) == 0:
            continue
        row = dp[s]
        sub = (s - 1) & s
        while sub:
            if sub < (s ^ sub):
                a, b = dp[sub], dp[s ^ sub]
                for v in range(n):
                    x = a[v] + b[v]
                    if x < row[v]:
                        row[v] = x
            sub = (sub - 1) & s
        # relax along edges: row[v] = min_u row[u] + d(u, v)
        best = row[:]
        for u in range(n):
            ru = row[u]
            if ru >= inf:
                continue
            for v in range(n):
                x = ru + rk[u ^ v]
                if x < best[v]:
                    best[v] = x
        dp[s] = best
    return min(dp[(1 << k) - 1])


def cells_2x2(L: int) -> tuple[int, list[tuple[int, int]], list[int]]:
    """Two rows' u (u_ij, i < 2) against two columns' coins (c_lj, l < 2), B = 1: words Q_il = sum_j u_ij c_lj."""
    nv = 4 * L
    pr = pairs(nv)
    u = lambda i, j: i * L + j
    c = lambda l, j: 2 * L + l * L + j
    qs = [form([(u(i, j), c(l, j)) for j in range(L)], pr) for i in range(2) for l in range(2)]
    return nv, pr, qs


if __name__ == "__main__":
    for L, k in [(1, 2), (1, 3), (2, 2)]:
        cost, _ = median(L, k)
        print(f"one row, L={L}, k={k} words: best one-fork star {cost} products; chains {k * L}")
    nv, pr, qs = cells_2x2(1)
    print(f"2x2 cells, L=1: Steiner tree {steiner_exact(nv, pr, [0] + qs)} products; chains 4")
    nv, pr, qs = row_words(1, 3)
    print(f"one row, L=1, k=3: Steiner tree {steiner_exact(nv, pr, [0] + qs)} products; chains 3")
    sys.exit(0)

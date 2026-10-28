"""Depth-one cost of integer bilinear schemes <m,k,n> of rank R on s8 blocks: kappa_1 = (1/mkn) sum_t c_h(N(u_t), N(v_t)).

N(u) is the number of values u.x takes over x in [-128, 127]^(mk), counted exactly (a sumset); c_h is the least cost of
one exact product over every slicing (sched, depth 0). kappa_1 is bracketed by c_h's grid tables (the lower at N rounded
down, the upper at the covering interval 255 |u|_1 + 1 rounded up), next to the closed form (R + R_wide)/(mkn), where
R_wide counts the products with an operand of two or more entries (each costs at least 2, the rest at least 1).

Schemes: Strassen, Strassen-Winograd and Laderman (written here, each checked against the matmul tensor), and every
factorization in AlphaTensor's factorizations_r.npz (github.com/google-deepmind/alphatensor, algorithms/), checked the same
way.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np

from .sched import Tables


def matmul_tensor(m: int, k: int, n: int) -> np.ndarray:
    T = np.zeros((m * k, k * n, m * n), dtype=np.int64)
    for i, j, l in itertools.product(range(m), range(k), range(n)):
        T[i * k + j, j * n + l, i * n + l] = 1
    return T


def computes(U, V, W, m, k, n) -> bool:
    """U, V, W (mk x R, kn x R, mn x R) are a scheme for <m,k,n>, with C read row-major or transposed."""
    S = np.einsum("ar,br,cr->abc", U.astype(float), V.astype(float), W.astype(float))
    S = np.round(S).astype(np.int64) if np.allclose(S, np.round(S)) else np.full(S.shape, -10 ** 9)
    T = matmul_tensor(m, k, n)
    Tt = np.zeros_like(T)
    for i, j, l in itertools.product(range(m), range(k), range(n)):
        Tt[i * k + j, j * n + l, l * m + i] = 1
    return bool((S == T).all() or (S == Tt).all())


def count(u) -> int:
    """|{u.x : x in [-128, 127]^d}|, exactly."""
    lo = 0
    ind = np.ones(1, dtype=bool)
    for c in (int(x) for x in u if x):
        a = abs(c)
        step = np.zeros(255 * a + 1, dtype=bool)
        step[::a] = True
        ind = np.convolve(ind, step) > 0
        lo += -128 * a if c > 0 else -127 * a
    return int(ind.sum())


def _from_strings(m, k, n, prods):
    """prods: [(A terms, B terms, C terms)], each a list of (coef, 'r c') 1-based."""
    R = len(prods)
    U, V, W = np.zeros((m * k, R), int), np.zeros((k * n, R), int), np.zeros((m * n, R), int)
    for t, (a, b, c) in enumerate(prods):
        for s, cols, X in ((a, k, U), (b, n, V), (c, n, W)):
            for coef, ij in s:
                i, j = int(ij[0]) - 1, int(ij[1]) - 1
                X[i * cols + j, t] += coef
    return U, V, W


def strassen():
    P = [([(1, "11"), (1, "22")], [(1, "11"), (1, "22")], [(1, "11"), (1, "22")]),
         ([(1, "21"), (1, "22")], [(1, "11")], [(1, "21"), (-1, "22")]),
         ([(1, "11")], [(1, "12"), (-1, "22")], [(1, "12"), (1, "22")]),
         ([(1, "22")], [(1, "21"), (-1, "11")], [(1, "11"), (1, "21")]),
         ([(1, "11"), (1, "12")], [(1, "22")], [(-1, "11"), (1, "12")]),
         ([(1, "21"), (-1, "11")], [(1, "11"), (1, "12")], [(1, "22")]),
         ([(1, "12"), (-1, "22")], [(1, "21"), (1, "22")], [(1, "11")])]
    return _from_strings(2, 2, 2, P)


def winograd():
    # S1 = A21 + A22, S2 = S1 - A11, S3 = A11 - A21, S4 = A12 - S2; T1 = B12 - B11, T2 = B22 - T1, T3 = B22 - B12,
    # T4 = T2 - B21; M1 = A11 B11, M2 = A12 B21, M3 = S4 B22, M4 = A22 T4, M5 = S1 T1, M6 = S2 T2, M7 = S3 T3;
    # U2 = M1 + M6, U3 = U2 + M7, U4 = U2 + M5; C11 = M1 + M2, C12 = U4 + M3, C21 = U3 - M4, C22 = U3 + M5.
    S4 = [(1, "12"), (-1, "21"), (-1, "22"), (1, "11")]
    T4 = [(1, "22"), (-1, "12"), (1, "11"), (-1, "21")]
    P = [([(1, "11")], [(1, "11")], [(1, "11"), (1, "12"), (1, "21"), (1, "22")]),
         ([(1, "12")], [(1, "21")], [(1, "11")]),
         (S4, [(1, "22")], [(1, "12")]),
         ([(1, "22")], T4, [(-1, "21")]),
         ([(1, "21"), (1, "22")], [(1, "12"), (-1, "11")], [(1, "12"), (1, "22")]),
         ([(1, "21"), (1, "22"), (-1, "11")], [(1, "22"), (-1, "12"), (1, "11")], [(1, "12"), (1, "21"), (1, "22")]),
         ([(1, "11"), (-1, "21")], [(1, "22"), (-1, "12")], [(1, "21"), (1, "22")])]
    return _from_strings(2, 2, 2, P)


def laderman():
    a = lambda *t: [(c, s) for c, s in t]
    P = [
        (a((1, "11"), (1, "12"), (1, "13"), (-1, "21"), (-1, "22"), (-1, "32"), (-1, "33")), a((1, "22")), None),
        (a((1, "11"), (-1, "21")), a((-1, "12"), (1, "22")), None),
        (a((1, "22")), a((-1, "11"), (1, "12"), (1, "21"), (-1, "22"), (-1, "23"), (-1, "31"), (1, "33")), None),
        (a((-1, "11"), (1, "21"), (1, "22")), a((1, "11"), (-1, "12"), (1, "22")), None),
        (a((1, "21"), (1, "22")), a((-1, "11"), (1, "12")), None),
        (a((1, "11")), a((1, "11")), None),
        (a((-1, "11"), (1, "31"), (1, "32")), a((1, "11"), (-1, "13"), (1, "23")), None),
        (a((-1, "11"), (1, "31")), a((1, "13"), (-1, "23")), None),
        (a((1, "31"), (1, "32")), a((-1, "11"), (1, "13")), None),
        (a((1, "11"), (1, "12"), (1, "13"), (-1, "22"), (-1, "23"), (-1, "31"), (-1, "32")), a((1, "23")), None),
        (a((1, "32")), a((-1, "11"), (1, "13"), (1, "21"), (-1, "22"), (-1, "23"), (-1, "31"), (1, "32")), None),
        (a((-1, "13"), (1, "32"), (1, "33")), a((1, "22"), (1, "31"), (-1, "32")), None),
        (a((1, "13"), (-1, "33")), a((1, "22"), (-1, "32")), None),
        (a((1, "13")), a((1, "31")), None),
        (a((1, "32"), (1, "33")), a((-1, "31"), (1, "32")), None),
        (a((-1, "13"), (1, "22"), (1, "23")), a((1, "23"), (1, "31"), (-1, "33")), None),
        (a((1, "13"), (-1, "23")), a((1, "23"), (-1, "33")), None),
        (a((1, "22"), (1, "23")), a((-1, "31"), (1, "33")), None),
        (a((1, "12")), a((1, "21")), None),
        (a((1, "23")), a((1, "32")), None),
        (a((1, "21")), a((1, "13")), None),
        (a((1, "31")), a((1, "12")), None),
        (a((1, "33")), a((1, "33")), None),
    ]
    # C from the products (Laderman 1976): c_ij as sums of m_t
    C = {"11": [6, 14, 19], "12": [1, 4, 5, 6, 12, 14, 15], "13": [6, 7, 9, 10, 14, 16, 18],
         "21": [2, 3, 4, 6, 14, 16, 17], "22": [2, 4, 5, 6, 20], "23": [14, 16, 17, 18, 21],
         "31": [6, 7, 8, 11, 12, 13, 14], "32": [12, 13, 14, 15, 22], "33": [6, 7, 8, 9, 23]}
    prods = []
    for t, (u, v, _) in enumerate(P, start=1):
        prods.append((u, v, [(1, ij) for ij, ms in C.items() if t in ms]))
    return _from_strings(3, 3, 3, prods)


def _integral(u) -> np.ndarray:
    """u scaled by the least positive integer making it integral (a product's output weight absorbs the factor)."""
    for s in range(1, 65):
        w = np.asarray(u, dtype=float) * s
        if np.allclose(w, np.round(w)):
            return np.round(w).astype(np.int64)
    raise ValueError(f"coefficients {u} are not rationals with a small denominator")


def evaluate(T: Tables, lo0, up0, U, V, m, k, n):
    R = U.shape[1]
    kl = ku = 0
    wide = 0
    for t in range(R):
        u, v = _integral(U[:, t]), _integral(V[:, t])
        ca, cb = count(u), count(v)
        sa, sb = 255 * int(np.abs(u).sum()) + 1, 255 * int(np.abs(v).sum()) + 1
        kl += int(lo0[T.down(ca), T.down(cb)])
        iu, ju = int(T.upi(sa)), int(T.upi(sb))
        assert T.g[iu] >= sa and T.g[ju] >= sb
        ku += int(up0[iu, ju])
        wide += int(np.count_nonzero(u) > 1 or np.count_nonzero(v) > 1)
    d = 8 * m * k * n
    return R, wide, kl / d, ku / d, (R + wide) / (m * k * n)


def main(name: str, npz: str | None):
    T = Tables(name, 4, [])
    lo0, up0, _, _ = T.level(None, None, None, None, 0)
    rows = [("strassen", 2, 2, 2, *strassen()), ("winograd", 2, 2, 2, *winograd()), ("laderman", 3, 3, 3, *laderman())]
    if npz:
        d = np.load(npz, allow_pickle=True)
        for key in d.files:
            m, k, n = map(int, key.split(","))
            f = d[key]
            U, V, W = (f[0], f[1], f[2])
            rows.append((f"alphatensor {key}", m, k, n, np.asarray(U), np.asarray(V), np.asarray(W)))
    print(f"{name}: scheme  <m,k,n>  R/mkn  R_wide  kappa_1 in [lo, up]  (R + R_wide)/mkn")
    for label, m, k, n, U, V, W in rows:
        Ui = np.stack([_integral(U[:, t]) for t in range(U.shape[1])], 1)
        Vi = np.stack([_integral(V[:, t]) for t in range(V.shape[1])], 1)
        if 255 * max(int(np.abs(Ui).sum(0).max()), int(np.abs(Vi).sum(0).max())) + 1 > T.g[-1]:
            print(f"  {label}: an operand wider than the grid, skipped")
            continue
        ok = computes(U, V, W, m, k, n)
        R, wide, kl, ku, simple = evaluate(T, lo0, up0, U, V, m, k, n)
        print(f"  {label:<22} <{m},{k},{n}> {R / (m * k * n):.4f} {wide:>3}/{R:<3} [{kl:.4f}, {ku:.4f}] {simple:.4f}"
              + ("" if ok else "  NOT A SCHEME"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

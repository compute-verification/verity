"""BabyBear linear algebra for the red team: exact rank of small matrices, Lagrange interpolation with slack
(a polynomial of bounded degree through prescribed values at given points, the remaining freedom uniform), and
solving a small overdetermined system with a consistency check.  Python ints / numpy int64 (every product of two
reduced values fits in 63 bits)."""
from __future__ import annotations

import numpy as np

from ..compile import P


def inv(a: int) -> int:
    return pow(a % P, P - 2, P)


def rank_mod_p(M: np.ndarray) -> int:
    """Rank over F_p of an (r, c) int64 matrix (Gaussian elimination, vectorised over rows)."""
    A = (np.asarray(M, dtype=np.int64) % P).copy()
    r, c = A.shape
    rank = 0
    for col in range(c):
        if rank == r:
            break
        piv = np.nonzero(A[rank:, col])[0]
        if piv.size == 0:
            continue
        pr = rank + int(piv[0])
        if pr != rank:
            A[[rank, pr]] = A[[pr, rank]]
        A[rank] = (A[rank] * inv(int(A[rank, col]))) % P
        others = np.nonzero(A[:, col])[0]
        others = others[others != rank]
        if others.size:
            A[others] = (A[others] - (A[others, col][:, None] * A[rank][None, :]) % P) % P
        rank += 1
    return rank


def solve_overdetermined(A: list[list[int]], b: list[int]) -> tuple[list[int] | None, bool]:
    """``A`` (e x u) with e >= u over F_p: the unique solution of a full-rank u x u subsystem and whether ALL e
    equations are satisfied by it.  Returns (None, False) when no full-rank subsystem exists."""
    e, u = len(A), len(A[0])
    M = [[x % P for x in row] + [bb % P] for row, bb in zip(A, b)]
    # row-reduce the augmented matrix
    rows = M
    for col in range(u):                     # every column yields a pivot (else no full-rank subsystem): pivot row = col
        r = col
        pr = next((i for i in range(r, e) if rows[i][col]), None)
        if pr is None:
            return None, False
        rows[r], rows[pr] = rows[pr], rows[r]
        iv = inv(rows[r][col])
        rows[r] = [(x * iv) % P for x in rows[r]]
        for i in range(e):
            if i != r and rows[i][col]:
                f = rows[i][col]
                rows[i] = [(x - f * y) % P for x, y in zip(rows[i], rows[r])]
    x = [rows[i][u] for i in range(u)]
    consistent = all(rows[i][u] == 0 for i in range(u, e))
    return x, consistent


def poly_mul(a: list[int], b: list[int]) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                out[i + j] = (out[i + j] + ai * bj) % P
    return out


def vanishing_poly(points: list[int]) -> list[int]:
    z = [1]
    for x in points:
        z = poly_mul(z, [(-x) % P, 1])
    return z


def lagrange(points: list[int], values: list[int]) -> list[int]:
    """Coefficients (degree < len(points)) of the polynomial through (points[i], values[i])."""
    n = len(points)
    Z = vanishing_poly(points)
    out = [0] * n
    for i, (xi, yi) in enumerate(zip(points, values)):
        # Z / (x - xi) by synthetic division
        q = [0] * n
        carry = 0
        for k in range(n, 0, -1):
            carry = (Z[k] + carry * xi) % P
            q[k - 1] = carry
        denom = 1
        for j, xj in enumerate(points):
            if j != i:
                denom = denom * (xi - xj) % P
        scale = yi * inv(denom) % P
        for k in range(n):
            out[k] = (out[k] + scale * q[k]) % P
    return out


def interp_with_slack(points: list[int], values: list[int], deg_bound: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random polynomial of degree < ``deg_bound`` (>= len(points)) with the prescribed values:
    Lagrange interpolant + Z_points(x) * (uniform of degree < deg_bound - len(points)).  int64 coefficients."""
    t = len(points)
    assert deg_bound >= t, (deg_bound, t)
    base = np.zeros(deg_bound, dtype=np.int64)
    base[:t] = lagrange(points, values)
    slack = deg_bound - t
    if slack:
        Z = np.asarray(vanishing_poly(points), dtype=np.int64)      # degree t, t + 1 coefficients
        r = rng.integers(0, P, size=slack, dtype=np.int64)
        for i in range(t + 1):
            if Z[i]:
                base[i:i + slack] = (base[i:i + slack] + (Z[i] * r) % P) % P
    return base


def poly_eval_many(coefs: np.ndarray, xs: list[int]) -> list[int]:
    out = []
    c = [int(v) for v in coefs]
    for x in xs:
        acc = 0
        for v in reversed(c):
            acc = (acc * x + v) % P
        out.append(acc)
    return out

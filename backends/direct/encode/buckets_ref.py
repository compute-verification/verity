"""numpy / Python-int references for ``buckets.py`` (small shapes; the GPU kernels are checked bit-exactly against these)."""
from __future__ import annotations

import numpy as np

from . import babybear as bb

P = bb.P
D = 6
NR = 31          # F_c = F_p[X] / (X^6 - 31)


def _u64(a):
    return np.asarray(a, dtype=np.uint64)


def coords_times_rows(coef: np.ndarray, M: np.ndarray) -> np.ndarray:
    """``out[c][j] = sum_i coef[i][c] M[i][j] mod p`` for ``coef`` ``(rows, D)`` and ``M`` ``(rows, ncols)`` (rows <= 2^33 / p)."""
    coef, M = _u64(coef), _u64(M)
    out = np.empty((D, M.shape[1]), dtype=np.uint64)
    for c in range(D):
        prod = (coef[:, c][:, None] * M) % np.uint64(P)
        out[c] = prod.sum(axis=0) % np.uint64(P)
    return out


def irs_row_combination(r: np.ndarray, W: np.ndarray) -> np.ndarray:
    return coords_times_rows(r, W)


def encode2(rows: np.ndarray) -> np.ndarray:
    """Cosets 0 and 1 of the ``[n, k]`` code: the 8,192-point domain used by the tests (``deg < 2k - 1``)."""
    return bb.encode_rows(rows, bb.N_CODE, layout="coset")[:, : 2 * bb.K_CODE]


def quadratic_test(r: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    X, Y, Z = encode2(x), encode2(y), encode2(z)
    term = (X * Y + (np.uint64(P) - Z)) % np.uint64(P)
    return coords_times_rows(r, term)


def linear_test(rA: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``rA`` ``(D, rows, k)`` plain coordinates of ``r^T A`` rows; ``u`` ``(rows, k)``: ``q[c][j] = sum_i R_c[i][j] U[i][j]``."""
    U = encode2(u)
    out = np.empty((D, U.shape[1]), dtype=np.uint64)
    for c in range(D):
        Rc = encode2(rA[c])
        out[c] = ((Rc * U) % np.uint64(P)).sum(axis=0) % np.uint64(P)
    return out


# -- F_c arithmetic on Python ints (coordinate lists) ---------------------------------------------------------------
def fc_mul(a: list[int], b: list[int]) -> list[int]:
    t = [0] * (2 * D - 1)
    for i in range(D):
        for j in range(D):
            t[i + j] = (t[i + j] + a[i] * b[j]) % P
    out = t[:D]
    for i in range(D, 2 * D - 1):
        out[i - D] = (out[i - D] + t[i] * NR) % P
    return out


def fc_pow(a: list[int], e: int) -> list[int]:
    acc = [1] + [0] * (D - 1)
    base = list(a)
    while e:
        if e & 1:
            acc = fc_mul(acc, base)
        base = fc_mul(base, base)
        e >>= 1
    return acc


def fc_inv(a: list[int]) -> list[int]:
    return fc_pow(a, P**D - 2)


def batch_inverse(d: np.ndarray) -> np.ndarray:
    """``d`` ``(D, N)`` plain coordinates -> ``(D, N)`` inverses (plain)."""
    d = _u64(d)
    out = np.empty_like(d)
    for j in range(d.shape[1]):
        out[:, j] = fc_inv([int(v) for v in d[:, j]])
    return out


def is_irreducible_binomial() -> bool:
    """``X^6 - 31`` is irreducible over ``F_p`` iff 31 is neither a square nor a cube (``6 | p - 1``, ``4 ∤ 6``)."""
    return pow(NR, (P - 1) // 2, P) != 1 and pow(NR, (P - 1) // 3, P) != 1

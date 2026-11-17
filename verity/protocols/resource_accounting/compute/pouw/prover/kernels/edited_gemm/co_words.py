"""The words CO-Dither's checked GEMMs commit, which their CPU references (`cgemm_ref.py`, `cpv_ref.py`, `apv_ref.py`,
`cwgemm_ref.py`) recompute and `calibration/pouw/theory/checked_strip.py` analyses (its docstring states the
construction, in Lean's coordinates). A tally, where given, is anything with checked_strip's `Tally.int8(A, B, what)`,
which prices each product as computed. Numpy only; its readers load it by file.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

M32 = np.uint32
MASK = (1 << 32) - 1
#: U's plane weights: U = −64 + Σ_p w_p β_p
W_RE = np.array([1, 2, 4, 8, 16, 32, 65], dtype=np.int64)


def s32(x: np.ndarray | int) -> np.ndarray:
    return (np.asarray(x, dtype=np.int64) & MASK).astype(M32)


def n_digits(lo: int, hi: int) -> int:
    """Base-128 digits `digits` needs for values in [lo, hi]: the least m with −128^m ≤ lo and ⌊hi/128^{m−1}⌋ ≤ 127."""
    m = 1
    while not (lo >= -(128 ** m) and hi // 128 ** (m - 1) <= 127):
        m += 1
    return m


def digits(Y: np.ndarray, z: np.ndarray | Sequence[np.ndarray], m: int) -> list[np.ndarray]:
    """Y = Σ_d 128^d A_d with m int8 digits: A_d = (R_d mod 128) − 128 z_d for d < m − 1 and A_{m−1} = R_{m−1}, where
    R_0 = Y and R_{d+1} = ⌊R_d/128⌋ + z_d. Each fresh coin z_d (one an entry) picks one of the digit's two int8
    representatives; the last is taken as 0 where ⌊R/128⌋ = 127, so that the top digit stays int8."""
    R, A = np.asarray(Y, np.int64), []
    for d in range(m - 1):
        q = R >> 7
        zd = np.where(q == 127, 0, z[d]) if d == m - 2 else z[d]
        A.append((R & 127) - 128 * zd)
        R = q + zd
    A.append(R)
    return A


def honest_words(P: np.ndarray, V: np.ndarray, U: np.ndarray, c: np.ndarray, b: np.ndarray, r: int, L: int) -> tuple[list[np.ndarray], np.ndarray]:
    k = P.shape[1]
    Ph = P + c
    Pt = Ph + U[:, np.arange(k) % r]
    X = V + b
    T = [Pt[:, :(t + 1) * L] @ X[:(t + 1) * L] for t in range(k // L)]
    O = Ph @ X
    return [s32(x) for x in T], s32(O)


def co32c_words(P: np.ndarray, V: np.ndarray, U: np.ndarray, c: np.ndarray, b: np.ndarray, z: np.ndarray | Sequence[np.ndarray], r: int, L: int, m: int,
                tally: Any = None) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """CO with committed digits: T_t, D_d = U·A_d (1 ≤ d < m) and O′ = T_last − U·A_0, for Y's digits A_d; the
    served output is O = O′ − Σ_d 128^d D_d. With a tally, every product is priced as computed."""
    k = P.shape[1]
    mm = tally.int8 if tally is not None else (lambda A_, B_, what: A_.astype(np.int64) @ B_.astype(np.int64))
    Pt = P + c + U[:, np.arange(k) % r]
    X = V + b
    T, acc = [], np.zeros((P.shape[0], V.shape[1]), np.int64)
    for t in range(k // L):
        sl = slice(t * L, (t + 1) * L)
        acc = acc + mm(Pt[:, sl], X[sl], f"T segment {t}")
        T.append(acc.copy())
    A = digits(X.reshape(k // r, r, -1).sum(0), z, m)
    D = [mm(U, A[d], f"D_{d}") for d in range(1, m)]
    Op = T[-1] + mm(-U, A[0], "O′")
    return [s32(x) for x in T], [s32(x) for x in D], s32(Op)


def pv_words(P: np.ndarray, V: np.ndarray, U: np.ndarray, c: np.ndarray, b: np.ndarray, z: np.ndarray, r: int, L: int, m: int, M: np.ndarray,
             tally: Any = None) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """P·V's words, batched over a leading axis (or none): T_t (masked P̃X), D_d = U·A_d(Ycum) (1 ≤ d < m) and
    O′ = T_last − U·A_0(Ycum) − W, W = Σ_{diagonal κ ≤ q} U(·, κ mod r) X(κ). With a tally (unbatched), every product
    is priced as computed, masked entries included (an IMMA computes them)."""
    k = P.shape[-1]
    mm = tally.int8 if tally is not None else (lambda A_, B_, what: np.matmul(A_.astype(np.int64), B_.astype(np.int64)))
    Ph = (P + c) * M
    Ue = U[..., np.arange(k) % r]
    Pt = Ph + Ue * M
    X = V + b
    T, acc = [], 0
    for t in range(k // L):
        sl = slice(t * L, (t + 1) * L)
        acc = acc + mm(Pt[..., sl], X[..., sl, :], f"T segment {t}")
        T.append(acc)
    g = (k - L) // r
    Yc = X[..., :k - L, :].reshape(*X.shape[:-2], g, r, X.shape[-1]).sum(-3) if g else 0 * X[..., :r, :]
    A = digits(Yc, z, m)
    D = [mm(U, a, f"D_{d + 1}") for d, a in enumerate(A[1:])]
    sl = slice(k - L, k)
    W = mm(Ue[..., sl] * M[..., sl], X[..., sl, :], "W (diagonal block's noise)")
    Op = T[-1] - mm(U, A[0], "U·A_0") - W
    return T, D, Op


def pv_served(Op: np.ndarray, D: list[np.ndarray], X: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(O, S, 2·served): O = O′ − Σ 128^d D_d = Σ_κ M P̂ X (Lean coordinates, s32), S = Σ_{κ ≤ q} X, and the served
    output doubled, 2(O + 64 S) − S = Σ M (2 P_prob + 2c − 1) X, exact (the bench's (P_prob + c − ½)·X)."""
    O = Op.astype(np.int64) - sum(d.astype(np.int64) << (7 * (j + 1)) for j, d in enumerate(D))
    S = M.astype(np.int64) @ X.astype(np.int64)
    return O, S, 2 * (O + 64 * S) - S

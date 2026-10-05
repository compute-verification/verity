"""numpy reference of the two-stage matrix-DFT encoder with INT8 limbs (the algorithm the tensor-core kernels run).

A size-``k`` DFT with ``k = N1 N2`` (Cooley-Tukey four-step, ``n = n1 + N1 n2``, ``kk = N2 k1 + k2``):

    A[n1, k2] = sum_n2 d[n1 + N1 n2] w_N2^(n2 k2)          (stage 1: N2-point DFTs, a dense (.. x N2) x (N2 x N2) GEMM)
    B[n1, k2] = A[n1, k2] w_k^(n1 k2)                       (twiddle, SIMT)
    X[N2 k1 + k2] = sum_n1 w_N1^(n1 k1) B[n1, k2]           (stage 2: N1-point DFTs, a GEMM after a transpose)

On tensor cores every GEMM operand is split into four balanced signed 8-bit limbs (``x = sum_i s_i 2^(8i)``,
``s_i in [-128, 127]``; ``x < p < 2^31 - 2^23`` keeps the top limb below 128) and the ``4 x 4`` limb products are
INT8 x INT8 -> INT32 dots (``|sum| <= 64 x 128^2 = 2^20`` at ``K = 64``: no chunking needed); the result is recombined
as ``sum_(i,j) Y_ij (2^(8(i+j)) mod p)`` in INT64 (``< 7 x 2^20 x 2^31 < 2^55``) and reduced once.  The encoder is the
coset formulation of ``encode_ntt_simt``: one inverse DFT (``w_k^-1``, scale ``k^-1`` folded into the coset shift
tables) and three forward DFTs on the shifted coefficients; all four are 64 x 64 two-stage DFTs at ``k = 4096``.

Base MACs per row: ``4 DFTs x 2 stages x k x sqrt(k)`` = 2,097,152 at ``k = 4096`` (the cost model's zero-padded
``n = 128 x 128`` evaluation formulation counts 4,718,592); x16 limb products = 33.6 M INT8 MACs per row.
"""
from __future__ import annotations

import math

import numpy as np

from . import babybear as bb

LIMBS = 4
LIMB_WEIGHTS_MOD_P = np.array([pow(2, 8 * s, bb.P) for s in range(2 * LIMBS - 1)], dtype=np.int64)


def balanced_limbs(x: np.ndarray) -> np.ndarray:
    """``(..., 4)`` int8 balanced limbs of non-negative int64 ``x < 2^31 - 2^23``."""
    x = np.asarray(x, dtype=np.int64)
    out = np.empty((*x.shape, LIMBS), dtype=np.int8)
    cur = x.copy()
    for i in range(LIMBS):
        s = ((cur + 128) & 255) - 128
        out[..., i] = s
        cur = (cur - s) >> 8
    if np.any(cur):
        raise ValueError("value does not fit in four balanced limbs")
    return out


def from_limbs(limbs: np.ndarray) -> np.ndarray:
    limbs = np.asarray(limbs, dtype=np.int64)
    return sum(limbs[..., i] << (8 * i) for i in range(limbs.shape[-1]))


def dft_matrix(N: int, w: int) -> np.ndarray:
    """``F[a, b] = w^(a b) mod p`` as int64 (``w`` a primitive ``N``-th root)."""
    e = (np.arange(N)[:, None] * np.arange(N)[None, :]) % N
    return bb.powers(w, N).astype(np.int64)[e]


def limb_gemm(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """``X @ F mod p`` (``X`` (M, K), ``F`` (K, N), both reduced mod p) computed exactly as the tensor cores do it:
    16 INT8 x INT8 -> INT32 GEMMs, INT64 recombination with ``2^(8(i+j)) mod p``, one reduction."""
    xl = balanced_limbs(X).astype(np.int32)     # (M, K, 4)
    fl = balanced_limbs(F).astype(np.int32)     # (K, N, 4)
    acc = np.zeros(X.shape[:-1] + F.shape[1:], dtype=np.int64)
    for i in range(LIMBS):
        for j in range(LIMBS):
            y = xl[..., i] @ fl[..., j]                       # int32 GEMM, |y| < 2^20 for K <= 64
            assert np.abs(y).max() < 2**31 - 1
            acc += y.astype(np.int64) * LIMB_WEIGHTS_MOD_P[i + j]
    return np.mod(acc, bb.P)


def dft_two_stage(d: np.ndarray, w: int, N1: int, N2: int) -> np.ndarray:
    """Row-wise size-``N1 N2`` DFT of ``d`` (R, k) with root ``w`` by the four-step algorithm on limb GEMMs; natural
    output order."""
    R, k = d.shape
    assert k == N1 * N2
    F2 = dft_matrix(N2, pow(w, N1, bb.P))
    F1 = dft_matrix(N1, pow(w, N2, bb.P))
    T = bb.powers(w, k).astype(np.int64)[(np.arange(N1)[:, None] * np.arange(N2)[None, :]) % k]      # w_k^(n1 k2)
    D = d.reshape(R, N2, N1).transpose(0, 2, 1).reshape(R * N1, N2)                                # [(r, n1), n2]
    A = limb_gemm(D, F2).reshape(R, N1, N2)                                                        # [r, n1, k2]
    B = (A * T[None]) % bb.P
    Bt = B.transpose(0, 2, 1).reshape(R * N2, N1)                                                  # [(r, k2), n1]
    Xt = limb_gemm(Bt, F1).reshape(R, N2, N1)                                                      # [r, k2, k1]
    return Xt.transpose(0, 2, 1).reshape(R, k)                                                     # N2 k1 + k2


def encode_rows_matdft(rows: np.ndarray, n: int) -> np.ndarray:
    """Coset-layout RS encoding of ``rows`` (R, k) by two-stage limb-GEMM DFTs (module docstring)."""
    rows = np.asarray(rows, dtype=np.int64) % bb.P
    R, k = rows.shape
    N1 = 1 << (int(math.log2(k)) // 2)
    N2 = k // N1
    wn = bb.root_of_unity(n)
    wk = pow(wn, n // k, bb.P)
    coef = dft_two_stage(rows, pow(wk, -1, bb.P), N1, N2)          # unscaled coefficients, natural order
    kinv = pow(k, -1, bb.P)
    out = np.empty((R, n), dtype=np.uint64)
    out[:, :k] = rows
    for c in range(1, n // k):
        shift = (bb.powers(pow(wn, c, bb.P), k).astype(np.int64) * kinv) % bb.P
        y = (coef * shift[None]) % bb.P
        out[:, c * k:(c + 1) * k] = dft_two_stage(y, wk, N1, N2)
    return out

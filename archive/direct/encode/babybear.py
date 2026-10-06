"""BabyBear arithmetic and the numpy reference Reed-Solomon encoder (b-encode lane).

The code is the protocols' Ligero code (``archive/direct/PROTOCOL.md`` section 3): ``[n = 2^14, k = 2^12]`` RS over
``F_t = BabyBear`` (``p = 2^31 - 2^27 + 1``).  A row holds ``k`` values (``l = 3840`` payload symbols + ``k - l``
randomizers, all *values*), so the encoder interpolates the unique degree ``< k`` polynomial through the ``k`` values
on the message domain and evaluates it on the ``n``-point domain: the INTT is part of the protocol, not an option
(the constraints refer to the message symbols as values, and the quadratic test multiplies rows pointwise).

Domains.  ``omega`` is a primitive ``n``-th root of unity; the evaluation domain is ``D = {omega^t : t < n}`` and the
message domain is its index-``rate^-1`` subgroup ``H = {omega^(4 j)} = {omega_k^j}``.  Message symbol ``j`` is the value
at ``omega_k^j``, i.e. at ``t = 4 j``: the code is *systematic* -- coset 0 of the stride-4 decomposition of ``D`` is the
message itself.  Two output layouts are used:

* ``natural``: codeword position ``t`` holds ``f(omega^t)``;
* ``coset``: position ``c * k + j`` holds ``f(omega^c omega_k^j) = f(omega^(4 j + c))``, ``c < 4`` -- the layout the
  GPU kernels write (coset 0 is a copy of the input row).  ``coset_to_natural`` is the public permutation between
  them; Ligero's column sampling is uniform over ``n`` positions, so either labelling of the columns is a valid
  evaluation domain and the verifier evaluates its Vandermonde / interpolation constants for whichever is used.

Everything here is numpy int64/uint64 (products of two 31-bit residues fit in 62 bits) and is the *reference*: the
GPU encoders are checked bit-exactly against ``encode_rows`` on small sizes.
"""
from __future__ import annotations

import numpy as np

P = 2013265921                  # 2^31 - 2^27 + 1
GENERATOR = 31                  # multiplicative generator of F_p^* (Plonky3's choice)
TWO_ADICITY = 27
N_CODE = 16384
K_CODE = 4096
RATE_INV = N_CODE // K_CODE     # 4 cosets

# Montgomery constants for R = 2^32 (used by the CUDA kernels; checked in the tests)
R_MOD_P = pow(2, 32, P)
R2_MOD_P = pow(2, 64, P)
P_INV_MOD_R = pow(P, -1, 2**32)          # p^{-1} mod 2^32
NEG_P_INV_MOD_R = (-P_INV_MOD_R) % 2**32  # -p^{-1} mod 2^32


def root_of_unity(order: int) -> int:
    """A primitive ``order``-th root of unity (``order`` a power of two up to ``2^27``)."""
    if order <= 0 or (P - 1) % order:
        raise ValueError(f"no root of unity of order {order} in F_p")
    return pow(GENERATOR, (P - 1) // order, P)


def to_mont(x: int) -> int:
    return (x * R_MOD_P) % P


def powers(w: int, count: int) -> np.ndarray:
    """``[w^0, ..., w^(count-1)] mod p`` as uint64 (repeated squaring/doubling in numpy)."""
    out = np.empty(count, dtype=np.uint64)
    out[0] = 1
    step = 1
    cur = w % P
    while step < count:
        m = min(step, count - step)
        out[step:step + m] = (out[:m] * np.uint64(cur)) % np.uint64(P)
        step *= 2
        cur = (cur * cur) % P
    return out


def bit_reverse_indices(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    idx = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def ntt_rows(x: np.ndarray, w: int) -> np.ndarray:
    """Row-wise DFT ``X[.., k] = sum_j x[.., j] w^(jk)`` over the last axis (length a power of two) with the given
    primitive root; iterative radix-2 in natural order (bit-reverse then Cooley-Tukey)."""
    x = np.asarray(x, dtype=np.uint64) % np.uint64(P)
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError("length must be a power of two")
    a = x[..., bit_reverse_indices(n)].copy()
    m = 1
    while m < n:
        wm = pow(w, n // (2 * m), P)
        tw = powers(wm, m)
        a = a.reshape(*a.shape[:-1], n // (2 * m), 2, m)
        lo = a[..., 0, :]
        hi = (a[..., 1, :] * tw) % np.uint64(P)
        a = np.stack([(lo + hi) % np.uint64(P), (lo + np.uint64(P) - hi) % np.uint64(P)], axis=-2)
        a = a.reshape(*a.shape[:-3], n)
        m *= 2
    return a


def intt_rows(x: np.ndarray, w: int) -> np.ndarray:
    n = np.asarray(x).shape[-1]
    inv_n = pow(n, -1, P)
    return (ntt_rows(x, pow(w, -1, P)) * np.uint64(inv_n)) % np.uint64(P)


def encode_rows(rows: np.ndarray, n: int = N_CODE, layout: str = "coset") -> np.ndarray:
    """Reference RS encoder: ``rows`` is ``(R, k)`` of values on the message domain ``H``; returns ``(R, n)`` codewords
    (uint64) in the requested layout (module docstring)."""
    rows = np.asarray(rows, dtype=np.uint64)
    k = rows.shape[-1]
    if n % k or (n // k) & (n // k - 1):
        raise ValueError("n / k must be a power of two")
    ratio = n // k
    wn = root_of_unity(n)
    wk = pow(wn, ratio, P)
    coeffs = intt_rows(rows, wk)                                   # (R, k) coefficients c_i
    idx = np.arange(k, dtype=np.uint64)
    cosets = []
    for c in range(ratio):
        shift = powers(pow(wn, c, P), k)                           # omega^(c i)
        cosets.append(ntt_rows((coeffs * shift[idx]) % np.uint64(P), wk))
    out = np.stack(cosets, axis=-2).reshape(*rows.shape[:-1], n)  # coset-major: [c][j]
    if layout == "coset":
        return out
    if layout == "natural":
        return out[..., coset_to_natural(n, k)]
    raise ValueError(layout)


def coset_to_natural(n: int = N_CODE, k: int = K_CODE) -> np.ndarray:
    """Index array ``perm`` with ``natural[t] = coset[perm[t]]``: ``t = ratio * j + c  ->  c * k + j``."""
    ratio = n // k
    t = np.arange(n)
    return (t % ratio) * k + t // ratio


def encode_rows_naive(rows: np.ndarray, n: int) -> np.ndarray:
    """O(n k) Lagrange-free check for tiny sizes: coefficients by inverse Vandermonde solve over the integers mod p
    (Gaussian elimination in Python ints), then direct evaluation.  Natural layout."""
    rows = np.asarray(rows, dtype=object)
    k = rows.shape[-1]
    wn = root_of_unity(n)
    wk = pow(wn, n // k, P)
    V = [[pow(wk, i * j, P) for j in range(k)] for i in range(k)]
    Vinv = _inverse_mod_p(V)
    out = np.empty((*rows.shape[:-1], n), dtype=np.uint64)
    for r in np.ndindex(*rows.shape[:-1]):
        vals = [int(v) for v in rows[r]]
        coeffs = [sum(Vinv[i][j] * vals[j] for j in range(k)) % P for i in range(k)]
        for t in range(n):
            x = pow(wn, t, P)
            acc = 0
            for i in reversed(range(k)):
                acc = (acc * x + coeffs[i]) % P
            out[r + (t,)] = acc
    return out


def _inverse_mod_p(M: list[list[int]]) -> list[list[int]]:
    n = len(M)
    A = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        piv = next(r for r in range(col, n) if A[r][col] % P)
        A[col], A[piv] = A[piv], A[col]
        inv = pow(A[col][col], -1, P)
        A[col] = [(v * inv) % P for v in A[col]]
        for r in range(n):
            if r != col and A[r][col]:
                f = A[r][col]
                A[r] = [(a - f * b) % P for a, b in zip(A[r], A[col])]
    return [row[n:] for row in A]


def fill_rows(R: int, k: int = K_CODE, seed: int = 0) -> np.ndarray:
    """Deterministic pseudo-random rows (values < p), uint32."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, P, size=(R, k), dtype=np.int64).astype(np.uint32)

"""Functions on the bit cube {0,1}^n, given as arrays indexed by the point sum_i x_i 2^i: their multilinear (Moebius)
coefficients, exactly over Z or in Z/2^bits, and the degree-2 part as an F_2 matrix.

The coefficient at index m is the coefficient of the monomial prod_{i in m} x_i, so f = sum_m c_m prod_{i in m} x_i on
every point.
"""
from __future__ import annotations

import numpy as np


def points(n: int) -> np.ndarray:
    """Every point's index, 0 .. 2^n - 1."""
    return np.arange(1 << n, dtype=np.uint64)


def bit(idx: np.ndarray, i: int) -> np.ndarray:
    """Coordinate x_i of each point."""
    return (np.asarray(idx, dtype=np.uint64) >> np.uint64(i)) & np.uint64(1)


def popcount(x) -> np.ndarray:
    """Set bits of each entry of an unsigned (or nonnegative) integer array."""
    return np.bitwise_count(np.asarray(x, dtype=np.uint64)).astype(np.int64)


def degree(n: int) -> np.ndarray:
    """The degree of each monomial index of the n-cube."""
    return popcount(np.arange(1 << n, dtype=np.uint64))


def mobius(values, n: int, bits: int | None = None) -> np.ndarray:
    """Multilinear coefficients of the function with these 2^n values.

    bits=None: exact over Z in int64 (the caller keeps 2^n times the largest |value| below 2^63). bits=k (1..64): in
    Z/2^k, returned unsigned; uint32 arithmetic for k <= 32, uint64 above."""
    if bits is None:
        a = np.array(values, dtype=np.int64)
    elif 1 <= bits <= 64:
        a = np.array(values).astype(np.uint32 if bits <= 32 else np.uint64)
    else:
        raise ValueError(f"bits = {bits} is not in 1..64")
    if a.shape != (1 << n,):
        raise ValueError(f"{a.shape} values for a {n}-cube")
    for i in range(n):
        a = a.reshape(-1, 2, 1 << i)
        a[:, 1, :] -= a[:, 0, :]
        a = a.reshape(-1)
    if bits is not None and bits not in (32, 64):
        a &= a.dtype.type((1 << bits) - 1)
    return a


def evaluate(coeff, n: int, bits: int | None = None) -> np.ndarray:
    """The inverse of `mobius`: the function's values from its coefficients."""
    a = np.array(coeff, dtype=np.int64) if bits is None else np.array(coeff).astype(np.uint32 if bits <= 32 else np.uint64)
    for i in range(n):
        a = a.reshape(-1, 2, 1 << i)
        a[:, 1, :] += a[:, 0, :]
        a = a.reshape(-1)
    if bits is not None and bits not in (32, 64):
        a &= a.dtype.type((1 << bits) - 1)
    return a


def quadratic_rows(coeff, n: int, plane: int = 0) -> list[int]:
    """The degree-2 part's bit `plane` as an alternating F_2 matrix (n row bitmasks): entry (u, v), u != v, is bit
    `plane` of the coefficient of x_u x_v."""
    c = np.asarray(coeff)
    rows = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if (int(c[(1 << u) | (1 << v)]) >> plane) & 1:
                rows[u] |= 1 << v
                rows[v] |= 1 << u
    return rows

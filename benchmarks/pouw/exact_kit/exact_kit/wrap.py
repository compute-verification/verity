"""Wraparound arithmetic on k-bit words, for Python ints and numpy integer arrays alike.

`u(x, k)` is x mod 2^k in [0, 2^k), `s(x, k)` the two's-complement reading in [-2^(k-1), 2^(k-1)); `u32`/`s32` fix k = 32.
Arrays come back as int64 (k <= 32: products of two words stay exact before reduction) or uint64 (u with k = 64).
"""
from __future__ import annotations

import numpy as np


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def u(x, k: int):
    if _is_int(x):
        return int(x) % (1 << k)
    if k == 64:
        return np.asarray(x).astype(np.uint64)
    return np.asarray(x, dtype=np.int64) % (1 << k) if k < 63 else np.asarray(x).astype(np.uint64) & np.uint64((1 << k) - 1)


def s(x, k: int):
    h = 1 << (k - 1)
    if _is_int(x):
        return (int(x) + h) % (1 << k) - h
    if k > 32:
        raise ValueError("signed arrays above 32 bits are not exact in int64 arithmetic here; use Python ints")
    return (np.asarray(x, dtype=np.int64) + h) % (1 << k) - h


def u32(x):
    return u(x, 32)


def s32(x):
    return s(x, 32)


def rotl(x, r: int, k: int = 32):
    """Rotate k-bit words left by r (taken mod k)."""
    r %= k
    m = (1 << k) - 1
    if _is_int(x):
        x = int(x) & m
        return ((x << r) | (x >> (k - r))) & m if r else x
    x = np.asarray(x).astype(np.uint64) & np.uint64(m)
    if not r:
        return x
    return ((x << np.uint64(r)) | (x >> np.uint64(k - r))) & np.uint64(m)


def rotr(x, r: int, k: int = 32):
    return rotl(x, (k - r % k) % k, k)


def v2(x: int) -> int:
    """2-adic valuation of a nonzero int."""
    x = int(x)
    if not x:
        raise ValueError("v2(0) is infinite")
    return (x & -x).bit_length() - 1

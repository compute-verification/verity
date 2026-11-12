"""Rotations of k-bit words, for Python ints and numpy integer arrays alike. An int comes back as an int in [0, 2^k), an
array as uint64."""
from __future__ import annotations

import numpy as np


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


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

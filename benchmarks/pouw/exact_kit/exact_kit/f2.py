"""Linear algebra over F_2 on rows given as int bitmasks: bit j of a row is its column j.

`rank`, and `from_matrix` for the rows of a 0/1 matrix. Rows are Python ints, so any width works.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

#: what `theory/checked_strip.py` and the tests read
__all__ = ["from_matrix", "rank"]


def rank(rows: Iterable[int]) -> int:
    """Rank over F_2 of int bitmask rows."""
    rows = [int(r) for r in rows if r]
    r = 0
    while rows:
        piv = rows.pop()
        r += 1
        low = piv & -piv
        rows = [x ^ piv if x & low else x for x in rows]
        rows = [x for x in rows if x]
    return r


def from_matrix(M) -> list[int]:
    """A 0/1 (or integer, read mod 2) matrix's rows as bitmasks."""
    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError("a matrix has two axes")
    bits = (M.astype(np.int64) % 2).astype(np.uint64)
    if M.shape[1] <= 64:
        return (bits << np.arange(M.shape[1], dtype=np.uint64)).sum(axis=1, dtype=np.uint64).tolist()
    return [sum(1 << j for j in np.flatnonzero(row).tolist()) for row in bits]

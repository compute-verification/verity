"""Linear algebra over F_2 on rows given as int bitmasks: bit j of a row is its column j.

Rank, reduced row echelon form, nullspace, span enumeration and membership, and the alternating (symmetric, zero-diagonal)
matrices that bilinear forms over F_2 are written as. Rows are Python ints, so any width works.
"""
from __future__ import annotations

import functools
import operator
from collections.abc import Iterable, Iterator

import numpy as np


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


def rref(rows: Iterable[int]) -> tuple[list[int], list[int]]:
    """Reduced row echelon form: (basis, pivots), the basis sorted by pivot (each row's lowest set bit), every other
    basis row zero in each pivot column. The basis is unique for the span."""
    basis: dict[int, int] = {}
    for r in rows:
        r = int(r)
        for p, b in basis.items():
            if r >> p & 1:
                r ^= b
        if not r:
            continue
        p = (r & -r).bit_length() - 1
        for q in basis:
            if basis[q] >> p & 1:
                basis[q] ^= r
        basis[p] = r
    pivots = sorted(basis)
    return [basis[p] for p in pivots], pivots


def reduce(v: int, rows: Iterable[int]) -> int:
    """v reduced against the span of rows: 0 iff v is in it."""
    basis, pivots = rref(rows)
    v = int(v)
    for p, b in zip(pivots, basis):
        if v >> p & 1:
            v ^= b
    return v


def in_span(v: int, rows: Iterable[int]) -> bool:
    return reduce(v, rows) == 0


def nullspace(rows: Iterable[int], n: int) -> list[int]:
    """Basis of {x in F_2^n : <row, x> = 0 for every row}, one vector per free column, as bitmasks."""
    basis, pivots = rref(rows)
    if basis and max(basis).bit_length() > n:
        raise ValueError(f"a row has a bit at column >= n = {n}")
    free = [j for j in range(n) if j not in set(pivots)]
    out = []
    for f in free:
        x = 1 << f
        for p, b in zip(pivots, basis):
            if b >> f & 1:
                x |= 1 << p
        out.append(x)
    return out


def span(rows: Iterable[int]) -> Iterator[int]:
    """Every element of the span of rows, 0 first, once each (Gray-code order over an echelon basis)."""
    basis, _ = rref(rows)
    v = 0
    yield v
    for i in range(1, 1 << len(basis)):
        v ^= basis[(i & -i).bit_length() - 1]
        yield v


def kernel(rows: Iterable[int], n: int) -> Iterator[int]:
    """Every x in F_2^n with <row, x> = 0 for every row."""
    return span(nullspace(rows, n))


def dot(a: int, b: int) -> int:
    return (int(a) & int(b)).bit_count() & 1


def from_matrix(M) -> list[int]:
    """A 0/1 (or integer, read mod 2) matrix's rows as bitmasks."""
    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError("a matrix has two axes")
    bits = (M.astype(np.int64) % 2).astype(np.uint64)
    if M.shape[1] <= 64:
        return (bits << np.arange(M.shape[1], dtype=np.uint64)).sum(axis=1, dtype=np.uint64).tolist()
    return [sum(1 << j for j in np.flatnonzero(row).tolist()) for row in bits]


def to_matrix(rows: Iterable[int], n: int) -> np.ndarray:
    rows = [int(r) for r in rows]
    return np.array([[(r >> j) & 1 for j in range(n)] for r in rows], dtype=np.uint8).reshape(len(rows), n)


def transpose(rows: Iterable[int], n: int) -> list[int]:
    """The n columns of the rows, as bitmasks over the row index."""
    rows = [int(r) for r in rows]
    return [sum(((r >> j) & 1) << i for i, r in enumerate(rows)) for j in range(n)]


def alternating(n: int, pairs: Iterable[tuple[int, int]]) -> list[int]:
    """The alternating matrix (as n row bitmasks) of the bilinear form sum over pairs (u, v) of x_u y_v + x_v y_u; a pair
    listed twice cancels."""
    rows = [0] * n
    for u, v in pairs:
        if u == v:
            raise ValueError(f"an alternating form has no diagonal pair ({u}, {v})")
        rows[u] ^= 1 << v
        rows[v] ^= 1 << u
    return rows


def wedge(n: int, a: int, b: int) -> list[int]:
    """The alternating matrix of a ^ b, the form of the product of linear forms a and b without its diagonal: row u is
    a_u b + b_u a."""
    return [((b if a >> u & 1 else 0) ^ (a if b >> u & 1 else 0)) for u in range(n)]


def xor_rows(*forms: list[int]) -> list[int]:
    """Row-wise sum of matrices given as row lists of one length."""
    return [functools.reduce(operator.xor, col) for col in zip(*forms, strict=True)]

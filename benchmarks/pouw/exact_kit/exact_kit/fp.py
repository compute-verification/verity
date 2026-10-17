"""Linear algebra over F_p (p prime) on rows given as lists of ints: rank, reduced row echelon form, nullspace, span
enumeration, and the projective normal form of a vector."""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence


def _check_p(p: int) -> None:
    if p < 2 or any(p % q == 0 for q in range(2, int(p ** 0.5) + 1)):
        raise ValueError(f"p = {p} is not prime")


def rref(rows: Iterable[Sequence[int]], p: int) -> tuple[list[list[int]], list[int]]:
    """Reduced row echelon form over F_p: (basis, pivots), sorted by pivot, each pivot entry 1 and every other basis row
    0 in each pivot column. Unique for the span."""
    _check_p(p)
    basis: dict[int, list[int]] = {}
    width = None
    for r in rows:
        r = [x % p for x in r]
        if width is None:
            width = len(r)
        elif len(r) != width:
            raise ValueError("rows of different lengths")
        for q, b in basis.items():
            if r[q]:
                f = r[q]
                r = [(x - f * y) % p for x, y in zip(r, b)]
        piv = next((i for i, x in enumerate(r) if x), None)
        if piv is None:
            continue
        inv = pow(r[piv], -1, p)
        r = [x * inv % p for x in r]
        for q in basis:
            if basis[q][piv]:
                f = basis[q][piv]
                basis[q] = [(x - f * y) % p for x, y in zip(basis[q], r)]
        basis[piv] = r
    pivots = sorted(basis)
    return [basis[q] for q in pivots], pivots


def rank(rows: Iterable[Sequence[int]], p: int) -> int:
    return len(rref(rows, p)[0])


def nullspace(rows: Iterable[Sequence[int]], p: int, n: int) -> list[list[int]]:
    """Basis of {x in F_p^n : <row, x> = 0 for every row}, one vector per free column."""
    basis, pivots = rref(rows, p)
    if basis and len(basis[0]) != n:
        raise ValueError(f"rows have length {len(basis[0])}, not n = {n}")
    free = [j for j in range(n) if j not in set(pivots)]
    out = []
    for f in free:
        x = [0] * n
        x[f] = 1
        for q, b in zip(pivots, basis):
            x[q] = (-b[f]) % p
        out.append(x)
    return out


def span(rows: Iterable[Sequence[int]], p: int) -> Iterator[tuple[int, ...]]:
    """Every element of the span, once each, 0 first."""
    basis, _ = rref(rows, p)
    if not basis:
        yield ()
        return
    n = len(basis[0])
    for coef in itertools.product(range(p), repeat=len(basis)):
        yield tuple(sum(c * b[i] for c, b in zip(coef, basis)) % p for i in range(n))


def kernel(rows: Iterable[Sequence[int]], p: int, n: int) -> Iterator[tuple[int, ...]]:
    ns = nullspace(rows, p, n)
    if not ns:
        yield (0,) * n
        return
    yield from span(ns, p)


def normal(v: Sequence[int], p: int) -> tuple[int, ...]:
    """The projective representative of a nonzero v: scaled so that its first nonzero entry is 1."""
    piv = next((x % p for x in v if x % p), None)
    if piv is None:
        raise ValueError("the zero vector has no projective normal form")
    inv = pow(piv, -1, p)
    return tuple(x * inv % p for x in v)


def lines(rows: Iterable[Sequence[int]], p: int) -> list[tuple[int, ...]]:
    """The points of the projective space of the span: each nonzero vector's normal form, once, sorted."""
    return sorted({normal(v, p) for v in span(rows, p) if any(v)})

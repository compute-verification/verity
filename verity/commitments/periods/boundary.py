"""An output field's leaves, each written once by dense reads: the Python twin of `Spot.BoundaryWritten`, round 1's
``link-written``, for a writer to hold its boundary reads to before it registers them (`interface.Circuit.boundary_reads`).

A read covers reader index ``i < prod(radix)``, whose source is leaf ``base + sum(d[k] * step[k])`` (``d`` the digits of
``i``, the first the slowest), the row there holding ``row`` leaves from its source on. A read is dense when its digits of
radix above 1 have steps that, sorted, are ``row``, then ``row * r0``, then ``row * r0 * r1`` and so on, each the one
before times the radix of the digit before: its rows then lay ``[base, base + row * prod(radix))`` end to end, each leaf
once (`extent`). A field's dense reads, sorted by base, tile ``[0, leaves)`` (`written`). The work is per read, never per
reader index or leaf; the Lean is the definition, and its difftest holds it to a reference that lists every leaf."""
from __future__ import annotations

from collections.abc import Sequence

#: a read: its first leaf, its radix and step (the first digit the slowest) and the leaves a row holds
Read = tuple[int, Sequence[int], Sequence[int], int]


def extent(base: int, radix: Sequence[int], step: Sequence[int], row: int) -> tuple[int, int] | None:
    """The leaves ``(lo, hi)`` a dense read lays end to end, or None when it is no dense read: a row of no leaves, a digit
    of radix 0, a fan-out (a step of 0), a negative step, or rows with a gap between them (`Spot.BoundaryWritten.extent?`)."""
    if row == 0 or 0 in radix:
        return None
    run = row
    for r, s in sorted(((r, s) for r, s in zip(radix, step) if r > 1), key=lambda d: d[1]):
        if s != run:
            return None
        run *= r
    return base, base + run


def written(leaves: int, reads: Sequence[Read]) -> bool:
    """Whether every read is dense and together they write each leaf of ``[0, leaves)`` once
    (`Spot.BoundaryWritten.written`)."""
    at = 0
    extents = [extent(*r) for r in reads]
    if None in extents:
        return False
    for lo, hi in sorted(extents):
        if lo != at:
            return False
        at = hi
    return at == leaves

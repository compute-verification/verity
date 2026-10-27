"""Half-open integer interval sets, and the leaf intervals a reference sequence names.

An `Intervals` value is a tuple of `(lo, hi)` pairs, sorted, disjoint and non-empty; `norm` makes one from any pairs.
The Program analyses (`liveness`, `boundary`, `query_ast`) keep sets of leaves, members and gates in this form, so a
weight table, a KV cache or a `batch` output stays O(#runs) instead of O(#leaves).
"""
from __future__ import annotations

import bisect
from typing import Callable, Iterable, Iterator

from .refs import Affine, Concat, Explicit, Refs, Strided

Intervals = tuple[tuple[int, int], ...]  # sorted, disjoint, non-empty half-open [lo, hi)

_ENUM_LIMIT = 1 << 20  # a Strided with no interval structure is enumerated up to this many leaves


class IntervalLimit(Exception):
    """A reference sequence has no compact interval form and is too large to enumerate."""


def norm(iv: Iterable[tuple[int, int]]) -> Intervals:
    """The pairs as `Intervals`: empty pairs dropped, overlapping and adjacent pairs merged."""
    iv = sorted(x for x in iv if x[1] > x[0])
    out: list[tuple[int, int]] = []
    for lo, hi in iv:
        if out and lo <= out[-1][1]:
            if hi > out[-1][1]:
                out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return tuple(out)


def full(n: int) -> Intervals:
    return ((0, n),) if n > 0 else ()


def total(iv: Intervals) -> int:
    return sum(hi - lo for lo, hi in iv)


def contains(iv: Intervals, x: int) -> bool:
    k = bisect.bisect_right(iv, (x, 1 << 62)) - 1
    return k >= 0 and iv[k][0] <= x < iv[k][1]


def isect(a: Intervals, b: Intervals) -> Intervals:
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tuple(out)


def restrict(iv: Intervals, lo: int, hi: int) -> Intervals:
    """`iv ∩ [lo, hi)` shifted by `-lo`."""
    out = []
    for a, b in iv:
        if b <= lo:
            continue
        if a >= hi:
            break
        out.append((max(a, lo) - lo, min(b, hi) - lo))
    return tuple(out)


def complement(iv: Intervals, n: int) -> Iterator[tuple[int, int]]:
    """The runs of `[0, n)` outside `iv`, ascending; `iv` must lie within `[0, n)`."""
    pos = 0
    for lo, hi in iv:
        if lo > pos:
            yield (pos, lo)
        pos = hi
    if pos < n:
        yield (pos, n)


def strided_intervals(r: Strided, add: Callable[[int, int], None]) -> None:
    """Call `add(lo, hi)` for leaf intervals whose union is exactly the leaves `r` references (repeats dropped).
    Raises `IntervalLimit` for a sequence with no interval structure and more than `_ENUM_LIMIT` leaves."""
    n, inner = r.count, r.inner
    if n == 0:
        return
    os_, is_, base = r.outer_stride, r.inner_stride, r.base
    outer = (n + inner - 1) // inner
    last = n - (outer - 1) * inner  # leaves in the last (possibly partial) outer run
    if os_ == 0 and is_ == 0:
        add(base, base + 1)
    elif is_ == 1:
        if os_ == inner and last == inner:
            add(base, base + n)  # a plain affine run written as a strided one
        elif os_ == 0:
            add(base, base + inner)  # every outer run is the same run: a broadcast
        else:
            for q in range(outer):
                add(base + q * os_, base + q * os_ + (inner if q < outer - 1 else last))
    elif os_ == 1 and is_ >= outer:
        if is_ == outer and last == inner:
            add(base, base + n)  # a transpose of a full block covers exactly the block
        else:
            for rr in range(inner):
                rows = outer if rr < last else outer - 1
                if rows:
                    add(base + rr * is_, base + rr * is_ + rows)
    elif is_ == 0:
        for q in range(outer):
            add(base + q * os_, base + q * os_ + 1)
    elif n <= _ENUM_LIMIT:
        for j in range(n):
            leaf = r[j][2]
            add(leaf, leaf + 1)
    else:
        raise IntervalLimit(f"{r!r} has no interval structure and {n} leaves")


def ref_intervals(r: Refs, out: dict[tuple[str, int], list], seen: set | None = None) -> None:
    """Accumulate the leaf intervals referenced by `r` into `out`, grouped by (space, idx) (unnormalised).

    With `seen`, a `Strided` already in it adds nothing (the caller has its intervals already) and one not in it is
    added to it: a table column every token reads (`[V][k]` read with stride k, no interval structure) is then
    enumerated once, not once per read."""
    if isinstance(r, Affine):
        if r.count:
            out.setdefault((r.space, r.idx), []).append((r.base, r.base + r.count))
    elif isinstance(r, Concat):
        for p in r.parts:
            ref_intervals(p, out, seen)
    elif isinstance(r, Explicit):
        for s, i, leaf in r.items:
            out.setdefault((s, i), []).append((leaf, leaf + 1))
    elif isinstance(r, Strided):
        if seen is not None:
            key = (r.space, r.idx, r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
            if key in seen:
                return
            seen.add(key)
        lst = out.setdefault((r.space, r.idx), [])
        strided_intervals(r, lambda lo, hi: lst.append((lo, hi)))
    else:  # pragma: no cover - unknown Refs subclass
        for s, i, leaf in r:
            out.setdefault((s, i), []).append((leaf, leaf + 1))


__all__ = ["Intervals", "IntervalLimit", "norm", "full", "total", "contains", "isect", "restrict", "complement",
           "strided_intervals", "ref_intervals"]

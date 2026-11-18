"""Value references and collections of references.

A *body-scoped reference* inside one composite body is a tuple `(space, idx, leaf)`:

    ("p", i, j)  leaf j of parameter i of the enclosing function
    ("n", k, j)  leaf j of the output collection of body node k

Leaf indices are flat positions in the (row-major) leaf order of the collection's type. A collection
never materialises anything but references; `Affine` sequences keep even multi-hundred-million-leaf
weight arrays O(1) in the descriptor.
"""

from __future__ import annotations

import bisect
import math
from itertools import chain, islice
from typing import Iterator

from .types import Array, RecordType, Tuple, Type, Value, remove_axis

Ref = tuple[str, int, int]


class Refs:
    """An ordered sequence of body-scoped references. Order and repeats are kept (IR §6)."""

    __slots__ = ()
    count: int

    def __getitem__(self, j: int) -> Ref:
        raise NotImplementedError

    def slice(self, lo: int, hi: int) -> "Refs":
        raise NotImplementedError

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Ref]:
        for j in range(self.count):
            yield self[j]

    def to_json(self):
        raise NotImplementedError


class Affine(Refs):
    """leaf j -> (space, idx, base + j)."""

    __slots__ = ("space", "idx", "base", "count")

    def __init__(self, space: str, idx: int, base: int, count: int) -> None:
        self.space, self.idx, self.base, self.count = space, idx, base, count

    def __getitem__(self, j: int) -> Ref:
        if j < 0:
            j += self.count
        if not (0 <= j < self.count):
            raise IndexError(j)
        return (self.space, self.idx, self.base + j)

    def slice(self, lo: int, hi: int) -> Refs:
        lo, hi = max(lo, 0), min(hi, self.count)
        return Affine(self.space, self.idx, self.base + lo, max(hi - lo, 0))

    def to_json(self):
        return {"affine": [self.space, self.idx, self.base, self.count]}

    def __repr__(self) -> str:
        return f"Affine({self.space},{self.idx},{self.base}+{self.count})"


class Explicit(Refs):
    __slots__ = ("items", "count")

    def __init__(self, items: tuple[Ref, ...]) -> None:
        self.items = tuple(items)
        self.count = len(self.items)

    def __getitem__(self, j: int) -> Ref:
        return self.items[j]

    def slice(self, lo: int, hi: int) -> Refs:
        return Explicit(self.items[lo:hi])

    def to_json(self):
        return {"explicit": [list(r) for r in self.items]}

    def __repr__(self) -> str:
        return f"Explicit({self.count})"


class Strided(Refs):
    """Two-level strided view of one node's/parameter's leaves: leaf j -> base + (j // inner) * outer_stride
    + (j % inner) * inner_stride. Describes transposes, axis selections and column gathers in O(1)
    (e.g. one KV head's rows out of a `[T][KVH][D]` cache, or a `[D][16]` column block of V).

    Zero strides denote repetition of the *same* reference: `Strided(s, i, b, 1, 0, 0, n)` is the leaf `b`
    of `(s, i)` referenced `n` times — one produced value, broadcast. It is what
    `concat` normalises `[z.refs] * n` to; it never merges distinct produced values."""

    __slots__ = ("space", "idx", "base", "inner", "outer_stride", "inner_stride", "count")

    def __init__(self, space: str, idx: int, base: int, inner: int, outer_stride: int, inner_stride: int, count: int) -> None:
        self.space, self.idx, self.base = space, idx, base
        self.inner, self.outer_stride, self.inner_stride, self.count = inner, outer_stride, inner_stride, count

    def __getitem__(self, j: int) -> Ref:
        if j < 0:
            j += self.count
        if not (0 <= j < self.count):
            raise IndexError(j)
        q, r = divmod(j, self.inner)
        return (self.space, self.idx, self.base + q * self.outer_stride + r * self.inner_stride)

    def slice(self, lo: int, hi: int) -> Refs:
        lo, hi = max(lo, 0), min(hi, self.count)
        if hi <= lo:
            return Explicit(())
        if lo % self.inner == 0 and (hi % self.inner == 0 or hi == self.count):
            return Strided(self.space, self.idx, self.base + (lo // self.inner) * self.outer_stride, self.inner,
                           self.outer_stride, self.inner_stride, hi - lo)
        if hi - lo <= self.inner - lo % self.inner:  # within one inner run
            return Affine(self.space, self.idx, self[lo][2], hi - lo) if self.inner_stride == 1 else \
                Strided(self.space, self.idx, self[lo][2], 1, self.inner_stride, 0, hi - lo)
        return Explicit(tuple(self[j] for j in range(lo, hi)))

    def to_json(self):
        return {"strided": [self.space, self.idx, self.base, self.inner, self.outer_stride, self.inner_stride, self.count]}

    def __repr__(self) -> str:
        return f"Strided({self.space},{self.idx},{self.base},inner={self.inner},{self.outer_stride}/{self.inner_stride},{self.count})"


Lane = tuple[str, int, int, int, tuple[tuple[int, int], ...]]


def grid_ref(run: int, lanes: tuple[Lane, ...], k: int) -> tuple[int, Ref]:
    """`(j, reference k)` of the grid `(run, lanes)`: lane j, period t's digits in the lane's levels (innermost first, the
    last unbounded), offset o in the run (PROTOCOL.md §3)."""
    t, r = divmod(k, len(lanes) * run)
    j, o = divmod(r, run)
    s, i, b, e, levels = lanes[j]
    leaf = b + o * e
    for n, (rad, d) in enumerate(levels):
        if n == len(levels) - 1:
            leaf += t * d
        else:
            t, digit = divmod(t, rad)
            leaf += digit * d
    return j, (s, i, leaf)


def lane_bounds(run: int, lane: Lane) -> tuple[int, int]:
    """The least and greatest leaf a lane of a whole grid reaches."""
    _s, _i, b, e, levels = lane
    lo = hi = b
    for rad, d in ((run, e), *levels):
        lo, hi = lo + (rad - 1) * min(d, 0), hi + (rad - 1) * max(d, 0)
    return lo, hi


def _merged_levels(levels) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for r, d in levels:
        if out and d == out[-1][0] * out[-1][1]:
            out[-1] = (out[-1][0] * r, out[-1][1])
        else:
            out.append((r, d))
    return tuple(out)


def spell(run: int, lanes) -> "Refs":
    """The grid `(run, lanes)` in the first form that spells it: `Strided` (`Affine`) for one lane and at most one level,
    `Explicit` for one reference a lane, `Grid` otherwise (PROTOCOL.md §3, merges first)."""
    lanes = [(s, i, b, e, _merged_levels(lv)) for s, i, b, e, lv in lanes]
    g = len(lanes)
    if g >= 2 and len({(s, i, e, lv) for s, i, _b, e, lv in lanes}) == 1:
        s, i, b0, e, lv = lanes[0]
        c = lanes[1][2] - b0
        if all(lanes[j][2] == b0 + j * c for j in range(g)) and (run == 1 or c == run * e):
            run, lanes = run * g, [(s, i, b0, c if run == 1 else e, lv)]
    if len(lanes) == 1:
        s, i, b, e, lv = lanes[0]
        while lv and (run == 1 or lv[0][1] == run * e):
            e = lv[0][1] if run == 1 else e
            run, lv = run * lv[0][0], _merged_levels(lv[1:])
        if run == 1:
            e = 1
        if not lv:
            return Affine(s, i, b, run) if e == 1 else Strided(s, i, b, run, 0, e, run)
        if len(lv) == 1:
            return Strided(s, i, b, run, lv[0][1], e, run * lv[0][0])
        lanes = [(s, i, b, e, lv)]
    if run == 1 and all(not lv for *_x, lv in lanes):
        return Explicit(tuple((s, i, b) for s, i, b, _e, _lv in lanes))
    return Grid(run, tuple(lanes))


def as_grid(p: "Refs") -> tuple[int, tuple[Lane, ...]]:
    """`(run, lanes)` of an affine, strided, explicit or grid part: the grid its spelling stands for (PROTOCOL.md §3)."""
    if isinstance(p, Grid):
        return p.run, p.lanes
    if isinstance(p, Affine):
        return p.count, ((p.space, p.idx, p.base, 1, ()),)
    if isinstance(p, Strided):
        T = max(1, -(-p.count // p.inner))
        return p.inner, ((p.space, p.idx, p.base, p.inner_stride, ((T, p.outer_stride),) if T > 1 else ()),)
    if isinstance(p, Explicit):
        return 1, tuple((s, i, j, 0, ()) for s, i, j in p.items)
    raise TypeError(f"{p!r} is not a part")


def recut(registered_bits: int, reader_bits: int, part: "Refs | None" = None, leaves: int | None = None) -> int:
    """q, the reader leaves one leaf of an input node of width ``registered_bits`` is read as at width ``reader_bits``:
    reader leaf j is bits [(j mod q) * reader_bits, (j mod q + 1) * reader_bits) of registered leaf j div q, the low part
    first (PROTOCOL.md §3). Given the ``part`` that reads it and the node's ``leaves``, its lanes are checked to read whole
    registered leaves inside the node. Refused (ValueError) by name: recut-width, recut-boundary, recut-overrun."""
    if not (reader_bits > 0 and registered_bits > 0 and registered_bits % reader_bits == 0):
        raise ValueError(f"recut-width: an input node of {registered_bits}-bit leaves read in {reader_bits}-bit leaves: the "
                         "reader's width divides the registered width")
    q = registered_bits // reader_bits
    if part is None or not part.count:
        return q
    run, lanes = as_grid(part)
    for ln in lanes:
        _s, _i, b, e, lv = ln
        if q > 1 and not (e == 1 and b % q == 0 and run % q == 0 and part.count % q == 0 and all(d % q == 0 for _r, d in lv)):
            raise ValueError(f"recut-boundary: {part.to_json()} reads an input node of {registered_bits}-bit leaves in "
                             f"{reader_bits}-bit leaves other than whole registered leaves")
        if isinstance(part, Strided):
            k, last = divmod(part.count - 1, part.inner)
            at = [b + x * part.outer_stride + y * e for x, y in
                  [(0, 0), (0, min(part.inner, part.count) - 1), (k, 0), (k, last)] + ([(k - 1, part.inner - 1)] if k else [])]
            lo, hi = min(at), max(at)
        else:
            lo, hi = lane_bounds(run, ln)
        if leaves is not None and not (0 <= lo and hi < leaves * q):
            raise ValueError(f"recut-overrun: {part.to_json()} reads past the {leaves} {registered_bits}-bit leaves of its "
                             "input node")
    return q


class Grid(Refs):
    """`len(lanes)` lanes of `run` references a period, interleaved, over `periods` periods (PROTOCOL.md §3). A lane is
    `(space, idx, base, inner_stride, ((radix, step), ...))`, innermost level first; every lane's radices multiply to
    `periods`."""

    __slots__ = ("run", "lanes", "periods", "count")

    def __init__(self, run: int, lanes: tuple[Lane, ...]) -> None:
        self.run, self.lanes = run, tuple((s, i, b, e, tuple(map(tuple, lv))) for s, i, b, e, lv in lanes)
        self.periods = math.prod(r for r, _d in self.lanes[0][4]) if self.lanes else 0
        self.count = len(self.lanes) * run * self.periods

    def __getitem__(self, k: int) -> Ref:
        if k < 0:
            k += self.count
        if not (0 <= k < self.count):
            raise IndexError(k)
        return grid_ref(self.run, self.lanes, k)[1]

    def pieces(self, t0: int = 0, t1: int | None = None) -> list[Refs]:
        """Periods `[t0, t1)` as one run a lane a period, in order."""
        run, g = self.run, len(self.lanes)
        out: list[Refs] = []
        for t in range(t0, self.periods if t1 is None else t1):
            for j, (s, i, _b, e, _lv) in enumerate(self.lanes):
                leaf = grid_ref(run, self.lanes, (t * g + j) * run)[1][2]
                out.append(Affine(s, i, leaf, run) if e == 1 or run == 1 else Strided(s, i, leaf, run, 0, e, run))
        return out

    def lane_parts(self) -> list[Refs]:
        """Each lane's references as one-level runs, in no order: the grid's leaves as a set, repeats aside."""
        out: list[Refs] = []
        for s, i, b, e, lv in self.lanes:
            if not lv:
                out.append(Strided(s, i, b, self.run, 0, e, self.run))
                continue
            (r1, d1), rest = lv[0], lv[1:]
            for q in range(math.prod(r for r, _d in rest)):
                off = 0
                for n, (r, d) in enumerate(rest):
                    q, digit = (0, q) if n == len(rest) - 1 else divmod(q, r)
                    off += digit * d
                out.append(Strided(s, i, b + off, self.run, d1, e, self.run * r1))
        return out

    def slice(self, lo: int, hi: int) -> Refs:
        lo, hi = max(lo, 0), min(hi, self.count)
        if hi <= lo:
            return Explicit(())
        p = len(self.lanes) * self.run
        t0, t1 = lo // p, -(-hi // p)
        return concat(self.pieces(t0, t1)).slice(lo - t0 * p, hi - t0 * p)

    def to_json(self):
        return {"grid": [self.run, [[s, i, b, e, [list(x) for x in lv]] for s, i, b, e, lv in self.lanes]]}

    def __repr__(self) -> str:
        return f"Grid({self.run}x{len(self.lanes)}x{self.periods})"


class Concat(Refs):
    """Concatenation of reference sequences (binary search on `[j]`); descriptor size = number of parts.

    `prefix` is `(log, j)` when the first `j` parts are `log.parts[:j]` (a `Concat` cut from a `PartLog`), else None. A cut's
    `parts` and `starts` are read-only sequences over the log's lists (`_Prefix`), not copies of their first `j` items."""

    __slots__ = ("parts", "starts", "count", "prefix")

    def __init__(self, parts: list[Refs]) -> None:
        self.parts = list(parts)
        self.starts = []
        pos = 0
        for p in self.parts:
            self.starts.append(pos)
            pos += p.count
        self.count = pos
        self.prefix = None

    def __getitem__(self, j: int) -> Ref:
        if j < 0:
            j += self.count
        if not (0 <= j < self.count):
            raise IndexError(j)
        k = _bisect(bisect.bisect_right, self.starts, j) - 1
        return self.parts[k][j - self.starts[k]]

    def slice(self, lo: int, hi: int) -> Refs:
        lo, hi = max(lo, 0), min(hi, self.count)
        out: list[Refs] = []
        starts = self.starts
        k0, k1 = max(_bisect(bisect.bisect_right, starts, lo) - 1, 0), _bisect(bisect.bisect_left, starts, hi)
        for st, p in zip(starts[k0:k1], self.parts[k0:k1]):
            a, b = max(lo, st) - st, min(hi, st + p.count) - st
            if b > a:
                out.append(p.slice(a, b))
        return concat(out)

    def to_json(self):
        return {"concat": [p.to_json() for p in self.parts]}

    def __repr__(self) -> str:
        return f"Concat({len(self.parts)} parts, {self.count})"


def refs_from_json(j) -> Refs:
    if "affine" in j:
        k, i, b, c = j["affine"]
        return Affine(k, i, b, c)
    if "strided" in j:
        return Strided(*j["strided"])
    if "concat" in j:
        return Concat([refs_from_json(x) for x in j["concat"]])
    if "grid" in j:
        return _grid_from_json(j["grid"])
    return Explicit(tuple((r[0], r[1], r[2]) for r in j["explicit"]))


def _grid_from_json(a) -> Grid:
    def nat(x, lo=0):
        return isinstance(x, int) and not isinstance(x, bool) and x >= lo

    def lane(x):
        return (isinstance(x, list) and len(x) == 5 and x[0] in ("p", "n") and nat(x[1]) and nat(x[2], -(1 << 62))
                and nat(x[3], -(1 << 62)) and isinstance(x[4], list)
                and all(isinstance(v, list) and len(v) == 2 and nat(v[0], 2) and nat(v[1], -(1 << 62)) for v in x[4]))
    if not (isinstance(a, list) and len(a) == 2 and nat(a[0], 1) and isinstance(a[1], list) and a[1] and all(map(lane, a[1]))):
        raise ValueError(f"grid-shape: grid {a!r} is not [run, [[space, index, base, inner_stride, [[radix, step], ...]], ...]]")
    if len({math.prod(r for r, _d in x[4]) for x in a[1]}) != 1:
        raise ValueError(f"grid-shape: grid {a!r}: its lanes' radices multiply to different period counts")
    g = Grid(a[0], tuple((s, i, b, e, tuple((r, d) for r, d in lv)) for s, i, b, e, lv in a[1]))
    if not isinstance(spell(g.run, g.lanes), Grid):
        raise ValueError(f"grid-spelled: grid {a!r}: an affine, strided or explicit part spells it")
    return g


def is_broadcast(p: Refs) -> bool:
    """A part that references one leaf `count` times: a single-leaf `Affine` or a zero-stride `Strided`."""
    return (isinstance(p, Strided) and p.outer_stride == 0 and p.inner_stride == 0) or (isinstance(p, Affine) and p.count == 1)


def broadcast(space: str, idx: int, leaf: int, count: int) -> Refs:
    """`count` references to leaf `leaf` of `(space, idx)` — one value, repeated (zero-stride `Strided`)."""
    return Affine(space, idx, leaf, 1) if count == 1 else Strided(space, idx, leaf, 1, 0, 0, count)


def _merge_broadcasts(parts: list[Refs]) -> list[Refs]:
    """Adjacent repetitions of the *same* leaf become one zero-stride `Strided`. Only parts that already denote a
    single leaf are merged, so `[z, z, z]` (one produced value) compacts while `[z0, z1, z2]` (three produced
    values) is untouched."""
    out: list[Refs] = []
    for p in parts:
        if out and is_broadcast(p) and is_broadcast(out[-1]):
            q = out[-1]
            if (q.space, q.idx, q.base) == (p.space, p.idx, p.base):
                out[-1] = broadcast(q.space, q.idx, q.base, q.count + p.count)
                continue
        out.append(p)
    return out


def concat(parts: list[Refs]) -> Refs:
    parts = _merge_broadcasts([p for p in parts if p.count])
    if not parts:
        return Explicit(())
    if len(parts) == 1:
        return parts[0]
    ok = all(isinstance(p, Affine) for p in parts)
    if ok:
        a0 = parts[0]
        pos = a0.base + a0.count
        for p in parts[1:]:
            if not (p.space == a0.space and p.idx == a0.idx and p.base == pos):
                ok = False
                break
            pos += p.count
        if ok:
            return Affine(a0.space, a0.idx, a0.base, pos - a0.base)
    total = sum(p.count for p in parts)
    if total > 64 or any(not isinstance(p, Explicit) for p in parts):
        return Concat(parts)
    items: list[Ref] = []
    for p in parts:
        items.extend(p)
    return Explicit(tuple(items))


class _Prefix:
    """`base[:n] + list(tail)` as a read-only sequence that shares `base` instead of copying it. `base` is a `PartLog`'s `parts` or
    `starts`, which only ever grow past `n`, so the sequence never changes. A causal attention cuts one log once per call, so copied
    prefixes cost O(rows) per call and O(rows^2) per sequence (GLM-5.3 at 2048 rows: tens of GB); a shared one costs O(1)."""

    __slots__ = ("base", "n", "tail")
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, base: list, n: int, tail: tuple = ()) -> None:
        self.base, self.n, self.tail = base, n, tail

    def __len__(self) -> int:
        return self.n + len(self.tail)

    def __getitem__(self, i):
        n = self.n
        if type(i) is int and 0 <= i < n:
            return self.base[i]
        if isinstance(i, slice):
            lo, hi, step = i.indices(n + len(self.tail))
            if step != 1:
                return list(self)[i]
            if hi <= lo:
                return []
            if lo >= n:
                return list(self.tail[lo - n:hi - n])
            if hi <= n:
                return self.base[lo:hi]
            return self.base[lo:n] + list(self.tail[:hi - n])
        if i < 0:
            i += n + len(self.tail)
            if i < 0:
                raise IndexError("index out of range")
        return self.base[i] if i < n else self.tail[i - n]

    def __iter__(self):
        return chain(islice(self.base, self.n), self.tail)

    def __reversed__(self):
        return chain(reversed(self.tail), islice(reversed(self.base), len(self.base) - self.n, None))

    def __eq__(self, other) -> bool:
        try:
            m = len(other)
        except TypeError:
            return NotImplemented
        return len(self) == m and all(a == b for a, b in zip(self, other))

    def __repr__(self) -> str:
        return repr(list(self))


def _bisect(search, a, x) -> int:
    """`search(a, x)` for `search` one of `bisect`'s, on an ascending list or `_Prefix`; a `_Prefix` is searched in C, not item by
    item through `__getitem__`."""
    if type(a) is not _Prefix:
        return search(a, x)
    k = search(a.base, x, 0, a.n)
    return k if k < a.n else a.n + search(a.tail, x)


class PartLog:
    """Parts in order, cut into `Concat`s that share them as a prefix. The key rows an attention call reads are the rows
    the call before read plus one: cut from one log, each call's sequence costs O(1) parts of work, not O(rows).

    `cut(j, extra)` is exactly `concat(parts[:j] + extra)`. The `Concat` it returns records `prefix = (log, j)`, so a
    consumer that walks parts (`runs`, the descriptor encoder, liveness, the correspondence emitter) keeps what it
    computed for `parts[:j]` in the log and pays only for the parts past the longest cut it has seen. `memo` holds those
    consumers' per-log state (keyed by consumer); a log is append-only, so a prefix never changes once cut."""

    __slots__ = ("parts", "starts", "memo", "_bad", "_chain", "_explicit", "_leaf", "_affine", "_runs", "_run_n", "_run_last")

    def __init__(self, parts: list[Refs] = ()) -> None:
        self.parts: list[Refs] = []
        self.starts: list[int] = [0]   # starts[i] = leaves before part i; starts[len] = all leaves
        self.memo: dict = {}
        self._bad: int | None = None   # first i with parts[:i + 1] not in `concat`'s normal form (an empty part, a broadcast merge)
        self._chain = 0                # parts[:_chain] is one contiguous run of `Affine`s (`concat` merges it into one `Affine`)
        self._explicit = 0             # parts[:_explicit] are all `Explicit`
        self._leaf = 0                 # parts[:_leaf] are not `Concat`s
        self._affine = 0               # parts[:_affine] are all `Affine`
        self._runs: list[tuple[str, int, int, int]] = []    # runs of parts[:len(_run_n) - 1]
        self._run_n: list[int] = [0]                        # _run_n[j] = len(runs(parts[:j]))
        self._run_last: list = [None]                       # _run_last[j] = runs(parts[:j])[-1]
        for p in parts:
            self.append(p)

    def __len__(self) -> int:
        return len(self.parts)

    def append(self, p: Refs) -> None:
        parts, i = self.parts, len(self.parts)
        if self._bad is None and (p.count == 0 or (i and is_broadcast(p) and is_broadcast(parts[-1])
                                                   and (p.space, p.idx, p.base) == (parts[-1].space, parts[-1].idx, parts[-1].base))):
            self._bad = i
        if self._chain == i and isinstance(p, Affine) and (i == 0 or (p.space, p.idx, p.base) == (
                parts[0].space, parts[0].idx, parts[0].base + self.starts[i])):
            self._chain = i + 1
        if self._explicit == i and isinstance(p, Explicit):
            self._explicit = i + 1
        if self._leaf == i and not isinstance(p, Concat):
            self._leaf = i + 1
        if self._affine == i and isinstance(p, Affine):
            self._affine = i + 1
        parts.append(p)
        self.starts.append(self.starts[-1] + p.count)

    def cut(self, j: int, extra: list[Refs] = ()) -> Refs:
        """`concat(parts[:j] + extra)`."""
        parts = self.parts
        if j < 2 or (self._bad is not None and self._bad < j) or j <= self._explicit:
            return concat(parts[:j] + list(extra))
        if j <= self._chain:
            if not extra:
                a0 = parts[0]
                return Affine(a0.space, a0.idx, a0.base, self.starts[j])
            return concat(parts[:j] + list(extra))
        tail = [p for p in extra if p.count]
        if tail:
            merged = _merge_broadcasts([parts[j - 1]] + tail)
            if merged[0] is not parts[j - 1]:
                return concat(parts[:j] + list(extra))
            tail = merged[1:]
        return self._cut(j, tail)

    def raw_cut(self, j: int, extra: list[Refs] = ()) -> Concat:
        """`Concat(parts[:j] + extra)`: the parts as they are, not in `concat`'s normal form (what the descriptor decoder builds)."""
        return self._cut(j, extra)

    def _cut(self, j: int, tail) -> Concat:
        c = Concat.__new__(Concat)
        tail = tuple(tail)
        starts, pos = [], self.starts[j]
        for p in tail:
            starts.append(pos)
            pos += p.count
        c.parts, c.starts = _Prefix(self.parts, j, tail), _Prefix(self.starts, j, tuple(starts))
        c.count, c.prefix = pos, (self, j)
        return c

    def leaf_parts(self, j: int) -> bool:
        """No part of `parts[:j]` is a `Concat`."""
        return j <= self._leaf

    def affine_parts(self, j: int) -> bool:
        """Every part of `parts[:j]` is an `Affine`."""
        return j <= self._affine

    def runs(self, j: int) -> tuple[list[tuple[str, int, int, int]], int]:
        """`(runs(Concat(parts[:j])), s)`: a fresh list, whose first `s` runs are also the first `s` runs of every longer cut."""
        n = self._run_n
        for k in range(len(n) - 1, j):
            _walk_runs(self._runs, self.parts[k])
            n.append(len(self._runs))
            self._run_last.append(self._runs[-1] if self._runs else None)
        c = n[j]
        if not c:
            return [], 0
        out = self._runs[:c]
        out[-1] = self._run_last[j]
        return out, c - 1

    def shared_runs(self, lo: int, hi: int) -> list[tuple[str, int, int, int]]:
        """Shared runs `lo..hi` (`hi` at most an `s` that `runs` returned): final, whatever is appended later."""
        return self._runs[lo:hi]


def _push_run(out, space, idx, lo, hi):
    if out and out[-1][0] == space and out[-1][1] == idx and out[-1][3] == lo:
        out[-1] = (space, idx, out[-1][2], hi)
    else:
        out.append((space, idx, lo, hi))


def _walk_runs(out, r):
    if isinstance(r, Affine):
        if r.count:
            _push_run(out, r.space, r.idx, r.base, r.base + r.count)
    elif isinstance(r, Strided):
        if r.inner_stride == 1 or r.inner == 1:
            rows = r.count // r.inner
            for i in range(rows):
                b = r.base + i * r.outer_stride
                _push_run(out, r.space, r.idx, b, b + r.inner)
        else:
            for space, idx, leaf in r:
                _push_run(out, space, idx, leaf, leaf + 1)
    elif isinstance(r, Concat):
        for p in r.parts:
            _walk_runs(out, p)
    elif isinstance(r, Grid):
        for p in r.pieces():
            _walk_runs(out, p)
    else:
        for space, idx, leaf in r:
            _push_run(out, space, idx, leaf, leaf + 1)


def prefix_runs(refs: Refs) -> tuple[list[tuple[str, int, int, int]], PartLog | None, int]:
    """`(runs(refs), log, s)`: for a `Concat` cut from a `PartLog`, the first `s` runs are `log`'s shared runs (the same for
    every cut at least as long); otherwise `log` is None and `s` is 0."""
    if isinstance(refs, Concat) and refs.prefix is not None:
        log, j = refs.prefix
        out, s = log.runs(j)
        for p in refs.parts[j:]:
            _walk_runs(out, p)
        return out, log, min(s, len(out))
    out = []
    _walk_runs(out, refs)
    return out, None, 0


def runs(refs: Refs) -> list[tuple[str, int, int, int]]:
    """Maximal contiguous runs `(space, idx, lo, hi)` of a reference sequence, in sequence order (pure; used by
    the derivation frontend, program families and the capture fold — none of which need torch for it)."""
    return prefix_runs(refs)[0]


def strided_view(c, base: int, inner: int, outer_stride: int, inner_stride: int, count: int) -> Refs:
    """A `Strided` over the leaves of a Coll whose refs are a single `Affine` run (parameter or node output)."""
    r = c.refs
    if not isinstance(r, Affine):
        # fall back to explicit resolution through the underlying refs
        return Explicit(tuple(r[base + (j // inner) * outer_stride + (j % inner) * inner_stride] for j in range(count)))
    return Strided(r.space, r.idx, r.base + base, inner, outer_stride, inner_stride, count)


class Coll:
    """A typed collection of value references. Immutable; all operations are reference rearrangements
    (no gates). Indexing an `Array` yields a `Coll` of the element type; `.field(name)` for records;
    `[i]` on a tuple yields the field."""

    __slots__ = ("type", "refs")

    def __init__(self, type: Type, refs: Refs) -> None:
        if refs.count != type.leaves:
            raise ValueError(f"reference count {refs.count} does not match type {type!r} ({type.leaves} leaves)")
        self.type, self.refs = type, refs

    # -- structure ------------------------------------------------------------------
    def __len__(self) -> int:
        t = self.type
        if isinstance(t, Array):
            return t.n
        if isinstance(t, (Tuple, RecordType)):
            return len(t.fields)
        raise TypeError("a scalar has no length")

    def __getitem__(self, i):
        t = self.type
        if isinstance(t, Array):
            if isinstance(i, slice):
                lo, hi, step = i.indices(t.n)
                if step != 1:
                    raise ValueError("only unit-stride slices")
                L = t.elem.leaves
                return Coll(Array(hi - lo, t.elem), self.refs.slice(lo * L, hi * L))
            if i < 0:
                i += t.n
            if not (0 <= i < t.n):
                raise IndexError(i)
            L = t.elem.leaves
            return Coll(t.elem, self.refs.slice(i * L, (i + 1) * L))
        if isinstance(t, Tuple):
            ft = t.fields[i]
            off = t.field_offset(i)
            return Coll(ft, self.refs.slice(off, off + ft.leaves))
        if isinstance(t, RecordType):
            if isinstance(i, str):
                return self.field(i)
            _, ft = t.fields[i]
            off = t.field_offset(i)
            return Coll(ft, self.refs.slice(off, off + ft.leaves))
        raise TypeError(f"cannot index a {t!r}")

    def field(self, name: str) -> "Coll":
        t = self.type
        if not isinstance(t, RecordType):
            raise TypeError(f"{t!r} has no fields")
        i = t.index(name)
        return self[i]

    def __getattr__(self, name: str) -> "Coll":
        if name in ("type", "refs"):
            raise AttributeError(name)
        try:
            return self.field(name)
        except (TypeError, KeyError):
            raise AttributeError(name)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def reshape(self, t: Type) -> "Coll":
        """Same flat leaf order, different static structure (e.g. `Array<K>` -> `Array<K/16, Array<16>>`)."""
        return Coll(t, self.refs)

    def take(self, axis: int, m: int) -> "Coll":
        """Member `m` along array axis `axis` (used by `batch`)."""
        if axis == 0:
            return self[m]
        t = self.type
        if not isinstance(t, Array):
            raise TypeError(f"axis {axis} requires nested Arrays, got {t!r}")
        n, member_t = remove_axis(t, axis)
        if m < 0:
            m += n
        if not 0 <= m < n:
            raise IndexError(m)
        # one run of the member's leaves per index over the axes before `axis`, cut from `refs` directly: slicing each
        # outer element first would cost every member a pass over all of `refs`
        outer, inner = 1, t
        for _ in range(axis):
            outer *= inner.n
            inner = inner.elem
        L = inner.elem.leaves
        stride = inner.n * L
        parts = [self.refs.slice(k * stride + m * L, k * stride + (m + 1) * L) for k in range(outer)]
        return Coll(member_t, concat(parts))

    @property
    def leaf(self) -> Ref:
        if not isinstance(self.type, Value):
            raise TypeError(f"{self.type!r} is not a scalar")
        return self.refs[0]

    def __repr__(self) -> str:
        return f"Coll({self.type!r}, {self.refs!r})"


def coll_concat(t: Type, parts: list[Coll]) -> Coll:
    """Assemble a collection of type `t` from parts in leaf order (arrays from elements, tuples/records
    from fields)."""
    return Coll(t, concat([p.refs for p in parts]))


def array_of(parts: list[Coll]) -> Coll:
    if not parts:
        raise ValueError("array_of needs at least one element (use Coll(Array(0, T), Explicit(())) otherwise)")
    et = parts[0].type
    for p in parts:
        if p.type != et:
            raise TypeError(f"array elements differ in type: {et!r} vs {p.type!r}")
    return coll_concat(Array(len(parts), et), parts)


def tuple_of(*parts: Coll) -> Coll:
    return coll_concat(Tuple(*[p.type for p in parts]), list(parts))


def record_of(**parts: Coll) -> Coll:
    return coll_concat(RecordType(**{k: v.type for k, v in parts.items()}), list(parts.values()))

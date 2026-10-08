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
from typing import Iterator

from .types import Array, Record, Tuple, Type, Value, remove_axis

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
    of `(s, i)` referenced `n` times — one produced value, broadcast (frontend rulings §8.5). It is what
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


class Concat(Refs):
    """Concatenation of reference sequences (binary search on `[j]`); descriptor size = number of parts.

    `prefix` is `(log, j)` when the first `j` parts are `log.parts[:j]` (a `Concat` cut from a `PartLog`), else None."""

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
        k = bisect.bisect_right(self.starts, j) - 1
        return self.parts[k][j - self.starts[k]]

    def slice(self, lo: int, hi: int) -> Refs:
        lo, hi = max(lo, 0), min(hi, self.count)
        out: list[Refs] = []
        starts, parts = self.starts, self.parts
        k, n = max(bisect.bisect_right(starts, lo) - 1, 0), len(parts)
        while k < n and starts[k] < hi:
            st, p = starts[k], parts[k]
            a, b = max(lo, st) - st, min(hi, st + p.count) - st
            if b > a:
                out.append(p.slice(a, b))
            k += 1
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
    return Explicit(tuple((r[0], r[1], r[2]) for r in j["explicit"]))


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
        c = Concat.__new__(Concat)
        c.parts = parts[:j] + tail
        c.starts = self.starts[:j]
        pos = self.starts[j]
        for p in tail:
            c.starts.append(pos)
            pos += p.count
        c.count, c.prefix = pos, (self, j)
        return c

    def raw_cut(self, j: int, extra: list[Refs] = ()) -> Concat:
        """`Concat(parts[:j] + extra)`: the parts as they are, not in `concat`'s normal form (what the descriptor decoder builds)."""
        c = Concat.__new__(Concat)
        c.parts, c.starts, pos = self.parts[:j], self.starts[:j], self.starts[j]
        for p in extra:
            c.parts.append(p)
            c.starts.append(pos)
            pos += p.count
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
        if isinstance(t, (Tuple, Record)):
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
        if isinstance(t, Record):
            if isinstance(i, str):
                return self.field(i)
            _, ft = t.fields[i]
            off = t.field_offset(i)
            return Coll(ft, self.refs.slice(off, off + ft.leaves))
        raise TypeError(f"cannot index a {t!r}")

    def field(self, name: str) -> "Coll":
        t = self.type
        if not isinstance(t, Record):
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
    return coll_concat(Record(**{k: v.type for k, v in parts.items()}), list(parts.values()))

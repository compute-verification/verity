"""Versioned memory model (pure): `(allocation, element interval) -> Refs` with generations.

Implemented by lane L2.  The model maps physical storage to *immutable* source-gate references in the fold's
root body (`("p", i, leaf)` parameters, `("n", k, leaf)` node outputs; see `verity.ir/refs.py`).  Device
addresses are capture-time information only: once a value is referenced by an instance, that reference never
changes, whatever later happens to the memory.

Discipline (design §9):

    Compute   1. resolve every read to Refs (raises Unresolved if any element has no producer)
              2. the fold emits the instance and learns its output gate refs
              3. apply every write:  view element j := refs[j]
    Move      refs = read(src) *now*; write(dst, refs)   (never "dst points at whatever src holds later")
    Alias     nothing to do: a view of the same allocation resolves through the same map
    Alloc     a new generation for the allocation; previous contents are unreachable

API
---
    m = MemoryModel()
    m.alloc(StorageAlloc)                  register/refresh an allocation (new generation invalidates old contents)
    m.free(alloc_id)                       drop the allocation (reads raise Unresolved)
    m.write(view, refs)                    refs.count == view.numel; element j (row-major over view.shape) := refs[j]
    m.read(view) -> Refs                   row-major refs over view.shape; raises Unresolved(view, missing) when any
                                           element is unwritten, freed, or written with a different element width
    m.read_rows(view, indices) -> Refs     concat(read(row_view(view, i)) for i in indices)
    m.coverage(view) -> float              fraction of elements resolvable (diagnostics)
    m.stats() -> dict                      allocations, live intervals, writes, reads

Width discipline: a written interval remembers the element width (bits) it was written with.  Reading it with
a different dtype width is `Unresolved` (reinterpretation across widths is Unsupported in the MVP).  Reading
with a different dtype of the *same* width (e.g. int32 vs uint32) is allowed only when `allow_same_width_reinterpret`.

Complexity: per allocation an ordered interval map keyed by byte offset; a write of a view with R contiguous
runs is O(R log n); reads are O(R log n + parts).  Refs returned are `concat` of run slices, so a contiguous
read of one producer is a single `Affine`.

Implementation notes (L2)
-------------------------
- Intervals are half-open byte ranges `[lo, hi)` of one allocation, sorted and pairwise disjoint, each holding
  one *primitive* `Refs` part (`Affine` / `Strided` / `Explicit`, never `Concat`) plus the width and dtype it
  was written with.  A write whose refs are a `Concat` is stored as one interval per part, so later row reads
  slice an `Affine`/`Strided` in O(1) instead of walking a `Concat` (a whole-buffer `copy_` followed by per-row
  reads stays linear).  A write splits the intervals it partially overlaps into their surviving pieces with
  re-based `refs.slice(...)`; a piece that no longer contains a whole element (an element clobbered by a
  narrower write) is dropped and its bytes become unwritten.
- `read` of a view whose byte extent lies inside one `Affine` interval (the common case: one producer, any
  view of it) is answered from the view geometry alone, in the same normal form the authored frontend uses for
  views (`verity_vllm/program/frontend/torch_frontend.py:_strided_refs`): `Affine` for a contiguous view, `Strided` for
  one strided axis or two collapsed axes (transposes, column slices, expands), a `Concat` over the leading axis
  beyond that.  Otherwise the read walks the element runs, collects the covering slices and compacts them
  (`compact_refs`): adjacent contiguous `Affine`s merge; equally spaced equal-length `Affine` rows of one
  producer become a `Strided`; equally spaced 1-D `Strided`s become a two-level `Strided`.  Compaction never
  changes the denoted sequence (checked element by element in `tests/test_memory_refs_semantics.py`).
- `Unresolved.missing` lists the unresolvable element runs `(start_elem, count)` in units of the view's dtype,
  absolute in the allocation (the same units as `views.element_runs`), in logical order; `reason` is one of
  `unwritten`, `width`, `dtype`, `freed`, `unknown`, or several joined with `+` (`misaligned` — an interval of
  the same width on a shifted byte grid — is checked defensively but cannot arise from `ViewDesc`s, whose
  offsets are in elements of their own dtype from the allocation start).
- A write into an allocation that was never announced (or was freed) registers it implicitly (counted in
  `stats().extra["implicit_allocs"]`) rather than failing: the write is a fact about the contents, and refusing
  it would only turn later reads into spurious `Unresolved`s.  Reads of unknown/freed allocations are
  `Unresolved(reason="unknown" | "freed")`.
- A view with zero-stride (broadcast) axes writes its repeated storage elements several times, in logical
  order (the last assignment wins); such a view reads back the repeated refs, once per logical position.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

from verity.ml.circuits.refs import Affine, Concat, Explicit, Refs, Strided, broadcast, concat

from verity_vllm.observe.events import StorageAlloc, ViewDesc
from verity_vllm.observe.fold.views import byte_extent, collapsed_dims, element_runs, row_view


class Unresolved(Exception):
    """A read touched elements with no live producer."""

    def __init__(self, view: ViewDesc, missing: list[tuple[int, int]], reason: str = "unwritten") -> None:
        self.view, self.missing, self.reason = view, missing, reason
        super().__init__(f"unresolved read of {view!r}: {reason}; missing element runs {missing[:4]}"
                         f"{'...' if len(missing) > 4 else ''}")


@dataclass
class MemoryStats:
    allocations: int = 0
    live_allocations: int = 0
    writes: int = 0
    reads: int = 0
    moves: int = 0
    intervals: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------------
# refs helpers
# ---------------------------------------------------------------------------------------------------------


def flat_parts(r: Refs) -> list[Refs]:
    """The primitive (non-`Concat`, non-empty) parts of `r` in order."""
    if not isinstance(r, Concat):
        return [r] if r.count else []
    out: list[Refs] = []
    stack: list[Refs] = [r]
    while stack:
        p = stack.pop()
        if isinstance(p, Concat):
            stack.extend(reversed(p.parts))
        elif p.count:
            out.append(p)
    return out


def geometry_refs(space: str, idx: int, base: int, shape: tuple[int, ...], strides: tuple[int, ...]) -> Refs:
    """Refs of a view with collapsed geometry `(shape, strides)` over one producer whose leaf for storage
    element `e` is `base + (e - e_0)` (`base` is the leaf of the view's first element).  Normal form as in
    `torch_frontend._strided_refs`: `Affine` (one contiguous axis), `Strided` (one strided axis or two axes),
    `Concat` over the leading axis beyond that."""
    if not shape:
        return Affine(space, idx, base, 1)
    if len(shape) == 1:
        n, st = shape[0], strides[0]
        if st == 1:
            return Affine(space, idx, base, n)
        if st == 0:
            return broadcast(space, idx, base, n)
        return Strided(space, idx, base, 1, st, 0, n)
    if len(shape) == 2:
        (n0, n1), (s0, s1) = shape, strides
        return Strided(space, idx, base, n1, s0, s1, n0 * n1)
    n0, s0 = shape[0], strides[0]
    return Concat([geometry_refs(space, idx, base + i * s0, shape[1:], strides[1:]) for i in range(n0)])




def access_runs(refs: Refs | list | None) -> list | None:
    """An access journal entry's refs as `runs()` lists: the fold keeps the `Refs` it read (shared with the instances, where a
    transposed weight's runs would be one per element); a list is already runs."""
    from verity.ml.circuits.refs import runs

    return runs(refs) if isinstance(refs, Refs) else refs


def _nested(view: ViewDesc) -> bool:
    """Whether `view`'s axes, by decreasing stride, nest (each stride at least the next axis's span): offsets decompose uniquely."""
    ax = sorted((d for d in range(view.rank) if view.shape[d] > 1), key=lambda d: -view.strides[d])
    return all(view.strides[x] > 0 and view.strides[x] >= view.strides[y] * view.shape[y] for x, y in zip(ax, ax[1:])) and \
        all(view.strides[d] > 0 for d in ax)


def _runs_meet(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    """Whether two address-sorted, disjoint element-run lists share an element."""
    i = j = 0
    while i < len(a) and j < len(b):
        (s, c), (t, e) = a[i], b[j]
        if s < t + e and t < s + c:
            return True
        if s + c <= t + e:
            i += 1
        else:
            j += 1
    return False


def address_ordered(view: ViewDesc, refs: Refs) -> tuple[ViewDesc, Refs]:
    """`view` written with `refs` (one producer's leaves base..base+numel in the view's row-major order) as the same element-to-leaf
    map with the axes in decreasing storage stride, so the write covers contiguous address runs: a transposed parameter view
    (GLM-5.3's absorbed W_UK_T, VLLM-14) writes one run per inner row, not one per element.  Other refs, and a view already in
    address order, pass through."""
    if not isinstance(refs, Affine) or len(view.shape) < 2:
        return view, refs
    order = sorted(range(len(view.shape)), key=lambda d: -view.strides[d])
    if order == list(range(len(view.shape))):
        return view, refs
    leaf, acc = [0] * len(view.shape), 1
    for d in reversed(range(len(view.shape))):
        leaf[d], acc = acc, acc * view.shape[d]
    pv = ViewDesc(view.alloc, view.offset, tuple(view.shape[d] for d in order), tuple(view.strides[d] for d in order), view.dtype, view.device)
    return pv, geometry_refs(refs.space, refs.idx, refs.base, pv.shape, tuple(leaf[d] for d in order))


def _merge_affine(parts: list[Refs]) -> list[Refs]:
    out: list[Refs] = []
    for p in parts:
        if out and isinstance(p, Affine) and isinstance(out[-1], Affine):
            q = out[-1]
            if q.space == p.space and q.idx == p.idx and q.base + q.count == p.base:
                out[-1] = Affine(q.space, q.idx, q.base, q.count + p.count)
                continue
        out.append(p)
    return out


def _rows_to_strided(parts: list[Refs]) -> list[Refs]:
    """Runs of >= 2 consecutive equal-length `Affine`s of one producer with a constant base step become one
    `Strided` (rows of a column slice / gather -> `Strided(inner=len, outer_stride=step, inner_stride=1)`; single
    leaves -> `Strided(inner=1, outer_stride=step, inner_stride=0)`, the broadcast form when the step is 0)."""
    out: list[Refs] = []
    i, n = 0, len(parts)
    while i < n:
        p = parts[i]
        if not isinstance(p, Affine):
            out.append(p)
            i += 1
            continue
        j = i + 1
        d = None
        while j < n:
            q = parts[j]
            if not (isinstance(q, Affine) and q.space == p.space and q.idx == p.idx and q.count == p.count):
                break
            step = q.base - parts[j - 1].base
            if d is None:
                d = step
            elif step != d:
                break
            j += 1
        k = j - i
        if k >= 2:
            assert d is not None
            if p.count == 1:
                out.append(broadcast(p.space, p.idx, p.base, k) if d == 0 else Strided(p.space, p.idx, p.base, 1, d, 0, k))
            else:
                out.append(Strided(p.space, p.idx, p.base, p.count, d, 1, k * p.count))
        else:
            out.append(p)
        i = j
    return out


def _strided_to_two_level(parts: list[Refs]) -> list[Refs]:
    """Runs of >= 2 consecutive one-axis `Strided`s (`inner == 1`) of one producer with equal stride/count and a
    constant base step become one two-level `Strided` (a transpose read row by row)."""
    out: list[Refs] = []
    i, n = 0, len(parts)
    while i < n:
        p = parts[i]
        if not (isinstance(p, Strided) and p.inner == 1):
            out.append(p)
            i += 1
            continue
        j = i + 1
        d = None
        while j < n:
            q = parts[j]
            if not (isinstance(q, Strided) and q.inner == 1 and q.space == p.space and q.idx == p.idx
                    and q.outer_stride == p.outer_stride and q.count == p.count):
                break
            step = q.base - parts[j - 1].base
            if d is None:
                d = step
            elif step != d:
                break
            j += 1
        k = j - i
        if k >= 2:
            assert d is not None
            out.append(Strided(p.space, p.idx, p.base, p.count, d, p.outer_stride, k * p.count))
        else:
            out.append(p)
        i = j
    return out


def compact_refs(parts: list[Refs]) -> Refs:
    """`concat(parts)` after structural compaction (same denoted sequence, smaller descriptor)."""
    flat: list[Refs] = []
    for p in parts:
        flat.extend(flat_parts(p))
    flat = _merge_affine(flat)
    flat = _rows_to_strided(flat)
    flat = _strided_to_two_level(flat)
    return concat(flat)


# ---------------------------------------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------------------------------------


class _Interval:
    __slots__ = ("bits", "dtype", "hi", "lo", "refs")

    def __init__(self, lo: int, hi: int, refs: Refs, bits: int, dtype: str) -> None:
        self.lo, self.hi, self.refs, self.bits, self.dtype = lo, hi, refs, bits, dtype

    def trimmed(self, lo: int, hi: int) -> _Interval | None:
        """The whole elements of this interval inside `[lo, hi)`, re-based; None if there is none."""
        w = self.bits // 8
        e_lo = -(-(max(lo, self.lo) - self.lo) // w)
        e_hi = (min(hi, self.hi) - self.lo) // w
        if e_hi <= e_lo:
            return None
        return _Interval(self.lo + e_lo * w, self.lo + e_hi * w, self.refs.slice(e_lo, e_hi), self.bits, self.dtype)

    def __repr__(self) -> str:
        return f"[{self.lo},{self.hi}) {self.dtype} {self.refs!r}"


#: [FOLD-01] distinct gathered-row geometries memoised per allocation; beyond this a geometry's rows are read uncached
_MAX_ROW_GEOMS = 256
#: [FOLD-01] a write overlapping more cached rows than this clears the allocation's row cache instead of popping row by row
_ROW_INVALIDATE_SCAN = 4096


class _Alloc:
    __slots__ = ("device", "generation", "id", "ivs", "live", "nbytes", "origin", "root_maps", "row_cache", "row_geoms", "starts")

    def __init__(self, id: int, generation: int, nbytes: int, device: str, origin: str) -> None:
        self.id, self.generation, self.nbytes, self.device, self.origin = id, generation, nbytes, device, origin
        self.live = True
        self.starts: list[int] = []          # ivs[k].lo, for bisect
        self.ivs: list[_Interval] = []
        # [FOLD-01] memo of `read(row_view(view, i))` for gathered rows (`read_rows`): `(geometry, i) -> Refs`, where geometry =
        # (offset, shape, strides, dtype) of the gathered view within this allocation.  A row's refs are a pure function of the
        # intervals under its byte extent, so an entry stays valid until a write / free / regeneration touches those bytes:
        # `_splice` pops exactly the rows whose extent overlaps the written range (`row_geoms` keeps each geometry's
        # (row-0 byte lo, row byte stride, row byte extent)), `free` / a new generation clear everything.  Identical results,
        # once per (row, write epoch) instead of once per read: attention gathers the whole KV context of every request at
        # every step (and every query row of a prefill), which made the fold's cost sum(context) x layers x steps.
        self.row_cache: dict[tuple, Refs] = {}
        self.row_geoms: dict[tuple, tuple[int, int, int]] = {}
        # [VLLM-14] the permuted weight roots written here (`write_root`): (address-ordered root view, space, idx, base, leaf stride per
        # axis, element runs), each an affine address -> leaf map that answers a read of any sub-box of its axes without slicing
        # intervals; dropped by an ordinary write over its byte extent or a root write over its elements
        self.root_maps: list[tuple[ViewDesc, str, int, int, tuple[int, ...], list[tuple[int, int]]]] = []

    def first_overlapping(self, lo: int) -> int:
        """Index of the first interval with `hi > lo` (== len(ivs) if none)."""
        i = bisect_right(self.starts, lo) - 1
        if i < 0 or self.ivs[i].hi <= lo:
            i += 1
        return i


class MemoryModel:
    def __init__(self, allow_same_width_reinterpret: bool = True) -> None:
        self.allow_same_width_reinterpret = allow_same_width_reinterpret
        self._allocs: dict[int, _Alloc] = {}
        self._n_allocs = 0
        self._n_writes = self._n_reads = self._n_moves = 0
        self._n_unresolved = self._n_implicit = self._n_generation_drops = self._n_frees = self._n_fast_reads = 0
        self._n_row_hits = 0   # [FOLD-01] gathered rows served from the per-allocation row cache (counted in `reads` too)

    # -- allocations --------------------------------------------------------------------------------------

    def alloc(self, ev: StorageAlloc) -> None:
        a = self._allocs.get(ev.alloc)
        if a is None:
            self._allocs[ev.alloc] = _Alloc(ev.alloc, ev.generation, ev.nbytes, ev.device, ev.origin)
            self._n_allocs += 1
            return
        if a.generation != ev.generation:
            if a.ivs:
                self._n_generation_drops += 1
            a.starts.clear()
            a.ivs.clear()
            a.row_cache.clear()
            a.root_maps.clear()
            a.generation = ev.generation
        a.live = True
        a.nbytes, a.device, a.origin = ev.nbytes, ev.device, ev.origin

    def free(self, alloc_id: int) -> None:
        a = self._allocs.get(alloc_id)
        if a is None:
            a = _Alloc(alloc_id, -1, 0, "", "unknown")
            self._allocs[alloc_id] = a
            self._n_allocs += 1
        a.starts.clear()
        a.ivs.clear()
        a.row_cache.clear()
        a.root_maps.clear()
        a.live = False
        self._n_frees += 1

    def _alloc_for_write(self, alloc_id: int) -> _Alloc:
        a = self._allocs.get(alloc_id)
        if a is None:
            a = _Alloc(alloc_id, -1, 0, "", "implicit")
            self._allocs[alloc_id] = a
            self._n_allocs += 1
            self._n_implicit += 1
        elif not a.live:
            a.live = True
            self._n_implicit += 1
        return a

    # -- writes -------------------------------------------------------------------------------------------

    def write(self, view: ViewDesc, refs: Refs) -> None:
        if refs.count != view.numel:
            raise ValueError(f"write of {refs.count} refs into {view!r} ({view.numel} elements)")
        a = self._alloc_for_write(view.alloc)
        isz, bits = view.itemsize, view.bits
        flat = flat_parts(refs)
        starts: list[int] = []
        pos = 0
        for p in flat:
            starts.append(pos)
            pos += p.count
        pos = 0
        for start, count in element_runs(view):
            pieces = _slice_parts(flat, starts, pos, pos + count)
            self._store_run(a, start * isz, (start + count) * isz, pieces, bits, view.dtype)
            pos += count
        self._n_writes += 1

    def write_root(self, view: ViewDesc, refs: Refs) -> None:
        """`write` of a weight root, in address order (`address_ordered`); a permuted root also keeps its address -> leaf map
        (`_root_map_read`), so a read along its permuted axes (MLA's `W_UV^T`: one element run per element) costs no interval scan."""
        pv, prefs = address_ordered(view, refs)
        a = self._alloc_for_write(view.alloc)
        keep, a.root_maps = a.root_maps, []
        self.write(pv, prefs)
        runs_ = element_runs(pv)
        # roots sharing an allocation interleave (W_UK_T and W_UV), so a map is dropped only when the new root's elements meet its own
        a.root_maps = [m for m in keep if not _runs_meet(m[5], runs_)]
        if pv is not view and _nested(pv):
            leaf, acc = [0] * view.rank, 1
            for d in reversed(range(view.rank)):
                leaf[d], acc = acc, acc * view.shape[d]
            order = sorted(range(view.rank), key=lambda d: -view.strides[d])
            a.root_maps.append((pv, refs.space, refs.idx, refs.base, tuple(leaf[d] for d in order), runs_))

    def _store_run(self, a: _Alloc, lo: int, hi: int, pieces: list[Refs], bits: int, dtype: str) -> None:
        w = bits // 8
        ivs: list[_Interval] = []
        pos = lo
        for p in pieces:
            ivs.append(_Interval(pos, pos + p.count * w, p, bits, dtype))
            pos += p.count * w
        if pos != hi:  # pragma: no cover - guarded by the count check in write()
            raise ValueError(f"refs cover {pos - lo} bytes, run needs {hi - lo}")
        self._splice(a, lo, hi, ivs)

    def _splice(self, a: _Alloc, lo: int, hi: int, pieces: list[_Interval]) -> None:
        """Replace the bytes `[lo, hi)` of `a` by `pieces`, keeping the whole-element survivors of the
        intervals cut at `lo` and `hi`."""
        ivs = a.ivs
        i0 = a.first_overlapping(lo)
        j = i0
        while j < len(ivs) and ivs[j].lo < hi:
            j += 1
        new: list[_Interval] = []
        if i0 < j and ivs[i0].lo < lo:
            left = ivs[i0].trimmed(ivs[i0].lo, lo)
            if left is not None:
                new.append(left)
        new.extend(pieces)
        if i0 < j and ivs[j - 1].hi > hi:
            right = ivs[j - 1].trimmed(hi, ivs[j - 1].hi)
            if right is not None:
                new.append(right)
        ivs[i0:j] = new
        a.starts[i0:j] = [x.lo for x in new]
        if a.root_maps:
            a.root_maps = [m for m in a.root_maps if not (byte_extent(m[0])[0] < hi and lo < byte_extent(m[0])[1])]
        if a.row_cache:
            self._invalidate_rows(a, lo, hi)

    def _invalidate_rows(self, a: _Alloc, lo: int, hi: int) -> None:
        """[FOLD-01] drop the memoised gathered rows whose byte extent overlaps `[lo, hi)` (the bytes a write just replaced)."""
        cache = a.row_cache
        for gk, (row0, stride, ext) in a.row_geoms.items():
            # rows i with row0 + i*stride < hi and row0 + i*stride + ext > lo
            i_lo = (lo - ext - row0) // stride + 1
            i_hi = -(-(hi - row0) // stride) - 1
            if i_lo < 0:
                i_lo = 0
            if i_hi < i_lo:
                continue
            if i_hi - i_lo + 1 > _ROW_INVALIDATE_SCAN:
                cache.clear()
                return
            for i in range(i_lo, i_hi + 1):
                cache.pop((gk, i), None)

    # -- reads --------------------------------------------------------------------------------------------

    def read(self, view: ViewDesc) -> Refs:
        self._n_reads += 1
        if view.numel == 0:
            return Explicit(())
        a = self._allocs.get(view.alloc)
        if a is None or not a.live:
            self._n_unresolved += 1
            raise Unresolved(view, element_runs(view), "unknown" if a is None else "freed")
        fast = self._fast_read(a, view) or (self._root_map_read(a, view) if a.root_maps else None)
        if fast is not None:
            self._n_fast_reads += 1
            return fast
        parts: list[Refs] = []
        missing: list[tuple[int, int]] = []
        reasons: set[str] = set()
        isz = view.itemsize
        for start, count in element_runs(view):
            self._collect(a, start * isz, (start + count) * isz, view, parts, missing, reasons)
        if missing:
            self._n_unresolved += 1
            raise Unresolved(view, _merge_runs(missing), "+".join(sorted(reasons)))
        return compact_refs(parts)

    def _fast_read(self, a: _Alloc, view: ViewDesc) -> Refs | None:
        """The read when one `Affine` interval covers the whole byte extent of `view` (else None)."""
        lo_b, hi_b = byte_extent(view)
        i = bisect_right(a.starts, lo_b) - 1
        if i < 0:
            return None
        x = a.ivs[i]
        if x.hi < hi_b or not isinstance(x.refs, Affine) or not self._compatible(x, view) or x.lo % view.itemsize:
            return None
        shape, strides = collapsed_dims(view)
        base = x.refs.base + (view.offset - x.lo // view.itemsize)
        return geometry_refs(x.refs.space, x.refs.idx, base, shape, strides)

    def _root_map_read(self, a: _Alloc, view: ViewDesc) -> Refs | None:
        """The read when `view` is a sub-box of a permuted weight root's axes (`write_root`): its leaf for element i is the root's
        affine map at i, so the refs are `geometry_refs` of the view's own geometry over the root's leaf strides; else None."""
        for rv, space, idx, base, leaf, _ in a.root_maps:
            if rv.bits != view.bits or view.offset < rv.offset or not (self.allow_same_width_reinterpret or rv.dtype == view.dtype):
                continue
            axes = [d for d in range(rv.rank) if rv.shape[d] > 1]     # rv is address ordered: strides decrease
            coord, rem = [0] * rv.rank, view.offset - rv.offset
            for d in axes:
                coord[d], rem = divmod(rem, rv.strides[d])
            if rem or any(coord[d] >= rv.shape[d] for d in axes):
                continue
            shape, strides, used = [], [], set()
            for n, st in zip(view.shape, view.strides):
                if n == 1:
                    continue
                d = next((d for d in axes if d not in used and rv.strides[d] == st), None)
                if d is None or coord[d] + n > rv.shape[d]:
                    break
                used.add(d)
                shape.append(n)
                strides.append(leaf[d])
            else:
                return geometry_refs(space, idx, base + sum(c * l for c, l in zip(coord, leaf)), tuple(shape), tuple(strides))
        return None

    def _compatible(self, x: _Interval, view: ViewDesc) -> bool:
        return x.bits == view.bits and (self.allow_same_width_reinterpret or x.dtype == view.dtype)

    def _collect(self, a: _Alloc, lo: int, hi: int, view: ViewDesc, parts: list[Refs],
                 missing: list[tuple[int, int]], reasons: set[str]) -> None:
        """Slices of the intervals covering the element run at bytes `[lo, hi)` into `parts`; the run's
        uncovered / incompatible elements into `missing` as `(start_elem, count)` runs (exact within the run:
        an element straddling two bad byte pieces is counted once), with their reasons."""
        ivs = a.ivs
        w = view.itemsize
        i = a.first_overlapping(lo)
        pos = lo
        bad: list[tuple[int, int]] = []
        while pos < hi:
            if i >= len(ivs) or ivs[i].lo >= hi:
                bad.append((pos, hi))
                reasons.add("unwritten")
                break
            x = ivs[i]
            if x.lo > pos:
                bad.append((pos, x.lo))
                reasons.add("unwritten")
                pos = x.lo
            b = min(hi, x.hi)
            if x.bits != view.bits:
                bad.append((pos, b))
                reasons.add("width")
            elif not self.allow_same_width_reinterpret and x.dtype != view.dtype:
                bad.append((pos, b))
                reasons.add("dtype")
            elif x.lo % w:
                bad.append((pos, b))
                reasons.add("misaligned")
            else:
                parts.append(x.refs.slice((pos - x.lo) // w, (b - x.lo) // w))
            pos = b
            i += 1
        if bad:
            missing.extend(_merge_runs([(p // w, -(-q // w) - p // w) for p, q in bad]))

    def read_rows(self, view: ViewDesc, indices: list[int] | tuple[int, ...]) -> Refs:
        a = self._allocs.get(view.alloc)
        if a is None or not a.live or not indices or view.rank == 0 or view.shape[0] == 0:
            return compact_refs([self.read(row_view(view, i)) for i in indices])
        gk = (view.offset, view.shape, view.strides, view.dtype)
        geom = a.row_geoms.get(gk)
        if geom is None:
            stride = view.strides[0] * view.itemsize
            if stride <= 0 or len(a.row_geoms) >= _MAX_ROW_GEOMS:
                return compact_refs([self.read(row_view(view, i)) for i in indices])
            lo0, hi0 = byte_extent(row_view(view, 0))
            a.row_geoms[gk] = (lo0, stride, hi0 - lo0)
        cache = a.row_cache
        parts: list[Refs] = []
        for i in indices:
            key = (gk, i)
            r = cache.get(key)
            if r is None:
                r = self.read(row_view(view, i))       # raises Unresolved exactly as before; only successful reads are memoised
                cache[key] = r
            else:
                self._n_reads += 1
                self._n_row_hits += 1
            parts.append(r)
        return compact_refs(parts)

    def move(self, src: ViewDesc, dst: ViewDesc) -> None:
        """read(src) now, then write(dst, ...).  Shapes must agree element-for-element (same numel, same width)."""
        if src.numel != dst.numel:
            raise ValueError(f"move of {src!r} into {dst!r}: element counts differ")
        if src.bits != dst.bits:
            raise ValueError(f"move of {src!r} into {dst!r}: element widths differ")
        refs = self.read(src)
        self.write(dst, refs)
        self._n_moves += 1

    # -- diagnostics --------------------------------------------------------------------------------------

    def coverage(self, view: ViewDesc) -> float:
        if view.numel == 0:
            return 1.0
        a = self._allocs.get(view.alloc)
        if a is None or not a.live:
            return 0.0
        parts: list[Refs] = []
        missing: list[tuple[int, int]] = []
        isz = view.itemsize
        for start, count in element_runs(view):
            self._collect(a, start * isz, (start + count) * isz, view, parts, missing, set())
        lost = sum(c for _, c in missing)          # per-run exact: logical positions, repeats counted each time
        return (view.numel - lost) / view.numel

    def stats(self) -> MemoryStats:
        return MemoryStats(
            allocations=self._n_allocs,
            live_allocations=sum(1 for a in self._allocs.values() if a.live),
            writes=self._n_writes, reads=self._n_reads, moves=self._n_moves,
            intervals=sum(len(a.ivs) for a in self._allocs.values()),
            extra={"unresolved": self._n_unresolved, "implicit_allocs": self._n_implicit,
                   "generation_drops": self._n_generation_drops, "frees": self._n_frees,
                   "fast_reads": self._n_fast_reads, "row_cache_hits": self._n_row_hits})

    def intervals(self, alloc_id: int) -> list[tuple[int, int, Refs, int, str]]:
        """`(lo_byte, hi_byte, refs, width_bits, dtype)` of the live intervals of an allocation (tests, reports)."""
        a = self._allocs.get(alloc_id)
        if a is None:
            return []
        return [(x.lo, x.hi, x.refs, x.bits, x.dtype) for x in a.ivs]


def _slice_parts(flat: list[Refs], starts: list[int], lo: int, hi: int) -> list[Refs]:
    """Primitive pieces denoting `[lo, hi)` of the sequence whose primitive parts are `flat` (part k starts at
    `starts[k]`); O(log parts + parts touched)."""
    out: list[Refs] = []
    k = max(bisect_right(starts, lo) - 1, 0)
    while k < len(flat) and starts[k] < hi:
        p, st = flat[k], starts[k]
        a, b = max(lo, st) - st, min(hi, st + p.count) - st
        if b > a:
            out.append(p if (a == 0 and b == p.count) else p.slice(a, b))
        k += 1
    return out


def _merge_runs(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """`(start, count)` element runs in sequence order, merging a run into the previous one when it starts
    inside or right after it.  The sequence need not be monotone (a transposed view visits storage backwards
    and forwards); runs that jump back are kept apart, so nothing is lost."""
    out: list[tuple[int, int]] = []
    for s, c in runs:
        if out and out[-1][0] <= s <= out[-1][0] + out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], s + c - out[-1][0]))
        else:
            out.append((s, c))
    return out

"""Physical view geometry (pure): element runs, leading-axis indexing, view algebra on `ViewDesc`.

Contract used by `memory.py`, `resolver.py` patterns and `fold.py`:

    row_view(v, i)            -> ViewDesc        v[i] along the leading axis (rank-1 lower; a 1-D view gives rank 0)
    rows_view(v, lo, hi)      -> ViewDesc        v[lo:hi] along the leading axis
    flat_view(v)              -> ViewDesc        the same elements as a contiguous 1-D view iff `v` is contiguous, else ValueError
    element_runs(v)           -> list[(start_elem, count)]
                                 the storage elements of `v` in row-major logical order, merged into maximal
                                 contiguous runs (unit inner stride), in *elements of v.dtype* relative to the
                                 allocation start (i.e. including v.offset)
    is_contiguous(v)          -> bool
    same_storage(a, b)        -> bool            same allocation id
    same_element(a, b)        -> bool            one element each, at the same storage offset of one allocation
    byte_extent(v)            -> (lo_byte, hi_byte)   the half-open byte range of the allocation `v` touches

Views never carry values; all functions are total on well-formed descriptors and raise ValueError otherwise.

Conventions:

- Well-formed means non-negative shape entries and non-negative strides (torch never produces negative strides;
  they are rejected with ValueError rather than silently mis-mapped).  Size-1 axes are free: their stride is
  irrelevant to contiguity, runs and extent.
- A rank-0 view is one element at `v.offset`; `row_view` of a rank-1 view yields one.
- Zero-stride (broadcast / `expand`) axes are allowed: `element_runs` then repeats the same storage elements,
  once per logical position, because the runs describe the *logical* element sequence (`element_runs(v)` always
  sums to `v.numel`).  A write through such a view assigns the repeated elements several times (last one wins).
- `element_runs` is canonical: adjacent runs are merged whenever the next run starts exactly where the previous
  one ends, so the result is the greedy maximal-run compression of the storage-element sequence.
- `collapsed_dims(v)` (helper) is the geometry the memory model reads refs through: size-1 axes dropped and
  row-major-contiguous axes merged (the same normalisation the authored frontend applies in
  `verity_vllm/program/frontend/strided_views.py:_strided_refs`), so a `Strided`/`Affine` produced from it has the same
  normal form as authored view refs.
"""

from __future__ import annotations

from verity_vllm.observe.events import ViewDesc


def _check(v: ViewDesc) -> None:
    """ValueError unless `v` is a well-formed descriptor (non-negative shape and strides)."""
    for n in v.shape:
        if n < 0:
            raise ValueError(f"negative shape entry in {v!r}")
    for s in v.strides:
        if s < 0:
            raise ValueError(f"negative stride in {v!r} (not produced by torch; unsupported)")


def collapsed_dims(v: ViewDesc) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """`(shape, strides)` of `v` with size-1 axes dropped and row-major-contiguous neighbours merged.

    Two adjacent axes `(n_o, s_o), (n_i, s_i)` merge into `(n_o * n_i, s_i)` iff `s_o == n_i * s_i` (this
    includes two zero-stride axes).  The result has rank 0 for one element, and never contains a size-1 axis.
    Requires `v.numel > 0` (a size-0 view has no geometry to speak of; callers handle it first)."""
    _check(v)
    dims: list[tuple[int, int]] = []
    for n, st in zip(v.shape, v.strides):
        if n == 1:
            continue
        if n == 0:
            raise ValueError(f"collapsed_dims of an empty view {v!r}")
        if dims and dims[-1][1] == n * st:
            dims[-1] = (dims[-1][0] * n, st)
        else:
            dims.append((n, st))
    return tuple(n for n, _ in dims), tuple(st for _, st in dims)


def row_view(v: ViewDesc, i: int) -> ViewDesc:
    _check(v)
    if v.rank == 0:
        raise ValueError(f"row_view of a rank-0 view {v!r}")
    if not (0 <= i < v.shape[0]):
        raise ValueError(f"row {i} out of range for {v!r}")
    return ViewDesc(v.alloc, v.offset + i * v.strides[0], v.shape[1:], v.strides[1:], v.dtype, v.device)


def rows_view(v: ViewDesc, lo: int, hi: int) -> ViewDesc:
    _check(v)
    if v.rank == 0:
        raise ValueError(f"rows_view of a rank-0 view {v!r}")
    if not (0 <= lo <= hi <= v.shape[0]):
        raise ValueError(f"rows [{lo}, {hi}) out of range for {v!r}")
    return ViewDesc(v.alloc, v.offset + lo * v.strides[0], (hi - lo,) + v.shape[1:], v.strides, v.dtype, v.device)


def flat_view(v: ViewDesc) -> ViewDesc:
    if not is_contiguous(v):
        raise ValueError(f"flat_view of a non-contiguous view {v!r}")
    return ViewDesc(v.alloc, v.offset, (v.numel,), (1,), v.dtype, v.device)


def is_contiguous(v: ViewDesc) -> bool:
    """Row-major with unit inner stride, size-1 axes free (torch's `is_contiguous`); empty views are contiguous."""
    _check(v)
    if v.numel == 0:
        return True
    expected = 1
    for n, st in zip(reversed(v.shape), reversed(v.strides)):
        if n == 1:
            continue
        if st != expected:
            return False
        expected *= n
    return True


def element_runs(v: ViewDesc) -> list[tuple[int, int]]:
    _check(v)
    if v.numel == 0:
        return []
    shape, strides = collapsed_dims(v)
    if not shape:
        return [(v.offset, 1)]
    if strides[-1] == 1:
        count, outer_shape, outer_strides = shape[-1], shape[:-1], strides[:-1]
    else:
        count, outer_shape, outer_strides = 1, shape, strides
    starts = [v.offset]
    for n, st in zip(outer_shape, outer_strides):   # row-major expansion: earlier axes vary slowest
        starts = [s + i * st for s in starts for i in range(n)]
    runs: list[tuple[int, int]] = []
    for s in starts:
        if runs and runs[-1][0] + runs[-1][1] == s:
            runs[-1] = (runs[-1][0], runs[-1][1] + count)
        else:
            runs.append((s, count))
    return runs


def same_storage(a: ViewDesc, b: ViewDesc) -> bool:
    return a.alloc == b.alloc


def same_element(a: ViewDesc, b: ViewDesc) -> bool:
    """Both views are one element of the same allocation at the same storage offset (same element width)."""
    return a.alloc == b.alloc and a.numel == 1 and b.numel == 1 and a.offset == b.offset and a.itemsize == b.itemsize


def byte_extent(v: ViewDesc) -> tuple[int, int]:
    _check(v)
    lo = v.byte_offset
    if v.numel == 0:
        return lo, lo
    last = v.offset + sum((n - 1) * st for n, st in zip(v.shape, v.strides))
    return lo, (last + 1) * v.itemsize

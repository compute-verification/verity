"""Physical view geometry (`verity_vllm.observe.fold.views`) — shapes, transposes, column slices, expands, rank 0."""

from __future__ import annotations

import itertools
import random

import pytest

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.views import (
    byte_extent,
    collapsed_dims,
    element_runs,
    flat_view,
    is_contiguous,
    row_view,
    rows_view,
    same_storage,
)


def V(shape, strides, offset=0, dtype="bfloat16", alloc=1):
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides), dtype, "cuda:0")


def logical_elems(v: ViewDesc) -> list[int]:
    """Brute-force oracle: the storage element of every logical position, row-major."""
    return [v.offset + sum(i * s for i, s in zip(idx, v.strides))
            for idx in itertools.product(*(range(n) for n in v.shape))]


def expand_runs(runs) -> list[int]:
    return [e for s, c in runs for e in range(s, s + c)]


# -- row_view / rows_view -------------------------------------------------------------------------------


def test_row_view_indexes_the_leading_axis():
    v = V((4, 576), (576, 1), offset=10)
    r = row_view(v, 2)
    assert r == V((576,), (1,), offset=10 + 2 * 576)
    assert row_view(V((3, 9, 64), (960, 64, 1), offset=5), 1) == V((9, 64), (64, 1), offset=965)
    with pytest.raises(ValueError):
        row_view(v, 4)
    with pytest.raises(ValueError):
        row_view(v, -1)


def test_row_view_of_rank1_is_rank0():
    v = V((8,), (1,), offset=3, dtype="int32")
    r = row_view(v, 5)
    assert r.shape == () and r.strides == () and r.offset == 8 and r.numel == 1 and r.rank == 0
    assert element_runs(r) == [(8, 1)] and byte_extent(r) == (32, 36)
    with pytest.raises(ValueError):
        row_view(r, 0)


def test_rows_view_slices_the_leading_axis():
    v = V((32, 960), (960, 1))
    s = rows_view(v, 4, 12)
    assert s == V((8, 960), (960, 1), offset=4 * 960)
    assert rows_view(v, 0, 32) == v
    assert rows_view(v, 7, 7).numel == 0 and element_runs(rows_view(v, 7, 7)) == []
    for lo, hi in [(-1, 3), (5, 3), (0, 33)]:
        with pytest.raises(ValueError):
            rows_view(v, lo, hi)
    with pytest.raises(ValueError):
        rows_view(V((), ()), 0, 1)


# -- contiguity / flat ----------------------------------------------------------------------------------


@pytest.mark.parametrize("v,expect", [
    (V((4, 576), (576, 1)), True),
    (V((4, 576), (1, 4)), False),                    # transpose
    (V((32, 192), (960, 1), offset=576), False),     # column slice
    (V((3, 1, 5), (5, 99, 1)), True),                # size-1 axis stride is free
    (V((1, 1), (7, 3)), True),
    (V((), ()), True),
    (V((6,), (1,)), True),
    (V((6,), (2,)), False),
    (V((4, 5), (0, 1)), False),                      # expand
    (V((0, 5), (5, 1)), True),                       # empty
    (V((2, 3), (3, 1), offset=100), True),
])
def test_is_contiguous(v, expect):
    assert is_contiguous(v) is expect


def test_flat_view():
    v = V((3, 1, 5), (5, 99, 1), offset=7)
    assert flat_view(v) == V((15,), (1,), offset=7)
    assert flat_view(V((), (), offset=2)) == V((1,), (1,), offset=2)
    with pytest.raises(ValueError):
        flat_view(V((4, 576), (1, 4)))


# -- element_runs ---------------------------------------------------------------------------------------


def test_runs_contiguous_and_rank0():
    assert element_runs(V((4, 576), (576, 1), offset=8)) == [(8, 2304)]
    assert element_runs(V((), (), offset=8)) == [(8, 1)]
    assert element_runs(V((1, 1, 1), (5, 5, 5), offset=8)) == [(8, 1)]
    assert element_runs(V((1_000_000,), (1,))) == [(0, 1_000_000)]


def test_runs_transpose_is_one_run_per_element():
    t = V((4, 576), (1, 4))          # transpose of a [576, 4] row-major buffer
    runs = element_runs(t)
    assert len(runs) == 4 * 576 and all(c == 1 for _, c in runs)
    assert expand_runs(runs) == logical_elems(t)
    t2 = V((576, 4), (1, 576))       # transpose of a [4, 576] row-major buffer
    assert len(element_runs(t2)) == 2304 and expand_runs(element_runs(t2)) == logical_elems(t2)


def test_runs_qkv_column_slices():
    q = V((32, 576), (960, 1), offset=0)
    k = V((32, 192), (960, 1), offset=576)
    v = V((32, 192), (960, 1), offset=768)
    assert element_runs(q) == [(r * 960, 576) for r in range(32)]
    assert element_runs(k) == [(r * 960 + 576, 192) for r in range(32)]
    assert len(element_runs(v)) == 32 and expand_runs(element_runs(v)) == logical_elems(v)


def test_runs_merge_collapsible_axes_and_adjacent_runs():
    # [2, 2, 2] with strides (5, 2, 1): rows [0,1],[2,3] are adjacent -> merged, then [5,6],[7,8]
    assert element_runs(V((2, 2, 2), (5, 2, 1))) == [(0, 4), (5, 4)]
    # a [3, 9, 64] head view of a [3, 960] buffer: 3 runs of 576
    assert element_runs(V((3, 9, 64), (960, 64, 1))) == [(0, 576), (960, 576), (1920, 576)]
    # the maximal-run property: the next run never starts exactly where the previous one ends
    for v in [V((2, 2, 2), (5, 2, 1)), V((3, 4), (5, 1)), V((6,), (2,)), V((2, 3), (1, 0))]:
        runs = element_runs(v)
        assert all(runs[i][0] + runs[i][1] != runs[i + 1][0] for i in range(len(runs) - 1))
        assert expand_runs(runs) == logical_elems(v)


def test_runs_zero_stride_repeat_elements():
    e = V((3, 4), (0, 1), offset=2)                       # unsqueeze(0).expand(3, 4)
    assert element_runs(e) == [(2, 4), (2, 4), (2, 4)]
    b = V((5,), (0,), offset=9)
    assert element_runs(b) == [(9, 1)] * 5
    assert element_runs(V((2, 3), (1, 0))) == [(0, 1), (0, 1), (0, 2), (1, 1), (1, 1)]
    assert expand_runs(element_runs(V((2, 3), (1, 0)))) == logical_elems(V((2, 3), (1, 0)))
    assert byte_extent(e) == (4, 12) and byte_extent(b) == (18, 20)


def test_runs_count_matches_numel_random():
    rng = random.Random(7)
    for _ in range(300):
        rank = rng.randrange(0, 4)
        shape = tuple(rng.randrange(1, 5) for _ in range(rank))
        strides = tuple(rng.choice([0, 1, 1, 2, 3, 5, 7]) for _ in range(rank))
        v = V(shape, strides, offset=rng.randrange(0, 4))
        runs = element_runs(v)
        assert sum(c for _, c in runs) == v.numel
        assert expand_runs(runs) == logical_elems(v)
        lo, hi = byte_extent(v)
        el = logical_elems(v)
        assert lo == min(el) * 2 and hi == (max(el) + 1) * 2


# -- extent / misc ------------------------------------------------------------------------------------


def test_byte_extent():
    assert byte_extent(V((4, 576), (576, 1))) == (0, 4 * 576 * 2)
    assert byte_extent(V((32, 192), (960, 1), offset=576)) == (576 * 2, (31 * 960 + 576 + 192) * 2)
    assert byte_extent(V((8,), (1,), offset=3, dtype="int32")) == (12, 44)
    assert byte_extent(V((0,), (1,), offset=3)) == (6, 6)
    assert byte_extent(V((), (), offset=3, dtype="float64")) == (24, 32)


def test_collapsed_dims():
    assert collapsed_dims(V((4, 576), (576, 1))) == ((2304,), (1,))
    assert collapsed_dims(V((4, 576), (1, 4))) == ((4, 576), (1, 4))
    assert collapsed_dims(V((3, 1, 9, 64), (960, 7, 64, 1))) == ((3, 576), (960, 1))
    assert collapsed_dims(V((3, 4), (0, 0))) == ((12,), (0,))
    assert collapsed_dims(V((), ())) == ((), ())
    with pytest.raises(ValueError):
        collapsed_dims(V((0, 3), (3, 1)))


def test_negative_strides_and_shapes_rejected():
    bad = V((4,), (-1,), offset=3)
    for fn in (element_runs, is_contiguous, byte_extent, flat_view):
        with pytest.raises(ValueError):
            fn(bad)
    with pytest.raises(ValueError):
        row_view(bad, 0)
    with pytest.raises(ValueError):
        element_runs(V((-1,), (1,)))


def test_same_storage():
    assert same_storage(V((2,), (1,), alloc=3), V((9,), (2,), alloc=3, offset=5))
    assert not same_storage(V((2,), (1,), alloc=3), V((2,), (1,), alloc=4))

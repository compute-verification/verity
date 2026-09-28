"""M2 torture suite for the versioned memory model (`verity_vllm.observe.fold.memory`), fully synthetic.

Every scenario builds `ViewDesc`s by hand over pretend allocations and uses `verity.ml.circuits.refs` objects as the
values the fold would hand the model (`Affine("n", k, 0, count)` = the whole output of body node k, `("p", i, j)`
parameter leaves).  Element-wise checks go through `list(refs)`; structural checks assert the descriptor type
(O(1) `Affine`/`Strided` where one producer is read through a view), never timing.  Gate G1 = this file green.
"""

from __future__ import annotations

import pytest

from verity_vllm.observe.events import StorageAlloc, ViewDesc
from verity_vllm.observe.fold.memory import MemoryModel, Unresolved, compact_refs, flat_parts
from verity_vllm.observe.fold.views import element_runs, row_view
from verity.ml.circuits.refs import Affine, Concat, Explicit, Refs, Strided, concat

DEV = "cuda:0"


def V(alloc, offset, shape, strides, dtype="bfloat16"):
    return ViewDesc(alloc, offset, tuple(shape), tuple(strides), dtype, DEV)


def contig(alloc, shape, dtype="bfloat16", offset=0):
    strides, s = [], 1
    for n in reversed(shape):
        strides.append(s)
        s *= n
    return V(alloc, offset, shape, tuple(reversed(strides)), dtype)


def node(k, count, base=0):
    return Affine("n", k, base, count)


def alloc_ev(alloc, nbytes=1 << 20, generation=0, origin="op"):
    return StorageAlloc(alloc, 0x10000 * (alloc + 1), nbytes, DEV, generation, origin)


def model(*allocs):
    m = MemoryModel()
    for a in allocs:
        m.alloc(alloc_ev(a))
    return m


def seq(r: Refs) -> list:
    return list(r)


# ---------------------------------------------------------------------------------------------------------
# (a) aliasing
# ---------------------------------------------------------------------------------------------------------


def test_a_alias_views_of_one_allocation_read_the_same_refs():
    m = model(1)
    A = contig(1, (4, 576))
    m.write(A, node(1, 2304))
    for B in (V(1, 0, (2304,), (1,)), V(1, 0, (2, 1152), (1152, 1)), V(1, 0, (4, 9, 64), (576, 64, 1)),
              V(1, 0, (1, 4, 576), (99999, 576, 1))):
        r = m.read(B)
        assert isinstance(r, Affine) and (r.space, r.idx, r.base, r.count) == ("n", 1, 0, 2304)
    # a sub-view aliases a slice of the producer
    r = m.read(V(1, 576 * 2, (2, 576), (576, 1)))
    assert isinstance(r, Affine) and (r.base, r.count) == (1152, 1152)
    # a rank-0 alias of one element
    assert seq(m.read(V(1, 700, (), ()))) == [("n", 1, 700)]


# ---------------------------------------------------------------------------------------------------------
# (b) non-contiguous views: transposes
# ---------------------------------------------------------------------------------------------------------


def test_b_transpose_read_is_elementwise_the_producer_leaf_and_o1_in_descriptor():
    m = model(1)
    m.write(contig(1, (4, 576)), node(1, 2304))
    t = V(1, 0, (576, 4), (1, 576))                  # x.t(): logical (i, j) -> storage i + 576 j
    runs = element_runs(t)
    assert len(runs) == 576 * 4 and all(c == 1 for _, c in runs)   # every element its own run (2304, not 576)
    r = m.read(t)
    assert r.count == 2304
    for j in range(2304):
        i, jj = divmod(j, 4)
        assert r[j] == ("n", 1, i + 576 * jj)
    assert isinstance(r, Strided) and (r.inner, r.outer_stride, r.inner_stride) == (4, 1, 576)
    # the other orientation: a [4, 576] view with strides (1, 4) of a [576, 4] producer
    m.write(contig(1, (576, 4)), node(2, 2304))
    t2 = V(1, 0, (4, 576), (1, 4))
    r2 = m.read(t2)
    assert isinstance(r2, Strided) and seq(r2) == [("n", 2, i + 4 * j) for i in range(4) for j in range(576)]


def test_b_transpose_over_two_producers_compacts_per_producer():
    # rows 0..1 from node 1, rows 2..3 from node 2; the transpose interleaves them -> Concat of Strided pieces
    m = model(1)
    x = contig(1, (4, 576))
    m.write(V(1, 0, (2, 576), (576, 1)), node(1, 1152))
    m.write(V(1, 1152, (2, 576), (576, 1)), node(2, 1152))
    t = V(1, 0, (576, 4), (1, 576))
    r = m.read(t)
    for j in range(2304):
        i, jj = divmod(j, 4)
        assert r[j] == (("n", 1, i + 576 * jj) if jj < 2 else ("n", 2, i + 576 * (jj - 2)))
    assert isinstance(r, Concat) and len(r.parts) <= 2 * 576
    assert m.read(x).count == 2304


# ---------------------------------------------------------------------------------------------------------
# (c) QKV column slices
# ---------------------------------------------------------------------------------------------------------


def test_c_qkv_column_slices():
    m = model(1)
    qkv = contig(1, (32, 960))
    m.write(qkv, node(5, 32 * 960))
    q = V(1, 0, (32, 576), (960, 1))
    k = V(1, 576, (32, 192), (960, 1))
    v = V(1, 768, (32, 192), (960, 1))
    rq, rk, rv = m.read(q), m.read(k), m.read(v)
    assert seq(rq) == [("n", 5, r * 960 + c) for r in range(32) for c in range(576)]
    assert seq(rk) == [("n", 5, r * 960 + 576 + c) for r in range(32) for c in range(192)]
    assert seq(rv) == [("n", 5, r * 960 + 768 + c) for r in range(32) for c in range(192)]
    for r in (rq, rk, rv):
        assert isinstance(r, Strided) and (r.inner_stride, r.outer_stride) == (1, 960)
    # per-row reads of the slices are single Affines (what a per-row instance consumes)
    rk3 = m.read(row_view(k, 3))
    assert isinstance(rk3, Affine) and (rk3.base, rk3.count) == (3 * 960 + 576, 192)
    # the head view q as [32, 9, 64] of the same bytes
    rq_heads = m.read(V(1, 0, (32, 9, 64), (960, 64, 1)))
    assert seq(rq_heads) == seq(rq)


def test_c_qkv_written_per_row_then_sliced():
    # the fold writes one Gemm instance per row: row r of qkv := node 100+r
    m = model(1)
    qkv = contig(1, (8, 960))
    for r in range(8):
        m.write(row_view(qkv, r), node(100 + r, 960))
    k = V(1, 576, (8, 192), (960, 1))
    rk = m.read(k)
    assert seq(rk) == [("n", 100 + r, 576 + c) for r in range(8) for c in range(192)]
    assert isinstance(rk, Concat) and len(rk.parts) == 8
    assert all(isinstance(p, Affine) for p in rk.parts)
    r5 = m.read(row_view(k, 5))
    assert isinstance(r5, Affine) and (r5.idx, r5.base, r5.count) == (105, 576, 192)


# ---------------------------------------------------------------------------------------------------------
# (d) in-place fused residual: operands captured before the writes are immutable
# ---------------------------------------------------------------------------------------------------------


def test_d_in_place_fused_residual_operands_are_immutable():
    m = model(1, 2)
    x, res = contig(1, (4, 576)), contig(2, (4, 576))
    m.write(x, node(1, 2304))          # p1
    m.write(res, node(2, 2304))        # p2
    # the Compute reads both operands first ...
    op_x, op_res = m.read(x), m.read(res)
    assert seq(op_x) == seq(node(1, 2304)) and seq(op_res) == seq(node(2, 2304))
    # ... the fold emits the instance (nodes 3, 4) and applies the in-place writes
    m.write(x, node(3, 2304))
    m.write(res, node(4, 2304))
    assert seq(m.read(x)) == seq(node(3, 2304))
    assert seq(m.read(res)) == seq(node(4, 2304))
    # the operand refs captured before the writes still name p1 / p2
    assert seq(op_x) == seq(node(1, 2304)) and seq(op_res) == seq(node(2, 2304))
    assert op_x[0] == ("n", 1, 0) and op_res[-1] == ("n", 2, 2303)


def test_d_in_place_per_row_norm_then_rows_read_new_producers():
    m = model(1)
    x = contig(1, (3, 576))
    m.write(x, node(1, 3 * 576))
    before = [m.read(row_view(x, i)) for i in range(3)]
    for i in range(3):
        m.write(row_view(x, i), node(10 + i, 576))
    for i in range(3):
        assert seq(before[i]) == [("n", 1, i * 576 + c) for c in range(576)]
        assert seq(m.read(row_view(x, i))) == [("n", 10 + i, c) for c in range(576)]
    full = m.read(x)
    assert isinstance(full, Concat) and len(full.parts) == 3


# ---------------------------------------------------------------------------------------------------------
# (e) partial overlapping writes
# ---------------------------------------------------------------------------------------------------------


def test_e_partial_overlapping_writes_split_the_old_interval():
    m = model(1)
    buf = contig(1, (100,), dtype="float32")
    m.write(buf, node(1, 100))                                        # A
    m.write(V(1, 50, (30,), (1,), "float32"), node(2, 30))           # B over [50, 80)
    r0 = m.read(V(1, 0, (50,), (1,), "float32"))
    r1 = m.read(V(1, 50, (30,), (1,), "float32"))
    r2 = m.read(V(1, 80, (20,), (1,), "float32"))
    assert isinstance(r0, Affine) and (r0.idx, r0.base, r0.count) == (1, 0, 50)
    assert isinstance(r1, Affine) and (r1.idx, r1.base, r1.count) == (2, 0, 30)
    assert isinstance(r2, Affine) and (r2.idx, r2.base, r2.count) == (1, 80, 20)
    full = m.read(buf)
    assert seq(full) == seq(node(1, 50)) + seq(node(2, 30)) + seq(node(1, 100, 80).slice(0, 20))
    assert isinstance(full, Concat) and len(full.parts) == 3
    assert m.stats().intervals == 3
    # a straddling read of [40, 60) is A[40:50] + B[0:10]
    assert seq(m.read(V(1, 40, (20,), (1,), "float32"))) == [("n", 1, j) for j in range(40, 50)] + [("n", 2, j) for j in range(10)]
    # overwrite that covers several intervals collapses them back to one
    m.write(V(1, 10, (85,), (1,), "float32"), node(3, 85))
    assert m.stats().intervals == 3
    assert seq(m.read(buf)) == seq(node(1, 10)) + seq(node(3, 85)) + [("n", 1, j) for j in range(95, 100)]


def test_e_overlapping_writes_through_strided_views():
    m = model(1)
    buf = contig(1, (10, 8), dtype="float32")
    m.write(buf, node(1, 80))
    col = V(1, 3, (10,), (8,), "float32")           # column 3
    m.write(col, node(2, 10))
    r = m.read(buf)
    exp = [("n", 2, i) if c == 3 else ("n", 1, i * 8 + c) for i in range(10) for c in range(8)]
    assert seq(r) == exp
    assert m.stats().intervals == 21
    rc = m.read(col)
    assert seq(rc) == seq(node(2, 10)) and isinstance(rc, Affine)      # ten single-element intervals re-merge
    # the same column of the old producer is a Strided of node 1
    rc2 = m.read(V(1, 2, (10,), (8,), "float32"))
    assert isinstance(rc2, Strided) and seq(rc2) == [("n", 1, i * 8 + 2) for i in range(10)]


# ---------------------------------------------------------------------------------------------------------
# (f) move-before-overwrite
# ---------------------------------------------------------------------------------------------------------


def test_f_move_then_overwrite_source_keeps_old_refs_in_destination():
    m = model(1, 2)
    src, dst = contig(1, (64,)), contig(2, (64,))
    m.write(src, node(1, 64))                 # A
    m.move(src, dst)
    m.write(src, node(3, 64))                 # C
    assert seq(m.read(dst)) == seq(node(1, 64))
    assert seq(m.read(src)) == seq(node(3, 64))
    assert isinstance(m.read(dst), Affine)
    # move within one allocation (copy_ between two disjoint regions)
    m.move(V(1, 0, (32,), (1,)), V(1, 32, (32,), (1,)))
    assert seq(m.read(V(1, 32, (32,), (1,)))) == [("n", 3, j) for j in range(32)]
    assert m.stats().moves == 2


def test_f_move_checks_shape_and_width():
    m = model(1, 2)
    m.write(contig(1, (64,)), node(1, 64))
    with pytest.raises(ValueError):
        m.move(contig(1, (64,)), contig(2, (32,)))
    with pytest.raises(ValueError):
        m.move(contig(1, (64,)), contig(2, (64,), dtype="float32"))
    # same width, different dtype is a bit-preserving move
    m.move(contig(1, (64,)), contig(2, (64,), dtype="int16"))
    assert seq(m.read(contig(2, (64,), dtype="int16"))) == seq(node(1, 64))
    with pytest.raises(Unresolved):
        m.move(contig(2, (64,), dtype="int16", offset=64), contig(1, (64,)))


# ---------------------------------------------------------------------------------------------------------
# (g) allocation reuse: same id, new generation
# ---------------------------------------------------------------------------------------------------------


def test_g_new_generation_makes_contents_unreachable():
    m = MemoryModel()
    m.alloc(StorageAlloc(7, 0xdead, 4096, DEV, 0, "op"))
    buf = contig(7, (1024,))
    m.write(buf, node(1, 1024))
    assert seq(m.read(buf)) == seq(node(1, 1024))
    m.alloc(StorageAlloc(7, 0xdead, 4096, DEV, 1, "op"))
    with pytest.raises(Unresolved) as ei:
        m.read(buf)
    assert ei.value.missing == [(0, 1024)] and ei.value.reason == "unwritten"
    assert m.coverage(buf) == 0.0
    m.write(buf, node(2, 1024))
    assert seq(m.read(buf)) == seq(node(2, 1024))
    # re-announcing the same generation keeps the contents
    m.alloc(StorageAlloc(7, 0xdead, 4096, DEV, 1, "op"))
    assert seq(m.read(buf)) == seq(node(2, 1024))
    st = m.stats()
    assert st.allocations == 1 and st.live_allocations == 1 and st.extra["generation_drops"] == 1


def test_g_free_and_unknown_allocations():
    m = model(1)
    buf = contig(1, (16,))
    m.write(buf, node(1, 16))
    m.free(1)
    with pytest.raises(Unresolved) as ei:
        m.read(buf)
    assert ei.value.reason == "freed" and ei.value.missing == [(0, 16)]
    with pytest.raises(Unresolved) as ei:
        m.read(contig(42, (16,)))
    assert ei.value.reason == "unknown"
    assert m.coverage(contig(42, (16,))) == 0.0
    # a write into an unannounced allocation is accepted (a fact) and counted
    m.write(contig(42, (16,)), node(2, 16))
    assert seq(m.read(contig(42, (16,)))) == seq(node(2, 16))
    assert m.stats().extra["implicit_allocs"] == 1 and m.stats().live_allocations == 1
    with pytest.raises(Unresolved):
        m.read_rows(contig(1, (4, 4)), [0])


# ---------------------------------------------------------------------------------------------------------
# (h) KV-slot reuse
# ---------------------------------------------------------------------------------------------------------


def test_h_kv_slots_move_read_rows_and_overwrite():
    KVH, D, BS, NB = 3, 64, 16, 4
    m = model(1, 2)
    cache = contig(1, (NB * BS, KVH, D))             # [slots, KVH, D]
    k = contig(2, (3, KVH, D))                        # the step's k rows (after RoPE), one producer per row
    for i in range(3):
        m.write(row_view(k, i), node(10 + i, KVH * D))
    slots = (5, 21, 37)
    for i, s in enumerate(slots):                      # Transparent(move, src=PerRow(k), dst=Gather(cache, slots))
        m.move(row_view(k, i), row_view(cache, s))
    r = m.read_rows(cache, list(slots))
    assert seq(r) == seq(node(10, 192)) + seq(node(11, 192)) + seq(node(12, 192))
    assert isinstance(r, Concat) and len(r.parts) == 3 and all(isinstance(p, Affine) for p in r.parts)
    # a slot that was never filled is Unresolved, reporting exactly that slot's run
    with pytest.raises(Unresolved) as ei:
        m.read_rows(cache, [5, 6])
    assert ei.value.missing == [(6 * KVH * D, KVH * D)] and ei.value.view == row_view(cache, 6)
    # slot 21 is reused by a later step
    m.write(row_view(cache, 21), node(20, KVH * D))
    r2 = m.read_rows(cache, list(slots))
    assert seq(r2) == seq(node(10, 192)) + seq(node(20, 192)) + seq(node(12, 192))
    # GatherMany for a request whose positions 0..2 live in slots (5, 21, 37): the same thing
    assert seq(m.read_rows(cache, slots)) == seq(r2)
    # a single KV head across the gathered slots: [KVH, D] rows -> head 1 is Strided per row
    head1 = [m.read(V(1, s * KVH * D + 1 * D, (D,), (1,))) for s in slots]
    assert [seq(h) for h in head1] == [[("n", 10, D + j) for j in range(D)], [("n", 20, D + j) for j in range(D)],
                                       [("n", 12, D + j) for j in range(D)]]


def test_h_kv_slots_from_one_producer_compact_to_strided_or_affine():
    KVH, D = 3, 64
    m = model(1, 2)
    cache = contig(1, (64, KVH, D))
    k = contig(2, (4, KVH, D))
    m.write(k, node(1, 4 * KVH * D))                 # one producer for all four rows
    for i, s in enumerate((8, 9, 10, 11)):           # consecutive slots -> one contiguous region -> Affine
        m.move(row_view(k, i), row_view(cache, s))
    r = m.read_rows(cache, [8, 9, 10, 11])
    assert isinstance(r, Affine) and (r.idx, r.base, r.count) == (1, 0, 4 * KVH * D)
    for i, s in enumerate((20, 22, 24, 26)):         # equally spaced slots -> the rows are a Strided of node 1
        m.move(row_view(k, i), row_view(cache, s))
    r = m.read_rows(cache, [20, 22, 24, 26])
    assert seq(r) == seq(node(1, 4 * KVH * D))
    assert isinstance(r, Affine)                      # the moved rows are node 1 rows 0..3 in order: contiguous leaves
    r = m.read_rows(cache, [26, 24, 22, 20])
    assert isinstance(r, Strided) and seq(r) == [("n", 1, (3 - q) * 192 + j) for q in range(4) for j in range(192)]


# ---------------------------------------------------------------------------------------------------------
# (i) width discipline
# ---------------------------------------------------------------------------------------------------------


def test_i_width_discipline():
    m = model(1)
    m.write(contig(1, (64,)), node(1, 64))                         # bf16
    as_i32 = contig(1, (32,), dtype="int32")
    with pytest.raises(Unresolved) as ei:
        m.read(as_i32)
    assert ei.value.reason == "width" and ei.value.missing == [(0, 32)]
    assert m.coverage(as_i32) == 0.0
    with pytest.raises(Unresolved) as ei:                          # a narrower view of a wider write, too
        m.read(contig(1, (64,), dtype="int8"))
    assert ei.value.reason == "width"
    # same width, different dtype
    m.write(contig(1, (32,), dtype="int32"), node(2, 32))
    assert seq(m.read(contig(1, (32,), dtype="uint32"))) == seq(node(2, 32))
    assert seq(m.read(contig(1, (32,), dtype="float32"))) == seq(node(2, 32))
    strict = MemoryModel(allow_same_width_reinterpret=False)
    strict.alloc(alloc_ev(1))
    strict.write(contig(1, (32,), dtype="int32"), node(2, 32))
    with pytest.raises(Unresolved) as ei:
        strict.read(contig(1, (32,), dtype="uint32"))
    assert ei.value.reason == "dtype"
    assert seq(strict.read(contig(1, (32,), dtype="int32"))) == seq(node(2, 32))


def test_i_narrow_write_into_wide_element_clobbers_only_that_element():
    m = model(1)
    m.write(contig(1, (8,), dtype="int32"), node(1, 8))
    m.write(V(1, 5, (2,), (1,), "bfloat16"), node(2, 2))          # bytes [10, 14): element 2 (bytes 8..12) and 3 (12..16)
    with pytest.raises(Unresolved) as ei:
        m.read(contig(1, (8,), dtype="int32"))
    # int32 elements 2 and 3 are gone: their outer half-bytes are unwritten (clobbered), the middle is bf16
    assert ei.value.missing == [(2, 2)] and ei.value.reason == "unwritten+width"
    assert seq(m.read(V(1, 0, (2,), (1,), "int32"))) == [("n", 1, 0), ("n", 1, 1)]
    assert seq(m.read(V(1, 4, (4,), (1,), "int32"))) == [("n", 1, j) for j in range(4, 8)]
    assert seq(m.read(V(1, 5, (2,), (1,), "bfloat16"))) == seq(node(2, 2))
    # the half-elements left over on both sides are unwritten for a bf16 reader
    with pytest.raises(Unresolved) as ei:
        m.read(V(1, 4, (4,), (1,), "bfloat16"))
    assert ei.value.missing == [(4, 1), (7, 1)] and ei.value.reason == "unwritten"
    assert m.coverage(V(1, 4, (4,), (1,), "bfloat16")) == 0.5
    assert m.coverage(contig(1, (8,), dtype="int32")) == 0.75


def test_i_mixed_width_neighbours_resolve_per_grid():
    m = model(1)
    m.write(V(1, 1, (4,), (1,), "int32"), node(1, 4))              # bytes [4, 20)
    m.write(V(1, 0, (2,), (1,), "int16"), node(2, 2))              # bytes [0, 4)
    m.write(V(1, 10, (2,), (1,), "int16"), node(3, 2))             # bytes [20, 24)
    assert seq(m.read(V(1, 1, (4,), (1,), "int32"))) == seq(node(1, 4))
    assert seq(m.read(V(1, 0, (2,), (1,), "int16"))) == seq(node(2, 2))
    assert seq(m.read(V(1, 10, (2,), (1,), "int16"))) == seq(node(3, 2))
    # an int16 reader spanning all three regions: only the int32 span [4, 20) = int16 elements 2..9 is unresolvable
    with pytest.raises(Unresolved) as ei:
        m.read(V(1, 0, (12,), (1,), "int16"))
    assert ei.value.reason == "width" and ei.value.missing == [(2, 8)]
    # an int32 reader over the int16 ends: elements 0 and 5
    with pytest.raises(Unresolved) as ei:
        m.read(V(1, 0, (6,), (1,), "int32"))
    assert ei.value.reason == "width" and ei.value.missing == [(0, 1), (5, 1)]
    assert m.coverage(V(1, 0, (6,), (1,), "int32")) == pytest.approx(4 / 6)


# ---------------------------------------------------------------------------------------------------------
# (j) Unresolved reports the exact missing runs
# ---------------------------------------------------------------------------------------------------------


def test_j_unresolved_reports_exact_missing_runs():
    m = model(1)
    f32 = lambda off, n: V(1, off, (n,), (1,), "float32")
    m.write(f32(10, 10), node(1, 10))
    m.write(f32(40, 20), node(2, 20))
    with pytest.raises(Unresolved) as ei:
        m.read(f32(0, 100))
    assert ei.value.missing == [(0, 10), (20, 20), (60, 40)] and ei.value.reason == "unwritten"
    assert ei.value.view == f32(0, 100)
    with pytest.raises(Unresolved) as ei:
        m.read(f32(15, 30))
    assert ei.value.missing == [(20, 20)]
    assert m.coverage(f32(15, 30)) == pytest.approx(10 / 30)
    assert m.coverage(f32(0, 100)) == pytest.approx(0.30)
    # a 2-D view: missing runs are per row, in logical order
    with pytest.raises(Unresolved) as ei:
        m.read(V(1, 0, (4, 25), (25, 1), "float32"))
    assert ei.value.missing == [(0, 10), (20, 20), (60, 40)]
    # a strided view: single-element gaps in logical order
    with pytest.raises(Unresolved) as ei:
        m.read(V(1, 0, (6,), (9,), "float32"))       # elements 0, 9, 18, 27, 36, 45
    assert ei.value.missing == [(0, 1), (9, 1), (27, 1), (36, 1)]
    assert "missing element runs" in str(ei.value)
    # the model still answers the resolvable part
    assert seq(m.read(f32(10, 10))) == seq(node(1, 10))
    # a mixture of reasons is reported as such
    m.write(V(1, 0, (4,), (1,), "int16"), node(3, 4))              # bytes [0, 8) = float32 elements 0..1
    with pytest.raises(Unresolved) as ei:
        m.read(f32(0, 12))
    assert ei.value.reason == "unwritten+width" and ei.value.missing == [(0, 10)]


# ---------------------------------------------------------------------------------------------------------
# (k) descriptor compactness (types, not timing)
# ---------------------------------------------------------------------------------------------------------


def test_k_large_contiguous_write_read_is_one_affine():
    m = model(1)
    big = contig(1, (1_000_000,))
    m.write(big, node(1, 1_000_000))
    r = m.read(big)
    assert isinstance(r, Affine) and (r.base, r.count) == (0, 1_000_000)
    assert m.stats().intervals == 1
    r2 = m.read(V(1, 1000, (500, 1000), (1000, 1)))
    assert isinstance(r2, Affine) and (r2.base, r2.count) == (1000, 500_000)
    r3 = m.read(V(1, 0, (1000, 500), (1000, 1)))          # a column block: one Strided
    assert isinstance(r3, Strided) and r3.count == 500_000


def test_k_thousand_row_writes_then_full_read_is_compact():
    m = model(1)
    buf = contig(1, (1000, 64))
    for i in range(1000):
        m.write(row_view(buf, i), node(1000 + i, 64))
    r = m.read(buf)
    assert isinstance(r, Concat) and len(r.parts) <= 1000
    assert r[64 * 517 + 3] == ("n", 1517, 3) and r.count == 64_000
    assert flat_parts(r) == r.parts and all(isinstance(p, Affine) for p in r.parts)
    # rows written by one producer in order re-merge into a single Affine on read ("or better")
    m2 = model(1)
    for i in range(1000):
        m2.write(row_view(buf, i), Affine("n", 1, 64 * i, 64))
    assert m2.stats().intervals == 1000
    r2 = m2.read(buf)
    assert isinstance(r2, Affine) and (r2.base, r2.count) == (0, 64_000)
    # a whole-buffer move of the 1000-part value keeps intervals primitive, so row reads stay O(1) slices
    m.alloc(alloc_ev(2))
    m.move(buf, contig(2, (1000, 64)))
    assert m.stats().intervals == 2000
    row = m.read(row_view(contig(2, (1000, 64)), 999))
    assert isinstance(row, Affine) and (row.idx, row.base, row.count) == (1999, 0, 64)


def test_k_strided_and_explicit_refs_survive_write_and_slice():
    m = model(1)
    s = Strided("n", 4, 0, 8, 100, 1, 64)             # 8 rows of 8 leaves at stride 100
    m.write(contig(1, (64,)), s)
    r = m.read(V(1, 8, (8,), (1,)))                   # row 1 of the strided value
    assert isinstance(r, Strided | Affine) and seq(r) == [("n", 4, 100 + j) for j in range(8)]
    e = Explicit(tuple(("p", 3, j * j) for j in range(16)))
    m.write(V(1, 64, (16,), (1,)), e)
    assert seq(m.read(V(1, 70, (4,), (1,)))) == [("p", 3, j * j) for j in range(6, 10)]
    full = m.read(contig(1, (80,)))
    assert seq(full) == seq(s) + seq(e)


# ---------------------------------------------------------------------------------------------------------
# (l) rank-0 views
# ---------------------------------------------------------------------------------------------------------


def test_l_rank0_token_slot_write_and_read():
    m = model(1)
    ids = contig(1, (8,), dtype="int32")               # input_ids of a step
    slot3 = row_view(ids, 3)
    assert slot3.shape == () and slot3.numel == 1
    m.write(slot3, Affine("p", 1, 3, 1))                # prompt token leaf
    m.write(V(1, 4, (), (), "int32"), Explicit((("n", 77, 0),)))   # a StepLink'd sampled token
    r = m.read(slot3)
    assert r.count == 1 and seq(r) == [("p", 1, 3)] and isinstance(r, Affine)
    assert seq(m.read(row_view(ids, 4))) == [("n", 77, 0)]
    with pytest.raises(Unresolved) as ei:
        m.read(row_view(ids, 5))
    assert ei.value.missing == [(5, 1)]
    with pytest.raises(Unresolved) as ei:
        m.read(ids)
    assert ei.value.missing == [(0, 3), (5, 3)]
    with pytest.raises(ValueError):
        m.write(slot3, node(1, 2))
    # PerRow over a rank-1 view = one rank-0 read per row
    m.write(ids, Affine("p", 2, 0, 8))
    assert seq(m.read_rows(ids, [7, 0, 3])) == [("p", 2, 7), ("p", 2, 0), ("p", 2, 3)]


# ---------------------------------------------------------------------------------------------------------
# realistic aliasing: RoPE in place on the q/k head views of the qkv buffer
# ---------------------------------------------------------------------------------------------------------


def test_rope_in_place_on_head_views_of_qkv():
    M, NH, KVH, D = 4, 9, 3, 64
    m = model(1)
    qkv = contig(1, (M, 960))
    for r in range(M):
        m.write(row_view(qkv, r), node(100 + r, 960))            # Gemm rows
    q = V(1, 0, (M, NH, D), (960, D, 1))
    k = V(1, NH * D, (M, KVH, D), (960, D, 1))
    v = V(1, (NH + KVH) * D, (M, KVH, D), (960, D, 1))
    q_in = [m.read(row_view(q, r)) for r in range(M)]
    k_in = [m.read(row_view(k, r)) for r in range(M)]
    for r in range(M):
        assert isinstance(q_in[r], Affine) and (q_in[r].idx, q_in[r].base, q_in[r].count) == (100 + r, 0, NH * D)
        assert isinstance(k_in[r], Affine) and (k_in[r].idx, k_in[r].base, k_in[r].count) == (100 + r, NH * D, KVH * D)
    for r in range(M):                                            # RoPE instances write q and k rows in place
        m.write(row_view(q, r), node(200 + r, NH * D))
        m.write(row_view(k, r), node(300 + r, KVH * D))
    for r in range(M):
        assert seq(m.read(row_view(q, r))) == seq(node(200 + r, NH * D))
        assert seq(m.read(row_view(k, r))) == seq(node(300 + r, KVH * D))
        assert seq(m.read(row_view(v, r))) == [("n", 100 + r, (NH + KVH) * D + j) for j in range(KVH * D)]
    assert m.stats().intervals == 3 * M
    full = m.read(qkv)
    assert isinstance(full, Concat) and len(full.parts) == 3 * M
    # the KV head 1 of k row 2 is a plain Affine of the RoPE output
    h = m.read(V(1, 2 * 960 + NH * D + D, (D,), (1,)))
    assert isinstance(h, Affine) and (h.idx, h.base) == (302, D)


# ---------------------------------------------------------------------------------------------------------
# broadcast / expand views and stats
# ---------------------------------------------------------------------------------------------------------


def test_expand_views_read_repeated_refs_and_write_last_wins():
    m = model(1)
    m.write(contig(1, (4,)), node(1, 4))
    e = V(1, 0, (3, 4), (0, 1))                                    # expand(3, 4)
    r = m.read(e)
    assert seq(r) == seq(node(1, 4)) * 3
    assert isinstance(r, Strided) and (r.inner, r.outer_stride, r.inner_stride) == (4, 0, 1)
    b = V(1, 2, (5,), (0,))
    rb = m.read(b)
    assert seq(rb) == [("n", 1, 2)] * 5 and isinstance(rb, Strided) and (rb.outer_stride, rb.inner_stride) == (0, 0)
    m.write(b, Explicit(tuple(("n", 9, j) for j in range(5))))    # element 2 := last assignment
    assert seq(m.read(contig(1, (4,)))) == [("n", 1, 0), ("n", 1, 1), ("n", 9, 4), ("n", 1, 3)]


def test_stats_and_empty_views():
    m = model(1)
    st = m.stats()
    assert (st.allocations, st.live_allocations, st.writes, st.reads, st.intervals) == (1, 1, 0, 0, 0)
    empty = V(1, 0, (0, 64), (64, 1))
    m.write(empty, Explicit(()))
    assert m.read(empty).count == 0 and m.coverage(empty) == 1.0
    m.write(contig(1, (8,)), node(1, 8))
    m.read(contig(1, (8,)))
    with pytest.raises(Unresolved):
        m.read(contig(1, (16,)))
    st = m.stats()
    assert st.writes == 2 and st.reads == 3 and st.intervals == 1 and st.extra["unresolved"] == 1
    assert m.intervals(1) == [(0, 16, m.intervals(1)[0][2], 16, "bfloat16")]
    assert m.intervals(99) == []


def test_compact_refs_preserves_sequences():
    parts = [Affine("n", 1, 0, 4), Affine("n", 1, 4, 4), Affine("n", 1, 100, 4), Affine("n", 1, 200, 4), Affine("n", 1, 300, 4),
             Affine("n", 2, 0, 1), Affine("n", 2, 5, 1), Affine("n", 2, 10, 1), Affine("n", 3, 7, 1), Affine("n", 3, 7, 1),
             Explicit((("p", 0, 1), ("p", 0, 9)))]
    r = compact_refs(parts)
    assert seq(r) == seq(concat(parts))
    kinds = [type(p).__name__ for p in flat_parts(r)]
    assert kinds == ["Affine", "Strided", "Strided", "Strided", "Explicit"]
    assert flat_parts(r)[2].outer_stride == 5 and flat_parts(r)[3].outer_stride == 0

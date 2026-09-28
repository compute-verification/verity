"""Element-by-element oracle for the memory model: for random small shapes / strides / offsets,
`read(view)[j]` equals the ref most recently written to the storage element logical position `j` maps to.

The oracle is a plain dict `storage element -> ref` (single dtype) or `storage byte -> (write, element, byte
in element, width, ref)` (mixed widths); it knows nothing about intervals, runs or refs compaction, so it
checks the whole chain `element_runs` -> interval splitting -> slicing -> `compact_refs` on the read side.
"""

from __future__ import annotations

import itertools
import random

import pytest

from verity_vllm.observe.events import DTYPE_BYTES, StorageAlloc, ViewDesc
from verity_vllm.observe.fold.memory import MemoryModel, Unresolved
from verity_vllm.observe.fold.views import row_view
from verity.ml.circuits.refs import Affine, Explicit, Refs, Strided

DEV = "cuda:0"


def logical_elems(v: ViewDesc) -> list[int]:
    return [v.offset + sum(i * s for i, s in zip(idx, v.strides))
            for idx in itertools.product(*(range(n) for n in v.shape))]


def rand_view(rng: random.Random, alloc: int, n_elems: int, dtype: str, shape=None) -> ViewDesc:
    for _ in range(1000):
        if shape is None:
            rank = rng.randrange(0, 4)
            sh = tuple(rng.randrange(1, 5) for _ in range(rank))
        else:
            sh = shape
        strides = tuple(rng.choice([0, 1, 1, 1, 2, 3, 4, 7]) for _ in sh)
        span = sum((n - 1) * s for n, s in zip(sh, strides))
        if span < n_elems:
            return ViewDesc(alloc, rng.randrange(0, n_elems - span), sh, strides, dtype, DEV)
    raise RuntimeError("no view fits")


def rand_refs(rng: random.Random, k: int, count: int) -> Refs:
    c = rng.randrange(3)
    if c == 0:
        return Affine("n", k, rng.randrange(0, 50), count)
    if c == 1:
        return Explicit(tuple(("p", rng.randrange(3), rng.randrange(40)) for _ in range(count)))
    inner = rng.randrange(1, 4)
    return Strided("n", k, rng.randrange(10), inner, rng.randrange(0, 6), rng.choice([0, 1, 2]), count)


def check_read(m: MemoryModel, v: ViewDesc, expected: list) -> None:
    """`expected[j]` is the oracle ref of logical element j or None when unresolvable."""
    missing = [e for e, r in zip(logical_elems(v), expected) if r is None]
    if not missing:
        got = m.read(v)
        assert got.count == v.numel
        assert list(got) == expected
        assert [got[j] for j in range(got.count)] == expected           # random access agrees with iteration
        assert list(m.read(v)) == expected                              # reads do not mutate
        assert m.coverage(v) == 1.0
        return
    with pytest.raises(Unresolved) as ei:
        m.read(v)
    runs = ei.value.missing
    reported = [e for s, c in runs for e in range(s, s + c)]
    assert set(reported) == set(missing), (runs, missing)
    assert runs[0][0] == missing[0]                                     # logical order
    assert all(c > 0 for _, c in runs)
    assert m.coverage(v) == pytest.approx(1 - len(missing) / v.numel)


def test_single_dtype_random_program_matches_oracle():
    rng = random.Random(12345)
    N = 40
    for trial in range(60):
        dtype = rng.choice(["bfloat16", "float32", "int64"])
        m = MemoryModel()
        m.alloc(StorageAlloc(1, 0x1000, N * DTYPE_BYTES[dtype], DEV, 0, "op"))
        m.alloc(StorageAlloc(2, 0x2000, N * DTYPE_BYTES[dtype], DEV, 0, "op"))
        storage: dict[tuple[int, int], tuple] = {}                     # (alloc, elem) -> ref
        k = 0
        for _ in range(40):
            op = rng.random()
            alloc = rng.choice([1, 2])
            if op < 0.45:
                v = rand_view(rng, alloc, N, dtype)
                k += 1
                refs = rand_refs(rng, k, v.numel)
                m.write(v, refs)
                for j, e in enumerate(logical_elems(v)):
                    storage[(alloc, e)] = refs[j]
            elif op < 0.85:
                v = rand_view(rng, alloc, N, dtype)
                check_read(m, v, [storage.get((alloc, e)) for e in logical_elems(v)])
            else:
                src = rand_view(rng, alloc, N, dtype)
                dalloc = rng.choice([1, 2])
                dst = rand_view(rng, dalloc, N, dtype, shape=src.shape)
                exp = [storage.get((alloc, e)) for e in logical_elems(src)]
                if any(r is None for r in exp):
                    with pytest.raises(Unresolved):
                        m.move(src, dst)
                    continue
                m.move(src, dst)
                for j, e in enumerate(logical_elems(dst)):
                    storage[(dalloc, e)] = exp[j]
        # final sweep: every element and a few random views
        for alloc in (1, 2):
            full = ViewDesc(alloc, 0, (N,), (1,), dtype, DEV)
            check_read(m, full, [storage.get((alloc, e)) for e in range(N)])
            for _ in range(5):
                v = rand_view(rng, alloc, N, dtype)
                check_read(m, v, [storage.get((alloc, e)) for e in logical_elems(v)])


def test_read_rows_matches_per_row_reads():
    rng = random.Random(7)
    N = 64
    for _ in range(30):
        m = MemoryModel()
        m.alloc(StorageAlloc(1, 0x1000, N * 2, DEV, 0, "op"))
        storage: dict[int, tuple] = {}
        for k in range(rng.randrange(1, 6)):
            v = rand_view(rng, 1, N, "bfloat16")
            refs = Affine("n", k, 0, v.numel)
            m.write(v, refs)
            for j, e in enumerate(logical_elems(v)):
                storage[e] = refs[j]
        rows = rng.randrange(1, 5)
        v = rand_view(rng, 1, N, "bfloat16", shape=(rows,) + tuple(rng.randrange(1, 4) for _ in range(rng.randrange(0, 3))))
        idx = [rng.randrange(rows) for _ in range(rng.randrange(1, 6))]
        exp = [storage.get(e) for i in idx for e in logical_elems(row_view(v, i))]
        if any(r is None for r in exp):
            with pytest.raises(Unresolved):
                m.read_rows(v, idx)
        else:
            assert list(m.read_rows(v, idx)) == exp


def test_read_rows_cache_is_invalidated_by_overlapping_writes_frees_and_generations():
    """[FOLD-01] `read_rows` memoises gathered rows per allocation; the memo must be invisible: after any interleaving of
    writes (partial rows, whole rows, strided views cutting through rows), frees, re-allocations and generation changes,
    a repeated gather equals the per-row oracle exactly (and a gather over a stale-then-rewritten row sees the rewrite)."""
    rng = random.Random(2026)
    N = 96
    for trial in range(80):
        m = MemoryModel()
        gen = 0
        m.alloc(StorageAlloc(1, 0x1000, N * 2, DEV, gen, "op"))
        storage: dict[int, tuple] = {}
        rows = rng.randrange(2, 7)
        inner = tuple(rng.randrange(1, 4) for _ in range(rng.randrange(0, 3)))
        views = [rand_view(rng, 1, N, "bfloat16", shape=(rows,) + inner) for _ in range(rng.randrange(1, 3))]
        k = 0
        for _ in range(60):
            op = rng.random()
            if op < 0.35:                                   # write: random view (may cut through gathered rows)
                v = rand_view(rng, 1, N, "bfloat16")
                k += 1
                refs = rand_refs(rng, k, v.numel)
                m.write(v, refs)
                for j, e in enumerate(logical_elems(v)):
                    storage[e] = refs[j]
            elif op < 0.45:                                 # write exactly one gathered row
                v = rng.choice(views)
                i = rng.randrange(rows)
                rv = row_view(v, i)
                k += 1
                refs = rand_refs(rng, k, rv.numel)
                m.write(rv, refs)
                for j, e in enumerate(logical_elems(rv)):
                    storage[e] = refs[j]
            elif op < 0.50:                                 # free, then re-allocate (same or new generation)
                m.free(1)
                storage.clear()
                if rng.random() < 0.5:
                    gen += 1
                m.alloc(StorageAlloc(1, 0x1000, N * 2, DEV, gen, "op"))
            elif op < 0.53:                                 # regeneration without a free (implied by the adapter)
                gen += 1
                m.alloc(StorageAlloc(1, 0x1000, N * 2, DEV, gen, "op"))
                storage.clear()
            else:                                           # gather (repeated gathers hit the memo)
                v = rng.choice(views)
                idx = [rng.randrange(rows) for _ in range(rng.randrange(1, 8))]
                exp = [storage.get(e) for i in idx for e in logical_elems(row_view(v, i))]
                if any(r is None for r in exp):
                    with pytest.raises(Unresolved):
                        m.read_rows(v, idx)
                else:
                    assert list(m.read_rows(v, idx)) == exp
        st = m.stats()
        assert st.extra["row_cache_hits"] >= 0 and st.reads >= st.extra["row_cache_hits"]


def test_mixed_width_random_program_matches_byte_oracle():
    """Bytes are the truth: an element is resolvable iff all its bytes come from one write, of the same width,
    of the same written element, in the right byte order (so clobbered half-elements are unwritten)."""
    rng = random.Random(99)
    NB = 64                                                            # bytes per allocation
    dtypes = ["int8", "bfloat16", "int16", "int32", "float32", "int64"]
    for trial in range(60):
        m = MemoryModel()
        m.alloc(StorageAlloc(1, 0x1000, NB, DEV, 0, "op"))
        bytes_: dict[int, tuple] = {}                                  # byte -> (write id, elem j, byte in elem, width, ref)
        k = 0
        for _ in range(40):
            dtype = rng.choice(dtypes)
            w = DTYPE_BYTES[dtype]
            v = rand_view(rng, 1, NB // w, dtype)
            if rng.random() < 0.5:
                k += 1
                refs = rand_refs(rng, k, v.numel)
                m.write(v, refs)
                for j, e in enumerate(logical_elems(v)):
                    for b in range(w):
                        bytes_[e * w + b] = (k, j, b, w, refs[j])
            else:
                expected = []
                for e in logical_elems(v):
                    cells = [bytes_.get(e * w + b) for b in range(w)]
                    ok = all(c is not None for c in cells) and len({(c[0], c[1]) for c in cells}) == 1 \
                        and cells[0][3] == w and all(c[2] == b for b, c in enumerate(cells))
                    expected.append(cells[0][4] if ok else None)
                check_read(m, v, expected)
        # every dtype's full view at the end
        for dtype in dtypes:
            w = DTYPE_BYTES[dtype]
            v = ViewDesc(1, 0, (NB // w,), (1,), dtype, DEV)
            expected = []
            for e in range(NB // w):
                cells = [bytes_.get(e * w + b) for b in range(w)]
                ok = all(c is not None for c in cells) and len({(c[0], c[1]) for c in cells}) == 1 \
                    and cells[0][3] == w and all(c[2] == b for b, c in enumerate(cells))
                expected.append(cells[0][4] if ok else None)
            check_read(m, v, expected)


def test_generation_bump_and_free_in_random_program():
    rng = random.Random(3)
    N = 32
    m = MemoryModel()
    gen = 0
    m.alloc(StorageAlloc(1, 0x1000, N * 4, DEV, gen, "op"))
    storage: dict[int, tuple] = {}
    for step in range(200):
        r = rng.random()
        if r < 0.1:
            gen += 1
            m.alloc(StorageAlloc(1, 0x1000, N * 4, DEV, gen, "op"))
            storage.clear()
        elif r < 0.15:
            m.free(1)
            storage.clear()
            with pytest.raises(Unresolved) as ei:
                m.read(ViewDesc(1, 0, (N,), (1,), "float32", DEV))
            assert ei.value.reason == "freed"
            gen += 1
            m.alloc(StorageAlloc(1, 0x1000, N * 4, DEV, gen, "op"))
        elif r < 0.6:
            v = rand_view(rng, 1, N, "float32")
            refs = Affine("n", step, 0, v.numel)
            m.write(v, refs)
            for j, e in enumerate(logical_elems(v)):
                storage[e] = refs[j]
        else:
            v = rand_view(rng, 1, N, "float32")
            check_read(m, v, [storage.get(e) for e in logical_elems(v)])
    st = m.stats()
    assert st.allocations == 1 and st.live_allocations == 1 and st.intervals == len(m.intervals(1))

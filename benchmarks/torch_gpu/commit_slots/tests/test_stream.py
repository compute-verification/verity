"""Each slot's tree streams through a frontier and gives the roots the leaf matrix gave: at unit counts around powers of two,
groups that end anywhere, and passes of any number of slots, the host twin's roots are the old path's and the reference's;
and a step holds O(slots x log units) words beside one pass's temporaries, never a slot's leaves (counted, not timed)."""
from __future__ import annotations

import json
import random
import shutil
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import slots as S
import slots_gpu as G

pytestmark = pytest.mark.skipif(not any(shutil.which(c) for c in ("g++", "c++", "clang++")), reason="needs a C++ compiler")

CTX = bytes(range(64))
KEY = bytes(range(100, 132))


def unit_roots(plan: G.Plan, grp: Any, leaves: np.ndarray) -> np.ndarray:
    """(units, S, 8): each unit's root over its rows' (G, S, 8) leaves, a one-row unit's its leaf (the host's arrays)."""
    shape, S_ = grp.shape, int(leaves.shape[1])
    n = np.bincount(shape.unit_of_row, minlength=shape.units)
    first = np.concatenate(([0], np.cumsum(n)[:-1]))
    roots = np.zeros((shape.units, S_, 8), dtype=np.uint64)
    for m in sorted(set(n.tolist())):
        units = np.flatnonzero(n == m)
        x = leaves[first[units][:, None] + np.arange(m)]
        if m == 1:
            roots[units] = x[:, 0]
            continue
        x = np.ascontiguousarray(x.transpose(0, 2, 1, 3)).reshape(-1, m, 8)
        roots[units] = plan.engine.root(x, np.repeat(grp.dids[units], S_, axis=0)).reshape(len(units), S_, 8)
    return roots


def matrix_commit(plan: G.Plan, step: int, salt_key: bytes, outputs: list[list[Any]], slots: int) -> S.StepCommit:
    """The leaf-matrix path this replaced, kept as the yardstick: every row's leaf from the leaf kernel alone, unit roots and
    their slot-tree leaves (``frame_leaves``, from the IV) as separate calls, and every slot's unit leaves in one (slots,
    units) matrix, padded and rooted at the end."""
    eng = plan.engine
    leaves = plan.empty(slots, plan.layout.unit_count)
    for grp, tensors in zip(plan.groups, outputs, strict=True):
        x = eng.leaves(grp.leaf_table, eng.row_digests(plan.rows(grp.shape, tensors)), G.E.SaltKey(salt_key))
        u0 = grp.first_unit
        leaves[:, u0:u0 + grp.shape.units] = eng.frame_leaves(plan.per_slot(unit_roots(plan, grp, x)), [plan.slot_did], u0)
    slot_roots = eng.root(leaves, [plan.slot_did])
    step_did = S.step_domain(plan.context, plan.layout, step, slots).domain_id
    root = eng.root(eng.frame_leaves(slot_roots.reshape(1, slots, 8), [step_did]), [step_did])
    return S.StepCommit(G.E.digests(root)[0], tuple(G.E.digests(slot_roots)))


def odd_layout(n: int, seed: int) -> S.Layout:
    """``n`` units in up to four groups cut at random, each unit one to three rows of one or two parts of 1 to 40 bytes."""
    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(1, n), min(n - 1, rng.randint(0, 3)))) if n > 1 else []
    groups = []
    for size in (b - a for a, b in zip([0, *cuts], [*cuts, n], strict=True)):
        off, units = [0, 0], []
        for _ in range(size):
            rows = []
            for _ in range(rng.choice((1, 1, 1, 2, 3))):
                parts = []
                for src in (0,) if rng.random() < 0.7 else (0, 1):
                    nb = rng.randint(1, 40)
                    parts.append(S.Part(src, off[src], nb))
                    off[src] += nb
                rows.append(S.Row(tuple(parts)))
            units.append(tuple(rows))
        groups.append(S.Group((max(off[0], 1), max(off[1], 1)), tuple(units)))
    return S.Layout(f"odd/{n}/{seed}", tuple(groups))


COUNTS = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257]


@pytest.mark.parametrize("n", COUNTS)
def test_streamed_roots_are_the_leaf_matrix_s_and_the_reference_s(n: int) -> None:
    layout = odd_layout(n, n)
    assert layout.unit_count == n
    slots = 1 + n % 5
    out = S.random_outputs(layout, slots, np.random.default_rng(n))
    want = S.commit(CTX, layout, 2, KEY, out, slots)
    plan = G.Plan(CTX, layout, "host")
    assert matrix_commit(plan, 2, KEY, out, slots) == want
    assert G.commit(plan, 2, KEY, out, slots) == want
    one = G.Plan(CTX, layout, "host", temp_bytes=1)
    assert all(one.per_pass(g.shape, slots) == 1 for g in one.groups)
    assert G.commit(one, 2, KEY, out, slots) == want


@pytest.mark.parametrize("n", [1023, 1024, 1025, 4097])
@pytest.mark.parametrize("passes", [1, 3])
def test_larger_trees_match_the_leaf_matrix_in_passes(n: int, passes: int) -> None:
    """Deeper trees (to 2^13 leaves, pads alone in chunks below PAD_CHUNK) against the old path, in passes of 1 and 2 slots
    of 3."""
    layout = odd_layout(n, 7 * n)
    out = S.random_outputs(layout, 3, np.random.default_rng(n))
    rows = max(g.shape.rows for g in G.Plan(CTX, layout, "host").groups)
    plan = G.Plan(CTX, layout, "host", temp_bytes=G.TEMP_BUFFERS * 64 * rows * (1 if passes == 3 else 2))
    assert plan.per_pass(max((g.shape for g in plan.groups), key=lambda s: s.rows), 3) == (1 if passes == 3 else 2)
    assert G.commit(plan, 5, KEY, out, 3) == matrix_commit(plan, 5, KEY, out, 3)


def test_the_fixture_streams_to_its_roots_in_one_slot_passes() -> None:
    d = json.loads(S.FIXTURE.read_text())
    layout = S.edge_layout()
    out = [[np.frombuffer(bytes.fromhex(h), np.uint8).reshape(d["slots"], n) for h, n in zip(t, g.sources, strict=True)]
           for t, g in zip(d["outputs"], layout.groups, strict=True)]
    for temp in (G.TEMP_BYTES, 1):
        plan = G.Plan(bytes.fromhex(d["context"]), layout, "host", temp_bytes=temp)
        got = G.commit(plan, d["step"], bytes.fromhex(d["salt_key"]), out, d["slots"])
        assert got == matrix_commit(plan, d["step"], bytes.fromhex(d["salt_key"]), out, d["slots"])
        assert got.root.hex() == d["root"]


def test_the_pads_alone_are_the_reference_s_above_a_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subtree of pads wider than PAD_CHUNK is built from chunk roots: with the chunk at 2, the closing digests are the ones
    hashed in one go, and the roots the reference's."""
    layout = odd_layout(37, 3)
    want = G.Plan(CTX, layout, "host")
    monkeypatch.setattr(G, "PAD_CHUNK", 2)
    got = G.Plan(CTX, layout, "host")
    assert np.array_equal(want.pad0, got.pad0)
    assert [None if p is None else p.tolist() for p in want.pads_right] == [None if p is None else p.tolist() for p in got.pads_right]
    assert [k for k, p in enumerate(got.pads_right) if p is not None] == [1, 3, 4]   # 37 = 100101b: 8 and 16 pads in chunks of 4
    out = S.random_outputs(layout, 2, np.random.default_rng(37))
    assert G.commit(got, 1, KEY, out, 2) == S.commit(CTX, layout, 1, KEY, out, 2)


def _traced(fn: Callable[[], Any]) -> tuple[Any, int]:
    """``fn()`` and the most bytes it held at once beyond what it was handed (numpy's buffers and Python's objects)."""
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        out = fn()
        return out, tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()


SMALL = S.Group((8,), tuple((S.Row((S.Part(0, 2 * i, 2),)),) for i in range(4)))


def _first_add(layout: S.Layout, slots: int) -> tuple[G.StepCommitter, int]:
    """A step of ``slots`` slots over ``layout`` (whose first group is SMALL) with that group added: the committer and the
    bytes it held at most."""
    plan = G.Plan(CTX, layout, "host")
    out = S.random_outputs(layout, slots, np.random.default_rng(1), groups=[0])

    def run() -> G.StepCommitter:
        sc = G.StepCommitter(plan, 0, slots, KEY)
        sc.add(out[0])
        return sc
    return _traced(run)


def _pass_bound(rows: int, slots: int) -> int:
    return G.TEMP_BUFFERS * 64 * rows * slots + (1 << 20)


def _coords(n: int) -> S.Group:
    return S.Group((2 * n,), tuple((S.Row((S.Part(0, 2 * i, 2),)),) for i in range(n)))


def test_a_step_holds_slots_times_log_units() -> None:
    """At 4,096 slots, doubling the units adds one frontier level (4,096 x 64 bytes) and nothing else: the most a step holds
    is its frontier and one pass over the first group."""
    slots = 4096
    held = {}
    for layers in (5, 10):
        layout = S.Layout(f"small-then-{layers}", (SMALL, *(_coords(1000),) * layers))
        sc, peak = _first_add(layout, slots)
        depth = G.merkle_depth(layout.unit_count)
        assert sc.frontier.shape == (slots, depth + 1, 8) and sc.frontier.nbytes == slots * (depth + 1) * 64
        assert peak <= sc.frontier.nbytes + _pass_bound(4, slots)
        held[layers] = (peak - sc.frontier.nbytes, depth)
    (a, da), (b, db) = held[5], held[10]
    assert db == da + 1 and abs(b - a) < 64 << 10


@pytest.mark.slow
def test_a_step_of_4096_slots_over_six_million_units_holds_its_frontier() -> None:
    """At 4,096 slots and 6,104,804 units (SMALL, then 52 layers of 117,000 one-row units and 400 two-row ones, test_scale's
    families-like layout), a step's committer holds 4,096 x 24 x 64 bytes of frontier (6.3 MB) and one pass over the first
    group; the leaf matrix was 4,096 x 6.1M x 64 bytes (1.6 TB). What it catches that test_a_step_holds_slots_times_log_units
    can't: anything a unit at the full count. Builds a 6M-unit host Plan: 20 s and 2.3 GB on a CPU node."""
    units = [(S.Row((S.Part(0, 2 * i, 2),)),) for i in range(117_000)]
    units += [(S.Row((S.Part(1, 64 * i, 64),)), S.Row((S.Part(2, 0, 4),))) for i in range(400)]
    layer = S.Group((2 * 117_000, 64 * 400, 4), tuple(units))
    layout = S.Layout("small-then-families-like", (SMALL, *(layer,) * 52))
    assert layout.unit_count == 6_104_804
    sc, peak = _first_add(layout, 4096)
    assert sc.plan.depth == 23 and sc.frontier.nbytes == 4096 * 24 * 64
    assert peak <= sc.frontier.nbytes + _pass_bound(4, 4096)

"""The engine's host twin (``../sha512_rows``), driven by slots_gpu's own plan and step code, gives the reference's roots;
regrouping a layout changes no root; the streaming call refuses what isn't the layout's."""
from __future__ import annotations

import json
import shutil
import weakref

import numpy as np
import pytest

import slots as S
import slots_gpu as G

pytestmark = pytest.mark.skipif(not any(shutil.which(c) for c in ("g++", "c++", "clang++")), reason="needs a C++ compiler")

CTX = bytes(range(64))
KEY = bytes(range(100, 132))


@pytest.mark.parametrize("slots", [1, 2, 5, 9])
def test_the_edge_layout(slots: int) -> None:
    layout = S.edge_layout()
    out = S.random_outputs(layout, slots, np.random.default_rng(slots))
    assert G.commit(G.Plan(CTX, layout, "host"), 3, KEY, out, slots) == S.commit(CTX, layout, 3, KEY, out, slots)


def test_the_fixtures_root() -> None:
    d = json.loads(S.FIXTURE.read_text())
    layout = S.edge_layout()
    out = [[np.frombuffer(bytes.fromhex(h), np.uint8).reshape(d["slots"], n) for h, n in zip(t, g.sources, strict=True)]
           for t, g in zip(d["outputs"], layout.groups, strict=True)]
    got = G.commit(G.Plan(bytes.fromhex(d["context"]), layout, "host"), d["step"], bytes.fromhex(d["salt_key"]), out, d["slots"])
    assert got.root.hex() == d["root"]
    assert [r.hex() for r in got.slot_roots] == [s["root"] for s in d["per_slot"]]


def test_two_h200_layers() -> None:
    layout = S.h200_layout(layers=2)
    out = S.random_outputs(layout, 3, np.random.default_rng(7))
    assert G.commit(G.Plan(CTX, layout, "host"), 9, KEY, out, 3) == S.commit(CTX, layout, 9, KEY, out, 3)


@pytest.mark.parametrize(("layout", "k"), [(S.h200_layout(layers=3), 2), (S.edge_layout(), 2)], ids=["h200-3-by-2", "edge-by-2"])
def test_regrouped_layers_give_the_same_digest_and_roots(layout: S.CommitShape, k: int) -> None:
    grouped = G.regroup(layout, k)
    assert len(grouped.groups) == -(-len(layout.groups) // k) and S.layout_digest(grouped) == S.layout_digest(layout)
    out = S.random_outputs(layout, 3, np.random.default_rng(11))
    joined = [[t for ts in out[i:i + k] for t in ts] for i in range(0, len(out), k)]
    assert G.commit(G.Plan(CTX, grouped, "host"), 4, KEY, joined, 3) == S.commit(CTX, layout, 4, KEY, out, 3)


def test_a_step_reuses_its_plan_and_its_root_moves_with_the_step_and_the_key() -> None:
    layout = S.edge_layout()
    plan = G.Plan(CTX, layout, "host")
    out = S.random_outputs(layout, 4, np.random.default_rng(8))
    a, b = G.commit(plan, 0, KEY, out, 4), G.commit(plan, 1, KEY, out, 4)
    c = G.commit(plan, 0, bytes(32), out, 4)
    assert a == S.commit(CTX, layout, 0, KEY, out, 4) and b == S.commit(CTX, layout, 1, KEY, out, 4)
    assert len({a.root, b.root, c.root}) == 3 and a.slot_roots == b.slot_roots != c.slot_roots


def test_the_streaming_call_refuses_what_is_not_the_layouts() -> None:
    layout = S.edge_layout()
    plan = G.Plan(CTX, layout, "host")
    out = S.random_outputs(layout, 2, np.random.default_rng(9))
    sc = G.StepCommitter(plan, 0, 2, KEY)
    with pytest.raises(ValueError, match="takes 1 tensors"):
        sc.add(out[1])
    with pytest.raises(ValueError, match="contiguous bytes a slot"):
        sc.add([out[0][0][:, :-1]])
    with pytest.raises(ValueError, match="slots=2"):
        sc.add([out[0][0][:1]])
    sc.add(out[0])
    with pytest.raises(ValueError, match="1 of 2 groups"):
        sc.finish()
    sc.add(out[1])
    with pytest.raises(ValueError, match="every group is in"):
        sc.add(out[1])
    assert sc.finish().result() == S.commit(CTX, layout, 0, KEY, out, 2)
    with pytest.raises(ValueError, match="32 bytes"):
        G.StepCommitter(plan, 0, 2, KEY[:31])


class _Read:
    def __init__(self) -> None:
        self.passed = False

    def query(self) -> bool:
        return self.passed


class _Producer:
    def __init__(self) -> None:
        self.waited: list[_Read] = []

    def wait_event(self, read: _Read) -> None:
        self.waited.append(read)


class _Table:
    pass


def test_a_plan_drops_tables_once_read_and_past_the_lag_after_the_producer_waits() -> None:
    """``Plan.release`` with stand-in events (a CUDA plan's adds hold their tensors so): a held table goes, oldest first,
    once its read has passed, and with ``keep`` every add's but the last ``keep``, an unread one's only after its producer's
    stream waits for its read; a later read never lets a table go ahead of an earlier unread one."""
    plan, producer = G.Plan(CTX, S.edge_layout(), "host"), _Producer()
    reads, tables = [_Read() for _ in range(4)], [_Table() for _ in range(4)]
    alive = [weakref.ref(t) for t in tables]
    for read, t in zip(reads[:3], tables[:3], strict=True):
        plan.hold(read, producer, [t])
    t3 = tables[3]
    del tables, t
    plan.release(2)
    assert producer.waited == [reads[0]] and [a() is not None for a in alive[:3]] == [False, True, True]
    reads[2].passed = True
    plan.release()
    assert alive[1]() is not None and alive[2]() is not None
    reads[1].passed = True
    plan.release()
    assert alive[1]() is None and alive[2]() is None and producer.waited == [reads[0]] and not plan._held.entries
    plan.hold(reads[3], producer, [t3])
    del t3
    plan.release(1)
    assert alive[3]() is not None
    plan.release(0)
    assert alive[3]() is None and producer.waited == [reads[0], reads[3]]

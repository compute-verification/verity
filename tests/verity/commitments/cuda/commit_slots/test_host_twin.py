"""The engine's host twin (``../sha512_rows``), driven by slots_gpu's own plan and step code, gives the reference's roots
under the same salt bytes, handed to both as data; regrouping a layout changes no root; the streaming call refuses what
isn't the layout's, and a step with no salts or a key in their place."""
from __future__ import annotations

import json
import shutil
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import BUILD, salts_of, target

from verity.commitments.cuda.commit_slots import slots as S
from verity.commitments.cuda.commit_slots import slots_gpu as G

pytestmark = pytest.mark.skipif(not any(shutil.which(c) for c in ("g++", "c++", "clang++")), reason="needs a C++ compiler")

CTX = bytes(range(64))


@pytest.mark.parametrize("slots", [1, 2, 5, 9])
def test_the_edge_layout(slots: int) -> None:
    layout = S.edge_layout()
    out = S.random_outputs(layout, slots, np.random.default_rng(slots))
    y = salts_of(layout, slots)
    assert G.commit(G.Plan(CTX, layout, "host", build=BUILD), 3, y, out, slots) == S.commit(CTX, layout, 3, y, out, slots)


def test_the_fixtures_root() -> None:
    d = json.loads(S.FIXTURE.read_text())
    layout = S.edge_layout()
    out = [[np.frombuffer(bytes.fromhex(h), np.uint8).reshape(d["slots"], n) for h, n in zip(t, g.sources, strict=True)]
           for t, g in zip(d["outputs"], layout.groups, strict=True)]
    y = np.stack([np.frombuffer(bytes.fromhex(h), np.uint8).reshape(-1, 192) for h in d["salts"]])
    got = G.commit(G.Plan(bytes.fromhex(d["context"]), layout, "host", build=BUILD), d["step"], y, out, d["slots"])
    assert got.root.hex() == d["root"]
    assert [r.hex() for r in got.slot_roots] == [s["root"] for s in d["per_slot"]]


def test_two_h200_layers() -> None:
    layout = S.h200_layout(layers=2)
    out = S.random_outputs(layout, 3, np.random.default_rng(7))
    y = salts_of(layout, 3)
    assert G.commit(G.Plan(CTX, layout, "host", build=BUILD), 9, y, out, 3) == S.commit(CTX, layout, 9, y, out, 3)


@pytest.mark.parametrize(("layout", "k"), [(S.h200_layout(layers=3), 2), (S.edge_layout(), 2)], ids=["h200-3-by-2", "edge-by-2"])
def test_regrouped_layers_give_the_same_digest_and_roots(layout: S.CommitShape, k: int) -> None:
    grouped = G.regroup(layout, k)
    assert len(grouped.groups) == -(-len(layout.groups) // k) and S.layout_digest(grouped) == S.layout_digest(layout)
    out = S.random_outputs(layout, 3, np.random.default_rng(11))
    joined = [[t for ts in out[i:i + k] for t in ts] for i in range(0, len(out), k)]
    y = salts_of(layout, 3)
    assert G.commit(G.Plan(CTX, grouped, "host", build=BUILD), 4, y, joined, 3) == S.commit(CTX, layout, 4, y, out, 3)


def test_a_step_reuses_its_plan_and_its_root_moves_with_the_step_and_the_salts() -> None:
    layout = S.edge_layout()
    plan = G.Plan(CTX, layout, "host", build=BUILD)
    out = S.random_outputs(layout, 4, np.random.default_rng(8))
    y, other = salts_of(layout, 4), salts_of(layout, 4, seed=1)
    a, b = G.commit(plan, 0, y, out, 4), G.commit(plan, 1, y, out, 4)
    c = G.commit(plan, 0, other, out, 4)
    assert a == S.commit(CTX, layout, 0, y, out, 4) and b == S.commit(CTX, layout, 1, y, out, 4)
    assert len({a.root, b.root, c.root}) == 3 and a.slot_roots == b.slot_roots != c.slot_roots


def test_one_rows_salt_moves_its_slots_root_alone() -> None:
    """A leaf is under its own 192 bytes: one bit of row g of slot s's salt moves slot s's root and the step's, no other
    slot's, in the reference and the host twin alike."""
    layout, slots = S.edge_layout(), 3
    plan = G.Plan(CTX, layout, "host", build=BUILD)
    out = S.random_outputs(layout, slots, np.random.default_rng(12))
    y = salts_of(layout, slots)
    base = G.commit(plan, 0, y, out, slots)
    for s, g, n in ((0, 0, 0), (1, layout.row_count - 1, 191), (2, 4, 77)):
        moved = y.copy()
        moved[s, g, n] ^= 0x80
        got = G.commit(plan, 0, moved, out, slots)
        assert got == S.commit(CTX, layout, 0, moved, out, slots) and got.root != base.root
        assert [i for i in range(slots) if got.slot_roots[i] != base.slot_roots[i]] == [s]


def test_a_step_takes_its_salts_as_data_and_derives_none() -> None:
    """No key makes a step's salts: the reference and the plan's calls take every row's 192 bytes, and refuse a key or a
    buffer that is not a salt a row; ``fresh_salts`` draws them from the operating system, new bytes each call."""
    layout, slots = S.edge_layout(), 2
    plan = G.Plan(CTX, layout, "host", build=BUILD)
    out = S.random_outputs(layout, slots, np.random.default_rng(13))
    assert not hasattr(S, "fresh_salt_key") and not hasattr(S, "row_salts")
    y = S.fresh_salts(layout, slots)
    assert y.shape == (slots, layout.row_count, 192) and y.dtype == np.uint8
    assert len({y[s, g].tobytes() for s in range(slots) for g in range(layout.row_count)}) == slots * layout.row_count
    assert not np.array_equal(y, S.fresh_salts(layout, slots))
    assert G.commit(plan, 0, y, out, slots) == S.commit(CTX, layout, 0, y, out, slots)
    for bad in (bytes(32), y[:1], y[:, :-1], y.astype(np.uint16)):
        with pytest.raises(ValueError, match="salt"):
            S.commit(CTX, layout, 0, bad, out, slots)
        with pytest.raises(ValueError, match="salt"):
            G.commit(plan, 0, bad, out, slots)
    with pytest.raises(TypeError):
        G.StepCommitter(plan, 0, slots, bytes(32), None, False, 1)


def test_the_streaming_call_refuses_what_is_not_the_layouts() -> None:
    layout = S.edge_layout()
    plan = G.Plan(CTX, layout, "host", build=BUILD)
    out = S.random_outputs(layout, 2, np.random.default_rng(9))
    y = salts_of(layout, 2)
    n0 = layout.group_rows[0]
    y0, y1 = y[:, :n0], y[:, n0:]
    sc = G.StepCommitter(plan, 0, 2)
    with pytest.raises(ValueError, match="takes 1 tensors"):
        sc.add(out[1], y1)
    with pytest.raises(ValueError, match="contiguous bytes a slot"):
        sc.add([out[0][0][:, :-1]], y0)
    with pytest.raises(ValueError, match="slots=2"):
        sc.add([out[0][0][:1]], y0)
    for bad in (None, bytes(32), y1, y0[:1], y0.transpose(1, 0, 2)):
        with pytest.raises(ValueError, match="group 0's salts are"):
            sc.add(out[0], bad)
    sc.add(out[0], y0)
    with pytest.raises(ValueError, match="1 of 2 groups"):
        sc.finish()
    sc.add(out[1], y1)
    with pytest.raises(ValueError, match="every group is in"):
        sc.add(out[1], y1)
    assert sc.finish().result() == S.commit(CTX, layout, 0, y, out, 2)


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
    plan, producer = G.Plan(CTX, S.edge_layout(), "host", build=BUILD), _Producer()
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


def test_a_graphs_key_names_the_sms_its_leaf_grid_was_sized_to() -> None:
    """``Plan.graph``'s key, as its own code builds it with no device: the plan's SM count is its ``sms``, by name, and the
    strides the graph read its salts at come after it."""
    class Held(dict):
        def __contains__(self, key):
            return True

        def __getitem__(self, key):
            return key
    plan = SimpleNamespace(temp_bytes=1 << 20, leaf_minb=0, sms=16, _graphs=Held())
    salts = SimpleNamespace(stride=lambda d: (14400, 192)[d])
    key = G.Plan.graph(plan, SimpleNamespace(k=2, slots=75), None, (), salts)
    assert key == (2, 75, 1 << 20, 0, 16, 14400, 192)
    assert key.sms == 16 and (key.row_stride, key.slot_stride) == (14400, 192) and key[-1] != key.sms


def test_a_plan_on_the_host_asks_for_no_cards_target_and_a_cuda_plan_for_its_card_files() -> None:
    """The device tests' plan takes its target from `conftest.target`: on the host it asks torch for no device's name
    (torch refuses the string ``"host"``) and passes none, which the engine's host twin takes; on a CUDA device it asks
    for that device's name and passes its card file's target, and a device no card file names is refused."""
    asked: list[str] = []

    def refuses(device: str) -> str:
        raise AssertionError(f"asked for the name of {device!r} with no CUDA device")

    assert target(SimpleNamespace(cuda=SimpleNamespace(get_device_name=refuses)), "host") is None
    assert G.Plan(CTX, S.edge_layout(), "host", target(None, "host"), build=BUILD).host
    names = {"cuda": "NVIDIA H100 80GB HBM3", "cuda:1": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}
    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda device: asked.append(device) or names[device]))
    assert [target(torch, d) for d in names] == ["90a", "120a"] and asked == list(names)
    with pytest.raises(KeyError, match="NVIDIA Tesla T4"):
        target(SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda device: "NVIDIA Tesla T4")), "cuda")

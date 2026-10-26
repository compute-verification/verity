"""On a CUDA device: the call on the engine's device build gives the reference's roots (the recorded runs are check.py's,
at the frozen shape)."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

import slots as S
import slots_gpu as G

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or shutil.which("nvcc") is None,
                                reason="needs a CUDA device and nvcc")

CTX = bytes(range(64))
KEY = bytes(range(100, 132))


def _cuda(out: list[list[np.ndarray]]) -> list[list[object]]:
    return [[torch.from_numpy(a).cuda() for a in tensors] for tensors in out]


@pytest.mark.parametrize("slots", [1, 5, 300])
def test_the_edge_layout(slots: int) -> None:
    layout = S.edge_layout()
    out = S.random_outputs(layout, slots, np.random.default_rng(slots))
    assert G.commit(G.Plan(CTX, layout), 1, KEY, _cuda(out), slots) == G.commit(G.Plan(CTX, layout, "host"), 1, KEY, out, slots)


def test_two_h200_layers() -> None:
    layout = S.h200_layout(layers=2)
    out = S.random_outputs(layout, 3, np.random.default_rng(3))
    assert G.commit(G.Plan(CTX, layout), 2, KEY, _cuda(out), 3) == S.commit(CTX, layout, 2, KEY, out, 3)


@pytest.mark.parametrize("leaf_minb", [1, 2])
def test_either_leaf_kernel_gives_the_reference_s_roots(leaf_minb: int) -> None:
    layout = S.h200_layout(layers=2)
    out = S.random_outputs(layout, 3, np.random.default_rng(5))
    plan = G.Plan(CTX, layout, leaf_minb=leaf_minb)
    assert G.commit(plan, 3, KEY, _cuda(out), 3) == S.commit(CTX, layout, 3, KEY, out, 3)


@pytest.mark.parametrize("temp_bytes", [1, 4 * 64 * 25 * 2], ids=["1-slot-passes", "2-slot-passes"])
def test_passes_on_the_device_give_the_reference_s_roots(temp_bytes: int) -> None:
    """Passes over a few slots at a time (each pass's row table moved on its first slot, its salts' nonces from there) and the
    frontier folded and closed on the device: three H200 layers (75 units in groups of 25 rows) at 5 slots."""
    layout = S.h200_layout(layers=3)
    out = S.random_outputs(layout, 5, np.random.default_rng(4))
    plan = G.Plan(CTX, layout, temp_bytes=temp_bytes)
    assert plan.per_pass(plan.groups[0].shape, 5) == (1 if temp_bytes == 1 else 2)
    assert G.commit(plan, 6, KEY, _cuda(out), 5) == S.commit(CTX, layout, 6, KEY, out, 5)


@pytest.mark.parametrize("temp_bytes", [G.TEMP_BYTES, 4 * 64 * 25 * 2], ids=["one-pass", "2-slot-passes"])
def test_steps_on_graphs_give_the_reference_s_roots(temp_bytes: int) -> None:
    """Each add one CUDA graph launch, captured on the first step and replayed on the next over other tensors (other
    addresses) under another salt key: three H200 layers at 5 slots, in one pass or in passes of 2 slots."""
    layout = S.h200_layout(layers=3)
    plan = G.Plan(CTX, layout, temp_bytes=temp_bytes)
    for step, key in ((6, KEY), (7, bytes(range(32))), (8, KEY)):
        out = S.random_outputs(layout, 5, np.random.default_rng(step))
        assert G.commit(plan, step, key, _cuda(out), 5, graphs=True) == S.commit(CTX, layout, step, key, out, 5)
    assert len(plan._graphs) == len(layout.groups)


def test_the_edge_layout_on_graphs() -> None:
    """Units of several rows (one padded) and a two-part row, replayed under a second key."""
    layout = S.edge_layout()
    plan, host = G.Plan(CTX, layout), G.Plan(CTX, layout, "host")
    for step, key in ((1, KEY), (2, bytes(32))):
        out = S.random_outputs(layout, 300, np.random.default_rng(step))
        assert G.commit(plan, step, key, _cuda(out), 300, graphs=True) == G.commit(host, step, key, out, 300)


def test_a_marked_step_on_graphs_marks_each_add() -> None:
    """Marks (e2e_h200 times its commit by the last) are one an add on graphs, none captured, and finish's as ever."""
    layout = S.h200_layout(layers=2)
    plan = G.Plan(CTX, layout)
    for step in (1, 2):
        out = S.random_outputs(layout, 3, np.random.default_rng(step))
        marks: list = []
        sc = G.StepCommitter(plan, step, 3, KEY, marks=marks, graphs=True)
        for tensors in _cuda(out):
            sc.add(tensors)
        assert sc.finish().result() == S.commit(CTX, layout, step, KEY, out, 3)
        assert [n for n, _ in marks] == ["add"] * len(layout.groups) + ["slot_trees", "step_tree"]
        assert marks[0][1].elapsed_time(marks[-1][1]) >= 0


def test_a_plan_on_some_sms_gives_the_reference_s_roots_on_graphs() -> None:
    """Graphs captured on a green context's stream, their leaf grid sized to its SMs, replayed under a second key."""
    layout = S.h200_layout(layers=3)
    plan = G.Plan(CTX, layout, sms=16)
    for step, key in ((6, KEY), (7, bytes(range(32)))):
        out = S.random_outputs(layout, 5, np.random.default_rng(step))
        assert G.commit(plan, step, key, _cuda(out), 5, graphs=True) == S.commit(CTX, layout, step, key, out, 5)
    assert {k[-1] for k in plan._graphs} == {plan.sms}


def test_a_plan_has_one_step_on_graphs_open() -> None:
    layout = S.edge_layout()
    plan = G.Plan(CTX, layout)
    out = _cuda(S.random_outputs(layout, 2, np.random.default_rng(1)))
    sc = G.StepCommitter(plan, 1, 2, KEY, graphs=True)
    sc.add(out[0])
    with pytest.raises(ValueError, match="open"):
        G.StepCommitter(plan, 2, 2, KEY, graphs=True)
    sc.add(out[1])
    sc.finish()
    G.StepCommitter(plan, 2, 2, KEY, graphs=True)

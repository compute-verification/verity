"""On a CUDA device: the call on the engine's device build gives the reference's roots (the recorded runs are check.py's,
at the frozen shape)."""
from __future__ import annotations

import shutil
import weakref

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


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
def test_tables_freed_once_added_take_no_allocator_retry_at_the_hbm_cap(graphs: bool) -> None:
    """At the HBM cap, with room for two groups' tables (the one being read and the one being made), a producer that
    frees each group's tables once it has made the next group's (e2e_h200's driver, its stream waiting for the reads
    ``LAG`` groups back) gets the group before's memory back for the group after while the side stream lags its reads,
    so across two steps the allocator never retries (a retry frees its cache and waits on every stream) and nothing
    runs out, and the roots are an unpressured run's. What it catches that the roots' tests can't: a commit that keeps a
    freed table from the producer's next allocation, as ``record_stream`` onto the side stream did (a table was reusable
    only once that stream had run everything queued on it when the table was freed, so the third group's retried), or
    as holding tables two adds did (the third group's ran out)."""
    dev, width, slots, lag = torch.device("cuda"), 4 << 20, 64, 2
    group = S.Group((width,), tuple((S.Row((S.Part(0, a, 32 << 10),)),) for a in range(0, width, 32 << 10)))
    plan = G.Plan(CTX, S.CommitShape("commit-slots/test-hbm-cap/v1", (group,) * 6))
    gen = torch.Generator(device=dev)

    def table(step: int, g: int) -> object:
        gen.manual_seed(100 * step + g)
        return torch.randint(-128, 128, (slots, width), dtype=torch.int8, device=dev, generator=gen)

    def steps(sleep: int) -> tuple[list[S.StepCommit], list[bool]]:
        out, lagged = [], []
        for step, key in ((0, KEY), (1, bytes(range(32)))):
            sc, reads, t = G.StepCommitter(plan, step, slots, key, graphs=graphs), [], None
            for g in range(len(plan.groups)):
                if sleep:
                    with torch.cuda.stream(plan.stream):
                        torch.cuda._sleep(sleep)
                t = table(step, g)
                reads.append(sc.add([t]))
                lagged.append(not reads[-1].query())
                if len(reads) > lag:
                    torch.cuda.current_stream(dev).wait_event(reads.pop(0))
            del t
            out.append(sc.finish().result())
        return out, lagged

    want, _ = steps(0)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(dev)
    left, ballast = (free - 2 * slots * width - (128 << 20)) // (2 << 20) * (2 << 20), []
    try:
        while left > 0:
            ballast.append(torch.empty(min(left, 1 << 30), dtype=torch.uint8, device=dev))
            left -= min(left, 1 << 30)
        before = torch.cuda.memory_stats(dev)
        got, lagged = steps(50_000_000)
        after = torch.cuda.memory_stats(dev)
    finally:
        del ballast
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
    assert all(lagged)
    assert {k: after[k] - before[k] for k in ("num_alloc_retries", "num_ooms")} == {"num_alloc_retries": 0, "num_ooms": 0}
    assert got == want


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
def test_a_table_freed_once_added_goes_at_its_read_with_no_later_add(graphs: bool) -> None:
    """A CUDA plan drops an add's tensors as soon as the side stream has read them, before any later add or result, so a
    producer that frees its tables has their memory back for whatever it allocates next. What it catches that the HBM-cap
    test can't: tables held to the next add, which kept each freed group from the forward's allocations in between (1.76
    GB more on the device at the frozen shape's 1 KiB rows); the roots are the reference's."""
    layout = S.h200_layout(layers=2)
    out = S.random_outputs(layout, 3, np.random.default_rng(9))
    plan = G.Plan(CTX, layout)
    sc = G.StepCommitter(plan, 4, 3, KEY, graphs=graphs)
    first, rest = _cuda(out[:1])[0], _cuda(out[1:])
    alive = [weakref.ref(t) for t in first]
    with torch.cuda.stream(plan.stream):
        torch.cuda._sleep(50_000_000)
    done = sc.add(first)
    del first
    done.synchronize()
    held = plan._held
    with held.lock:
        assert held.thread is not None and held.thread.is_alive()
        held.drained.wait_for(lambda: not held.entries)
    assert [a() for a in alive] == [None] * len(alive)
    for tensors in rest:
        sc.add(tensors)
    assert sc.finish().result() == S.commit(CTX, layout, 4, KEY, out, 3)


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

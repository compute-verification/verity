"""On a CUDA device: the call on the engine's device build gives the reference's roots under the same salt bytes, read
from a device buffer as data (the recorded runs are check.py's, at the frozen shape)."""
from __future__ import annotations

import shutil
from pathlib import Path
import weakref

import numpy as np
import pytest

from conftest import BUILD, salts_of

from verity.commitments.cuda.commit_slots import slots as S
from verity.commitments.cuda.commit_slots import slots_gpu as G
from verity.device import compute_capability, gpu_card_reporting

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or shutil.which("nvcc") is None,
                                reason="needs a CUDA device and nvcc")

CTX = bytes(range(64))


class _Engine(G.Plan.ENGINE):
    def runtime(self) -> object:
        return torch


#: the GPU card files (`verity.device`), whose ``arch`` is the target the device library is built for
CARD_FILES = Path(__file__).resolve().parents[5] / "calibration/hardware/cards"


def arch(name: str) -> str:
    """The target a library is built for on the device reporting ``name`` (torch's ``get_device_name``): its card file's
    ``arch`` (`verity.device.gpu_card_reporting`) without ``sm_``, such as ``90a``; a device no card file names is refused."""
    card = gpu_card_reporting(name, directory=CARD_FILES)
    compute_capability(card)
    return card["arch"].removeprefix("sm_")


class Plan(G.Plan):
    """``slots_gpu.Plan`` on torch, which verity doesn't import, built for the target its device's card file names."""
    ENGINE = _Engine

    def __init__(self, context: bytes, layout: S.CommitShape, device: str, **kw: object) -> None:
        super().__init__(context, layout, device, arch(torch.cuda.get_device_name(device)), **kw)


def _cuda(out: list[list[np.ndarray]]) -> list[list[object]]:
    return [[torch.from_numpy(a).cuda() for a in tensors] for tensors in out]


def _dev(salts: np.ndarray) -> object:
    return torch.from_numpy(salts).cuda()


def _by_group(layout: S.CommitShape, salts: object) -> list[object]:
    """A step's salts cut into its groups' (views of the one buffer, as a caller hands them to ``add``)."""
    return [salts[:, g0:g0 + n] for g0, n in zip(layout.group_row_base, layout.group_rows, strict=True)]


@pytest.mark.parametrize("slots", [1, 5, 300])
def test_the_edge_layout(slots: int) -> None:
    layout = S.edge_layout()
    out = S.random_outputs(layout, slots, np.random.default_rng(slots))
    y = salts_of(layout, slots)
    assert G.commit(Plan(CTX, layout, "cuda", build=BUILD), 1, _dev(y), _cuda(out), slots) == G.commit(Plan(CTX, layout, "host", build=BUILD), 1, y, out, slots)


def test_two_h200_layers() -> None:
    layout = S.h200_layout(layers=2)
    out = S.random_outputs(layout, 3, np.random.default_rng(3))
    y = salts_of(layout, 3)
    assert G.commit(Plan(CTX, layout, "cuda", build=BUILD), 2, _dev(y), _cuda(out), 3) == S.commit(CTX, layout, 2, y, out, 3)


@pytest.mark.parametrize("leaf_minb", [1, 2])
def test_either_leaf_kernel_gives_the_reference_s_roots(leaf_minb: int) -> None:
    layout = S.h200_layout(layers=2)
    out = S.random_outputs(layout, 3, np.random.default_rng(5))
    plan = Plan(CTX, layout, "cuda", leaf_minb=leaf_minb, build=BUILD)
    y = salts_of(layout, 3)
    assert G.commit(plan, 3, _dev(y), _cuda(out), 3) == S.commit(CTX, layout, 3, y, out, 3)


@pytest.mark.parametrize("temp_bytes", [1, 4 * 64 * 25 * 2], ids=["1-slot-passes", "2-slot-passes"])
def test_passes_on_the_device_give_the_reference_s_roots(temp_bytes: int) -> None:
    """Passes over a few slots at a time (each pass's row table moved on its first slot, its salts read from that slot's on) and the
    frontier folded and closed on the device: three H200 layers (75 units in groups of 25 rows) at 5 slots."""
    layout = S.h200_layout(layers=3)
    out = S.random_outputs(layout, 5, np.random.default_rng(4))
    plan = Plan(CTX, layout, "cuda", temp_bytes=temp_bytes, build=BUILD)
    assert plan.per_pass(plan.groups[0].shape, 5) == (1 if temp_bytes == 1 else 2)
    y = salts_of(layout, 5)
    assert G.commit(plan, 6, _dev(y), _cuda(out), 5) == S.commit(CTX, layout, 6, y, out, 5)


@pytest.mark.parametrize("temp_bytes", [G.TEMP_BYTES, 4 * 64 * 25 * 2], ids=["one-pass", "2-slot-passes"])
def test_steps_on_graphs_give_the_reference_s_roots(temp_bytes: int) -> None:
    """Each add one CUDA graph launch, captured on the first step and replayed on the next over other tensors (other
    addresses) under other salts in another buffer: three H200 layers at 5 slots, in one pass or in passes of 2 slots."""
    layout = S.h200_layout(layers=3)
    plan = Plan(CTX, layout, "cuda", temp_bytes=temp_bytes, build=BUILD)
    for step in (6, 7, 8):
        out = S.random_outputs(layout, 5, np.random.default_rng(step))
        y = salts_of(layout, 5, seed=step)
        assert G.commit(plan, step, _dev(y), _cuda(out), 5, graphs=True) == S.commit(CTX, layout, step, y, out, 5)
    assert len(plan._graphs) == len(layout.groups)


def test_the_edge_layout_on_graphs() -> None:
    """Units of several rows (one padded) and a two-part row, replayed under a second step's salts."""
    layout = S.edge_layout()
    plan, host = Plan(CTX, layout, "cuda", build=BUILD), Plan(CTX, layout, "host", build=BUILD)
    for step in (1, 2):
        out = S.random_outputs(layout, 300, np.random.default_rng(step))
        y = salts_of(layout, 300, seed=step)
        assert G.commit(plan, step, _dev(y), _cuda(out), 300, graphs=True) == G.commit(host, step, y, out, 300)


def test_salts_are_read_in_place_from_a_wider_buffer_and_misaligned_ones_refused() -> None:
    """The leaf kernel reads each salt where the caller put it: a step's salts as a strided view of a wider buffer (a
    16-byte-aligned offset in) give the reference's roots, eagerly and on graphs; a view the vector loads can't read
    (an odd byte offset) is refused by name rather than copied."""
    layout, slots = S.edge_layout(), 5
    plan = Plan(CTX, layout, "cuda", build=BUILD)
    out = S.random_outputs(layout, slots, np.random.default_rng(21))
    y = salts_of(layout, slots)
    wide = torch.zeros((slots + 2, layout.row_count + 3, 192), dtype=torch.uint8, device="cuda")
    wide[1:1 + slots, 2:2 + layout.row_count] = _dev(y)
    view = wide[1:1 + slots, 2:2 + layout.row_count]
    for graphs in (False, True):
        assert G.commit(plan, 1, view, _cuda(out), slots, graphs=graphs) == S.commit(CTX, layout, 1, y, out, slots)
    odd = torch.zeros(y.size + 1, dtype=torch.uint8, device="cuda")[1:].view(y.shape)
    with pytest.raises(ValueError, match="salt"):
        G.commit(plan, 1, odd, _cuda(out), slots)


def test_a_marked_step_on_graphs_marks_each_add() -> None:
    """Marks (a caller times its commit by the last) are one an add on graphs, none captured, and finish's as ever."""
    layout = S.h200_layout(layers=2)
    plan = Plan(CTX, layout, "cuda", build=BUILD)
    for step in (1, 2):
        out = S.random_outputs(layout, 3, np.random.default_rng(step))
        marks: list = []
        y = salts_of(layout, 3, seed=step)
        sc = G.StepCommitter(plan, step, 3, marks=marks, graphs=True)
        for tensors, salts in zip(_cuda(out), _by_group(layout, _dev(y)), strict=True):
            sc.add(tensors, salts)
        assert sc.finish().result() == S.commit(CTX, layout, step, y, out, 3)
        assert [n for n, _ in marks] == ["add"] * len(layout.groups) + ["slot_trees", "step_tree"]
        assert marks[0][1].elapsed_time(marks[-1][1]) >= 0


def test_a_plan_on_some_sms_gives_the_reference_s_roots_on_graphs() -> None:
    """Graphs captured on a green context's stream, their leaf grid sized to its SMs, replayed under a second step's salts.
    Skipped where this Python's torch has no driver bindings to make a green context with."""
    try:
        G.driver_bindings(torch)
    except RuntimeError as e:
        pytest.skip(str(e))
    layout = S.h200_layout(layers=3)
    plan = Plan(CTX, layout, "cuda", sms=16, build=BUILD)
    for step in (6, 7):
        out = S.random_outputs(layout, 5, np.random.default_rng(step))
        y = salts_of(layout, 5, seed=step)
        assert G.commit(plan, step, _dev(y), _cuda(out), 5, graphs=True) == S.commit(CTX, layout, step, y, out, 5)
    assert plan.sms == 16 and {k.sms for k in plan._graphs} == {plan.sms}


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
def test_tables_freed_once_added_take_no_allocator_retry_at_the_hbm_limit(graphs: bool) -> None:
    """At the HBM limit, with room for two groups' tables (the one being read and the one being made), a producer that
    frees each group's tables once it has made the next group's (a serving loop's driver, its stream waiting for the
    reads a fixed number of groups back) gets the group before's memory back for the group after while the side stream
    lags its reads, so across two steps the allocator never retries (a retry frees its cache and waits on every stream)
    and nothing runs out, and the roots are an unpressured run's. What it catches that the roots' tests can't: a commit
    that keeps a freed table from the producer's next allocation, as ``record_stream`` onto the side stream did (a table
    was reusable only once that stream had run everything queued on it when the table was freed, so the third group's
    retried), or as holding tables two adds did (the third group's ran out)."""
    dev, width, slots, lag = torch.device("cuda"), 4 << 20, 64, 2
    group = S.Group((width,), tuple((S.Row((S.Part(0, a, 32 << 10),)),) for a in range(0, width, 32 << 10)))
    layout = S.CommitShape("commit-slots/test-hbm-cap/v1", (group,) * 6)
    plan = Plan(CTX, layout, "cuda", build=BUILD)
    gen = torch.Generator(device=dev)
    salts = [_by_group(layout, _dev(salts_of(layout, slots, seed=step))) for step in (0, 1)]

    def table(step: int, g: int) -> object:
        gen.manual_seed(100 * step + g)
        return torch.randint(-128, 128, (slots, width), dtype=torch.int8, device=dev, generator=gen)

    def steps(sleep: int) -> tuple[list[S.StepCommit], list[bool]]:
        out, lagged = [], []
        for step in (0, 1):
            sc, reads, t = G.StepCommitter(plan, step, slots, graphs=graphs), [], None
            for g in range(len(plan.groups)):
                if sleep:
                    with torch.cuda.stream(plan.stream):
                        torch.cuda._sleep(sleep)
                t = table(step, g)
                reads.append(sc.add([t], salts[step][g]))
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
    producer that frees its tables has their memory back for whatever it allocates next. What it catches that the HBM-limit
    test can't: tables held to the next add, which kept each freed group from the forward's allocations in between (1.76
    GB more on the device at the frozen shape's 1 KiB rows); the roots are the reference's."""
    layout = S.h200_layout(layers=2)
    out = S.random_outputs(layout, 3, np.random.default_rng(9))
    plan = Plan(CTX, layout, "cuda", build=BUILD)
    y = salts_of(layout, 3)
    salts = _by_group(layout, _dev(y))
    sc = G.StepCommitter(plan, 4, 3, graphs=graphs)
    first, rest = _cuda(out[:1])[0], _cuda(out[1:])
    alive = [weakref.ref(t) for t in first]
    with torch.cuda.stream(plan.stream):
        torch.cuda._sleep(50_000_000)
    done = sc.add(first, salts[0])
    del first
    done.synchronize()
    held = plan._held
    with held.lock:
        assert held.thread is not None and held.thread.is_alive()
        held.drained.wait_for(lambda: not held.entries)
    assert [a() for a in alive] == [None] * len(alive)
    for tensors, group_salts in zip(rest, salts[1:], strict=True):
        sc.add(tensors, group_salts)
    assert sc.finish().result() == S.commit(CTX, layout, 4, y, out, 3)


def test_a_plan_has_one_step_on_graphs_open() -> None:
    layout = S.edge_layout()
    plan = Plan(CTX, layout, "cuda", build=BUILD)
    out = _cuda(S.random_outputs(layout, 2, np.random.default_rng(1)))
    salts = _by_group(layout, _dev(salts_of(layout, 2)))
    sc = G.StepCommitter(plan, 1, 2, graphs=True)
    sc.add(out[0], salts[0])
    with pytest.raises(ValueError, match="open"):
        G.StepCommitter(plan, 2, 2, graphs=True)
    sc.add(out[1], salts[1])
    sc.finish()
    G.StepCommitter(plan, 2, 2, graphs=True)

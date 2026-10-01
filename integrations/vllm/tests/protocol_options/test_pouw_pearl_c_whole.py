"""Pearl-C's whole-step replay (`pouw_pearl_c_whole`, `pouw_pearl_c_graphs.GraphCalls`' whole-forward half), CPU only: the
hooks on a stand-in of Model Runner V2's graph manager (each key's graphs, eager where a key has none, the calls a capture
recorded handed to the run before each replay, the scratch pass dropped), and the run's bookkeeping at a replay (each call's
slot and offsets, its records appended as `_after` appends them).  The bytes on the card are the e2e's ``whole`` gate's.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from verity_pouw.protocol import Matmul, Shape
from verity_pouw.serving import SERIAL
from verity_vllm.engine.hooks import Hooks
from verity_vllm.linear.api import Arm, PassSpec, Stack
from verity_vllm.protocol_options import pouw_pearl_c_device as PPD
from verity_vllm.protocol_options import pouw_pearl_c_graphs as PG
from verity_vllm.protocol_options.pouw_pearl_c_whole import VLLM, WholeStep

NONE, FULL = "NONE", "FULL"


@dataclass(frozen=True)
class Desc:
    cg_mode: str
    num_tokens: int
    num_reqs: int | None = None


class Manager:
    """Model Runner V2's `CudaGraphManager`, as far as `WholeStep` reads it: FULL graphs of 32 tokens."""

    def __init__(self):
        self.full = Desc(FULL, 32, 32)
        self.graphs = {self.full: "vllm's graph"}
        self._graphs_captured = True
        self.replayed: list = []

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras=0, max_query_len=None):
        if self._graphs_captured and num_tokens <= 32 and uniform_token_count == 1:
            return self.full
        return Desc(NONE, num_tokens, num_reqs)

    def run_fullgraph(self, desc):
        assert desc in self.graphs
        self.replayed.append(self.graphs[desc])
        return "hidden states"


class Runner:
    """The runner's ``capture_model``: per descriptor, a warm-up forward outside the capture, then the captured one, then
    the graph stored."""

    def __init__(self, run):
        self.cudagraph_manager, self.run = Manager(), run

    def capture_model(self):
        mgr = self.cudagraph_manager
        for desc in (mgr.full,):
            self.run.forward(capturing=False)
            self.run.forward(capturing=True)
            mgr.graphs[desc] = f"graph {len(mgr.graphs)}"


class Run:
    """`PearlCDeviceRun`'s side of the hooks: its pass kind's key, the capture's calls, and the replays it is handed."""

    def __init__(self):
        self.graphs, self.hashing, self.retain, self.passes = True, True, None, 0
        self._whole = None
        self.replays: list = []
        self.details = {"passes": []}

    def whole_key(self):
        return None if self.retain or not self.graphs else (self.hashing, frozenset(), "serial")

    def whole_capture_begin(self):
        self._whole = []

    def whole_capture_end(self):
        calls, self._whole = tuple(self._whole or ()), None
        return calls

    def forward(self, capturing):
        for w in (0, 1, 2):
            if self._whole is not None and capturing:
                self._whole.append(PG.WholeCall(w, 32, 4096, 4096, 32, 64))

    def whole_replay(self, calls):
        self.replays.append(calls)


class Executor:
    name, keeps_weight_state, accepts = "pouw:pearl-c", True, frozenset({"bf16"})
    controls, schedules = frozenset({"poison", "omit"}), frozenset({"serial"})

    def __init__(self, run):
        self.run = run

    def refusals(self, sites):
        return []

    def begin(self, spec):
        run = self.run
        run.hashing, run.retain = spec.hashing, spec.retain

        class Pass:
            def __init__(self):
                self.spec = spec

            def end(self):
                run.details["passes"].append({"units": 0})
                return {"units": 0}
        return Pass()


def setup():
    run = Run()
    stack, runner = Stack([], Executor(run)), Runner(run)
    return run, stack, WholeStep(run, stack, runner, Hooks(tag="test")), runner.cudagraph_manager


def step(mgr, tokens=32, uniform=1):
    desc = mgr.dispatch(tokens, tokens, uniform, num_active_loras=0)
    if desc.cg_mode == FULL:
        mgr.run_fullgraph(desc)
    return desc


def test_vllms_own_graphs_serve_while_no_pearl_c_arm_executes():
    run, _, whole, mgr = setup()
    assert whole.key() == VLLM
    assert step(mgr).cg_mode == FULL and mgr.replayed == ["vllm's graph"] and run.replays == []
    assert step(mgr, 8192, None).cg_mode == NONE
    assert whole.take() == {"replayed": 1, "eager": 1, "captured": 0}


def test_an_arm_runs_eagerly_until_prepared_then_replays_its_own_graphs_with_the_captured_calls():
    run, stack, whole, mgr = setup()
    stack.activate(Arm("pearlc", "bf16", "pouw:pearl-c", spec=PassSpec(hashing=True)))
    with stack.pass_():
        assert step(mgr) == Desc(NONE, 32, 32) and mgr.replayed == []
    assert whole.prepare() is True
    assert stack.passes == [{"units": 0}] and run.details["passes"] == [{"units": 0}]   # the timed pass's, not the scratch's
    assert whole.prepare() is False and len(stack.passes) == 1
    with stack.pass_():
        assert step(mgr).cg_mode == FULL
    assert mgr.replayed == ["graph 0"]
    assert run.replays == [tuple(PG.WholeCall(w, 32, 4096, 4096, 32, 64) for w in (0, 1, 2))]
    assert whole.take()["captured"] == 1


def test_each_pass_kind_has_its_own_graphs_and_vllms_stay_its_own():
    run, stack, whole, mgr = setup()
    for hashing in (True, False):
        stack.activate(Arm("pearlc" if hashing else "pearlc-nohash", "bf16", "pouw:pearl-c", spec=PassSpec(hashing=hashing)))
        assert whole.prepare() is True
    assert set(whole.sets) == {VLLM, (True, frozenset(), "serial"), (False, frozenset(), "serial")}
    stack.activate(None)
    step(mgr)
    assert mgr.replayed == ["vllm's graph"] and run.replays == []


def test_retained_passes_graphs_off_and_forwards_outside_a_pass_run_eagerly():
    run, stack, whole, mgr = setup()
    stack.activate(Arm("pearlc", "bf16", "pouw:pearl-c", spec=PassSpec(hashing=True)))
    whole.prepare()
    assert whole.key() is None and step(mgr).cg_mode == NONE
    with stack.pass_(PassSpec(hashing=True, retain="/tmp/x")):
        assert step(mgr).cg_mode == NONE
    run.graphs = False
    with stack.pass_():
        assert step(mgr).cg_mode == NONE
    assert mgr.replayed == [] and run.replays == [] and mgr._graphs_captured


def test_whole_step_refuses_an_engine_without_decode_graphs_it_can_capture_whole():
    run = Run()
    runner = Runner(run)
    runner.cudagraph_manager.graphs = {}
    with pytest.raises(ValueError, match="no CUDA graphs"):
        WholeStep(run, Stack([], Executor(run)), runner, Hooks())
    runner.cudagraph_manager.graphs = {Desc(FULL, 24, 24): "g"}
    with pytest.raises(ValueError, match=r"\[24\]"):
        WholeStep(run, Stack([], Executor(run)), runner, Hooks())


def test_uninstalling_the_hooks_puts_the_managers_methods_back():
    run = Run()
    runner, hooks = Runner(run), Hooks()
    WholeStep(run, Stack([], Executor(run)), runner, hooks)
    assert "dispatch" in vars(runner.cudagraph_manager)
    hooks.uninstall()
    assert "dispatch" not in vars(runner.cudagraph_manager) and "run_fullgraph" not in vars(runner.cudagraph_manager)


# ---- the run's half ----------------------------------------------------------------------------------------------------

class Event:
    def __init__(self):
        self.synced = self.recorded = 0

    def synchronize(self):
        self.synced += 1

    def record(self, stream=None):
        self.recorded += 1


class Calls(PG.GraphCalls):
    """`GraphCalls` over CPU tensors: three weights, keys that name their call."""

    def __init__(self, torch):
        self.entry, self.hashing, self.graphs, self.omit = "test", True, True, frozenset()
        self.schedule, self.capacity = PPD.schedule_of(SERIAL, PPD.scheme_of(PPD.SCHEMES[2])), PPD.LEAF_CAPACITY
        self.order = {7: 0, 8: 1, 9: 2}
        self.roots = {w: bytes([w]) * 32 for w in self.order}
        self.keys = lambda root, index, m16: bytes([root[0], index, m16]) * 45 + b"\0"
        self.key_slots = torch.zeros((3, PG.SLOT_BYTES), dtype=torch.uint8)
        self.offs = torch.zeros((3, 3), dtype=torch.int64)
        self._host_keys, self._host_offs = torch.zeros_like(self.key_slots), torch.zeros_like(self.offs)
        self._slots_copied, self._predicted = Event(), {}
        self.roots_buf, self.flag_buf = torch.zeros(32 * 8, dtype=torch.uint8), torch.zeros((256, 2), dtype=torch.uint8)
        self.cur = PPD._Pass(None)


def test_a_replay_uploads_every_calls_slot_once_and_appends_its_records():
    torch = pytest.importorskip("torch")
    run = Calls(torch)
    run.cur.matmuls = [Matmul(0, Shape(16, 4096, 4096), 8)]
    run.cur.tiles, run.cur.flag_rows = 5, 16
    run._predicted = {8: ("stale",), 9: ("stale",)}
    calls = (PG.WholeCall(9, 32, 4096, 1024, 32, 8), PG.WholeCall(7, 32, 1024, 4096, 32, 32))
    run.whole_replay(calls)
    assert run.cur.matmuls[1:] == [Matmul(1, Shape(32, 4096, 1024), 9), Matmul(2, Shape(32, 1024, 4096), 7)]
    assert (run.cur.tiles, run.cur.flag_rows) == (5 + 8 + 32, 16 + 64)
    keys, offs = run.key_slots.numpy(), run.offs.numpy()
    assert keys[2, :136].tobytes() == run.keys(run.roots[9], 1, 32) and keys[0, :136].tobytes() == run.keys(run.roots[7], 2, 32)
    assert offs[2].tolist() == [32 * 5, 2 * 16, 32 * 1] and offs[0].tolist() == [32 * 13, 2 * 48, 32 * 2]
    assert run._predicted == {9: (1, 32, 5, 16), 7: (2, 32, 13, 48)}     # 8's slot was rewritten from the host copy
    assert (run._slots_copied.synced, run._slots_copied.recorded) == (1, 1)


def test_a_replay_without_hashing_appends_the_matmuls_only():
    torch = pytest.importorskip("torch")
    run = Calls(torch)
    run.hashing = False
    run.whole_replay((PG.WholeCall(7, 32, 1024, 4096, 32, 32),))
    assert run.cur.matmuls == [Matmul(0, Shape(32, 1024, 4096), 7)] and (run.cur.tiles, run.cur.flag_rows) == (0, 0)
    assert not run.key_slots.any() and run._slots_copied.recorded == 0


def test_a_replay_over_the_passs_capacity_is_refused():
    torch = pytest.importorskip("torch")
    two = (PG.WholeCall(7, 32, 1024, 4096, 32, 32), PG.WholeCall(8, 32, 1024, 4096, 32, 32))
    for bound in ("capacity", "roots_buf", "flag_buf"):
        run = Calls(torch)
        setattr(run, bound, {"capacity": 40, "roots_buf": torch.zeros(32), "flag_buf": torch.zeros((40, 2))}[bound])
        with pytest.raises(RuntimeError, match="capacity"):
            run.whole_replay(two)
        assert run.cur.matmuls == []


def test_inside_vllms_capture_a_call_is_recorded_and_launched_into_it(monkeypatch):
    torch = pytest.importorskip("torch")
    run = Calls(torch)
    into = []
    monkeypatch.setattr(PG, "_capturing", lambda: True)
    monkeypatch.setattr(run, "_into", lambda *a: into.append(a) or ("y", True), raising=False)
    sh = type("Sh", (), {"m16": 32, "k": 1024, "n": 4096})()
    run.whole_capture_begin()
    assert run._graph_call(7, sh, "pipe", 0, 32, 32, 0, None) == ("y", True)
    with pytest.raises(RuntimeError, match="filler rows"):
        run._graph_call(8, sh, "pipe", 0, 32, 20, 1, None)
    assert run.whole_capture_end() == (PG.WholeCall(7, 32, 1024, 4096, 32, 32),) and len(into) == 1
    run.whole_capture_begin()
    run._graph_call(7, sh, "pipe", 0, 32, 32, 0, None)
    run._graph_call(7, sh, "pipe", 0, 32, 32, 1, None)
    with pytest.raises(RuntimeError, match="twice"):
        run.whole_capture_end()


def test_a_pass_replays_whole_only_when_graphed_serial_and_not_retained():
    torch = pytest.importorskip("torch")
    run = Calls(torch)
    assert run.whole_key() == (True, frozenset(), run.schedule.name)
    run.cur = PPD._Pass(Path("/tmp/retained"))
    assert run.whole_key() is None
    run.cur, run.graphs = PPD._Pass(None), False
    assert run.whole_key() is None
    run.graphs, run.schedule = True, PPD.schedule_of("deferred", PPD.scheme_of(PPD.SCHEMES[2]))
    assert run.whole_key() is None

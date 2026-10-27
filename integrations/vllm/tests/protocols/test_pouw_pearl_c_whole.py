"""Pearl-C's whole-step replay (`pouw_pearl_c_whole`, `pouw_pearl_c_graphs.GraphCalls`' whole-forward half), CPU only: the
hooks on a stand-in of Model Runner V2's graph manager (each key's graphs, a retained pass's kind among them, eager where a
key has none, the calls a capture recorded handed to the run around each replay, the scratch pass dropped), and the run's
bookkeeping (each call's slot and offsets, the next forward's staged after a replay and uploaded by the next only from where
they were staged for, its records appended as `_after` appends them, a retained capture's stages and the records and files
kept from them, and a deferred capture's forks and joins over stand-in streams).  The bytes on the card are the e2e's
``whole`` gate's.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from verity.core.protocols.pouw.protocol import Matmul, Shape
from verity.core.protocols.pouw.serving import SERIAL
from verity_vllm.engine.hooks import Hooks
from verity_vllm.protocols.linear.api import Arm, PassSpec, Stack
from verity_vllm.protocols.pouw.pearl_c import passes as PP
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.pearl_c import graphs as PG
from verity_vllm.protocols.pouw.pearl_c.whole import VLLM, WholeStep

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
        self.replayed: list = []
        self.begun: list = []
        self.details = {"passes": []}
        self.eager = 0

    def eager_forward(self):
        self.eager += 1

    def whole_key(self, retained=None):
        retained = self.retain is not None if retained is None else retained
        return (self.hashing, frozenset(), "serial", self.hashing and retained) if self.graphs else None

    def whole_capture_begin(self, key=None):
        self._whole = []
        self.begun.append(key)

    def whole_capture_end(self):
        calls, self._whole = tuple(self._whole or ()), None
        return calls

    def forward(self, capturing):
        for w in (0, 1, 2):
            if self._whole is not None and capturing:
                self._whole.append(PG.WholeCall(w, 32, 4096, 4096, 32, 64))

    def whole_replay(self, calls):
        self.replays.append(calls)
        return len(self.replays) > 1

    def whole_replayed(self, calls):
        self.replayed.append(calls)


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
    assert whole.take() == {"replayed": 1, "staged": 0, "eager": 1, "captured": 0}


CALLS = tuple(PG.WholeCall(w, 32, 4096, 4096, 32, 64) for w in (0, 1, 2))
HASHED, RETAINED = (True, frozenset(), "serial", False), (True, frozenset(), "serial", True)


def test_an_arm_runs_eagerly_until_prepared_then_replays_its_own_graphs_with_the_captured_calls():
    run, stack, whole, mgr = setup()
    stack.activate(Arm("pearlc", "bf16", "pouw:pearl-c", spec=PassSpec(hashing=True)))
    with stack.pass_():
        assert step(mgr) == Desc(NONE, 32, 32) and mgr.replayed == []
    assert whole.prepare() is True
    assert stack.passes == [{"units": 0}] and run.details["passes"] == [{"units": 0}]   # the timed pass's, not the scratch's
    assert run.begun == [HASHED, HASHED, RETAINED, RETAINED]    # each kind's capture, and again after its descriptor's graph
    assert whole.prepare() is False and len(stack.passes) == 1
    with stack.pass_():
        assert step(mgr).cg_mode == FULL and step(mgr).cg_mode == FULL
    assert mgr.replayed == ["graph 0"] * 2 and run.replays == run.replayed == [CALLS] * 2
    assert whole.take() == {"replayed": 2, "staged": 1, "eager": 1, "captured": 2}


def test_each_pass_kind_has_its_own_graphs_and_vllms_stay_its_own():
    run, stack, whole, mgr = setup()
    for hashing in (True, False):
        stack.activate(Arm("pearlc" if hashing else "pearlc-nohash", "bf16", "pouw:pearl-c", spec=PassSpec(hashing=hashing)))
        assert whole.prepare() is True
    assert set(whole.sets) == {VLLM, HASHED, RETAINED, (False, frozenset(), "serial", False)}
    stack.activate(None)
    step(mgr)
    assert mgr.replayed == ["vllm's graph"] and run.replays == []


def test_a_retained_pass_replays_its_kinds_own_capture():
    run, stack, whole, mgr = setup()
    stack.activate(Arm("pearlc", "bf16", "pouw:pearl-c", spec=PassSpec(hashing=True)))
    whole.prepare()
    with stack.pass_(PassSpec(hashing=True, retain="/tmp/x")):
        assert whole.key() == RETAINED and step(mgr).cg_mode == FULL
    assert whole.sets[RETAINED] is not whole.sets[HASHED] and run.replays == run.replayed == [CALLS]


def test_enabled_off_graphs_off_and_forwards_outside_a_pass_run_eagerly():
    run, stack, whole, mgr = setup()
    stack.activate(Arm("pearlc", "bf16", "pouw:pearl-c", spec=PassSpec(hashing=True)))
    whole.prepare()
    assert whole.key() is None and step(mgr).cg_mode == NONE
    whole.enabled = False
    with stack.pass_():
        assert step(mgr).cg_mode == NONE
    assert whole.prepare() is False
    whole.enabled, run.graphs = True, False
    with stack.pass_():
        assert step(mgr).cg_mode == NONE
    assert mgr.replayed == [] and run.replays == [] and mgr._graphs_captured


def test_the_run_hears_of_each_eager_forward_of_a_pearl_c_pass_and_of_nothing_else():
    """`PearlCDeviceRun.eager_forward` (``whole_eager``'s switch) runs once per forward a Pearl-C pass runs eagerly: before
    the arm's graphs exist, and for a batch they lack (a prefill's); never for a replayed step, vLLM's own eager steps, a
    forward outside a pass or with `enabled` off, or while `prepare` captures."""
    run, stack, whole, mgr = setup()
    step(mgr, 8192, None)
    stack.activate(Arm("pearlc", "bf16", "pouw:pearl-c", spec=PassSpec(hashing=True)))
    with stack.pass_():
        step(mgr)
    assert run.eager == 1
    whole.prepare()
    assert run.eager == 1
    with stack.pass_():
        assert step(mgr).cg_mode == FULL and run.eager == 1
        assert step(mgr, 640, None).cg_mode == NONE and run.eager == 2
    step(mgr, 640, None)
    whole.enabled = False
    with stack.pass_():
        step(mgr, 640, None)
    whole.enabled, whole.preparing = True, True
    with stack.pass_():
        step(mgr, 640, None)
    assert run.eager == 2 and whole.take()["eager"] == 6


def test_whole_step_refuses_an_engine_without_decode_graphs_it_can_capture_whole():
    run = Run()
    runner = Runner(run)
    runner.cudagraph_manager.graphs = {}
    with pytest.raises(ValueError, match="no CUDA graphs"):
        WholeStep(run, Stack([], Executor(run)), runner, Hooks())
    runner.cudagraph_manager.graphs = {Desc(FULL, 24, 24): "g", Desc(FULL, 250, 250): "g", Desc(FULL, 264, 264): "g"}
    with pytest.raises(ValueError, match=r"\[264\]"):            # 272 rows with its filler, over GRAPH_ROWS
        WholeStep(run, Stack([], Executor(run)), runner, Hooks())
    runner.cudagraph_manager.graphs = {Desc(FULL, 24, 24): "g", Desc(FULL, 250, 250): "g"}
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


class Calls(PG.GraphCalls, PP.EpochCalls):
    """`GraphCalls` over CPU tensors: three weights, keys that name their call."""

    def __init__(self, torch):
        self.entry, self.hashing, self.graphs, self.omit = "test", True, True, frozenset()
        self.schedule, self.capacity = PPD.schedule_of(SERIAL, PPD.scheme_of(PPD.SCHEMES[0])), PPD.LEAF_CAPACITY
        self.order, self.nb = {7: 0, 8: 1, 9: 2}, 64
        self.roots = {w: bytes([w]) * 32 for w in self.order}
        self.keys = lambda root, index, m16: bytes([root[0], index, m16]) * 56
        self.key_slots = torch.zeros((3, PG.SLOT_BYTES), dtype=torch.uint8)
        self.offs = torch.zeros((3, 3), dtype=torch.int64)
        self._host_keys, self._host_offs = torch.zeros_like(self.key_slots), torch.zeros_like(self.offs)
        self._slots_copied, self._predicted = Event(), {}
        self.roots_buf, self.flag_buf = torch.zeros(64 * 8, dtype=torch.uint8), torch.zeros((256, 2), dtype=torch.uint8)
        self.cur = PPD._Pass(None)


def test_a_replay_uploads_every_calls_slot_once_and_appends_its_records():
    torch = pytest.importorskip("torch")
    run = Calls(torch)
    run.cur.matmuls = [Matmul(0, Shape(16, 4096, 4096), 8)]
    run.cur.tiles, run.cur.flag_rows = 5, 16
    run._predicted = {8: ("stale",), 9: ("stale",)}
    calls = (PG.WholeCall(9, 32, 4096, 1024, 32, 8), PG.WholeCall(7, 32, 1024, 4096, 32, 32))
    assert run.whole_replay(calls) is False and run.cur.matmuls[1:] == []
    run.whole_replayed(calls)
    assert run.cur.matmuls[1:] == [Matmul(1, Shape(32, 4096, 1024), 9), Matmul(2, Shape(32, 1024, 4096), 7)]
    assert (run.cur.tiles, run.cur.flag_rows) == (5 + 8 + 32, 16 + 64)
    keys, offs = run.key_slots.numpy(), run.offs.numpy()
    assert keys[2, :168].tobytes() == run.keys(run.roots[9], 1, 32) and keys[0, :168].tobytes() == run.keys(run.roots[7], 2, 32)
    assert offs[2].tolist() == [64 * 5, 2 * 16, 64 * 1] and offs[0].tolist() == [64 * 13, 2 * 48, 64 * 2]
    assert run._predicted == {9: (1, 32, 5, 16), 7: (2, 32, 13, 48)}     # 8's slot was rewritten from the host copy
    assert (run._slots_copied.synced, run._slots_copied.recorded) == (1, 1)


def test_a_replay_uploads_the_slots_the_last_one_staged_only_from_where_they_were_staged_for():
    torch = pytest.importorskip("torch")
    run, made = Calls(torch), []
    make = run.keys
    run.keys = lambda *a: made.append(a[1:]) or make(*a)
    calls = (PG.WholeCall(9, 32, 4096, 1024, 32, 8), PG.WholeCall(7, 32, 1024, 4096, 32, 32))
    run.whole_replay(calls)
    assert made == [(0, 32), (1, 32)]
    run.whole_replayed(calls)
    assert made[2:] == [(2, 32), (3, 32)]                    # the next forward's, while the GPU runs this one
    assert run.whole_replay(calls) is True and made[4:] == []
    run.whole_replayed(calls)
    assert run.cur.matmuls[2:] == [Matmul(2, Shape(32, 4096, 1024), 9), Matmul(3, Shape(32, 1024, 4096), 7)]
    keys, offs = run.key_slots.numpy(), run.offs.numpy()
    assert keys[2, :168].tobytes() == run.keys(run.roots[9], 2, 32) and keys[0, :168].tobytes() == run.keys(run.roots[7], 3, 32)
    assert offs[2].tolist() == [64 * 40, 2 * 64, 64 * 2] and offs[0].tolist() == [64 * 48, 2 * 96, 64 * 3]
    assert run._predicted == {9: (2, 32, 40, 64), 7: (3, 32, 48, 96)} and run._slots_copied.recorded == 2
    run.cur = PPD._Pass(None)                                 # a new pass is elsewhere (numbered on from the epoch's 4)
    assert run.whole_replay(calls) is False and offs[2].tolist() == [0, 0, 0]
    run.whole_replayed(calls)
    assert run.whole_replay(calls[:1]) is False               # and so do other calls
    assert keys[2, :168].tobytes() == run.keys(run.roots[9], 4 + 2, 32)


def test_a_replay_without_hashing_appends_the_matmuls_only():
    torch = pytest.importorskip("torch")
    run = Calls(torch)
    run.hashing = False
    calls = (PG.WholeCall(7, 32, 1024, 4096, 32, 32),)
    assert run.whole_replay(calls) is False
    run.whole_replayed(calls)
    assert run.cur.matmuls == [Matmul(0, Shape(32, 1024, 4096), 7)] and (run.cur.tiles, run.cur.flag_rows) == (0, 0)
    assert not run.key_slots.any() and run._slots_copied.recorded == 0 and run._next is None


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
    assert run._graph_call(8, sh, "pipe", 0, 32, 20, 1, None) == ("y", True)     # filler rows too: vLLM's batch fixes m
    assert run.whole_capture_end() == (PG.WholeCall(7, 32, 1024, 4096, 32, 32), PG.WholeCall(8, 20, 1024, 4096, 32, 32))
    assert len(into) == 2
    run.whole_capture_begin()
    run._graph_call(7, sh, "pipe", 0, 32, 32, 0, None)
    run._graph_call(7, sh, "pipe", 0, 32, 32, 1, None)
    with pytest.raises(RuntimeError, match="twice"):
        run.whole_capture_end()


def test_a_graph_screens_its_filler_rows_call_only_inside_a_whole_capture_where_it_is_that_ms_own():
    torch = pytest.importorskip("torch")
    run = Calls(torch)
    sh = SimpleNamespace(m16=32)
    full, filler = run._graph_key(7, sh, 32), run._graph_key(7, sh, 20)
    assert full[1] is True and filler[1] is False and filler[0] == run._graph_key(7, sh, 17)[0]   # one graph for every m
    run.whole_capture_begin()                                 # from the capture's warm-up on
    inside = run._graph_key(7, sh, 20)
    assert inside[1] is True and inside[0] not in (full[0], filler[0], run._graph_key(7, sh, 17)[0])
    assert run._graph_key(7, sh, 32) == full
    run.whole_capture_end()
    assert run._graph_key(7, sh, 20) == filler
    run.hashing = False
    run.whole_capture_begin()
    assert run._graph_key(7, sh, 20)[1] is False              # nothing to screen without hashing


def test_a_pass_replays_whole_only_when_graphed_and_serial_and_its_key_says_whether_it_is_retained():
    torch = pytest.importorskip("torch")
    run = Calls(torch)
    name = run.schedule.name
    assert run.whole_key() == (True, frozenset(), name, False, False) and run.whole_key(True) == (True, frozenset(), name, True, False)
    run.cur = PPD._Pass(Path("/tmp/retained"))
    assert run.whole_key() == (True, frozenset(), name, True, False) and run.whole_key(False) == (True, frozenset(), name, False, False)
    run.whole_defer = True
    assert run.whole_key() == (True, frozenset(), name, True, True)
    run.whole_defer_a = True
    assert run.whole_key() == (True, frozenset(), name, True, True, "a_tree")
    run.whole_fork_stats = True
    assert run.whole_key() == (True, frozenset(), name, True, True, "a_tree", "stats")
    run.whole_tree_sm = True
    assert run.whole_key() == (True, frozenset(), name, True, True, "a_tree", "tree_sm", "stats")
    run.whole_prefetch_b = True
    assert run.whole_key() == (True, frozenset(), name, True, True, "a_tree", "tree_sm", "stats", "prefetch_b")
    run.whole_shape_joins = True
    assert run.whole_key() == (True, frozenset(), name, True, True, "a_tree", "tree_sm", "stats", "prefetch_b", "shape_joins")
    run.hashing = False
    assert run.whole_key() == (False, frozenset(), name, False, False)        # nothing to defer without hashing
    run.cur, run.graphs = PPD._Pass(None), False
    assert run.whole_key() is None
    run.graphs, run.schedule = True, PPD.schedule_of("deferred", PPD.scheme_of(PPD.SCHEMES[0]))
    assert run.whole_key() is None


def test_a_retained_kinds_capture_copies_each_calls_a_words_and_tree_to_the_stage_its_warm_up_made(monkeypatch):
    torch = pytest.importorskip("torch")
    run, capturing = Calls(torch), [False]
    monkeypatch.setattr(PG, "_capturing", lambda: capturing[0])
    for name in ("_into", "_replay", "_slot"):
        monkeypatch.setattr(run, name, lambda *a: ("y", True))
    sh = SimpleNamespace(m16=32, k=8, n=16, xa=torch.zeros(48, 8), b={"tree_a": torch.zeros(100, dtype=torch.uint8)})
    sh.buf = sh.b.__getitem__
    pipe = SimpleNamespace(size={"tree_a": 64})
    run.whole_capture_begin(run.whole_key(True))
    run._graph_call(7, sh, pipe, 0, 4, 32, 0, None)          # vLLM's warm-up, outside the capture
    xa, tree = run._stages[7, 32]
    assert xa.shape == (32, 8) and tree.shape == (64,)
    sh.xa[:32], sh.b["tree_a"][:64] = 1.5, 7
    capturing[0] = True
    assert run._graph_call(7, sh, pipe, 0, 4, 32, 0, None) == ("y", True)
    assert torch.equal(xa, sh.xa[:32]) and torch.equal(tree, sh.b["tree_a"][:64])
    assert run.whole_capture_end() == (PG.WholeCall(7, 32, 8, 16, 32, 4),) and not run._whole_retained
    sh.xa[:32] = 2.0
    run.whole_capture_begin(run.whole_key(False))
    run._graph_call(7, sh, pipe, 0, 4, 32, 0, None)
    run.whole_capture_end()
    assert float(xa[0, 0]) == 1.5                            # an unretained kind's capture leaves the stages alone
    run._graphs = {}
    assert [t.dtype for t in run._graph_buffers()] == [torch.uint8, torch.uint8]


def test_an_enabled_statement_is_in_the_key_stages_each_captured_call_and_commits_after_each_replay(monkeypatch):
    torch = pytest.importorskip("torch")
    run, capturing, log = Calls(torch), [False], []
    monkeypatch.setattr(PG, "_capturing", lambda: capturing[0])
    for name in ("_into", "_replay", "_slot"):
        monkeypatch.setattr(run, name, lambda *a: ("y", True))
    run.statement = SimpleNamespace(enabled=True, prepare=lambda w, sh: log.append(("prepare", w)),
                                    stage=lambda w, sh: log.append(("stage", w)),
                                    after=lambda calls, at: log.append(("after", tuple(c.w for c in calls), at)))
    name = run.schedule.name
    assert run.whole_key() == (True, frozenset(), name, False, False, "statement")
    sh = SimpleNamespace(m16=32, k=8, n=16, xa=torch.zeros(48, 8), b={"tree_a": torch.zeros(100, dtype=torch.uint8)})
    sh.buf = sh.b.__getitem__
    run.whole_capture_begin(run.whole_key())
    run._graph_call(7, sh, "pipe", 0, 4, 32, 0, None)          # vLLM's warm-up, outside the capture
    capturing[0] = True
    run._graph_call(7, sh, "pipe", 0, 4, 32, 0, None)
    calls = run.whole_capture_end()
    assert log == [("prepare", 7), ("stage", 7)]
    run.cur.tiles = 5
    run.whole_replay(calls)
    run.whole_replayed(calls)
    assert log[-1] == ("after", (7,), 5) and run.cur.tiles == 9   # the forward's first tile, where the pass was
    run.statement.enabled = False
    assert run.whole_key() == (True, frozenset(), name, False, False)
    run.whole_capture_begin(run.whole_key())
    run._graph_call(7, sh, "pipe", 0, 4, 32, 0, None)
    run.whole_capture_end()
    run.whole_replay(calls)
    run.whole_replayed(calls)
    run.statement.enabled, run.hashing = True, False
    assert run.whole_key() == (False, frozenset(), name, False, False)  # nothing to commit without hashing
    assert len(log) == 3


def test_after_a_retained_replay_each_call_keeps_its_stage_and_flags_as_a_call_made_alone_would(tmp_path):
    torch = pytest.importorskip("torch")
    run = Calls(torch)
    run.weights = {7: SimpleNamespace(excluded=(3,)), 9: SimpleNamespace(excluded=())}
    run.cur = PPD._Pass(tmp_path)
    run._stages = {(9, 32): (torch.full((32, 4), 2.0), torch.full((16,), 4, dtype=torch.uint8)),
                   (7, 32): (torch.full((32, 8), 1.0), torch.full((64,), 3, dtype=torch.uint8))}
    calls = (PG.WholeCall(9, 32, 4, 16, 32, 2), PG.WholeCall(7, 32, 8, 16, 32, 4))
    run.flag_buf[32 + 5, 0] = 1                               # weight 7's call: row 5 fails the screen
    run.whole_capture_begin(run.whole_key(True))              # replays follow a capture, which leaves its state behind
    run.whole_capture_end()
    run.whole_replay(calls)
    run.whole_replayed(calls)
    want = []
    for mm, (w, ex) in zip(run.cur.matmuls, ((9, ()), (7, (5,))), strict=True):
        want.append(PPD.PW.UnitRecord(mm, 32, mm.shape.k, ex, run.weights[w].excluded))
    assert run.cur.records == want
    assert [f["tree"]["path"] for f in run.cur.files] == ["calls/0.tree", "calls/1.tree"]
    assert (tmp_path / "calls/1.tree").read_bytes() == bytes([3]) * 64
    assert (tmp_path / "calls/0.rows.bf16").read_bytes() == torch.full((32, 4), 2.0).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    run.cur = PPD._Pass(None)
    run.whole_replay(calls)
    run.whole_replayed(calls)                                 # an unretained pass keeps nothing
    assert run.cur.records == [] and run.cur.files == []


def test_a_retained_kinds_capture_at_two_sizes_keeps_each_calls_stage_at_its_m16(monkeypatch, tmp_path):
    """vLLM's ``capture_model`` captures a forward at each of its batch sizes, the largest first, each after a warm-up outside
    the capture.  A call's A words and tree are sized by its m₁₆ (a shape's buffers and a pipeline per m₁₆), so the second
    size's capture copies into a stage of its own (one stage a weight, sized at 128 tokens, refuses the 32-token
    capture's copy), and each size's replay keeps the records and files its calls keep made alone."""
    torch = pytest.importorskip("torch")
    run, capturing = Calls(torch), [False]
    run.weights = {7: SimpleNamespace(k=8, excluded=(1,)), 8: SimpleNamespace(k=4, excluded=())}
    monkeypatch.setattr(PG, "_capturing", lambda: capturing[0])
    for name in ("_into", "_replay", "_slot"):
        monkeypatch.setattr(run, name, lambda *a: ("y", True))
    sizes = ((64, 64), (16, 8))                               # (m₁₆, m): the larger size first, the smaller with filler rows
    sh = {(w, m16): SimpleNamespace(m16=m16, k=wt.k, n=16, xa=torch.zeros(m16 + 16, wt.k),
                                    b={"tree_a": torch.zeros(4 * m16, dtype=torch.uint8)})
          for w, wt in run.weights.items() for m16, _ in sizes}
    for s in sh.values():
        s.buf = s.b.__getitem__
    pipe = {m16: SimpleNamespace(size={"tree_a": 2 * m16}) for m16, _ in sizes}
    calls = {}
    run.whole_capture_begin(run.whole_key(True))
    for m16, m in sizes:
        for i, w in enumerate(run.weights):                   # vLLM's warm-up forward, outside the capture
            run._graph_call(w, sh[w, m16], pipe[m16], 0, m16 // 4, m, i, None)
        capturing[0] = True
        for i, w in enumerate(run.weights):                   # each call's A words and tree as its steps leave them
            s = sh[w, m16]
            s.xa[:m16] = torch.arange(m16 * s.k, dtype=torch.float32).reshape(m16, s.k) + 1000 * w + m16
            s.b["tree_a"][:] = (w + m16) % 251
            run._graph_call(w, s, pipe[m16], 0, m16 // 4, m, i, None)
        capturing[0] = False
        calls[m16] = run.whole_capture_end()                  # vLLM stores the size's graph: the next one's calls begin
        run.whole_capture_begin(run.whole_key(True))
    run.whole_capture_end()
    assert {k: tuple(t.shape for t in v) for k, v in run._stages.items()} == {
        (7, 64): ((64, 8), (128,)), (8, 64): ((64, 4), (128,)), (7, 16): ((16, 8), (32,)), (8, 16): ((16, 4), (32,))}
    run.flag_buf[3, 0] = 1                                    # the first call's row 3 fails the screen
    for m16, m in sizes:
        assert calls[m16] == tuple(PG.WholeCall(w, m, wt.k, 16, m16, m16 // 4) for w, wt in run.weights.items())
        run.cur, alone = PPD._Pass(tmp_path / f"whole-{m16}"), PPD._Pass(tmp_path / f"alone-{m16}")
        run.whole_replay(calls[m16])
        run.whole_replayed(calls[m16])
        for i, (mm, w) in enumerate(zip(run.cur.matmuls, run.weights, strict=True)):
            run._retained(alone, mm, m, run.flag_buf[i * m:(i + 1) * m], *run._kept(sh[w, m16], pipe[m16]))
        assert run.cur.records == alone.records and run.cur.records[0].excluded_a == (3,)
        assert run.cur.files == alone.files and len(alone.files) == 2
        for f in (p for x in alone.files for p in (x["rows"]["path"], x["tree"]["path"])):
            assert (tmp_path / f"whole-{m16}" / f).read_bytes() == (tmp_path / f"alone-{m16}" / f).read_bytes()


class Lane:
    """A stand-in of torch's CUDA stream: its handle, and each wait it is told to make and event it records, in `log`."""

    def __init__(self, log, name="side", device=None):
        self.log, self.name, self.cuda_stream = log, name, {"main": 1, "side": 2, "tree": 3, "stats": 4}[name]

    def wait_stream(self, other):
        self.log.append(f"{self.name} waits for {other.name}")

    def record_event(self):
        event = f"{self.name} event {sum(' event ' in x and 'waits' not in x for x in self.log)}"
        self.log.append(event)
        return event

    def wait_event(self, event):
        at = f"{event.at}'s mark" if isinstance(event, Mark) else event
        self.log.append(f"{self.name} waits for {at}")


class Mark:
    """A stand-in of torch's CUDA event: where it was recorded, the record in the streams' log."""

    def __init__(self, log):
        self.log, self.at = log, None

    def record(self, stream):
        self.at = stream.name
        self.log.append(f"{stream.name} marks")


def _lanes(monkeypatch, torch, log):
    """torch.cuda's current stream, `Stream` and `stream` over `Lane`s and `Event` over `Mark`s, the current one the
    capturing stream ("main"); the streams the run makes are the side stream, then the tree stream, then the stats stream."""
    current, made = [Lane(log, "main")], iter(("side", "tree", "stats"))

    @contextlib.contextmanager
    def stream(s):
        held, current[0] = current[0], s
        try:
            yield
        finally:
            current[0] = held
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current[0])
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: Lane(log, next(made)))
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: Mark(log))
    return current


def test_a_deferred_capture_runs_each_calls_tile_hashing_beside_the_next_calls_a_tree(monkeypatch):
    torch = pytest.importorskip("torch")
    run, log = Calls(torch), []
    run.whole_defer, run.device, run._graphs = True, "cuda", {}
    run.kernel = SimpleNamespace(dev=SimpleNamespace(stream=None))
    current = _lanes(monkeypatch, torch, log)
    monkeypatch.setattr(PG, "_capturing", lambda: True)
    shapes = {7: SimpleNamespace(m16=32, k=8, n=16), 8: SimpleNamespace(m16=32, k=8, n=32)}
    shapes[9] = shapes[8]
    names = ("a_tree", "forming", "gemm", "hash_leaf", "tree_leaves", "scatter")

    def step(w, name):
        return lambda: log.append(f"{w}:{name} on {current[0].name} ({run.kernel.dev.stream})")
    for w, sh in shapes.items():
        run._graphs[run._graph_key(w, sh, 32)[0]] = PG._Graph(torch.zeros(1), torch.zeros(1), "graph",
                                                               [step(w, x) for x in names], (1, 3))

    def forward():
        run.whole_capture_begin()
        for i, (w, sh) in enumerate(shapes.items()):
            run.join_before(sh)                               # as `_gemm` does, before it writes the call's A words
            run._graph_call(w, sh, SimpleNamespace(b={}), 1, 4, 32, i, None)
        return run.whole_capture_end()

    calls = forward()
    side = lambda w: [f"{w}:{x} on side (2)" for x in names[3:]]   # noqa: E731
    assert log == ["7:a_tree on main (1)", "7:forming on main (1)", "7:gemm on main (1)", "side waits for main", *side(7),
                   "8:a_tree on main (1)", "main waits for side", "8:forming on main (1)", "8:gemm on main (1)",
                   "side waits for main", *side(8),
                   "main waits for side",                     # 9's call is of 8's shape: joined before it writes
                   "9:a_tree on main (1)", "9:forming on main (1)", "9:gemm on main (1)", "side waits for main", *side(9),
                   "main waits for side"]                     # the forward's last call joins its own
    assert [c.w for c in calls] == [7, 8, 9] and run._pending is None and run.kernel.dev.stream is None
    log.clear()
    run.whole_defer = False
    forward()
    assert log == [f"{w}:{x} on main (1)" for w in shapes for x in names]


class Stage:
    """A stand-in of a retained capture's stage tensor: each copy into it, in `log`, with the stream it runs on."""

    def __init__(self, log, current, name):
        self.log, self.current, self.name = log, current, name

    def copy_(self, other):
        self.log.append(f"{self.name} staged on {self.current[0].name}")


def test_a_deferred_a_capture_runs_each_calls_a_tree_beside_its_forming_and_gemm_and_joins_it_before_its_root_is_read(
        monkeypatch):
    """Under ``whole_defer_a`` a call's A tree (``tree_seed_a``, after its per-row keying, ``rows_a``) goes on the tree
    stream beside its forming and GEMM.  Its side steps' scatter, which copies A's root to the pass, waits for it (the
    tile hashing before it, which reads nothing of A's tree, doesn't), and so does a retained capture's copy of A's tree
    to its stage.  The forward's last call joins the tree stream, and a call
    joining a call of its shape joins that call's tree through its side steps."""
    torch = pytest.importorskip("torch")
    run, log = Calls(torch), []
    run.whole_defer, run.whole_defer_a, run.device, run._graphs = True, True, "cuda", {}
    run.kernel = SimpleNamespace(dev=SimpleNamespace(stream=None))
    current = _lanes(monkeypatch, torch, log)
    monkeypatch.setattr(PG, "_capturing", lambda: True)
    shapes = {7: SimpleNamespace(m16=32, k=8, n=16), 8: SimpleNamespace(m16=32, k=8, n=32)}
    shapes[9] = shapes[8]
    for sh in shapes.values():
        sh.xa, sh.buf = torch.zeros(48, 8), lambda name: torch.zeros(100, dtype=torch.uint8)
    names = ("rows_a", "tree_seed_a", "forming", "gemm", "hash_leaf", "scatter")
    pipe, grouped = SimpleNamespace(b={}, size={"tree_a": 64}, tiles_group=False), []

    def step(w, name):
        def f():
            log.append(f"{w}:{name} on {current[0].name} ({run.kernel.dev.stream})")
            grouped.append(pipe.tiles_group)
        return f
    for w, sh in shapes.items():
        run._graphs[run._graph_key(w, sh, 32)[0]] = PG._Graph(torch.zeros(1), torch.zeros(1), "graph",
                                                               [step(w, x) for x in names], (2, 4), 1)
    run._stages = {(w, 32): (Stage(log, current, f"{w}:rows"), Stage(log, current, f"{w}:tree")) for w in shapes}

    def forward(retained):
        run.whole_capture_begin(run.whole_key(retained))
        for i, (w, sh) in enumerate(shapes.items()):
            run.join_before(sh)
            run._graph_call(w, sh, pipe, 1, 4, 32, i, None)
        return run.whole_capture_end()

    assert [c.w for c in forward(False)] == [7, 8, 9]
    main = lambda w, *xs: [f"{w}:{x} on main (1)" for x in xs]  # noqa: E731
    tree = lambda w: ["tree waits for main", f"{w}:tree_seed_a on tree (3)"]  # noqa: E731
    side = lambda w: ["side waits for main", f"{w}:hash_leaf on side (2)", "side waits for tree",  # noqa: E731
                      f"{w}:scatter on side (2)"]
    assert log == [*main(7, "rows_a"), *tree(7), *main(7, "forming", "gemm"), *side(7),
                   *main(8, "rows_a"), *tree(8), "main waits for side", *main(8, "forming", "gemm"), *side(8),
                   "main waits for side",                     # 9's call is of 8's shape: 8's tree and side steps joined first
                   *main(9, "rows_a"), *tree(9), *main(9, "forming", "gemm"), *side(9),
                   "main waits for side", "main waits for tree"]   # the forward's last call joins both
    assert not run._tree_open and run.kernel.dev.stream is None
    steps_run = [x.split(" on ")[0].split(":")[1] for x in log if " on " in x and "staged" not in x]
    assert [g for x, g in zip(steps_run, grouped, strict=True) if g] == [True] * 6   # the side steps' tiles, PD_G a CTA
    assert {x for x, g in zip(steps_run, grouped, strict=True) if g} == {"hash_leaf", "scatter"} and not pipe.tiles_group
    log.clear()
    forward(True)
    assert [x for x in log if "staged" in x or x == "main waits for tree"] == [
        *(x for w in shapes for x in ("main waits for tree", f"{w}:rows staged on main", f"{w}:tree staged on main"))]
    log.clear()
    run.whole_defer_a = False                                 # the same graphs, the tree on the capturing stream
    forward(False)
    assert "tree waits for main" not in log and log[:4] == main(7, "rows_a", "tree_seed_a", "forming", "gemm")


def test_a_shape_joining_capture_joins_a_calls_side_steps_only_before_the_next_call_of_its_shape(monkeypatch):
    """Under ``whole_shape_joins`` no call joins the call before's side steps: the side stream records an event after each
    call's, and a call waits at its shape's last one before it writes the shape's buffers (`join_before`), dropping the
    events before it (the one side stream ran them first); at the last pending call's it waits for the side stream itself,
    and the forward's last call joins the side and tree streams.  Every step keeps its stream and order."""
    torch = pytest.importorskip("torch")
    run, log = Calls(torch), []
    run.whole_defer, run.whole_defer_a, run.whole_shape_joins, run.device, run._graphs = True, True, True, "cuda", {}
    run.kernel = SimpleNamespace(dev=SimpleNamespace(stream=None))
    current = _lanes(monkeypatch, torch, log)
    monkeypatch.setattr(PG, "_capturing", lambda: True)
    x, y = SimpleNamespace(m16=32, k=8, n=16), SimpleNamespace(m16=32, k=8, n=32)
    shapes = {7: x, 8: y, 9: x, 10: y, 11: y}
    run.order = {w: i for i, w in enumerate(shapes)}
    names = ("rows_a", "tree_seed_a", "forming", "gemm", "hash_leaf", "scatter")

    def step(w, name):
        return lambda: log.append(f"{w}:{name} on {current[0].name} ({run.kernel.dev.stream})")
    pipe = SimpleNamespace(b={}, size={"tree_a": 64}, tiles_group=False)
    for w, sh in shapes.items():
        run._graphs[run._graph_key(w, sh, 32)[0]] = PG._Graph(torch.zeros(1), torch.zeros(1), "graph",
                                                               [step(w, n) for n in names], (2, 4), 1)
    run.whole_capture_begin(run.whole_key(False))
    for i, (w, sh) in enumerate(shapes.items()):
        run.join_before(sh)
        run._graph_call(w, sh, pipe, 1, 4, 32, i, None)
    assert [c.w for c in run.whole_capture_end()] == list(shapes)
    call = lambda w, e: [f"{w}:rows_a on main (1)", "tree waits for main", f"{w}:tree_seed_a on tree (3)",  # noqa: E731
                         f"{w}:forming on main (1)", f"{w}:gemm on main (1)", "side waits for main", f"{w}:hash_leaf on side (2)",
                         "side waits for tree", f"{w}:scatter on side (2)", f"side event {e}"]
    assert log == [*call(7, 0), *call(8, 1),
                   "main waits for side event 0", *call(9, 2),   # 7's shape: at 7's event, 8's side steps still running
                   "main waits for side event 1", *call(10, 3),  # 8's shape: at 8's event, 9's side steps still running
                   "main waits for side", *call(11, 4),          # 10's shape, the last pending call: the side stream itself
                   "main waits for side", "main waits for tree"]
    assert run._pending is None and not run._tree_open and run.kernel.dev.stream is None


def test_a_stats_forking_capture_runs_each_calls_a_scales_beside_its_a_commitment_and_joins_them_in_their_place(
        monkeypatch):
    """Under ``whole_fork_stats`` a call's ``stats_a``, which reads A's words alone, goes on the stats stream after the
    call's A words, beside its A commitment (``rows_a``) and tree, launched after the commitment (so the graph launches
    the commitment first), and the capturing stream joins it in its place, before ``eprime_a`` reads A's scales (the mark
    after it); every other step keeps its stream and order.  Under ``whole_prefetch_b`` ``stats_a`` and then B̃'s prefetch
    are launched there before the commitment, the join not waiting for the prefetch, which the forward's last call joins;
    alone, the prefetch forks only itself."""
    torch = pytest.importorskip("torch")
    run, log = Calls(torch), []
    run.whole_defer, run.whole_defer_a, run.whole_fork_stats, run.device, run._graphs = True, True, True, "cuda", {}
    run.kernel = SimpleNamespace(dev=SimpleNamespace(stream=None))
    current = _lanes(monkeypatch, torch, log)
    monkeypatch.setattr(PG, "_capturing", lambda: True)
    shapes = {7: SimpleNamespace(m16=32, k=8, n=16), 8: SimpleNamespace(m16=32, k=8, n=32)}
    shapes[9] = shapes[8]
    names = ("rows_a", "tree_seed_a", "stats_a", "eprime_a", "gemm", "hash_leaf", "scatter")

    def step(w, name):
        return lambda: log.append(f"{w}:{name} on {current[0].name} ({run.kernel.dev.stream})")
    for w, sh in shapes.items():
        run._graphs[run._graph_key(w, sh, 32)[0]] = PG._Graph(torch.zeros(1), torch.zeros(1), "graph",
                                                               [step(w, x) for x in names], (2, 5), 1, 2)
    pipes = {w: SimpleNamespace(b={}, size={"tree_a": 64}, prefetch_b=step(w, "prefetch_b")) for w in shapes}

    def forward():
        run.whole_capture_begin(run.whole_key(False))
        for i, (w, sh) in enumerate(shapes.items()):
            run.join_before(sh)
            run._graph_call(w, sh, pipes[w], 1, 4, 32, i, None)
        return run.whole_capture_end()

    assert [c.w for c in forward()] == [7, 8, 9]
    main = lambda w, *xs: [f"{w}:{x} on main (1)" for x in xs]  # noqa: E731
    stats = lambda w, *xs: ["stats waits for main", f"{w}:stats_a on stats (4)", "stats marks",  # noqa: E731
                            *(f"{w}:{x} on stats (4)" for x in xs)]
    joined = "main waits for stats's mark"
    tree = lambda w: ["tree waits for main", f"{w}:tree_seed_a on tree (3)"]  # noqa: E731
    commit = lambda w: ["stats waits for main", *main(w, "rows_a"), f"{w}:stats_a on stats (4)", "stats marks",  # noqa: E731
                        *tree(w)]
    side = lambda w: ["side waits for main", f"{w}:hash_leaf on side (2)", "side waits for tree",  # noqa: E731
                      f"{w}:scatter on side (2)"]
    assert log == [*commit(7), joined, *main(7, "eprime_a", "gemm"), *side(7),
                   *commit(8), "main waits for side", joined,
                   *main(8, "eprime_a", "gemm"), *side(8),
                   "main waits for side",                     # 9's call is of 8's shape: joined before it writes A's words
                   *commit(9), joined, *main(9, "eprime_a", "gemm"), *side(9),
                   "main waits for side", "main waits for tree"]
    assert run.kernel.dev.stream is None
    log.clear()
    run.whole_prefetch_b = True                               # B̃'s prefetch after stats_a, joined at the forward's end
    forward()
    call = lambda w: [*stats(w, "prefetch_b"), *main(w, "rows_a"), *tree(w),  # noqa: E731
                      *(["main waits for side"] if w == 8 else []), joined, *main(w, "eprime_a", "gemm"), *side(w)]
    assert log == [*call(7), *call(8), "main waits for side", *call(9), "main waits for side", "main waits for tree",
                   "main waits for stats"]
    assert not run._prefetch_open
    log.clear()
    run.whole_fork_stats = False                              # the prefetch alone: stats_a in its place, nothing waits on it
    forward()
    assert log[:7] == ["stats waits for main", "7:prefetch_b on stats (4)", *main(7, "rows_a"), *tree(7),
                       *main(7, "stats_a", "eprime_a")]
    assert joined not in log and log[-1] == "main waits for stats"
    log.clear()
    run.whole_prefetch_b = False                              # the same graphs, stats_a on the capturing stream in its place
    forward()
    assert not any("stats waits" in x or "waits for stats" in x for x in log)
    assert log[:5] == [*main(7, "rows_a"), *tree(7), *main(7, "stats_a", "eprime_a")]


def test_a_stats_forking_capture_widens_a_calls_a_words_beside_its_commitment_where_the_commitment_reads_its_bf16_rows(
        monkeypatch):
    """A call's A words left to its `_into` (`_widening`, as `LaneCalls._stage_a` leaves them in such a capture): where its
    steps before the scales' join read them only in ``rows_a`` and the pipeline reads BF16 rows, ``rows_a`` runs with the
    call's rows set on the pipeline (`rows_bf16`, cleared after) and the widening goes on the stats stream before
    ``stats_a``, plainly; else the widening runs first on the capturing stream, as `_stage_a` launches it."""
    torch = pytest.importorskip("torch")
    run, log = Calls(torch), []
    run.whole_defer, run.whole_defer_a, run.whole_fork_stats, run.device, run._graphs = True, True, True, "cuda", {}
    current = _lanes(monkeypatch, torch, log)
    monkeypatch.setattr(PG, "_capturing", lambda: True)
    names = ("rows_a", "tree_seed_a", "stats_a", "eprime_a", "gemm", "hash_leaf", "scatter")
    pipe = SimpleNamespace(b={}, size={"tree_a": 64}, rows_bf16=None, reads=True, names=names)
    pipe.steps, pipe.reads_bf16 = (lambda: [(x, None, None) for x in pipe.names]), (lambda: pipe.reads)
    run._call_names = lambda p: p.names
    widen = lambda dev, src, out, m, filled, k, stream=None, pdl=True: log.append(  # noqa: E731
        f"widen {src} {m} {filled} {k} on {current[0].name} ({stream}) pdl {pdl}")
    run.kernel = SimpleNamespace(dev=SimpleNamespace(stream=None), run=SimpleNamespace(widen_a=widen))
    shapes = {7: SimpleNamespace(m16=32, k=8, n=16, xa=torch.zeros(48, 8)),
              8: SimpleNamespace(m16=32, k=8, n=32, xa=torch.zeros(48, 8))}

    def step(w, name):
        return lambda: log.append(f"{w}:{name} on {current[0].name} ({run.kernel.dev.stream})"
                                  + (f" rows {pipe.rows_bf16}" if name == "rows_a" else ""))
    for w, sh in shapes.items():
        run._graphs[run._graph_key(w, sh, 32)[0]] = PG._Graph(torch.zeros(1), torch.zeros(1), "graph",
                                                               [step(w, x) for x in names], (2, 5), 1, 2)

    def forward():
        run.whole_capture_begin(run.whole_key(False))
        for i, (w, sh) in enumerate(shapes.items()):
            run.join_before(sh)
            assert run._widen_later(sh)
            run._widening = (sh, 1000 + w, 3, 32)
            run._graph_call(w, sh, pipe, 1, 4, 32, i, None)
            assert run._widening is None and pipe.rows_bf16 is None
        return run.whole_capture_end()

    forward()
    assert log[:6] == ["stats waits for main", "7:rows_a on main (1) rows (1007, 3)", "widen 1007 3 32 8 on stats (4) pdl False",
                       "7:stats_a on stats (4)", "stats marks", "tree waits for main"]
    assert "main waits for stats's mark" in log
    assert "widen 1008 3 32 8 on stats (4) pdl False" in log and not any("on main" in x for x in log if x.startswith("widen"))
    for reads, before in ((False, names), (True, ("rows_a", "lines_ea", *names[1:]))):
        log.clear()
        pipe.reads, pipe.names = reads, before
        forward()
        assert log[:3] == ["widen 1007 3 32 8 on main (1) pdl True", "stats waits for main", "7:rows_a on main (1) rows None"]
    run.whole_capture_begin(run.whole_key(False))
    run._widening = (shapes[8], 1, 3, 32)
    with pytest.raises(RuntimeError, match="another shape's"):
        run._graph_call(7, shapes[7], pipe, 1, 4, 32, 0, None)
    monkeypatch.setattr(PG, "_capturing", lambda: False)
    assert not run._widen_later(shapes[7])


def test_a_tree_sm_capture_launches_the_steps_beside_each_calls_a_tree_with_the_pipelines_beside_set(monkeypatch):
    """Under ``whole_tree_sm`` the steps a call launches between its cuts, beside its forked A tree (its forming and GEMM),
    see the pipeline's ``beside`` at 1, so its persistent GEMMs leave the tree's CTA its SM; its A commitment before them,
    its tile hashing after them, and every step without the option or without a forked tree see 0."""
    torch = pytest.importorskip("torch")
    run, log = Calls(torch), []
    run.whole_defer, run.whole_defer_a, run.whole_tree_sm, run.device, run._graphs = True, True, True, "cuda", {}
    run.kernel = SimpleNamespace(dev=SimpleNamespace(stream=None))
    _lanes(monkeypatch, torch, log)
    monkeypatch.setattr(PG, "_capturing", lambda: True)
    sh, pipe, seen = SimpleNamespace(m16=32, k=8, n=16), SimpleNamespace(b={}, size={"tree_a": 64}, beside=0), []
    names = ("rows_a", "tree_seed_a", "eprime_a", "gemm", "hash_leaf", "scatter")
    steps = [lambda x=x: seen.append((x, pipe.beside)) for x in names]
    run._graphs[run._graph_key(7, sh, 32)[0]] = PG._Graph(torch.zeros(1), torch.zeros(1), "graph", steps, (2, 4), 1)

    def forward():
        seen.clear()
        run.whole_capture_begin(run.whole_key(False))
        run._graph_call(7, sh, pipe, 1, 4, 32, 0, None)
        run.whole_capture_end()
        return seen

    assert forward() == [("rows_a", 0), ("tree_seed_a", 0), ("eprime_a", 1), ("gemm", 1), ("hash_leaf", 0), ("scatter", 0)]
    assert pipe.beside == 0
    run.whole_tree_sm = False
    assert [b for _, b in forward()] == [0] * 6
    run.whole_tree_sm, run.whole_defer_a = True, False         # A's tree on the capturing stream: nothing beside the GEMM
    assert [b for _, b in forward()] == [0] * 6


def test_a_calls_steps_split_after_its_a_tree_and_before_its_tile_hashing():
    pipe = SimpleNamespace(steps=lambda: [(x, None, None) for x in PPD.CALL_STEPS if x != "pad_a"])
    run = SimpleNamespace(entry="test", hashing=True, omit=frozenset())
    run._call_names = lambda p: PPD.PearlCDeviceRun._call_names(run, p)
    a, b = PPD.PearlCDeviceRun._call_cuts(run, pipe)
    names = [x for x in run._call_names(pipe) if x != "pad_a"]
    # A's tree with its row digests and row seeds, before forming; the tile hashing is the leaves alone
    assert names[:a] == ["rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a"] and names[b:] == ["hash_leaf"]
    run.hashing = False
    assert PPD.PearlCDeviceRun._call_cuts(run, pipe) == (0, len(run._call_names(pipe)))
    run._call_names = lambda p: ("lines_ea", "hash_rows_a", "gemm_pearlc", "hash_leaf")
    with pytest.raises(RuntimeError, match="A's commitment"):
        PPD.PearlCDeviceRun._call_cuts(run, pipe)


def test_a_fused_step_takes_its_parts_place_and_an_omitted_part_drops_it():
    parts = ("pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "lines_ea")
    pipe = SimpleNamespace(steps=lambda: [(x, None, None) for x in ("commit_a", *PPD.CALL_STEPS) if x not in parts],
                           FUSED={"commit_a": parts})
    run = SimpleNamespace(entry="test", hashing=True, omit=frozenset())
    run._call_names = lambda p: PPD.PearlCDeviceRun._call_names(run, p)
    names = run._call_names(pipe)
    assert names == ("commit_a", "stats_a", "eprime_a", "form_a", "af", "pa", "gemm_pearlc", "hash_leaf")
    assert PPD.PearlCDeviceRun._call_cuts(run, pipe) == (1, 7)
    run.omit = frozenset({"seeds_a"})
    assert "commit_a" not in run._call_names(pipe) and run._call_names(pipe)[0] == "stats_a"
    run.omit, run.hashing = frozenset(), False
    assert run._call_names(pipe) == ("stats_a", "eprime_a", "form_a", "af", "pa", "gemm_pearlc")
    ep = ("eprime_a", "form_a")
    pipe = SimpleNamespace(steps=lambda: [(x, None, None) for x in ("form_ep_a", *PPD.CALL_STEPS) if x not in ep],
                           FUSED={"form_ep_a": ep})
    assert run._call_names(pipe) == ("stats_a", "form_ep_a", "af", "pa", "gemm_pearlc")
    assert PPD.STEP_GROUPS["form_ep_a"] == "forming"
    # with A'F_B's windows in the forming launch too, that step takes the words' place as well; a pipeline with both entries
    # in FUSED runs one of them
    epw = (*ep, "af")
    pipe = SimpleNamespace(steps=lambda: [(x, None, None) for x in ("form_epw_a", *PPD.CALL_STEPS) if x not in epw],
                           FUSED={"form_ep_a": ep, "form_epw_a": epw})
    assert run._call_names(pipe) == ("stats_a", "form_epw_a", "pa", "gemm_pearlc")
    assert PPD.STEP_GROUPS["form_epw_a"] == "forming"


def test_a_split_commitments_tree_forks_from_the_step_before_the_first_cut():
    """A pipeline that splits A's commitment (``rows_a``, then ``tree_seed_a``) has its tree last before `_call_cuts`' first
    cut, where a deferred-a whole capture forks it (`_call_tree`); one that doesn't has none to fork, and a split whose tree
    isn't A's commitment's last step is refused."""
    parts = ("pad_a", "rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a", "lines_ea")
    fused = {"commit_a": parts, "rows_a": tuple(x for x in parts if x not in ("tree_a", "seed_a")),
             "tree_seed_a": ("tree_a", "seed_a")}
    split = SimpleNamespace(steps=lambda: [(x, None, None) for x in ("rows_a", "tree_seed_a", *PPD.CALL_STEPS) if x not in parts],
                            FUSED=fused)
    whole = SimpleNamespace(steps=lambda: [(x, None, None) for x in ("commit_a", *PPD.CALL_STEPS) if x not in parts], FUSED=fused)
    run = SimpleNamespace(entry="test", hashing=True, omit=frozenset())
    run._call_names = lambda p: PPD.PearlCDeviceRun._call_names(run, p)
    run._call_cuts = lambda p: PPD.PearlCDeviceRun._call_cuts(run, p)
    assert run._call_names(split)[:3] == ("rows_a", "tree_seed_a", "stats_a") and run._call_cuts(split) == (2, 8)
    assert PPD.PearlCDeviceRun._call_tree(run, split) == 1 and PPD.PearlCDeviceRun._call_tree(run, whole) is None
    assert PPD.PearlCDeviceRun._call_stats(run, split) == 2 and PPD.PearlCDeviceRun._call_stats(run, whole) == 1
    run._call_names = lambda p: ("tree_seed_a", "rows_a", "stats_a", "gemm_pearlc", "hash_leaf")
    with pytest.raises(RuntimeError, match="A's tree"):
        PPD.PearlCDeviceRun._call_tree(run, split)


def test_deferring_a_trees_needs_a_deferred_whole_capture_on_a_kernel_that_splits_a_commitment():
    def kernel(fused):
        return SimpleNamespace(run=SimpleNamespace(Pipeline=SimpleNamespace(FUSED=fused)))
    for k, defer in ((kernel({"commit_a": ()}), True), (kernel({"tree_seed_a": ()}), False)):
        with pytest.raises(ValueError, match="whole_defer_a"):
            PPD.PearlCDeviceRun(None, "test", k, {}, {}, None, None, whole_defer=defer, whole_defer_a=True)


def test_forking_a_scales_needs_a_deferred_whole_capture():
    k = SimpleNamespace(run=SimpleNamespace(Pipeline=SimpleNamespace(FUSED={"tree_seed_a": ()})))
    with pytest.raises(ValueError, match="whole_fork_stats"):
        PPD.PearlCDeviceRun(None, "test", k, {}, {}, None, None, whole_defer=False, whole_fork_stats=True)


def test_leaving_a_tree_its_sm_needs_a_deferred_a_tree_on_a_kernel_whose_gemms_take_beside():
    splits_a = {"tree_seed_a": ()}
    for pipeline, defer_a in ((SimpleNamespace(FUSED=splits_a, beside=0), False), (SimpleNamespace(FUSED=splits_a), True)):
        k = SimpleNamespace(run=SimpleNamespace(Pipeline=pipeline))
        with pytest.raises(ValueError, match="whole_tree_sm"):
            PPD.PearlCDeviceRun(None, "test", k, {}, {}, None, None, whole_defer=True, whole_defer_a=defer_a, whole_tree_sm=True)


def test_joining_side_steps_by_shape_needs_a_deferred_whole_capture():
    k = SimpleNamespace(run=SimpleNamespace(Pipeline=SimpleNamespace(FUSED={"tree_seed_a": ()})))
    with pytest.raises(ValueError, match="whole_shape_joins"):
        PPD.PearlCDeviceRun(None, "test", k, {}, {}, None, None, whole_defer=False, whole_shape_joins=True)


def test_prefetching_b_needs_a_deferred_whole_capture_on_a_kernel_with_the_prefetch():
    for pipeline, defer in ((SimpleNamespace(FUSED={}, prefetch_b=None), False), (SimpleNamespace(FUSED={}), True)):
        k = SimpleNamespace(run=SimpleNamespace(Pipeline=pipeline))
        with pytest.raises(ValueError, match="whole_prefetch_b"):
            PPD.PearlCDeviceRun(None, "test", k, {}, {}, None, None, whole_defer=defer, whole_prefetch_b=True)


def test_an_end_aside_needs_lagged_ends():
    k = SimpleNamespace(run=SimpleNamespace(Pipeline=SimpleNamespace(FUSED={"tree_seed_a": ()})))
    with pytest.raises(ValueError, match="aside_ends"):
        PPD.PearlCDeviceRun(None, "test", k, {}, {}, None, None, aside_ends=True)


def test_a_calls_a_words_are_one_widen_a_where_the_kernel_has_it_else_torchs_copy_and_fill(monkeypatch):
    """`LaneCalls._stage_a`: on the card, with a cubin that has widen_a and x contiguous BF16, one launch on the current
    stream over the call's rows and the padding up to `filled`, or none, left to the call's `_into` (`_widening`), where
    `GraphCalls._widen_later`; with an older cubin, torch's copy and fill, the same words."""
    torch = pytest.importorskip("torch")
    from verity_vllm.protocols.pouw.pearl_c import lanes as L
    monkeypatch.setattr(L, "_stream", lambda device: 77)
    x, launched = torch.arange(24, dtype=torch.float32).reshape(3, 8).to(torch.bfloat16), []
    for has, later in ((True, False), (True, True), (False, True)):
        xa = torch.full((5, 8), 7.0)
        run = SimpleNamespace(widen_a=lambda *a, **kw: launched.append((a[1:], kw)))
        dev = SimpleNamespace(has=lambda name, has=has: has and name == "widen_a")
        calls = SimpleNamespace(device=SimpleNamespace(type="cuda"), kernel=SimpleNamespace(dev=dev, run=run),
                                _widen_later=lambda sh, later=later: later, _widening=None)
        sh = SimpleNamespace(xa=xa)
        L.LaneCalls._stage_a(calls, sh, x, 3, 4)
        if has and not later:
            assert launched == [((x.data_ptr(), xa.data_ptr(), 3, 4, 8), {"stream": 77})] and (xa == 7).all()
        elif has:
            assert calls._widening == (sh, x.data_ptr(), 3, 4) and len(launched) == 1 and (xa == 7).all()
        else:
            assert torch.equal(xa[:3], x.float()) and (xa[3] == 0).all() and (xa[4] == 7).all() and len(launched) == 1
            assert calls._widening is None


def test_a_lagged_end_gets_its_commitment_in_place_at_the_next_end_and_only_an_engine_steps_end_lags():
    """A lagged end's readback (its copies behind an event) completes at `settle`: the next end's first step, and
    `finish`'s.  It waits for the event once and fills the ended pass's own dict, the one `details["passes"]` and the
    stack hold, and `commits`.  The executor's pass end and `end_pass` with ``step`` lag where the run's `lag_ends` is
    set; `end_pass` without it (a caller's) never does."""
    import hashlib

    import torch

    run = PPD.PearlCDeviceRun.__new__(PPD.PearlCDeviceRun)
    run.commits, run._lagged, waited = {"transcript_root": None}, None, []
    run.cur, run.hashing, run.schedule, run.details = PPD._Pass(None), False, SimpleNamespace(name="serial"), {"passes": []}
    run.kernel = None
    roots_a, root = bytes(range(128)), bytes(range(64, 128))

    def lagged(units, tiles):
        out = {"units": units, "screened_rows": 3}
        got = [torch.frombuffer(bytearray(roots_a), dtype=torch.uint8), torch.frombuffer(bytearray(root), dtype=torch.uint8),
               torch.tensor([1, 2], dtype=torch.int64)]
        run._lagged = (out, SimpleNamespace(synchronize=lambda: waited.append(units)), got, units, tiles)
        return out

    out = lagged(2, 7)
    assert "transcript_root" not in out
    ended = run.end()                                        # nothing this pass: it reads back nothing, the last one settles
    assert waited == [2] and run._lagged is None and ended == {"units": 0, "tiles": 0, "hashing": False, "schedule": "serial"}
    assert out == {"units": 2, "screened_rows": 3, "transcript_root": root.hex(),
                   "activation_roots": hashlib.sha512(roots_a).hexdigest(), "failing_rows": 1, "unsure_rows": 2}
    assert run.commits == {"transcript_root": root.hex(), "units": 2, "tiles": 7}
    run.settle()
    assert waited == [2]
    out = lagged(4, 9)
    run.finish()
    assert waited == [2, 4] and out["transcript_root"] == root.hex() and run.commits["tiles"] == 9

    lags = []
    run.end, run.begin = (lambda lag=False: lags.append(lag) or {}), (lambda *a, **kw: None)
    run.cur = PPD._Pass(None, matmuls=[None])
    for flag in (False, True):
        run.lag_ends = flag
        PPD.PearlCDeviceRun.end_pass(run, step=True)
        PPD.PearlCDeviceRun.end_pass(run)
        PPD._ExecutorPass.end(SimpleNamespace(run=SimpleNamespace(end=run.end, lag_ends=flag, omit=None)))
    assert lags == [False, False, False, True, False, True]

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
        self.replayed: list = []
        self.begun: list = []
        self.details = {"passes": []}

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


class Calls(PG.GraphCalls):
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
    run.cur = PPD._Pass(None)                                 # a new pass is elsewhere: it makes its own
    assert run.whole_replay(calls) is False and offs[2].tolist() == [0, 0, 0]
    run.whole_replayed(calls)
    assert run.whole_replay(calls[:1]) is False               # and so do other calls
    assert keys[2, :168].tobytes() == run.keys(run.roots[9], 2, 32)


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
    pipe = SimpleNamespace(size={"tree_a": 64})
    run.whole_capture_begin(run.whole_key(True))
    run._graph_call(7, sh, pipe, 0, 4, 32, 0, None)          # vLLM's warm-up, outside the capture
    xa, tree = run._stages[7]
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
    run._stages = {9: (torch.full((32, 4), 2.0), torch.full((16,), 4, dtype=torch.uint8)),
                   7: (torch.full((32, 8), 1.0), torch.full((64,), 3, dtype=torch.uint8))}
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


class Lane:
    """A stand-in of torch's CUDA stream: its handle, and each wait it is told to make, in `log`."""

    def __init__(self, log, name="side", device=None):
        self.log, self.name, self.cuda_stream = log, name, {"main": 1, "side": 2}[name]

    def wait_stream(self, other):
        self.log.append(f"{self.name} waits for {other.name}")


def _lanes(monkeypatch, torch, log):
    """torch.cuda's current stream, `Stream` and `stream` over `Lane`s, the current one the capturing stream ("main")."""
    current = [Lane(log, "main")]

    @contextlib.contextmanager
    def stream(s):
        held, current[0] = current[0], s
        try:
            yield
        finally:
            current[0] = held
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current[0])
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: Lane(log, "side"))
    monkeypatch.setattr(torch.cuda, "stream", stream)
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
    assert [c.w for c in calls] == [7, 8, 9] and run._deferred is None and run.kernel.dev.stream is None
    log.clear()
    run.whole_defer = False
    forward()
    assert log == [f"{w}:{x} on main (1)" for w in shapes for x in names]


def test_a_calls_steps_split_after_its_a_tree_and_before_its_tile_hashing():
    pipe = SimpleNamespace(steps=lambda: [(x, None, None) for x in PPD.CALL_STEPS if x != "pad_a"])
    run = SimpleNamespace(entry="test", hashing=True, omit=frozenset())
    run._call_names = lambda: PPD.PearlCDeviceRun._call_names(run)
    a, b = PPD.PearlCDeviceRun._call_cuts(run, pipe)
    names = [x for x in run._call_names() if x != "pad_a"]
    # A's tree with its row digests and row seeds, before forming; the tile hashing is the leaves alone
    assert names[:a] == ["rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a"] and names[b:] == ["hash_leaf"]
    run.hashing = False
    assert PPD.PearlCDeviceRun._call_cuts(run, pipe) == (0, len(run._call_names()))
    run._call_names = lambda: ("lines_ea", "hash_rows_a", "gemm_pearlc", "hash_leaf")
    with pytest.raises(RuntimeError, match="A's commitment"):
        PPD.PearlCDeviceRun._call_cuts(run, pipe)

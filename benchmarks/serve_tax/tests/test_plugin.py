"""The plugin's changes on a stand-in for vLLM's V1 scheduler (the methods and attributes they wrap and read, as vLLM
d9105ea80 has them)."""
from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace as NS

import servetax_plugin as P


class Request:
    def __init__(self, rid: str, prompt: list[int]):
        self.request_id, self.prompt_token_ids, self.num_prompt_tokens = rid, prompt, len(prompt)
        self.sampling_params = NS(temperature=0.0, max_tokens=4, ignore_eos=True)
        self.eos_token_id = None


class Scheduler:
    def __init__(self):
        self.requests: dict[str, Request] = {}
        self.order: list[str] = []
        self.finished: list[str] = []
        self.scheduler_config = NS(max_num_seqs=8, max_num_batched_tokens=64, policy="fcfs", async_scheduling=False)
        self.cache_config = NS(block_size=16, enable_prefix_caching=True, num_gpu_blocks=100)

    def add_request(self, request):
        self.requests[request.request_id] = request
        self.order.append(request.request_id)

    def schedule(self):
        return NS(total_num_scheduled_tokens=0)

    def update_from_output(self, so, mro):
        return {}

    def finish_requests(self, request_ids, finished_status):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids or ())
        self.finished += ids
        return [self.requests.pop(r) for r in ids if r in self.requests]


def eid(run: str, n: int) -> str:
    return f"cmpl-{run}-{n:05d}-0-0123abcd"


def test_release_key_reads_the_client_number_inside_the_engine_id():
    assert P.release_key(eid("saturated2", 17)) == ("saturated2", 17)
    assert P.release_key("warm-00003") == ("warm", 3)
    assert P.release_key("cmpl-noid-0-0123abcd") is None


def test_the_gate_adds_in_release_order_and_the_trace_sees_that_order(tmp_path):
    sched = Scheduler()
    done = P.install(sched, {"out": str(tmp_path), "record": True, "release_order": True})
    for n in (2, 0, 3, 1):
        sched.add_request(Request(eid("light1", n), [n]))
    assert sched.order == [eid("light1", n) for n in range(4)]
    done[0].drain()
    (trace,) = tmp_path.glob("step_trace.*.jsonl")
    recs = [json.loads(x) for x in trace.read_text().splitlines()]
    assert [r["request"] for r in recs if r["kind"] == "arrive"] == sched.order
    done[-1].uninstall()
    done[0].uninstall()


def test_a_missing_number_is_skipped_after_its_hold_and_a_held_abort_never_reaches_the_scheduler():
    sched, now = Scheduler(), [0.0]
    gate = P.ReleaseGate(sched.add_request, lambda ids: sched.finish_requests(ids, "aborted"), key=P.release_key,
                         hold_s=5.0, clock=lambda: now[0])
    gate.add(Request(eid("r", 1), [1]))
    gate.add(Request(eid("r", 2), [2]))
    assert sched.order == [] and gate.held()
    assert gate.finished([eid("r", 2), "other"]) == ["other"]
    now[0] = 4.9
    gate.poll()
    assert sched.order == []
    now[0] = 5.0
    gate.poll()
    assert sched.order == [eid("r", 1)] and not gate.held()


def test_a_new_run_starts_from_zero_and_unnumbered_requests_pass():
    sched = Scheduler()
    gate = P.ReleaseGate(sched.add_request, lambda ids: None, key=P.release_key)
    gate.add(Request(eid("a", 0), [0]))
    gate.add(Request(eid("b", 0), [0]))
    gate.add(Request("plain-request", [0]))
    assert sched.order == [eid("a", 0), eid("b", 0), "plain-request"]


class Rows:
    def __init__(self, n: int, k: int):
        self.shape = (n, k)

    def __getitem__(self, s: slice) -> "Rows":
        return Rows(len(range(*s.indices(self.shape[0]))), self.shape[1])


class Launched:
    def __init__(self):
        self.done = False

    def ready(self) -> bool:
        return self.done

    def kernel_ms(self) -> float:
        return 2.0


def test_kv_edge_keeps_at_most_inflight_chunks_out_and_reports_each_edge_once():
    now, got, out = [0.0], [], []

    def launch(edge, rows):
        out.append((edge, rows.shape[0], Launched()))
        return out[-1][2]

    kv = P.KvEdge(10.0, got.append, chunk_rows=2, inflight=1, clock=lambda: now[0], launch=launch)
    kv.rows = lambda caches: [Rows(5, 8)]
    kv.tick(None)
    assert out == []
    now[0] = 10.0
    for _ in range(3):
        kv.tick(None)
        assert sum(not o[2].done for o in out) == 1
        out[-1][2].done = True
    kv.tick(None)
    assert [o[:2] for o in out] == [(0, 2), (0, 2), (0, 1)]
    assert got == [{"edge": 0, "bytes": 40, "chunks": 3, "kernel_ms": 6.0, "gb_per_s": round(40 / 6.0 / 1e6, 1), "span_s": 0.0,
                    "t": got[0]["t"]}]


def test_token_commit_commits_each_window_prompt_and_sampled_ids_in_order():
    now, got, seen = [0.0], [], []
    tc = P.TokenCommit(10.0, got.append, clock=lambda: now[0], commit=lambda w, data: (seen.append(data), b"\x01" * 64)[1])
    tc.prompt(Request("a", [5, 6, 7]))
    tc.sampled({0: NS(outputs=[NS(new_token_ids=[8]), NS(new_token_ids=[])])})
    tc.tick()
    now[0] = 10.0
    tc.tick()
    tc.prompt(Request("b", [9]))
    tc.stop()
    assert [g["tokens"] for g in got] == [4, 1] and [g["window"] for g in got] == [0, 1]
    assert seen[0] == bytes([5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0])


def test_kv_edge_starts_at_the_first_step_with_a_kv_cache_outside_a_capture():
    ticks, capturing = [], [False]

    class Edge:
        def tick(self, kv):
            ticks.append(kv)

    made = []
    run = P._kv_runner(lambda self: "out", {"out": "."}, capturing=lambda: capturing[0],
                       edge_of=lambda: made.append(1) or Edge())
    runner = NS()
    assert run(runner) == "out" and not made
    runner.kv_caches = []
    run(runner)
    runner.kv_caches, capturing[0] = ["kv"], True
    run(runner)
    assert not made
    capturing[0] = False
    run(runner), run(runner)
    assert made == [1] and ticks == [["kv"], ["kv"]]


def test_history_tables_start_on_rank_0_at_the_first_step_with_a_kv_cache_outside_a_capture(monkeypatch):
    """``boundary_rows.tables``: rank 0's step hook, channel and builder (`rank0_tables`) are made at the first step with
    a KV cache that is neither a dummy run nor captured, the builder starts answering the channel then, and the hook
    reads that step and every later one; another rank makes none."""
    made, steps, capturing = [], [], [False]

    class Hook:
        def after(self, runner, so):
            steps.append(so)

    class Channel:
        def start(self, builder):
            made.append(("start", builder))
            return "stop"

    monkeypatch.setattr(P, "rank0_tables", lambda runner, cfg: made.append(("tables", runner)) or (Hook(), Channel(), "b"))
    run = P._tables_runner(lambda self, so, **kw: "out", {"out": "."}, capturing=lambda: capturing[0], rank=lambda: 0)
    runner = NS()
    assert run(runner, 0) == "out" and not made
    runner.kv_caches = ["kv"]
    run(runner, 1, dummy_run=True)
    capturing[0] = True
    run(runner, 2)
    capturing[0] = False
    assert not made
    run(runner, 3), run(runner, 4)
    assert made == [("tables", runner), ("start", "b")] and steps == [3, 4] and run.state["stop"] == "stop"
    other = P._tables_runner(lambda self, so, **kw: None, {"out": "."}, capturing=lambda: False, rank=lambda: 1)
    other(runner, 5), other(runner, 6)
    assert other.state == {"hook": None} and steps == [3, 4] and len(made) == 2


def test_tables_that_cant_start_fail_every_step_and_a_refused_cache_stops_the_engine_as_it_binds_its_caches(monkeypatch):
    """A step whose tables can't start raises, and so does every step after it: none is served with the tables off. The
    guard itself (the step hook's construction, stood in for here) runs on rank 0 as Model Runner V2 binds its caches,
    so an engine whose cache the tables can't read never serves a step."""
    import pytest

    def refuse(runner, cfg):
        raise ValueError("serve with --kv-cache-dtype auto")

    monkeypatch.setattr(P, "rank0_tables", refuse)
    run = P._tables_runner(lambda self, so, **kw: "out", {"out": "."}, capturing=lambda: False, rank=lambda: 0)
    runner = NS(kv_caches=["kv"])
    for step in range(3):
        with pytest.raises(ValueError, match="--kv-cache-dtype auto"):
            run(runner, step)
    assert run.state == {}

    class Channel:
        def start(self, builder):
            raise RuntimeError("no thread")

    monkeypatch.setattr(P, "rank0_tables", lambda runner, cfg: ("hook", Channel(), "b"))
    for step in range(2):
        with pytest.raises(RuntimeError, match="no thread"):
            run(runner, step)
    assert run.state == {}

    hooks, bound = [], []

    def step_hook(model, layers):
        hooks.append((model, layers))
        if model == "fp8":
            raise ValueError("its cache doesn't hold DsaLatentRow_v1's 576 bf16 words a row; serve with --kv-cache-dtype auto")

    tables = types.ModuleType("verity_vllm.pipeline.boundary_tables")
    tables.StepHook, tables.layers_of = step_hook, lambda spec: tuple(range(spec))
    monkeypatch.setitem(sys.modules, tables.__name__, tables)
    if "verity_vllm.pipeline" in sys.modules:
        monkeypatch.setattr(sys.modules["verity_vllm.pipeline"], "boundary_tables", tables, raising=False)
    cfg = {"boundary_rows": {"tables": {"layers": 2}}}

    def initialize(self, config, is_profiling=False):
        bound.append(config)
        self.kv_caches = ["kv"]
        return "bound"

    guard = P._tables_guard(initialize, cfg, rank=lambda: 0)
    assert guard(NS(model="bf16"), "c", is_profiling=False) == "bound" and hooks == [("bf16", (0, 1))]
    with pytest.raises(ValueError, match="DsaLatentRow_v1.*serve with --kv-cache-dtype auto"):
        guard(NS(model="fp8"), "c")
    # another rank reads no rows, and a runner that bound no cache has none to refuse
    assert P._tables_guard(initialize, cfg, rank=lambda: 1)(NS(model="fp8"), "c") == "bound"
    assert P._tables_guard(lambda self, c: "none", cfg, rank=lambda: 0)(NS(model="fp8"), "c") == "none"
    assert len(bound) == 3 and len(hooks) == 2


class RankExecutor:
    """A stand-in tensor-parallel executor: three ranks answering `boundary.rank_pool` and `boundary.rank_take`, rank 2's
    root different."""

    def __init__(self):
        self.calls: list = []

    def collective_rpc(self, fn, args=()):
        import boundary as B

        self.calls.append((fn.__name__, args))
        if fn is B.rank_pool:
            return [{"rank": r, "bytes": 10} for r in (1, 0)]
        return [{"rank": r, "root": "aa" if r < 2 else "bb", "hash_kernel_ms": 4.0 + r, "copy_ms": 9.0 if r == 0 else None,
                 "copied_bytes": 10 if r == 0 else 0, "kv_s": 0.5, "runner_s": 0.1 * r} for r in (2, 0, 1)]


def test_a_tensor_parallel_edge_takes_each_ranks_part_through_collective_rpc(monkeypatch):
    import boundary as B
    import pytest

    sched = NS(requests={"a": 1}, running=[1], waiting=[], kv_cache_manager=NS(block_pool=NS(blocks=[0] * 4)))
    core = NS(scheduler=sched, model_executor=RankExecutor())
    calls = core.model_executor.calls
    monkeypatch.setattr(B, "capture", lambda core, runner=True: {"requests": b"xyz", "runner": runner})
    now, recs = [10.0], []
    with pytest.raises(RuntimeError, match="no save_kv"):
        P.ranks_take(core, {"out": "x", "save_kv": [1]}, NS(scheduled=7), None, None, [None], lambda: now[0],
                     record=recs.append)
    take = P.ranks_take(core, {"out": "x", "kv_copy": "first"}, NS(scheduled=7), None, None, ["medium2"],
                        lambda: now[0], key=lambda: b"k" * 32, record=recs.append)
    assert calls == [("rank_pool", (4, "first"))] and recs[0]["ranks"][0] == {"rank": 1, "bytes": 10}
    now[0] = 12.5
    take(NS(edge=0), {"edge": 0, "due_t": 12.0, "drained_t": 12.25, "drain_calls": 2})
    assert calls[-1] == ("rank_take", (b"k" * 32,))
    r = recs[-1]
    assert (r["edge"], r["load"], r["next_step"], r["drain_s"], r["pause_s"]) == (0, "medium2", 7, 0.25, 0.25)
    assert [x["rank"] for x in r["ranks"]] == [0, 1, 2] and r["state_requests_bytes"] == 3
    assert r["kv"] == {"root": "aa", "roots_equal": False, "bytes": 20, "copied_bytes": 10, "hash_kernel_ms": 6.0,
                       "copy_ms": 9.0, "rank_kv_s": 0.5, "runner_s": 0.2}


def test_a_tensor_parallel_engine_takes_an_edge_each_period_of_each_loads_serving(tmp_path, monkeypatch):
    import boundary as B

    sched = Scheduler()
    sched._verity_servetax = P.install(sched, {"out": str(tmp_path), "record": True})
    sched.running, sched.waiting = [], []
    sched.kv_cache_manager = NS(block_pool=NS(blocks=[0] * 4))
    recs: list = []
    monkeypatch.setattr(P, "_lines_behind", lambda path: recs.append)
    monkeypatch.setattr(B, "capture", lambda core, runner=True: {"requests": b""})
    core = NS(scheduler=sched, model_executor=RankExecutor(), batch_queue=[])
    now, steps = [0.0], []
    edges = P.edges_of(core, {"out": str(tmp_path), "every_s": 30.0, "kv_copy": "first"}, lambda: now[0])

    def step(t: float) -> None:
        now[0] = t
        edges.step(lambda c: steps.append(t))

    step(5.0)
    sched.add_request(Request(eid("light1", 0), [1]))
    step(34.0)
    step(35.0)
    now[0] = 40.0
    sched.add_request(Request(eid("medium1", 0), [1]))
    step(69.0)
    step(70.0)
    assert [(r["edge"], r["load"], r["drain_s"]) for r in recs if "edge" in r] == [(0, "light1", 0.0), (1, "medium1", 0.0)]
    assert steps == [5.0, 34.0, 35.0, 69.0, 70.0]


def test_boundary_rows_installs_the_integrations_boundary_windows_from_the_configs_object(tmp_path, monkeypatch):
    """``boundary_rows`` hands its object to the integration's boundary windows (`verity_vllm.pipeline.boundary_windows`, a
    stand-in here), which commit each class under out/boundary/ and write a line each to boundary.<pid>.jsonl. Behind
    the certifier's release order they also get the gate's key, each request's release; without it, none."""
    got: dict = {}

    class Windows:
        @classmethod
        def of(cls, cfg, out, sink, release=None, public=()):
            got.update(cfg=cfg, out=out, release=release, public=public[:1])
            sink({"window": 0, "spans": 2})
            return cls()

        def install(self, sched, hooks):
            got["sched"] = sched
            return self

    stub = types.ModuleType("verity_vllm.pipeline.boundary_windows")
    stub.BoundaryWindows = Windows
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    if "verity_vllm.pipeline" in sys.modules:
        monkeypatch.setattr(sys.modules["verity_vllm.pipeline"], "boundary_windows", stub, raising=False)
    sched, cfg = Scheduler(), {"check": "t", "per_prompt": 0, "per_output": 1, "window_s": 5.0}
    done = P.install(sched, {"out": str(tmp_path), "boundary_rows": cfg})
    assert isinstance(done[0], Windows)
    assert got == {"cfg": cfg, "out": tmp_path / "boundary", "public": (tmp_path,), "sched": sched, "release": None}
    (lines,) = tmp_path.glob("boundary.*.jsonl")
    assert json.loads(lines.read_text()) == {"window": 0, "spans": 2}
    done[-1].uninstall()
    (tmp_path / "ordered").mkdir()
    done = P.install(Scheduler(), {"out": str(tmp_path / "ordered"), "boundary_rows": cfg, "release_order": True})
    assert got["release"] is P.release_key
    done[-1].uninstall()
    # a class's directory holds its requests' tokens and salts: with a private directory named it goes there, and
    # out keeps only the classes' public lines and stays the directory the history tables are refused under
    out, private = tmp_path / "out", tmp_path / "kept"
    out.mkdir()
    done = P.install(Scheduler(), {"out": str(out), "private": str(private), "boundary_rows": cfg})
    assert got["out"] == private / "boundary" and got["public"] == (out,)
    assert [f.name.split(".")[0] for f in out.iterdir()] == ["boundary"]
    done[-1].uninstall()

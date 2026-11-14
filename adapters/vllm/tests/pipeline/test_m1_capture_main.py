"""`verity_vllm.pipeline.m1_capture`: the Match stage's capture and control arms, `main` whole, on the CPU.

Stood in, each because it needs a GPU, vLLM or a checkpoint: the workload's reading, the engine's build, its run of the
requests, its version facts and its scheduler's lag, the observer and its header, the acquisition's stage and its hooks, and
the protocols.  Each stand-in keeps what it was handed.  The engine's model holds real buffers, so the analytic tables' words
are the module's own.  The module's clock is one the stand-in run moves (`Clock`), and its timeline keeps its events, so every
number in a line is one the test states.

Held: the files `main` writes (`versions.json`, `spec_tokens.json`, `tokens.json`, `stats.json`,
`acquisition_observed.json`, the tables' words), what each stand-in was handed, every line it prints and how `main` ends: by
returning or by exit 3.  CPU only."""
from __future__ import annotations

import contextlib
import copy
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

import verity_vllm.pipeline.m1_capture as M1
from verity_vllm.acquire import install, stage
from verity_vllm.observe import capture as OC

Record = dict[str, Any]
Lines = Callable[[], list[str]]

ROTARY = "model.layers.0.self_attn.rotary_emb.cos_sin_cache"
REQUESTS: list[Record] = [{"request_id": "r0", "prompt_token_ids": [11, 12, 13], "max_tokens": 4, "arrive_step": 0},
                          {"request_id": "r1", "prompt_token_ids": [21, 22], "max_tokens": 4}]
SAMPLING: Record = {"temperature": 0.0}
ACCEPTED = {"r0": [7, 8, 9, 10], "r1": [5, 2]}
SERVED = {"r0": [7, 8, 9, 10], "r1": [5, 2]}
STEPS: Record = {"r0": {"arrive_step_realized": 0, "first_step": 0, "finish_step": 3, "finish_reason": "length"},
                 "r1": {"arrive_step_realized": 1, "first_step": 1, "finish_step": 2, "finish_reason": "stop"}}
KWARGS: Record = {"model": "/weights", "tokenizer": "/weights", "enforce_eager": True, "max_num_seqs": 2}
SHOWN: Record = {"enforce_eager": True, "max_num_seqs": 2}
STATS: Record = {"events_written": 40, "steps": 5, "recorder": "pinned", "storage": "arena", "snapshot_bytes": 0, "errors": []}


def workload(**more: Any) -> Record:
    return {"requests": copy.deepcopy(REQUESTS), "sampling": dict(SAMPLING), "case": None, "accepted": copy.deepcopy(ACCEPTED),
            "row_id": "the-row", "serving_profile": "mixed", "context_class": "i256", "expected_engine_steps": 5} | more


def timing(wall_s: float, *, steps: Record | None = None, schedule: Record | None = None) -> Record:
    """A run's timing as `vllm_adapter.run_requests` returns it."""
    return {"wall_s": wall_s, "engine_steps": 5, "requests": copy.deepcopy(STEPS if steps is None else steps),
            "static_schedule": schedule}


def declared(digest: str, **more: Any) -> Record:
    """A run's static-schedule record, one the run kept."""
    return {"ok": True, "problems": [], "schedule": {"digest": digest, "engine_steps": 5}} | more


class Clock:
    """`m1_capture`'s `time`: seconds the stand-ins move."""

    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        return self.now


class Model:
    def __init__(self, buffers: dict[str, torch.Tensor]) -> None:
        self.buffers = buffers

    def named_buffers(self) -> list[tuple[str, torch.Tensor]]:
        return list(self.buffers.items())


class Observer:
    """The observer `main` installs, as it reads it: its roots, the ids it sampled and the stats it closes with."""

    def __init__(self, world: World, eng: Any, *, header: Any, log_path: Path, config: OC.ObserverConfig) -> None:
        self.world, self.roots, self.sample_ids = world, ["a", "b", "c"], {0: [7, 5], 1: [8, 2]}
        world.calls.append(("observer", eng, header, log_path, config))

    def install(self) -> None:
        self.world.calls.append(("installed",))

    def close(self) -> Record:
        self.world.calls.append(("closed",))
        return copy.deepcopy(self.world.stats)


class Hooks:
    """The acquisition plan's hooks beside the observer."""

    def __init__(self, world: World) -> None:
        self.world, self.installed, self.deferred = world, ["s0", "s1"], ["tap"]

    def keys(self) -> list[tuple[str, str, str]]:
        return [("module_outputs", "model.layers.0.mlp", "out"), ("module_outputs", "model.norm", "out")]

    def remove(self) -> None:
        self.world.calls.append(("hooks removed",))


class Protocols:
    """A target's executors, installed for the capture."""
    entries = ("pouw:pearl-c",)

    def __init__(self, world: World) -> None:
        self.world = world

    def close(self) -> None:
        self.world.calls.append(("protocols closed",))


class World:
    """What `main` reaches outside itself.  A test sets what a stand-in answers before it calls `main`, and reads `calls`."""

    def __init__(self, out: Path) -> None:
        self.out, self.clock = out, Clock()
        self.calls: list[tuple[Any, ...]] = []
        self.timeline: list[tuple[str, str, Record]] = []
        self.workload = workload()
        self.buffers: dict[str, torch.Tensor] = {}
        self.runs: list[tuple[Record, Record]] = [({"w0": [1], "w1": [2]}, timing(0.5)), (copy.deepcopy(SERVED), timing(2.0))]
        self.lag: bool | None = True
        self.plan: Any = None
        self.plan_refusal: str | None = None
        self.protocols: Protocols | None = None
        self.stats: Record = copy.deepcopy(STATS)
        self.engine = SimpleNamespace(build_s=1.5, cp={"repo": "org/model", "revision": "rev1"}, kwargs=dict(KWARGS), execution="eager",
                                      effective={"enforce_eager": True}, model=Model(self.buffers), runner="the runner", llm="the llm")
        self.header = SimpleNamespace(profile_id="profile-1", notes={"reenter_impls": {"op": "impl"}})

    def called(self, name: str) -> list[tuple[Any, ...]]:
        return [c[1:] for c in self.calls if c[0] == name]

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    # vllm_adapter's
    def load_workload(self, path: str, max_tokens: int | None = None) -> Record:
        self.calls.append(("workload", path, max_tokens))
        return self.workload

    def target_of_workload(self, wl: Record, target: str | None = None) -> str:
        return f"target of {wl['row_id']} over {target}"

    def build_engine(self, checkpoints: str, **kw: Any) -> Any:
        self.calls.append(("engine", checkpoints, kw))
        return self.engine

    def versions_doc(self, eng: Any) -> Record:
        assert eng is self.engine
        return {"vllm": "0.0.0", "torch": "2.0"}

    def throwaway_requests(self, eng: Any, like: list[Record], seed: int) -> list[Record]:
        self.calls.append(("throwaway", like, seed))
        return [{"request_id": f"w{i}"} for i in range(len(like))]

    def run_requests(self, eng: Any, requests: list[Record], sampling: Record) -> tuple[Record, Record]:
        tokens, took = self.runs.pop(0)
        self.calls.append(("run", [r["request_id"] for r in requests], sampling))
        self.clock.now += took["wall_s"]
        return tokens, took

    def resolved_async_scheduling(self, llm: Any) -> bool | None:
        assert llm == "the llm"
        return self.lag

    # the observer's, the acquisition's, the protocols'
    def make_header(self, eng: Any, wl: Record, **kw: Any) -> Any:
        self.calls.append(("header", kw))
        return self.header

    def plan_at_stage(self, model: Any, runner: Any, manifest: Any, out_dir: Path, *, source: str) -> stage.StageDocs:
        self.calls.append(("stage", runner, manifest, Path(out_dir), source))
        if self.plan_refusal is not None:
            raise install.IncompletePlan(self.plan_refusal)
        runtime: Any = SimpleNamespace(modules=["m0", "m1"], buffers=["b0"])
        return stage.StageDocs(runtime=runtime, plan=self.plan, paths={"runtime_tree": "runtime_tree.json"})

    def install_from_plan(self, plan: Any, model: Any, runner: Any = None, *, sink: Any = None) -> Hooks:
        self.calls.append(("hooks", plan, runner, sink))
        return Hooks(self)

    def capture_protocols(self, wl: Record, args: Any, model: Any) -> tuple[Protocols | None, list[str]]:
        self.calls.append(("protocols", model))
        return self.protocols, list(self.protocols.entries) if self.protocols is not None else []

    # the timeline's
    def from_env(self, proc: str, **defaults: Any) -> World:
        self.timeline.append(("timeline", proc, defaults))
        return self

    def event(self, name: str, ev: str = "mark", **fields: Any) -> None:
        self.timeline.append((name, ev, fields))

    def start(self, name: str, **fields: Any) -> None:
        self.timeline.append((name, "start", fields))

    def end(self, name: str, **fields: Any) -> None:
        self.timeline.append((name, "end", fields))

    @contextlib.contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[None]:
        self.timeline.append((name, "start", fields))
        yield
        self.timeline.append((name, "end", {}))


class Sink:
    """The occurrences the plan's hooks counted."""

    def to_json(self) -> Record:
        return {"sites": [{"site": "s0", "occurrences": 5}, {"site": "s1", "occurrences": 2}]}


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    w = World(tmp_path / "capture")
    for name in ("load_workload", "target_of_workload", "build_engine", "versions_doc", "throwaway_requests", "run_requests",
                 "resolved_async_scheduling"):
        monkeypatch.setattr(M1.va, name, getattr(w, name))
    monkeypatch.setattr(OC, "make_header", w.make_header)
    monkeypatch.setattr(OC, "Capture", lambda eng, **kw: Observer(w, eng, **kw))
    monkeypatch.setattr(stage, "plan_at_stage", w.plan_at_stage)
    monkeypatch.setattr(install, "install_from_plan", w.install_from_plan)
    monkeypatch.setattr(install, "OccurrenceSink", Sink)
    monkeypatch.setattr(M1, "capture_protocols", w.capture_protocols)
    monkeypatch.setattr(M1, "Timeline", w)
    monkeypatch.setattr(M1, "time", w.clock)
    monkeypatch.setattr(M1, "_T_PROCESS0", 998.5)
    monkeypatch.setattr(M1.sys, "argv", ["verity-vllm", "m1-capture"])
    monkeypatch.delenv(stage.MANIFEST_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    return w


@pytest.fixture
def said(capsys: pytest.CaptureFixture[str]) -> Lines:
    """The lines `m1_capture` has printed since the last call."""
    return lambda: capsys.readouterr().out.splitlines()


def options(w: World, **given: Any) -> M1.Options:
    return M1.Options(**{"out": str(w.out)} | given)


def refused(run: Callable[[], Any]) -> None:
    """``run`` ends the process with exit status 3."""
    with pytest.raises(SystemExit) as e:
        run()
    assert e.value.code == 3


def read(w: World, name: str) -> Any:
    return json.loads((w.out / name).read_text())


def row(rid: str, served: list[int], accepted: list[int] | None, equal: bool | None, *, finish: Record, executed: int | None,
        exhausted: bool, stopped: bool, lag: int | None) -> Record:
    """A request's row of `tokens.json`."""
    r = next(q for q in REQUESTS if q["request_id"] == rid)
    return {"request_id": rid, "prompt_token_ids": r["prompt_token_ids"], "max_tokens": 4, "generated_token_ids": served,
            "accepted_record_token_ids": accepted, "equals_accepted": equal, "prompt_len": len(r["prompt_token_ids"]),
            "n_generated": len(served), "arrive_step": int(r.get("arrive_step") or 0), **finish, "served_output_len": len(served),
            "executed_prefix_len": executed, "cap_exhausted": exhausted, "eos_stop_before_cap": stopped, "lag": lag}


ROW0 = row("r0", [7, 8, 9, 10], [7, 8, 9, 10], True, finish=STEPS["r0"], executed=4, exhausted=True, stopped=False, lag=1)
ROW1 = row("r1", [5, 2], [5, 2], True, finish=STEPS["r1"], executed=3, exhausted=False, stopped=True, lag=1)
BUILT = "[m1] engine built in 1.5s (case B0, org/model); kwargs={'enforce_eager': True, 'max_num_seqs': 2}"
TREE = "[m1] runtime tree 2 modules, 1 buffers -> runtime_tree.json"
WARMED = "[m1] warm-up (uninstrumented, 2 throwaway requests): 0.50s, 5 steps"
INSTALLED = "[m1] observer installed: 3 roots, profile_id=profile-1, reenter impls={'op': 'impl'}"
OBSERVED = "[m1] observed run 2.00s: 40 events, 5 steps, recorder=pinned, storage=arena, snapshot_bytes=0, errors=0"
SAID0 = ["[m1] r0: prompt_len=3 arrive=0 steps=0..3 (length) generated [7, 8, 9, 10]", "[m1]   accepted  [7, 8, 9, 10]  equal=True"]
SAID1 = ["[m1] r1: prompt_len=2 arrive=0 steps=1..2 (stop) generated [5, 2]", "[m1]   accepted  [5, 2]  equal=True"]


def test_the_capture_arm_serves_the_workload_under_the_observer_and_records_what_was_served(world: World, said: Lines) -> None:
    M1.main(options(world, snapshot_steps="0,2", max_tokens=4, engine_arg=["max_num_batched_tokens=64"], target="t.json"))
    out = world.out
    assert said() == [BUILT, "[m1] cos_sin_cache -> None", TREE, WARMED, INSTALLED, OBSERVED, *SAID0, *SAID1, f"[m1] done -> {out}"]
    assert world.names() == ["workload", "engine", "stage", "throwaway", "run", "protocols", "header", "observer", "installed", "run",
                             "closed"]
    assert world.called("workload") == [("workloads/workload_32x16_1req.json", 4)]
    assert world.called("engine") == [("manifests/checkpoints.json", {
        "max_num_seqs": 8, "case": "B0", "verify_weights": False, "engine_args": {"max_num_batched_tokens": 64},
        "target": "target of the-row over t.json", "execution": None})]
    assert world.called("stage") == [("the runner", None, out, "m1_capture")]
    assert world.called("throwaway") == [(world.workload["requests"], M1.va.WARMUP_SEED)]
    assert world.called("run") == [(["w0", "w1"], SAMPLING), (["r0", "r1"], SAMPLING)]
    assert world.called("header") == [({"adapter_revision": "unknown", "wheel_sha256": M1.Options(out="").wheel_sha256,
                                        "vllm_commit": "d9105ea80", "extra_notes": {
                                            "argv": ["verity-vllm", "m1-capture"], "snapshot_steps": [0, 2], "weights_verified": False,
                                            "protocols": []}},)]
    (eng, header, log_path, config), = world.called("observer")
    assert eng is world.engine and header is world.header and log_path == out / "log.jsonl.gz"
    assert config == OC.ObserverConfig(mode="record", reenter=True, triton=True, module_hooks=True, write_log=True,
                                       snapshot_steps=frozenset({0, 2}), snapshot_dir=out, snapshot_max_bytes=32 << 20,
                                       snapshot_max_bytes_op=None)
    assert read(world, "versions.json") == {"vllm": "0.0.0", "torch": "2.0"}
    assert read(world, "stats.json") == STATS
    assert read(world, "spec_tokens.json") == {
        "schema": "verity-gen/spec-tokens/v1", "known_before_execution": True, "tokenizer": {"repo": "org/model", "revision": "rev1"},
        "requests": [{"request_id": "r0", "prompt_token_ids": [11, 12, 13], "prompt_len": 3, "max_tokens": 4, "arrive_step": 0},
                     {"request_id": "r1", "prompt_token_ids": [21, 22], "prompt_len": 2, "max_tokens": 4, "arrive_step": 0}],
        "sampling": SAMPLING, "engine_kwargs": SHOWN, "row_id": "the-row", "declared_execution": "eager",
        "effective_execution": {"enforce_eager": True}}
    tokens = read(world, "tokens.json")
    assert list(tokens) == ["requests", "per_step_sampled", "timing", "observer", "protocols", "profile_id", "case", "model",
                            "serving_profile", "context_class", "expected_engine_steps", "engine_steps", "lag", "all_equal_accepted"]
    assert [list(r) for r in tokens["requests"]] == [list(ROW0), list(ROW1)]
    assert tokens.pop("lag").items() >= {"value": 1, "async_scheduling": True}.items()
    assert tokens == {"requests": [ROW0, ROW1], "per_step_sampled": {"0": [7, 5], "1": [8, 2]}, "timing": timing(2.0), "observer": True,
                      "protocols": [], "profile_id": "profile-1", "case": "B0", "model": "org/model", "serving_profile": "mixed",
                      "context_class": "i256", "expected_engine_steps": 5, "engine_steps": 5, "all_equal_accepted": True}
    assert sorted(p.name for p in out.iterdir()) == ["spec_tokens.json", "stats.json", "tokens.json", "versions.json"]
    assert world.timeline == [
        ("timeline", "m1_capture", {"hot": False, "arm": "capture"}), ("startup.imports", "end", {"s": 1.5}),
        ("engine.build", "start", {"case": "B0", "max_num_seqs": 8}),
        ("engine.build", "end", {"build_s": 1.5, "model": "org/model", "enforce_eager": True, "engine_reused": False,
                                 "restart_reason": "fresh process (one m1_capture per Match arm)"}),
        ("prep.warmup_control", "start", {}), ("prep.warmup_control", "end", {}), ("work.capture_run", "start", {}),
        ("work.capture_run", "end", {"run_s": 2.0, "events": 40, "steps": 5}), ("process.end", "mark", {"rc": 0, "s": 4.0})]


def test_the_control_arm_serves_the_same_workload_with_no_observer(world: World, said: Lines) -> None:
    M1.main(options(world, no_observer=True, case="B1", max_num_seqs=2, verify_weights=True, adapter_revision="abc123"))
    assert said() == ["[m1] engine built in 1.5s (case B1, org/model); kwargs={'enforce_eager': True, 'max_num_seqs': 2}",
                      "[m1] cos_sin_cache -> None", TREE, WARMED, "[m1] control run (no observer): 2.00s", *SAID0, *SAID1,
                      f"[m1] done -> {world.out}"]
    assert world.names() == ["workload", "engine", "stage", "throwaway", "run", "protocols", "run"]
    (_, built), = world.called("engine")
    assert (built["case"], built["max_num_seqs"], built["verify_weights"]) == ("B1", 2, True)
    tokens = read(world, "tokens.json")
    assert (tokens["observer"], tokens["per_step_sampled"], tokens["profile_id"], tokens["case"]) == (False, None, None, "B1")
    assert tokens["requests"] == [ROW0, ROW1] and tokens["all_equal_accepted"] is True
    assert sorted(p.name for p in world.out.iterdir()) == ["spec_tokens.json", "tokens.json", "versions.json"]
    assert world.timeline[0] == ("timeline", "m1_capture", {"hot": False, "arm": "control"})
    assert [e[:2] for e in world.timeline[4:]] == [("prep.warmup_control", "start"), ("prep.warmup_control", "end"),
                                                   ("work.control_run", "start"), ("work.control_run", "end"), ("process.end", "mark")]


def test_with_no_warm_up_the_run_of_record_is_the_engine_s_first(world: World, said: Lines) -> None:
    del world.runs[0]
    M1.main(options(world, no_warmup=True, no_reenter=True, no_triton=True, no_module_hooks=True, snapshot_max_bytes=64,
                    snapshot_max_bytes_op=8))
    assert said() == [BUILT, "[m1] cos_sin_cache -> None", TREE, INSTALLED, OBSERVED, *SAID0, *SAID1, f"[m1] done -> {world.out}"]
    assert world.called("throwaway") == [] and world.called("run") == [(["r0", "r1"], SAMPLING)]
    (_, _, _, config), = world.called("observer")
    assert config == OC.ObserverConfig(mode="record", reenter=False, triton=False, module_hooks=False, write_log=True,
                                       snapshot_steps=frozenset(), snapshot_dir=None, snapshot_max_bytes=64, snapshot_max_bytes_op=8)


@pytest.mark.parametrize("lag,want", [(False, (0, 2)), (None, (None, None))])
def test_a_request_stopped_early_executes_the_lag_s_steps_past_what_it_served(world: World, said: Lines, lag: bool | None,
                                                                             want: tuple[int | None, int | None]) -> None:
    """The lag is the engine's resolved scheduling, read back: with none attested, the executed length of a stopped request is
    not stated."""
    world.lag = lag
    M1.main(options(world))
    tokens = read(world, "tokens.json")
    value, executed = want
    assert (tokens["lag"]["value"], tokens["lag"]["async_scheduling"]) == (value, lag)
    assert tokens["requests"] == [ROW0 | {"lag": value}, ROW1 | {"lag": value, "executed_prefix_len": executed}]
    assert len(said()) == 11


def test_a_request_that_stops_on_its_last_allowed_token_executes_no_step_past_it(world: World) -> None:
    """A stop on the last token the request may have is the limit reached, with nothing run past what was served."""
    world.runs[1] = (copy.deepcopy(SERVED), timing(2.0, steps=STEPS | {"r0": STEPS["r0"] | {"finish_reason": "stop"}}))
    M1.main(options(world))
    assert read(world, "tokens.json")["requests"] == [ROW0 | {"finish_reason": "stop"}, ROW1]


UNTIMED: Record = {"arrive_step_realized": None, "first_step": None, "finish_step": None, "finish_reason": None}


@pytest.mark.parametrize("steps", [None, {}, {"r0": STEPS["r0"]}, {"r0": STEPS["r0"], "r1": None}, {"r0": STEPS["r0"], "r1": {}}])
def test_a_request_the_run_s_timing_has_no_record_of_has_no_realized_step_and_did_not_stop_early(world: World, said: Lines,
                                                                                               steps: Record | None) -> None:
    """The timing may hold no `requests` at all, or none for one request: that request's row says which steps it ran as unknown,
    and its executed length is what it served."""
    run_timing = timing(2.0, steps=steps)
    if steps is None:
        del run_timing["requests"]
    world.runs[1] = (copy.deepcopy(SERVED), run_timing)
    M1.main(options(world))
    untimed = row("r1", [5, 2], [5, 2], True, finish=UNTIMED, executed=2, exhausted=False, stopped=False, lag=1)
    assert read(world, "tokens.json")["requests"] == [ROW0 if steps else ROW0 | UNTIMED, untimed]
    assert said()[-3:-1] == ["[m1] r1: prompt_len=2 arrive=0 steps=None..None (None) generated [5, 2]",
                             "[m1]   accepted  [5, 2]  equal=True"]


def test_one_request_s_unequal_tokens_make_the_run_s_unequal(world: World) -> None:
    world.runs[1] = ({"r0": [7, 8, 9, 10], "r1": [5, 3]}, timing(2.0))
    M1.main(options(world))
    tokens = read(world, "tokens.json")
    assert [r["equals_accepted"] for r in tokens["requests"]] == [True, False] and tokens["all_equal_accepted"] is False


def test_tokens_that_differ_from_the_accepted_ones_are_recorded_as_unequal(world: World, said: Lines) -> None:
    world.runs[1] = ({"r0": [7, 8, 9, 10], "r1": [5, 3, 2], "1_r1": [5, 4]}, timing(2.0))
    world.workload["accepted"].pop("r0")
    M1.main(options(world))
    tokens = read(world, "tokens.json")
    assert tokens["all_equal_accepted"] is False
    assert tokens["requests"] == [
        row("r0", [7, 8, 9, 10], None, None, finish=STEPS["r0"], executed=4, exhausted=True, stopped=False, lag=1),
        row("r1", [5, 3, 2], [5, 2], False, finish=STEPS["r1"], executed=4, exhausted=False, stopped=True, lag=1)]
    assert tokens["children"] == [{"request_id": "1_r1", "generated_token_ids": [5, 4]}] and list(tokens)[-1] == "children"
    assert said()[6:10] == ["[m1] r0: prompt_len=3 arrive=0 steps=0..3 (length) generated [7, 8, 9, 10]", "[m1]   accepted  None  equal=None",
                            "[m1] r1: prompt_len=2 arrive=0 steps=1..2 (stop) generated [5, 3, 2]", "[m1]   accepted  [5, 2]  equal=False"]


def test_the_observer_s_errors_and_its_log_s_size_are_said(world: World, said: Lines) -> None:
    world.stats["errors"] = [{"step": 1, "why": "a view outside its storage"}]
    world.out.mkdir()
    (world.out / "log.jsonl.gz").write_bytes(bytes(2_500_000))
    M1.main(options(world))
    first, *rest = json.dumps(world.stats["errors"], indent=1).splitlines()
    assert said()[5:-5] == ["[m1] observed run 2.00s: 40 events, 5 steps, recorder=pinned, storage=arena, snapshot_bytes=0, errors=1",
                            f"[m1] ERRORS: {first}", *rest, "[m1] log size 2.5 MB"]
    assert read(world, "stats.json")["errors"] == world.stats["errors"]


def test_the_plan_s_hooks_count_beside_the_observer_and_come_off_before_it_closes(world: World, said: Lines,
                                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    world.plan = SimpleNamespace(digest="d" * 64, values=["v0", "v1"], unsupported=[], residuals=[], hooks=lambda: ["h0", "h1", "h2"])
    monkeypatch.setenv(stage.MANIFEST_ENV, "/row/required_manifest.json")
    M1.main(options(world))
    observed = world.out / "acquisition_observed.json"
    hooked = ("[m1] acquisition plan hooks installed beside the observer: 2 sites (1 deferred to their sources: taps / registered "
              "inputs / prescribed)")
    assert said()[2:9] == [
        TREE, f"[m1] acquisition plan {'d' * 16}: 2 Value classes, 0 unsupported, 3 hooks, 0 residuals -> None", WARMED, INSTALLED,
        hooked, f"[m1] acquisition observed: 7 occurrences over 2 sites -> {observed}", OBSERVED]
    assert world.called("stage") == [("the runner", "/row/required_manifest.json", world.out, "m1_capture")]
    (plan, runner, sink), = world.called("hooks")
    assert plan is world.plan and runner == "the runner" and isinstance(sink, Sink)
    assert world.names()[-5:] == ["installed", "hooks", "run", "hooks removed", "closed"]
    assert json.loads(observed.read_text()) == {
        "sites": [{"site": "s0", "occurrences": 5}, {"site": "s1", "occurrences": 2}], "plan_digest": "d" * 64,
        "installed": [["module_outputs", "model.layers.0.mlp", "out"], ["module_outputs", "model.norm", "out"]]}


def test_the_option_names_the_manifest_before_the_environment_does(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(stage.MANIFEST_ENV, "/row/required_manifest.json")
    M1.main(options(world, acquisition_manifest="/given/manifest.json", no_observer=True))
    assert world.called("stage") == [("the runner", "/given/manifest.json", world.out, "m1_capture")]


def test_a_plan_with_an_unsupported_value_refuses_before_any_run(world: World, said: Lines) -> None:
    world.plan_refusal = "1 unsupported Value(s): v0: no site"
    refused(lambda: M1.main(options(world)))
    assert said() == [BUILT, "[m1] cos_sin_cache -> None", "[m1] REFUSE: acquisition plan: 1 unsupported Value(s): v0: no site"]
    assert world.called("run") == [] and not (world.out / "tokens.json").exists()
    assert sorted(p.name for p in world.out.iterdir()) == ["spec_tokens.json", "versions.json"]


def test_the_protocols_serve_the_run_and_are_closed_after_it(world: World, said: Lines) -> None:
    world.protocols = Protocols(world)
    M1.main(options(world))
    (header,), = world.called("header")
    assert header["extra_notes"]["protocols"] == ["pouw:pearl-c"] and read(world, "tokens.json")["protocols"] == ["pouw:pearl-c"]
    assert world.called("protocols") == [(world.engine.model,)] and world.names()[-1] == "protocols closed"
    assert len(said()) == 11


def test_a_declared_static_schedule_the_warm_up_also_ran_is_reported(world: World, said: Lines) -> None:
    world.runs = [({}, timing(0.5, schedule=declared("5c" * 32))), (copy.deepcopy(SERVED), timing(2.0, schedule=declared("5c" * 32)))]
    M1.main(options(world))
    assert said()[-2:] == [f"[m1] static schedule: ok (digest {'5c' * 8}, warm-up equal)", f"[m1] done -> {world.out}"]
    assert read(world, "tokens.json")["timing"]["static_schedule"]["content_invariance"]["ok"] is True


def test_a_run_that_is_not_its_declared_static_schedule_ends_with_exit_3_after_its_tokens_are_written(world: World,
                                                                                                    said: Lines) -> None:
    failed = declared("5c" * 32, ok=False, problems=["r1 stopped at step 2 of 4"])
    world.runs = [(copy.deepcopy(SERVED), timing(2.0, schedule=failed))]
    refused(lambda: M1.main(options(world, no_warmup=True, no_observer=True)))
    assert said()[-3:] == [f"[m1] static schedule: FAILED (digest {'5c' * 8}, no warm-up to compare)",
                           "[m1] REFUSE: the workload declares the static schedule (sampling.ignore_eos) and the run is not it:",
                           "  r1 stopped at step 2 of 4"]
    assert read(world, "tokens.json")["requests"] == [ROW0, ROW1]
    assert world.timeline[-1][0] == "work.control_run"


def test_a_warm_up_that_ran_another_schedule_fails_the_run_s_declared_one(world: World, said: Lines) -> None:
    """The warm-up serves the same lengths with other tokens: a schedule that differs from it depends on the prompts' contents."""
    world.runs = [({}, timing(0.5, schedule=declared("aa" * 32))), (copy.deepcopy(SERVED), timing(2.0, schedule=declared("5c" * 32)))]
    refused(lambda: M1.main(options(world, no_observer=True)))
    assert said()[-3:] == [f"[m1] static schedule: FAILED (digest {'5c' * 8}, warm-up DIFFERS)",
                           "[m1] REFUSE: the workload declares the static schedule (sampling.ignore_eos) and the run is not it:",
                           "  the schedule depends on the prompts' contents, not only on the declaration: digests differ"]
    assert read(world, "tokens.json")["timing"]["static_schedule"]["ok"] is False


def words(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def test_the_rotary_table_s_words_are_written_as_the_engine_holds_them(world: World, said: Lines) -> None:
    table = torch.arange(12, dtype=torch.float32).reshape(4, 3).to(torch.bfloat16)
    world.buffers.update({"model.embed.weight_scale": torch.ones(2), ROTARY: table, "model.normalizer": torch.tensor(8.0).to(torch.bfloat16)})
    M1.main(options(world, no_observer=True))
    out = world.out
    assert said()[1:3] == [f"[m1] cos_sin_cache -> {out / 'cos_sin_cache.npy'}", f"[m1] normalizer -> {out / 'normalizer.npy'}"]
    got = np.load(out / "cos_sin_cache.npy")
    assert got.dtype == np.uint16 and got.shape == (4, 3) and (got == words(table)).all()
    assert read(world, "cos_sin_cache.json") == {"dtype": "bfloat16", "shape": [4, 3], "file": "cos_sin_cache.npy",
                                                "words": "uint16 (raw bfloat16 words)", "buffer": ROTARY}
    one = np.load(out / "normalizer.npy")
    assert one.dtype == np.uint16 and one.shape == () and int(one) == int(words(torch.tensor([8.0]).to(torch.bfloat16))[0])


def test_a_half_precision_table_is_written_under_its_own_name(world: World, tmp_path: Path) -> None:
    table = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    world.buffers[ROTARY] = table
    path = M1.dump_cos_sin(world.engine, tmp_path)
    assert path == tmp_path / "cos_sin_cache.float16.npy" and (np.load(path) == words(table)).all()
    assert json.loads((tmp_path / "cos_sin_cache.json").read_text()) == {
        "dtype": "float16", "shape": [2, 3], "file": "cos_sin_cache.float16.npy", "words": "uint16 (raw IEEE binary16 words)",
        "buffer": ROTARY}
    assert not (tmp_path / "cos_sin_cache.npy").exists()


@pytest.mark.parametrize("buffers", [{}, {ROTARY: torch.zeros(2, 3)}, {"model.normalizer": torch.zeros(1).to(torch.bfloat16)},
                                     {"model.normalizer": torch.tensor(8.0)}])
def test_a_table_of_another_type_or_shape_is_not_written(world: World, tmp_path: Path, buffers: dict[str, torch.Tensor]) -> None:
    world.buffers.update(buffers)
    assert M1.dump_cos_sin(world.engine, tmp_path) is None and M1.dump_normalizer(world.engine, tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_the_adapter_s_revision_is_the_option_s_then_the_exported_tree_s(world: World, tmp_path: Path) -> None:
    assert M1.adapter_revision(options(world, adapter_revision="abc123")) == "abc123"
    assert M1.adapter_revision(options(world)) == "unknown"
    (tmp_path / "EXPORT.json").write_text(json.dumps({"commit": "f00d"}))
    assert M1.adapter_revision(options(world)) == "f00d"
    (tmp_path / "EXPORT.json").write_text("[]")
    assert M1.adapter_revision(options(world)) == "unknown"


def test_the_steps_to_snapshot_are_the_option_s_numbers() -> None:
    assert M1.parse_steps(None) == frozenset() and M1.parse_steps("") == frozenset()
    assert M1.parse_steps("0,1, 8,") == frozenset({0, 1, 8})

"""The recorder's trace writer on CPU: the open-loop schedule, chunk stamping on an injected clock, the conversion to the
calibration tool's format with its exclusions, the merge, and the offline engine's loop and a whole job on a fake
engine, evicted or not. No GPU, no server, no clock."""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import signal
import sys
import types
from pathlib import Path

import pytest
from verity.core.protocols.certifier import calibration

HERE = Path(__file__).resolve().parents[1]


def _record():
    spec = importlib.util.spec_from_file_location("network_traces_record", HERE / "record.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rec = _record()
WINDOW_MS = rec.WINDOW_BUCKETS * rec.BUCKET_MS


def request(rid: str, send_ms: float, chunks: list, *, prompt_len: int = 100, ok: bool = True) -> dict:
    return {"rid": rid, "send_ms": send_ms, "chunks": chunks, "prompt_len": prompt_len, "request_bytes": 9000,
            "ok": ok, "max_tokens": sum(n for _, n in chunks)}


def samples(windows: int, **at: dict) -> list[dict]:
    """A sample every SAMPLE_S from t0 through the tail, quiet unless `at` overrides the sample at that t_ms."""
    step = int(1000 * rec.SAMPLE_S)
    out = []
    for t in range(step, windows * WINDOW_MS + rec.TAIL_MS + step, step):
        s = {"t_ms": float(t), "dt_s": rec.SAMPLE_S, "hb_max_ms": 1.0, "loop_max_ms": 2.0, "foreign_pids": [],
             "others_busy": 0.05, "load": [1.0, 1.0, 1.0], "own_cores": 2.0, "sm_mhz": 2400}
        out.append(s | at.get(f"t{t}", {}))
    return out


def test_the_schedule_is_open_loop_seeded_and_never_shares_a_prefix() -> None:
    wl = [{"index": 0, "offset_s": 0.0, "prompt": [7, 8, 9], "max_tokens": 4},
          {"index": 1, "offset_s": 0.5, "prompt": [7, 8, 9, 10], "max_tokens": 2}]
    a = rec.schedule([("b4", wl, 2.0)], -2.0, 6.0, seed=0)
    assert a == rec.schedule([("b4", wl, 2.0)], -2.0, 6.0, seed=0)
    assert a != rec.schedule([("b4", wl, 2.0)], -2.0, 6.0, seed=1)
    assert [q["instance"] for q in a if q["index"] == 0] == [0, 1, 2, 3]
    for q in a:
        k = q["instance"]
        base = q["at_s"] - wl[q["index"]]["offset_s"]
        assert -2.0 + 2 * k <= base < -2.0 + 2 * k + 1.0
    assert [q["at_s"] for q in a] == sorted(q["at_s"] for q in a)
    assert len({tuple(q["prompt"]) for q in a}) == len(a)
    assert all(q["prompt"][1:] == wl[q["index"]]["prompt"][1:] for q in a)


def test_a_stream_stamps_each_chunk_with_its_tokens_on_the_injected_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.25, 10.5, 10.75, 11.0])
    monkeypatch.setattr(rec, "clock", lambda: next(times))
    s = rec.Stream(t0=10.0)
    for line in [b": keep-alive\n",
                 b'data: {"choices": [{"token_ids": [5], "prompt_token_ids": [1, 2]}]}\n', b"\n",
                 b'data: {"choices": [{"token_ids": [6, 7]}]}\n',
                 b'data: {"choices": [{"token_ids": [], "finish_reason": "length"}]}\n',
                 b"data: [DONE]\n"]:
        s.line(line)
    assert s.chunks == [[250.0, 1], [500.0, 2]] and s.done


def test_a_chunk_becomes_frames_in_canonical_order_and_ingress_frames() -> None:
    records = [request("a", 50.0, [[120.0, 1], [130.5, 2]], prompt_len=1000),
               request("b", 99.9, [[125.0, 1], [210.0, 1]]),
               request("c", WINDOW_MS + 50.0, [[WINDOW_MS + 210.0, 1], [WINDOW_MS + 300.0, 1]])]
    configs, summary = rec.build_trace(records, samples(2), name="t", windows=2)
    assert summary["segments"] == [(0, 2)] and summary["windows_kept"] == 2 and not summary["frozen"]
    (cfg,) = configs
    frames = [(f["session"], f["index"], f["anchor"], f["ready"]) for f in cfg["egress"][0]["frames"]]
    assert frames == [("a", 0, 0, 2), ("b", 0, 0, 2), ("a", 1, 0, 2), ("a", 2, 0, 2), ("b", 1, 0, 3),
                      ("c", 0, 600, 603), ("c", 1, 600, 604)]
    assert cfg["ingress"][0]["arrival_ms"] == [50.0, 50.0, 99.9, WINDOW_MS + 50.0]
    (body,), _ = rec.build_trace(records, samples(2), name="t", windows=2, ingress="body")
    assert body["ingress"][0]["arrival_ms"] == [50.0] * 3 + [99.9] * 3 + [WINDOW_MS + 50.0] * 3
    report = calibration.calibrate(rec.trace_doc(configs))
    assert (report["egress"][0]["honest_windows"], report["egress"][0]["honest_misses"]) == (2, 0)


def test_contamination_and_failures_drop_windows_and_split_the_chunk() -> None:
    records = [request("a", 50.0, [[120.0, 1]], ok=False), request("b", WINDOW_MS + 50.0, [[WINDOW_MS + 150.0, 1]]),
               request("c", 2 * WINDOW_MS + 50.0, [[2 * WINDOW_MS + 150.0, 1]]),
               request("d", 3 * WINDOW_MS + 50.0, [[3 * WINDOW_MS + 150.0, 1]])]
    quiet_gpu = {f"t{2 * WINDOW_MS + 2000}": {"foreign_pids": [4242]}}
    configs, summary = rec.build_trace(records, samples(4, **quiet_gpu), name="t", windows=4)
    assert summary["windows_dropped"] == {"0": "request failed", "2": "foreign GPU process"}
    assert summary["segments"] == [(1, 2), (3, 4)] and summary["foreign_pids"] == [4242]
    assert [c["name"] for c in configs] == ["t/w1-2", "t/w3-4"]
    assert [f["anchor"] for c in configs for f in c["egress"][0]["frames"]] == [0, 0]
    busy = {f"t{WINDOW_MS}": {"others_busy": 0.9}}
    _, summary = rec.build_trace(records[1:], samples(4, **busy), name="t", windows=4)
    assert summary["windows_dropped"] == {"0": "NUMA node busy", "1": "NUMA node busy"}
    crowded = {f"t{WINDOW_MS}": {"affinity_busy": 0.99}}
    _, summary = rec.build_trace(records[1:], samples(4, **crowded), name="t", windows=4, affinity_cpus=32)
    assert summary["windows_dropped"] == {"0": "co-tenants on the job's CPUs", "1": "co-tenants on the job's CPUs"}


def test_the_job_cpu_rule_counts_co_tenants_not_our_own_use() -> None:
    """On an 8-CPU cpuset, window 1 is 95% busy with the job using 6 cores (co-tenants 20%), and window 3 is 95% busy
    with the job using 0.8 (co-tenants 85%, under the line) and then 0.16 (93%). The co-tenant rule keeps window 1 and
    drops window 3; the total rule drops both."""
    records = [request(c, w * WINDOW_MS + 50.0, [[w * WINDOW_MS + 150.0, 1]]) for w, c in enumerate("abcd")]
    at = {f"t{WINDOW_MS + 30_000}": {"affinity_busy": 0.95, "own_cores": 6.0},
          f"t{3 * WINDOW_MS + 20_000}": {"affinity_busy": 0.95, "own_cores": 0.8},
          f"t{3 * WINDOW_MS + 30_000}": {"affinity_busy": 0.95, "own_cores": 0.16}}
    configs, summary = rec.build_trace(records, samples(4, **at), name="t", windows=4, affinity_cpus=8)
    assert summary["windows_dropped"] == {"3": "co-tenants on the job's CPUs"} and summary["cpu_rule"] == "cotenants"
    assert [c["name"] for c in configs] == ["t/w0-3"] and summary["cotenant_busy"]["max"] == pytest.approx(0.93)
    _, total = rec.build_trace(records, samples(4, **at), name="t", windows=4, affinity_cpus=8, cpu_rule="total")
    assert total["windows_dropped"] == {"1": "job's CPUs saturated", "3": "job's CPUs saturated"}
    _, unsized = rec.build_trace(records, samples(4, **at), name="t", windows=4)
    assert unsized["windows_dropped"] == total["windows_dropped"] | {"1": "co-tenants on the job's CPUs",
                                                                      "3": "co-tenants on the job's CPUs"}
    with pytest.raises(ValueError, match="cpu_rule"):
        rec.build_trace(records, samples(4), name="t", windows=4, cpu_rule="own")


def write_job(out: Path, windows: int, records: list[dict], sampled: list[dict]) -> None:
    """A job directory as `run` leaves it, on an 8-CPU cpuset, for `convert`."""
    meta = {"t0_mono": 0.0, "windows": windows, "mode": "plaintext", "model": "m", "mix": "b4=1", "host": "h",
            "gpu": {"name": "g", "uuid": "u"}, "env": {"RESEARCH_RUN_ID": "r1"}, "affinity": list(range(8))}
    (out / "meta.json").write_text(json.dumps(meta))
    (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (out / "samples.jsonl").write_text("".join(json.dumps(s | {"mono": s["t_ms"] / 1000}) + "\n" for s in sampled))


def test_convert_takes_the_cpuset_from_meta_and_the_rule_from_its_flag(tmp_path: Path) -> None:
    records = [request(c, w * WINDOW_MS + 50.0, [[w * WINDOW_MS + 150.0, 1]]) for w, c in enumerate("ab")]
    write_job(tmp_path, 2, records, samples(2, **{f"t{30_000}": {"affinity_busy": 0.95, "own_cores": 6.0}}))
    assert rec.main(["convert", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "summary.json").read_text())["windows_kept"] == 2
    assert rec.main(["convert", str(tmp_path), "--cpu-rule", "total"]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["windows_dropped"] == {"0": "job's CPUs saturated"} and summary["cpu_rule"] == "total"


@pytest.mark.parametrize("field", ["hb_max_ms", "loop_max_ms"])
@pytest.mark.parametrize(("at", "stall", "dropped"), [(WINDOW_MS + 10_000, 3000.0, [1, 2]),
                                                      (WINDOW_MS + 2000, 4000.0, [0, 1, 2]),
                                                      (WINDOW_MS + 10_000, rec.FREEZE_REST_MS, [1, 2])])
def test_a_stall_drops_the_windows_it_overlaps_and_the_next(field: str, at: int, stall: float, dropped: list) -> None:
    """Inside window 1, from window 0 into window 1, and at the guard but not over it."""
    records = [request(c, w * WINDOW_MS + 50.0, [[w * WINDOW_MS + 150.0, 1]]) for w, c in enumerate("abcde")]
    configs, summary = rec.build_trace(records, samples(5, **{f"t{at}": {field: stall}}), name="t", windows=5)
    assert not summary["frozen"] and summary["freeze_scope"] == "window"
    assert summary["windows_dropped"] == {str(w): "stalled" for w in dropped}
    assert summary["windows_kept"] == 5 - len(dropped) == sum(hi - lo for lo, hi in summary["segments"])
    assert [c["name"] for c in configs] == [f"t/w{lo}-{hi}" for lo, hi in summary["segments"]]


def test_a_stall_over_the_guard_drops_the_rest_of_its_chunk() -> None:
    records = [request(c, w * WINDOW_MS + 50.0, [[w * WINDOW_MS + 150.0, 1]]) for w, c in enumerate("abcde")]
    long = {f"t{WINDOW_MS + 10_000}": {"hb_max_ms": rec.FREEZE_REST_MS + 1}}
    configs, summary = rec.build_trace(records, samples(5, **long), name="t", windows=5)
    assert not summary["frozen"] and summary["windows_dropped"] == {str(w): "stalled" for w in range(1, 5)}
    assert [c["name"] for c in configs] == ["t/w0-1"]
    _, raised = rec.build_trace(records, samples(5, **long), name="t", windows=5, freeze_rest_ms=10_000.0)
    assert raised["windows_dropped"] == {"1": "stalled", "2": "stalled"} and raised["segments"] == [(0, 1), (3, 5)]


@pytest.mark.parametrize("stall", [{"hb_max_ms": 1500.0}, {"loop_max_ms": 1500.0}])
def test_the_chunk_scope_voids_the_whole_chunk_on_any_stall(stall: dict) -> None:
    """The old rule, kept for sensitivity: where the window rule drops windows 1 and 2, the chunk rule keeps nothing."""
    records = [request(c, w * WINDOW_MS + 50.0, [[w * WINDOW_MS + 150.0, 1]]) for w, c in enumerate("abcd")]
    at = {f"t{WINDOW_MS + 10_000}": stall}
    configs, summary = rec.build_trace(records, samples(4, **at), name="t", windows=4, freeze_scope="chunk")
    assert configs == [] and summary["frozen"] and summary["windows_kept"] == 0 and summary["freeze_scope"] == "chunk"
    _, window = rec.build_trace(records, samples(4, **at), name="t", windows=4)
    assert not window["frozen"] and window["segments"] == [(0, 1), (3, 4)]
    with pytest.raises(ValueError, match="freeze_scope"):
        rec.build_trace(records, samples(4), name="t", windows=4, freeze_scope="segment")


def test_convert_takes_the_freeze_rule_from_its_flags(tmp_path: Path) -> None:
    records = [request(c, w * WINDOW_MS + 50.0, [[w * WINDOW_MS + 150.0, 1]]) for w, c in enumerate("abcde")]
    write_job(tmp_path, 5, records, samples(5, **{f"t{WINDOW_MS + 10_000}": {"loop_max_ms": 8000.0}}))

    def kept(*flags: str) -> tuple:
        assert rec.main(["convert", str(tmp_path), *flags]) == 0
        summary = json.loads((tmp_path / "summary.json").read_text())
        return summary["segments"], summary["frozen"]

    assert kept() == ([[0, 1]], False)
    assert kept("--freeze-rest-ms", "10000") == ([[0, 1], [3, 5]], False)
    assert kept("--freeze-scope", "chunk") == ([], True)


def test_convert_refuses_a_job_without_its_requests(tmp_path: Path) -> None:
    """An evicted `requests.jsonl` would otherwise convert to kept windows with no traffic."""
    write_job(tmp_path, 2, [], samples(2))
    assert rec.main(["convert", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "summary.json").read_text())["windows_kept"] == 2
    (tmp_path / "requests.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="requests.jsonl"):
        rec.main(["convert", str(tmp_path)])


def test_the_windows_after_the_monitor_stopped_are_dropped() -> None:
    records = [request(c, w * WINDOW_MS + 50.0, [[w * WINDOW_MS + 150.0, 1]]) for w, c in enumerate("abc")]
    died = [s for s in samples(3) if s["t_ms"] <= WINDOW_MS + 30_000]
    configs, summary = rec.build_trace(records, died, name="t", windows=3)
    assert summary["windows_dropped"] == {"1": "unmonitored", "2": "unmonitored"}
    assert [c["name"] for c in configs] == ["t/w0-1"]


def test_a_late_sample_while_the_heartbeat_ran_drops_only_its_windows() -> None:
    records = [request("a", 50.0, [[120.0, 1]]), request("b", WINDOW_MS + 50.0, [[WINDOW_MS + 150.0, 1]])]
    late = {f"t{WINDOW_MS + 30_000}": {"dt_s": 5.0}}
    configs, summary = rec.build_trace(records, samples(2, **late), name="t", windows=2)
    assert not summary["frozen"] and summary["windows_dropped"] == {"1": "monitor late"}
    assert summary["segments"] == [(0, 1)] and [c["name"] for c in configs] == ["t/w0-1"]
    assert summary["sample_gap_ms_max"] == 3000.0


def test_a_stopped_chunk_keeps_the_whole_windows_before_the_stop() -> None:
    records = [request("a", 50.0, [[120.0, 1]]), request("b", WINDOW_MS + 50.0, [[WINDOW_MS + 150.0, 1]], ok=False)]
    _, summary = rec.build_trace(records, samples(3), name="t", windows=3, stopped_ms=WINDOW_MS + 30_000.0)
    assert summary["windows_planned"] == 1 and summary["segments"] == [(0, 1)] and summary["requests"] == 1


def test_lines_cut_by_a_kill_are_dropped_and_merge_checks_the_grid(tmp_path: Path) -> None:
    log = rec.JsonLines(tmp_path / "requests.jsonl")
    log({"rid": "a"})
    log.close()
    with open(tmp_path / "requests.jsonl", "a") as fh:
        fh.write('{"rid": "b", "chu')
    assert rec.read_lines(tmp_path / "requests.jsonl") == [{"rid": "a"}]
    one = rec.trace_doc([{"name": "x", "egress": [], "ingress": []}])
    (tmp_path / "trace.json").write_text(json.dumps(one))
    (tmp_path / "other.json").write_text(json.dumps(one | {"bucket_ms": 50}))
    assert rec.merge([tmp_path, tmp_path / "trace.json"])["configurations"] == one["configurations"] * 2
    with pytest.raises(ValueError, match="grid"):
        rec.merge([tmp_path / "other.json"])


def test_a_repo_id_resolves_to_its_snapshot_with_or_without_refs(tmp_path: Path) -> None:
    repo = tmp_path / "hub" / "models--org--m"
    (repo / "snapshots" / "abc").mkdir(parents=True)
    assert rec.local_model("org/m", tmp_path) == str(repo / "snapshots" / "abc")
    (repo / "snapshots" / "def").mkdir()
    with pytest.raises(RuntimeError, match="2 snapshots"):
        rec.local_model("org/m", tmp_path)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("def\n")
    assert rec.local_model("org/m", tmp_path) == str(repo / "snapshots" / "def")
    assert rec.local_model(str(repo), tmp_path) == str(repo)


class FakeEngine:
    """vLLM's LLMEngine surface the offline loop uses: each step adds one token to every running request, in 20 ms."""

    def __init__(self, now: list[float]):
        self.now, self.running = now, {}

    def add_request(self, rid, prompt, params):
        self.running[rid] = [params["max_tokens"], 0]

    def has_unfinished_requests(self):
        return bool(self.running)

    def step(self):
        self.now[0] += 0.02
        outs = []
        for rid, st in list(self.running.items()):
            st[1] += 1
            outs.append(types.SimpleNamespace(request_id=rid + "-0", finished=st[1] == st[0],
                                              outputs=[types.SimpleNamespace(token_ids=list(range(st[1])))]))
            if st[1] == st[0]:
                del self.running[rid]
        return outs


class FakeCommitter:
    def __init__(self):
        self.calls, self.errors = [], []

    def finalize(self):
        self.calls.append("finalize")
        return types.SimpleNamespace(steps=[types.SimpleNamespace(n_bytes=8)] * 2, run_root=bytes(32),
                                     root_ready_delay_s=0.001, extra={"errors_n": 0, "native_collect": {
                                         "plans_dropped": 4, "plans_pending": 0, "plans_pending_emptied_step": 2, "copies": 9}})

    def __getattr__(self, name):
        return lambda: self.calls.append(name)


def test_the_offline_loop_admits_on_time_stamps_each_step_and_commits_one_run_per_window(monkeypatch) -> None:
    now = [100.0]

    def clock():
        now[0] += 0.001
        return now[0]

    class Event:
        def is_set(self):
            return False

        def set(self):
            pass

        def wait(self, timeout):
            now[0] += timeout

    monkeypatch.setattr(rec, "clock", clock)
    monkeypatch.setattr(rec, "threading", types.SimpleNamespace(Event=Event))
    monkeypatch.setattr(rec.signal, "signal", lambda *_: None)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "verity_vllm.engine.engine_driver",
                        types.SimpleNamespace(external_id=lambda s: s.rsplit("-", 1)[0]))
    eng, com = types.SimpleNamespace(llm=types.SimpleNamespace(llm_engine=FakeEngine(now))), FakeCommitter()
    reqs = [{"rid": f"r{i}", "workload": "b4", "instance": i, "index": 0, "at_s": -5 + 1.5 * i, "prompt": [1, 2, 3],
             "max_tokens": 3} for i in range(90)]
    out, steps, commits = [], [], []
    t0 = now[0] + 5
    ended = rec.offline_replay(eng, com, reqs, t0, t0 + 200, out.append, steps.append, commits.append)
    assert ended == {"stopped_ms": None, "unfinished": 0} and [r["ok"] for r in out] == [True] * 90
    r = next(r for r in out if r["rid"] == "r10")
    assert r["send_ms"] == r["sched_ms"] == 10_000.0 and 0 <= r["admit_ms"] - r["send_ms"] < 25
    assert [n for _, n in r["chunks"]] == [1, 1, 1] and r["chunks"][0][0] > r["admit_ms"]
    assert [c["t_ms"] // 60_000 for c in commits] == [0, 1, 2] and commits[0]["steps"] == 2
    assert {k: v for k, v in commits[0].items() if k.startswith("plans")} == {
        "plans_dropped": 4, "plans_pending": 0, "plans_pending_emptied_step": 2} and "copies" not in commits[0]
    assert com.calls[:6] == ["reset", "install", "finalize", "uninstall", "reset", "install"]
    assert sum(s["tokens"] for s in steps) == 270 and all(s["gc_ms"] >= 0 for s in steps + commits)
    com.errors.append("worker step 7: ImportError('no research')")
    t0 = now[0] + 5
    with pytest.raises(RuntimeError, match="no research"):
        rec.offline_replay(eng, com, [q | {"rid": "again." + q["rid"]} for q in reqs], t0, t0 + 200,
                           out.append, steps.append, commits.append)
    assert not any(isinstance(c, rec.Collections) for c in gc.callbacks)


@pytest.fixture
def thaw():
    """An offline `run` freezes this process's heap: unfreeze it after the test, so later tests' garbage is collected."""
    yield
    gc.unfreeze()


@pytest.mark.parametrize("evict", ["in the replay", "at startup", None])
def test_an_evicted_chunk_converts_what_it_recorded_and_exits_143(tmp_path: Path, monkeypatch, thaw, evict) -> None:
    """A whole offline `run` on the fake engine and the injected clock, with SIGTERM delivered 90 s after t0, while the
    engine loads (seeds 62-69's eviction), or never. The schedule is frozen out of the collector's reach before the
    first request is admitted, and `meta.json` records the freeze."""
    now, handlers, evict_at, first = [100.0], {}, [math.inf], []

    def clock():
        now[0] += 0.001
        return now[0]

    class Event:
        def __init__(self):
            self.flag = False

        def is_set(self):
            return self.flag

        def set(self):
            self.flag = True

        def wait(self, timeout):
            if not self.flag:
                now[0] += timeout

    class Monitor:
        """The spawned monitor: when joined, a quiet sample every SAMPLE_S since it started."""

        def __init__(self, target, args, daemon):
            self.path = args[0]

        def start(self):
            self.began = now[0]

        def join(self, timeout):
            lines, t = rec.JsonLines(self.path), self.began
            while (t := t + rec.SAMPLE_S) <= now[0]:
                lines({"mono": t, "dt_s": rec.SAMPLE_S, "hb_max_ms": 1.0, "loop_max_ms": 1.0, "foreign_pids": [],
                       "others_busy": 0.05, "affinity_busy": 0.1, "load": [1.0, 1.0, 1.0], "own_cores": 0.5,
                       "sm_mhz": 2400})
            lines.close()

    class Evicting(FakeEngine):
        def add_request(self, rid, prompt, params):
            if not first:
                first.append((gc.get_freeze_count(), any(o is prompt["prompt_token_ids"] for o in gc.get_objects())))
            super().add_request(rid, prompt, params)

        def step(self):
            if now[0] >= evict_at[0]:
                evict_at[0] = math.inf
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return super().step()

    def offline_engine(args, commit, meta, mix):
        if evict == "at startup":
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        if evict == "in the replay":
            evict_at[0] = now[0] + args.warmup_s + 90
        return types.SimpleNamespace(llm=types.SimpleNamespace(llm_engine=Evicting(now))), None

    monkeypatch.setattr(rec, "clock", clock)
    monkeypatch.setattr(rec, "threading", types.SimpleNamespace(Event=Event))
    monkeypatch.setattr(rec, "multiprocessing", types.SimpleNamespace(
        get_context=lambda _: types.SimpleNamespace(Event=Event, Process=Monitor)))
    monkeypatch.setattr(rec, "signal", types.SimpleNamespace(
        SIGTERM=signal.SIGTERM, SIG_IGN=signal.SIG_IGN, signal=lambda sig, h: handlers.__setitem__(sig, h)))
    monkeypatch.setattr(rec, "resource", types.SimpleNamespace(
        RLIMIT_NOFILE=0, getrlimit=lambda _: (1024, 1024), setrlimit=lambda *_: None))
    monkeypatch.setattr(rec, "identity", lambda: {"host": "h", "gpu": {"index": 0, "uuid": "GPU-0", "name": "fake"},
                                                  "affinity": list(range(8)), "watched": [], "env": {}})
    monkeypatch.setattr(rec, "local_model", lambda name, _: name)
    monkeypatch.setattr(rec, "offline_engine", offline_engine)
    monkeypatch.setattr(rec, "load_workload",  # 8 s per request: the last outlasts the last window, so it's monitored
                        lambda _: [{"index": 0, "offset_s": 0.0, "prompt": [1, 2, 3], "max_tokens": 400}])
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "verity_vllm.engine.engine_driver",
                        types.SimpleNamespace(external_id=lambda s: s.rsplit("-", 1)[0]))
    frozen = gc.get_freeze_count()  # not 0: CPython starts with objects in the permanent generation
    rc = rec.main(["run", "--out", str(tmp_path), "--model", "unsloth/Llama-3.2-1B", "--mode", "bi-eager",
                   "--mix", "b4=5", "--minutes", "3.5", "--warmup-s", "10"])
    meta = json.loads((tmp_path / "meta.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert rc == (143 if evict else 0) and meta["ended"]["evicted"] is bool(evict)
    if evict == "at startup":
        assert "t0_mono" not in meta and summary["error"] == "the job ended before t0"
        assert "gc" not in meta and first == []
        return
    assert meta["gc"]["frozen_before_t0"] > frozen and first == [(meta["gc"]["frozen_before_t0"], False)]
    assert meta["windows"] == 2 and summary["windows_planned"] == (1 if evict else 2)
    assert summary["windows_kept"] == summary["windows_planned"]
    assert len(json.loads((tmp_path / "trace.json").read_text())["configurations"]) == 1
    if evict:
        assert 90_000 <= meta["ended"]["stopped_ms"] < 96_000


def test_collections_add_up_each_collection_and_take_resets(monkeypatch) -> None:
    times = iter([1.0, 1.25, 2.0, 2.5])
    monkeypatch.setattr(rec, "clock", lambda: next(times))
    c = rec.Collections()
    c("start", {"generation": 2})
    c("stop", {"generation": 2})
    c("start", {"generation": 0})
    assert c.take() == 0.25
    c("stop", {"generation": 0})
    assert c.take() == 0.5 and c.take() == 0.0

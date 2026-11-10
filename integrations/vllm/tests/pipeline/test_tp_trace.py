"""`tp-commit --trace-dir` (`pipeline.tp.trace`): one extra pair under torch.profiler after the timed pairs, outside runs.jsonl -- the
instrumented run installed, profiled, finalized for its roots only and reset --, the per-thread CPU readings around every timed run
(`trace.thread_cpu`) and the per-rank Commit timing every finalize carries (`pipeline.tp.rank_worker._commit_timing`).  CPU-only: stand-in ranks behind an LLM that dispatches `collective_rpc` to them."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from verity_vllm.acquire.committer_api import StepCommitment
from verity_vllm.pipeline.tp.rank_worker import _commit_timing
from verity_vllm.pipeline.tp import trace as TRACE


class _Rank:
    def __init__(self, rank: int) -> None:
        self._tp2_commit = {"rank": rank}
        self.seen: list[tuple] = []

    def tp2_profiler_start(self):
        self.seen.append(("start",))
        return {"rank": self._tp2_commit["rank"]}

    def tp2_profiler_stop(self, *, out_dir, keep_trace=False, keep_ranks=None):
        self.seen.append(("stop", keep_trace, keep_ranks))
        return {"rank": self._tp2_commit["rank"], "n_kernels": 7}

    def tp2_commit_install(self):
        self.seen.append(("install",))

    def tp2_commit_finalize(self, checks=True):
        self.seen.append(("finalize", checks))
        return {"rank": self._tp2_commit["rank"], "run_root": f"{self._tp2_commit['rank']:02x}" * 32}

    def tp2_commit_reset(self):
        self.seen.append(("reset",))


class _LLM:
    def __init__(self, ranks: list[_Rank]) -> None:
        self.ranks = ranks

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        return [method(r, *args) if callable(method) else getattr(r, method)(**(kwargs or {})) for r in self.ranks]


def _once(walls):
    def run():
        w = walls.pop(0)
        return {"wall_s": sum(w), "step_wall_s": w, "step_tokens_out": [1] * len(w)}
    return run


def test_torch_profiles_a_control_and_an_instrumented_run_that_is_finalized_for_its_roots_only(tmp_path):
    llm = _LLM([_Rank(0), _Rank(1)])
    rows = TRACE.run(llm, _once([[0.5, 0.1], [0.7, 0.2]]), tmp_path / "trace", [0], 60.0)
    assert [(r["arm"], r["step_wall_s"]) for r in rows] == [("control", [0.5, 0.1]), ("instrumented", [0.7, 0.2])]
    assert "run_roots" not in rows[0] and rows[1]["run_roots"] == {0: "00" * 32, 1: "01" * 32}
    for x in llm.ranks:
        assert x.seen == [("start",), ("stop", True, [0]), ("install",), ("start",), ("stop", True, [0]), ("finalize", False), ("reset",)]
    traced = [json.loads(ln) for ln in (tmp_path / "trace" / "traced.jsonl").read_text().splitlines()]
    assert [(t["arm"], t["wall_s"]) for t in traced] == [("control", 0.6), ("instrumented", pytest.approx(0.9))]
    assert (tmp_path / "trace" / "torch.control").is_dir() and (tmp_path / "trace" / "torch.instrumented").is_dir()


def test_thread_cpu_reads_this_process_threads_by_python_name_and_the_delta_keeps_what_ran():
    done = threading.Event()

    def spin():
        x = 0
        while not done.is_set():
            x += 1

    t = threading.Thread(target=spin, name="cmt-worker", daemon=True)
    before = [TRACE.thread_cpu(_Rank(3))]
    t.start()
    while TRACE.thread_cpu(_Rank(3))["cpu_s"].get("cmt-worker", 0.0) <= 0.0:
        pass
    after = [TRACE.thread_cpu(_Rank(3))]
    done.set()
    t.join()
    assert after[0]["rank"] == 3 and "MainThread" in after[0]["cpu_s"]
    d = TRACE.cpu_delta(before, after)
    assert d[3]["cmt-worker"] > 0 and all(v > 0 for v in d[3].values())
    assert TRACE.cpu_delta([{"rank": 0, "cpu_s": {"a": 1.0, "b": 2.0}}], [{"rank": 0, "cpu_s": {"a": 1.0, "b": 2.5, "c": 0.25}}]) == {0: {"b": 0.5, "c": 0.25}}


@pytest.mark.parametrize("release", [False, True])
def test_the_gpus_go_back_after_the_last_traced_run_only_when_asked(tmp_path, monkeypatch, release):
    stops = []
    monkeypatch.setattr(TRACE.TPL, "stop", lambda llm, timeout, after: stops.append(after))
    TRACE.run(_LLM([_Rank(0)]), _once([[0.5], [0.6]]), tmp_path, [0], 60.0, release=release)
    assert stops == (["work.trace"] if release else [])


def test_commit_timing_projects_steps_spans_ring_and_source_counters():
    class _Moe:
        short_name = "moe"
        stats = {"calls": 3}

    class _Route:
        short_name = "route"

        def stats(self):
            return {"layers": 2}

    com = SimpleNamespace(sources=[_Moe(), _Route()])
    rc = SimpleNamespace(steps=[StepCommitment(0, b"", b"", 4, 1024, 3, b"", 10.0, 10.25, 1, hashing_s=0.2)],
                         spans={"total_s": {"wait": 0.5}, "per_step": {0: {"wait": 0.5, "worker_idle": 0.1}}},
                         extra={"windows": {"waits": 2, "wait_s": 0.5}, "native_collect": {"copies": 9, "fa2_wrap": {"x": 1}},
                                "gc": {"min_s": 1e-3, "pauses": [(2, 10.1, 0.7, "MainThread")], "total_s": 0.7}})
    t = _commit_timing(com, rc)
    assert t["steps"] == [{"step": 0, "t_end": 10.0, "lag_s": 0.25, "hashing_s": 0.2, "backlog": 1, "bytes": 1024, "tensors": 3}]
    assert t["spans"]["per_step"][0]["worker_idle"] == 0.1 and t["windows"] == {"waits": 2, "wait_s": 0.5}
    assert t["gc"]["pauses"] == [(2, 10.1, 0.7, "MainThread")]
    assert t["collector"] == {"copies": 9} and t["sources"] == {"moe": {"calls": 3}, "route": {"layers": 2}}

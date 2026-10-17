"""`tp-commit --trace-dir` (`pipeline.tp.trace`): one extra pair per profiler after the timed pairs, outside runs.jsonl -- the instrumented
run installed, profiled, finalized for its roots only and reset -- and the per-rank Commit timing every finalize carries
(`engine.rank_worker._commit_timing`).  CPU-only: stand-in ranks behind an LLM that dispatches `collective_rpc` to them."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from verity_vllm.acquire.committer_api import StepCommitment
from verity_vllm.engine.rank_worker import _commit_timing
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


def test_tools_are_named_and_pyspy_needs_its_binary():
    assert TRACE.tools_of("torch", None) == ["torch"]
    for bad in ("", "nsys", "torch,perf"):
        with pytest.raises(SystemExit, match="--trace-tools"):
            TRACE.tools_of(bad, "/opt/py-spy")
    with pytest.raises(SystemExit, match="--trace-pyspy"):
        TRACE.tools_of("torch,pyspy", None)
    assert TRACE.tools_of("torch,pyspy", "/opt/py-spy") == ["torch", "pyspy"]


def test_torch_profiles_a_control_and_an_instrumented_run_that_is_finalized_for_its_roots_only(tmp_path):
    llm = _LLM([_Rank(0), _Rank(1)])
    rows = TRACE.run(llm, _once([[0.5, 0.1], [0.7, 0.2]]), tmp_path / "trace", ["torch"], [0], 60.0)
    assert [(r["tool"], r["arm"], r["step_wall_s"]) for r in rows] == [("torch", "control", [0.5, 0.1]), ("torch", "instrumented", [0.7, 0.2])]
    assert "run_roots" not in rows[0] and rows[1]["run_roots"] == {0: "00" * 32, 1: "01" * 32}
    assert rows[0]["pids"] == rows[1]["pids"] and set(rows[0]["pids"]) == {0, 1}
    for x in llm.ranks:
        assert x.seen == [("start",), ("stop", True, [0]), ("install",), ("start",), ("stop", True, [0]), ("finalize", False), ("reset",)]
    traced = [json.loads(ln) for ln in (tmp_path / "trace" / "traced.jsonl").read_text().splitlines()]
    assert [(t["arm"], t["wall_s"]) for t in traced] == [("control", 0.6), ("instrumented", pytest.approx(0.9))]
    assert (tmp_path / "trace" / "torch.control").is_dir() and (tmp_path / "trace" / "torch.instrumented").is_dir()


def test_pyspy_samples_the_traced_ranks_from_outside_for_a_fixed_time_and_is_waited_for(tmp_path, monkeypatch):
    started = []

    class _Proc:
        def __init__(self, argv, stdin=None, stdout=None, stderr=None):
            self.argv, self.stdin, self.waited = argv, stdin, False
            started.append(self)

        def wait(self):
            self.waited = True
            return 0

    monkeypatch.setattr(TRACE.subprocess, "Popen", _Proc)
    llm = _LLM([_Rank(0), _Rank(1)])
    rows = TRACE.run(llm, _once([[0.5], [0.6]]), tmp_path, ["pyspy"], [1], 60.0, pyspy="/opt/py-spy", pyspy_s=25)
    assert [r["profiler"] for r in rows] == [{1: 0}, {1: 0}] and len(started) == 2
    for p, arm in zip(started, ("control", "instrumented")):
        assert p.argv[:4] == ["sudo", "-n", "/opt/py-spy", "record"] and p.argv[p.argv.index("--pid") + 1] == str(rows[0]["pids"][1])
        assert p.argv[p.argv.index("--duration") + 1] == "25" and p.stdin == TRACE.subprocess.DEVNULL and p.waited
        assert {"--idle", "--threads", "--nonblocking"} <= set(p.argv) and p.argv[-1] == str(tmp_path / f"pyspy.{arm}" / "rank1.speedscope.json")
    assert all(("start",) not in x.seen for x in llm.ranks)


@pytest.mark.parametrize("release", [False, True])
def test_the_gpus_go_back_after_the_last_traced_run_only_when_asked(tmp_path, monkeypatch, release):
    stops = []
    monkeypatch.setattr(TRACE.TPL, "stop", lambda llm, timeout, after: stops.append(after))
    TRACE.run(_LLM([_Rank(0)]), _once([[0.5], [0.6]]), tmp_path, ["torch"], [0], 60.0, release=release)
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

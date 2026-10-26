"""`tp-commit`'s fail-fast: the Commit stops after its first failed run, FAIL, the runs after it and the value check's run not run,
the lease's GPUs back at that run (`lease.stop`), and `summary.json` and `sampled_replay.json` saying which runs ran; `--all-pairs`
runs every one.  CPU-only: `tp.commit.main` over a stand-in LLM of two ranks, the failure a negative's (``--omit-rank 1``: one rank's
commitment missing from the vector fails every instrumented run closed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline.tp import commit as TPCM

from tests.pipeline import served_ranks

WORKLOAD = {"requests": [{"request_id": "q0", "prompt_token_ids": [1, 2, 3], "max_tokens": 4}], "sampling": {"max_tokens": 4}, "sha256": "w",
            "execution": "bi-eager"}
RUN = {"wall_s": 0.5, "iterations": 4, "step_wall_s": [0.2, 0.1, 0.1, 0.1], "step_tokens_out": [1, 1, 1, 1], "generated_tokens": 4, "prompt_tokens": 3, "tokens": {"q0": [5, 6, 7, 8]}}


class _LLM:
    """`vllm.LLM` over two ranks: every `collective_rpc` but the CPU readings around a run goes on ``events`` and answers one dict per rank."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __call__(self, **kw) -> _LLM:
        return self

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if getattr(method, "__name__", None) != "thread_cpu":
            self.events.append(f"rpc.{getattr(method, '__name__', method)}")
        if (served := served_ranks.answer(method, WORKLOAD)) is not None:
            return served
        return [{"rank": r, "leaves": 3, "errors_n": 0, "errors": [], "committer": "native_collect_v2b", "verdict": "PASS", "cpu_s": {}} for r in (0, 1)]


def _commit(tmp_path: Path, monkeypatch, *flags: str) -> tuple[int, list[str], Path]:
    pytest.importorskip("torch")
    from verity_vllm import protocols as PO
    from verity_vllm.engine import engine_profile as prof
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import commit as PC
    from verity_vllm.pipeline import gpu_lease as GL
    from verity_vllm.pipeline.tp import lease as TPL

    events: list[str] = []
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    vllm = ModuleType("vllm")
    vllm.LLM = _LLM(events)
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    def run(eng, requests, sampling):
        events.append("run")
        return dict(RUN)

    for owner, name, value in [(prof, "apply_env", lambda: {}), (prof, "checkpoint_for", lambda case, path: {"repo": "org/model"}),
                               (prof, "engine_kwargs", lambda cp, **kw: dict(kw)), (va, "load_workload", lambda path, max_tokens: WORKLOAD),
                               (va, "target_of_workload", lambda wl, *a: None), (va, "case_of", lambda wl, case: case),
                               (va, "throwaway_requests", lambda eng, reqs, seed=0: reqs), (PO, "commit_guard", lambda *a, **k: None),
                               (PC, "git_head", lambda: None), (PC, "run_workload", run), (TPCM, "tree_of_record", lambda: {}),
                               (GL, "_ENVIRON", []), (GL, "_HELD", []),
                               (TPL, "stop", lambda llm, timeout, after: events.append(f"stop after {after}"))]:
        monkeypatch.setattr(owner, name, value)
    out = tmp_path / "commit"
    opts = cli.parse("tp-commit", ["--workload", str(tmp_path / "w.json"), "--out", str(out), "--case", "C", "--pairs", "3", "--warmup", "1",
                                   "--omit-rank", "1", "--value-check", *flags], environ={})
    return TPCM.main(opts), events[events.index("rpc.tp2_commit_reset") + 1:], out


def test_the_commit_stops_after_its_first_failed_run_and_says_which_runs_it_did_not_run(tmp_path, monkeypatch):
    rc, events, out = _commit(tmp_path, monkeypatch)
    assert rc == 1
    assert events[:6] == ["run", "rpc.tp2_commit_install", "run", "rpc.tp2_commit_finalize", "rpc.tp2_commit_reset",
                          "stop after work.pair0.instrumented"]
    assert events.count("run") == 2 and not any("value_check" in e for e in events)
    summ = json.loads((out / "summary.json").read_text())
    assert summ["pass"] is False and summ["pairs"] == 3 and summ["pairs_run"] == 1 and summ["value_check"] is None
    assert summ["commit_pass"] == [False] and summ["delta_wall"] == [0.0]
    stopped = summ["stopped"]
    assert (stopped["after"], stopped["pair"], stopped["arm"]) == ("work.pair0.instrumented", 0, "instrumented")
    assert stopped["not_run"] == ["pair1.instrumented", "pair1.control", "pair2.control", "pair2.instrumented", "value_check"]
    assert stopped["why"] == "fail-closed: 1/2 ranks finalized"
    assert [(r["pair"], r["arm"]) for r in map(json.loads, (out / "runs.jsonl").read_text().splitlines())] == [(0, "control"), (0, "instrumented")]
    assert not (out / "value_check.json").exists()


@pytest.mark.slow
def test_all_pairs_runs_every_pair_and_the_value_check_after_a_failed_run(tmp_path, monkeypatch):
    rc, events, out = _commit(tmp_path, monkeypatch, "--all-pairs")
    assert rc == 1 and not any(e.startswith("stop after") for e in events)
    assert events.count("run") == 7 and "rpc.tp2_commit_value_check" in events      # 3 pairs x 2 arms, then the value check's run
    summ = json.loads((out / "summary.json").read_text())
    assert summ["pass"] is False and summ["pairs_run"] == 3 and summ["stopped"] is None and summ["commit_pass"] == [False] * 3
    assert (out / "value_check.json").exists()


def test_the_schedule_alternates_the_arms_and_keeps_only_arm():
    assert TPCM.schedule(3, None) == [(0, "control"), (0, "instrumented"), (1, "instrumented"), (1, "control"), (2, "control"), (2, "instrumented")]
    assert TPCM.schedule(2, "instrumented") == [(0, "instrumented"), (1, "instrumented")]

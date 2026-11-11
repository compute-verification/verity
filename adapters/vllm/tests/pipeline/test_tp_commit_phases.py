"""`tp-commit`'s phases on its row's timeline (`pipeline.tp.phases`), which a held-idle reading attributes a TP Commit's lease to:
the engine through its probe, the warm-ups, the committers' setup and every measured run, each span ended however its step ends, and
no `gpu_lease.*` span.  CPU-only: `tp.commit.main` runs over a stand-in LLM of two ranks and a stand-in workload run, which mark the
same timeline as they run, so the file says which step ran inside which span."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import timeline as T
from verity_vllm.pipeline.tp import phases as PH

from tests.pipeline import served_ranks

WORKLOAD = {"requests": [{"request_id": "q0", "prompt_token_ids": [1, 2, 3], "max_tokens": 4}], "sampling": {"max_tokens": 4}, "sha256": "w",
            "execution": "bi-eager"}
RUN = {"wall_s": 0.5, "iterations": 4, "step_wall_s": [0.2, 0.1, 0.1, 0.1], "step_tokens_out": [1, 1, 1, 1], "generated_tokens": 4, "prompt_tokens": 3, "tokens": {"q0": [5, 6, 7, 8]}}


class _LLM:
    """`vllm.LLM` over two ranks: building it and every `collective_rpc` but the CPU readings around a run mark the timeline (``LLM``,
    ``rpc.<method>``); each rpc answers one dict per rank, but the ``n``-th call of the method ``fails`` names raises."""

    def __init__(self, mark: T.Timeline, fails: tuple[str, int] | None = None, ranks: int = 2) -> None:
        self.mark, self.fails, self.ranks, self.calls = mark, fails, ranks, {}

    def __call__(self, **kw) -> _LLM:
        self.mark.event("LLM")
        return self

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        name = getattr(method, "__name__", method)
        if name != "thread_cpu":
            self.mark.event(f"rpc.{name}")
        self.calls[name] = self.calls.get(name, 0) + 1
        if (name, self.calls[name]) == self.fails:
            raise RuntimeError(f"rank 1: {name} failed")
        if (served := served_ranks.answer(method, WORKLOAD, range(self.ranks))) is not None:
            return served
        return [{"rank": r, "leaves": 3, "errors_n": 0, "errors": [], "committer": "native_collect_v2b", "cpu_s": {}} for r in range(self.ranks)]


def _events(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines()]


def _steps(events: list[dict]) -> list:
    """The file in order: a span event as (stage, ev[, ok]), a stand-in's mark as its name."""
    return [e["stage"] if e["proc"] == "stand-in" else (e["stage"], e["ev"], *([e["ok"]] if e["ev"] == "end" else [])) for e in events]


@pytest.fixture
def timeline(tmp_path, monkeypatch) -> Path:
    """The row's timeline as `tp-commit` finds it (``VERITY_TIMELINE``), written through `phases.TL`'s process label and fields."""
    path = tmp_path / "row" / "timeline.jsonl"
    monkeypatch.setenv("VERITY_TIMELINE", str(path))
    assert (PH.TL.proc, PH.TL.defaults) == ("tp_commit", {"hot": False})
    monkeypatch.setattr(PH, "TL", T.Timeline.from_env(PH.TL.proc, **PH.TL.defaults))
    return path


def test_a_tp_commit_writes_each_phase_around_its_gpu_steps_and_ends_the_one_a_failed_step_was_in(timeline, tmp_path, monkeypatch):
    """The first instrumented run's finalize fails on a rank: every span before it ended ok, the run's span ended not ok with the
    error on its way out, and nothing is left open (the reader would run an open span on to `now`)."""
    pytest.importorskip("torch")
    from verity_vllm import protocols as PO
    from verity_vllm.engine import engine_profile as prof
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import commit as PC
    from verity_vllm.pipeline import gpu_lease as GL
    from verity_vllm.pipeline.tp import commit as TPCM

    mark = T.Timeline(timeline, proc="stand-in")
    llm = _LLM(mark, fails=("tp2_commit_finalize", 2))      # the warm-up's finalize, then the first measured one's

    def run(eng, requests, sampling):
        mark.event("run")
        return dict(RUN)

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    vllm = ModuleType("vllm")
    vllm.LLM = llm
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    for owner, name, value in [(prof, "apply_env", lambda: {}), (prof, "checkpoint_for", lambda case, path: {"repo": "org/model"}),
                               (prof, "engine_kwargs", lambda cp, **kw: dict(kw)), (va, "load_workload", lambda path, max_tokens: WORKLOAD),
                               (va, "target_of_workload", lambda wl, *a: None), (va, "case_of", lambda wl, case: case),
                               (va, "throwaway_requests", lambda eng, reqs, seed=0: reqs), (PO, "commit_guard", lambda *a, **k: None),
                               (PC, "git_head", lambda: None), (PC, "run_workload", run), (TPCM, "tree_of_record", lambda: {}),
                               (GL, "_ENVIRON", []), (GL, "_HELD", [])]:
        monkeypatch.setattr(owner, name, value)
    opts = cli.parse("tp-commit", ["--workload", str(tmp_path / "w.json"), "--out", str(tmp_path / "commit"), "--case", "C", "--pairs", "1",
                                   "--warmup", "1"], environ={})
    with pytest.raises(RuntimeError, match="rank 1: tp2_commit_finalize failed"):
        TPCM.main(opts)

    events = _events(timeline)
    assert _steps(events) == [
        ("engine.build", "start"), "LLM", "rpc.tp2_probe", ("engine.build", "end", True), "rpc.read_rank",
        ("prep.warmup_control0", "start"), "run", ("prep.warmup_control0", "end", True),
        ("prep.committer_setup", "start"), "rpc.tp2_commit_make", ("prep.committer_setup", "end", True),
        ("prep.warmup_instrumented", "start"), "rpc.warming", "rpc.tp2_commit_install", "run", "rpc.tp2_commit_finalize", "rpc.warming",
        "rpc.tp2_commit_reset", ("prep.warmup_instrumented", "end", True),
        ("work.pair0.control", "start"), "run", ("work.pair0.control", "end", True),
        ("work.pair0.instrumented", "start"), "rpc.tp2_commit_install", "run", "rpc.tp2_commit_finalize", ("work.pair0.instrumented", "end", False)]
    ours = [e for e in events if e["proc"] == "tp_commit"]
    assert {(e["pid"], e["hot"]) for e in ours} == {(os.getpid(), False)} and T.report(events)["unclosed"] == []
    starts, ends = ({e["stage"]: e for e in ours if e["ev"] == ev} for ev in ("start", "end"))
    assert {k: starts["engine.build"][k] for k in ("tp", "case", "max_num_seqs")} == {"tp": 2, "case": "C", "max_num_seqs": 8}
    assert {k: ends["engine.build"][k] for k in ("model", "enforce_eager", "engine_reused", "restart_reason")} == {
        "model": "org/model", "enforce_eager": True, "engine_reused": False, "restart_reason": "fresh process (one tp-commit per Commit stage)"}
    assert ends["engine.build"]["build_s"] >= 0 and ends["prep.warmup_instrumented"]["wall_s"] == 0.5
    assert starts["prep.committer_setup"]["kind"] == starts["prep.warmup_instrumented"]["kind"] == "native_collect_v2b"
    assert [(starts[s]["pair"], starts[s]["arm"], ends[s]["wall_s"], ends[s]["steps"]) for s in ("work.pair0.control", "work.pair0.instrumented")] == [
        (0, "control", 0.5, 4), (0, "instrumented", 0.5, 4)]
    assert not any(e["stage"].startswith("gpu_lease.") for e in events)


@pytest.mark.parametrize("ranks,build_s", [(1, True), (None, False)], ids=["one-rank-answers-the-probe", "the-engine-does-not-start"])
def test_an_engine_build_that_fails_ends_its_span(timeline, ranks, build_s):
    """Fail closed when fewer ranks than ``tp`` answer the probe (SystemExit) or the engine raises: `engine.build` ended not ok,
    with the build's seconds once the engine was built."""
    llm = _LLM(T.Timeline(timeline, proc="stand-in"), ranks=ranks or 2)

    def build() -> _LLM:
        if ranks is None:
            raise RuntimeError("CUDA out of memory")
        return llm()

    with pytest.raises(SystemExit if ranks else RuntimeError, match="fail-closed: 1 ranks answered the probe, tp=2" if ranks else "out of memory"):
        PH.engine(build, 2, 60.0, model="org/model", enforce_eager=True, case="C")
    events = _events(timeline)
    end = events[-1]
    assert (end["stage"], end["ev"], end["ok"], "build_s" in end) == ("engine.build", "end", False, build_s) and T.report(events)["unclosed"] == []

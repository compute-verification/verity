"""The TP Commit's phase helpers at one rank, where a single-GPU Commit stands: vLLM's one worker runs in the driver's process
(`engine.env` pins ``VLLM_ENABLE_V1_MULTIPROCESSING=0``), so `LLM.collective_rpc` calls it here.  On the single-GPU Commit's own
timeline (`pipeline.commit`'s ``commit_delta``), with its restart reason and its log tag, `tp.phases` writes the spans and lines
`commit.main` writes for its engine and its control warm-ups, the warm-up instrumented run (`tp.warmup`) learns on the one rank, and
the target check (`tp.target`) reads that rank's engine.  CPU-only: the worker is a stand-in."""
from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from tests.pipeline.test_tp2_warmup_learn_only import _Com, _Rank
from verity_vllm.pipeline import timeline as T
from verity_vllm.pipeline.tp import phases as PH
from verity_vllm.pipeline.tp import warmup as WU

#: `commit.main`'s reason for a fresh engine outside a hot worker
FRESH = "fresh process (one commit_delta per Commit stage)"


class _InProcess:
    """`vllm.LLM` under the uniprocess executor: `collective_rpc` runs a callable as ``fn(worker, *args, **kwargs)`` on the one worker,
    a name as the worker's method, and answers a list of one."""

    def __init__(self, worker) -> None:
        self.worker, self.rpcs = worker, []

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.rpcs.append(getattr(method, "__name__", method))
        call = getattr(self.worker, method) if isinstance(method, str) else (lambda *a, **k: method(self.worker, *a, **k))
        return [call(*args, **(kwargs or {}))]


class _Worker(_Rank):
    def tp2_probe(self) -> dict:
        return {"rank": 0, "world": 1}


@pytest.fixture
def commit_tl(tmp_path, monkeypatch) -> T.Timeline:
    monkeypatch.setenv("VERITY_TIMELINE", str(tmp_path / "timeline.jsonl"))
    return T.Timeline.from_env("commit_delta", hot=False)


def test_the_engine_phase_at_one_rank_is_the_single_gpu_commits_engine_build(commit_tl):
    llm = _InProcess(_Worker(0, _Com()))
    built, _, topo = PH.engine(lambda: llm, 1, 60.0, model="org/model", enforce_eager=True, tl=commit_tl, restart_reason=FRESH,
                               case="c", max_num_seqs=4, execution="bi-eager")
    assert built is llm and topo == [{"rank": 0, "world": 1}] and llm.rpcs == ["tp2_probe"]
    start, end = T.read(commit_tl.path)
    assert (start["proc"], start["stage"], start["ev"], start["hot"]) == ("commit_delta", "engine.build", "start", False)
    assert {k: start[k] for k in ("case", "max_num_seqs", "execution", "tp")} == {"case": "c", "max_num_seqs": 4, "execution": "bi-eager", "tp": 1}
    assert {k: end[k] for k in ("model", "enforce_eager", "engine_reused", "restart_reason", "ok")} == {
        "model": "org/model", "enforce_eager": True, "engine_reused": False, "restart_reason": FRESH, "ok": True}
    assert T.report(T.read(commit_tl.path))["engine"]["restart_reasons"] == {FRESH: 1}


def test_an_engine_whose_probe_is_not_answered_by_one_rank_fails_closed_at_one_rank(commit_tl):
    class _Two(_InProcess):
        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            return super().collective_rpc(method, timeout, args, kwargs) * 2

    with pytest.raises(SystemExit, match="2 ranks answered the probe, tp=1"):
        PH.engine(lambda: _Two(_Worker(0, _Com())), 1, 60.0, model="org/model", enforce_eager=True, tl=commit_tl, restart_reason=FRESH)
    assert T.read(commit_tl.path)[-1]["ok"] is False


def test_the_control_warm_ups_at_one_rank_write_the_single_gpu_commits_spans_and_lines(commit_tl, capsys):
    PH.warmups(2, lambda i: {"wall_s": 0.5 + i, "iterations": 4}, tl=commit_tl, tag="delta")
    assert [(e["proc"], e["stage"], e["ev"]) for e in T.read(commit_tl.path)] == [
        ("commit_delta", f"prep.warmup_control{i}", ev) for i in (0, 1) for ev in ("start", "end")]
    assert capsys.readouterr().out.splitlines() == ["[delta] warm-up control 0: 0.500s 4 it", "[delta] warm-up control 1: 1.500s 4 it"]


def test_the_warm_up_instrumented_run_at_one_rank_learns_on_that_rank_and_records_nothing_of_it(tmp_path, monkeypatch):
    monkeypatch.setattr("verity_vllm.pipeline.commit.assert_pristine", lambda eng, base: None)
    rank = _Worker(0, _Com())
    llm = _InProcess(rank)
    assert WU.run(llm, lambda: {"wall_s": 1.0}, tmp_path, "native_collect_v2b", 60.0) == {"wall_s": 1.0}
    assert llm.rpcs == ["warming", "tp2_commit_install", "tp2_commit_finalize", "warming", "tp2_commit_reset"]
    assert rank._tp2_commit["com"].seen == [("reset", True), ("install", True), ("finalize", True), ("uninstall", True), ("reset", False)]
    assert rank.calls == []
    (b,) = json.loads((tmp_path / "bounded_warmup_native_collect_v2b.json").read_text())
    assert (b["rank"], b["learn_only"], b["learned_only_steps"], b["plans_cached"]) == (0, True, [0], 5)


def test_the_target_check_at_one_rank_reads_that_ranks_engine_and_names_it_when_it_differs(monkeypatch):
    pytest.importorskip("torch")
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline.tp import target as TT

    distributed = ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: 0
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    worker, target, model = _Worker(0, _Com()), object(), object()
    worker.vllm_config, worker.get_model = object(), lambda: model
    seen = []
    monkeypatch.setattr(va, "target_mismatches", lambda eng: seen.append(eng) or [])
    assert TT.require_ranks(_InProcess(worker), target, 60.0) == [{"rank": 0, "mismatches": []}]
    (eng,) = seen
    assert eng.target is target and eng.model is model and eng.llm.llm_engine.vllm_config is worker.vllm_config
    monkeypatch.setattr(va, "target_mismatches", lambda eng: ["num_sms declared 132, device 114"])
    with pytest.raises(SystemExit, match="rank 0: num_sms declared 132, device 114"):
        TT.require_ranks(_InProcess(worker), target, 60.0)

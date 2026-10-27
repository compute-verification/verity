"""A TP Commit's warm-up instrumented run is bounded staging's learn-only pass on every rank, as the single-rank Commit's
(`pipeline.tp.warmup`, `acquire.sources.taps.warmup`): learn-only goes on for that run alone, the ranks' finalize binds, attributes,
compares and replays nothing of it, and `bounded_warmup_<kind>.json` records each rank's pass.  Without it the warm-up is a record run:
a learning step over the transient budget raises on every rank ('its plan was not pre-learned').
The record runs that follow finalize with learn-only off and carry their bounded-staging record for the verdict.  CPU-only: the ranks
are `TP2CaptureWorkerExtension`'s commit methods over a stand-in committer, and the LLM dispatches `collective_rpc` to them."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from verity_vllm.acquire.committer_api import StepCommitment
from verity_vllm.pipeline.tp.rank_worker import TP2CaptureWorkerExtension as _cls
from verity_vllm.pipeline.tp import warmup as WU


class _Placeholder:
    placeholder = False


class _Com:
    """A committer that learns a plan over budget at step 0 when learn-only, raises there otherwise; `seen` = (call, learn_only)."""

    def __init__(self, bounded: bool = True) -> None:
        self.bounded, self.learn_only, self.sources, self.seen = bounded, False, [_Placeholder()], []

    def reset(self) -> None:
        self.seen.append(("reset", self.learn_only))

    def install(self) -> None:
        self.seen.append(("install", self.learn_only))

    def uninstall(self) -> None:
        self.seen.append(("uninstall", self.learn_only))

    def finalize(self):
        self.seen.append(("finalize", self.learn_only))
        learned = [0] if self.bounded and self.learn_only else []
        bs = {"enabled": self.bounded, "learn_only": self.learn_only, "learned_only_steps": learned, "learned_only_bytes": (4352 << 20) if learned else 0,
              "transient_peak_bytes": 0, "learn_budget_bytes": 4096 << 20, "plan_mismatches": 0}
        return SimpleNamespace(steps=[StepCommitment(0, b"", b"", 3, 12, 2, b"", 1.0, 1.5, 0)], run_root=(b"" if learned else b"\x01" * 32), root_ready_delay_s=0.0,
                               t_finalize_s=0.0, spans={"total_s": 0.0}, weight_registration=None, persistent_bytes=0, temporary_bytes_peak=0,
                               extra={"bounded_staging": bs, "native_collect": {"plans_cached": 5}, "pinned_pool_bytes": 1 << 20, "errors": []})

    def coverage_declaration(self, rc):
        return {"classes": []}


class _Rank:
    tp2_commit_install = _cls.tp2_commit_install
    tp2_commit_finalize = _cls.tp2_commit_finalize
    tp2_commit_reset = _cls.tp2_commit_reset

    def __init__(self, rank: int, com: _Com) -> None:
        self._tp2_commit = {"com": com, "rank": rank, "eng": None, "base": None, "replay": {"programs": {}}, "match_capture_dir": "/match"}
        self.calls: list[str] = []

    def _tp2_binding(self, rc):
        self.calls.append("binding")
        return {"ok": True}

    def _tp2_attribution(self, rc):
        self.calls.append("attribution")

    def _tp2_sampled_replay(self, rc):
        self.calls.append("sampled_replay")
        return {"sampled_replay": {"result": True}}


class _LLM:
    """`LLM.collective_rpc` over the ranks: a callable runs as `fn(worker, *args)` on each, a name as the worker's method."""

    def __init__(self, ranks: list[_Rank]) -> None:
        self.ranks, self.rpcs = ranks, []

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.rpcs.append((getattr(method, "__name__", method), args))
        return [method(r, *args) if callable(method) else getattr(r, method)(**(kwargs or {})) for r in self.ranks]


@pytest.fixture(autouse=True)
def _pristine(monkeypatch):
    monkeypatch.setattr("verity_vllm.pipeline.commit.assert_pristine", lambda eng, base: None)


def _run(llm: _LLM, tmp_path):
    states = []
    r = WU.run(llm, lambda: states.append([(k.learn_only, k.sources[0].placeholder) for k in (x._tp2_commit["com"] for x in llm.ranks)]) or {"wall_s": 1.0},
               tmp_path, "native_collect_v2b", 60.0)
    return r, states


def test_warming_toggles_the_rank_warm_up_its_committer_and_its_placeholder_sources():
    for com in (_Com(), _Com(bounded=False)):
        worker = SimpleNamespace(_tp2_commit={"com": com, "rank": 1})
        on = WU.warming(worker, True)
        assert on["rank"] == 1 and on["learn_only"] is com.bounded and com.sources[0].placeholder is True and worker._tp2_commit["warmup"] is True
        assert set(on["host_mem_kb"]) == {"rss", "hwm"}
        off = WU.warming(worker, False)
        assert off["learn_only"] is False and com.learn_only is False and com.sources[0].placeholder is False and worker._tp2_commit["warmup"] is False


def test_the_warm_up_is_learn_only_on_every_rank_and_records_nothing_of_it(tmp_path):
    llm = _LLM([_Rank(0, _Com()), _Rank(1, _Com())])
    r, states = _run(llm, tmp_path)
    assert r == {"wall_s": 1.0} and states == [[(True, True), (True, True)]]
    assert llm.rpcs == [("warming", (True,)), ("tp2_commit_install", ()), ("tp2_commit_finalize", ()), ("warming", (False,)), ("tp2_commit_reset", ())]
    for x in llm.ranks:
        com = x._tp2_commit["com"]
        assert com.seen == [("reset", True), ("install", True), ("finalize", True), ("uninstall", True), ("reset", False)]
        assert com.learn_only is False and com.sources[0].placeholder is False and x.calls == []
    fin = json.loads((tmp_path / "finalize_warmup.json").read_text())
    assert [(f["rank"], f["binding"], f["attribution"], f["match_oracle"], f["sampled_replay"]) for f in fin] == [(0, None, None, None, None), (1, None, None, None, None)]
    bw = json.loads((tmp_path / "bounded_warmup_native_collect_v2b.json").read_text())
    assert [(b["rank"], b["learn_only"], b["learned_only_steps"], b["plans_cached"], b["pinned_pool_bytes"]) for b in bw] == [(0, True, [0], 5, 1 << 20), (1, True, [0], 5, 1 << 20)]
    assert all(set(b["host_mem_kb"]) == {"rss", "hwm"} for b in bw)


def test_the_record_run_after_the_warm_up_is_not_learn_only_and_is_bound_and_replayed(tmp_path):
    llm = _LLM([_Rank(0, _Com()), _Rank(1, _Com())])
    _run(llm, tmp_path)
    llm.collective_rpc("tp2_commit_install")
    rec = llm.collective_rpc("tp2_commit_finalize")
    for x, f in zip(llm.ranks, rec):
        assert x._tp2_commit["com"].seen[-3:] == [("install", False), ("finalize", False), ("uninstall", False)]
        assert x.calls == ["binding", "attribution", "sampled_replay"] and f["binding"] == {"ok": True} and f["sampled_replay"] == {"sampled_replay": {"result": True}}
        assert f["match_oracle"]["verdict"] == "FAIL" and f["bounded_staging"]["learn_only"] is False and f["bounded_staging"]["learned_only_steps"] == []


def test_an_unbounded_committer_warms_up_without_learn_only_and_records_nothing_of_it(tmp_path):
    llm = _LLM([_Rank(0, _Com(bounded=False)), _Rank(1, _Com(bounded=False))])
    _, states = _run(llm, tmp_path)
    assert states == [[(False, True), (False, True)]] and all(x.calls == [] for x in llm.ranks)
    assert not (tmp_path / "bounded_warmup_native_collect_v2b.json").exists() and (tmp_path / "finalize_warmup.json").exists()

"""`verity_vllm.pipeline.tp.rank_worker`: what a rank answers when its committed run ends
(`TP2CaptureWorkerExtension.tp2_commit_finalize`, with `_commit_timing`).  The rank fixes its run root once, takes its
committer off the engine and answers the root with the run's counts, its coverage declaration and where the Commit spent its
time.  A run of record is also bound, attributed, compared with the Match capture and replayed; a warm-up is none of these.
A field the committer did not state reads as none: no errors, no collector.  CPU-only: the committer is the library's host
committer over `tests.check.opened`'s steps, wrapped so `finalize` hands the commitment the test made, and the rank's
binding, attribution and replay are stand-ins that say what they were given."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from tests.check.opened import commit_host_steps
from verity_vllm.check.commit_rules import class_coverage
from verity_vllm.pipeline.commit import baseline_state
from verity_vllm.pipeline.tp import rank_worker as RW
from verity_vllm.pipeline.tp.rank_worker import TP2CaptureWorkerExtension

STEPS = {0: [("m.linear/out", b"a" * 8), ("m.norm/out", b"b" * 8)], 1: [("m.linear/out", b"c" * 8)]}
ANSWER = ["attribution", "binding", "bounded_staging", "bytes_bound", "classes", "collector_failure", "committed_tensors", "declaration",
          "errors", "errors_n", "expected_manifest", "expected_tensors", "finalize_s", "identity_check", "leaves", "leaves_per_step",
          "match_oracle", "persistent_bytes", "pinned_pool_bytes", "plans_cached", "rank", "root_ready_delay_s", "run_root", "sampled_replay",
          "spans_total_s", "steps", "temporary_bytes_peak", "tensors", "timing", "weight_registration", "weights_pin"]
ABSENT: Any = object()


class _Finalized:
    """The rank's committer at the end of a run: `finalize` hands the run's commitment, and every read goes to the host
    committer that made it."""

    def __init__(self, com: Any, rc: Any) -> None:
        self._com, self._rc = com, rc
        self.calls: list[str] = []

    def finalize(self) -> Any:
        self.calls.append("finalize")
        return self._rc

    def uninstall(self) -> None:
        self.calls.append("uninstall")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._com, name)


class _Rank(TP2CaptureWorkerExtension):
    """A rank whose checks of a run of record say what they were given."""

    _tp2_commit: dict[str, Any]

    def _tp2_binding(self, rc: Any) -> dict[str, Any]:
        return {"bound": rc.run_root.hex()}

    def _tp2_attribution(self, rc: Any) -> dict[str, Any]:
        return {"attributed": rc.run_root.hex()}

    def _tp2_sampled_replay(self, rc: Any) -> dict[str, Any]:
        return {"replayed": rc.run_root.hex()}

    def _deferred_replay(self, rc: Any, defer: dict[str, Any]) -> dict[str, Any]:
        return {"deferred": dict(defer)}


def _rank(extra: dict[str, Any] | None = None, **state: Any) -> tuple[_Rank, _Finalized, Any]:
    """(a rank whose run committed `STEPS`, its committer, the run's commitment); ``extra`` sets fields of what the committer
    states beside the root (`ABSENT`: the field is not stated), ``state`` fields of the rank's Commit state."""
    com, rc = commit_host_steps(STEPS)
    for k, v in (extra or {}).items():
        if v is ABSENT:
            rc.extra.pop(k, None)
        else:
            rc.extra[k] = v
    eng: Any = SimpleNamespace(runner=SimpleNamespace(), model=torch.nn.Linear(2, 2))
    w = _Rank()
    w._tp2_commit = {"rank": 1, "com": _Finalized(com, rc), "eng": eng, "base": baseline_state(eng), "rc": None, **state}
    return w, w._tp2_commit["com"], rc


def test_a_warm_up_answers_its_root_counts_coverage_and_timing_and_is_not_checked(tmp_path):
    w, com, rc = _rank(warmup=True, replay={"programs": {}}, match_capture_dir=str(tmp_path), weights_pin={"checkpoint_dir": "/ckpt"})
    res = w.tp2_commit_finalize()
    assert sorted(res) == ANSWER and com.calls == ["finalize", "uninstall"] and w._tp2_commit["rc"] is rc
    assert (res["rank"], res["run_root"], res["weights_pin"]) == (1, rc.run_root.hex(), {"checkpoint_dir": "/ckpt"})
    assert (res["binding"], res["attribution"], res["match_oracle"], res["sampled_replay"]) == (None, None, None, None)
    assert (res["steps"], res["leaves_per_step"], res["leaves"], res["bytes_bound"], res["tensors"]) == (2, [1, 1], 2, 24, 3)
    assert (res["root_ready_delay_s"], res["finalize_s"], res["spans_total_s"]) == (rc.root_ready_delay_s, rc.t_finalize_s, rc.spans["total_s"])
    assert (res["errors_n"], res["errors"], res["collector_failure"], res["plans_cached"]) == (0, [], None, None)
    assert [res[k] for k in ("expected_tensors", "committed_tensors", "identity_check", "pinned_pool_bytes")] == \
        [rc.extra[k] for k in ("expected_tensors", "committed_tensors", "identity_check", "pinned_pool_bytes")]
    assert res["expected_manifest"] == com.expected_manifest() and res["weight_registration"] == rc.weight_registration
    assert (res["persistent_bytes"], res["temporary_bytes_peak"]) == (rc.persistent_bytes, rc.temporary_bytes_peak)
    declared = RW._jsonable(w._tp2_commit["decl"])
    assert res["declaration"] == declared and res["classes"] == class_coverage(declared) and "instance_outputs_native" in res["classes"]
    timing = res["timing"]
    assert sorted(timing) == ["collector", "gc", "sources", "spans", "steps", "windows"]
    assert timing["steps"] == [{"step": s.step, "t_end": s.t_step_end, "lag_s": s.t_root_ready - s.t_step_end, "hashing_s": s.hashing_s,
                                "backlog": s.backlog_at_step_end, "bytes": s.n_bytes, "tensors": s.n_tensors} for s in rc.steps]
    assert [t["bytes"] for t in timing["steps"]] == [16, 8] and timing["spans"] == RW._jsonable(rc.spans)
    assert (timing["windows"], timing["gc"], timing["collector"], timing["sources"]) == (None, rc.extra["gc"], {}, {})


def test_a_run_of_record_is_bound_and_attributed_and_replayed_only_with_a_replay():
    w, _, rc = _rank()
    res = w.tp2_commit_finalize()
    root = rc.run_root.hex()
    assert (res["binding"], res["attribution"], res["sampled_replay"], res["match_oracle"]) == ({"bound": root}, {"attributed": root}, None, None)
    w, _, rc = _rank(replay={"programs": {}})
    assert w.tp2_commit_finalize()["sampled_replay"] == {"replayed": rc.run_root.hex()}
    w, _, _ = _rank(replay={"programs": {}})
    assert w.tp2_commit_finalize(defer={"out": "/o", "pair": 2})["sampled_replay"] == {"deferred": {"out": "/o", "pair": 2}}
    w, _, _ = _rank()
    assert w.tp2_commit_finalize(defer={"out": "/o", "pair": 2})["sampled_replay"] is None    # no replay configured: nothing deferred


def test_a_replay_that_takes_turns_says_its_slot(tmp_path):
    w, _, rc = _rank(replay={"programs": {}, "ranks_at_once": 2, "slot_dir": str(tmp_path)})
    replay = w.tp2_commit_finalize()["sampled_replay"]
    slot = replay.pop("host_slot")
    assert replay == {"replayed": rc.run_root.hex()} and (slot["slot"], slot["ranks_at_once"]) == (1, 2)
    assert [p.name for p in tmp_path.iterdir()] == ["replay_slot1.lock"]


def test_the_root_alone_is_answered_without_checks_and_the_run_is_finalized_once():
    w, com, rc = _rank()
    assert w.tp2_commit_finalize(checks=False) == {"rank": 1, "run_root": rc.run_root.hex()}
    assert com.calls == ["finalize", "uninstall"] and "decl" not in w._tp2_commit
    assert w.tp2_commit_finalize()["binding"] == {"bound": rc.run_root.hex()}
    assert com.calls == ["finalize", "uninstall"]                                   # the leased Commit's second call checks the same root


def test_a_hook_left_on_the_model_fails_the_finalize():
    w, com, _ = _rank()
    w._tp2_commit["eng"].model.register_forward_hook(lambda m, a, o: None)
    with pytest.raises(AssertionError, match="forward hooks present: 1 != baseline 0"):
        w.tp2_commit_finalize()
    assert com.calls == ["finalize", "uninstall"]


TWELVE = [f"e{i}" for i in range(12)]


@pytest.mark.parametrize("extra, errors_n, errors", [
    ({"errors": ABSENT, "errors_n": ABSENT}, 0, []), ({"errors": None, "errors_n": ABSENT}, 0, []), ({"errors": [], "errors_n": ABSENT}, 0, []),
    ({"errors": TWELVE, "errors_n": ABSENT}, 12, TWELVE[:10]), ({"errors": TWELVE, "errors_n": 40}, 40, TWELVE[:10]),
    ({"errors": ABSENT, "errors_n": 3}, 3, []), ({"errors": None, "errors_n": 3}, 3, []),
], ids=["neither stated", "errors none", "no error", "twelve, uncounted", "twelve, counted 40", "counted, not listed", "counted, listed none"])
def test_the_answer_counts_the_committers_errors_and_lists_the_first_ten(extra, errors_n, errors):
    w, _, _ = _rank(extra)
    res = w.tp2_commit_finalize()
    assert (res["errors_n"], res["errors"]) == (errors_n, errors)


@pytest.mark.parametrize("stated, plans, collector", [
    (ABSENT, None, {}), (None, None, {}), ({}, None, {}), ({"fa2_wrap": {"calls": 9}}, None, {}),
    ({"plans_cached": 7, "fa2_wrap": {"calls": 9}, "bytes": 5}, 7, {"plans_cached": 7, "bytes": 5}), ({"plans_cached": 0}, 0, {"plans_cached": 0}),
], ids=["not stated", "none", "empty", "the wrap alone", "plans, the wrap and bytes", "no plan cached"])
def test_the_collectors_counters_are_answered_without_its_wrap_and_none_when_it_states_none(stated, plans, collector):
    w, _, _ = _rank({"native_collect": stated})
    res = w.tp2_commit_finalize()
    assert (res["plans_cached"], res["timing"]["collector"]) == (plans, collector)


class _Counted:
    """A source that states its counters by a method, under its short name."""
    short_name = "counted"

    def stats(self) -> dict[str, int]:
        return {"calls": 4}


class _Held:
    """A source that holds its counters, and has no short name."""
    stats = {"bytes": 16}


class _Silent:
    """A source with no counters."""
    short_name = "silent"


def test_the_timing_names_each_sources_counters():
    _, _, rc = _rank()
    timing = RW._commit_timing(SimpleNamespace(sources=[_Counted(), _Held(), _Silent()]), rc)
    assert timing["sources"] == {"counted": {"calls": 4}, "_Held": {"bytes": 16}, "silent": None} and len(timing["steps"]) == 2
    assert RW._commit_timing(SimpleNamespace(), rc)["sources"] == {}                 # a committer that lists no source


class _Partials:
    """The rank's TP partial source at the compare with the Match capture: it says ``said`` (an exception: raises it)."""

    def __init__(self, said: Any) -> None:
        self.said = said
        self.read: list[tuple[str, Any]] = []

    def compare_match_oracle(self, capture_dir: str, reader: Any) -> Any:
        self.read.append((capture_dir, reader))
        if isinstance(self.said, Exception):
            raise self.said
        return dict(self.said)


def test_a_run_of_record_is_compared_with_the_match_capture_through_the_committed_bytes(tmp_path):
    tps = _Partials({"rank": 1, "verdict": "PASS"})
    w, com, rc = _rank(match_capture_dir=str(tmp_path), tps=tps, schedule_replay={"armed": True})
    assert w.tp2_commit_finalize()["match_oracle"] == {"rank": 1, "verdict": "PASS", "schedule_replay": {"armed": True}}
    ((capture_dir, reader),) = tps.read
    assert capture_dir == str(tmp_path) and (reader.com, reader.run) == (com, rc)


@pytest.mark.parametrize("tps, said", [
    (None, {"why": "no tp partial source on this rank (partials not bound)"}),
    (_Partials(RuntimeError("x" * 400)), {"error": repr(RuntimeError("x" * 400))[:300]}),
], ids=["no partial source", "the compare raises"])
def test_a_compare_that_cannot_run_fails_the_rank(tmp_path: Path, tps, said):
    w, _, _ = _rank(match_capture_dir=str(tmp_path), tps=tps)
    assert w.tp2_commit_finalize()["match_oracle"] == {"rank": 1, "verdict": "FAIL", **said, "schedule_replay": None}

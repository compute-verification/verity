"""`verity_vllm.pipeline.tp.rank_worker`: a rank's verdict of its value-check run (`TP2CaptureWorkerExtension.tp2_commit_value_check`).
The rank finalizes, compares the committed bytes with the module checker's references and with the TP partial source's, runs the
late-read check, and leaves the engine as it found it.  PASS needs both compares to pass.  The late-read result is the
detector's, made False by a committed `/out` that differs from its bind-time copy, and, where the detector passed, by a
compare that counted no `/out` at all.  CPU-only: the bytes are committed by the host committer (`tests.check.opened`) and read
through the opened reader; the two compares are stand-ins that say what a test gives them."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from tests.check.opened import commit_host_steps
from verity_vllm.pipeline.commit import baseline_state
from verity_vllm.pipeline.tp.rank_worker import TP2CaptureWorkerExtension
from verity_vllm.records.schema_ids import LATE_READ_CHECK_SCHEMA

OUT, NORM = "model.layers.0.mlp.experts/out", "model.norm/0"
#: what a rank committed, by the late-read detector's answer for it: passes, does not apply (no `/out` member), finds a late read
MOE = {0: [(OUT, b"a" * 8), (NORM, b"d" * 8)]}
DENSE = {0: [("model.layers.0.self_attn.o_proj/0", b"o" * 8), (NORM, b"d" * 8)]}
LATE = {0: [(OUT, b"z" * 8), (NORM, b"z" * 8)]}


class _Finalized:
    """The rank's committer at the end of the value-check run: `finalize` hands the run's commitment, and every read goes to the
    host committer that made it."""

    def __init__(self, com: Any, rc: Any) -> None:
        self._com, self._rc = com, rc
        self.calls: list[str] = []

    def finalize(self) -> Any:
        self.calls.append("finalize")
        return self._rc

    def uninstall(self) -> None:
        self.calls.append("uninstall")

    def reset(self) -> None:
        self.calls.append("reset")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._com, name)


class _Compare:
    """A checker whose compare says ``said`` (an exception: raises it)."""

    def __init__(self, said: Any) -> None:
        self.said, self.reference, self.uninstalled = said, True, 0
        self.read: list[Any] = []

    def compare(self, reader: Any) -> Any:
        self.read.append(reader)
        if isinstance(self.said, Exception):
            raise self.said
        return self.said

    def uninstall(self) -> None:
        self.uninstalled += 1


class _Rank(TP2CaptureWorkerExtension):
    _tp2_commit: dict[str, Any]


def _partials(out: Any = ..., **more: Any) -> dict[str, Any]:
    """A passing TP partials compare; ``out``: its count for the `/out` member."""
    return {"verdict": "PASS", **({} if out is ... else {"by_member": {"out": out}}), **more}


def _rank(committed: dict, *, modules: Any = ..., partials: Any = ...) -> tuple[_Rank, _Finalized, Any, Any]:
    """(the rank, its committer, its module checker, its TP partial source) after a value-check run that committed ``committed``;
    ``modules`` / ``partials``: what each compare says (None: the rank has no such checker)."""
    com = _Finalized(*commit_host_steps(committed))
    vc = None if modules is None else _Compare({"verdict": "PASS"} if modules is ... else modules)
    tps = None if partials is None else _Compare(_partials({"tensors": 1, "tap_equal": 1}) if partials is ... else partials)
    eng: Any = SimpleNamespace(runner=SimpleNamespace(), model=torch.nn.Linear(2, 2))
    w = _Rank()
    w._tp2_commit = {"rank": 1, "com": com, "tps": tps, "eng": eng, "base": baseline_state(eng), "rc": "the run before"}
    if vc is not None:
        w._tp2_commit["vc"] = vc
    return w, com, vc, tps


def test_a_rank_passes_when_both_compares_pass_and_leaves_the_engine_as_it_found_it():
    w, com, vc, tps = _rank(MOE)
    res = w.tp2_commit_value_check()
    assert (res["verdict"], res["modules"], res["rank"]) == ("PASS", {"verdict": "PASS"}, 1) and res["run_root"] == com._rc.run_root.hex()
    assert res["tp_partials"] == _partials({"tensors": 1, "tap_equal": 1})
    assert res["t6_4"]["schema"] == LATE_READ_CHECK_SCHEMA and res["t6_4"]["applicable"] is True and res["t6_4"]["result"] is True
    assert (res["t6_4"]["out_tensors_vs_clone"], res["t6_4"]["clone_mismatches"]) == (1, 0)
    assert "oracle-compare" in res["method"] and "not the Match fold" in res["oracle"] and json.loads(json.dumps(res)) == res
    # each compare read the committed bytes through a reader of this run; then the checkers come off and the committer is emptied
    assert len(vc.read) == len(tps.read) == 1 and com.calls == ["finalize", "uninstall", "reset"]
    assert vc.uninstalled == 1 and tps.reference is False and w._tp2_commit["rc"] is None and "vc" not in w._tp2_commit


@pytest.mark.parametrize("committed, partials, result, tensors, mismatches", [
    (MOE, _partials({"tensors": 2, "tap_equal": 2}), True, 2, 0),
    (MOE, _partials({"tensors": 2, "tap_equal": 1}), False, 2, 1),
    (MOE, _partials({"tensors": 2}), False, 2, 2),
    (MOE, _partials({"tensors": None, "tap_equal": None}), True, 0, 0),
    (MOE, _partials({}), False, 0, None),
    (MOE, _partials(None), False, 0, None),
    (MOE, _partials(), False, 0, None),
    (MOE, _partials(by_member=None), False, 0, None),
    (MOE, _partials(by_member={}), False, 0, None),
    (DENSE, _partials({"tensors": 1, "tap_equal": 0}), False, 1, 1),
    (DENSE, _partials({"tensors": 1, "tap_equal": 1}), None, 1, 0),
    (DENSE, _partials(), None, 0, None),
    (LATE, _partials({"tensors": 1, "tap_equal": 1}), False, 1, 0),
], ids=["every /out equals its copy", "one /out differs from its copy", "no count of equal copies", "counts that are None",
        "an empty count for /out", "no count for /out", "no member counts", "member counts that are None", "empty member counts",
        "no late read to detect, and an /out differs", "no late read to detect, every /out equal", "no late read to detect, no /out",
        "a late read"])
def test_the_late_read_result_is_the_detectors_unless_an_out_differs_from_its_copy_or_a_pass_counted_none(
        committed, partials, result, tensors, mismatches):
    w, *_ = _rank(committed, partials=partials)
    res = w.tp2_commit_value_check()
    t = res["t6_4"]
    assert (t["result"], t["out_tensors_vs_clone"], t["clone_mismatches"]) == (result, tensors, mismatches), t
    assert t["schema"] == LATE_READ_CHECK_SCHEMA and "clone_rule" in t and res["verdict"] == "PASS"   # the verdict is the two compares'


@pytest.mark.parametrize("modules, partials, said", [
    ({"verdict": "FAIL", "tap_mismatch": ["m"]}, ..., ("FAIL", "PASS")),
    (..., _partials({"tensors": 1, "tap_equal": 1}) | {"verdict": "FAIL"}, ("PASS", "FAIL")),
    (None, ..., ("NOT_RUN", "PASS")),
    (..., None, ("PASS", "NOT_RUN")),
    (..., {}, ("PASS", None)),
], ids=["the modules differ", "the partials differ", "no module checker", "no partial source", "a partials compare that says nothing"])
def test_a_rank_fails_unless_both_compares_ran_and_passed(modules, partials, said):
    w, com, vc, tps = _rank(MOE, modules=modules, partials=partials)
    res = w.tp2_commit_value_check()
    assert res["verdict"] == "FAIL" and (res["modules"].get("verdict"), res["tp_partials"].get("verdict")) == said
    if partials is None:
        assert res["tp_partials"] == {"verdict": "NOT_RUN", "why": "no tp partial source"}
    if partials is None or partials == {}:                                      # no `/out` was counted, so the detector's pass does not stand
        assert (res["t6_4"]["result"], res["t6_4"]["out_tensors_vs_clone"], res["t6_4"]["clone_mismatches"]) == (False, 0, None)
    assert com.calls == ["finalize", "uninstall", "reset"] and (vc is None or vc.uninstalled == 1) and (tps is None or tps.reference is False)


def test_a_compare_that_raises_fails_its_part_by_the_errors_first_300_characters_and_the_rank_still_cleans_up():
    long = ValueError("x" * 400)
    w, com, vc, tps = _rank(MOE, modules=long, partials=RuntimeError("the partials could not be opened"))
    res = w.tp2_commit_value_check()
    assert res["modules"] == {"verdict": "FAIL", "error": repr(long)[:300]} and len(res["modules"]["error"]) == 300
    assert res["tp_partials"] == {"verdict": "FAIL", "error": repr(RuntimeError("the partials could not be opened"))}
    assert res["verdict"] == "FAIL" and res["t6_4"]["result"] is False and res["t6_4"]["clone_mismatches"] is None
    assert com.calls == ["finalize", "uninstall", "reset"] and vc.uninstalled == 1 and tps.reference is False


def test_a_late_read_check_that_raises_is_a_failed_check_of_the_same_schema_with_the_error():
    w, *_ = _rank(MOE, partials=_partials({"tensors": "many"}))
    res = w.tp2_commit_value_check()
    assert res["t6_4"] == {"schema": LATE_READ_CHECK_SCHEMA, "result": False, "error": repr(ValueError("invalid literal for int() with base 10: 'many'"))}
    assert res["verdict"] == "PASS"


def test_a_hook_left_on_the_model_after_the_check_is_an_error_and_no_verdict():
    w, com, *_ = _rank(MOE)
    w._tp2_commit["eng"].model.register_forward_hook(lambda module, args, output: None)
    with pytest.raises(AssertionError, match="forward hooks present: 1 != baseline 0"):
        w.tp2_commit_value_check()
    assert com.calls == ["finalize", "uninstall", "reset"]

"""`verity_vllm.check.replay.served_check.population` over the notes of single steps: what the reconciliation reads of a
step's notes, and what it needs each to state.

The served check's own tests reconcile the reports of whole served runs; these hold the cases a run of theirs does not
reach: a step the Build plans nothing for, and notes that are not the check's.
"""
from __future__ import annotations

import pytest

from verity_vllm.check.replay import served_check as SC
from verity_vllm.query.manifest.format import SERVED_FAMILY, SERVED_IDS

H = 4


def _build(identities=(), steps_plan=()):
    program = {"values": {SERVED_IDS: {"width": 1}, "h": {"width": H}}, "logits": {"gemm": [H, 8]}, "calls": []}
    return {"program": program, "identities": list(identities), "steps_plan": list(steps_plan), "kind": "Q_served",
            "manifest_digest": "m", "program_digest": "p"}


def _identity(step, request, op_path, member, rows=1, width=1):
    return {"family": SERVED_FAMILY, "engine_step": step, "request_id": request, "op_path": op_path, "output_member": member,
            "element_range": [0, rows * width], "rank": 0}


def test_a_step_the_build_plans_nothing_for_notes_no_rows_and_none_of_it_is_reached():
    build = _build([_identity(7, "r0", "c00.input", "ids", rows=2)])
    c = SC.ServedCheck(build, {}.__getitem__, seed=None)
    c.step(SC.Step(7, [("r0", 0, 2)], {}))
    rep = c.report()
    assert rep["steps"]["7"] == {"batch": {"ok": False, "why": "engine step 7 is in no step plan of the Build"}}
    checks = SC.population(build, rep, ranks=[0])["query_checks"]
    assert not checks["ok"] and checks["accounted"]
    assert checks["identities_without_rows"] == {"n": 1, "first": [{"request_id": "r0", "step": 7, "op_path": "c00.input", "member": "ids"}]}
    assert checks["vus_outside_query"]["n"] == 0 and checks["rows_disagree"]["n"] == 0


def test_the_ids_are_reached_for_the_requests_their_note_names_and_no_other():
    build = _build([_identity(0, "r0", "c00.input", "ids", rows=2), _identity(0, "r1", "c00.input", "ids", rows=3)])
    rep = {"steps": {"0": {"batch": {"ok": True, "rows": {"r0": 2, "r1": 3}}, SERVED_IDS: {"ok": True, "requests": ["r0"]}}}}
    checks = SC.population(build, rep, ranks=[0])["query_checks"]
    assert checks["identities_without_rows"]["first"] == [{"request_id": "r1", "step": 0, "op_path": "c00.input", "member": "ids"}]
    assert checks["rows_disagree"]["n"] == 0 and checks["accounted"]


@pytest.mark.parametrize("notes, missing", [({}, "batch"),
                                            ({"batch": {"ok": True, "rows": {"r0": 2}}, SERVED_IDS: {"ok": True}}, "requests")])
def test_notes_that_do_not_state_what_the_check_always_writes_are_no_report_of_it(notes, missing):
    with pytest.raises(KeyError, match=missing):
        SC.population(_build(), {"steps": {"0": notes}}, ranks=[0])

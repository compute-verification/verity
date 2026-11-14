"""`verity_vllm.check.replay.served_check.population` over the notes of single steps: what the reconciliation reads of a
step's notes, and what it answers where they leave a note out.

The served check's own tests reconcile the reports of whole served runs; these hold the cases a run of theirs does not
reach: a step the Build plans nothing for, notes that are not the check's, and more findings of a kind than it names.
Last, on that run itself (`tests.check.test_served_check`): the rows a step's attention note counts as not finite.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.check import test_served_check as SVT
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


ABSENT: Any = object()


def _checks(build: dict, notes: dict) -> dict:
    return SC.population(build, {"steps": {"0": notes}}, ranks=[0])["query_checks"]


def test_each_finding_names_its_first_six_and_counts_them_all():
    """Seven identities of one step's ids, each stating three rows.  Under notes that reach none, seven are without rows; under
    notes whose batch gives each request two rows, seven disagree; under those notes and a Build that holds no identity, seven
    noted values are outside the query.  `NAMED` of each are named."""
    requests = [f"r{i}" for i in range(7)]
    build = _build([_identity(0, r, "c00.input", "ids", rows=3) for r in requests])
    notes = {"batch": {"ok": True, "rows": dict.fromkeys(requests, 2)}, SERVED_IDS: {"ok": True, "requests": requests}}
    at = [{"request_id": r, "step": 0, "op_path": "c00.input", "member": "ids"} for r in requests]
    without, disagree = _checks(build, {})["identities_without_rows"], _checks(build, notes)["rows_disagree"]
    assert without == {"n": 7, "first": at[:6]}
    assert disagree == {"n": 7, "first": [a | {"manifest_height": 3.0, "program_rows": 2} for a in at[:6]]}
    outside = _checks(_build(), notes)["vus_outside_query"]
    assert (outside["n"], len(outside["first"])) == (7, 6)


@pytest.mark.parametrize("batch", [ABSENT, None, {}, {"ok": True}, {"ok": True, "rows": None}, {"ok": True, "rows": {}}],
                         ids=["absent", "null", "empty", "no rows", "rows null", "rows empty"])
def test_a_step_whose_notes_state_no_rows_reaches_no_value_of_a_batch(batch):
    """The check writes the batch note at every step; notes without it, or without its rows, are a step with no request."""
    build = _build([_identity(0, "r0", "h", "out", width=H)])
    checks = _checks(build, ({} if batch is ABSENT else {"batch": batch}) | {"h": {"ok": True}})
    assert checks["identities_without_rows"] == {"n": 1, "first": [{"request_id": "r0", "step": 0, "op_path": "h", "member": "out"}]}
    assert (checks["vus_outside_query"]["n"], checks["rows_disagree"]["n"], checks["accounted"], checks["ok"]) == (0, 0, True, False)
    reached = _checks(build, {"batch": {"ok": True, "rows": {"r0": 1}}, "h": {"ok": True}})
    assert reached["identities_without_rows"]["n"] == 0 and reached["ok"]


@pytest.mark.parametrize("requests", [ABSENT, None, []], ids=["absent", "null", "empty"])
def test_an_ids_note_that_names_no_request_reaches_no_request_s_ids(requests):
    build = _build([_identity(0, "r0", "c00.input", "ids", rows=2)])
    note = {"ok": True} | ({} if requests is ABSENT else {"requests": requests})
    checks = _checks(build, {"batch": {"ok": True, "rows": {"r0": 2}}, SERVED_IDS: note})
    assert checks["identities_without_rows"] == {"n": 1, "first": [{"request_id": "r0", "step": 0, "op_path": "c00.input", "member": "ids"}]}
    assert (checks["vus_outside_query"]["n"], checks["rows_disagree"]["n"], checks["accounted"], checks["ok"]) == (0, 0, True, False)


def test_an_attention_note_counts_the_rows_whose_output_is_not_finite():
    """The served run's attention outputs are finite at every step, and each step's note counts none.  One query word made an
    infinity at the last step gives one row that is not finite: the note counts it, and counts it once among the wrong words,
    beside every word of the head that read it."""
    build, weights = SVT._build(), SVT._weights()
    steps, prompts = SVT._serve(build, weights)

    def notes(served: list[SC.Step]) -> dict[str, tuple]:
        rep = SC.check(build, served, weights.__getitem__, seed=None, rows=8, prompts=prompts)
        return {step: tuple(noted[SVT.ATT][k] for k in ("ok", "rows", "nonfinite", "bad")) for step, noted in rep["steps"].items()}

    def infinite(q: np.ndarray) -> np.ndarray:
        q[0, 0] = 0x7F80
        return q

    assert notes(steps) == {"0": (True, 3, 0, 0), "1": (True, 3, 0, 0), "2": (True, 8, 0, 0), "3": (True, 3, 0, 0)}
    assert notes(SVT._with(steps, 3, SVT.Q, infinite)) == notes(steps) | {"3": (False, 3, 1, SVT.D + 1)}

"""`records.global_match`: the global match's document loads as the typed Match result and dumps back to the same JSON; a missing or unknown
field is refused by name.  A document the checker itself writes is held to the same in `tests/check/test_global_match.py`."""
from __future__ import annotations

import copy

import pytest

from verity_vllm.records.fields import MissingField, RecordError, UnknownField
from verity_vllm.records.global_match import SCHEMA, GlobalMatch

LAG = {"declared": 1, "record_attests": 1, "tokens_json_lag": 1, "exercised": True, "variable_lag": False}


def _doc() -> dict:
    """The fields `check.match.global_match.check` and its CLI write, for a two-request row whose r0 stopped at an EOS."""
    checks = [{"id": f"G{i}", "what": f"check {i}", "status": "PASS", "problems": [], "n_problems": 0} for i in range(1, 9)]
    checks[5] |= {"lengths_by_request": {"r0": {"served_output_len": 11, "executed_prefix_len": 12, "cap": 32, "cap_exhausted": False},
                                         "r1": {"served_output_len": 32, "executed_prefix_len": 32, "cap": 32, "cap_exhausted": True}},
                  "lag": LAG, "step_links": 40}
    return {"schema": SCHEMA, "comparator": "cb_a.global_match/v1", "rules_version": "r14-v1", "utc": "2026-10-25T00:00:00Z",
            "oracle_dir": "/row/match", "log": "/row/match/capture/log.jsonl.gz",
            "global_program": {"schema": "cb-a/global-program/v1", "builder": "b", "workload": None, "workload_digest": "ab" * 32,
                               "n_requests": 2, "digest_verification": None, "relocated_components": {}},
            "chronology_unit": "engine_step", "request_id_convention": "r<k>", "within_step_criterion": "dag", "within_step_note": "n",
            "checks": checks,
            "requests": [{"request_id": "r0", "workload_request_id": "u0", "index": 0, "eos_truncated": True, "eos_verified": True},
                         {"request_id": "r1", "workload_request_id": None, "index": 1, "eos_truncated": False}],
            "x09": {"verdict": "PASS"}, "request_ids": {"table": {"u0": "r0"}}, "execution": {"declared_execution": "bi-eager"},
            "achieved_concurrency": 2, "sampler_geometry": {}, "scheduled_live": {}, "lag": LAG,
            "verdict": "PASS", "failed_checks": [], "reasons": [], "eos_truncated_requests": ["r0"], "eos_verified": {"r0": True},
            "seconds": 1.5, "alternate_verdict": {"criterion": "sequence", "verdict": "PASS"}, "phases": {}, "impl": {"name": "fast"}}


def test_a_writers_document_loads_and_dumps_back_to_itself():
    doc = _doc()
    gm = GlobalMatch.of(doc, "global_match.json")
    assert gm.dump() == doc and GlobalMatch.of(gm.dump(), "global_match.json") == gm and GlobalMatch.of(gm, "x") is gm
    g6 = gm.check("G6")
    assert (gm.verdict, gm.failed_checks, gm.lag, g6.status, g6.lag) == ("PASS", (), LAG, "PASS", LAG)
    assert g6.lengths_by_request["r0"]["executed_prefix_len"] == 12 and gm.check("G1").lengths_by_request is None
    assert [(r.request_id, r.workload_request_id, r.index) for r in gm.requests] == [("r0", "u0", 0), ("r1", None, 1)]
    assert gm.check("G9") is None


def test_a_document_refused_before_the_checks_ran_loads():
    doc = {k: v for k, v in _doc().items() if k not in ("x09", "request_ids", "execution", "achieved_concurrency", "sampler_geometry",
                                                          "scheduled_live", "lag", "failed_checks", "eos_truncated_requests",
                                                          "eos_verified", "seconds", "alternate_verdict", "phases", "impl")}
    doc |= {"checks": [], "requests": [], "verdict": "FAIL", "reasons": ["global program schema 'x' != 'cb-a/global-program/v1'"]}
    gm = GlobalMatch.of(doc, "w")
    assert (gm.failed_checks, gm.lag, gm.checks, gm.check("G6")) == (None, None, (), None) and gm.dump() == doc


@pytest.mark.parametrize("path", ("schema", "verdict", "reasons", "checks", "requests", "checks.5.id", "checks.5.status",
                                  "checks.5.problems", "requests.0.request_id"))
def test_a_missing_field_is_refused_by_name(path):
    doc = _doc()
    *up, last = path.split(".")
    d = doc
    for k in up:
        d = d[int(k)] if k.isdigit() else d[k]
    del d[last]
    with pytest.raises(MissingField) as e:
        GlobalMatch.of(doc, "global_match.json")
    assert (e.value.where, e.value.field) == ("global_match.json", path.replace(".5.", "[5].").replace(".0.", "[0]."))


def test_an_unknown_field_is_refused_by_name_and_a_wrong_one_named():
    with pytest.raises(UnknownField) as e:
        GlobalMatch.of(_doc() | {"verdict_v2": "PASS"}, "w")
    assert e.value.field == "verdict_v2"
    with pytest.raises(RecordError, match="'schema' is 'cb-a/global-match/v2'"):
        GlobalMatch.of(_doc() | {"schema": "cb-a/global-match/v2"}, "w")
    doc = copy.deepcopy(_doc())
    doc["checks"][5]["lag"] = 1
    with pytest.raises(RecordError, match=r"'checks\[5\].lag' is a int, not a Mapping"):
        GlobalMatch.of(doc, "w")
    with pytest.raises(RecordError, match="'reasons' holds a value that is not a string"):
        GlobalMatch.of(_doc() | {"reasons": [1]}, "w")

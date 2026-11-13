"""`verity_vllm.check.replay.coverage.query_population` on a manifest whose header carries a dataflow guard and a cross-check: an older header's,
which a manifest of record may still carry and no writer in this tree writes.  Their state travels onto the record
(`query_total.guard`, `query_total.cross_check`) and into the scope line."""
from __future__ import annotations

import pytest

from verity_vllm.check.replay import coverage as COV
from verity_vllm.records.required_manifest import SCHEMA

OLDER_ID = "Q_module (older header, no query_id)"
GUARD = {"engine": "traversal", "armed": True, "available": True, "fully_wired": True, "audited": 7}
CROSS = {"agree": True, "primary_engine": "traversal", "other_engine": "inventory", "identities": {"traversal": 3, "inventory": 2}, "seconds": 1.5}


def _manifest(**header) -> dict:
    return {"schema": SCHEMA, "program_digest": "p" * 8, "manifest_digest": "m" * 8, "requests": ["r0"], "identities": [], **header}


def test_the_guard_and_the_cross_check_reach_the_record_with_the_fields_the_record_keeps():
    q = COV.query_population(_manifest(guard=GUARD, cross_check=CROSS), {}, set(), ranks=[])
    assert q["query_total"]["guard"] == {"engine": "traversal", "armed": True, "available": True, "fully_wired": True}
    assert q["query_total"]["cross_check"] == {"agree": True, "primary_engine": "traversal", "other_engine": "inventory",
                                               "identities": {"traversal": 3, "inventory": 2}}
    assert q["query_total"]["engine"] == "traversal"
    assert q["scope"] == f"{OLDER_ID}(P) = 0 identities (guard armed (traversal); cross-check AGREED (traversal vs inventory: 3 / 2 identities))"
    assert q["query_checks"]["ok"] is True


@pytest.mark.parametrize("header, line", [
    ({"guard": {"engine": "inventory", "armed": False}},
     "guard NOT armed (inventory); no cross-check on the manifest (one engine asked for by name: inventory)"),
    ({"guard": dict(GUARD, armed=False), "cross_check": dict(CROSS, agree=False)},
     "guard NOT armed (traversal); cross-check DISAGREED (traversal vs inventory: 3 / 2 identities)"),
    ({"cross_check": {"agree": True, "primary_engine": "traversal", "other_engine": "inventory"}},
     "guard state not on the manifest (its header predates the guard); cross-check AGREED (traversal vs inventory: None / None identities)"),
    ({}, "guard state not on the manifest (its header predates the guard); no cross-check on the manifest (one engine asked for by name: inventory)"),
])
def test_the_scope_line_says_the_guards_state_and_the_cross_checks(header, line):
    q = COV.query_population(_manifest(**header), {}, set(), ranks=[])
    assert q["scope"] == f"{OLDER_ID}(P) = 0 identities ({line})"
    assert q["query_total"]["guard"] == ({k: v for k, v in header["guard"].items() if k != "audited"} if "guard" in header else None)
    assert q["query_total"]["cross_check"] == ({k: v for k, v in header["cross_check"].items() if k != "seconds"} if "cross_check" in header else None)


def test_no_manifest_and_a_refused_one_carry_neither_and_say_which():
    none = COV.query_population(None, {}, set(), ranks=[])
    assert none["scope"] == "Q(P) = 0 identities (no manifest)" and none["query_total"]["guard"] is None and none["query_total"]["cross_check"] is None
    refused = COV.query_population({"schema": SCHEMA, "guard": GUARD}, {}, set(), ranks=[])
    assert refused["scope"] == "Q(P) = 0 identities (manifest refused)" and refused["query_total"]["guard"] is None
    assert refused["query_checks"]["ok"] is False and refused["query_checks"]["manifest_refused"]["field"] == "program_digest"


def test_an_identity_no_row_produces_that_states_no_step_is_not_read():
    """No record refuses it by name: beside an executed prefix its step is read with `int`, which refuses None."""
    man = _manifest(identities=[{"family": "input_token_ids", "request_id": "r0", "op_path": "x", "output_member": "out"}])
    prefix = {"r0": {"n_exec": 2}}
    with pytest.raises(TypeError, match="NoneType"):
        COV.query_population(man, {}, set(), ranks=[0], executed_prefix=prefix)
    assert COV.query_population(man, {}, set(), ranks=[0])["query_total"]["identities"] == 1      # no prefix: the step is not read

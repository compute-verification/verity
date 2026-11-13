"""`verity_vllm.check.replay.coverage.query_population` on a manifest whose header carries a dataflow guard and a cross-check: an older header's,
which a manifest of record may still carry and no writer in this tree writes.  Their state travels onto the record
(`query_total.guard`, `query_total.cross_check`) and into the scope line.  Beside them, on manifests written here: what each
finding names and counts, and what the record says of each request under an executed prefix."""
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


def test_a_stopped_request_s_last_manifest_step_is_the_largest_its_identities_state_below_its_stop():
    """Identities no row produces, of a request that stopped after two of its four steps.  One at a step past the stop is
    counted not executed; the cut names the last step the manifest states below the stop, in whatever order it states them."""
    prefix = {"r0": {"n_exec": 2, "cap": 4, "a_r": 0, "lag": 1, "faults": []}}

    def cut(*steps: int) -> dict:
        ids = [{"family": "input_token_ids", "request_id": "r0", "step": s, "op_path": "x", "output_member": "out"} for s in steps]
        q = COV.query_population(_manifest(identities=ids), {}, set(), ranks=[0], executed_prefix=prefix)
        (named,) = q["query_checks"]["executed_prefix_rule"]["manifest_cut_before_classification"]["first"]
        return {k: v for k, v in named.items() if k != "why"}

    assert cut(3) == {"request_id": "r0", "manifest_last_step": None, "cap_last_step": 3, "n_exec": 2, "identities_at_not_executed_steps": 1}
    assert cut(1, 0, 3)["manifest_last_step"] == cut(0, 1, 3)["manifest_last_step"] == 1
    assert cut(0, 2, 3) == {"request_id": "r0", "manifest_last_step": 0, "cap_last_step": 3, "n_exec": 2, "identities_at_not_executed_steps": 2}


def test_each_finding_names_its_first_and_counts_them_all():
    """One more than `FIRST` of each: addresses the manifest does not require; identities whose block is two rows high where one
    row produces it, annotated as three; identities at a second invocation that no alias reaches and no row produces."""
    n = COV.FIRST + 1
    addresses = {("r0", 0, f"m{i}", "out"): {"spec": "X_v1{}", "class": "vu", "rows": 1, "why": None} for i in range(n)}
    ids = [{"family": "instance_outputs", "request_id": "r0", "step": 0, "op_path": f"m{i}", "output_member": "out", "element_range": [0, 8],
            "width": 4, "rows": 3} for i in range(n)]
    outside = COV.query_population(_manifest(), addresses, set(), ranks=[0])["query_checks"]
    rows = COV.query_population(_manifest(identities=ids), addresses, set(), ranks=[0])["query_checks"]
    second = COV.query_population(_manifest(identities=[dict(i, invocation=1) for i in ids]), {}, set(), ranks=[0])["query_checks"]
    for finding in (outside["vus_outside_query"], rows["rows_disagree"], rows["rows_annotation_inconsistent"],
                    second["identities_unreachable_by_alias"], second["identities_without_rows"]):
        assert finding["n"] == n and [f["op_path"] for f in finding["first"]] == [f"m{i}" for i in range(COV.FIRST)]
    at = {"request_id": "r0", "step": 0, "op_path": "m0", "member": "out"}
    assert rows["rows_disagree"]["first"][0] == at | {"manifest_height": 2, "program_rows": 1}
    assert rows["rows_annotation_inconsistent"]["first"][0] == at | {"rows_annotation": 3, "height": 2}
    assert outside["ok"] is rows["ok"] is second["ok"] is False


def test_a_request_served_to_its_cap_counts_nothing_not_executed_beside_one_that_stopped():
    """r0 served to its cap of four steps; r1 stopped after two, with a row identity at each of its four."""
    prefix = {"r0": {"n_exec": 4, "cap": 4, "a_r": 4, "lag": 1, "faults": [], "r3_one_mechanism": True, "mechanism": "recurrence: x"},
              "r1": {"n_exec": 2, "cap": 4, "a_r": 1, "lag": 1, "faults": [], "r3_one_mechanism": True, "mechanism": "arithmetic"}}
    ids = [{"family": "instance_outputs", "request_id": "r1", "step": s, "op_path": "m", "output_member": "out"} for s in range(4)]
    addresses = {("r1", s, "m", "out"): {"spec": "X_v1{}", "class": "vu", "rows": 1, "why": None} for s in range(4)}
    strata = {"a": {"evaluated": 0}, "b": {"evaluated": 1}, "c": {"evaluated": 2}, "d": {}}
    q = COV.query_population(_manifest(identities=ids, requests=["r0", "r1"]), addresses, set(), ranks=[0], strata=strata, executed_prefix=prefix)
    assert q["query_checks"]["ok"] is True and q["query_checks"]["accounted"] is True
    by_request = q["not_executed_by_rule"]["by_request"]
    counts = ("identities", "excluded_unexecuted_identities", "excluded_row_identities", "required_row_identities")
    assert [by_request["r0"][k] for k in counts] == [0, 0, 0, 0] and [by_request["r1"][k] for k in counts] == [2, 2, 2, 2]
    rule = q["not_executed_rule"]
    assert (rule["n"], rule["requests"]["r0"]["n"], rule["requests"]["r1"]["n"]) == (2, 0, 2)
    assert rule["mechanism"] == {"arithmetic": 1, "recurrence": 1} and rule["r3_one_mechanism"] is True
    assert (q["sampled_strata"]["strata"], q["sampled_strata"]["strata_evaluated"]) == (4, 2)
    for other in (None, False):                                                   # one request whose prefix does not say it
        one = prefix | {"r1": prefix["r1"] | {"r3_one_mechanism": other}}
        assert COV.query_population(_manifest(), {}, set(), ranks=[0], executed_prefix=one)["not_executed_rule"]["r3_one_mechanism"] is False
    no_prefix = COV.query_population(_manifest(), {}, set(), ranks=[0])
    assert no_prefix["not_executed_rule"]["r3_one_mechanism"] is False and no_prefix["sampled_strata"]["strata_evaluated"] is None

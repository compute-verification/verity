"""`verity_vllm.check.replay.coverage.query_population` on a manifest whose header carries a dataflow guard and a cross-check: an older header's,
which a manifest of record may still carry and no writer in this tree writes.  Their state travels onto the record
(`query_total.guard`, `query_total.cross_check`) and into the scope line.  Beside them, on manifests written here: what each
finding names and counts, what the record says of each request under an executed prefix, and that a module a mechanism names
is one the tree has."""
from __future__ import annotations

import importlib.util
import re

import pytest

from verity_vllm.check.replay import coverage as COV
from verity_vllm.records.required_manifest import SCHEMA

OLDER_ID = "Q_module (older header, no query_id)"
#: a module as a mechanism's text names it: a dotted lower-case name in parentheses or backticks
MODULE_NAMED = re.compile(r"[(`]((?:[a-z_][a-z0-9_]*\.)+[a-z_][a-z0-9_]*)[)`]")
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
    more = _manifest(guard=GUARD | {"flags": ["hidden_bn"], "named_input": True},
                     cross_check=CROSS | {"digests": {"traversal": "t"}, "complete": True, "why": "w"},
                     query={"query_digest": "q" * 8, "implemented_query": "the module-boundary query"})
    total = COV.query_population(more, {}, set(), ranks=[])["query_total"]
    assert total["guard"] == {k: v for k, v in more["guard"].items() if k != "audited"}
    assert total["cross_check"] == {k: v for k, v in more["cross_check"].items() if k != "seconds"}
    assert (total["query_digest"], total["implemented_query"]) == ("q" * 8, "the module-boundary query")


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
    # no alias table: said of the identities it leaves unreached, and of none where there are none
    assert rows["identities_unreachable_by_alias"] == {"n": 0, "first": [], "alias_entries": 0, "why": None}
    unreached = second["identities_unreachable_by_alias"]
    assert unreached["alias_entries"] == 0 and unreached["why"].startswith("invocation >= 1 identities that no `op_path_aliases` entry maps")
    assert unreached["why"].endswith("read as vus_outside_query and these identities as identities_without_rows")




def _stands(name: str) -> bool:
    for full in (name, f"verity_vllm.{name}"):
        try:
            if importlib.util.find_spec(full) is not None:
                return True
        except ModuleNotFoundError:
            continue
    return False


def test_a_module_a_mechanism_names_is_one_the_tree_has():
    """A mechanism is free text on the record (`population.uncovered_by_name`).  Where it names a module, as itself or by its
    place under `verity_vllm`, a reader of the record finds the module there."""
    named = {(family, m) for family, text in COV.QUERY_FAMILY_MECHANISM.items() for m in MODULE_NAMED.findall(text)}
    assert {m for _family, m in named} >= {"query.norm_scales", "query.router_softmax", "query.vocab_range", "query.scale_products",
                                           "check.replay.checks"}
    assert {family for family, m in named if "exactness" in m} == {"router_softmax", "vocab_range"}
    assert sorted(fm for fm in named if not _stands(fm[1])) == []

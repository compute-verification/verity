"""`records.query_checks`: the sampled replay's reconciliation against Q(P) as its writers write it, refused by name when a
field they all write is missing, and graded by the C2 rules (`check.c2_rules`) to the same texts as the parsed JSON.

The blocks come from the replay's coverage (`check.replay.coverage.query_population`) over `data/required_manifest.json` and
its row addresses, reconciled and with one fault each, beside the served check's and a TP rank's field sets.
`tests/check/test_served_check.py` reads the served check's own blocks through the same rules.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from verity_vllm.check import c2_rules as C2
from verity_vllm.check.replay import coverage as COV
from verity_vllm.records.fields import MissingField, RecordError
from verity_vllm.records.query_checks import Findings, QueryChecks, Unreachable

DATA = Path(__file__).parent / "data"
#: sha256 of the four C2 readers' outcomes over `_corpus()` (`json.dumps(..., sort_keys=True, default=str)`), computed with the
#: readers on the parsed JSON before they read the record
OUTCOMES_SHA256 = "57da25a573b80ddc29057f1b344f4c23fd8c0ee04eca109151c26706e6469449"
REQUIRED = ("vus_outside_query", "identities_without_rows", "rows_disagree", "ranks", "unstated_families", "vus_outside_query.n",
            "vus_outside_query.first", "rows_disagree.first", "ranks.replay", "identities_unreachable_by_alias.alias_entries",
            "identities_unreachable_by_alias.n", "executed_prefix_rule.applied", "executed_prefix_rule.faults",
            "executed_prefix_rule.evaluated_past_stop", "executed_prefix_rule.manifest_cut_before_classification",
            "executed_prefix_rule.evaluated_past_stop.first")
#: present on every block, null where the writer has nothing to state
STATED = ("ok", "accounted", "ranks.manifest", "ranks.equal")


def _manifest() -> dict:
    return json.loads((DATA / "required_manifest.json").read_text())


def _addresses() -> dict[tuple, dict]:
    return {tuple(k): v for *k, v in json.loads((DATA / "replay_addresses.json").read_text())}


def _coverage(**kw) -> dict:
    return COV.query_population(_manifest(), _addresses(), set(), ranks=[0], **kw)["query_checks"]


def _served() -> dict:
    """The fields the served check writes (`check.replay.served_check.population`), and no others."""
    return {"vus_outside_query": {"n": 0, "first": []}, "identities_without_rows": {"n": 0, "first": []}, "rows_disagree": {"n": 0, "first": []},
            "ranks": {"replay": [0], "manifest": [0], "equal": True}, "accounted": True, "unstated_families": [], "ok": True}


def _corpus() -> dict[str, object]:
    def sr_of(qp, facts=None):
        sr = {"result": True, "grade": "complete", "population": {"query_total": qp["query_total"], "query_checks": qp["query_checks"], "scope": qp["scope"]}}
        if facts is not None:
            sr["executed_prefix_of_record"] = {"facts_of_record": facts}
        return sr

    def qp(manifest=None, addresses=None, evaluated=(), ranks=(0,), **kw):
        return COV.query_population(_manifest() if manifest is None else manifest, _addresses() if addresses is None else addresses,
                                    set(evaluated), ranks=list(ranks), **kw)

    def prefix(n_exec, faults=()):
        return {"r0": {"faults": list(faults), "n_exec": n_exec, "cap": 3, "a_r": 0, "lag": 1}}

    def edited(name, edit):
        q = qp()
        edit(q)
        out[name] = sr_of(q)

    out: dict[str, object] = {}
    A = _addresses()
    out["coverage reconciled"] = sr_of(qp())
    out["coverage no rows"] = sr_of(qp(addresses={}))
    out["coverage no manifest"] = sr_of(COV.query_population(None, A, set(), ranks=[0]))
    out["coverage refused"] = sr_of(qp({k: v for k, v in _manifest().items() if k != "program_digest"}))
    k0 = next(iter(A))
    out["coverage outside"] = sr_of(qp(addresses=A | {(k0[0], k0[1], "model.extra", k0[3]): A[k0]}))
    out["coverage other rank"] = sr_of(qp(ranks=(1,)))
    faulted = qp(executed_prefix=prefix(1, ["the tokens.json leg is gone"]))
    out["coverage prefix faulted, account missing"] = sr_of(faulted, {"r0": {"account_missing": True, "linked": False}})
    out["coverage prefix faulted, linked"] = sr_of(faulted, {"r0": {"account_missing": False, "linked": True}})
    out["coverage prefix faulted, no facts"] = sr_of(faulted)
    out["coverage prefix applied"] = sr_of(qp(executed_prefix=prefix(3)))
    out["coverage prefix not executed"] = sr_of(qp(executed_prefix=prefix(2)))
    out["coverage evaluated past stop"] = sr_of(qp(evaluated={(k[0], k[1], k[2]) for k in A}, executed_prefix=prefix(1)))
    cut = _manifest()
    cut["identities"] = [i for i in cut["identities"] if i["step"] != 2]
    out["coverage manifest cut"] = sr_of(qp(cut, executed_prefix=prefix(2)))
    alias = _manifest()
    alias["identities"][0]["invocation"] = 1
    out["coverage unreachable by alias"] = sr_of(qp(alias))
    out["coverage controller rows disagree"] = sr_of(qp(controller_rows={"r0": 3}))
    rows = _manifest()
    i0 = next(i for i in rows["identities"] if i["op_path"] == "logits_processor")
    i0["element_range"] = [0, i0["element_range"][1] * 2]
    out["coverage rows disagree"] = sr_of(qp(rows))
    edited("coverage cross-check disagrees", lambda q: q["query_total"].update(cross_check={"agree": False, "why": "the inventory has 3 more identities"}))
    edited("coverage missing acquisition", lambda q: q["query_checks"].update(missing_acquisition_families=["future_family"], ok=False))
    edited("coverage unstated", lambda q: q["query_checks"].update(unstated_families=["future_family"], ok=False))
    edited("coverage ok False with nothing named", lambda q: q["query_checks"].update(ok=False))

    first = [{"request_id": "r1", "step": 2, "op_path": "c00.input", "member": "ids"}]
    qt = {"source": "the served Build of record (acquire.served)", "query_id": "served-v1", "identities": 140, "by_family": {"x": 140}}
    out["served reconciled"] = {"population": {"query_total": qt, "query_checks": _served(), "scope": "served-v1(P) = 140 identities"}}
    out["served outside and without"] = {"population": {"query_total": qt, "scope": "s", "query_checks": _served() | {
        "ok": False, "vus_outside_query": {"n": 2, "first": first}, "identities_without_rows": {"n": 1, "first": first}}}}
    out["served other rank"] = {"population": {"query_total": qt, "scope": "s", "query_checks": _served() | {
        "ok": False, "ranks": {"replay": [1], "manifest": [0], "equal": False}}}}

    def tp(ok, outside=0, without=0, unreachable=None):
        f = [{"request_id": "r0", "step": 0, "op_path": "model.layers.1.self_attn.rotary_emb", "member": "q", "rank": 0}]
        chk = _served() | {"vus_outside_query": {"n": outside, "first": f if outside else []},
                           "identities_without_rows": {"n": without, "first": f if without else []},
                           "ranks": {"replay": [0], "manifest": [0, 1], "equal": True}, "ok": ok,
                           "executed_prefix_rule": {"applied": True, "faults": [], "evaluated_past_stop": {"n": 0, "first": []},
                                                    "manifest_cut_before_classification": {"n": 0, "first": []}}}
        if unreachable is not None:
            chk["identities_unreachable_by_alias"] = {"n": unreachable, "first": f[:1] if unreachable else [], "alias_entries": 0}
        return {"result": True, "grade": "complete", "population": {
            "vus": 8067, "scope": "Q_module_v1(P) = 8067 identities (...)",
            "query_total": {"identities": 8067, "query_id": "Q_module_v1", "cross_check": {"agree": True}}, "query_checks": chk}}
    out["tp reconciled"] = tp(True)
    out["tp not reconciled"] = tp(None)
    out["tp outside and without"] = tp(False, 960, 960)
    out["tp unreachable"] = tp(False, 960, 960, 960)
    out["legacy no query_total"] = {"population": {"query_checks": _served(), "scope": "s"}}
    out["legacy checks not a dict"] = {"population": {"query_total": qt, "query_checks": [_served()], "scope": "s"}}
    out["legacy no population"] = {"result": True}
    out["not a dict"] = None
    return out


def _outcomes() -> dict[str, dict]:
    res = {}
    for name, sr in _corpus().items():
        r: dict = {"scope": C2._query_population_scope(sr)}
        if isinstance(sr, dict):
            r |= {"facts": C2._faulted_prefix_facts(sr), "account_missing": C2._scope_fail_account_missing(sr)}
        res[name] = r | {"replay_scope": C2._replay_scope(sr)}
    return res


def _drop(doc: dict, path: str) -> dict:
    doc = copy.deepcopy(doc)
    *up, last = path.split(".")
    d = doc
    for k in up:
        d = d[k]
    del d[last]
    return doc


def _full() -> dict:
    """A coverage block with every optional check present: the alias table's finding and the executed-prefix rule."""
    return _coverage(executed_prefix={"r0": {"faults": [], "n_exec": 3, "cap": 3, "a_r": 0, "lag": 1}})


def test_the_replays_and_the_served_checks_blocks_load():
    doc = _full()
    c = QueryChecks.of(doc, "the replay's block")
    assert c.where == "the replay's block" and c.ok is True and c.accounted is True
    assert c.vus_outside_query == Findings(0, []) and c.ranks.replay == [0] and c.ranks.manifest == [0] and c.ranks.equal is True
    assert isinstance(c.identities_unreachable_by_alias, Unreachable) and c.identities_unreachable_by_alias.alias_entries == 0
    assert c.executed_prefix_rule is not None and c.executed_prefix_rule.applied is True and c.executed_prefix_rule.faults == []
    assert c.rows_annotation_inconsistent is not None and c.controller_rows_disagree is not None
    assert (c.manifest_refused, c.missing_acquisition_families) == (None, None)
    assert QueryChecks.of(c, "elsewhere") is c
    s = QueryChecks.of(_served(), "the served check's block")
    assert (s.identities_unreachable_by_alias, s.rows_annotation_inconsistent, s.controller_rows_disagree, s.executed_prefix_rule) == (None,) * 4


def test_a_block_without_a_manifest_states_nothing_reconciled():
    c = QueryChecks.of(COV.query_population(None, _addresses(), set(), ranks=[0])["query_checks"], "the replay's block")
    assert (c.ok, c.accounted, c.ranks.manifest, c.ranks.equal) == (None, None, None, None)


@pytest.mark.parametrize("field", REQUIRED + STATED)
def test_a_block_missing_a_field_its_writers_write_is_refused_naming_the_field(field):
    with pytest.raises(MissingField) as e:
        QueryChecks.of(_drop(_full(), field), "the test's block")
    assert (e.value.where, e.value.field) == ("the test's block", field)
    assert str(e.value) == f"the test's block: field {field!r} is missing: every writer of the record writes it"


@pytest.mark.parametrize("field", ("vus_outside_query", "ranks", "unstated_families", "vus_outside_query.n", "ranks.replay"))
def test_a_null_required_field_is_refused(field):
    doc = _full()
    *up, last = field.split(".")
    (doc[up[0]] if up else doc)[last] = None
    with pytest.raises(MissingField, match=f"field {field!r} is missing"):
        QueryChecks.of(doc, "the test's block")


@pytest.mark.parametrize("field, value", [("ok", "yes"), ("accounted", 1), ("vus_outside_query", []), ("unstated_families", "x"),
                                          ("manifest_refused", "refused"), ("missing_acquisition_families", "x"),
                                          ("executed_prefix_rule", [])])
def test_a_mistyped_field_is_refused_by_name(field, value):
    with pytest.raises(RecordError) as e:
        QueryChecks.of(dict(_full(), **{field: value}), "the test's block")
    assert not isinstance(e.value, MissingField) and e.value.field == field
    assert str(e.value).startswith(f"the test's block: field {field!r} is a ")


@pytest.mark.parametrize("edit, field", [(lambda d: d["vus_outside_query"].update(n="3"), "vus_outside_query.n"),
                                         (lambda d: d["rows_disagree"].update(first=["r0"]), "rows_disagree.first[0]"),
                                         (lambda d: d["ranks"].update(equal=1), "ranks.equal"),
                                         (lambda d: d["executed_prefix_rule"].update(faults=["the tokens.json leg is gone"]), "executed_prefix_rule.faults[0]")])
def test_a_mistyped_nested_field_is_refused_by_its_path(edit, field):
    doc = _full()
    edit(doc)
    with pytest.raises(RecordError) as e:
        QueryChecks.of(doc, "the test's block")
    assert e.value.field == field


def test_a_block_that_is_not_a_json_object_is_refused():
    with pytest.raises(RecordError, match="the test's block: field 'ok' cannot be read: the block is a list"):
        QueryChecks.of([_served()], "the test's block")


def test_the_c2_rules_grade_every_writers_block_to_the_same_texts():
    outcomes = _outcomes()
    assert hashlib.sha256(json.dumps(outcomes, sort_keys=True, default=str).encode()).hexdigest() == OUTCOMES_SHA256
    assert outcomes["coverage reconciled"]["scope"][0].startswith("C2 population scope (query of record Q_module_body_v1)")
    assert "ranks replay [1] != manifest [0]" in outcomes["served other rank"]["scope"][1]
    assert outcomes["coverage prefix faulted, account missing"]["account_missing"] is True


@pytest.mark.parametrize("field", ("ok", "ranks.equal", "executed_prefix_rule.manifest_cut_before_classification"))
def test_a_refused_block_fails_the_population_scope_by_name_never_a_crash_or_a_pass(field):
    q = COV.query_population(_manifest(), _addresses(), set(), ranks=[0], executed_prefix={"r0": {"faults": [], "n_exec": 3, "cap": 3, "a_r": 0, "lag": 1}})
    sr = {"population": {"query_total": q["query_total"], "query_checks": _drop(q["query_checks"], field), "scope": q["scope"]}}
    note, fail = C2._query_population_scope(sr)
    assert note is None and fail == (f"C2 population reconciliation refused, nothing graded ({C2.QUERY_CHECKS}: field {field!r} is missing: "
                                     "every writer of the record writes it)")
    assert C2._replay_scope(sr)["query_checks_ok"] is None and C2._faulted_prefix_facts(sr) == [] and C2._scope_fail_account_missing(sr) is False

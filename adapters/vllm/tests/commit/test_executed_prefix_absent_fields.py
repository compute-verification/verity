"""`verity_vllm.commit.executed_prefix`: what its readers take a field to mean that the record they read leaves out, holds null or
holds empty.  A classification that states no request or family counts none there, a binding map without entries commits no row, a
step that declares no member is dropped, and a linkage that states no horizon links no step."""
from __future__ import annotations

from typing import Any

import pytest

from tests.records.test_commit_binding import EOS, _binding
from verity_vllm.commit import executed_prefix as EP
from verity_vllm.records.schema_ids import NOT_EXECUTED_RECONCILIATION_SCHEMA

#: the identity coverage `commit.binding.coverage_check` writes, which the account links sit beside
COVERED = {"ok": True, "expected_n": 4, "bound_n": 4, "missing_n": 0, "ambiguous_n": 0}
#: a Commit's token record link (`check.commit_tokens`) that holds for r0, with no Match record beside the Commit
LINK = {"ok": True, "required": True, "lag_attested": True, "problems": [], "record": {"path": "/row/tokens.json", "sha256": "ab" * 32},
        "served_by_request": {"r0": 2}, "tokens_committed": {"r0": True}}


def _epr() -> dict:
    """The executed prefixes of the records' fixture binding: r0 stopped by EOS after 2 tokens under LAG 1 and executed 3 steps; r1
    served all 3 of its own."""
    return _binding()["coverage"]["executed_prefix_of_record"]


@pytest.mark.parametrize("epr", [{}, {"requests": None}, {"requests": {}}])
def test_a_record_of_no_request_gives_no_fact(epr):
    assert EP.executed_prefix_facts(epr, None, None, {}, {}) == {}


def test_the_facts_read_the_programs_stop_rule_as_declared():
    """`stop.eos` left out or null is an undeclared stop rule (`eos_declared` None, no `eos_ids`); an empty list declares no id, so
    the last served token is not one."""
    epr, served = _epr(), {"r0": [5, EOS]}
    for stop in ({}, {"eos": None}):
        f = EP.executed_prefix_facts(epr, None, None, {}, stop, served_tokens=served)["r0"]
        assert f["eos_declared"] is None and f["eos_ids"] is None and f["lag"] == -1
    empties: list[Any] = [[], ()]
    for empty in empties:
        f = EP.executed_prefix_facts(epr, None, None, {}, {"eos": empty}, served_tokens=served)["r0"]
        assert f["eos_declared"] is False and f["eos_ids"] == []
    f = EP.executed_prefix_facts(epr, None, None, {}, {"eos": [EOS, 3], "declared_lag": 1}, served_tokens=served)["r0"]
    assert f["eos_declared"] is True and f["eos_ids"] == [3, EOS] and f["lag"] == 1 and f["served"] == [5, EOS]


@pytest.mark.parametrize("linkage, linked", [
    (None, False),
    ({}, False),
    ({"result": True}, False),
    ({"result": True, "horizon": None}, False),
    ({"result": True, "horizon": {}}, False),
    ({"result": True, "horizon": {"r1": {"linked_steps": 3}}}, False),
    ({"result": True, "horizon": {"r0": None}}, False),
    ({"result": True, "horizon": {"r0": {}}}, False),
    ({"result": True, "horizon": {"r0": {"linked_steps": 2}}}, False),
    ({"result": True, "horizon": {"r0": {"linked_steps": 3}}}, True),
    ({"result": False, "horizon": {"r0": {"linked_steps": 3}}}, False),
])
def test_a_linkage_that_states_no_horizon_for_the_request_does_not_link_its_prefix(linkage, linked):
    """On the Commit's token record the prefix is linked only where the boundary linkage passed and ran over exactly the request's
    executed prefix: no linkage, no `horizon`, none for the request and one without `linked_steps` link nothing."""
    epr = _epr()
    assert epr["requests"]["r0"]["executed_prefix_len"] == 3
    f = EP.executed_prefix_facts(epr, {"coverage": COVERED | {"commit_account_link": LINK}}, linkage, {"r0": 0}, {"eos": [EOS], "declared_lag": 1})
    assert f["r0"]["linked"] is linked and f["r0"]["a_r"] == 2 and f["r0"]["lag_attested"] is True


def test_a_binding_map_without_entries_commits_no_row_past_the_stop():
    none = {"ok": True, "entries_n": 0, "requests": {}, "steps_past_by_request": {}, "padding_entries_n": 0, "padded_record": False,
            "padding_inside_prefix": {}, "error": None}
    bmaps: list[dict[str, Any]] = [{}, {"entries": None}, {"entries": []}]
    for bmap in bmaps:
        assert EP.rows_past_the_stop(bmap, {"r0": [0, 1]}) == none


def test_a_leaf_without_an_identity_is_no_requests_row():
    """A leaf entry whose `identity` is left out, null or empty is skipped, whatever the declared requests are named; one that names
    a declared request at a step outside its prefix is execution past the stop."""
    blank = [{"kind": "leaf"}, {"kind": "leaf", "identity": None}, {"kind": "leaf", "identity": {}}, {"kind": "node", "identity": {"request_id": "r0", "step": 9}}]
    for steps in ({"r0": [0, 1]}, {"None": [0]}):
        got = EP.rows_past_the_stop({"entries": blank}, steps)
        assert got["ok"] and got["entries_n"] == 0 and got["requests"] == {}
    past = EP.rows_past_the_stop({"entries": blank + [{"kind": "leaf", "identity": {"request_id": "r0", "step": 2}},
                                                      {"kind": "leaf", "identity": {"request_id": "r0", "step": 1}},
                                                      {"kind": "leaf", "identity": {"request_id": "r0"}},
                                                      {"kind": "leaf", "identity": {"request_id": "r9", "step": 2}}]}, {"r0": [0, 1]})
    assert not past["ok"] and past["entries_n"] == 1 and past["requests"] == {"r0": [2]} and past["steps_past_by_request"] == {"r0": 1}


def test_an_engine_step_that_declares_no_member_is_dropped():
    assert EP.restrict_engine_steps(None, {"r0": [0]}) == ([], {"members_dropped": 0, "steps_dropped": 0, "steps_changed": 0})
    steps: list[dict[str, Any]] = [
        {"engine_step": 0}, {"engine_step": 1, "members": None}, {"engine_step": 2, "members": []},
        {"engine_step": 3, "members": [{"request_id": "r0", "rows": 2}, {"request_id": "r1", "rows": 3}], "rows": 5, "n_seq": 2},
        {"engine_step": 4, "members": [{"request_id": "r0", "rows": 2}], "rows": 2, "n_seq": 1},
        {"engine_step": 5, "members": [{"request_id": "r1", "rows": 1}], "rows": 1, "n_seq": 1}]
    out, counts = EP.restrict_engine_steps(steps, {"r0": [3], "r1": [3, 5]})
    assert out == [steps[3], steps[5]] and counts == {"members_dropped": 1, "steps_dropped": 4, "steps_changed": 0}
    out, counts = EP.restrict_engine_steps(steps[3:], {"r0": [4], "r1": [3, 5]})
    assert out == [dict(steps[3], members=[{"request_id": "r1", "rows": 3}], rows=3, n_seq=1), steps[4], steps[5]]
    assert counts == {"members_dropped": 1, "steps_dropped": 0, "steps_changed": 1}


#: one plane row of two requests at engine step 3, and one named-input row of r0 at engine step 3
PLANE = {"family": "plane", "request_id": "*", "requests": ["r0", "r1"], "engine_step": 3}
PROMPT = {"family": "input_token_ids", "request_id": "r0", "engine_step": 3}


@pytest.mark.parametrize("population", [{}, {"by_request": None}, {"by_request": {}}, {"by_request": {"r0": None, "r1": 5}},
                                        {"by_request": {"r0": {}, "r1": {"arrive": 1}}}])
def test_a_population_that_states_no_requests_n_exec_places_no_cell(population):
    """The arithmetic counts each family's rows; a cell is past the stop only where the population states its request's `n_exec`."""
    assert EP.expected_cells_from_population([PLANE, PROMPT, {"family": "row", "request_id": "r0", "engine_step": 9}], population) == {
        "plane": {"cells": 0, "rows": 1}, "input_token_ids": {"cells": 0, "rows": 1}}
    assert EP.expected_cells_from_population(None, population) == {}


def test_the_arithmetic_reads_n_exec_and_arrival_where_the_population_states_them():
    by_request = {"r0": {"n_exec": 3}, "r1": {"n_exec": 3, "arrive": 1}, "r2": {"n_exec": None, "arrive": None}}
    assert EP.expected_cells_from_population([PLANE, PROMPT], {"by_request": by_request}) == {
        "plane": {"cells": 1, "rows": 1}, "input_token_ids": {"cells": 1, "rows": 1}}


def _reconciled(**over) -> dict:
    return {"schema": NOT_EXECUTED_RECONCILIATION_SCHEMA, "ok": True, "problems": [], "n_exec": {}, "by_family": {},
            "outside_population_reach": {}, "rule": EP.RULE, "scope": EP.ACTIVE_REFERENCE_ONLY, "total": True, **over}


def test_a_classification_that_states_nothing_reconciles_with_one_that_counts_nothing():
    """A population block that is applied and states no request, total, kind or batch-level cell agrees with a Commit that classified
    nothing, whether the Commit's classification and its prefix record are absent, null inside or empty."""
    classes: list[dict[str, Any] | None] = [None, {}, {"by_family": None, "kind_by_family": None}, {"by_family": {}, "kind_by_family": {}}]
    prefixes: list[dict[str, Any] | None] = [None, {}, {"requests": None}, {"requests": {}}]
    populations: list[dict[str, Any]] = [
        {"applied": True}, {"applied": True, "by_request": None, "totals": None, "kind_by_family": None, "batch_level": None},
        {"applied": True, "by_request": {}, "totals": {}, "kind_by_family": {}, "batch_level": {}},
        {"applied": True, "totals": {"by_family": None}}, {"applied": True, "totals": {"by_family": {}}}]
    for commit_class in classes:
        for epr in prefixes:
            for population in populations:
                assert EP.reconcile_not_executed(commit_class, population, epr) == _reconciled()
    absent: list[dict[str, Any] | None] = [None, {}]
    for no_block in absent:
        got = EP.reconcile_not_executed(None, no_block, None)
        assert got["ok"] is False and "total" not in got and "no not_executed_by_rule block" in got["problems"][0]


@pytest.mark.parametrize("by_request", [None, {}, {"r1": {"n_exec": 3}}, {"r0": None}, {"r0": {}}, {"r0": 3}, {"r0": {"n_exec": None}}])
def test_a_request_the_population_states_no_n_exec_for_is_not_reconciled(by_request):
    got = EP.reconcile_not_executed(None, {"applied": True, "by_request": by_request}, _epr())
    assert got["ok"] is False and got["total"] is False
    assert got["n_exec"]["r0"] == {"commit": 3, "population": None, "equal": False}
    assert got["problems"][0] == "r0: executed prefix of record 3 != the population's n_exec None (two prefixes on one record)"


def test_a_request_is_reconciled_on_the_populations_n_exec():
    by_request = {"r0": {"n_exec": 3}, "r1": {"n_exec": 3, "arrive": 1}}
    got = EP.reconcile_not_executed(None, {"applied": True, "by_request": by_request}, _epr())
    assert got == _reconciled(n_exec={r: {"commit": 3, "population": 3, "equal": True} for r in ("r0", "r1")})
    off = EP.reconcile_not_executed(None, {"by_request": by_request}, _epr())
    assert off["problems"] == ["population.not_executed_by_rule not applied on a record with an EOS-stopped request"] and off["n_exec"]["r0"]["equal"]


@pytest.mark.parametrize("totals, n_ex", [(None, 0), ({}, 0), ({"by_family": None}, 0), ({"by_family": {"rows": None}}, 0),
                                          ({"by_family": {"rows": {}}}, 0), ({"by_family": {"rows": 4}}, 0),
                                          ({"by_family": {"rows": {"N_excluded": 4}}}, 4)])
def test_a_row_family_the_populations_totals_do_not_name_excluded_none(totals, n_ex):
    """A family the Commit classified is a row family unless its `kind_by_family` says otherwise, and the population excluded none of
    it where its totals leave the family out or state no `N_excluded`."""
    named: list[dict[str, Any]] = [{}, {"kind_by_family": None}, {"kind_by_family": {}}, {"kind_by_family": {"other": "batch-level"}},
                                   {"kind_by_family": {"rows": "row"}}]
    for kinds in named:
        got = EP.reconcile_not_executed({"by_family": {"rows": 4}, **kinds}, {"applied": True, "totals": totals}, None)
        assert got["by_family"] == {"rows": {"commit": 4, "population": n_ex, "equal": n_ex == 4}} and got["ok"] is (n_ex == 4)
        assert got["outside_population_reach"] == {}
    if n_ex:
        mine = EP.reconcile_not_executed(None, {"applied": True, "totals": totals}, None)
        assert mine["by_family"] == {"rows": {"commit": 0, "population": 4, "equal": False}}, "a family only the population names"


@pytest.mark.parametrize("stated, cells", [
    ({}, None),
    ({"batch_level": None}, None),
    ({"batch_level": {}}, None),
    ({"batch_level": {"plane": None}}, None),
    ({"batch_level": {"plane": {}}}, None),
    ({"batch_level": {"plane": {"cells_not_executed": None}}}, None),
    ({"batch_level": {"plane": {"cells_not_executed": 2}}}, 2),
    ({"totals": {"by_family": {"plane": {"plane_cells_not_executed": {"cells": 2}}}}}, 2),
    ({"totals": {"by_family": {"plane": {"plane_cells_not_executed": {"cells": 2}}}}, "batch_level": {"plane": {"cells_not_executed": 5}}}, 2),
    ({"totals": {"by_family": {"plane": {"plane_cells_not_executed": {}}}}, "batch_level": {"plane": {"cells_not_executed": 5}}}, 5),
])
def test_a_batch_level_family_is_reconciled_on_the_cells_the_population_states(stated, cells):
    """The population reaches a batch-level family where its totals or its `batch_level` block state the family's cells; a population
    that states neither does not reach it, and the family is named outside its reach, pending."""
    commit_class = {"by_family": {"plane": 2}, "kind_by_family": {"plane": EP.KIND_BATCH}}
    got = EP.reconcile_not_executed(commit_class, {"applied": True, **stated}, None)
    if cells is None:
        assert got["by_family"] == {} and got["ok"] and got["total"] is False
        assert got["outside_population_reach"]["plane"]["population_reaches"] is False and got["outside_population_reach"]["plane"]["commit"] == 2
    else:
        assert got["outside_population_reach"] == {} and got["ok"] is (cells == 2)
        assert got["by_family"] == {"plane": {"commit": 2, "population": cells, "equal": cells == 2, "kind": EP.KIND_BATCH, "unit": "cells",
                                              "population_N_excluded": 0}}


@pytest.mark.parametrize("stated, reaches", [({}, False), ({"kind_by_family": None}, False), ({"kind_by_family": {}}, False),
                                             ({"kind_by_family": {"input_token_ids": None}}, False), ({"kind_by_family": {"input_token_ids": "row"}}, False),
                                             ({"kind_by_family": {"other": "named-input"}}, False), ({"kind_by_family": {"input_token_ids": "named-input"}}, True)])
def test_a_named_input_family_is_reconciled_once_the_population_names_its_kind(stated, reaches):
    commit_class = {"by_family": {"input_token_ids": 2}, "kind_by_family": {"input_token_ids": EP.KIND_INPUT}}
    population = {"applied": True, "totals": {"by_family": {"input_token_ids": {"N_excluded": 2}}}, **stated}
    got = EP.reconcile_not_executed(commit_class, population, None)
    if reaches:
        assert got == _reconciled(by_family={"input_token_ids": {"commit": 2, "population": 2, "equal": True, "kind": EP.KIND_INPUT, "unit": "identities"}})
    else:
        assert got["by_family"] == {} and got["outside_population_reach"]["input_token_ids"]["population_N_excluded"] == 2 and got["total"] is False


def test_a_family_outside_the_populations_reach_is_held_to_its_own_arithmetic():
    """With the manifest's rows, a family the population does not reach is still equal only to what the population's `n_exec` and
    arrivals imply, and a population without `by_request` implies no cell."""
    commit_class = {"by_family": {"plane": 1, "input_token_ids": 1}, "kind_by_family": {"plane": EP.KIND_BATCH, "input_token_ids": EP.KIND_INPUT}}
    by_request = {"r0": {"n_exec": 3}, "r1": {"n_exec": 3, "arrive": 1}}
    got = EP.reconcile_not_executed(commit_class, {"applied": True, "by_request": by_request}, None, manifest_identities=[PLANE, PROMPT])
    assert got["ok"] and got["total"] and {k: (v["equal"], v["expected_from_population_facts"]) for k, v in got["outside_population_reach"].items()} == {
        "plane": (True, 1), "input_token_ids": (True, 1)}
    bare = EP.reconcile_not_executed(commit_class, {"applied": True}, None, manifest_identities=[PLANE, PROMPT])
    assert bare["ok"] is False and bare["total"] is False and len(bare["problems"]) == 2
    assert {k: v["expected_from_population_facts"] for k, v in bare["outside_population_reach"].items()} == {"plane": 0, "input_token_ids": 0}

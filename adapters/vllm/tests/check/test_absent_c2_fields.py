"""What the C2 rules make of a field the Commit record leaves out.

The rules read the sampled replay's record, the oracle compare's, the weights pin's and the route pin's, each a
section of the Commit record.  Each test leaves one field out (absent, null or empty) and states what the rule then
says, beside what it says where the record states the field.
"""
from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.check.test_absent_commit_fields import ABSENT, put
from tests.check.test_absent_link_fields import left_out
from tests.check.test_commit_verdict_selector_no_evaluator_gap import (
    K_ROPE,
    K_SEL,
    OC_STOCH,
)
from tests.check.test_rev_r16_named_population_gap import (
    K_06F,
    K_06G,
    OC_COMPLETE,
    SR_COMPLETE_INTERIOR_ONLY,
    SR_COMPLETE_WITH_NAMED_EXCLUSIONS,
)
from tests.check.test_verdict import _base
from verity_vllm.check import c2_rules as C2
from verity_vllm.check import verdict as V
from verity_vllm.check.replay import replay_codes as RC

NO_GAP = (None, None)
BY_FORM_B = "C2 of record by form (B) [oracle compare complete: 3 compared, 0 mismatch, instance outputs 2]"
PLANE_FED = "C2 of record not established: 64 plane-fed VUs "
NORM = "RMSNormFusedCuda_v2{N=2048}|r0|rank0|first-decode|model.layers.1.input_layernorm|fresh-window"
BIAS = "BiasAdd_v1{N=2048}|r0|rank0|first-decode|model.layers.1.self_attn.qkv_proj|fresh-window"
WEIGHTS, FACTS = "checkpoint_binding/weights_of_record", ("sampled_replay", "executed_prefix_of_record", "facts_of_record", "r0", "account_missing")


def without(record: dict[str, Any], path: tuple[str, ...], stated: Any) -> dict[str, Any]:
    """A copy of `record` with `stated` at `path`, or without the field where `stated` is ABSENT."""
    return put(copy.deepcopy(record), path, stated)


def stamped(replay: dict[str, Any]) -> dict[str, Any]:
    """`replay` with the code of every reason of its population stamped on it, as the sampled replay writes its record."""
    return without(replay, ("population", "not_evaluable_codes"), RC.not_evaluable_codes(replay["population"]))


def partial_replay() -> dict[str, Any]:
    """A partial replay with no mismatch that names its two unevaluated strata, each under a class form (B) covers."""
    strata = [{"stratum": NORM, "spec": "RMSNormFusedCuda_v2{N=2048}", "op_path": "model.layers.1.input_layernorm", "n": 2,
               "why": "no evaluator for RMSNormFusedCuda_v2{N=2048} variant", "classes": ["no-evaluator"]},
              {"stratum": BIAS, "spec": "BiasAdd_v1{N=2048}", "op_path": "model.layers.1.self_attn.qkv_proj", "n": 2,
               "why": "fed by interior producer(s) without an evaluator (Gemm_v2)", "classes": ["interior-composed-v2"]}]
    return {"result": None, "grade": "partial", "why": "2 strata not evaluated", "strata_unevaluated": [NORM, BIAS], "unevaluated_strata": strata,
            "sample": {"evaluated": 598, "equal": 598, "mismatch_n": 0, "not_evaluated_n": 224}}


# ------------------------------------------------------------------------------------------------------------ the oracle compare, form (B)

@left_out({})
def test_an_oracle_compare_that_states_no_attribution_is_complete_over_the_values(stated):
    assert C2._form_b_complete_over_values(without(OC_COMPLETE, ("attribution",), stated)) is None
    assert C2._form_b_complete_over_values(without(OC_COMPLETE, ("attribution", "complete"), False)) == "form (B) request attribution not complete"


@left_out({})
def test_an_oracle_compare_that_states_no_attribution_does_not_cover_the_selector_outputs(stated):
    assert C2._form_b_selector_coverage(OC_STOCH, 32) == (None, "sampled_token_ids compared 32/32 (step, request) pairs, not compared 0, 0 mismatch")
    assert C2._form_b_selector_coverage(without(OC_STOCH, ("attribution",), stated), 32) == ("form (B) request attribution not complete", "")


@left_out({})
def test_an_oracle_compare_that_counts_no_family_compared_no_sampled_id(stated):
    counted = "sampled_token_ids compared 0/32 (step, request) pairs, not compared 0, 0 mismatch"
    assert C2._form_b_selector_coverage(without(OC_STOCH, ("compared_by_family",), stated), 32) == (
        f"form (B) does not cover the selector outputs ({counted}; 32 selector VUs)", counted)


@left_out({})
def test_an_oracle_compare_that_counts_no_family_left_out_left_no_sampled_id_out(stated):
    assert C2._form_b_selector_coverage(without(OC_STOCH, ("not_compared_by_family",), stated), 32) == (
        None, "sampled_token_ids compared 32/32 (step, request) pairs, not compared 0, 0 mismatch")
    left = without(OC_STOCH, ("not_compared_by_family", "sampled_token_ids"), 1)
    assert C2._form_b_selector_coverage(left, 32)[1] == "sampled_token_ids compared 32/32 (step, request) pairs, not compared 1, 0 mismatch"


@left_out([])
def test_an_attribution_that_lists_no_ambiguous_step_counts_none(stated):
    open_attribution = {"complete": False, "pairs": 3, "unverified_n": 1, "ambiguous_steps": [4, 5]}
    note = C2._batch_attribution_note(2, without(OC_COMPLETE, ("attribution",), open_attribution))
    assert note is not None and "(pairs 3, unverified 1, ambiguous steps 2)" in note
    note = C2._batch_attribution_note(2, without(OC_COMPLETE, ("attribution",), put(open_attribution, ("ambiguous_steps",), stated)))
    assert note is not None and "(pairs 3, unverified 1, ambiguous steps 0)" in note


# ------------------------------------------------------------------------------------------------------------ the replay's population

@left_out({})
def test_a_reason_the_replay_stamped_no_code_for_is_in_no_class(stated):
    counts = {K_SEL: 32, K_ROPE: 5}
    codes = RC.not_evaluable_codes({"not_evaluable": counts})
    assert C2._no_evaluator_reasons(counts, codes) == [(K_SEL, "GumbelTopPTokenSelect_v1", 32)]
    assert C2._no_evaluator_reasons(counts, without(codes, (K_SEL,), stated)) == []


@left_out({})
def test_a_complete_replay_that_states_no_population_names_no_gap(stated):
    assert C2._complete_replay_population_gap(without(SR_COMPLETE_INTERIOR_ONLY, ("population",), stated), OC_COMPLETE) == NO_GAP


@left_out({})
def test_a_population_that_excludes_nothing_by_name_names_no_gap(stated):
    note, fail = C2._complete_replay_population_gap(SR_COMPLETE_INTERIOR_ONLY, OC_COMPLETE)
    assert fail is None and note is not None and note.startswith(f"{BY_FORM_B} over 140 VUs not evaluable by name (interior-composed-v2; MAN-06g)")
    assert C2._complete_replay_population_gap(without(SR_COMPLETE_INTERIOR_ONLY, ("population", "not_evaluable"), stated), OC_COMPLETE) == NO_GAP


@left_out({})
def test_an_excluded_reason_without_a_stamped_code_is_no_gap_the_rule_names(stated):
    replay = stamped(SR_COMPLETE_INTERIOR_ONLY)
    note, fail = C2._complete_replay_population_gap(replay, OC_COMPLETE)
    assert fail is None and note is not None and note.startswith(f"{BY_FORM_B} over 140 VUs not evaluable by name (interior-composed-v2; MAN-06g)")
    assert C2._complete_replay_population_gap(without(replay, ("population", "not_evaluable_codes", K_06G), stated), OC_COMPLETE) == NO_GAP


@left_out({})
def test_a_plane_fed_reason_without_a_stamped_code_does_not_fail_by_name(stated):
    replay = stamped(SR_COMPLETE_WITH_NAMED_EXCLUSIONS)
    note, fail = C2._complete_replay_population_gap(replay, OC_COMPLETE)
    assert note is None and fail is not None and fail.startswith(PLANE_FED)
    note, fail = C2._complete_replay_population_gap(without(replay, ("population", "not_evaluable_codes", K_06F), stated), OC_COMPLETE)
    assert fail is None and note is not None and note.startswith(f"{BY_FORM_B} over 140 VUs not evaluable by name (interior-composed-v2; MAN-06g)")


# ------------------------------------------------------------------------------------------------------------ a partial replay

def test_a_partial_replay_that_names_its_strata_passes_by_form_b():
    note, fail = C2._partial_replay_named_gap(partial_replay(), OC_COMPLETE)
    assert fail is None and note is not None
    assert note.startswith(f"{BY_FORM_B}; sampled replay partial [598 evaluated, 0 mismatch, 2 strata named not evaluated (interior-composed-v2, no-evaluator)")


@pytest.mark.parametrize(("field", "empty"), [("sample", {}), ("strata_unevaluated", []), ("unevaluated_strata", [])])
def test_a_partial_replay_that_leaves_out_its_sample_or_its_strata_names_no_gap(field, empty):
    assert [C2._partial_replay_named_gap(without(partial_replay(), (field,), stated), OC_COMPLETE) for stated in (ABSENT, None, empty)] == [NO_GAP] * 3


# ------------------------------------------------------------------------------------------------------------ the scope and the inputs

def replay_scope(replay: dict[str, Any]) -> tuple[Any, Any, Any]:
    scope = C2._replay_scope(replay)
    return scope["requests"], scope["strata"], scope["strata_unevaluated"]


def counted_replay() -> dict[str, Any]:
    """A complete replay that lists two requests, three strata and one stratum it did not evaluate."""
    replay = _base()["sampled_replay"]
    replay["population"]["requests"] = ["r0", "r1"]
    return {**replay, "strata": ["a", "b", "c"], "strata_unevaluated": ["a"]}


@left_out([])
def test_a_replay_that_lists_no_stratum_counts_the_number_it_states_or_none(stated):
    assert replay_scope(counted_replay()) == (2, 3, 1)
    assert replay_scope(without(counted_replay(), ("strata",), stated)) == (2, None, 1)
    assert replay_scope(without({**counted_replay(), "strata_n": 7}, ("strata",), stated)) == (2, 7, 1)
    by_population = without(counted_replay(), ("population", "strata_n"), 9)
    assert replay_scope(without(by_population, ("strata",), stated)) == (2, 9, 1)


@left_out([])
def test_a_replay_that_lists_no_unevaluated_stratum_counts_none(stated):
    assert replay_scope(without(counted_replay(), ("strata_unevaluated",), stated)) == (2, 3, 0)


@pytest.mark.parametrize(("stated", "requests"), [(ABSENT, None), (None, None), ([], 0), ({}, 0), ({"r0": 1}, 1)])
def test_a_population_counts_its_requests_only_where_it_lists_them(stated, requests):
    assert replay_scope(without(counted_replay(), ("population", "requests"), stated)) == (requests, 3, 1)


@left_out({})
def test_a_context_without_file_digests_gives_the_replay_no_digest(stated):
    replay, digests = _base()["sampled_replay"], {"sampled_replay_p0.json": "aa" * 32, "runs.jsonl": "bb" * 32}
    seed = {"sample_seed": replay["sample"]["seed"], "seed_source": replay["sample"]["seed_source"]}
    assert C2._replay_inputs(replay, {"file_digests": digests}) == {**seed, "sampled_replay_p0.json": "aa" * 32}
    assert C2._replay_inputs(replay, put({"file_digests": digests}, ("file_digests",), stated)) == seed


# ------------------------------------------------------------------------------------------------------------ the pins and the account

def weights_differ(commit: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    rule = next(c for c in C2.c2_rules(commit, {}) if c.name == WEIGHTS)
    return rule.outcome, rule.differences


def weights_that_differ() -> dict[str, Any]:
    commit = _base()
    commit["weights_pin"].update(result=False, mismatched=[f"w{i}" for i in range(10)], missing=["m0"], unresolved=["u0", "u1"])
    return commit


@pytest.mark.parametrize("field", ["mismatched", "missing", "unresolved"])
def test_a_weights_pin_that_lists_no_name_of_a_kind_carries_an_empty_list_for_it(field):
    listed = {"name": "weights_differ", "mismatched": [f"w{i}" for i in range(8)], "missing": ["m0"], "unresolved": ["u0", "u1"]}
    assert weights_differ(weights_that_differ()) == (V.FAIL, [listed])
    left_out_three_ways: list[Any] = [ABSENT, None, []]
    for stated in left_out_three_ways:
        assert weights_differ(put(weights_that_differ(), ("weights_pin", field), stated)) == (V.FAIL, [{**listed, field: []}])


@left_out([])
def test_a_route_pin_that_lists_no_departure_carries_an_empty_list(stated):
    commit = {**_base(), "cublas_route_pin": {"ok": False, "why": "served off the pin", "departures": [{"k": 1}], "families": {"X": 1}, "pin": "p"}}
    rule = C2.rule_route_pin(commit, {})
    assert (rule.outcome, rule.reason, rule.differences) == (V.FAIL, "served off the pin", [{"name": "served_off_route_pin", "departures": [{"k": 1}]}])
    rule = C2.rule_route_pin(put(commit, ("cublas_route_pin", "departures"), stated), {})
    assert (rule.outcome, rule.reason, rule.differences) == (V.FAIL, "served off the pin", [{"name": "served_off_route_pin", "departures": []}])


def account_link(commit: dict[str, Any]) -> tuple[str, Any]:
    rule = next(c for c in C2.execution_extent_rules(commit, {}) if c.name == "execution_extent/match_account_link")
    return rule.outcome, rule.scope["account_missing"]


@left_out([])
def test_a_request_whose_facts_name_no_missing_account_file_misses_none(stated):
    required = put(_base(), ("binding", "coverage", "match_account_link", "required"), True)
    assert account_link(put(copy.deepcopy(required), FACTS, ["tokens.json"])) == (V.INSUFFICIENT, ["tokens.json"])
    assert account_link(put(copy.deepcopy(required), FACTS, stated)) == (V.PASS, None)

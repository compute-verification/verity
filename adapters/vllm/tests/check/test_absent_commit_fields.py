"""What the Commit's rules and its v1 decision make of a field the Commit record leaves out.

A field that is absent, null or empty reads alike, with one exception the rules state: a record that carries a deferred
position's key, even empty, was written by a committer that knows positions, and one without the keys by an older one.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tests.check.test_verdict import _base, _both
from verity_vllm.check import commit_rules as CR
from verity_vllm.check import verdict as V
from verity_vllm.check.commit_pass import commit_verdict

ABSENT = object()   # a field its record leaves out, beside the null and the empty one
SCOPED = "PASS (C2 population scope (query of record Q_module (older header)): )"
NO_COUNTERS = "reference integrity counters missing (expected_tensors/committed_tensors): committer does not report them"
UNCOVERED = "manifest coverage: 1 required identities uncovered ({'instance_outputs': 1}); first: step 3 m/out inv 0 want None got None"
NOT_REPLAYED = "value correspondence FAIL (sampled-exact-replay(complete) + boundary-linkage)"
BY_COUNT = ("deferred picks recorded by count (1 challenge, 0 population) but their positions are not on the record (0, 0) -- "
            "the after-release stage cannot be checked against them")
NOT_RECEIVED = ("1 challenge + 0 population pick(s) deferred to replay; 1 position(s) not received by openings after release "
                "(first: [(3, 7)]) -- every deferred pick must be opened by replay")
NO_RELEASE_RECORD = "picks deferred to replay (excluded families) and no openings-after-release record"
OPENINGS, AFTER_RELEASE = "runtime_match/openings", "runtime_match/openings_after_release"
MANIFEST_COVERAGE, IDENTITY_LAYOUT = "required_value_coverage/manifest_coverage", "program_source_identity/identity_layout"


def put(record: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """`record` with `value` at `path`, or without the field where `value` is ABSENT."""
    at = record
    for key in path[:-1]:
        at = at[key]
    if value is ABSENT:
        at.pop(path[-1], None)
    else:
        at[path[-1]] = value
    return record


def decided(commit: dict[str, Any]) -> tuple[bool, str, str]:
    """The v1 decision and its reason, and the rules' outcome, for one instrumented run."""
    (ok, why), outcome, _ = _both(commit)
    return ok, why, outcome


def check(commit: dict[str, Any], name: str) -> CR.CheckResult:
    """The result of the rule `name` on one instrumented run."""
    return next(c for c in _both(commit)[2] if c.name == name)


def without_an_error_count() -> dict[str, Any]:
    commit = _base()
    del commit["extra"]["errors_n"]
    return commit


def one_opening_unverified() -> dict[str, Any]:
    return put(_base(), ("openings", "verified"), 63)


def one_identity_uncovered() -> dict[str, Any]:
    commit = _base()
    commit["extra"]["coverage"].update(ok=False, missing_n=1, missing_by_family={"instance_outputs": 1},
                                       missing=[{"step": 3, "op_path": "m", "output_member": "out", "invocation": 0}])
    return commit


def layout_not_ok() -> dict[str, Any]:
    return put(_base(), ("extra", "identity_check", "ok"), False)


def values_differ() -> dict[str, Any]:
    return put(_base(), ("value_correspondence", "result"), False)


def one_pick_deferred(**after_release: Any) -> dict[str, Any]:
    """A run whose challenge deferred one pick to replay, and whose openings after release received one, by count."""
    commit = put(_base(), ("openings", "deferred_replay_n"), 1)
    commit["openings_after_release"] = {"n": 64, "verified": 64, "source": "host", "deferred_received_n": 1, **after_release}
    return commit


def one_pick_deferred_at_a_position() -> dict[str, Any]:
    return put(one_pick_deferred(), ("openings", "deferred_positions"), [[3, 7]])


def one_population_pick_received() -> dict[str, Any]:
    """The population stage deferred one pick, by position, and the openings after release received it."""
    commit = _base()
    commit["binding"]["openings_population"].update(deferred_replay_n=1, deferred_positions=[[2, 5]])
    commit["openings_after_release"] = {"n": 64, "verified": 64, "source": "host", "deferred_received_positions": [[2, 5]]}
    return commit


def one_challenge_pick_received() -> dict[str, Any]:
    return put(one_pick_deferred_at_a_position(), ("openings_after_release", "deferred_received_positions"), [[3, 7]])


#: (the record, the field left out, its empty value, the decision): absent, null and empty decide alike
ALIKE: list[tuple[Callable[[], dict[str, Any]], tuple[str, ...], Any, tuple[bool, str, str]]] = [
    (without_an_error_count, ("extra", "errors"), [], (True, SCOPED, V.PASS)),
    (one_opening_unverified, ("openings", "failed"), [], (False, "openings verified 63 != checked 64", V.FAIL)),
    (_base, ("extra",), {}, (False, NO_COUNTERS, V.INSUFFICIENT)),
    (_base, ("openings",), {}, (False, "openings checked 0 < 64", V.FAIL)),
    (one_identity_uncovered, ("extra", "coverage", "families_missing_entirely"), [], (False, UNCOVERED, V.FAIL)),
    (_base, ("extra", "coverage", "steps_committed"), [], (True, SCOPED, V.PASS)),
    (values_differ, ("value_correspondence", "oracle_compare"), {}, (False, NOT_REPLAYED, V.FAIL)),
    (values_differ, ("value_correspondence", "components"), {}, (False, NOT_REPLAYED, V.FAIL)),
    (_base, ("value_correspondence",), {}, (True, SCOPED, V.PASS)),
    (_base, ("binding", "openings_population"), {}, (True, SCOPED, V.PASS)),
    (one_pick_deferred, ("binding", "openings_population"), {}, (True, SCOPED, V.PASS)),
    (one_pick_deferred_at_a_position, ("openings_after_release", "deferred_received_positions"), [], (False, NOT_RECEIVED, V.FAIL)),
    (one_population_pick_received, ("openings", "deferred_positions"), [], (True, SCOPED, V.PASS)),
    (one_challenge_pick_received, ("binding", "openings_population", "deferred_positions"), [], (True, SCOPED, V.PASS)),
]


@pytest.mark.parametrize("make, path, empty, decision", ALIKE)
def test_a_field_the_commit_record_leaves_out_null_or_empty_decides_alike(make, path, empty, decision):
    assert [decided(put(make(), path, value)) for value in (ABSENT, None, empty)] == [decision] * 3


@pytest.mark.parametrize("path", [("openings", "deferred_positions"), ("binding", "openings_population", "deferred_positions"),
                                  ("openings_after_release", "deferred_received_positions")])
@pytest.mark.parametrize("stated, decision", [(ABSENT, (True, SCOPED, V.PASS)), (None, (False, BY_COUNT, V.FAIL)), ([], (False, BY_COUNT, V.FAIL))])
def test_a_deferred_positions_key_tells_a_record_that_knows_positions_from_an_older_one(path, stated, decision):
    commit = put(one_pick_deferred(), path, stated)
    assert decided(commit) == decision
    assert check(commit, AFTER_RELEASE).differences == ([] if decision[0] else [
        {"name": "deferred_positions_missing", "challenge": 1, "population": 0, "positions": [0, 0]}])


def test_a_pick_deferred_without_a_record_of_the_openings_after_release_is_not_enough():
    commit = _base()
    commit["binding"]["openings_population"]["deferred_replay_n"] = 1
    assert decided(commit) == (False, NO_RELEASE_RECORD, V.INSUFFICIENT)
    assert check(commit, AFTER_RELEASE).differences == [{"name": "deferred_picks_unopened", "challenge": 0, "population": 1}]


@pytest.mark.parametrize("value", [ABSENT, None, {}])
def test_under_a_floor_of_no_openings_a_record_without_openings_passes_where_no_stage_deferred_a_pick(value):
    commit = put(_base(), ("openings",), value)
    assert commit_verdict(commit, True, min_openings=0) == (True, SCOPED)
    checks = V.run_checks(commit, True, min_openings=0)
    assert [(c.outcome, c.differences) for c in checks if c.name in (OPENINGS, AFTER_RELEASE)] == [(V.PASS, []), (V.NOT_RUN, [])]
    commit["binding"]["openings_population"]["deferred_replay_n"] = 1
    assert commit_verdict(commit, True, min_openings=0) == (False, NO_RELEASE_RECORD)


@pytest.mark.parametrize("stated, decision", [(ABSENT, (True, "PASS", V.PASS)), (None, (True, "PASS", V.PASS)),
                                              ({}, (False, "binding-map context: None", V.FAIL))])
def test_a_record_without_a_binding_block_is_decided_by_the_older_rule_and_an_empty_block_is_refused(stated, decision):
    commit = put(one_pick_deferred(), ("binding",), stated)
    assert decided(commit) == decision and check(commit, AFTER_RELEASE).outcome == V.PASS


@pytest.mark.parametrize("value", [ABSENT, None, []])
def test_the_rules_carry_an_empty_list_for_what_the_record_does_not_list(value):
    unverified = check(put(one_opening_unverified(), ("openings", "failed"), value), OPENINGS)
    assert unverified.differences == [{"name": "openings_unverified", "checked": 64, "verified": 63, "failed": []}]
    uncovered = check(put(one_identity_uncovered(), ("extra", "coverage", "families_missing_entirely"), value), MANIFEST_COVERAGE)
    assert uncovered.differences[0]["families_missing_entirely"] == [] and uncovered.differences[0]["missing_n"] == 1
    assert check(put(_base(), ("extra", "coverage", "steps_committed"), value), MANIFEST_COVERAGE).scope["steps_committed_n"] == 0
    assert check(put(without_an_error_count(), ("extra", "errors"), value), "runtime_match/collector_errors").scope == {"errors_n": 0}
    layout = check(put(layout_not_ok(), ("extra", "identity_check", "violations"), value), IDENTITY_LAYOUT)
    assert (layout.outcome, layout.differences) == (V.FAIL, [])


def test_the_lists_a_record_states_are_carried():
    unverified = check(put(one_opening_unverified(), ("openings", "failed"), list(range(10))), OPENINGS)
    assert unverified.differences[0]["failed"] == list(range(8))
    uncovered = check(put(one_identity_uncovered(), ("extra", "coverage", "families_missing_entirely"), ["a", "b"]), MANIFEST_COVERAGE)
    assert uncovered.differences[0]["families_missing_entirely"] == ["a", "b"] and "whole families missing: ['a', 'b']; " in decided(
        put(one_identity_uncovered(), ("extra", "coverage", "families_missing_entirely"), ["a", "b"]))[1]
    assert check(put(_base(), ("extra", "coverage", "steps_committed"), [0, 1, 2]), MANIFEST_COVERAGE).scope["steps_committed_n"] == 3
    errors = put(without_an_error_count(), ("extra", "errors"), ["step 3: staging refused", "step 4: late"])
    assert decided(errors) == (False, "2 collector error(s): step 3: staging refused", V.FAIL)
    violations = [{"step": 2, "missing_modules": ["m"]}, "not a violation"]
    assert check(put(layout_not_ok(), ("extra", "identity_check", "violations"), violations), IDENTITY_LAYOUT).differences == [
        {"name": "missing_modules", "steps": [2], "steps_n": 1, "first": {"step": 2, "missing_modules": ["m"]}},
        {"name": "sampler_ids", "steps": [2], "steps_n": 1, "first": {"step": 2, "sampler_ids": None}}]


@pytest.mark.parametrize("manifest", [{}, {"populations": None}, {"populations": {}}, {"populations": {"by_family": None}}, {"populations": {"by_family": {}}}])
def test_a_manifest_that_states_no_population_by_family_names_no_hidden_class(manifest):
    assert CR.hidden_classes_required(manifest) is False


def test_a_manifest_names_a_hidden_class_by_its_count_and_no_manifest_says_nothing():
    assert CR.hidden_classes_required(None) is None
    assert CR.hidden_classes_required({"populations": {"by_family": {"fa3_hidden_m1_stream": 2}}}) is True
    assert CR.hidden_classes_required({"populations": {"by_family": {"fa3_hidden_m1_stream": 0, "fa2_hidden_m1_stream": None}}}) is False


@pytest.mark.parametrize("declaration", [None, {}, {"classes": None}, {"classes": []}])
def test_a_declaration_that_lists_no_class_covers_none(declaration):
    assert CR.class_coverage(declaration) == {}
    assert CR.class_coverage({"classes": [{"class": "weights", "how": "registered-input", "tensors": 1}]}) == {
        "weights": {"how": "registered-input", "tensors": 1, "bytes": None}}


@pytest.mark.parametrize("stated", [ABSENT, None, {}])
def test_a_binding_that_states_no_population_openings_passes_on_its_coverage(stated):
    assert CR.binding_check(put(_base()["binding"], ("openings_population",), stated)) == (True, "PASS")

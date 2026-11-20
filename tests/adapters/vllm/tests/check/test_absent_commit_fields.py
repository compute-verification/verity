"""What the Commit's rules make of a field the Commit record leaves out: absent, null and empty read alike."""
from __future__ import annotations

from typing import Any

import pytest
from verity_vllm.check import commit_rules as CR

from tests.check.test_verdict import _base

ABSENT = object()   # a field its record leaves out, beside the null and the empty one


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


#: every value that tests false: each is nothing stated to a reader that takes a value by its truth
NOTHING: tuple[Any, ...] = (None, {}, [], 0, "", False)


def left_out(empty: Any) -> pytest.MarkDecorator:
    """A field absent, null and empty (`empty`: the empty value of its kind), and every other value that tests false.  The readers
    take a field by its truth: one that took only an absent or a null field as nothing stated would answer the others differently."""
    others = {"zero": 0, "no-text": "", "false": False, "no-item": [], "no-key": {}}
    names = [name for name, value in others.items() if type(value) is not type(empty)]
    return pytest.mark.parametrize("stated", [ABSENT, None, empty, *(others[name] for name in names)], ids=["absent", "null", "empty", *names])

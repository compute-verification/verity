"""The input nodes a parameter's type lowers to (`verity_vllm.pipeline.registration_log.lower`): a type of one width is one node of
all its leaves; a mixed tuple or record gives its fields' nodes in order, a mixed array repeats its element's, and a mixed type of
no such form is refused."""
from __future__ import annotations

import pytest
from verity.circuits.types import Array, RecordType, Tuple, Type, Value

import verity_vllm.pipeline.registration_log as RL

PAIR = Tuple(Value(8), Array(2, Value(32)))
RECORD = RecordType(("a", Value(16)), ("b", PAIR))


def test_a_type_of_one_width_is_one_node_of_all_its_leaves():
    assert RL.lower(Value(16)) == [(16, 1)] and RL.lower(Array(4, Value(8))) == [(8, 4)]
    assert RL.lower(Tuple(Value(8), Array(3, Value(8)))) == [(8, 4)] and RL.lower(Array(2, Tuple(Value(8), Value(8)))) == [(8, 4)]


def test_a_mixed_type_lowers_field_by_field_and_a_mixed_array_repeats_its_elements_nodes():
    assert RL.lower(PAIR) == [(8, 1), (32, 2)]
    assert RL.lower(Array(3, PAIR)) == [(8, 1), (32, 2), (8, 1), (32, 2), (8, 1), (32, 2)]
    assert RL.lower(RECORD) == [(16, 1), (8, 1), (32, 2)]
    assert RL.lower(Tuple(RECORD, Array(2, Value(16)), Array(2, PAIR))) == [(16, 1), (8, 1), (32, 2), (16, 2), (8, 1), (32, 2), (8, 1), (32, 2)]


def test_a_type_with_no_leaf_lowers_to_no_node_and_a_mixed_type_of_no_known_form_is_refused():
    assert RL.lower(Tuple()) == [] and RL.lower(Array(0, PAIR)) == []

    class Odd(Type):
        @property
        def leaves(self) -> int:
            return 2

        def width_set(self) -> frozenset:
            return frozenset({8, 16})

        def __repr__(self) -> str:
            return "Odd"
    with pytest.raises(ValueError, match="^a parameter of type Odd lowers to no input nodes$"):
        RL.lower(Odd())

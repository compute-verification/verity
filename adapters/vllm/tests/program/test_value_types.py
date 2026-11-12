"""`value_types.flat_types`: the tensors of a type, in leaf order. Needs torch (the module reads its dtypes); skips cleanly
without it."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verity.circuits.types import Array, RecordType, Value
from verity.circuits.types import Tuple as TupleT

from verity_vllm.program.frontend.value_types import flat_types

V16 = Value(16)
ROW = Array(3, V16)


def test_a_tensor_is_one_chunk_whatever_its_rank():
    assert flat_types(V16) == [V16]
    assert flat_types(Array(2, ROW)) == [Array(2, ROW)]


def test_a_tuple_and_a_record_give_their_fields_tensors_in_order():
    assert flat_types(TupleT(ROW, V16)) == [ROW, V16]
    assert flat_types(RecordType(("a", V16), ("b", TupleT(ROW, V16)))) == [V16, ROW, V16]


def test_an_array_of_tuples_repeats_its_elements_tensors():
    assert flat_types(Array(2, TupleT(ROW, V16))) == [ROW, V16, ROW, V16]

"""`query.type_repr`: the `repr` of a `verity.circuits.types.Type` reads back to the Type, for each form a Program's port types
take and for one inside another, and a text that is no Type's repr is refused."""
from __future__ import annotations

import pytest
from verity.circuits.types import Array, RecordType, Value
from verity.circuits.types import Tuple as TupleT

from verity_vllm.query.type_repr import TypeReprError, parse_type_repr

V32, BIT = Value(32), Value(1)
TYPES = (V32, Array(31, V32), Array(0, BIT), TupleT(), TupleT(V32), TupleT(V32, Array(2, BIT)), TupleT(TupleT(V32), V32),
         RecordType(("a", V32), ("b", Array(3, V32))), Array(2, TupleT(V32, RecordType(("s", Value(8))))),
         RecordType(("t", TupleT(BIT, BIT)), ("u", RecordType(("v", V32)))))


@pytest.mark.parametrize("t", TYPES, ids=repr)
def test_a_types_repr_reads_back_to_the_type(t) -> None:
    assert parse_type_repr(repr(t)) == t


def test_the_space_between_the_parts_is_skipped() -> None:
    assert parse_type_repr(" ( Value<32>,\n\tArray<2, Value<1> > ) ") == TupleT(V32, Array(2, BIT))
    assert parse_type_repr("{ a : Value<32> ,b:Value<1>}") == RecordType(("a", V32), ("b", BIT))


@pytest.mark.parametrize(("text", "why"), [
    ("Array<2,Value<32>", r"expected '>' at 17 in 'Array<2,Value<32>'"),                 # the text ends where the array closes
    ("Array<2,Value<32>)", r"expected '>' at 17"),
    ("Arr<2>", r"unexpected 'Arr<2>' at 0"),
    ("(Value<32>, value<1>)", r"unexpected 'value<1>\)' at 12"),
    ("Value<32> x", r"trailing 'x'"),
    ("(Value<32>)(", r"trailing '\('"),
])
def test_a_text_that_is_no_types_repr_is_refused(text: str, why: str) -> None:
    with pytest.raises(TypeReprError, match=why):
        parse_type_repr(text)

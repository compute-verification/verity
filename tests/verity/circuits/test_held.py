"""A root's return held to a supplied input (PROTOCOL §5.4): the decoder reads a held row `[lo, hi, {"equals": [input,
at, n]}]` of the root's `ports.ret` into `ports.PortTypes.held`, writes it back, and refuses one `supplied.check_held`
refuses (`E_HELD`), one not of that form, one off the root and overlapping ones (`E_PORT_DECL`)."""

from __future__ import annotations

import copy

import pytest

from verity.circuits import codec, ports as P
from verity.circuits.defs import REGISTRY, CompositeDefinition, bind
from verity.circuits.program import Program
from verity.circuits.types import Array, Value

ONE = {"supplied": "const.uint/v1", "args": {"bits": 16, "value": 1}, "layout": "le", "bits": 16}
BITS16 = Array(16, Value(1))
INNER = bind(CompositeDefinition("HeldInner", 1, (), lambda S: ((("x", BITS16),), BITS16), lambda B, S, x: x,
                                 register=False))
ROOT = bind(CompositeDefinition("HeldRoot", 1, (), lambda S: ((("a", BITS16), ("one", BITS16)), BITS16),
                                lambda B, S, a, one: B.call(INNER, a), register=False))
DESC = codec.encode_program(Program(ROOT, supplied={"one": ONE}))


def held(rows, desc=DESC) -> dict:
    j = copy.deepcopy(desc)
    j["definitions"][j["root"]]["ports"] = {"ret": rows}
    return j


def test_a_verdict_held_to_the_supplied_one_decodes_and_writes_back():
    j = held([[0, 16, {"equals": ["one", 0, 16]}]])
    root = codec.decode_program(j, REGISTRY).root
    assert P.port_types(root).held == ((0, 16, "one", 0, 16),)
    assert P.port_types(root).to_json() == j["definitions"][j["root"]]["ports"]


@pytest.mark.parametrize("rows, error, text", [
    ([[0, 16, {"equals": ["a", 0, 16]}]], "E_HELD",
     "E_HELD: output row [0, 16) of 1-bit leaves: held to ['a', 0, 16], which is no run of leaves of a supplied input "
     "of that width whose length divides the row's"),
    ([[0, 16, {"equals": ["one", 8, 16]}]], "E_HELD",
     "E_HELD: output row [0, 16) of 1-bit leaves: held to ['one', 8, 16], which is no run of leaves of a supplied input "
     "of that width whose length divides the row's"),
    ([[0, 16, {"equals": ["one", 0]}]], "E_PORT_DECL",
     "E_PORT_DECL: Root[HeldRoot_v1]_v1 return: a held row is [lo, hi, {\"equals\": [input, at, n]}] inside the return, "
     "not [0, 16, {'equals': ['one', 0]}]"),
    ([[0, 8, {"equals": ["one", 0, 8]}], [4, 16, {"equals": ["one", 0, 4]}]], "E_PORT_DECL",
     "E_PORT_DECL: Root[HeldRoot_v1]_v1: its held return rows overlap"),
], ids=["an unmarked input", "a run past the input", "not the held form", "overlapping rows"])
def test_refusals(rows, error, text):
    with pytest.raises(Exception) as e:
        codec.decode_program(held(rows), REGISTRY)
    assert str(e.value) == text


def test_only_the_root_holds_an_output():
    j = copy.deepcopy(DESC)
    callee = next(k for k in j["definitions"] if "HeldInner" in k)
    j["definitions"][callee]["ports"] = {"ret": [[0, 1, {"equals": ["one", 0, 1]}]]}
    with pytest.raises(Exception, match="only the root holds an output to a supplied input"):
        codec.decode_program(j, REGISTRY)

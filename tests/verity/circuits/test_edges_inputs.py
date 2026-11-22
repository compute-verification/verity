"""`edges`' input nodes are `program.input_id`'s, Lean's `Flock.Program.inputId?`: a run, a registered and a supplied
input node are inputs of the root, so a root reading a supplied value gets its cuts; a template node is none, and a
template node no range names is still refused as ``edge-template``."""

from __future__ import annotations

import pytest

from verity.circuits import codec
from verity.circuits import edges as EG
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.types import Array, Value

XOR16 = PrimitiveDefinition("EdgesInputsXor16", 1, (("a", Value(16)), ("b", Value(16))), Value(16), lambda a, b: a ^ b,
                            register=False)
ROW = Array(2, Value(16))
A = bind(CompositeDefinition("EdgesInputsA", 0, (), lambda S: ((("x", ROW),), ROW),
                             lambda B, S, x: B.batch(XOR16, x, x, axes=(0, 0)), register=False))
T = bind(CompositeDefinition(
    "EdgesInputsT", 0, (), lambda S: ((("x", ROW), ("r", Value(16)), ("s", Value(16))), ROW),
    lambda B, S, x, r, s: B.batch(XOR16, B.batch(XOR16, x, r, axes=(0, None)), s, axes=(0, None)), register=False))


def _root():
    """A root batching A (2 members) over its run input and T (2) over A's outputs, its registered and its supplied
    input, its descriptor and the ranges naming both templates' nodes."""
    def body(B, S, x, r, s):
        return B.batch(T, B.batch(A, x, axes=(0,)), r, s, axes=(0, None, None))

    root = bind(CompositeDefinition("EdgesInputsRoot", 0, (),
                                    lambda S: ((("x", Array(2, ROW)), ("r", Value(16)), ("s", Value(16))), Array(2, ROW)),
                                    body, register=False))
    desc = codec.encode_program(Program(root, registered=("r",), supplied=("s",)))
    ids = desc["callees"]
    nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
    at = {ids[x["fn"]]: k for k, x in enumerate(nodes)}
    return desc, [ids[x["fn"]] for x in nodes], [[at[codec.descriptor_id(d)], codec.descriptor_id(d), 0, 2] for d in (A, T)]


def test_a_run_a_registered_and_a_supplied_input_node_are_inputs_and_a_template_node_is_none():
    desc, ids, _ = _root()
    flags = {i.split("_v")[0].rstrip("0123456789"): n["input"] for i, n in zip(ids, EG._Root(desc).nodes)}
    assert flags == {"Input": True, "RegisteredInput": True, "SuppliedInput": True, "EdgesInputsA": False,
                     "EdgesInputsT": False}


def test_a_root_reading_a_supplied_input_gets_its_cuts():
    desc, _, ranges = _root()
    assert EG.io_cuts(desc, ranges) == {}


def test_a_template_node_no_range_names_is_refused():
    desc, _, ranges = _root()
    with pytest.raises(ValueError, match="edge-template"):
        EG.io_cuts(desc, ranges[:1])

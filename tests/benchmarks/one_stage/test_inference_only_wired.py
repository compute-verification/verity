"""`inference_only.py` on a wired Program of the form a slot-grouped `wired_population_program` gives: a root calling one
Definition a slot, each holding its slot's template batches and the wires between them. The count's reference
(`inference_only_reference.evaluate_count`, held to the Lean's `InferenceOnlyCheck.Count` by its difftest) counts a row a
wire inside the slot fills as computed in the block, and the policy's k (`wired_k`) is that same count with every weight
held, split by kind with the opened rows a kind of their own; a block that takes in more than k, a weight left unheld or a
wire from another slot, is refused ``fresh``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "one_stage"))

import inference_only as IO  # noqa: E402
from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind  # noqa: E402
from verity.circuits.types import Array, Value  # noqa: E402
from verity_vllm.program.predict import scan_state  # noqa: E402

V16 = Value(16)
A2 = Array(2, V16)
Add = PrimitiveDefinition("TSlotAdd16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a + b) & 0xFFFF, register=False)
Mul = PrimitiveDefinition("TSlotMul16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a * b + 1) & 0xFFFF,
                          register=False)
TABLE = {descriptor_id(Add): 16, descriptor_id(Mul): 100}
#: the root's values: the weight (θ), a Pearl+ y (``linears``), the scan state (``scan_state``), an opened row
#: (``openings``), each two words a slot
Y, SCAN, OPENED = "pearl-y/0/0", scan_state.SCAN_STATE, "tbl/row"
RECORD = {"registered": {"tables": [], "roots": [], "openings": [{"table": "tbl", "ports": ["row"]}]}}


def _template(name, prim):
    return bind(CompositeDefinition(name, 1, (), lambda S: ((("a", A2), ("b", A2)), A2),
                                    lambda B, S, a, b: B.batch(prim, a, b, axes=(0, 0)), register=False))


TA, TB = _template("TSlotA", Add), _template("TSlotB", Mul)


def _slot(name, crossing=False):
    """A slot: ``a = TA(y, s)``, ``b = TB(a, w)`` (a wire inside the slot), ``TA(b, o)``; with ``crossing``, TB reads
    ``prev``, another slot's output, in place of its own ``a``."""
    params = (("y", A2), ("s", A2), ("w", A2), ("o", A2)) + ((("prev", A2),) if crossing else ())

    def body(B, S, y, s, w, o, *prev):
        a = B.call(TA, y, s)
        return B.call(TA, B.call(TB, prev[0] if crossing else a, w), o)
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, A2), body, register=False))


SLOT, CROSS = _slot("TSlotDef"), _slot("TSlotCross", crossing=True)
VALUES = (("w", Array(2, A2)), (Y, Array(2, A2)), (SCAN, Array(2, A2)), (OPENED, Array(2, A2)))


def _root(crossing=False):
    """Two slots of one Definition called as a batch; with ``crossing``, slot 1 is `CROSS` reading slot 0's output."""
    def body(B, S, w, y, s, o):
        if not crossing:
            return B.batch(SLOT, y, s, w, o, axes=(0, 0, 0, 0))
        first = B.call(SLOT, y[0], s[0], w[0], o[0])
        return B.call(CROSS, y[1], s[1], w[1], o[1], first)
    ret = A2 if crossing else Array(2, A2)
    return bind(CompositeDefinition("TSlotRoot" + ("Cross" if crossing else ""), 1, (), lambda S: (VALUES, ret), body,
                                    register=False))


KINDS = {"w": None, Y: IO.wired_kind(Y, RECORD), SCAN: IO.wired_kind(SCAN, RECORD), OPENED: IO.wired_kind(OPENED, RECORD)}


def _count(fn, slots, k, held=("w",)):
    return IO._reference().evaluate_count(IO.wired_instance(fn, slots, TABLE, 10**9, k, 10**9, held))


def test_a_wire_inside_the_slot_is_computed_in_the_block_and_k_is_the_counts_own_rule():
    assert KINDS == {"w": None, Y: "linears", SCAN: "scan_state", OPENED: "openings"}
    root = _root()
    got = IO.wired_k(root, [descriptor_id(SLOT)], TABLE, ("w",), KINDS)
    # y, s and o, two 16-bit words each: TB's a is TA's output inside the block, and w is θ
    assert got == {"fresh": 96, "k_by_kind": {"linears": 32, "scan_state": 32, "openings": 32}}
    out = _count(root, [descriptor_id(SLOT)], got["fresh"])
    assert out["blocks"] == [[descriptor_id(SLOT), 2 * 16 + 2 * 100 + 2 * 16, 96, 2]] and out["complies"]


def test_a_block_past_k_is_refused_fresh_a_weight_left_unheld():
    root = _root()
    k = IO.wired_k(root, [descriptor_id(SLOT)], TABLE, ("w",), KINDS)["fresh"]
    out = _count(root, [descriptor_id(SLOT)], k, held=())
    assert out["blocks"][0][2] == k + 32 and out["failure"] == [0, "fresh"]


def test_a_wire_from_another_slot_is_a_fresh_input_of_the_block_that_reads_it():
    root = _root(crossing=True)
    keys = [descriptor_id(SLOT), descriptor_id(CROSS)]
    out = _count(root, keys, 96)
    fresh = {key: x for key, _, x, _ in out["blocks"]}
    assert fresh == {keys[0]: 96, keys[1]: 96 + 32} and out["failure"] == [1, "fresh"]


def test_the_policys_kinds_must_sum_to_k():
    with pytest.raises(ValueError, match="no kind"):
        IO.wired_k(_root(), [descriptor_id(SLOT)], TABLE, ("w",), {**KINDS, OPENED: None})

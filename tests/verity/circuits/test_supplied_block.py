"""A supplied input (PROTOCOL.md §5.4) through the descriptor and `restrict`: C's `inputs` entry carries its mark, and a
block reading part of it takes a parameter bound `supplied` whose mark names C's input and the leaves it covers."""

from __future__ import annotations

import json
import random

import pytest

from verity.circuits import boolean as BL
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks, descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.evaluation.reference import evaluate_region
from verity.circuits.program import Program, input_id
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple

A4, A16 = Array(4, BL.BIT), Array(16, BL.BIT)
MARK = {"supplied": "const.uint/v1", "args": {"bits": 16, "value": 5}, "layout": "le", "bits": 16}


def _t_body(B, S, a, k):
    return array_of([B.call(BL.XOR, BL.operands(a[i], B.call(BL.AND, BL.operands(k[i], a[(i + 1) % 4])))) for i in range(4)])


T = bind(CompositeDefinition("SuppliedBlockToyT", 1, (), lambda S: ((("a", A4), ("k", A4)), A4), _t_body, register=False))
PARTITION = {"query": {"name": "Q_template_instances", "version": 0, "params": {"templates": [descriptor_id(T)]}}}


def program(supplied=None) -> Program:
    """x, then k: unit 0 reads k's leaves 0..3, unit 1 its leaves 4..7."""
    def body(B, S, x, k):
        u0 = B.call(T, x, array_of([k[i] for i in range(4)]))
        return tuple_of(B.call(T, u0, array_of([k[i] for i in range(4, 8)])))
    fn = CompositeDefinition("SuppliedBlockToy", 1, (), lambda S: ((("x", A4), ("k", A16)), Tuple(A4)), body, register=False)
    return Program(fn, supplied={"k": MARK} if supplied is None else supplied)


def bit(t: dict[int, int], bits: R.Bits, x: int) -> int:
    g = bits.gate_of(x)
    return (t[g] >> bits.out[g].index(x)) & 1


def test_cs_entry_carries_the_mark_and_a_block_its_part():
    p = program()
    assert p.inputs == [["x", 1], ["k", 1, MARK]]
    c = b"".join(descriptor_chunks(p))
    assert json.loads(c)["inputs"] == [["x", 1], ["k", 1, MARK]]
    bits = R.Bits(R.program(c))
    b = R.restrict(c, PARTITION, [(1, 2)], bits)
    d = json.loads(b.descriptor)
    assert d["inputs"] == [["k/p0", 1, {**MARK, "bits": 4, "of": ["k", 4, 8]}], ["p1", 1]]
    assert any(k.startswith("SuppliedInput1_v1") for k in d["definitions"])
    R.program(b.descriptor)
    C, t, rng = p.circuit, {}, random.Random(1)
    for g in C.gates():
        if input_id(g.prim.id) is not None:
            t[g.index] = rng.getrandbits(1)
    evaluate_region(C, C[0:C.size], t)
    text = b.listing()
    assert R.evaluate(text, [bit(t, bits, x) for x in b.input_bits]) == [bit(t, bits, x) for x in b.output_bits]
    assert text.count(b"supplied") == 4


def test_a_mark_its_nodes_do_not_hold_is_refused():
    with pytest.raises(Exception, match="E_SUPPLIED_WIDTH"):
        R.program(b"".join(descriptor_chunks(program({"k": {**MARK, "supplied": "log.base/v1", "args": {}}}))))


def test_registered_and_supplied_at_once_is_refused():
    with pytest.raises(ValueError, match="registered and supplied"):
        Program(CompositeDefinition("SuppliedBlockToy2", 1, (), lambda S: ((("k", A16),), A16), lambda B, S, k: k,
                                    register=False), registered=["k"], supplied={"k": MARK})

"""The Boolean profile (`verity/ir/PROTOCOL.md` §9): the three gates and the constant bits, the word view, and a small Boolean
program pinned byte for byte with its gate table and truth table (`boolean_vectors.json`).

`python packages/verity/tests/ir/test_boolean.py --write` rewrites the vectors from the reference implementation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity.ir.evaluate import evaluate_call
from verity.ir.program import Program
from verity.ir.refs import Coll, concat
from verity.ir.types import Array, Record, Tuple, Value, type_from_json

PATH = Path(__file__).parent / "boolean_vectors.json"
BIT = BL.BIT


def _cat(t, *cs) -> Coll:
    return Coll(t, concat([c.refs for c in cs]))


def _full_adder(B, S, a, b, c):
    t = B.call(BL.XOR, a, b)
    s = B.call(BL.XOR, t, c)
    carry = B.call(BL.XOR, B.call(BL.AND, a, b), B.call(BL.AND, t, c))
    return _cat(Array(2, BIT), s, carry)


FULL_ADDER = CompositeDefinition("ExampleFullAdder", 1, (), lambda S: ((("a", BIT), ("b", BIT), ("c", BIT)), Array(2, BIT)),
                                 _full_adder, register=False)


def _add2(B, S, x, y):
    """x + y for 2-bit elements: a batch of Xor gates, two calls of the full adder (carry in: the constant 0), and a NOT of the
    constant 0 as a fourth, always-set output bit."""
    _ = B.batch(BL.XOR, x, y, axes=(0, 0))
    zero = B.call(BL.constant(0))
    lo = B.call(FULL_ADDER, x[0], y[0], zero)
    hi = B.call(FULL_ADDER, x[1], y[1], lo[1])
    one = B.call(BL.NOT, zero)
    return _cat(Array(4, BIT), lo[0], hi[0], hi[1], one)


ADD2 = CompositeDefinition("ExampleAdd2", 1, (), lambda S: ((("x", Array(2, BIT)), ("y", Array(2, BIT))), Array(4, BIT)), _add2,
                           register=False)
PROGRAM = CompositeDefinition("ExampleAdd2Program", 1, (), lambda S: ADD2.signature(S), lambda B, S, x, y: B.call(ADD2, x, y),
                              doc="one call of ExampleAdd2 over Input gates", register=False)

WORD_VIEW_TYPES = [BIT, Array(16, BIT), Array(4, Array(16, BIT)), Tuple(Array(32, BIT), BIT, Array(2, Array(8, BIT))),
                   Record(("acc", Array(32, BIT)), ("mask", Array(3, BIT)))]


def _words(bits: list[int], widths: list[int]) -> list[int]:
    out, i = [], 0
    for w in widths:
        out.append(sum(b << j for j, b in enumerate(bits[i:i + w])))
        i += w
    return out


def build() -> dict:
    prog = Program(bind(PROGRAM))
    desc = codec.encode_program(prog)
    C = prog.circuit
    gates = [[g.index, g.prim.id, list(g.operands)] for g in map(C.gate, range(C.size))]
    (call,) = [c for c in C.root_calls() if not c.is_primitive]
    table = []
    for x in range(4):
        for y in range(4):
            bits = [(x >> j) & 1 for j in range(2)] + [(y >> j) & 1 for j in range(2)]
            out = evaluate_call(C, call, bits)
            table.append([[x, y], _words(list(out), BL.elements(Array(4, BIT)))])
    return {
        "about": "verity/ir/PROTOCOL.md §9: written by test_boolean.py --write",
        "gates": [[g.id, BL.KINDS[g.id][0], [t.to_json() for _, t in g.params], g.ret.to_json(),
                   [[list(a), g.evaluate(*a)] for a in ([(0,), (1,)] if len(g.params) == 1 else [(0, 0), (0, 1), (1, 0), (1, 1)])]]
                  for g in (BL.AND, BL.XOR, BL.NOT)],
        "constants": [[c.id, c.evaluate()] for c in BL.CONSTANTS],
        "word_view": [[t.to_json(), BL.elements(t), BL.word_type(t).to_json()] for t in WORD_VIEW_TYPES],
        "program": {"descriptor": codec.canonical_json(desc).decode(), "program_digest": codec.program_digest(desc),
                    "is_boolean": BL.is_boolean(prog.root), "gates": gates, "words": table},
    }


@pytest.fixture(scope="module")
def pinned() -> dict:
    return json.loads(PATH.read_text())


def test_vectors_are_current(pinned):
    assert build() == pinned


def test_gates_are_registered_and_reserved():
    for g in (BL.AND, BL.XOR, BL.NOT, *BL.CONSTANTS):
        assert REGISTRY[g.id] is g
    with pytest.raises(ValueError):
        PrimitiveDefinition("And", 1, (("a", BIT), ("b", BIT)), BIT, lambda a, b: a | b)


def test_a_constant_bit_is_0_or_1():
    assert [c.evaluate() for c in BL.CONSTANTS] == [0, 1]
    with pytest.raises(ValueError):
        BL.constant(2)


def test_the_word_view(pinned):
    for t_json, widths, word in pinned["word_view"]:
        t = type_from_json(t_json)
        assert BL.elements(t) == widths and sum(widths) == t.leaves
        assert BL.word_type(t).to_json() == word
        assert BL.bit_type(BL.word_type(t)) == t
    with pytest.raises(TypeError):
        BL.elements(Value(16))


def test_the_program_decodes_and_evaluates(pinned):
    desc = json.loads(pinned["program"]["descriptor"])
    D = codec.decode_program(desc, REGISTRY)
    assert codec.program_digest(desc) == pinned["program"]["program_digest"]
    assert BL.is_boolean(D.root)
    assert [[g.index, g.prim.id, list(g.operands)] for g in map(D.circuit.gate, range(D.circuit.size))] == pinned["program"]["gates"]
    (call,) = [c for c in D.circuit.root_calls() if not c.is_primitive]
    for (x, y), (out,) in pinned["program"]["words"]:
        assert out == x + y + 8
        bits = [(x >> j) & 1 for j in range(2)] + [(y >> j) & 1 for j in range(2)]
        assert _words(list(evaluate_call(D.circuit, call, bits)), [4]) == [out]


def test_is_boolean_refuses_a_word_primitive():
    word = PrimitiveDefinition("ExampleWordNeg", 1, (("a", Value(16)),), Value(16), lambda a: a ^ 0x8000, register=False)
    assert not BL.is_boolean(word)
    assert BL.is_boolean(BL.AND) and BL.is_boolean(bind(ADD2))


if __name__ == "__main__" and "--write" in sys.argv:
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH}")

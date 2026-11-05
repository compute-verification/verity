"""`boolean.build`: a hand-built body's constant bits, its references as a list and back (contiguous runs as `Affine`, as a
traced body's are), and a Definition's AND gates as the census counts them."""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.build import Consts, ands, coll, refs
from verity.ml.circuits.boolean.forms import and_reduce
from verity.ml.circuits.census import BooleanWork
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array

BIT = BL.BIT


def test_a_list_of_references_becomes_a_collection_with_its_runs_affine_and_reads_back_as_the_same_list() -> None:
    run = [("node", 3, j) for j in range(5)]
    loose = [("node", 1, 0), ("node", 2, 7)]
    assert coll(Array(5, BIT), run).refs.to_json() == {"affine": ["node", 3, 0, 5]}
    assert coll(Array(2, BIT), loose).refs.to_json() == {"explicit": [list(r) for r in loose]}
    for leaves in (run, loose, run + loose):
        assert refs(coll(Array(len(leaves), BIT), leaves)) == leaves


def test_a_body_reads_each_constant_bit_from_one_node_and_computes_with_it() -> None:
    def body(B, S, x):
        K = Consts(B)
        one = K.bits(1, 1)
        assert K.bits(0b101, 3) == [one[0], K.bits(0, 1)[0], one[0]]
        return B.call(BL.and_gate(3), coll(Array(3, BIT), refs(x) + one))

    d = bind(CompositeDefinition("BuildConstsProbe", 1, (), lambda S: ((("x", Array(2, BIT)),), BIT), body, register=False))
    consts = [nd for nd in d.body.nodes if BL.is_constant_bit(nd.fn)]
    assert sorted(nd.fn.id for nd in consts) == ["Const1[0x0]_v1", "Const1[0x1]_v1"]
    assert [int(evaluate_bits(d, [[a, b]]).reshape(-1)[0]) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))] == [0, 0, 0, 1]
    assert ands(d) == 2


def test_ands_counts_a_traced_definitions_and_gates_and_shares_one_count_across_calls() -> None:
    d = bind(TR.definition("BuildAndsProbe", 1, lambda C, x: [and_reduce(C, x)], [("x", Array(4, BIT))], Array(1, BIT),
                           register=False))
    work = BooleanWork()
    assert ands(d) == ands(d, work) == ands(d, work) == 3
    assert ands(BL.and_gate(5)) == 4 and ands(BL.XOR) == 0 and ands(BL.NOT) == 0

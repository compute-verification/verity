"""`program_view.from_descriptor` on a request Build's descriptor, whose program function is inlined into the root (circuits,
VLLM: the lookup of `program_fn` among the definitions raised KeyError)."""
from __future__ import annotations

import pytest
from verity.circuits.codec import encode_program
from verity.circuits.program import is_input

from verity_vllm.query import program_view as PV

pytest.importorskip("torch")


def test_a_request_build_descriptor_views_its_roots_body_with_its_inputs_as_parameters():
    from tests.pipeline import test_single_request_build_path as T

    rep = T._derive(V=64, S=2, LP=4, STEPS=2)
    desc = encode_program(rep.program)
    assert desc["program_fn"] not in desc["definitions"]
    nodes = rep.program.root.body.nodes
    view = PV.from_descriptor(desc)
    assert len(view.calls) == sum(not is_input(n.fn) for n in nodes) > 0
    assert len(view.params) == sum(is_input(n.fn) for n in nodes) > 0 and view.outputs
    ids = {p.id for p in view.params}
    for c in view.calls:
        assert all(r.call in ids or 0 <= r.call < c.id for r in c.reads), c.id


def test_a_descriptor_whose_root_reads_an_input_node_recut_is_refused_by_name():
    from verity.circuits.defs import CompositeDefinition, DefinitionTable, PrimitiveDefinition, bind
    from verity.circuits.program import Program
    from verity.circuits.refs import Affine, Coll
    from verity.circuits.types import Array, Value

    bit = Value(1)
    not1 = PrimitiveDefinition("TViewRecutNot1", 1, (("a", bit),), bit, lambda a: a ^ 1, register=False)

    def toy(B, S, s):
        sp, ix, _ = s.refs[0]
        return B.batch(not1, Coll(Array(32, bit), Affine(sp, ix, 0, 32)), axes=(0,))

    prog = Program(bind(CompositeDefinition("TViewRecutToy", 1, (), lambda S: ((("s", Array(2, Value(16))),), Array(32, bit)),
                                            toy, register=False)))
    reg = DefinitionTable()
    reg.add(not1)
    with pytest.raises(ValueError, match="recut-unread: program_view.from_descriptor"):
        PV.from_descriptor(encode_program(prog), reg)

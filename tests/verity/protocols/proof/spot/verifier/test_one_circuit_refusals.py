"""The refusals of `one_circuit.block` and `one_circuit.cuts`, one case each: the smallest partition object that trips that clause
and no earlier one, asserted by the clause's own text, so deleting the clause fails its case."""
from __future__ import annotations

import json

import pytest

from test_proofs import _two_templates
from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.boolean import BIT, xor_gate
from verity.circuits.codec import descriptor_chunks, descriptor_id
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program, input_id
from verity.circuits.refs import Coll, array_of
from verity.circuits.types import Array
from verity.protocols.proof.spot.verifier.one_stage import one_circuit as OC

NO_TEMPLATE = {"name": "Q_template_instances", "version": 0, "params": {"templates": ["Nope_v1"]}}


def _object(blocks=None, query=None):
    c, part, _ = _two_templates(2, 2)
    return c, PO.build(json.loads(c), query or part["query"], blocks)


def _not_listed(c: bytes) -> str:
    """The reason a query listing no template of C is refused: C's first template node is no instance of the listed one."""
    D = R.program(c)
    k, nd = next((k, nd) for k, nd in enumerate(D.root.body.nodes) if input_id(nd.fn.id) is None)
    return f"E_QUERY: root node {k} is a {nd.form} of {nd.fn.id!r}: an instance is a call or batch of one of ['Nope_v1']"


def _refused(f) -> str:
    with pytest.raises(OC.Refused) as e:
        f()
    return str(e.value)


def test_a_partition_object_that_is_not_valid_is_refused_by_its_codes():
    c, part = _object([[(0, 4)]])
    assert _refused(lambda: OC.block(c, dict(part, format="nope"), 0)) == "E_PARTITION: the partition object is not valid: ['format']"


def test_a_partition_object_of_two_levels_of_blocks_is_refused():
    c, part = _object([[(0, 2), (2, 4)], [(0, 1), (1, 2), (2, 3), (3, 4)]])
    assert _refused(lambda: OC.block(c, part, 0)) == (
        "E_BLOCKS: partition: blocks-levels: the object has 2 levels of blocks; this reader takes one")


def test_a_partition_object_with_no_blocks_is_refused():
    c, part = _object()
    assert _refused(lambda: OC.block(c, part, 0)) == "E_BLOCKS: the partition object names no blocks"


def test_a_block_of_a_query_naming_no_template_of_c_is_refused_by_the_querys_reason():
    c, part = _object([[(0, 4)]], NO_TEMPLATE)
    assert _refused(lambda: OC.block(c, part, 0)) == _not_listed(c)


def test_blocks_ending_short_of_the_population_are_refused():
    c, part = _object([[(0, 1), (1, 3)]])
    assert _refused(lambda: OC.block(c, part, 0)) == "E_BLOCKS: the partition's blocks end at 3, not at C's population 4"


@pytest.mark.parametrize("k", [4, -1, 1.0, True])
def test_a_block_that_is_no_index_of_the_partitions_blocks_is_refused(k):
    c, part = _object([[(u, u + 1) for u in range(4)]])
    assert _refused(lambda: OC.block(c, part, k)) == f"E_BLOCKS: no block {k!r} among the partition's 4"


def test_cuts_of_a_partition_that_is_no_template_query_are_a_named_hole():
    c, part = _object([[(0, 4)]], {"name": "Q_word", "version": 1, "params": {"X": 16}})
    with pytest.raises(OC.NotDerived) as e:
        OC.cuts(c, part)
    assert str(e.value) == f"{OC.PREFIX}{OC.LOWER}"


def test_cuts_of_a_query_naming_no_template_of_c_are_refused_by_the_querys_reason():
    c, part = _object([[(0, 4)]], NO_TEMPLATE)
    assert _refused(lambda: OC.cuts(c, part)) == _not_listed(c)


def test_cuts_of_a_root_reading_a_port_in_rows_no_writer_field_divides_are_refused():
    """A batch of two writers of two leaves each, read three leaves at a time by one reader: its port's three leaves are no
    whole number of the writer's two-leaf rows."""
    A2, A3 = Array(2, BIT), Array(3, BIT)
    writer = bind(CompositeDefinition("DoorCutW", 0, (), lambda S: ((("x", A2),), A2),
                                      lambda B, S, x: array_of([B.call(xor_gate(2), x), B.call(xor_gate(2), x)]), register=False))
    reader = bind(CompositeDefinition("DoorCutR", 0, (), lambda S: ((("x", A3),), BIT), lambda B, S, x: B.call(xor_gate(3), x),
                                      register=False))

    def body(B, S, a):
        return B.call(reader, Coll(A3, B.batch(writer, a, axes=(0,)).refs.slice(0, 3)))
    root = CompositeDefinition("DoorCutRows", 0, (), lambda S: ((("a", Array(2, A2)),), BIT), body, register=False)
    c = b"".join(descriptor_chunks(Program(root)))
    part = {"query": {"name": "Q_template_instances", "version": 0, "params": {"templates": [descriptor_id(writer), descriptor_id(reader)]}}}
    assert _refused(lambda: OC.cuts(c, part)) == "E_LOWER: in(S)'s cuts: edge-rows"


def test_cuts_of_a_query_listing_a_template_of_no_gate_are_refused_in_leans_words():
    """A template that returns its parameter has no gate: the query is refused as `Partition.checkTemplates` refuses it,
    though the other template, whose instance a block might hold alone, has gates."""
    A2 = Array(2, BIT)
    same = bind(CompositeDefinition("DoorCutSame", 0, (), lambda S: ((("x", A2),), A2), lambda B, S, x: x, register=False))
    xor = bind(CompositeDefinition("DoorCutXor", 0, (), lambda S: ((("x", A2),), BIT), lambda B, S, x: B.call(xor_gate(2), x),
                                   register=False))
    root = CompositeDefinition("DoorCutGateless", 0, (), lambda S: ((("a", A2),), BIT),
                               lambda B, S, a: B.call(xor, B.call(same, a)), register=False)
    c = b"".join(descriptor_chunks(Program(root)))
    part = {"query": {"name": "Q_template_instances", "version": 0, "params": {"templates": [descriptor_id(same), descriptor_id(xor)]}}}
    assert _refused(lambda: OC.cuts(c, part)) == "E_QUERY: query-refused: template DoorCutSame_v0 has no gate"

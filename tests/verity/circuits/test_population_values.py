"""`mixed_population_program`'s values: values a window registers, which the population's root takes as parameters ahead of
the members' and no node reads, named in order in its binding record. The members' parameters, ports, proof blocks and
results are what they are without values, and a population without values keeps its id and digest (every population's
recorded id and digest is `test_population_recorded.py`'s)."""
from __future__ import annotations

import json

import pytest

from verity.circuits import partition_object as PO
from verity.circuits import ports as P
from verity.circuits.codec import decode_program, descriptor_id, encode_program, program_digest
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, DefinitionTable, bind
from verity.circuits.population import leaf_type, mixed_population_program
from verity.circuits.refs import runs as ref_runs
from verity.circuits.types import Array, Value

V32 = Value(32)
TAdd = PrimitiveDefinition("TPopValuesAdd32", 1, (("a", V32), ("b", V32)), V32, lambda a, b: (a + b) & 0xFFFFFFFF,
                           register=False)
TKeyFold = PrimitiveDefinition("TPopValuesKeyFold32", 1, (("x", V32), ("key", V32)), V32,
                               lambda x, key: ((x ^ key) * 0x9E3779B1) & 0xFFFFFFFF, register=False,
                               ports=P.Ports(params={"key": P.In(kind="coin")}, ret=P.Out(sink=True)))
Plain = bind(CompositeDefinition("TPopValuesPlain", 1, (), lambda S: ((("a", V32), ("b", V32)), V32),
                                 lambda B, S, a, b: B.call(TAdd, a, b), register=False))
#: a unit: a coin and a position in, its receipt a sink
Unit = bind(CompositeDefinition("TPopValuesUnit", 1, (), lambda S: ((("salt", V32), ("position", V32)), V32),
                                lambda B, S, salt, position: B.call(TKeyFold, position, salt), register=False,
                                ports=lambda S: P.Ports(params={"salt": P.In(kind="coin"),
                                                                "position": P.In(kind="position")},
                                                        ret=P.Out(sink=True))))
MEMBERS = [(Plain, 2), (Unit, 3)]
#: a round's A table of K = 2048 and its input-ids table, by the names H1's Commit record registers them under, in 16-bit words
VALUES = (("row-table/0/k2048", 6, 16), ("input-ids/0", 4, 16))


def _kinds(pt):
    return [[(lo, hi, p.kind) for lo, hi, p, _w in rs if p.kind in ("coin", "position")] for rs in pt.params]


def test_the_values_are_the_first_parameters_and_named_in_the_binding_record():
    prog = mixed_population_program(MEMBERS, values=VALUES)
    params = prog.fn.signature()[0]
    assert params[:2] == (("row-table/0/k2048", Array(6, Value(16))), ("input-ids/0", Array(4, Value(16))))
    assert [n for n, _ in params[2:]] == ["0.a", "0.b", "1.salt", "1.position"]
    assert prog.fn.bindings["values"] == VALUES
    assert descriptor_id(prog.fn).endswith(',values=[["row-table/0/k2048",6,16],["input-ids/0",4,16]],counts=[2,3]}')
    assert prog.fn.signature()[1] == mixed_population_program(MEMBERS).fn.signature()[1]


def test_a_population_without_values_names_none():
    plain, empty = mixed_population_program(MEMBERS), mixed_population_program(MEMBERS, values=())
    assert descriptor_id(plain.fn) == descriptor_id(empty.fn) and program_digest(plain) == program_digest(empty)
    assert "values" not in plain.fn.bindings and "values=" not in descriptor_id(plain.fn)
    assert program_digest(mixed_population_program(MEMBERS, values=VALUES)) != program_digest(plain)


def test_each_value_is_one_input_node_that_no_node_reads():
    plain, held = mixed_population_program(MEMBERS), mixed_population_program(MEMBERS, values=VALUES)
    nodes = held.root.body.nodes
    assert [(n.fn.id, n.gates) for n in nodes[:2]] == [("Input16_v1", 6), ("Input16_v1", 4)]
    read = {k for n in nodes for c in n.args for _s, k, _a, _b in ref_runs(c.refs)}
    read |= {k for _s, k, _a, _b in ref_runs(held.root.body.ret)}
    assert not read & {0, 1}
    assert held.input_gates == plain.input_gates + 10 and held.gates == plain.gates + 10


def test_the_template_query_and_the_ports_see_the_members_alone():
    plain, held = mixed_population_program(MEMBERS), mixed_population_program(MEMBERS, values=VALUES)
    q = PO.template_instances_query([Plain, Unit])
    a, b = PO.evaluate(plain, q), PO.evaluate(held, q)
    assert [r[1:] for r in b.ranges] == [r[1:] for r in a.ranges] and b.population == a.population == 5
    assert [r[0] for r in b.ranges] == [r[0] + 2 for r in a.ranges]
    pa, pb = P.port_types(plain.fn), P.port_types(held.fn)
    assert _kinds(pb) == [[], []] + _kinds(pa) == [[], [], [], [], [(0, 3, "coin")], [(0, 3, "position")]]
    assert [(lo, hi, o.sink) for lo, hi, o, _w in pb.ret] == [(lo, hi, o.sink) for lo, hi, o, _w in pa.ret] == [(2, 5, True)]


def test_a_value_of_a_leaf_type_is_an_array_of_it_and_decodes_to_itself():
    """A value whose width is a shape (`leaf_type`): two `Digest`s, ``(8, 64, 1)``, are ``Array(2, Digest)``, the binding
    record holds the shape as a list (given as a list or a tuple, one program), and the population decodes to itself.
    An integer width is ``Value(width)``, its binding record unchanged."""
    prog = mixed_population_program(MEMBERS, values=(("hist", 2, (8, 64, 1)), *VALUES))
    assert prog.fn.signature()[0][:2] == (("hist", Array(2, Array(8, Array(64, Value(1))))),
                                          ("row-table/0/k2048", Array(6, Value(16))))
    assert ',values=[["hist",2,[8,64,1]],["row-table/0/k2048",6,16],' in descriptor_id(prog.fn)
    assert program_digest(mixed_population_program(MEMBERS, values=(("hist", 2, [8, 64, 1]), *VALUES))) == \
        program_digest(prog)
    assert leaf_type(16) == Value(16) and leaf_type((4, 16)) == Array(4, Value(16))
    registry = DefinitionTable()
    for prim in (TAdd, TKeyFold):
        registry.add(prim)
    decoded = decode_program(json.loads(json.dumps(encode_program(prog))), registry)
    assert program_digest(decoded) == program_digest(prog)


@pytest.mark.parametrize("values, version", [
    (VALUES, 0),
    ((("t", 1, 16), ("t", 2, 16)), 1),
    ((("0.a", 1, 16),), 1),
    ((("t", 0, 16),), 1),
    ((("t", 1, 0),), 1),
    ((("t", True, 16),), 1),
    ((("", 1, 16),), 1),
    (((3, 1, 16),), 1),
    ((("t", 1, (16,)),), 1),
    ((("t", 1, ()),), 1),
    ((("t", 1, (8, 0)),), 1),
    ((("t", 1, (8, True)),), 1),
])
def test_a_value_that_is_no_new_name_of_a_leaf_and_a_bit_or_of_a_v0_population_is_refused(values, version):
    with pytest.raises(ValueError):
        mixed_population_program(MEMBERS, values=values, version=version)


def test_a_population_with_values_decodes_to_itself():
    prog = mixed_population_program(MEMBERS, values=VALUES)
    registry = DefinitionTable()
    for prim in (TAdd, TKeyFold):
        registry.add(prim)
    decoded = decode_program(json.loads(json.dumps(encode_program(prog))), registry)
    assert program_digest(decoded) == program_digest(prog)

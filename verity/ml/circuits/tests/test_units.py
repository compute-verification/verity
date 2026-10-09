"""Each unit of a partition as a standalone word circuit (`verity.ml.circuits.units`): on the `partition_object` test program, the units of
`Q_word` v1, of the template queries and of an owner function cover every computing gate once, read only parameter leaves or other
units' committed gates, output exactly their committed gates, and share shapes exactly when their circuits are equal; `classes`
counts what iteration yields without expanding separable cuts; and a unit's circuit, run gate by gate on its inputs, gives the
values its Call computes."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

import pytest

from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import codec, partition_object as PO
from verity.ml.circuits.cut import derived_committed
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluate import evaluate_call
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.units import PartitionUnits
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.ops.scalar import F32Fabs, F32Fmaxf, I32Add

F32 = I32 = Value(32)


def _c(name, params, ret, body):
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False))


#: fmaxf(fmaxf(x0, x1), |x2|): |x2| is wiring
Max3 = _c("UMax3", (("x", Array(3, F32)),), F32, lambda B, S, x: B.call(F32Fmaxf, B.call(F32Fmaxf, x[0], x[1]), B.call(F32Fabs, x[2])))
#: (s, s + x0), s = x0 + x1: two units, s committed
Pair = _c("UPair", (("x", Array(2, I32)),), Tuple(I32, I32),
          lambda B, S, x: (lambda s: tuple_of(s, B.call(I32Add, s, x[0])))(B.call(I32Add, x[0], x[1])))
#: a separable body: each row's Max3 its own cut
Rows = _c("URows", (("x", Array(2, Array(3, F32))),), Array(2, F32), lambda B, S, x: B.batch(Max3, x, axes=(0,)))
#: x0 + (1 + 2): the sum of constants is structure
AddOne = _c("UAddOne", (("x", Array(1, I32)),), I32,
            lambda B, S, x: B.call(I32Add, x[0], B.call(I32Add, B.call(const(32, 1)), B.call(const(32, 2)))))


def _fan(B, S, x):
    a = B.call(I32Add, x[0], x[1])
    B.call(I32Add, x[0], x[2])                                         # dead
    return tuple_of(B.call(I32Add, a, x[2]), B.call(I32Add, a, x[3]))


Fan = _c("UFan", (("x", Array(4, I32)),), Tuple(I32, I32), _fan)
CALLS = (Max3, Pair, Rows, AddOne, Fan, Pair)
Top = _c("UTop", tuple((n, fn.signature()[0][0][1]) for n, fn in zip("abcdef", CALLS)), Tuple(*[fn.signature()[1] for fn in CALLS]),
         lambda B, S, *a: tuple_of(*(B.call(fn, v) for fn, v in zip(CALLS, a))))


@pytest.fixture(scope="module")
def program():
    return Program(Top)


def _check_cover(units: PartitionUnits) -> list:
    placements = list(units)
    by_call: dict[int, list] = {}
    for pl in placements:
        by_call.setdefault((pl.node, pl.member, pl.path), []).append(pl.circuit)
    for (_k, _m, _path), circuits in by_call.items():
        act = units.activation_of(circuits[0].definition)
        keep = set(act.graph.keep)
        owned = sorted(j for uc in circuits for j in uc.gates if j in keep)
        assert owned == sorted(keep)                                            # every computing gate in exactly one unit
        mine = {j: uc.unit for uc in circuits for j in uc.gates if j in keep}
        outputs = {j for uc in circuits for j in uc.outputs}
        owner = [mine[j] for j in act.graph.keep]
        assert outputs == {act.graph.keep[c] for c in derived_committed(act.graph, owner)}
        for uc in circuits:
            for x in uc.inputs:                                                 # a parameter leaf, or another unit's committed gate
                assert x < 0 or (x in outputs and mine[x] != uc.unit)
    return placements


def test_q_word_units(program):
    units = PartitionUnits(program, PO.query(16, 32))
    placements = _check_cover(units)
    assert [pl.index for pl in placements] == list(range(units.partition.population))
    for pl in placements:                                                       # the partition's own numbering
        p, k, u, path = units.partition.locate(pl.index)
        assert pl.node == k and [(n, m) for n, m, _ in path] == list(pl.path)
    assert Counter(pl.circuit.shape for pl in placements) == Counter({s: c.units for s, c in units.classes().items()})


def test_separable_members_share_their_circuits(program):
    units = PartitionUnits(program, PO.query(16, 32))
    rows = [pl.circuit for pl in units if pl.path and units.program.root.body.nodes[pl.node].fn is Rows]
    assert len(rows) == 2 and rows[0] == rows[1] and rows[0].definition == Max3.id


def test_template_query_makes_each_instance_one_unit():
    prog = Program(Rows)                                                       # the root inlines URows: a batch of two UMax3
    units = PartitionUnits(prog, PO.template_instance_query(Max3))
    placements = list(units)
    assert [(pl.node, pl.member) for pl in placements] == [(1, 0), (1, 1)]
    for pl in placements:
        assert pl.circuit.computing == units.activation(Max3).graph.n and all(x < 0 for x in pl.circuit.inputs)


def test_an_owner_function(program):
    units = PartitionUnits(program, owners=lambda fn, graph: [c % 2 for c in range(graph.n)])
    _check_cover(units)
    with pytest.raises(ValueError):
        PartitionUnits(program, PO.query(), owners=lambda fn, g: [])


def test_equal_shapes_are_equal_circuits(program):
    units = PartitionUnits(program, owners=lambda fn, graph: list(range(graph.n)))      # every computing gate its own unit
    by_shape: dict[str, set] = {}
    for pl in units:
        by_shape.setdefault(pl.circuit.shape, set()).add(pl.circuit.form)
    assert all(len(forms) == 1 for forms in by_shape.values())
    form = json.loads(next(iter(by_shape.values())).pop())
    assert set(form) == {"in", "gates", "out"}


def test_a_unit_computes_its_calls_values(program):
    """Each unit's gates, evaluated on the values its flat Definition's transcript gives its inputs, reproduce that transcript at
    the unit's outputs."""
    rng = random.Random(7)
    units = PartitionUnits(program, PO.query(16, 32))
    seen = set()
    for pl in units:
        uc = pl.circuit
        act = units.activation_of(uc.definition)
        if (uc.definition, uc.unit) in seen:
            continue
        seen.add((uc.definition, uc.unit))
        args = [rng.getrandbits(32) for _ in range(act.params)]
        prog, call = standalone_call(act.fn)
        t: dict[int, int] = {}
        evaluate_call(prog.circuit, call, args, t)
        value = {j: t[act.absolute(j)] for j in range(act.fn.gates)} | {-1 - p: args[p] for p in range(act.params)}
        mine = {x: value[x] for x in uc.inputs}
        for j in uc.gates:
            g = act.gate(j)
            mine[j] = g.prim.evaluate(*(mine[act.local(o)] for o in g.operands))
        assert all(mine[j] == value[j] for j in uc.outputs)
    assert len(seen) >= 6


def test_pinned_shapes(program):
    """The shapes of the test program's `Q_word` units: a change here changes every class digest a backend has recorded."""
    got = sorted(c.representative.form for c in PartitionUnits(program, PO.query(16, 32)).classes().values())
    assert len(got) == len(set(got))
    assert hashlib.sha512(codec.canonical_json(got)).hexdigest() == "94f3cb71f5e3cd2637f39cd3df9998db80f1ef692a3aa5d77bab14516564452ac26ab702b5a520ced5ab3c6a2eaa9a41fa1102c273be1b1846c03fea86ada2db"

"""Binding times of a program's constants (`verity.ir.constants`): a constant node of the root body is bound per run, a constant
inside a callee's body at registration, and a Call's run-bound parameter leaves are exactly the leaves whose argument is a run
constant."""
from __future__ import annotations

import pytest

from verity.ir import constants as K
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.program import Program
from verity.ir.refs import tuple_of
from verity.ir.types import Tuple, Value
from verity.ml.prims import const
from verity.ml.scalar import I32Add

I32 = Value(32)
TEMP, TOP_P, SEVEN, POS = 0x3F4CCCCD, 0x3F733333, 7, 255


def _c(name, params, ret, body):
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, register=False))


#: (x + t) + 7: the 7 is the callee's own constant
Scale = _c("KScale", (("x", I32), ("t", I32)), I32, lambda B, S, x, t: B.call(I32Add, B.call(I32Add, x, t), B.call(const(32, SEVEN))))


def _top(B, S, x):
    temp = B.call(const(32, TEMP))
    positions = B.batch(const(32, POS), axes=(), n=3)
    a = B.call(Scale, x, temp)
    b = B.call(Scale, a, positions[1])
    return tuple_of(b, B.call(I32Add, x, B.call(const(32, TOP_P))))


Top = _c("KTop", (("x", I32),), Tuple(I32, I32), _top)


@pytest.fixture(scope="module")
def circuit():
    return Program(Top).circuit


def _calls(circuit):
    return [c for c in circuit.root_calls() if c.id == Scale.id]


def test_root_constants_are_run_constants(circuit):
    run = K.run_constants(circuit)
    assert sorted(run.values()) == sorted([TEMP, POS, POS, POS, TOP_P])
    for g in run:
        assert K.binding_time(circuit, g) == K.RUN


def test_every_constant_has_one_binding_time(circuit):
    run = K.run_constants(circuit)
    seen = {K.RUN: 0, K.REGISTRATION: 0}
    for g in range(circuit.size):
        t = K.binding_time(circuit, g)
        if t is None:
            assert not K.is_constant(circuit.gate(g).prim)
            continue
        seen[t] += 1
        assert (t == K.RUN) == (g in run)
        if t == K.REGISTRATION:
            assert circuit.gate(g).prim.evaluate() == SEVEN
    assert seen == {K.RUN: 5, K.REGISTRATION: 2}


def test_inputs_and_computed_gates_have_no_binding_time(circuit):
    for g in range(circuit.size):
        prim = circuit.gate(g).prim
        if not K.is_constant(prim):
            assert K.binding_time(circuit, g) is None
    assert not K.is_constant(I32Add)


def test_run_arguments_name_the_bound_leaves(circuit):
    first, second = _calls(circuit)
    assert K.run_arguments(first) == {1: TEMP}
    assert K.run_arguments(second) == {1: POS}
    for call in (first, second):
        for leaf, value in K.run_arguments(call).items():
            assert K.binding_time(circuit, call.args[leaf]) == K.RUN
            assert circuit.gate(call.args[leaf]).prim.evaluate() == value


def test_no_callee_body_holds_a_run_constant(circuit):
    for call in _calls(circuit):
        for g in range(call.lo, call.hi):
            assert K.binding_time(circuit, g) != K.RUN

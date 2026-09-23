"""``verity.verification.query``: the one-unit / two-unit contract of the design on tiny synthetic Programs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from verity.verification.query import (PartitionError, Policy, ValueRef, VerificationUnit, admissibility, boundary, boundary_values,
                                       partition_by, required_values, unit_boundary)


@dataclass(frozen=True)
class C:
    id: int
    definition: str
    ports: tuple[str, ...]
    reads: tuple[ValueRef, ...]
    module: str = ""


@dataclass(frozen=True)
class P:
    calls: tuple[C, ...]
    inputs: tuple[ValueRef, ...] = ()
    outputs: tuple[ValueRef, ...] = ()


X = ValueRef(-1, "out")          # the prescribed input x
U = ValueRef(0, "out")           # Matmul's output
Y = ValueRef(1, "out")           # Activation's output


def chain(module_of=("m", "m")) -> P:
    """``x -- Matmul -- u -- Activation -- y``; ``y`` is the Program output."""
    return P(calls=(C(0, "Matmul", ("out",), (X,), module_of[0]), C(1, "Activation", ("out",), (U,), module_of[1])),
             inputs=(X,), outputs=(Y,))


def by_module(P_):
    return partition_by(lambda c: c.module)(P_)


def test_one_unit_boundary_is_x_and_y():
    Pr = chain(("m", "m"))
    assert boundary_values(Pr, by_module) == {X, Y}
    (unit,) = by_module(Pr)
    ins, outs = unit_boundary(Pr, unit)
    assert (ins, outs) == ({X}, {Y})


def test_two_units_boundary_is_x_u_y():
    Pr = chain(("a", "b"))
    assert boundary_values(Pr, by_module) == {X, U, Y}
    ua, ub = by_module(Pr)
    assert unit_boundary(Pr, ua) == ({X}, {U})
    assert unit_boundary(Pr, ub) == ({U}, {Y})


def test_interior_value_read_by_two_calls_of_one_unit_is_not_a_boundary_value():
    # x -> A -> u ; u read by B and by C, all in one unit; C's output is the Program output
    calls = (C(0, "A", ("out",), (X,), "m"), C(1, "B", ("out",), (U,), "m"), C(2, "C", ("out",), (U, ValueRef(1, "out")), "m"))
    Pr = P(calls, inputs=(X,), outputs=(ValueRef(2, "out"),))
    assert boundary_values(Pr, by_module) == {X, ValueRef(2, "out")}


def test_tuple_ports_are_distinct_values():
    # Norm returns (normed, residual); only the residual crosses into the next unit; normed is dead
    calls = (C(0, "Norm", ("0", "1"), (X,), "n"), C(1, "Add", ("out",), (ValueRef(0, "1"),), "r"))
    Pr = P(calls, inputs=(X,), outputs=(ValueRef(1, "out"),))
    assert boundary_values(Pr, by_module) == {X, ValueRef(0, "1"), ValueRef(1, "out")}
    b = boundary(Pr, by_module)
    assert b.produced_in[ValueRef(0, "1")] == "n" and b.read_by[ValueRef(0, "1")] == ("r",)
    assert b.produced_in[X] is None                                       # a prescribed input belongs to no unit


def test_boundary_records_readers_in_body_order_once_each():
    calls = (C(0, "A", ("out",), (X,), "p"), C(1, "B", ("out",), (U,), "q"), C(2, "C", ("out",), (U,), "r"), C(3, "D", ("out",), (U,), "q"))
    Pr = P(calls, inputs=(X,), outputs=(ValueRef(3, "out"),))
    b = boundary(Pr, by_module)
    assert b.read_by[U] == ("q", "r")


def test_units_must_partition_the_calls():
    Pr = chain()
    with pytest.raises(PartitionError):
        boundary_values(Pr, lambda P_: (VerificationUnit("only-first", frozenset({0})),))
    with pytest.raises(PartitionError):
        boundary_values(Pr, lambda P_: (VerificationUnit("a", frozenset({0, 1})), VerificationUnit("b", frozenset({1}))))
    with pytest.raises(ValueError):
        VerificationUnit("empty", frozenset())


def test_required_values_keeps_the_three_sources_apart():
    Pr = chain(("a", "b"))
    W = ValueRef(-2, "w")
    Pr = P(Pr.calls, inputs=(X, W), outputs=Pr.outputs)                 # w: a prescribed input no Call reads
    rv = required_values(Pr, by_module, Policy(prescribed_inputs=frozenset({W}), protocol_required=frozenset({U})))
    assert rv.boundary == {X, U, Y}
    assert rv.prescribed == {W} and rv.protocol == {U}
    assert rv.values == {X, U, Y, W} and len(rv) == 4 and W in rv
    assert required_values(Pr, by_module, Policy.none()).values == {X, U, Y}


def test_policy_naming_an_unknown_value_is_an_error():
    Pr = chain()
    with pytest.raises(ValueError):
        required_values(Pr, by_module, Policy(prescribed_inputs=frozenset({ValueRef(-9, "out")})))
    with pytest.raises(ValueError):
        required_values(Pr, by_module, Policy(protocol_required=frozenset({ValueRef(0, "nope")})))
    with pytest.raises(ValueError):
        required_values(Pr, by_module, Policy(protocol_required=frozenset({ValueRef(7, "out")})))


def test_admissibility_is_a_pure_table_lookup():
    req = {X, U, Y}
    assert admissibility(req, {X, Y}) == [U]
    assert admissibility(req, {X: True, U: False, Y: True}) == [U]
    assert admissibility(req, lambda v: v.port == "out") == []
    assert admissibility(req, set()) == sorted(req)


def test_partition_by_orders_units_when_asked():
    Pr = chain(("b", "a"))
    assert [u.id for u in partition_by(lambda c: c.module)(Pr)] == ["b", "a"]
    assert [u.id for u in partition_by(lambda c: c.module, order=str)(Pr)] == ["a", "b"]

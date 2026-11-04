"""`verity.ml.circuits.layout`'s scope walk (`resolve_gate`) on a scan of no iterations, as Lean's `Flock.Program.resolve`
walks it (`verity/core/service/flock/Flock/Program.lean`): a gate reading the scan's output is refused by name, and a
program that holds such a scan without reading it resolves every gate. `Circuit.gates` over any range is `Circuit.gate` of
each index in it: the same gates, operands and scopes, and the same exception where the range leaves the circuit or a gate
reads a scan of no iterations."""
from __future__ import annotations

import pytest

from verity.ml.circuits import layout
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

V8 = Value(8)
Op = PrimitiveDefinition("TEmptyScanOp8", 1, (("a", V8), ("b", V8)), V8, lambda a, b: a ^ b, register=False)
Step = bind(CompositeDefinition("TEmptyScanStep", 1, (), lambda S: ((("c", V8),), Tuple(V8, Tuple())),
                                lambda B, S, c: tuple_of(c, tuple_of()), register=False))
Skip = bind(CompositeDefinition("TEmptyScanSkip", 1, (), lambda S: ((("c", V8), ("x", V8)), V8),
                                lambda B, S, c, x: B.call(Op, x, x), register=False))


def _program(kind: str) -> Program:
    """The root holds `Input8_v1` (x) and a scan of no iterations whose carry is x; the scan's carry is read by a primitive
    (`read`), passed to a Call that reads none of it (`passed`), or read by nothing (`unread`)."""
    def body(B, S, x):
        s = B.scan(Step, x, n=0)
        if kind == "passed":
            return B.call(Skip, s[0], x)
        return B.call(Op, s[0], x) if kind == "read" else B.call(Op, x, x)
    return Program(bind(CompositeDefinition(f"TEmptyScan{kind.capitalize()}", 1, (), lambda S: ((("x", V8),), V8), body,
                                            register=False)))


@pytest.fixture
def bounded(monkeypatch):
    """Every `Scope` the walk makes, counted, and refused past a bound, so an unbounded walk fails here instead of hanging."""
    made = [0]

    class Bounded(layout.Scope):
        __slots__ = ()

        def __init__(self, *a) -> None:
            made[0] += 1
            if made[0] > 1000:
                raise RuntimeError("the scope walk made over 1,000 scopes")
            super().__init__(*a)

    monkeypatch.setattr(layout, "Scope", Bounded)
    return made


def test_a_gate_reading_a_scan_of_no_iterations_is_refused_by_name(bounded):
    C = _program("read").circuit
    assert [n.fn.id for n in C.root.body.nodes] == ["Input8_v1", "TEmptyScanStep_v1", "TEmptyScanOp8_v1"]
    with pytest.raises(ValueError, match="^a scan with no iterations has no output to read$"):
        C.gate(1)
    assert bounded[0] <= 2


@pytest.mark.parametrize("kind", ["unread", "passed"])
def test_a_scan_of_no_iterations_no_gate_reads_resolves_every_gate(bounded, kind):
    C = _program(kind).circuit
    assert [tuple(C.gate(g).operands) for g in range(C.size)] == [(), (0, 0)]


N_WIDE = layout.WIDE_FAN_IN + 3
Add = PrimitiveDefinition("TGatesAdd8", 1, (("a", V8), ("b", V8)), V8, lambda a, b: (a + b) & 255, register=False)
Neg = PrimitiveDefinition("TGatesNeg8", 1, (("a", V8),), V8, lambda a: (-a) & 255, register=False)
Sum3 = PrimitiveDefinition("TGatesSum3", 1, (("v", Array(3, V8)),), V8, lambda a, b, c: (a + b + c) & 255, register=False)
Wide = PrimitiveDefinition("TGatesWide", 1, (("i", V8), ("t", Array(N_WIDE, V8))), V8, lambda i, *t: t[i % N_WIDE], register=False)


def _composite(name: str, params: tuple, ret, body):
    return bind(CompositeDefinition(f"TGates{name}", 1, (), lambda S: (tuple((f"p{i}", t) for i, t in enumerate(params)), ret), body,
                                    register=False))


def _flat(parts: list) -> Coll:
    return Coll(Array(sum(p.type.leaves for p in parts), V8), concat([p.refs for p in parts]))


Swap = _composite("Swap", (V8, V8), Tuple(V8, V8), lambda B, S, a, b: tuple_of(b, a))
Inner = _composite("Inner", (V8, V8), V8, lambda B, S, a, b: B.call(Add, B.call(Neg, B.call(Add, a, b)), a))
Crossed = _composite("Crossed", (Array(2, Array(2, V8)),), V8,
                     lambda B, S, q: (lambda s: B.call(Inner, s[0], s[1]))(B.call(Swap, q[0][0], q[1][1])))
Inner3 = _composite("Inner3", (Array(3, V8),), V8, lambda B, S, v: B.call(Add, B.call(Sum3, v), v[1]))
Step3 = _composite("Step", (V8, V8, V8), Tuple(V8, V8),
                   lambda B, S, c, x, s: (lambda c2: tuple_of(c2, B.call(Neg, c2)))(B.call(Add, c, B.call(Add, x, s))))
PARAMS = (Array(6, V8), V8, Array(3, Array(2, V8)), Array(2, Array(3, Array(2, V8))), Array(N_WIDE, V8))


def _every_construct(B, S, x, y, m, z, w):
    b0 = B.batch(Add, x[0:3], y, axes=(0, None))
    c1 = B.call(Inner, x[0], y)
    sw = B.call(Swap, y, x[5])
    return _flat([b0, c1, B.batch(Sum3, m, axes=(1,)), B.batch(Inner3, m, axes=(1,)), B.batch(Inner, x[0:3], x[3:6], axes=(0, 0)),
                  B.batch(Crossed, z, axes=(1,)), B.scan(Step3, c1, xs=[x], shared=[b0[2]])[0],
                  B.scan(Step3, y, shared=[x[2], x[3]], n=2)[1], B.call(Wide, y, w), B.batch(Wide, x[0:2], w, axes=(0, None)),
                  B.call(Add, sw[0], sw[1])])


EVERY = _composite("Every", PARAMS, Array(21, V8), _every_construct)


def _key(g: layout.Gate) -> tuple:
    s = g.scope
    return (g.index, g.prim, type(g.operands), tuple(g.operands), s.spec, s.offset, s.node_idx, s.member, tuple(s.path()))


def _walked(gates) -> tuple:
    out = []
    try:
        for g in gates:
            out.append(_key(g))
    except (IndexError, ValueError) as e:
        return out, (type(e).__name__, str(e))
    return out, None


def test_the_gates_of_any_range_are_the_gates_found_one_by_one():
    """Calls, a call of no gates whose returns are its parameters, batches along no axis, axis 0 and a later one (of a
    primitive, of composites, one whose gates read every leaf of its parameter), scans with xs and shared arguments, wide gates
    alone and batched; every range across their
    boundaries, empty ranges, and ranges past either end, which raise where the gate-by-gate walk raises."""
    C = Program(EVERY).circuit
    n = C.size
    cuts = sorted({0, n} | {i for i in range(0, n, 37)} | {n - k for k in range(1, 40)} | {g.index for g in map(C.gate, range(n - 60, n))})
    ranges = [(0, n), (0, 0), (n, n), (5, 4), (-1, 3), (-2, -1), (n - 2, n + 2), (n + 1, n + 3), (n, n + 1)]
    ranges += [(a, b) for a in cuts for b in cuts if a < b and (b - a < 90 or a == 0)]
    for lo, hi in ranges:
        assert _walked(C.gates(lo, hi)) == _walked(C.gate(i) for i in range(lo, hi)), (lo, hi)
    assert _walked(C[0:n]) == _walked(C.gates()) and _walked(C.gates())[1] is None


@pytest.mark.parametrize("kind", ["read", "unread", "passed"])
def test_the_gates_past_a_scan_of_no_iterations_are_the_gates_found_one_by_one(kind):
    C = _program(kind).circuit
    got = _walked(C.gates())
    assert got == _walked(C.gate(i) for i in range(C.size)) and (got[1] is not None) == (kind == "read")

"""`verity.ml.circuits.layout`'s scope walk (`resolve_gate`) on a scan of no iterations, as Lean's `Flock.Program.resolve`
walks it (`verity/core/service/flock/Flock/Program.lean`): a gate reading the scan's output is refused by name, and a
program that holds such a scan without reading it resolves every gate."""
from __future__ import annotations

import pytest

from verity.ml.circuits import layout
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Tuple, Value

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

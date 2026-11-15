"""A Program whose root is one call of a registered function, for the tests that evaluate one Definition gate by gate."""

from __future__ import annotations

from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.program import Program

_STANDALONE = CompositeDefinition("Standalone", 1, ("FN",), lambda S: S.FN.signature(),
                                  lambda B, S, *args: B.call(S.FN, *args),
                                  doc="root wrapper: one call of F over Input gates", register=False)


def _standalone(spec):
    """A program whose root is exactly one call of `spec`; returns (program, the FunctionCall)."""
    prog = Program(bind(_STANDALONE, FN=spec))
    circ = prog.circuit
    calls = [c for c in circ.root_calls() if not c.is_primitive]
    assert len(calls) == 1, calls
    return prog, calls[0]

"""The resource-accounting circuits `door_walls.py` puts through the door, built by their protocols' own provers, and the
smallest block with an internal read."""
from __future__ import annotations

from typing import Any

#: the internal read's template: its read row's bits and its wide field's, past compose's floor
READ, WIDE = 16, 2048


def internal_read() -> tuple[Any, dict]:
    """C(a, w1, w2) = T(T(a, w1), w2), T(x : Array 16 (Value 1), w : Array 2048 (Value 1)) = map Not x: two units of T,
    the second's x the first's output, and its template query."""
    from verity.circuits import boolean as BL
    from verity.circuits import partition_object as PO
    from verity.circuits.defs import CompositeDefinition, bind
    from verity.circuits.program import Program
    from verity.circuits.types import Array, Value

    def composite(name, params, ret, body):
        return bind(CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False))

    x, w = Array(READ, Value(1)), Array(WIDE, Value(1))
    t = composite("WallsRead", (("x", x), ("w", w)), x, lambda B, S, x_, _w: B.batch(BL.NOT, x_, axes=[0], n=READ))
    root = composite("WallsReadRoot", (("a", x), ("w1", w), ("w2", w)), x,
                     lambda B, S, a, w1, w2: B.call(t, B.call(t, a, w1), w2))
    return Program(root), PO.template_instances_query([t])


def pous_setup() -> Any:
    """PoUS's setup C at its test's point (`window.Setup(64, 189, 16, 4, 4)`), an `IF.Circuit`."""
    from verity.protocols.accounting.memory.pous.prover import window as WIN
    return WIN.Setup(64, 189, 16, 4, 4).circuit()

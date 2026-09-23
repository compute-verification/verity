"""`Q_module_body_v1`: units = module BODIES.

A unit is the maximal set of Calls whose recorded owning module path (the correspondence's `module`) is identical.  A non-leaf
module's own Calls -- the ones its forward issues itself, not through a child -- form their own body (a decoder's own `embed *
scale` under `model`, beside the bodies of `model.layers.0.input_layernorm`, ...); Calls with no module form the root body
(unit id "").  Nothing about a module's class, name or family enters: two modules with identical paths are one body because the
correspondence says the same module issued their Calls.

Under this query the boundary Values are exactly the Values that leave a module body for another module's body (or for the
Program's output) and the prescribed inputs a module body reads.  v1's "promotion" of interior rows whose output another module
reads (F-r19-int-20) is not a pass here: such a Value crosses the body boundary and so is a boundary Value by definition.

Literals.  The IR has no literal operand (a reference is `("p", i, j)` -- a parameter leaf -- or `("n", k, j)` -- a body node's
output leaf): a literal is a Call that reads no Value (`Const<w>[bits]`, a zero-arity gate), whose output is a function of its binding
record alone and so is written in the Program descriptor.  Both ProgramView readers read the Program function's body, whose prescribed
inputs are its parameters (never `Input<w>` gates, which only the lowered root circuit has), so there a Call with no reads is always
such a literal.  A literal is part of the Program, not a runtime Value to acquire: the Values of `literal_calls(P)` are never boundary
Values, wherever the correspondence places the Call and whichever body reads it (`module_body_boundary`).
"""

from __future__ import annotations

from verity.verification.query import Boundary, Partition, boundary, partition_by

from .correspondence import Correspondence
from .program_view import ProgramView

__all__ = ["Q_MODULE_BODY_ID", "Q_MODULE_BODY_TEXT", "Q_module_body_v1", "literal_calls", "without_literals", "module_body_boundary"]

Q_MODULE_BODY_ID = "Q_module_body_v1"
Q_MODULE_BODY_TEXT = ("Q_module_body_v1: unit = module body = the maximal set of Calls whose recorded owning module path (correspondence: "
                      "nn_module_stack-derived) is identical; a non-leaf module's own Calls are their own body; Calls with no module are the "
                      "root body; boundary values = U_V (in(V) u out(V)) over these units (verity.verification.query.boundary_values) minus "
                      "the Program's literals (the Values of Calls that read no Value: written in the descriptor, never acquired)")


def Q_module_body_v1(P: ProgramView, correspondence: Correspondence) -> Partition:
    """The partition of `P`'s Calls by owning module path (unit id = the Program module path; "" for the root body)."""
    if correspondence.P is not P:
        raise ValueError("Q_module_body_v1: the correspondence was read off another Program")
    return partition_by(lambda c: correspondence.module_of(c.id))


def literal_calls(P: ProgramView) -> frozenset[int]:
    """The Calls of `P` that read no Value: the Program's literals (module docstring).  O(1) per Call on a columnar body."""
    cols = P.columns
    if cols is not None:
        off = cols.read_off
        return frozenset(k for k in range(cols.n) if off[k + 1] == off[k])
    return frozenset(c.id for c in P.calls if not len(c.reads))


def without_literals(B: Boundary, literals: frozenset[int]) -> Boundary:
    """`B` without the Values a literal Call produces (the units and the Call -> unit map unchanged)."""
    drop = frozenset(v for v in B.values if v.call in literals)
    if not drop:
        return B
    return Boundary(B.values - drop, {v: u for v, u in B.produced_in.items() if v not in drop},
                    {v: r for v, r in B.read_by.items() if v not in drop}, B.units, B.owner)


def module_body_boundary(P: ProgramView, correspondence: Correspondence, literals: frozenset[int] | None = None) -> Boundary:
    """The boundary of `Q_module_body_v1(P, correspondence)`: the crossing Values of the module-body partition minus the literals."""
    B = boundary(P, Q_module_body_v1(P, correspondence))
    return without_literals(B, literal_calls(P) if literals is None else literals)

"""`Q_module_body_v1`: units = module BODIES.

A unit is the maximal set of Calls whose recorded owning module path (the correspondence's `module`) is identical.  A non-leaf
module's own Calls -- the ones its forward issues itself, not through a child -- form their own body (a decoder's own `embed *
scale` under `model`, beside the bodies of `model.layers.0.input_layernorm`, ...); Calls with no module form the root body
(unit id "").  Nothing about a module's class, name or family enters: two modules with identical paths are one body because the
correspondence says the same module issued their Calls.

Under this query the boundary Values are exactly the Values that leave a module body for another module's body (or for the
Program's output) and the prescribed inputs a module body reads.  v1's "promotion" of interior rows whose output another module
reads (F-r19-int-20) is not a pass here: such a Value crosses the body boundary and so is a boundary Value by definition.
"""

from __future__ import annotations

from verity.verification.query import Partition, partition_by

from .correspondence import Correspondence
from .program_view import ProgramView

__all__ = ["Q_MODULE_BODY_ID", "Q_MODULE_BODY_TEXT", "Q_module_body_v1"]

Q_MODULE_BODY_ID = "Q_module_body_v1"
Q_MODULE_BODY_TEXT = ("Q_module_body_v1: unit = module body = the maximal set of Calls whose recorded owning module path (correspondence: "
                      "nn_module_stack-derived) is identical; a non-leaf module's own Calls are their own body; Calls with no module are the "
                      "root body; boundary values = U_V (in(V) u out(V)) over these units (verity.verification.query.boundary_values)")


def Q_module_body_v1(P: ProgramView, correspondence: Correspondence) -> Partition:
    """The partition of `P`'s Calls by owning module path (unit id = the Program module path; "" for the root body)."""
    if correspondence.P is not P:
        raise ValueError("Q_module_body_v1: the correspondence was read off another Program")
    return partition_by(lambda c: correspondence.module_of(c.id))

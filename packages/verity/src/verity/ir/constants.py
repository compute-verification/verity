"""When each constant of a program is fixed: its binding time (Project doc `constant-api-public.md` §1.3).

A constant is a gate of a parameterless primitive that isn't an Input (`PROTOCOL.md` §8). `Const{w}[0x…]_v1` carries its
value in its id. Every constant is bound in one of two ways:

* **registration**: a constant inside a callee's body. A Definition is fixed when it is registered, whatever run calls it:
  a norm's epsilon, the attention scale, rounding constants.
* **run**: a constant node of the root body. The root inlines the program function for one run (`program.Program`), so its
  constant nodes are what that run fixes: row #101's temperature, top-p and per-step Philox positions.

A run constant reaches a Definition only as a Call's argument, a parameter leaf of the callee, and never inside the
callee's own body. So no Definition depends on the run. `run_arguments` lists which of a Call's parameter leaves the run
binds, and to what: a lowering tags exactly those bindings ``run``, and a private build can turn them into run inputs.
"""
from __future__ import annotations

from typing import Any

from .defs import PrimitiveDefinition
from .layout import Circuit, FunctionCall
from .program import is_input

REGISTRATION = "registration"
RUN = "run"
BINDING_TIMES = (REGISTRATION, RUN)


def is_constant(fn: Any) -> bool:
    """A constant primitive: no parameters, and not an Input."""
    return isinstance(fn, PrimitiveDefinition) and not fn.params and not is_input(fn)


def binding_time(circuit: Circuit, gate: int) -> str | None:
    """``run`` for a constant of the root body, ``registration`` for a constant inside a callee, ``None`` for any other gate."""
    g = circuit.gate(gate)
    if not is_constant(g.prim):
        return None
    return RUN if g.scope.parent is None else REGISTRATION


def run_constants(circuit: Circuit) -> dict[int, int]:
    """Every run constant of the program, by gate: its value."""
    out: dict[int, int] = {}
    scope = circuit.root_scope
    body = scope.spec.body
    for k, node in enumerate(body.nodes):
        if not is_constant(node.fn):
            continue
        lo = scope.offset + body.offsets[k]
        for m in range(1 if node.form == "primitive" else node.n):
            out[lo + m] = node.fn.evaluate()
    return out


def run_arguments(call: FunctionCall) -> dict[int, int]:
    """For one Call, the parameter leaves (in the callee's leaf order) that a run constant binds: leaf -> value."""
    out: dict[int, int] = {}
    for leaf, gate in enumerate(call.args):
        if binding_time(call.circuit, gate) == RUN:
            out[leaf] = call.circuit.gate(gate).prim.evaluate()
    return out


__all__ = ["BINDING_TIMES", "REGISTRATION", "RUN", "binding_time", "is_constant", "run_arguments", "run_constants"]

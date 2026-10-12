"""What a program brings, and when it is bound (Daniel's ruling, 1 Oct 2026, 3:06 PM PDT).

An operation's own definition is gates: the floating-point formats' constants, SHA-512's and the measured GPU tables (exp2,
rcp, tanh) are gates of the operations that read them. Everything a program or a request brings is a value, committed under a
hiding leaf and always hidden, so a Program holds no data. A value enters a Program only as an input gate, of one of two
binding times (`program.Program(fn, registered=...)`):

* **registration** (`RegisteredInput<w>_v1`): what the program brings, bound when it is registered: its weights, a norm's
  epsilon, the attention scale, a softcap, and its tables (`gelu_tanh_bf16`, `tanh_rn`).
* **run** (`Input<w>_v1`): what a request brings, bound per run: its data, and its parameters such as temperature, top-p and
  the Philox positions.

A constant (`Const<w>[0x…]_v1`: a parameterless primitive that is not an input) is an operation's own bits. A constant
anywhere else is data the Program still holds, and `program_data` lists it by the Definition that holds it:

* a constant of the root body is always data (the root is no operation): it becomes a run input;
* a constant inside a Definition is data unless an enclosing Definition is an operation, which the caller names (`operation`).

A table a primitive reads by index is no gate, so the primitive declares it (`PrimitiveDefinition.tables`): a ``library``
table is an operation's own definition, and a ``registered`` one is the program's, which the primitive still holds. `verity.ml.circuits`
names no operations and no library tables; the guard is `verity.ml.circuits.operations.program_data`, which passes in both.
"""
from __future__ import annotations

from typing import Any, Callable

from .defs import PrimitiveDefinition
from .layout import Circuit
from .program import REGISTRATION, RUN, is_input

BINDING_TIMES = (REGISTRATION, RUN)


def is_constant(fn: Any) -> bool:
    """A constant primitive: no parameters, and not an input."""
    return isinstance(fn, PrimitiveDefinition) and not fn.params and not is_input(fn)


def binding_time(circuit: Circuit, gate: int) -> str | None:
    """``registration`` for a `RegisteredInput` gate, ``run`` for an `Input` gate, ``None`` for any other gate."""
    prim = circuit.gate(gate).prim
    return getattr(prim, "binding", None) if is_input(prim) else None


def inputs(circuit: Circuit) -> dict[str, list[int]]:
    """The Program's input gates by binding time: ``{registration: [...], run: [...]}``, ascending."""
    out: dict[str, list[int]] = {REGISTRATION: [], RUN: []}
    scope = circuit.root_scope
    body = scope.spec.body
    for k, node in enumerate(body.nodes):
        if is_input(node.fn):
            lo = scope.offset + body.offsets[k]
            out[node.fn.binding].extend(range(lo, lo + body.offsets[k + 1] - body.offsets[k]))
    return out


def held_tables(prim: Any, library: Callable[[str], bool] = lambda _name: False) -> list[str]:
    """The tables a primitive declares and holds as data: every ``registered`` one, and a ``library`` one unless
    `library(name)` names it."""
    return [n for n, kind in getattr(prim, "tables", ()) if kind != "library" or not library(n)]


def program_data(root: Any, operation: Callable[[Any], bool] = lambda _fn: False,
                 library: Callable[[str], bool] = lambda _name: False) -> dict[str, list[tuple[int, str]]]:
    """The data `root` (a Program's root Definition) still holds, by the Definition that holds it: descriptor id -> [(local
    gate, constant id or table name)], each Definition once, in first-reached order. Empty when the Program holds no data.

    A Definition `operation(fn)` names, and everything inside it when reached only through operations, holds none: a
    Definition reached first under an operation is not visited there, nor marked seen, so a path to it outside any operation
    still lists what it holds. A primitive reached outside any operation that holds a table (`held_tables`) is
    listed under its own id, at its one gate, by the table's name."""
    out: dict[str, list[tuple[int, str]]] = {}
    seen: set[str] = set()

    def walk(fn: Any, is_root: bool) -> None:
        fid = fn.id
        if fid in seen or (not is_root and operation(fn)):
            return
        seen.add(fid)
        body = fn.body
        mine = out[fid] = []
        for k, node in enumerate(body.nodes):
            callee = node.fn
            if is_constant(callee):
                mine.extend((body.offsets[k] + m, callee.id) for m in range(body.offsets[k + 1] - body.offsets[k]))
            elif getattr(callee, "body", None) is not None:
                walk(callee, False)
            elif callee.id not in seen and (held := held_tables(callee, library)):
                seen.add(callee.id)
                out[callee.id] = [(0, n) for n in held]
        if not mine:
            del out[fid]

    walk(root, True)
    return out


__all__ = ["BINDING_TIMES", "REGISTRATION", "RUN", "binding_time", "held_tables", "inputs", "is_constant", "program_data"]

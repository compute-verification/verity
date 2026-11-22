"""A circuit's Boolean gate count, from its own Boolean bodies: `count` over a descriptor, `of_block` over a proof block.

C names things at word level; its bit-level circuit, each word primitive replaced by its body (Lean's `inline`), is a
function of C, and `count` is exactly that function's size: the sum over C's gates of their bodies' counts, never a bound.

**The rule** (`gate`), the census's (`census.BooleanWork`) and the one the inference-only count reads: a gate of the Boolean
profile (PROTOCOL.md §9) counts its two-input gates, an `And<n>_v1` n - 1 of them, all ANDs, an `Xor<n>_v1` n - 1 and a
`Not_v1` 1, none an AND; a constant bit and an input of either binding and any width count 0. A word primitive counts what
its body in the descriptor's `boolean` map (§4.9) counts, and nothing else does. A composite counts the sum over its
nodes of members (`parts.n_members`: a batch's n, a scan's n, else 1) times its callee's count, once per specialized
Definition, so a batch or a scan is multiplied and never unrolled. This is not §6's gate count, which lays an `And<n>_v1`
or a constant bit out as one gate, nor `Definition.gates`, which counts a word primitive as one gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import boolean as BL
from .codec import descriptor_id
from .defs import PrimitiveDefinition
from .parts import n_members
from .program import input_id


@dataclass(frozen=True)
class Count:
    """Two-input Boolean gates and, among them, ANDs."""

    gates: int = 0
    ands: int = 0

    def __add__(self, o: Count) -> Count:
        return Count(self.gates + o.gates, self.ands + o.ands)

    def __mul__(self, k: int) -> Count:
        return Count(self.gates * k, self.ands * k)


ZERO = Count()


def gate(fn: PrimitiveDefinition) -> Count | None:
    """A primitive's count by the rule (module docstring), or None for a word primitive, which its body counts."""
    gk = BL.gate_kind(fn)
    if gk is not None:
        kind, n = gk
        return Count(max(1, n - 1), n - 1 if kind == "and" else 0)
    if BL.is_constant_bit(fn) or input_id(fn.id) is not None:
        return ZERO
    return None


class NoBody(KeyError):
    """A word primitive the circuit's `boolean` map gives no body."""


def _decoded(c: Any) -> Any:
    if isinstance(c, (bytes, str, Mapping)):
        from .restrict import program

        return program(c if isinstance(c, bytes) else (c.encode() if isinstance(c, str) else json.dumps(dict(c)).encode()))
    return c


def count(c: Any) -> dict[str, Count]:
    """Each Definition's count per activation (module docstring), by descriptor key: every Definition the root of ``c``
    reaches, each body of its `boolean` map and what that body reaches, the root included. ``c`` is a descriptor (its bytes,
    text or JSON object, decoded from its bytes alone) or a Program, built or decoded. `NoBody` names a word primitive the
    map lacks."""
    P = _decoded(c)
    bodies = P.boolean
    memo: dict[int, tuple[Any, Count]] = {}
    out: dict[str, Count] = {}

    def of(root: Any) -> Count:
        stack: list = [(root, False)]
        while stack:
            fn, ready = stack.pop()
            if id(fn) in memo:
                continue
            if isinstance(fn, PrimitiveDefinition):
                g = gate(fn)
                if g is None:
                    body = bodies.get(fn.id)
                    if body is None:
                        raise NoBody(f"the circuit's boolean map names no body for {fn.id}")
                    if id(body) not in memo:
                        stack += [(fn, False), (body, False)]
                        continue
                    g = memo[id(body)][1]
                memo[id(fn)] = (fn, g)
            elif ready:
                total = ZERO
                for nd in fn.body.nodes:
                    total += memo[id(nd.fn)][1] * n_members(nd)
                memo[id(fn)] = (fn, total)
            else:
                stack.append((fn, True))
                stack += [(nd.fn, False) for nd in fn.body.nodes if id(nd.fn) not in memo]
            if id(fn) in memo:
                out[descriptor_id(fn)] = memo[id(fn)][1]
        return memo[id(root)][1]

    of(P.root)
    for body in bodies.values():
        of(body)
    return out


def table(c: Any) -> dict[str, int]:
    """Each primitive the root of ``c`` reaches, inputs aside, by descriptor key at its two-input gates: a gate by the rule, a
    word primitive by its body. A count of a block reads it (`InferenceOnlyCheck.Count`'s ``table``)."""
    P = _decoded(c)
    got = count(P)
    keys: set[str] = set()
    seen: set[int] = set()
    todo = [P.root]
    while todo:
        fn = todo.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        if isinstance(fn, PrimitiveDefinition):
            if input_id(fn.id) is None:
                keys.add(descriptor_id(fn))
        else:
            todo += [nd.fn for nd in fn.body.nodes]
    return {k: got[k].gates for k in sorted(keys)}


def of_block(c: bytes, partition: Mapping[str, Any], units: Sequence[tuple[int, int]]) -> Count:
    """The count of the proof block of ``units`` (`restrict.restrict`'s block of C): the root of its Program."""
    from .restrict import restrict

    b = restrict(c, partition, units)
    return count(b.program)[descriptor_id(b.program.root)]


__all__ = ["Count", "NoBody", "ZERO", "count", "gate", "of_block", "table"]

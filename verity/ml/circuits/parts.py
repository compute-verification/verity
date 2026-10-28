"""Shared, frozen interface types for the general structural query API.

Circuit terminology (gate-centric): DAG nodes are *gates*, edges are *wires*.  Input gates are gates whose values the
circuit input supplies; some gates are designated output gates (`Program.root_outputs()`).  Every gate produces one
`Value<w>`.  For a gate set S:

    in(S)  = { u not in S : exists v in S, (u, v) in E }
    out(S) = { u in S : u designated output  or  exists v not in S, (u, v) in E }
    w_out(S) = sum_{g in out(S)} w(g)

Both are sets of *boundary gates* (a producer counts once regardless of fan-out).  A declared but unread parameter is not
a wire; an unread nested return is not automatically an escaping value.

This module is the only thing the four query-track modules share by import:

    verity/ml/circuits/query_ast.py  GateSet, Family, Query AST, evaluate       (Query API owner)
    verity/ml/circuits/boundary.py   compact in/out/w_out over Parts            (Boundary owner)
    verity/ml/circuits/partition.py  partition + width validation, verdicts     (Partition owner)
    tests/…                 oracle suite, fixtures, canonical queries  (Query-test owner)

Add to it only by agreement of the owners; never remove or rename.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .defs import SpecializedDefinition
    from .layout import Scope

# (node index j in the body, member m) -- m = 0 for primitive / call nodes, the batch member or scan iteration otherwise.
Item = tuple[int, int]


@dataclass(frozen=True)
class Part:
    """A set of gates inside ONE activation of a specialization: the union of the gate intervals of `items` of that
    activation's body.  `scope` is the layout Scope of the activation (its `offset` is the root gate index of its first
    gate; `scope.spec.body` is the body the items index).  Parts are the structural currency of every module: a GateSet
    is a union of Parts, a Family is an indexable sequence of GateSets, boundary analysis takes a Part.

    Invariant: every (j, m) in `items` is a valid item of `scope.spec.body` (0 <= j < len(nodes); 0 <= m < n_members(j))."""
    scope: "Scope"
    items: frozenset[Item]

    @property
    def spec(self) -> "SpecializedDefinition":
        return self.scope.spec

    def intervals(self) -> list[tuple[int, int]]:
        """Half-open root gate intervals [lo, hi) of the items, ascending, adjacent runs merged.  O(|items| log |items|)."""
        body = self.scope.spec.body
        ivs = []
        for j, m in self.items:
            node = body.nodes[j]
            lo = self.scope.offset + body.offsets[j] + m * node.fn.gates
            ivs.append((lo, lo + node.fn.gates))
        ivs.sort()
        out: list[tuple[int, int]] = []
        for lo, hi in ivs:
            if out and out[-1][1] == lo:
                out[-1] = (out[-1][0], hi)
            else:
                out.append((lo, hi))
        return out

    def size(self) -> int:
        return sum(hi - lo for lo, hi in self.intervals())


@dataclass(frozen=True)
class Unsupported:
    """The honest non-answer: the compact analysis cannot justify an exact result for this input.  Callers must treat it
    as *neither* pass nor fail (RA discipline).  `reason` is a short machine-readable code, `detail` free text."""
    reason: str
    detail: str = ""

    def __bool__(self) -> bool:  # so `if result:` never mistakes Unsupported for a value
        return False


class Limits:
    """Caps that turn an expensive answer into Unsupported rather than into an unbounded computation."""
    max_intervals: int = 5_000_000       # tier-B partition sweep (interval endpoints)
    max_members_exact: int = 4_096       # per-member exact out() before falling back to Unsupported
    max_items_per_part: int = 1_000_000


def n_members(node) -> int:
    """Members of a body node: 1 for primitive/call, `n` for batch/scan."""
    return node.n if node.form in ("batch", "scan") else 1

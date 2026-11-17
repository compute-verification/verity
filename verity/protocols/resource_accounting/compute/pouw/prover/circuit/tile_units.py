"""What PoUW's tile checks share in building and cutting their Programs: the body emitter (`_Emit`), the ok row's word
(`ok_word`), the hiding roots' tile digests (`DIGESTS`), the cut's attributes (`RowK`), and the units of a Program
built from unit templates with its cut (`ProofBlock`, `ROLES`, `units`, `unit_cut`).  Pearl-C's tile check (`pc8`), its
inert-work bound (`cap`), its row cut (`rowk`), Pearl+'s tile (`pearl_plus_tile`) and experimental tile checks build on
them; none of them is a Definition, so where they live changes no gate.
"""
from __future__ import annotations

import bisect
import struct
from dataclasses import dataclass, field
from typing import Any

from verity.circuits.codec import descriptor_id
from verity.circuits.cut import CallGraph, check_cut
from verity.circuits.defs import SpecializedDefinition
from verity.circuits.partition import CutVerdict
from verity.circuits.types import Array
from verity.circuits.ops.prims import const
from verity.circuits.ops.scalar import SelectBf16

from . import boolean as _boolean  # noqa: F401  registers the _v2 Definitions Flock's lowering falls back to
from . import leaves as L


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


class _Emit:
    """A body under construction: integer operands are constants of the parameter's width, one node per value and body, and
    an operation on constants alone is evaluated, not emitted."""

    def __init__(self, B) -> None:
        self.B = B
        self._consts: dict[tuple[int, int], Any] = {}

    def c(self, w: int, v: int):
        got = self._consts.get((w, v))
        if got is None:
            got = self._consts[(w, v)] = self.B.call(const(w, v))
        return got

    def call(self, fn, *args):
        if all(isinstance(a, int) for a in args):
            return fn.evaluate(*args)
        return self.B.call(fn, *(self.c(t.w, a) if isinstance(a, int) else a for a, (_n, t) in zip(args, fn.params)))


def ok_word(e: _Emit, ok):
    """The ok row's word of the admission bit ``ok`` (`leaves.OK`): a select between the constants 1 and 0, so bit 0 is
    ``ok`` and bits 1-15 are zero in gates, never free bits of the committed row."""
    return e.call(SelectBf16, ok, 1, 0)


DIGESTS = lambda S_: Array(S_.M // S_.TM, Array(S_.N // S_.TN, L.DIGEST))          # noqa: E731


class RowK:
    """A tile check's `Shape` attribute (``ports``, ``derivations``) that is ``today`` on the class and on a shape without
    ``row_k``, and with ``row_k`` set `rowk`'s attribute ``name``: the cut's roles and anchors.  So the class and every
    shape without the cut read today's dicts, which `rowk` leaves alone."""

    def __init__(self, today: dict, name: str) -> None:
        self.today, self.name = today, name

    def __get__(self, shape, cls=None) -> dict:
        if shape is None or shape.row_k is None:
            return self.today
        from . import rowk
        return getattr(rowk, self.name)


# ---- the partition -------------------------------------------------------------------------------------------------------

ROLES = {"weight": ("row_b",), "call": ("row_a", "tile"), "check": ("row_b", "row_a", "tile")}


@dataclass(frozen=True)
class ProofBlock:
    index: int
    role: str                   # a key of a tile check's `templates`, or of `rowk.ROLES`
    node: int                   # its root-body node
    member: int                 # its member of that batch node
    template: str = ""          # its template's descriptor id (`plan.template_strata`'s stratum)
    where: tuple[int, ...] = field(default=())    # a row unit's row; a tile or tally unit's (tile row, tile column); a
                                                  # word unit's (row, column); a `rowk` form's (row, segment)

#: the roles whose units sit on a grid, one batch node per grid row (`cap`'s word, row and tally units beside the tile)
GRID_ROLES = ("tile", "word", "row_digest", "row_tally", "tally")


def units(fn: SpecializedDefinition, role_of: dict[int, str]) -> tuple[ProofBlock, ...]:
    """Every unit of a Program built from the unit templates, in the partition's numbering (node order, then member);
    ``role_of`` maps each template's Definition (by id) to its role."""
    from . import rowk

    out: list[ProofBlock] = []
    grid_rows = dict.fromkeys(GRID_ROLES, 0)
    for j, nd in enumerate(fn.body.nodes):
        role, tid = role_of[id(nd.fn.defn)], descriptor_id(nd.fn)
        for m in range(nd.n if nd.form == "batch" else 1):
            where = (grid_rows[role], m) if role in grid_rows else (m, rowk.segment(nd.fn)) if role in rowk.FORMS else (m,)
            out.append(ProofBlock(len(out), role, j, m, tid, where))
        if role in grid_rows:
            grid_rows[role] += 1
    return tuple(out)


def unit_cut(fn: SpecializedDefinition, role_of: dict[int, str]) -> tuple[CallGraph, list[int], CutVerdict]:
    """The whole Program's cut with each computing gate owned by its root node's member, checked by `cut.check_cut`
    (every gate certified once, reads across units only through committed values, no value computed in two units), with no
    width rule: no unit here reaches a served output."""
    body, us = fn.body, units(fn, role_of)
    first, at = [], 0
    for nd in body.nodes:
        first.append(at)
        at += nd.n if nd.form == "batch" else 1
    g = CallGraph(fn)
    owner = []
    for c in range(g.n):
        j = g.keep[c]
        node = bisect.bisect_right(body.offsets, j) - 1
        nd = body.nodes[node]
        owner.append(first[node] + ((j - body.offsets[node]) // nd.fn.gates if nd.form == "batch" else 0))
    assert len(us) == at
    return g, owner, check_cut(g, owner, limits=None)


__all__ = ["DIGESTS", "GRID_ROLES", "ROLES", "ProofBlock", "RowK", "ok_word", "unit_cut", "units"]

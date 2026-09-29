"""The circuits' partition: `Q_template_instances` over the unit templates, its invariant, and the Z separator.

A `Circuit` is one call's Definition with its unit templates by role; `ncp_circuit` is `ncp-v2`'s, `h1t_circuit` H-1T's
(`circuit.h1t`), and the checks below read either.  The functions named for `ncp-v2` take its (shape, kind) as before.

* `query(shape, kind)` / `partition(program, ...)`: core's `verity/partition/v1` object naming the templates (`ncp2.templates`;
  `NcpNode` only when the call has two strips or more, since a listed template needs an instance). The weight's own
  Program (`NcpWeight`, its Y units) has `weight_query`; `window_units` and `window_cut` number and cut C calls on one
  weight in one root (`NcpWindow`).
* `units(shape, kind)`: every unit with its role, its template's descriptor id, and where it sits (strip, column strip, tile,
  tree node), numbered as `partition_object.evaluate` numbers them (node order, then member).
* `unit_cut(shape, kind)`: the whole call as one flat `cut.CallGraph` with the owner of every computing gate, checked by
  `cut.check_cut` (`partition.validate_unit_cut`: every gate certified once, reads across units only through committed
  values, and **no value computed in two units**), without the width rule.
* `separator(shape, kind)`: the downstream cut at Z (Draft 2, Definition 4.6 and Lemma 4.7). Every path from a gate of a
  PoUW unit (digest, node, key, form-X, form-Y, tile) to a delivered output (y) passes through a tile's Z, and what crosses
  is Z alone: m·n values of 32 bits, 8,192 bits per tile. The credited words are Program outputs committed for the verifier
  and never delivered; they reach y only through Z.
"""
from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from verity.ir import partition_object as PO
from verity.ir.codec import descriptor_id
from verity.ir.cut import CallGraph, boundary_widths, check_cut, fits
from verity.ir.partition import CutVerdict
from verity.ir.program import Program

from . import h1t, ncp2
from .ncp2 import D, Shape

#: the units whose values are PoUW's alone: behind the Z separator
POUW_ROLES = ("digest", "node", "key", "form_y", "form_x", "tile")
#: H-1T's: the strips' salt keys, the forming and the tiles
H1T_POUW_ROLES = ("key", "form_x", "tile")


@dataclass(frozen=True)
class Circuit:
    """One call's Definition and its unit templates by role."""

    fn: Any
    templates: Any                      # role -> unit template (the tile's is the separator port's)
    m: int
    n: int
    pouw_roles: tuple[str, ...]
    variants: tuple[tuple[str, Any], ...] = ()   # (role, template) for every template, when a role has several

    @property
    def tiles(self) -> int:
        return (self.m // D) * (self.n // D)


def ncp_circuit(shape: Shape, kind: str = "shift24", fn: Any = None) -> Circuit:
    return Circuit(ncp2.linear(shape, kind) if fn is None else fn, ncp2.templates(shape.k, kind), shape.m, shape.n, POUW_ROLES)


def h1t_circuit(shape: h1t.Shape, fn: Any = None) -> Circuit:
    return Circuit(h1t.linear(shape) if fn is None else fn, h1t.templates(shape.k), shape.m, shape.n, H1T_POUW_ROLES)


def h1t_window_circuit(shape: h1t.Shape, calls: int) -> Circuit:
    """``calls`` H-1T calls on one weight in one root (`h1t.window`)."""
    return h1t_circuit(shape, h1t.window(shape, calls))


def listed_of(fn: Any) -> list[Any]:
    """A call's templates in the order its root body first uses them."""
    out: list[Any] = []
    for nd in fn.body.nodes:
        if all(nd.fn is not t for t in out):
            out.append(nd.fn)
    return out


def query_of(c: Circuit) -> dict[str, Any]:
    return PO.template_instances_query(listed_of(c.fn))


def listed(shape: Shape, kind: str = "shift24") -> list[Any]:
    """The templates the call instantiates, in the order its root body first uses them."""
    t = ncp2.templates(shape.k, kind)
    order = ["digest"] + (["node"] if shape.m // D > 1 else []) + ["key", "form_x", "tile", "dequant"]
    return [t[r] for r in order]


def weight_query(shape: Shape, kind: str = "shift24") -> dict[str, Any]:
    """The weight Program's query: its Y units (`NcpFormY`), one per 16-column strip."""
    return PO.template_instances_query([ncp2.templates(shape.k, kind)["form_y"]])


@lru_cache(maxsize=8)
def weight_program(shape: Shape, kind: str = "shift24") -> Program:
    return Program(ncp2.weight(shape, kind))


def query(shape: Shape, kind: str = "shift24") -> dict[str, Any]:
    return PO.template_instances_query(listed(shape, kind))


@lru_cache(maxsize=8)
def program(shape: Shape, kind: str = "shift24") -> Program:
    return Program(ncp2.linear(shape, kind))


def partition(shape: Shape, kind: str = "shift24") -> dict[str, Any]:
    """The `verity/partition/v1` object."""
    return PO.build(program(shape, kind), query(shape, kind))


@dataclass(frozen=True)
class Unit:
    index: int
    role: str                   # a key of `ncp2.templates`
    template: str               # its descriptor id (the stratum key)
    node: int                   # its root-body node in `ncp2.linear`'s body
    member: int                 # its member of a batch node (0 for a call)
    where: tuple[int, ...] = field(default=())   # strip s; column strip c; tile (rs, cs); dequant (rs, cs, e); node: ()


@lru_cache(maxsize=8)
def units(shape: Shape, kind: str = "shift24") -> tuple[Unit, ...]:
    """Every unit, in the partition's numbering."""
    return units_of(ncp2.linear(shape, kind), shape, kind)


def units_of(fn: Any, shape: Shape, kind: str = "shift24") -> tuple[Unit, ...]:
    """Every unit of a call Definition built from the unit templates (`ncp2.linear`, or a variant a test builds)."""
    return circuit_units(ncp_circuit(shape, kind, fn))


def circuit_units(c: Circuit) -> tuple[Unit, ...]:
    """Every unit of a circuit, in the partition's numbering."""
    roles = {id(v): r for r, v in (*c.templates.items(), *c.variants)}
    ms, ns = c.m // D, c.n // D
    out: list[Unit] = []
    tile_row = deq_node = 0
    strips: Counter = Counter()                            # a strip role's units so far, across its batch nodes
    for k, nd in enumerate(c.fn.body.nodes):
        role = roles[id(nd.fn)]
        count = nd.n if nd.form == "batch" else 1
        for m in range(count):
            if role in ("digest", "form_x", "form_y", "pad"):
                where: tuple[int, ...] = (strips[role] + m,)
            elif role == "tile":
                where = (tile_row % ms, m)
            elif role == "dequant":
                where = ((deq_node // ns) % ms, deq_node % ns, m)
            else:
                where = ()
            out.append(Unit(len(out), role, descriptor_id(nd.fn), k, m, where))
        if role == "tile":
            tile_row += 1
        elif role == "dequant":
            deq_node += 1
        elif role in ("digest", "form_x", "form_y", "pad"):
            strips[role] += count
    assert tile_row % max(ms, 1) == 0
    return tuple(out)


def window_units(shape: Shape, calls: int, kind: str = "shift24") -> tuple[Unit, ...]:
    """`NcpWindow`'s units: the weight's Y units, then each call's, in the partition's numbering."""
    return units_of(ncp2.window(shape, calls, kind), shape, kind)


def window_cut(shape: Shape, calls: int, kind: str = "shift24") -> "UnitCut":
    """C calls on one weight in one root, cut as one Program: the invariant over a window's composition."""
    return unit_cut_of(ncp2.window(shape, calls, kind), shape, kind)


@dataclass
class UnitCut:
    graph: CallGraph
    owner: list[int]                                     # the unit of each computing gate (CallGraph order)
    units: tuple[Unit, ...]
    verdict: CutVerdict


@lru_cache(maxsize=4)
def unit_cut(shape: Shape, kind: str = "shift24") -> UnitCut:
    """The whole call's cut: owners from the root body's node and member ranges, checked with no width rule."""
    return unit_cut_of(ncp2.linear(shape, kind), shape, kind)


def unit_cut_of(fn: Any, shape: Shape, kind: str = "shift24") -> UnitCut:
    return circuit_cut(ncp_circuit(shape, kind, fn))


def circuit_cut(c: Circuit) -> UnitCut:
    """A circuit's whole-call cut, checked by `cut.check_cut` with no width rule."""
    fn = c.fn
    body = fn.body
    us = circuit_units(c)
    first: list[int] = []                                 # the first unit of each root node
    at = 0
    for nd in body.nodes:
        first.append(at)
        at += nd.n if nd.form == "batch" else 1
    g = CallGraph(fn)
    owner = []
    for c in range(g.n):
        j = g.keep[c]
        k = bisect.bisect_right(body.offsets, j) - 1
        nd = body.nodes[k]
        m = (j - body.offsets[k]) // nd.fn.gates if nd.form == "batch" else 0
        owner.append(first[k] + m)
    return UnitCut(g, owner, us, check_cut(g, owner, limits=None))


@dataclass(frozen=True)
class Separator:
    ok: bool                    # the pinned Z ports are a downstream cut for every PoUW unit toward the served outputs
    z_gates: int                # the Z values the served path reads
    width_bits: int             # their total declared width
    per_tile_bits: int
    reached_pouw: int           # PoUW-unit gates reached backward from y without crossing a pinned port (0 when ok)
    words_delivered: int        # credited words among y's gates (0)


def separator(shape: Shape, kind: str = "shift24") -> Separator:
    """Trace backward from the delivered outputs (y) without crossing a pinned separator port (the tiles' Z): no gate of a
    PoUW unit (digest, node, key, form-X, form-Y, tile) may be reached, and the ports the trace stops at are every tile's Z
    (Draft 2, Definition 4.6 and Lemma 4.7)."""
    return circuit_separator(ncp_circuit(shape, kind), unit_cut(shape, kind))


def circuit_separator(c: Circuit, uc: UnitCut | None = None) -> Separator:
    """`separator` for any circuit: no gate of one of its PoUW roles is reached from y without crossing a pinned port."""
    uc = circuit_cut(c) if uc is None else uc
    g, owner, us = uc.graph, uc.owner, uc.units
    ports = _port_gates(uc, c)
    preds: list[list[int]] = [[] for _ in range(g.n)]
    for p, q in zip(g.src, g.dst):
        preds[q].append(p)
    y_gates = list(g.outputs[:c.m * c.n])
    words_delivered = sum(1 for q in y_gates if us[owner[q]].role != "dequant")
    seen, stack, stopped, reached = set(y_gates), list(y_gates), set(), 0
    while stack:
        q = stack.pop()
        for p in preds[q]:
            if p in seen:
                continue
            seen.add(p)
            if p in ports:
                stopped.add(p)
            elif us[owner[p]].role in c.pouw_roles:
                reached += 1
            else:
                stack.append(p)
    width = sum(int(g.width[p]) for p in stopped)
    return Separator(reached == 0 and stopped == ports and words_delivered == 0, len(stopped), width, width // c.tiles, reached,
                     words_delivered)


#: The separator ports the protocol pins, not the registrant: a template role and the return field that crosses the
#: separator. For NCP-INT's layout (a) it is the tile's Z, field 1 of `NcpTile`'s return (the red team's X-SPC-66).
SEPARATOR_PORTS: dict[str, int] = {"tile": 1}


def _port_gates(uc: UnitCut, c: Circuit) -> set[int]:
    """The computing gates (CallGraph order) of every pinned separator port of every unit."""
    body, g = c.fn.body, uc.graph
    index = {j: q for q, j in enumerate(g.keep)}
    tmpl = c.templates
    local: dict[str, list[int]] = {}
    for role, field_ in SEPARATOR_PORTS.items():
        t = tmpl[role]
        _, rt = t.signature()
        lo = rt.field_offset(field_)
        gates = []
        for j in range(lo, lo + rt.fields[field_].leaves):
            space, node, leaf = t.body.ret[j]
            if space != "n" or t.body.nodes[node].form != "primitive":
                raise AssertionError(f"{t.id}: separator port leaf {j} is not a primitive's output")
            gates.append(t.body.offsets[node] + leaf)
        local[role] = gates
    out: set[int] = set()
    for u in uc.units:
        if u.role in local:
            base = body.offsets[u.node] + u.member * body.nodes[u.node].fn.gates
            out.update(index[base + j] for j in local[u.role])
    return out


@dataclass(frozen=True)
class WidthRule:
    ok: bool                              # every unit that reaches a served output other than through a pinned port fits
    checked: tuple[int, ...]              # those units (reachable from the served outputs without crossing a port)
    too_wide: tuple[tuple[int, int, int], ...]   # (unit, out bits, out gates) of each checked unit that fails
    exempt: int                           # units behind the separator
    separator_bits: int                   # the pinned ports' total width


def width_rule(shape: Shape, kind: str = "shift24") -> WidthRule:
    """The width rule by reachability, with the separator pinned (X-SPC-60, X-SPC-66), on the protocol's own Program
    (`ncp2.linear`, which the verifier builds; X-SPC-73): remove the pinned ports' values, find every unit from which a
    served output (y's m·n entries, the protocol's designation) is still reachable, and hold each such unit to
    `cut.WIDTH_RULE`; every other unit is behind the separator and exempt. The ports are the protocol's
    (`SEPARATOR_PORTS`), so a wide unit cannot name its own output a separator: a pinned tile whose word leaf reaches a
    served output is itself reachable, and fails."""
    return width_rule_of(ncp2.linear(shape, kind), shape, kind, shape.m * shape.n)


def width_rule_of(fn: Any, shape: Shape, kind: str, served: int) -> WidthRule:
    """`width_rule` on a given Definition and served count: the check a test runs on a variant. It is a property of the
    Definition checked, so it certifies nothing about any other; only `width_rule`'s Program is the protocol's."""
    return circuit_width_rule(ncp_circuit(shape, kind, fn), served)


def circuit_width_rule(c: Circuit, served: int | None = None, uc: UnitCut | None = None) -> WidthRule:
    """`width_rule` for any circuit: its served outputs are its first ``served`` root outputs (y's m·n by default)."""
    served = c.m * c.n if served is None else served
    if not 1 <= served:
        raise ValueError("the served outputs are at least one: a width rule over none checks nothing")
    uc = circuit_cut(c) if uc is None else uc
    g, owner = uc.graph, uc.owner
    ports = _port_gates(uc, c)
    preds: list[list[int]] = [[] for _ in range(g.n)]
    for p, q in zip(g.src, g.dst):
        preds[q].append(p)
    served_gates = list(g.outputs[:served])
    seen, stack = set(served_gates), list(served_gates)
    while stack:
        q = stack.pop()
        for p in preds[q]:
            if p not in seen and p not in ports:
                seen.add(p)
                stack.append(p)
    reached = sorted({owner[q] for q in seen})
    ob, og = boundary_widths(g, owner)
    bad = tuple((u, int(ob[u]), int(og[u])) for u in reached if not fits(ob[u], og[u]))
    bits = sum(int(g.width[p]) for p in ports)
    return WidthRule(not bad, tuple(reached), bad, len(uc.units) - len(reached), bits)


def boundary_bits(shape: Shape, kind: str = "shift24") -> dict[str, Counter]:
    """Each role's units by their outgoing boundary width in bits (`cut.boundary_widths`): what the width rule would read."""
    uc = unit_cut(shape, kind)
    ob, _og = boundary_widths(uc.graph, uc.owner)
    out: dict[str, Counter] = {}
    for u, bits in zip(uc.units, ob):
        out.setdefault(u.role, Counter())[int(bits)] += 1
    return out


__all__ = ["H1T_POUW_ROLES", "POUW_ROLES", "SEPARATOR_PORTS", "Circuit", "Separator", "Unit", "UnitCut", "WidthRule",
           "boundary_bits", "circuit_cut", "circuit_separator", "circuit_units", "circuit_width_rule", "h1t_circuit", "h1t_window_circuit",
           "listed", "listed_of", "ncp_circuit", "partition", "program", "query", "query_of", "separator", "unit_cut",
           "unit_cut_of", "units", "units_of", "weight_program", "weight_query", "width_rule", "width_rule_of", "window_cut",
           "window_units"]

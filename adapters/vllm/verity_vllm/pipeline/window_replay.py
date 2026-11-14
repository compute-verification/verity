"""A window as one replay Definition (`replay`): an `interface.Circuit`'s templates, units, cuts and wires restated as a
composite Definition's body, so what reads a replay block's wiring from its registered Definition reads the window's.

**The body** is one node a range of a template's units, a call for one unit and a batch for more, in an order of the
ranges in which every range comes after the ranges it reads and each template's ranges come in unit order: unit u of a
template is the u-th member of its nodes in body order. The ranges are the circuit's own (`Circuit.order`), cut again at
each wire's first and end reader unit, so within a node each row of a port has one source.

**Parameters**: one for each template's port with a row no wire fills, ``<descriptor id>/<port>``, an array of those
rows by unit then row; a row is the port, or one of the k rows a port cut k ways is read as (`RowLayout.cut`): its
outer elements ``[j·n/k, (j+1)·n/k)``. What fills such a row (a registered read, an ingress, nothing) is the circuit's
to say, not the body's.

**The return**: a tuple of one array for each output port the circuit names external (`Circuit.external`), that
output of each of its template's units in unit order; the empty tuple for a circuit that names none.

**A wire** is restated in the argument of the node that reads it: the reader row is the writer unit's output field,
whole, a reference into the writer's node at leaf ``(unit - node's first)·(leaves a unit returns) + (the field's first
leaf)``. A pattern's mixed radix is the reference's strides: a `Strided` holds two levels, so the row's leaves and one
digit are one part, digits whose steps continue each other merge (``step[k] = radix[k+1]·step[k+1]``), and every further
digit costs a part for each of its values. A part ends where the writer's units pass to another node. A port every
member of a batch reads at one writer row is a shared argument; any other is mapped over the batch's members, and a
member whose port's rows come from more than one source reads a concat of them, a part a source a member.

**The id** names the window by its record: ``window`` is the SHA-512 of its ``circuit.json`` less the fields its run
fixes (`interface.RUN_FIELDS`), in canonical JSON, so every run of one circuit gives one Definition. The Definition is a reference a test evaluates; no verifier reads it.

Refused (ValueError): a wire whose reader row isn't its writer's output field leaf for leaf (the same words at another
leaf width, as a Boolean template's bits read as words: a body's argument names each leaf at its own width); a
circuit whose order runs a template's later units before its earlier ones; a record that isn't the circuit's (one that
lists its members states no wires); and a return of more than 262,144 bits, the widest proof block's
(`layout.UNIT_MAX_BITS`)."""
from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any

import numpy as np
from verity.circuits.codec import canonical_json
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.refs import Coll, Refs, concat, strided_view
from verity.circuits.types import Array, Tuple, Type
from verity.commitments.layout import UNIT_MAX_BITS
from verity.commitments.windows import interface as IF

NAME, VERSION = "WindowReplay", 1

#: a view of one source's leaves, whole rows: leaf j is ``base + (j // inner)·outer + (j % inner)·step``, as a `Strided`'s
View = tuple[int, int, int, int, int]


def _levels(levels: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """A mixed radix's ``(radix, stride)`` digits, the first the slowest, as the fewest that give each index the same leaf:
    without the digits of radix 1, and two neighbours one where the outer's stride is the inner's whole extent."""
    out = [(r, s) for r, s in levels if r > 1]
    for i in range(len(out) - 2, -1, -1):
        (ra, sa), (rb, sb) = out[i], out[i + 1]
        if sa == rb * sb:
            out[i:i + 2] = [(ra * rb, sb)]
    return out


def _views(base: int, levels: list[tuple[int, int]], lo: int, hi: int) -> list[View]:
    """Indices ``[lo, hi)`` of the map index -> ``base + sum(digit·stride)`` over ``levels`` (`_levels`'), in order, as
    views: one for a map of two levels or fewer over whole rows, else one for each value of each digit above the last two
    and for each row cut short."""
    if lo >= hi:
        return []
    if not levels:
        return [(base, 1, 0, 0, 1)]
    (_r0, s0), rest = levels[0], levels[1:]
    if not rest:
        return [(base + lo * s0, 1, s0, 0, hi - lo)]
    size = math.prod(r for r, _ in rest)
    a, b = -(-lo // size), hi // size
    if len(rest) == 1 and a < b:
        return (_views(base + (a - 1) * s0, rest, lo - (a - 1) * size, size)
                + [(base + a * s0, size, s0, rest[0][1], (b - a) * size)]
                + _views(base + b * s0, rest, 0, hi - b * size))
    return [v for q in range(lo // size, (hi - 1) // size + 1)
            for v in _views(base + q * s0, rest, max(lo - q * size, 0), min(hi - q * size, size))]


def _split(view: View, bounds: list[int]) -> list[tuple[int, View]]:
    """``view`` cut where its leaves pass from one source to the next, each source's first leaf in ``bounds`` (sorted, the
    first 0): ``(source, its part)`` in order. A view inside one source costs no pass over its rows."""
    base, inner, outer, step, count = view
    rows, reach = count // inner, (inner - 1) * step
    ends = (base + min(0, reach), base + max(0, reach), base + (rows - 1) * outer + min(0, reach),
            base + (rows - 1) * outer + max(0, reach))
    if (k := bisect.bisect_right(bounds, min(ends)) - 1) == bisect.bisect_right(bounds, max(ends)) - 1:
        return [(k, view)]
    q = np.arange(rows, dtype=np.int64) * outer + base
    first, last = (np.searchsorted(bounds, q + x, side="right") - 1 for x in (min(0, reach), max(0, reach)))
    key = np.where(first == last, first, -1)
    cuts = [0, *(np.flatnonzero(np.diff(key)) + 1).tolist(), rows]
    out: list[tuple[int, View]] = []
    for a, b in zip(cuts, cuts[1:]):
        if key[a] >= 0:
            out.append((int(key[a]), (base + a * outer, inner, outer, step, (b - a) * inner)))
        else:
            out += [x for r in range(a, b) for x in _split((base + r * outer, 1, step, 0, inner), bounds)]
    return out


def _refs(c: Coll, view: View) -> Refs:
    """``view`` of ``c``'s leaves: a slice where its leaves are consecutive."""
    base, inner, outer, step, count = view
    if count <= inner:
        inner, outer, step = 1, step, 0
    if count == 1 or (inner == 1 and outer == 1) or (step == 1 and outer == inner):
        return c.refs.slice(base, base + count)
    return strided_view(c, base, inner, outer, step, count)


def _fields(ret: Type) -> tuple[tuple[int, Type], ...]:
    """A return's fields as `interface.layout` names them (``o<j>``): each one's first leaf and type."""
    types = tuple(f[1] if isinstance(f, tuple) else f for f in (getattr(ret, "fields", None) or (ret,)))
    starts = [0]
    for t in types:
        starts.append(starts[-1] + t.leaves)
    return tuple(zip(starts, types))


@dataclass
class _Port:
    """A template's port: its row's type and leaves, its parameter (None with no free row), its wires by first reader
    unit, and, as its template's nodes are laid out in unit order, the wires reading the node and the free rows before it."""
    row: Type
    leaves: int
    k: int
    wires: list[IF.Wire]
    param: int | None = None
    live: list[IF.Wire] = field(default_factory=list)
    nxt: int = 0
    free: int = 0


@dataclass(frozen=True)
class _Plan:
    """What the body emits: its nodes ``(template, first unit, units)`` in order; each node's arguments, ``(axis, type,
    pieces)``, a piece ``(space, index, view)`` of a parameter (``p``) or an earlier node's output (``n``); and the
    return's pieces."""
    params: tuple[tuple[str, Type], ...]
    ret: Type
    nodes: tuple[tuple[Any, int, int], ...]
    args: tuple[tuple[tuple[int | None, Type, tuple[tuple[str, int, View], ...]], ...], ...]
    out: tuple[tuple[str, int, View], ...]


def _ranges(circuit: IF.Circuit) -> tuple[tuple[str, int, int], ...]:
    """The body's ranges in order (the module docstring)."""
    units, wires = circuit.units, circuit.wires
    starts: dict[str, set[int]] = {t: {0} for t in units}
    for t, at, _n in circuit.order or ():
        starts[t].add(at)
    for w in wires:
        starts[w.reader] |= {w.at, w.at + w.reader_units}
    chain = [IF.Wire(t, "", t, "", s, (1,), s - 1, (0,)) for t, ss in starts.items() for s in sorted(ss) if 0 < s < units[t]]
    try:
        return IF.ordered(units, [*wires, *chain], {t: sorted(ss) for t, ss in starts.items()})
    except ValueError as e:
        raise ValueError(f"{circuit.name}: no order of its units' ranges runs each template's units in unit order, as a "
                         f"body numbers them ({e})") from None


def _plan(circuit: IF.Circuit) -> _Plan:
    fns = {tid: circuit.template(tid) for tid in circuit.ids}
    sigs = {tid: fn.signature() for tid, fn in fns.items()}
    outs = {tid: _fields(ret) for tid, (_ps, ret) in sigs.items()}
    units = circuit.units
    ports: dict[tuple[str, str], _Port] = {}
    params: list[tuple[str, Type]] = []
    for tid, (ps, _ret) in sigs.items():
        for p, ty in ps:
            k = int(circuit.cuts.get(tid, {}).get(p, 1))
            ports[tid, p] = _Port(ty if k == 1 else Array(ty.n // k, ty.elem), ty.leaves // k, k, [])
    for w in circuit.wires:
        port, (_f0, got) = ports[w.reader, w.port], outs[w.writer][int(w.out[1:])]
        if port.row.leaf_widths() != got.leaf_widths():
            raise ValueError(f"{circuit.name}: wire {w.doc()}: a row of {w.port} is {port.row!r} and {w.out} is {got!r}; a "
                             "body's argument names each leaf at its own width, so a wire a body restates reads a field of "
                             "its row's leaves")
        port.wires.append(w)
    for (tid, p), port in ports.items():
        port.wires.sort(key=lambda w: w.at)
        if free := units[tid] * port.k - sum(w.units for w in port.wires):
            port.param = len(params)
            params.append((f"{tid}/{p}", Array(free, port.row)))
    returned = [(tid, int(o[1:])) for tid, _, o in (x.rpartition("/") for x in circuit.external)
                if o not in dict(sigs[tid][0])]

    ranges = _ranges(circuit)
    where: dict[str, tuple[list[int], list[int]]] = {t: ([], []) for t in units}
    for i, (t, at, _n) in enumerate(ranges):
        where[t][0].append(at)
        where[t][1].append(i)
    leaves = {tid: ret.leaves for tid, (_ps, ret) in sigs.items()}
    bounds = {t: [at * leaves[t] for at in ats] for t, (ats, _) in where.items()}
    maps: dict[IF.Wire, tuple[int, list[tuple[int, int]]]] = {}

    def wire(w: IF.Wire, lo: int, hi: int) -> list[tuple[str, int, View]]:
        """Leaves ``[lo, hi)`` of ``w``'s reader rows, counted from its first reader index, in its writer's nodes."""
        if w not in maps:
            L = leaves[w.writer]
            maps[w] = (w.base * L + outs[w.writer][int(w.out[1:])][0],
                       _levels([(r, s * L) for r, s in zip(w.radix, w.step)] + [(ports[w.reader, w.port].leaves, 1)]))
        (ats, at), L = where[w.writer], leaves[w.writer]
        return [("n", at[k], (b - ats[k] * L, *rest)) for v in _views(*maps[w], lo, hi)
                for k, (b, *rest) in _split(v, bounds[w.writer])]

    args = []
    for t, at, n in ranges:
        node = []
        for p, ty in sigs[t][0]:
            port = ports[t, p]
            while port.nxt < len(port.wires) and port.wires[port.nxt].at <= at:
                port.live.append(port.wires[port.nxt])
                port.nxt += 1
            port.live = [w for w in port.live if at < w.at + w.reader_units]
            rows: list[IF.Wire | None] = [None] * port.k
            for w in port.live:
                rows[w.cut[0]:w.cut[0] + w.cut[1]] = [w] * w.cut[1]
            free = [j for j in range(port.k) if rows[j] is None]
            cuts = [j for j in range(port.k) if j == 0 or rows[j] is not rows[j - 1]] + [port.k]
            bands = [(j0, j1, rows[j0]) for j0, j1 in zip(cuts, cuts[1:])]
            L, first = port.leaves, port.free

            def unit(u: int, bands: Any = bands, free: Any = free, port: _Port = port, L: int = L, first: int = first,
                     rows: int = 1) -> list[tuple[str, int, View]]:
                """Units ``u`` to ``u + rows``' rows of the port; more than one unit only of a port of one band."""
                got: list[tuple[str, int, View]] = []
                for j0, j1, w in bands:
                    if w is None:
                        lo = (first + (u - at) * len(free) + free.index(j0)) * L
                        got.append(("p", port.param, (lo, 1, 1, 0, (rows * (j1 - j0)) * L)))
                    else:
                        i = (u - w.at) * w.cut[1] + j0 - w.cut[0]
                        got += wire(w, i * L, (i + rows * (j1 - j0)) * L)
                return got

            fixed = not free and all(s == 0 for w in port.live
                                     for r, s in zip(w.radix, w.step[:len(w.step) - (w.cut[1] > 1)]) if r > 1)
            if n == 1 or fixed:
                node.append((None, ty, tuple(unit(at))))
            elif len(bands) == 1:
                node.append((0, Array(n, ty), tuple(unit(at, rows=n))))
            else:
                node.append((0, Array(n, ty), tuple(x for u in range(at, at + n) for x in unit(u))))
            port.free += n * len(free)
        args.append(tuple(node))

    out = []
    for tid, j in returned:
        f0, ft = outs[tid][j]
        for at, i in zip(*where[tid]):
            n = ranges[i][2]
            out.append(("n", i, (f0, ft.leaves, leaves[tid], 1, n * ft.leaves)))
    ret = Tuple(*(Array(units[tid], outs[tid][j][1]) for tid, j in returned))
    return _Plan(tuple(params), ret, tuple((fns[t], at, n) for t, at, n in ranges), tuple(args), tuple(out))


def window_digest(record: Mapping[str, Any]) -> str:
    """The SHA-512 of a window's ``circuit.json`` less its run fields (`interface.circuit_of`), in canonical JSON
    (`codec.canonical_json`, no newline after it): the same for every run of one circuit."""
    return hashlib.sha512(canonical_json(IF.circuit_of(record))).hexdigest()


def replay(circuit: IF.Circuit, record: Mapping[str, Any]) -> SpecializedDefinition:
    """``circuit`` as one replay Definition (the module docstring), `WindowReplay_v1{window}`: ``window`` the
    `window_digest` of ``record``, the circuit's record: its templates, wires, declarations and external ports are the
    circuit's. The plan is made once, in work proportional to the circuit's ranges and its arguments' parts, and the body
    built from it wherever the Definition is."""
    plan = _plan(circuit)
    want = {**circuit.description(), "external": list(circuit.external)}
    if got := [k for k in sorted({*want, *IF.DECLARED}) if record.get(k) != want.get(k)]:
        raise ValueError(f"{circuit.name}: the record is not its record: its {', '.join(got)} differ")
    if (bits := sum(plan.ret.leaf_widths())) > UNIT_MAX_BITS:
        raise ValueError(f"{circuit.name}: its external ports return {bits:,} bits; a proof block returns at most "
                         f"{UNIT_MAX_BITS:,}")

    def body(B: Any, S: Any, *params: Coll) -> Coll:
        outs: list[Coll] = []

        def coll(ty: Type, pieces: Any) -> Coll:
            return Coll(ty, concat([_refs(params[i] if space == "p" else outs[i], v) for space, i, v in pieces]))

        for (fn, _at, n), node in zip(plan.nodes, plan.args):
            given = [coll(ty, pieces) for _axis, ty, pieces in node]
            outs.append(B.call(fn, *given) if n == 1 else B.batch(fn, *given, axes=[axis for axis, _ty, _ps in node], n=n))
        return coll(plan.ret, plan.out)

    defn = CompositeDefinition(NAME, VERSION, ("window",), lambda S: (plan.params, plan.ret), body, register=False,
                               conformance="derived: its templates' calls, wired as the window's circuit wires them",
                               doc=f"{circuit.name}: {sum(n for _f, _a, n in plan.nodes)} units in {len(plan.nodes)} nodes")
    return bind(defn, window=window_digest(record))

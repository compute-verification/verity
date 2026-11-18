"""The replay block's links, wires and rows, read from its Definition and its callers in the program's body: what a
two-stage audit's second stage holds each session's ports to (`SecondStage.replayLinks`, `replayWires`, `replayRows`)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.circuits.codec import definitions, descriptor_id
from verity.circuits.defs import PrimitiveDefinition, recut_unread
from verity.circuits.refs import Affine, Concat, Explicit
from verity.circuits.types import Array, RecordType, Tuple


@dataclass(frozen=True)
class Part:
    """A template's part of the interior population: ``q·n`` slots from `lo`, `n` in each sampled replay block."""

    template: str
    lo: int
    n: int

    def holds(self, q: int, u: int) -> bool:
        return self.lo <= u < self.lo + q * self.n

    def span(self, u: int) -> int:
        """Slot `u`'s place in the sample, `u` one of the part's: the index of its cell's replay block."""
        return (u - self.lo) // self.n

    def unit(self, u: int) -> int:
        """`Part.unit`: slot `u`'s unit in its cell, its place among the `n` slots there."""
        return (u - self.lo) % self.n


@dataclass(frozen=True)
class Link:
    """A link of the two-stage design: template `template`'s session port `port` reads the boundary registration's port
    `value`."""

    template: str
    port: str
    value: str


@dataclass(frozen=True)
class Wire:
    """A wire of the two-stage design: template `template`'s session port `port` reads template `writer`'s output port
    `wport`, one value the interior commits once, at the writer's port."""

    template: str
    port: str
    writer: str
    wport: str


@dataclass(frozen=True)
class Run:
    """`SecondStage.Run`: a run of units' rows, `count` units from unit `unit`, the first reading row `row`, each next
    `step` rows on."""

    unit: int
    count: int
    row: int
    step: int


@dataclass(frozen=True)
class Rows:
    """`SecondStage.Rows`: the rows template `template`'s units read at its session port `port`, one a link or a wire
    names: of the value's `per` rows a replay block, the boundary port's (a link) or its writer's units' (a wire), unit
    `j` reads the one its `runs` give it (`row_at`)."""

    template: str
    port: str
    per: int
    runs: tuple[Run, ...]

    def to_json(self) -> dict[str, Any]:
        return {"per": self.per, "port": self.port, "runs": [[r.unit, r.count, r.row, r.step] for r in self.runs],
                "template": self.template}


def parts(q: int, strata: Sequence[tuple[str, Sequence[Sequence[int]]]]) -> tuple[Part, ...] | None:
    """One part per stratum of the interior program (`partition.strata`), its one range of a positive multiple of the
    sample's `q` slots; None when a stratum is not such a range."""
    out = []
    for t, rs in strata:
        if len(rs) != 1 or q <= 0:
            return None
        lo, hi = rs[0]
        if not (lo < hi and (hi - lo) % q == 0):
            return None
        out.append(Part(t, lo, (hi - lo) // q))
    return tuple(out)


def _get(j: Any, k: str) -> Any:
    return j.get(k) if isinstance(j, dict) else None


def _by_port(rec: Any) -> dict[str, Any]:
    """`Record.byPort`: the record's last root of each port."""
    roots = _get(rec, "roots")
    out: dict[str, Any] = {}
    for r in roots if isinstance(roots, list) else ():
        if isinstance(_get(r, "port"), str):
            out.pop(r["port"], None)
            out[r["port"]] = r
    return out


def _port_of(named: bool, t: str, p: str) -> str:
    return f"{t}/{p}" if named else p


def _by_template(rec: Any) -> bool:
    """`SecondStage.byTemplate`: the record names its ports ``<template>/<port>``."""
    return any("/" in x for x in _by_port(rec))


def _outputs(t: Any) -> int:
    """`SecondStage.outputs`: the output ports ``o<j>`` of a circuit returning `t`, one per field: a tuple's or a
    record's fields, else one."""
    return len(t.fields) if isinstance(t, (Tuple, RecordType)) else 1


def _fields(t: Any) -> list[Any]:
    """`SecondStage.fields`: the fields of a circuit returning `t`, one per output port: a tuple's or a record's, else
    `t`."""
    if isinstance(t, Tuple):
        return list(t.fields)
    return [f for _name, f in t.fields] if isinstance(t, RecordType) else [t]


def _parts(r: Any) -> list[Any]:
    """A body reference's parts, its concatenations spliced in, as the descriptor's ``Seq`` holds them."""
    return [x for q in r.parts for x in _parts(q)] if isinstance(r, Concat) else [r]


def _sources(r: Any) -> list[tuple[bool, int]]:
    """`SecondStage.sources`: what a body's reference `r` reads, parameters (True) and nodes (False), by index."""
    out: list[tuple[bool, int]] = []
    todo = [r]
    while todo:
        q = todo.pop()
        if isinstance(q, Concat):
            todo.extend(reversed(q.parts))
        elif isinstance(q, Explicit):
            out += [(s == "p", i) for s, i, _j in q.items]
        else:
            out.append((q.space == "p", q.idx))
    return out


def _callers(defs: Mapping[str, Any]) -> dict[str, int]:
    """`SecondStage.callCounts`: how many nodes of the bodies of `defs` call each definition, by descriptor id."""
    out: dict[str, int] = {}
    for f in defs.values():
        for node in () if isinstance(f, PrimitiveDefinition) else f.body.nodes:
            out[descriptor_id(node.fn)] = out.get(descriptor_id(node.fn), 0) + 1
    return out


def _members(node: Any) -> int:
    """`Node.members`: a batch's members or a scan's iterations, else one."""
    return node.n if node.form in ("batch", "scan") else 1


def _axis_dims(t: Any, ax: int) -> tuple[int, int] | None:
    """`Ty.axisDims`: the length of array axis `ax` of `t` and the leaves of its elements; None for none."""
    while isinstance(t, Array):
        if ax == 0:
            return t.n, t.elem.leaves
        t, ax = t.elem, ax - 1
    return None


def _dims(node: Any, a: int) -> tuple[int, int] | None:
    """`Arg.dims`: for a batch's argument `a` mapped on an axis, or a scan's sequence, its length and its elements'
    leaves; None for any other argument."""
    t = node.args[a].type
    if node.form == "batch":
        ax = node.axes[a] if a < len(node.axes) else None
        return None if ax is None else _axis_dims(t, ax)
    if node.form == "scan" and 1 <= a <= node.n_xs and isinstance(t, Array):
        return t.n, t.elem.leaves
    return None


@dataclass(frozen=True)
class _Callers:
    """`SecondStage.Callers`: a replay block's body's templates (each node's, None for a node no template's), each
    node's first unit among its definition's, and each definition's units in the body, by descriptor id."""

    template: tuple[Any, ...]
    first: tuple[int, ...]
    units: Mapping[str, int]


def _callers_of(defs: Mapping[str, Any], fn: Any) -> _Callers:
    """`SecondStage.callersOf`: replay block `fn`'s body's templates and their units, once over `defs` and its body:
    each node's template, the composite definition it calls or batches when every node of `defs` that calls it is one
    of `fn`'s body's; each node's first unit among its definition's in a replay block, its callers before it in body
    order holding the units before it, a batch its members' (`_members`), never walked; and each definition's units."""
    every, here = _callers(defs), _callers({"": fn})
    units: dict[str, int] = {}
    template, first = [], []
    for node in fn.body.nodes:
        d = descriptor_id(node.fn)
        first.append(units.get(d, 0))
        units[d] = units.get(d, 0) + _members(node)
        c = node.fn
        template.append(c if node.form in ("call", "batch") and not isinstance(c, PrimitiveDefinition)
                        and every.get(d) == here.get(d) else None)
    return _Callers(tuple(template), tuple(first), units)


def _push_run(rs: list[Run], r: Run) -> None:
    """`SecondStage.pushRun`: run `r` after runs `rs`, merged into the last when `r` is one unit right after it
    continuing its rows (the last one unit, at a row no later than `r`'s, or `r`'s row the last's next), else
    appended."""
    last = rs[-1] if rs else None
    if last is not None and r.count == 1 and last.unit + last.count == r.unit:
        if last.count == 1 and last.row <= r.row:
            rs[-1] = Run(last.unit, 2, last.row, r.row - last.row)
            return
        if last.count > 1 and r.row == last.row + last.count * last.step:
            rs[-1] = Run(last.unit, last.count + 1, last.row, last.step)
            return
    rs.append(r)


def row_at(rs: Sequence[Run], j: int) -> int | None:
    """`SecondStage.rowAt`: unit `j`'s row under runs `rs`, in order of their first units: the run holding it, found by
    bisection over their first units, gives ``row + (j - unit)·step``; None for a unit no run holds."""
    lo, hi = 0, len(rs)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rs[mid].unit <= j:
            lo = mid
        else:
            hi = mid
    if not rs:
        return None
    r = rs[lo]
    return r.row + (j - r.unit) * r.step if r.unit <= j < r.unit + r.count else None


def _fits(rs: Sequence[Run], n: int) -> bool:
    """`SecondStage.fits`: runs `rs` hold units 0 to ``n - 1``, each once, in order: the first from unit 0, each next
    from the last's end, none empty, the last ending at `n`."""
    e = 0
    for r in rs:
        if r.unit != e or r.count <= 0:
            return False
        e += r.count
    return e == n


@dataclass(frozen=True)
class _Share:
    """`SecondStage.Share`: one caller's share of a link or a wire `edge`: its template's descriptor id `fn`, the
    value's rows a replay block `per`, and its units' rows `run`."""

    edge: Any
    fn: str
    per: int
    run: Run


def _gather(port: Callable[[Any], tuple[str, str]], same: Callable[[Any, Any], bool], units: Mapping[str, int],
            xs: Sequence[_Share]) -> list[tuple[Any, int, tuple[Run, ...]]] | None:
    """`SecondStage.gather`: shares `xs` gathered one per template port (`port`), in order of first appearance, each
    port's runs in order (`_push_run`), with its value's rows a block; None when two of a port name different values
    (`same`) or rows a block, or a port's runs don't hold each of its template's `units` once, in order (`_fits`)."""
    g: list[tuple[_Share, list[Run]]] = []
    for x in xs:
        j = next((i for i, (y, _rs) in enumerate(g) if port(y.edge) == port(x.edge)), None)
        if j is None:
            g.append((x, [x.run]))
            continue
        y, rs = g[j]
        if not (same(y.edge, x.edge) and y.per == x.per):
            return None
        _push_run(rs, x.run)
    if not all(_fits(rs, units.get(y.fn, 0)) for y, rs in g):
        return None
    return [(y.edge, y.per, tuple(rs)) for y, rs in g]


def _slice(node: Any, a: int, w: int) -> tuple[int, int] | None:
    """`SecondStage.slice`: the row of its collection's rows node `node`'s argument `a` reads, `w` leaves a member, and
    each next member's step, when the argument is one aligned slice of `w` leaves a member of one collection, its one
    affine part: the same slice for every member (a call's argument, or a batch's shared one, step 0), or a member's
    from the first, outer axis mapped (step 1). None for any other argument."""
    rs = _parts(node.args[a].refs)
    r = rs[0] if len(rs) == 1 else None
    if w <= 0 or not isinstance(r, Affine) or r.base % w:
        return None
    dims = _dims(node, a)
    if dims is None:
        return (r.base // w, 0) if r.count == w else None
    return (r.base // w, 1) if node.form == "batch" and dims[1] == w and r.count == _members(node) * w else None


def _node_links(cs: _Callers, fn: Any, k: int, node: Any) -> list[_Share] | None:
    """`SecondStage.nodeLinks`: the links node `k` of replay block `fn`'s body makes, each its share: for each argument
    that reads one of `fn`'s parameters, its template's parameter there reads that one, the parameter's rows a block
    those of the template parameter's width, its units the rows its slice is (`_slice`). None when the argument reads
    anything else too, or anything but such a slice, or `node` is no template's."""
    params, out = fn.signature()[0], []
    for a, arg in enumerate(node.args):
        src = _sources(arg.refs)
        first = next((x for x in src if x[0]), None)
        if first is None:
            continue
        c = cs.template[k]
        ports = [] if c is None else c.signature()[0]
        if not all(x == first for x in src) or c is None or a >= len(ports) or first[1] >= len(params):
            return None
        (port, ty), (name, pty) = ports[a], params[first[1]]
        at = _slice(node, a, ty.leaves)
        if at is None or pty.leaves % ty.leaves:
            return None
        out.append(_Share(Link(descriptor_id(c), port, name), descriptor_id(c), pty.leaves // ty.leaves,
                          Run(cs.first[k], _members(node), *at)))
    return out


def _field_at(t: Any, off: int, c: int) -> int | None:
    """`SecondStage.fieldAt`: the field of a value of type `t` from leaf `off`, of `c` leaves: its index among
    `_fields`; None for none."""
    s = 0
    for j, f in enumerate(_fields(t)):
        if s == off and f.leaves == c:
            return j
        s += f.leaves
    return None


def _field_of(t: Any, s: int, c: int) -> tuple[int, int] | None:
    """`SecondStage.fieldOf`: the field of a value of type `t` holding its leaves ``[s, s + c)``: its index and first
    leaf; None for none."""
    o = 0
    for j, f in enumerate(_fields(t)):
        if o <= s and s + c <= o + f.leaves:
            return j, o
        o += f.leaves
    return None


def _return_links(cs: _Callers, fn: Any) -> list[_Share] | None:
    """`SecondStage.returnLinks`: the links replay block `fn`'s return makes, each its share: each part of its return,
    which holds all of its leaves, an output of node `k`, a template's: one whole field `f` of its output where `k` is
    one instance (`_field_at`), the field's leaves its width, or else `k`'s whole output, the template's of one output
    (`f` = 0), its leaves its width. Within one field `j` of the return at a multiple of that width from the field's
    first, it makes the template's ``o<f>`` `fn`'s ``o<j>``, the field's rows a block those of that width, and `k`'s
    units the field's rows from there on, one each. None when a part is anything else."""
    nodes, rty, parts = fn.body.nodes, fn.signature()[1], _parts(fn.body.ret)
    if sum(r.count for r in parts) != rty.leaves:
        return None
    out, s = [], 0
    for r in parts:
        start, s = s, s + r.count
        if not (isinstance(r, Affine) and r.space == "n" and r.idx < len(nodes)):
            return None
        x, t = nodes[r.idx], cs.template[r.idx]
        if t is None:
            return None
        tty = t.signature()[1]
        if _members(x) == 1:
            f, w = _field_at(tty, r.base, r.count) if x.out.leaves == tty.leaves else None, r.count
        else:
            f, w = 0 if r.base == 0 and _outputs(tty) == 1 and r.count == x.out.leaves else None, tty.leaves
        if f is None or w <= 0 or r.count != _members(x) * w:
            return None
        at = _field_of(rty, start, r.count)
        if at is None:
            return None
        j, off = at
        g = _fields(rty)[j]
        if (start - off) % w or g.leaves % w:
            return None
        out.append(_Share(Link(descriptor_id(t), f"o{f}", f"o{j}"), descriptor_id(t), g.leaves // w,
                          Run(cs.first[r.idx], _members(x), (start - off) // w, 0 if _members(x) == 1 else 1)))
    return out


def link_runs(defs: Mapping[str, Any], key: str) -> list[tuple[Link, int, tuple[Run, ...]]] | None:
    """`SecondStage.linkRuns`: the design's links with their rows a block and runs, from replay block ``defs[key]``'s
    registered Definition, its parameters and its body: those its nodes make (`_node_links`) and its return
    (`_return_links`), one per template port (`_gather`), each value its own port's name. None when its body reads a
    parameter, or returns, any other way, or a template's callers read a linked port unlike each other."""
    fn = defs.get(key)
    if fn is None or isinstance(fn, PrimitiveDefinition):
        return None
    recut_unread(fn, "second_stage.link_runs")
    cs, shares = _callers_of(defs, fn), []
    for k, node in enumerate(fn.body.nodes):
        got = _node_links(cs, fn, k, node)
        if got is None:
            return None
        shares += got
    ret = _return_links(cs, fn)
    if ret is None:
        return None
    return _gather(lambda x: (x.template, x.port), lambda a, b: a.value == b.value, cs.units, shares + ret)


def links_of(defs: Mapping[str, Any], key: str) -> tuple[Link, ...] | None:
    """`SecondStage.linksOf`: the design's links (`link_runs`)."""
    runs = link_runs(defs, key)
    return None if runs is None else tuple(x for x, _per, _rs in runs)


def replay_links(program: Any, key: str, rec: Any) -> tuple[Link, ...] | None:
    """`SecondStage.replayLinks`: the links of the replay block `program` defines under descriptor id `key` (`links_of`
    over the definitions its descriptor records, `codec.definitions`), each value the boundary docket's port for
    it, as record `rec` names its ports."""
    links, named = links_of(definitions(program), key), _by_template(rec)
    return None if links is None else tuple(Link(x.template, x.port, _port_of(named, key, x.value)) for x in links)


def _node_wires(cs: _Callers, fn: Any, k: int, node: Any) -> list[_Share] | None:
    """`SecondStage.nodeWires`: the wires node `k` of replay block `fn`'s body makes, each its share: for each argument
    that reads a node, its template's parameter there reads that node's output field `f`, one of its template's
    (`_field_at`), the node one instance and the argument the field whole, the same for each of `node`'s members: the
    writer's units a block its rows, `node`'s units reading the node's. None when such an argument reads anything else
    too, or part of a field, several nodes, a node no template's, or `node` is no template's. An argument that reads
    only `fn`'s parameters is a link's (`_node_links`)."""
    nodes, out = fn.body.nodes, []
    for a, arg in enumerate(node.args):
        if all(p for p, _i in _sources(arg.refs)):
            continue
        rs = _parts(arg.refs)
        r = rs[0] if len(rs) == 1 else None
        if not (isinstance(r, Affine) and r.space == "n" and r.idx < len(nodes)):
            return None
        w, tw, t = nodes[r.idx], cs.template[r.idx], cs.template[k]
        params = [] if t is None else t.signature()[0]
        if tw is None or a >= len(params):
            return None
        port, ty = params[a]
        if not (_dims(node, a) is None and _members(w) == 1 and 0 < r.count == ty.leaves
                and w.out.leaves == tw.signature()[1].leaves):
            return None
        f = _field_at(tw.signature()[1], r.base, r.count)
        if f is None:
            return None
        out.append(_Share(Wire(descriptor_id(t), port, descriptor_id(tw), f"o{f}"), descriptor_id(t),
                          cs.units.get(descriptor_id(tw), 0), Run(cs.first[k], _members(node), cs.first[r.idx], 0)))
    return out


def wire_runs(defs: Mapping[str, Any], key: str) -> list[tuple[Wire, int, tuple[Run, ...]]] | None:
    """`SecondStage.wireRuns`: the design's wires with their rows a block and runs, from replay block ``defs[key]``'s
    registered Definition: those its nodes make (`_node_wires`), one per template port (`_gather`). None when its body
    reads a node any other way, or a template's callers read a wired port unlike each other."""
    fn = defs.get(key)
    if fn is None or isinstance(fn, PrimitiveDefinition):
        return None
    recut_unread(fn, "second_stage.wire_runs")
    cs, shares = _callers_of(defs, fn), []
    for k, node in enumerate(fn.body.nodes):
        got = _node_wires(cs, fn, k, node)
        if got is None:
            return None
        shares += got
    return _gather(lambda w: (w.template, w.port), lambda a, b: a.writer == b.writer and a.wport == b.wport, cs.units,
                   shares)


def replay_rows(program: Any, key: str) -> tuple[Rows, ...]:
    """`SecondStage.replayRows`: the design's rows, those of the ports the links and the wires of the replay block
    `program` defines under descriptor id `key` name (`link_runs`, `wire_runs`), none for either it makes none of."""
    defs = definitions(program)
    if key not in defs:
        return ()
    runs = [*(link_runs(defs, key) or ()), *(wire_runs(defs, key) or ())]
    return tuple(Rows(x.template, x.port, per, rs) for x, per, rs in runs)

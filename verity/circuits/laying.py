"""A Program's root units within a limit (`lay`), and its proof blocks under it (`pack`).

**The limit** is a predicate ``fits(ins, outs, ands)`` on a block of whole root units: ``ins`` the bits of each parameter of
the block's Program as `restrict` cuts it (one a maximal run of in(S) of one width and one binding), ``outs`` the bits of
each of its returned arrays (one a maximal run of out(S) of one width), ``ands`` its units' ANDs (`count`). It is the
composer's limit when the caller passes the composer's rule; nothing here names one.

**Laying** (`lay`). While a root unit, alone a block, does not fit, its root node is replaced by its callee's body at the
root: each body node, in body order, is a root node of the same form and arguments, a primitive node one batch of one
member, and a batch's members are laid in member order, each its callee's body. The gates are C's own gates in C's own
order, each reading the same gates (PROTOCOL.md §6 lays a call node out as its callee's body nodes, a batch as its members,
a batch of one primitive as its one gate): only the root's nodes, and so the units, change. A root unit that is one
primitive and does not fit, or a body that holds a scan, is refused (`Unlaid`).

**Packing** (`pack`). The root units in order, a block closed when the next unit would take it past ``fits``: the block's
ports are its in(S) and out(S), so a wire between two of its units is no port of it.
"""

from __future__ import annotations

import bisect
import heapq
from typing import Any, Callable, Iterator, Sequence

from . import partition_object as PO
from .count import count
from .codec import descriptor_id
from .defs import CompositeDefinition, PrimitiveDefinition, bind
from .program import REGISTRATION, RUN, Program, input_id, is_input
from .refs import Affine, Coll, Concat, Explicit, Grid, Strided, concat
from .restrict import _spell
from .types import Value, axis_runs

Fits = Callable[[tuple[int, ...], tuple[int, ...], int], bool]


class Unlaid(ValueError):
    """What `lay` and `pack` refuse, by code: ``lay-primitive`` (a unit that is one primitive and does not fit), ``lay-scan``
    (a body to lay that holds a scan), ``lay-form`` (a root node to lay that is no call or batch)."""

    def __init__(self, code: str, what: str) -> None:
        super().__init__(f"{code}: {what}")
        self.code = code


def query(program: Any) -> dict[str, Any]:
    """The template query whose units are the root's Calls: each Definition a root node calls, in first-call order."""
    return PO.template_instances_query(list(dict.fromkeys(descriptor_id(fn) for _k, fn in PO.calls(program))))


def lay(program: Any, fits: Fits) -> Program:
    """``program`` with every root unit that does not fit laid at the root, recursively (module docstring)."""
    while True:
        m = Measure(program, query(program))
        over = sorted({m.units[u][0] for u in range(len(m.units)) if not fits(*m.block(u, u + 1))})
        if not over:
            return program
        nodes = program.root.body.nodes
        for k in over:
            fn = nodes[k].fn
            if isinstance(fn, PrimitiveDefinition):
                raise Unlaid("lay-primitive", f"root node {k}, {descriptor_id(fn)}: one primitive a unit, past the limit")
            if any(nd.form == "scan" for nd in fn.body.nodes):
                raise Unlaid("lay-scan", f"root node {k}, {descriptor_id(fn)}: its body holds a scan, which no root unit is")
        program = _laid(program, set(over))


def _laid(program: Any, over: set[int]) -> Program:
    fn0, root = program.fn, program.root.body
    params, ret = fn0.signature()
    inputs = next((k for k, nd in enumerate(root.nodes) if not is_input(nd.fn)), len(root.nodes))

    def body(B, S, *_colls):
        new: dict[int, Any] = {k: Affine("n", k, 0, root.nodes[k].out.leaves) for k in range(inputs)}
        look = lambda s, i: new[i]
        for k in range(inputs, len(root.nodes)):
            node = root.nodes[k]
            args = [Coll(x.type, _subst(x.refs, look)) for x in node.args]
            new[k] = _open(B, node, args) if k in over else _emit(B, node, args)
        return Coll(ret, _subst(root.ret, look))

    fn = bind(CompositeDefinition(f"{fn0.name}Laid", fn0.version, (), lambda S: (params, ret), body, register=False))
    names = [n for n, _t in params]
    return Program(fn, annotations=program.annotations, theta=program.theta,
                   registered=[n for n, b in zip(names, program.bindings) if b == REGISTRATION])


def _emit(B, node, args: list) -> Any:
    if node.form == "primitive":
        return B.call(node.fn, *args, name=node.name).refs
    if node.form == "call":
        return B.call(node.fn, *args, name=node.name).refs
    if node.form == "batch":
        return B.batch(node.fn, *args, axes=node.axes, n=node.n, name=node.name).refs
    return B.scan(node.fn, args[0], args[1:1 + node.n_xs], args[1 + node.n_xs:], n=node.n, name=node.name).refs


def _open(B, node, args: list) -> Any:
    if node.form == "call":
        return _body(B, node.fn, args)
    if node.form != "batch":
        raise Unlaid("lay-form", f"a {node.form} node of {descriptor_id(node.fn)} at the root")
    return concat([_body(B, node.fn, [a if ax is None else a.take(ax, m) for a, ax in zip(args, node.axes)])
                   for m in range(node.n)])


def _body(B, fn, args: list) -> Any:
    body, local = fn.body, {}
    look = lambda s, i: args[i].refs if s == "p" else local[i]
    for j, nd in enumerate(body.nodes):
        a = [Coll(x.type, _subst(x.refs, look)) for x in nd.args]
        if nd.form == "primitive":
            local[j] = B.batch(nd.fn, *a, axes=(None,) * len(a), n=1, name=nd.name).refs
        else:
            local[j] = _emit(B, nd, a)
    return _subst(body.ret, look)


def _subst(refs: Any, look: Callable[[str, int], Any]) -> Any:
    """``refs`` with each part's (space, index) replaced by the references ``look`` gives it: structurally where that is one
    node's leaves from a base, leaf by leaf otherwise."""
    out, todo = [], [refs]
    while todo:
        p = todo.pop()
        if isinstance(p, Concat):
            todo.extend(reversed(p.parts))
            continue
        if isinstance(p, Affine):
            out.append(look(p.space, p.idx).slice(p.base, p.base + p.count))
            continue
        if isinstance(p, Strided) and isinstance(t := look(p.space, p.idx), Affine):
            out.append(Strided(t.space, t.idx, t.base + p.base, p.inner, p.outer_stride, p.inner_stride, p.count))
            continue
        if isinstance(p, Grid) and all(isinstance(look(s, i), Affine) for s, i, *_x in p.lanes):
            lanes = []
            for s, i, b, e, lv in p.lanes:
                t = look(s, i)
                lanes.append((t.space, t.idx, t.base + b, e, lv))
            out.append(Grid(p.run, tuple(lanes)))
            continue
        out.append(_spell([look(s, i)[j] for s, i, j in p]))
    return concat(out)


def _width(fn: PrimitiveDefinition) -> int:
    _, ret = fn.signature()
    if not isinstance(ret, Value):
        raise Unlaid("lay-form", f"{fn.id} returns {ret!r}, not one value")
    return ret.w


class Measure:
    """A Program's root units under a template query (`PO.TemplatePartition`), each as the gates it reads and the gates it
    holds that a later unit reads or C returns, by intervals of C's gates read from the descriptor; ``block(a, b)`` is the
    ``fits`` arguments of units a .. b - 1."""

    def __init__(self, program: Any, q: Any) -> None:
        self.program, self.root = program, program.root.body
        tp = PO.TemplatePartition(program, q)
        ands = {k: c.ands for k, c in count(program).items()}
        self.units: list[tuple[int, int]] = []
        self.lo: list[int] = []
        self.ands: list[int] = []
        for k, tid, _a, n in tp.ranges:
            node = self.root.nodes[k]
            for m in range(n):
                self.units.append((k, m))
                self.lo.append(self.root.offsets[k] + m * node.fn.gates)
                self.ands.append(ands[tid])
        self.hi = [lo + self.root.nodes[k].fn.gates for (k, _m), lo in zip(self.units, self.lo)]
        self.first = {k: a for k, _t, a, _n in tp.ranges}
        self._rets: dict[int, tuple[list, list]] = {}
        by: dict[int, list] = {}
        self.reads = []
        for u in range(len(self.units)):
            runs = list(self._unit_reads(u))
            self.reads.append(_merge([r[:4] for r in runs]))
            for lo, hi, w, _b, o in runs:
                if o is not None:
                    by.setdefault(o, []).append((lo, hi, w, u))
        for lo, hi, w, _b, o in self._refs(self.root.ret):
            if o is not None:
                by.setdefault(o, []).append((lo, hi, w, len(self.units)))
        self.held = {o: _last(ev) for o, ev in by.items()}

    def block(self, a: int, b: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        start = self.lo[a]
        ins = _merge([(lo, min(hi, start), w, x) for u in range(a, b) for lo, hi, w, x in self.reads[u] if lo < start])
        outs = _merge([(lo, hi, w, RUN) for u in range(a, b) for lo, hi, w, last in self.held.get(u, ()) if last >= b])
        return _rows(ins, lambda r: (r[2], r[3])), _rows(outs, lambda r: r[2]), sum(self.ands[a:b])

    def _unit_reads(self, u: int) -> Iterator[tuple[int, int, int, str]]:
        k, m = self.units[u]
        node = self.root.nodes[k]
        for x, ax in zip(node.args, node.axes if node.form == "batch" else (None,) * len(node.args)):
            if ax is None:
                yield from self._refs(x.refs)
            elif ax == 0:
                L = x.type.elem.leaves
                yield from self._refs(x.refs.slice(m * L, (m + 1) * L))
            else:
                yield from self._refs(x.take(ax, m).refs)

    def _refs(self, refs: Any, owner: int | None = None) -> Iterator[tuple[int, int, int, str, int | None]]:
        todo = [refs]
        while todo:
            p = todo.pop()
            if isinstance(p, Concat):
                todo.extend(p.parts)
            elif isinstance(p, Affine):
                yield from self._leaves(p.idx, p.base, p.base + p.count, owner)
            elif isinstance(p, Strided) and p.outer_stride == 0 and p.inner_stride == 0:
                yield from self._leaves(p.idx, p.base, p.base + 1, owner)
            elif isinstance(p, Strided) and p.inner_stride == 1:
                for q in range(-(-p.count // p.inner)):
                    b = p.base + q * p.outer_stride
                    yield from self._leaves(p.idx, b, b + min(p.inner, p.count - q * p.inner), owner)
            elif isinstance(p, Grid):
                todo.extend(p.lane_parts())
            else:
                for _s, i, j in p:
                    yield from self._leaves(i, j, j + 1, owner)

    def _leaves(self, k: int, lo: int, hi: int, owner: int | None) -> Iterator[tuple[int, int, int, str, int | None]]:
        node, off = self.root.nodes[k], self.root.offsets[k]
        fn, first = node.fn, self.first.get(k)
        if isinstance(fn, PrimitiveDefinition):
            inp = input_id(fn.id)
            if first is None or owner is not None:
                yield (off + lo, off + hi, _width(fn), inp[0] if inp else RUN, owner)
            else:
                yield from ((off + j, off + j + 1, _width(fn), RUN, first + j) for j in range(lo, hi))
            return
        if node.form == "scan":
            raise Unlaid("lay-form", f"root node {k} is a scan")
        L, G = fn.out_leaves, fn.gates
        starts, runs = self._ret_runs(fn)
        for m in range(lo // L, -(-hi // L)):
            o = owner if owner is not None or first is None else first + m
            a, b = max(lo, m * L) - m * L, min(hi, (m + 1) * L) - m * L
            i = bisect.bisect_right(starts, a) - 1
            while i < len(runs) and runs[i][0] < b:
                j0, _j1, g = runs[i][:3]
                x, y = max(a, j0), min(b, runs[i][1])
                if isinstance(g, tuple):
                    yield from self._refs(Explicit((_member_arg(node, g[1], g[2], m),)), o)
                else:
                    yield (off + m * G + g + x - j0, off + m * G + g + y - j0, runs[i][3], RUN, o)
                i += 1

    def _ret_runs(self, fn: Any) -> tuple[list, list]:
        """``fn``'s returned leaves as runs (first leaf, end leaf, first gate relative to fn's first, width), or a leaf it
        returns of its parameters as (leaf, leaf + 1, ("p", i, l)), with the runs' first leaves."""
        if id(fn) in self._rets:
            return self._rets[id(fn)]
        runs: list = []
        for j in range(fn.out_leaves):
            e = _resolve(fn, fn.body.ret[j], self._ret_runs)
            if e[0] == "p":
                runs.append((j, j + 1, e, 0))
            elif runs and not isinstance(runs[-1][2], tuple) and runs[-1][1] == j and runs[-1][3] == e[2] and \
                    runs[-1][2] + j - runs[-1][0] == e[1]:
                runs[-1] = (runs[-1][0], j + 1, runs[-1][2], e[2])
            else:
                runs.append((j, j + 1, e[1], e[2]))
        self._rets[id(fn)] = ([r[0] for r in runs], runs)
        return self._rets[id(fn)]


def _last(events: list) -> list[tuple[int, int, int, int]]:
    events.sort()
    points = sorted({x for lo, hi, _w, _u in events for x in (lo, hi)})
    heap: list = []
    i, pieces = 0, []
    for a, b in zip(points, points[1:]):
        while i < len(events) and events[i][0] <= a:
            lo, hi, w, u = events[i]
            heapq.heappush(heap, (-u, hi, w))
            i += 1
        while heap and heap[0][1] <= a:
            heapq.heappop(heap)
        if heap:
            u, _hi, w = heap[0]
            if pieces and pieces[-1][1] == a and pieces[-1][2:] == (w, -u):
                pieces[-1] = (pieces[-1][0], b, w, -u)
            else:
                pieces.append((a, b, w, -u))
    return pieces


def _ret_leaf(runs: tuple[list, list], j: int) -> tuple:
    i = bisect.bisect_right(runs[0], j) - 1
    j0, _j1, g, w = runs[1][i]
    return g if isinstance(g, tuple) else ("g", g + j - j0, w)


def _member_arg(node: Any, pi: int, leaf: int, m: int) -> tuple[str, int, int]:
    a = node.args[pi]
    if node.form == "call":
        return a.refs[leaf]
    if node.form == "batch":
        ax = node.axes[pi]
        if ax is None:
            return a.refs[leaf]
        if ax == 0:
            return a.refs[m * a.type.elem.leaves + leaf]
        _outer, n, L = axis_runs(a.type, ax)
        return a.refs[(leaf // L) * n * L + m * L + leaf % L]
    raise Unlaid("lay-form", f"a {node.form} node's parameter")


def _resolve(fn: Any, ref: tuple, rets: Callable[[Any], list]) -> tuple:
    """Leaf ``ref`` of ``fn``'s body as ("g", gate relative to fn's first, width) or ("p", i, l), its parameter's leaf."""
    body = fn.body
    s, i, l = ref
    while s != "p":
        nd, off = body.nodes[i], body.offsets[i]
        g = nd.fn
        if isinstance(g, PrimitiveDefinition):
            return ("g", off + (l if nd.form == "batch" else 0), _width(g))
        if nd.form == "scan":
            c = nd.out.fields[0].leaves
            m, j = (nd.n - 1, l) if l < c else (divmod(l - c, nd.out.fields[1].elem.leaves)[0],
                                                 c + (l - c) % nd.out.fields[1].elem.leaves)
            while True:
                e = _ret_leaf(rets(g), j)
                if e[0] == "g":
                    return ("g", off + m * g.gates + e[1], e[2])
                _p, pi, pl = e
                if pi == 0 and m > 0:
                    m, j = m - 1, pl
                    continue
                if pi == 0:
                    s, i, l = nd.args[0].refs[pl]
                elif pi <= nd.n_xs:
                    s, i, l = nd.args[pi].refs[m * nd.args[pi].type.elem.leaves + pl]
                else:
                    s, i, l = nd.args[pi].refs[pl]
                break
            continue
        m, j = (0, l) if nd.form == "call" else divmod(l, g.out_leaves)
        e = _ret_leaf(rets(g), j)
        if e[0] == "g":
            return ("g", off + m * g.gates + e[1], e[2])
        s, i, l = _member_arg(nd, e[1], e[2], m)
    return ("p", i, l)


def _merge(runs: list) -> list[tuple[int, int, int, str]]:
    out: list = []
    for lo, hi, w, b in sorted(r for r in runs if r[0] < r[1]):
        if out and (lo < out[-1][1] or lo == out[-1][1] and out[-1][2:] == (w, b)):
            if hi > out[-1][1]:
                out[-1] = (out[-1][0], hi, w, b)
            continue
        out.append((lo, hi, w, b))
    return out


def _rows(runs: list, key: Callable[[tuple], Any]) -> tuple[int, ...]:
    rows: list[int] = []
    last = None
    for r in runs:
        if rows and key(r) == last:
            rows[-1] += (r[1] - r[0]) * r[2]
        else:
            rows.append((r[1] - r[0]) * r[2])
            last = key(r)
    return tuple(rows)


def pack(program: Any, fits: Fits, q: Any = None) -> list[tuple[int, int]]:
    """The root units in order, ``q``'s (default `query`), as the partition object's ``blocks``: a block closed when the
    next unit would take it past ``fits`` (module docstring). A unit that alone does not fit is refused (`Unlaid`)."""
    m = Measure(program, q or query(program))
    n, out, a = len(m.units), [], 0
    while a < n:
        if not fits(*m.block(a, a + 1)):
            k = m.units[a][0]
            raise Unlaid("lay-primitive" if isinstance(m.root.nodes[k].fn, PrimitiveDefinition) else "pack-unit",
                         f"unit {a} (root node {k}, {descriptor_id(m.root.nodes[k].fn)}) alone does not fit: `lay` it first")
        b = a + 1
        while b < n and fits(*m.block(a, b + 1)):
            b += 1
        out.append((a, b))
        a = b
    return out


__all__ = ["Fits", "Measure", "Unlaid", "lay", "pack", "query"]

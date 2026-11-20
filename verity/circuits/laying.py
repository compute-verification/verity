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
ports are its in(S) and out(S), so a wire between two of its units is no port of it. A block closes at each of ``ends``
(a coarser level's ends, which the blocks then refine). With a floor ``enough(ins, outs, ands, units)``, a block under it
(it took every unit the limit and the next end let it) joins the block before it where that block, so grown, still fits
and is not at an end; elsewhere `pack` is refused (`pack-floor`) and returns nothing. A block holds no unit that reads another
unit of it (the door's `operands-outside`): a block closes before a unit that reads a unit of it, and where the floor would join
such a unit to the block it reads, `pack` is refused (`pack-inside-read`).
"""

from __future__ import annotations

import bisect
import heapq
from typing import Any, Callable, Iterator, Sequence

from . import partition_object as PO
from .count import count
from .codec import descriptor_id
from .defs import CompositeDefinition, PrimitiveDefinition, bind
from .program import REGISTRATION, RUN, SUPPLIED, Program, input_id, is_input
from .refs import Affine, Coll, Concat, Explicit, Grid, Strided, concat
from .restrict import _spell
from .types import Value, axis_runs

Fits = Callable[[tuple[int, ...], tuple[int, ...], int], bool]
Enough = Callable[[tuple[int, ...], tuple[int, ...], int, int], bool]


class Unlaid(ValueError):

    def __init__(self, code: str, what: str) -> None:
        super().__init__(f"{code}: {what}")
        self.code = code


def query(program: Any) -> dict[str, Any]:
    return PO.template_instances_query(list(dict.fromkeys(descriptor_id(fn) for _k, fn in PO.calls(program))))


def lay(program: Any, fits: Fits) -> Program:
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
        def look(_s, i):
            return new[i]
        for k in range(inputs, len(root.nodes)):
            node = root.nodes[k]
            args = [Coll(x.type, _subst(x.refs, look)) for x in node.args]
            new[k] = _open(B, node, args) if k in over else _emit(B, node, args)
        return Coll(ret, _subst(root.ret, look))

    fn = bind(CompositeDefinition(f"{fn0.name}Laid", fn0.version, (), lambda S: (params, ret), body, register=False))
    names = [n for n, _t in params]
    return Program(fn, annotations=program.annotations, theta=program.theta,
                   registered=[n for n, b in zip(names, program.bindings) if b == REGISTRATION],
                   supplied={n: program.supplied[n] for n, b in zip(names, program.bindings) if b == SUPPLIED})


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
    def look(s, i):
        return args[i].refs if s == "p" else local[i]
    for j, nd in enumerate(body.nodes):
        a = [Coll(x.type, _subst(x.refs, look)) for x in nd.args]
        if nd.form == "primitive":
            local[j] = B.batch(nd.fn, *a, axes=(None,) * len(a), n=1, name=nd.name).refs
        else:
            local[j] = _emit(B, nd, a)
    return _subst(body.ret, look)


def _subst(refs: Any, look: Callable[[str, int], Any]) -> Any:
    out, todo = [], [refs]
    while todo:
        p = todo.pop()
        if isinstance(p, Concat):
            todo.extend(reversed(p.parts))
            continue
        if isinstance(p, Affine):
            t = look(p.space, p.idx)
            out.append(p if isinstance(t, Affine) and (t.space, t.idx, t.base) == (p.space, p.idx, 0)
                       else t.slice(p.base, p.base + p.count))
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
        self.reads_unit: list[list[int]] = []
        for v, runs in enumerate(self.reads):
            us: set[int] = set()
            for lo, hi, _w, _b in runs:
                u = max(bisect.bisect_right(self.lo, lo) - 1, 0)
                while u < v and self.lo[u] < hi:
                    if lo < self.hi[u]:
                        us.add(u)
                    u += 1
            self.reads_unit.append(sorted(us))

    def inside_read(self, lo: int, hi: int) -> tuple[int, int] | None:
        """The first unit ``v`` of ``[lo, hi)`` and a unit ``u`` of it that ``v`` reads, where a block of those units holds
        one: a proof block's units read no unit of their own block."""
        for v in range(lo, hi):
            if (u := next((u for u in self.reads_unit[v] if u >= lo), None)) is not None:
                return u, v
        return None

    def block(self, a: int, b: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        for c, cost in self.grow(a):
            if c == b:
                return cost
        raise IndexError(b)

    def grow(self, a: int) -> Iterator[tuple[int, tuple[tuple[int, ...], tuple[int, ...], int]]]:
        start, ins, outs, alive, ands, foreign = self.lo[a], _Rows(), _Rows(), [], 0, 0
        for b in range(a + 1, len(self.units) + 1):
            u = b - 1
            for lo, hi, w, x in self.reads[u]:
                if lo < start:
                    ins.add(lo, min(hi, start), w, (w, x))
            for lo, hi, w, last in self.held.get(u, ()):
                own = self.lo[u] <= lo and hi <= self.hi[u]
                foreign += not own
                heapq.heappush(alive, (last, lo, hi, w, own))
                outs.add(lo, hi, w, w)
            rebuild = False
            while alive and alive[0][0] < b:
                _l, lo, hi, w, own = heapq.heappop(alive)
                if own and not foreign:
                    outs.remove(lo, hi, w)
                else:
                    rebuild, foreign = True, foreign - (not own)
            if rebuild:
                outs = _Rows()
                for _l, lo, hi, w, _o in sorted(alive, key=lambda e: e[1]):
                    outs.add(lo, hi, w, w)
            ands += self.ands[u]
            yield b, (ins.rows(), outs.rows(), ands)

    def _unit_reads(self, u: int) -> Iterator[tuple[int, int, int, str]]:
        k, m = self.units[u]
        node = self.root.nodes[k]
        for x, ax in zip(node.args, node.axes if node.form == "batch" else (None,) * len(node.args)):
            rw = x.type.homogeneous_width()
            if ax is None:
                yield from self._refs(x.refs, None, rw)
            elif ax == 0:
                L = x.type.elem.leaves
                yield from self._refs(x.refs.slice(m * L, (m + 1) * L), None, rw)
            else:
                yield from self._refs(x.take(ax, m).refs, None, rw)

    def _refs(self, refs: Any, owner: int | None = None,
              rw: int | None = None) -> Iterator[tuple[int, int, int, str, int | None]]:
        todo = [refs]
        while todo:
            p = todo.pop()
            if isinstance(p, Concat):
                todo.extend(p.parts)
            elif isinstance(p, Affine):
                yield from self._leaves(p.idx, p.base, p.base + p.count, owner, rw)
            elif isinstance(p, Strided) and p.outer_stride == 0 and p.inner_stride == 0:
                yield from self._leaves(p.idx, p.base, p.base + 1, owner, rw)
            elif isinstance(p, Strided) and p.inner_stride == 1:
                for q in range(-(-p.count // p.inner)):
                    b = p.base + q * p.outer_stride
                    yield from self._leaves(p.idx, b, b + min(p.inner, p.count - q * p.inner), owner, rw)
            elif isinstance(p, Grid):
                todo.extend(p.lane_parts())
            else:
                for _s, i, j in p:
                    yield from self._leaves(i, j, j + 1, owner, rw)

    def _leaves(self, k: int, lo: int, hi: int, owner: int | None,
                rw: int | None = None) -> Iterator[tuple[int, int, int, str, int | None]]:
        node, off = self.root.nodes[k], self.root.offsets[k]
        fn, first = node.fn, self.first.get(k)
        if isinstance(fn, PrimitiveDefinition):
            inp = input_id(fn.id)
            if inp and rw and _width(fn) > rw:
                q = _width(fn) // rw
                lo, hi = lo // q, -(-hi // q)
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


class _Rows:

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.keys: list[Any] = []
        self.widths: list[int] = []
        self.ivs: list[list[tuple[int, int]]] = []
        self.bits: list[int] = []

    def rows(self) -> tuple[int, ...]:
        return tuple(self.bits)

    def add(self, lo: int, hi: int, w: int, key: Any) -> None:
        if lo >= hi:
            return
        i = bisect.bisect_right(self.starts, lo) - 1
        if i >= 0 and lo < self.ends[i] and self.keys[i] != key:
            L, gk, gw = self.ivs[i], self.keys[i], self.widths[i]
            k = bisect.bisect_left(L, (lo,))
            lb = sum(h - l for l, h in L[:k]) * gw
            self._insert(i + 1, gk, gw, L[k:], self.bits[i] - lb)
            self._insert(i + 1, key, w, [(lo, hi)], (hi - lo) * w)
            self.ivs[i], self.bits[i], self.ends[i] = L[:k], lb, L[k - 1][1]
        elif i >= 0 and self.keys[i] == key:
            self._put(i, lo, hi)
        elif i + 1 < len(self.keys) and self.keys[i + 1] == key:
            self._put(i + 1, lo, hi)
        else:
            self._insert(i + 1, key, w, [(lo, hi)], (hi - lo) * w)

    def remove(self, lo: int, hi: int, w: int) -> None:
        i = bisect.bisect_right(self.starts, lo) - 1
        L = self.ivs[i]
        k = bisect.bisect_right(L, (lo, float("inf"))) - 1
        l, h = L[k]
        L[k:k + 1] = [x for x in ((l, lo), (hi, h)) if x[0] < x[1]]
        self.bits[i] -= (hi - lo) * w
        if L:
            self.starts[i], self.ends[i] = L[0][0], L[-1][1]
            return
        for col in (self.starts, self.ends, self.keys, self.widths, self.ivs, self.bits):
            del col[i]
        if 0 < i < len(self.keys) and self.keys[i - 1] == self.keys[i]:
            self.ivs[i - 1] += self.ivs[i]
            self.bits[i - 1] += self.bits[i]
            self.ends[i - 1] = self.ends[i]
            for col in (self.starts, self.ends, self.keys, self.widths, self.ivs, self.bits):
                del col[i]

    def _insert(self, i: int, key: Any, w: int, L: list[tuple[int, int]], bits: int) -> None:
        self.starts.insert(i, L[0][0])
        self.ends.insert(i, L[-1][1])
        self.keys.insert(i, key)
        self.widths.insert(i, w)
        self.ivs.insert(i, L)
        self.bits.insert(i, bits)

    def _put(self, i: int, lo: int, hi: int) -> None:
        L = self.ivs[i]
        k = bisect.bisect_left(L, (lo,))
        if k > 0 and L[k - 1][1] >= lo:
            k -= 1
        m, a, b, covered = k, lo, hi, 0
        while m < len(L) and L[m][0] <= hi:
            a, b, covered = min(a, L[m][0]), max(b, L[m][1]), covered + L[m][1] - L[m][0]
            m += 1
        L[k:m] = [(a, b)]
        self.bits[i] += (b - a - covered) * self.widths[i]
        self.starts[i], self.ends[i] = L[0][0], L[-1][1]


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


def _inside_read(lo: int, hi: int, writer: int, reader: int) -> Unlaid:
    return Unlaid("pack-inside-read", f"units [{lo}, {hi}) would be one proof block to meet the floor, and unit {reader} reads unit "
                  f"{writer} of it: a proof block holds no unit that reads a unit of its own")


def pack(program: Any, fits: Fits, q: Any = None, m: Measure | None = None, enough: Enough | None = None,
         ends: Sequence[int] = ()) -> list[tuple[int, int]]:
    m = m or Measure(program, q or query(program))
    n, out, a = len(m.units), [], 0
    stops = sorted(set(ends))
    if any(not 0 < e <= n for e in stops):
        raise Unlaid("pack-ends", f"ends {stops}: each a unit number in (0, {n}]")
    while a < n:
        if not fits(*m.block(a, a + 1)):
            k = m.units[a][0]
            raise Unlaid("lay-primitive" if isinstance(m.root.nodes[k].fn, PrimitiveDefinition) else "pack-unit",
                         f"unit {a} (root node {k}, {descriptor_id(m.root.nodes[k].fn)}) alone does not fit: `lay` it first")
        e = next((x for x in stops if x > a), n)
        b, got, reader = a, None, None
        for c, cost in m.grow(a):
            if c > e or not fits(*cost):
                break
            if m.reads_unit[c - 1] and m.reads_unit[c - 1][-1] >= a:
                reader = c - 1
                break
            b, got = c, cost
        if enough is not None and not enough(*got, b - a):
            lo = out[-1][0] if out and a not in stops else None
            j = None if lo is None else m.block(lo, b)
            if j is None or not fits(*j) or not enough(*j, b - lo):
                if reader is not None:
                    raise _inside_read(a, reader + 1, m.reads_unit[reader][-1], reader)
                raise Unlaid("pack-floor", f"units [{a}, {b}) are under the floor, the next unit does not join them within "
                             "the limit or past an end, and the block before does not take them")
            if (held := m.inside_read(lo, b)) is not None:
                raise _inside_read(lo, b, *held)
            out[-1] = (lo, b)
        else:
            out.append((a, b))
        a = b
    return out


ZK_M_MIN = 25


def zk_min_units(k_log: int, g: int) -> int:
    """The fewest units of ``2^k_log`` rows, ``g`` a VU, that a zero-knowledge Flock statement holds (``m = k_log +
    max(3, ⌈log₂⌈n/g⌉⌉) >= 25``); `flock-circuit`'s ``zk_min_units`` is its twin."""
    g = max(g, 1)
    return 1 if k_log + 3 >= ZK_M_MIN else (g << (ZK_M_MIN - 1 - k_log)) + 1


__all__ = ["Enough", "Fits", "Measure", "Unlaid", "lay", "pack", "query", "zk_min_units"]

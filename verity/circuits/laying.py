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

**Batchwise** (`lay(..., batchwise=True)`). A batch of n members past the limit is opened node by node instead: each body
node, in body order, becomes one root batch over the n members (a body batch of n' members one batch of n·n', member m's
n' members together), its mapped arguments member-major along axis 0, and a body node whose every argument is the same
for every member is emitted once. The laid root then has one tile's nodes whatever n is, and its gates are C's gates
reordered node by node: the outputs are the same, the gate order and so the units' gate ranges are not C's.

**Packing** (`pack`). The root units in order, a block closed when the next unit would take it past ``fits``: the block's
ports are its in(S) and out(S), so a wire between two of its units is no port of it. A block closes at each of ``ends``
(a coarser level's ends, which the blocks then refine). With a floor ``enough(ins, outs, ands, units)``, a block under it
(it took every unit the limit and the next end let it) joins the block before it where that block, so grown, still fits
and is not at an end; elsewhere `pack` is refused (`pack-floor`) and returns nothing.
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
from .refs import Affine, Coll, Concat, Explicit, Grid, Strided, as_grid, concat, spell
from .restrict import _spell
from .types import Array, Value, axis_runs, remove_axis

Fits = Callable[[tuple[int, ...], tuple[int, ...], int], bool]
Enough = Callable[[tuple[int, ...], tuple[int, ...], int, int], bool]


class Unlaid(ValueError):

    def __init__(self, code: str, what: str) -> None:
        super().__init__(f"{code}: {what}")
        self.code = code


def query(program: Any) -> dict[str, Any]:
    return PO.template_instances_query(list(dict.fromkeys(descriptor_id(fn) for _k, fn in PO.calls(program))))


def lay(program: Any, fits: Fits, batchwise: bool = False) -> Program:
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
        program = _laid(program, set(over), batchwise)


def _laid(program: Any, over: set[int], batchwise: bool = False) -> Program:
    fn0, root = program.fn, program.root.body
    params, ret = fn0.signature()
    inputs = next((k for k, nd in enumerate(root.nodes) if not is_input(nd.fn)), len(root.nodes))
    sp = _periodic if batchwise else _spell

    def body(B, S, *_colls):
        new: dict[int, Any] = {k: Affine("n", k, 0, root.nodes[k].out.leaves) for k in range(inputs)}
        def look(_s, i):
            return new[i]
        for k in range(inputs, len(root.nodes)):
            node = root.nodes[k]
            args = [Coll(x.type, _subst(x.refs, look, sp)) for x in node.args]
            if k not in over:
                new[k] = _emit(B, node, args)
            elif batchwise and node.form == "batch":
                new[k] = _open_batchwise(B, node, args)
            else:
                new[k] = _open(B, node, args)
        return Coll(ret, _subst(root.ret, look, sp))

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


def _open_batchwise(B, node, args: list) -> Any:
    n, body = node.n, node.fn.body
    mapped = [ax is not None for ax in node.axes]
    params = [[a if ax is None else _member(a, ax, m) for a, ax in zip(args, node.axes)] for m in range(n)]
    local: dict[int, tuple] = {}

    def looks(m):
        def look(s, i):
            if s == "p":
                return params[m][i].refs
            kind, refs, L = local[i]
            return refs if kind == "s" else refs.slice(m * L, (m + 1) * L)
        return look

    def shared(refs) -> bool:
        return all(not mapped[i] if s == "p" else local[i][0] == "s" for s, i in _sources(refs))

    for j, nd in enumerate(body.nodes):
        if all(shared(x.refs) for x in nd.args):
            a = [Coll(x.type, _subst(x.refs, looks(0), _periodic)) for x in nd.args]
            out = B.batch(nd.fn, *a, axes=(None,) * len(a), n=1, name=nd.name).refs if nd.form == "primitive" \
                else _emit(B, nd, a)
            local[j] = ("s", out, 0)
            continue
        if nd.form == "scan":
            raise Unlaid("lay-scan", f"{descriptor_id(nd.fn)}: a scan in a body opened batchwise")
        inner = nd.n if nd.form == "batch" else 1
        axes2 = nd.axes if nd.form == "batch" else (None,) * len(nd.args)
        cols, axes = [], []
        for x, ax2 in zip(nd.args, axes2):
            whole = [Coll(x.type, _subst(x.refs, looks(m), _periodic)) for m in range(n)]
            if ax2 is None and shared(x.refs):
                cols.append(whole[0])
                axes.append(None)
                continue
            mt = x.type if ax2 is None else remove_axis(x.type, ax2)[1]
            members = [w.refs if ax2 is None else _member(w, ax2, m2).refs for w in whole for m2 in range(inner)]
            cols.append(Coll(Array(n * inner, mt), _stack(members, (inner, n))))
            axes.append(0)
        out = B.batch(nd.fn, *cols, axes=axes, n=n * inner, name=nd.name)
        local[j] = ("b", out.refs, inner * out.type.elem.leaves)
    return _stack([_subst(body.ret, looks(m), _periodic) for m in range(n)])


def _member(c: Coll, axis: int, m: int) -> Coll:
    if axis:
        return c.take(axis, m)
    L = c.type.elem.leaves
    return Coll(c.type.elem, _slice(c.refs, m * L, (m + 1) * L))


def _slice(r: Any, lo: int, hi: int) -> Any:
    """``r.slice(lo, hi)``, a grid cut at whole periods of its levels kept one grid."""
    if not isinstance(r, Grid) or (lo, hi) == (0, r.count):
        return r if (lo, hi) == (0, r.count) else r.slice(lo, hi)
    per, lv = len(r.lanes) * r.run, r.lanes[0][4]
    if lo % per or hi % per or hi <= lo:
        return r.slice(lo, hi)
    t0, t1, i = lo // per, hi // per, 0
    while i < len(lv) and t0 % (_prod(lv[:i + 1])) == 0 and t1 % (_prod(lv[:i + 1])) == 0 and \
            (t1 - t0) % _prod(lv[:i + 1]) == 0 and _prod(lv[:i + 1]) <= t1 - t0:
        i += 1
    P = _prod(lv[:i])
    q0, cnt = t0 // P, (t1 - t0) // P
    if i < len(lv) and i < len(lv) - 1 and q0 % lv[i][0] + cnt > lv[i][0]:
        return r.slice(lo, hi)
    lanes = []
    for s, x, b, e, levels in r.lanes:
        q, off = q0, 0
        for k, (rad, d) in enumerate(levels[i:]):
            q, digit = (0, q) if k == len(levels) - i - 1 else divmod(q, rad)
            off += digit * d
        top = ((cnt, levels[i][1]),) if cnt > 1 else ()
        lanes.append((s, x, b + off, e, (*levels[:i], *top)))
    return spell(r.run, lanes)


def _prod(levels) -> int:
    out = 1
    for rad, _d in levels:
        out *= rad
    return out


def _sources(refs: Any) -> Iterator[tuple[str, int]]:
    todo = [refs]
    while todo:
        p = todo.pop()
        if isinstance(p, Concat):
            todo.extend(p.parts)
        elif isinstance(p, (Affine, Strided)):
            yield p.space, p.idx
        elif isinstance(p, Grid):
            yield from ((s, i) for s, i, *_x in p.lanes)
        else:
            yield from ((s, i) for s, i, _j in p)


def _atoms(refs: Any, out: list) -> list:
    if isinstance(refs, Concat):
        for p in refs.parts:
            _atoms(p, out)
    elif isinstance(refs, Strided) and refs.count % refs.inner:
        s, i, b, inner, o, e, c = (refs.space, refs.idx, refs.base, refs.inner, refs.outer_stride, refs.inner_stride,
                                   refs.count)
        if c > inner:
            _atoms(Strided(s, i, b, inner, o, e, c - c % inner), out)
        out.append((c % inner, ((s, i, b + (c // inner) * o, e, ()),)))
    elif refs.count:
        out.append(as_grid(refs))
    return out


def _stack(rs: list, radices: tuple[int, ...] | None = None) -> Any:
    """The references of members ``rs`` in member order, member μ's digits in ``radices`` (innermost first; one digit
    by default): one grid level a digit where each member's references are the first member's moved by a fixed step a
    lane and digit, else their concatenation."""
    if len(rs) == 1:
        return rs[0]
    atoms = [_atoms(r, []) for r in rs]
    for rad in dict.fromkeys(x for x in ((radices or (len(rs),)), (len(rs),)) if x):
        if (out := _grid(atoms, tuple(r for r in rad if r > 1))) is not None:
            return out
    N = len(rs)
    for g in range(2, N // 2 + 1):
        if N % g == 0 and (out := _grid([sum(atoms[i:i + g], []) for i in range(0, N, g)], (N // g,))) is not None:
            return out
    if rs[0].count <= N * len(atoms[0]) and (out := _grid([[(1, ((s, i, j, 1, ()),)) for s, i, j in r] for r in rs],
                                                          (N,))) is not None:
        return out
    return concat(rs)


def _grid(atoms: list, radices: tuple[int, ...]) -> Any:
    if len(atoms) == 1 or not atoms[0]:
        return None

    def sig(a):
        return [(run, tuple((s, i, e, lv) for s, i, _b, e, lv in lanes)) for run, lanes in a]

    def bases(a):
        return [b for _r, lanes in a for _s, _i, b, _e, _lv in lanes]

    s0, b0 = sig(atoms[0]), bases(atoms[0])
    flat = [ln for _r, lanes in atoms[0] for ln in lanes]
    if all(sig(a) == s0 for a in atoms[1:]) and (len(atoms[0]) == 1 or
                                                  all(run == atoms[0][0][0] and not lanes[0][4] for run, lanes in atoms[0])):
        units, d = 1, []
        for r in radices:
            d.append([y - x for x, y in zip(b0, bases(atoms[units]))])
            units *= r
        ok = True
        for mu in range(1, len(atoms)):
            q, want = mu, list(b0)
            for r, dk in zip(radices, d):
                q, digit = divmod(q, r)
                want = [w + digit * y for w, y in zip(want, dk)]
            if bases(atoms[mu]) != want:
                ok = False
                break
        if ok:
            return spell(atoms[0][0][0], [(s, i, b, e, (*lv, *((r, dk[j]) for r, dk in zip(radices, d))))
                                          for j, (s, i, b, e, lv) in enumerate(flat)])
    return None


def _periodic(items: list) -> Any:
    """`_spell`'s references, as one grid where they repeat with a period whose every position moves by a fixed step."""
    r = _spell(items)
    if not isinstance(r, Concat):
        return r
    for seq, run in ((r.parts, None), (items, 1)):
        N = len(seq)
        for P in range(1, N // 2 + 1):
            if N % P:
                continue
            k = N // P
            if run is None:
                if len({a.count for a in seq[:P]}) != 1:
                    continue
                at = [(a.space, a.idx, a.base, a.count) for a in seq]
            else:
                at = [(s, i, j, 1) for s, i, j in seq]
            d = [at[P + j][2] - at[j][2] for j in range(P)]
            if all(at[q * P + j][:2] == at[j][:2] and at[q * P + j][3] == at[j][3] and
                   at[q * P + j][2] == at[j][2] + q * d[j] for q in range(1, k) for j in range(P)):
                return spell(at[0][3], [(at[j][0], at[j][1], at[j][2], 1, ((k, d[j]),)) for j in range(P)])
    return r


def _subst(refs: Any, look: Callable[[str, int], Any], spell_: Callable[[list], Any] = _spell) -> Any:
    out, todo = [], [refs]
    while todo:
        p = todo.pop()
        if isinstance(p, Concat):
            todo.extend(reversed(p.parts))
            continue
        if isinstance(p, Affine):
            t = look(p.space, p.idx)
            out.append(p if isinstance(t, Affine) and (t.space, t.idx, t.base) == (p.space, p.idx, 0)
                       else _slice(t, p.base, p.base + p.count))
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
        out.append(spell_([look(s, i)[j] for s, i, j in p]))
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
        b, got = a, None
        for c, cost in m.grow(a):
            if c > e or not fits(*cost):
                break
            b, got = c, cost
        if enough is not None and not enough(*got, b - a):
            lo = out[-1][0] if out and a not in stops else None
            j = None if lo is None else m.block(lo, b)
            if j is None or not fits(*j) or not enough(*j, b - lo):
                raise Unlaid("pack-floor", f"units [{a}, {b}) are under the floor, the next unit does not join them within "
                             "the limit or past an end, and the block before does not take them")
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

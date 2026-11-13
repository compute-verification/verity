"""A structural lowering's table reads in Flock lookup slots: `struct_lower`'s template, lowered with ``slots`` (each read of
a pinned MUFU table only its value columns, `struct_lower._SlotRead`), split by read depth into one ``flock-ir-tmpl/v2``
template a net, the unit and then a net a stage (``stage0`` and on, `net_name`), wired to ``lookup:<table>`` slots as
`read_slots` wires a flat unit's, the last stage returning the outputs.

Depth is `read_slots`' (an input bit 0, a row its operands' greatest, a slot read's value bits one more than its index's
greatest), found once per Definition and pattern of its inputs' depths (`_Info`), so a Definition is split once per depth
pattern however many copies place it, and the split never walks the unrolled rows. A row is placed in the stage of its depth,
or as late as what reads it allows (`_Split.plan`, once per depth pattern and pattern of its caller's reads), so a value only a
later stage reads is computed there from what it was computed from; `lower` keeps the placement whose nets pass fewer bits
on (`boundary`). A Definition's part in stage s (`_Proj`)
holds, in their order: its rows placed in stage s; the index bits of each read whose index is of depth s, committed; the value bits
of each read whose value is of depth s, copied from the read's slot word; and its copies' parts in stage s. Its inputs are
the inputs it reads in stage s; then each earlier value it reads (its own or a copy's, of depth p < s), grouped by p, its
own first and then each copy's in copy order; then each slot word it reads, its own then its copies': so a run of copies
binds each group as one segment. Each net's root (`_Split.root`) takes the unit's input ports (the unit) or a port per group
and a port per slot word (a later stage), and returns, as copy rows, a port per later stage that reads its values, the index
of each read of its depth, and (the last stage) the unit's output ports. A read of a table ``slots`` doesn't map stays inline,
of its index's depth. :func:`evaluate` runs the nets and the slots in order."""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import ir_lower as IL
from . import struct_lower as SL
from . import tail as TL
from . import tmpl as T
from .fp import ceil_pow2

#: the tables a lookup slot reads, each by ``(index bits, value bits a slot word gives)`` (`read_slots.SLOT_TABLES`)
SLOTS = {t: (TL.TABLE_BITS[t], TL.VALUE_BITS) for t in IL.TABLES}
VB = TL.VALUE_BITS


def net_name(s: int) -> str:
    return "unit" if s == 0 else f"stage{s - 1}"


@dataclass
class Split:
    """The nets (the unit, then each stage) as templates, each net's input and output port tags and widths (the unit's
    inputs: its own ports, tagged None), per slot word ``g`` its table and the net computing its index, and the wires
    ``[dst net, 0, dst port, src net, src index, src port]`` (`read_slots`' form)."""

    nets: list[T.Tmpl]
    ins: list[list]
    outs: list[list]
    in_widths: list[list[int]]
    out_widths: list[list[int]]
    lookups: list[tuple[str, int]]
    wires: list[list]


@dataclass
class _Info:
    """A Definition at a pattern of its inputs' depths (``prof``), or a plan of it (`_Split.plan`): each own column's stage
    (``dep``), its copies (``kids``, each at its start), each slot read's index depth, the stages it has rows in, and (a
    plan) the earliest stage reading each of its inputs (``in_dem``)."""

    e: int
    prof: Any
    dep: dict = field(default_factory=dict)
    starts: list = field(default_factory=list)
    kids: list = field(default_factory=list)
    site_d: list = field(default_factory=list)
    stages: set = field(default_factory=set)
    in_dem: dict = field(default_factory=dict)


@dataclass
class _Proj:
    index: int
    cols: int
    args: list
    imp: dict
    words: list
    own: dict
    at: dict


class _Prep:
    def __init__(self, d: T.Def, sites) -> None:
        self.d = d
        self.sites = [s for s in sites if s.slot and s.vals]
        items = [(col, 0, i) for i, (col, _a, _b) in enumerate(d.rows)]
        items += [(s.lo, 1, k) for k, s in enumerate(self.sites)]
        items += [(c.base + j * c.stride, 2, (ci, j)) for ci, c in enumerate(d.calls) for j in range(c.count)]
        self.items = sorted(items)
        self.zero = d.cols > 0 and (not self.items or self.items[0][0] > 0)


def _bound(c: T.Call, j: int) -> list[int]:
    return [x for ln, at, step, reps, rs in c.segs for r in range(reps)
            for x in range(at + r * rs + j * step, at + r * rs + j * step + ln)]


def _runs(placed: list, cols: int) -> list[T.Call]:
    """Consecutive copies of one part whose bases step by its columns and whose bindings step by one fixed vector, as one
    call each (`struct_lower._runs`' rule)."""
    out, i = [], 0
    while i < len(placed):
        base, kp, r0 = placed[i]
        own0 = r0 < cols
        j, step = i + 1, None
        while j < len(placed):
            b, kq, r = placed[j]
            if kq is not kp or b != base + (j - i) * kp.cols or not np.array_equal(r < cols, own0):
                break
            d = r - r0
            if step is None:
                if (d < 0).any():
                    break
                step = d
            elif not np.array_equal(d, (j - i) * step):
                break
            j += 1
        st = step if step is not None else np.zeros(len(r0), np.int64)
        out.append(T.Call(kp.index, j - i, base, kp.cols, SL._segs(r0.tolist(), st.tolist(), cols)))
        i = j
    return out


class _Split:
    def __init__(self, t: T.Tmpl, sites: dict[int, list]) -> None:
        self.t = t
        self.prep = [_Prep(d, sites.get(e, ())) for e, d in enumerate(t.defs)]
        self.infos: dict[tuple, _Info] = {}
        self.projs: dict[tuple, _Proj] = {}
        self.defs: dict[int, list[T.Def]] = {}

    def depth(self, I: _Info, x: int) -> int:
        d = self.prep[I.e].d
        if x >= d.cols:
            return I.prof[x - d.cols] if x < d.cols + d.ins else 0
        return self.dcol(I, x)

    def _kid(self, I: _Info, x: int) -> tuple[int, int, _Info] | None:
        i = bisect.bisect_right(I.starts, x) - 1
        if i >= 0:
            start, kid = I.kids[i]
            if x - start < self.t.defs[kid.e].cols:
                return i, start, kid
        return None

    def dcol(self, I: _Info, x: int) -> int:
        v = I.dep.get(x)
        if v is not None:
            return v
        k = self._kid(I, x)
        if k is not None:
            return self.dcol(k[2], x - k[1])
        if x == 0 and self.prep[I.e].zero:
            return 0
        raise NotImplementedError(f"Definition {I.e}: column {x} is no row's, slot value's or copy's")

    def addr(self, I: _Info, x: int) -> tuple:
        if x in I.dep:
            return ("c", x)
        i, start, kid = self._kid(I, x)
        return ("s", i, self.addr(kid, x - start))

    def is_zero(self, I: _Info, x: int) -> bool:
        while x not in I.dep:
            k = self._kid(I, x)
            if k is None:
                return True
            _i, start, I = k
            x -= start
        return False

    def info(self, e: int, prof: bytes) -> _Info:
        key = (e, prof)
        I = self.infos.get(key)
        if I is not None:
            return I
        P = self.prep[e]
        d = P.d
        I = _Info(e, prof, site_d=[None] * len(P.sites))
        for pos, kind, p in P.items:
            if kind == 0:
                col, a, b = d.rows[p]
                v = max((self.depth(I, x) for x in a + b), default=0)
                I.dep[col] = v
                I.stages.add(v)
            elif kind == 1:
                s = P.sites[p]
                v = max((self.depth(I, x) for f in s.idx for x in f), default=0)
                I.site_d[p] = v
                for _j, c in s.vals:
                    I.dep[c] = v + 1
                I.stages.update((v, v + 1))
            else:
                ci, j = p
                c = d.calls[ci]
                prof_k = [self.depth(I, x) for x in _bound(c, j)]
                if max(prof_k, default=0) > 255:
                    raise NotImplementedError(f"Definition {e}: a read depth past 255")
                kid = self.info(c.e, bytes(prof_k))
                I.starts.append(pos)
                I.kids.append((pos, kid))
                I.stages |= kid.stages
        self.infos[key] = I
        return I

    def plan(self, I: _Info, dem: dict[int, int]) -> _Info:
        """``I`` with each row as late as what reads it allows: its stage the earliest of its readers', never before its
        depth (a row nothing reads at its depth), ``dem`` the earliest stage reading each column of it its caller reads. A
        slot read's index stays at its depth and its value bits one later, so the slots are wired as the depths place them;
        reading the items last to first, every reader of an item is placed before it."""
        key = (I.e, I.prof, tuple(sorted(dem.items())))
        J = self.infos.get(key)
        if J is not None:
            return J
        P = self.prep[I.e]
        d = P.d
        c0, c1 = d.cols, d.cols + d.ins
        cons: dict[int, int] = {}
        kcons: dict[int, dict[int, int]] = {}
        J = _Info(I.e, key[1:], {}, I.starts, [None] * len(I.kids), I.site_d)

        def read(x: int, s: int) -> None:
            if x >= c1:
                return
            if x >= c0:
                J.in_dem[x - c0] = min(J.in_dem.get(x - c0, s), s)
            elif x in I.dep:
                cons[x] = min(cons.get(x, s), s)
            else:
                k = self._kid(I, x)
                if k is not None:
                    m = kcons.setdefault(k[0], {})
                    m[x - k[1]] = min(m.get(x - k[1], s), s)

        for x, s in dem.items():
            read(x, s)
        ki = len(I.kids)
        for _pos, kind, p in reversed(P.items):
            if kind == 0:
                col, a, b = d.rows[p]
                v = max(I.dep[col], cons.get(col, I.dep[col]))
                J.dep[col] = v
                J.stages.add(v)
                for x in a + b:
                    read(x, v)
            elif kind == 1:
                site, q = P.sites[p], I.site_d[p]
                for _j, c in site.vals:
                    J.dep[c] = q + 1
                J.stages.update((q, q + 1))
                for f in site.idx:
                    for x in f:
                        read(x, q)
            else:
                ki -= 1
                start, kid = I.kids[ki]
                kp = self.plan(kid, kcons.pop(ki, {}))
                J.kids[ki] = (start, kp)
                J.stages |= kp.stages
                bound = _bound(d.calls[p[0]], p[1])
                for k, v in kp.in_dem.items():
                    read(bound[k], v)
        self.infos[key] = J
        return J

    def col_in(self, pr: _Proj, I: _Info, x: int) -> int:
        c = pr.own.get(x)
        if c is not None:
            return c
        i, start, kid = self._kid(I, x)
        base, kp = pr.at[i]
        return base + self.col_in(kp, kid, x - start)

    def at_addr(self, pr: _Proj, a: tuple) -> int:
        if a[0] == "c":
            return pr.own[a[1]]
        if a[0] == "i":
            return pr.own[a]
        base, kp = pr.at[a[1]]
        return base + self.at_addr(kp, a[2])

    def site_of(self, I: _Info, a: tuple):
        while a[0] == "s":
            I = I.kids[a[1]][1]
            a = a[2]
        return self.prep[I.e].sites[a[1]]

    def proj(self, I: _Info, s: int) -> _Proj:
        key = (I.e, I.prof, s)
        pr = self.projs.get(key)
        if pr is not None:
            return pr
        P = self.prep[I.e]
        d = P.d
        c0, c1 = d.cols, d.cols + d.ins
        pr = _Proj(-1, 0, [], {}, [], {0: 0} if P.zero else {}, {})
        n = int(P.zero)
        rows, placed, need, args, own_words = [], [], {}, set(), []

        def ref(x: int) -> tuple:
            if x >= c1:
                return ("k",)
            if x >= c0:
                args.add(x - c0)
                return ("a", x - c0)
            if self.is_zero(I, x):
                return ("n", 0) if P.zero and x == 0 else ("z",)
            q = self.dcol(I, x)
            if q == s:
                return ("n", self.col_in(pr, I, x))
            a = self.addr(I, x)
            need.setdefault(q, {})[a] = None
            return ("f", q, a)

        ki = 0
        for _pos, kind, p in P.items:
            if kind == 0:
                col, a, b = d.rows[p]
                if I.dep[col] == s:
                    pr.own[col] = n
                    rows.append((n, [ref(x) for x in a], [ref(x) for x in b]))
                    n += 1
            elif kind == 1:
                site, q = P.sites[p], I.site_d[p]
                if q == s:
                    for bi, f in enumerate(site.idx):
                        pr.own[("i", p, bi)] = n
                        rows.append((n, [ref(x) for x in f], [("k",)]))
                        n += 1
                if q + 1 == s:
                    w = len(own_words)
                    own_words.append(("w", p))
                    for j, c in site.vals:
                        pr.own[c] = n
                        rows.append((n, [("w", w, j)], [("k",)]))
                        n += 1
            else:
                i = ki
                ki += 1
                _start, kid = I.kids[i]
                if s not in kid.stages:
                    continue
                kp = self.proj(kid, s)
                pr.at[i] = (n, kp)
                bound = _bound(d.calls[p[0]], p[1])
                placed.append((i, kp, [ref(bound[a]) for a in kp.args]))
                n += kp.cols

        imp: dict[int, list] = {}
        kid_off: dict[tuple[int, int], int] = {}
        for q in range(s):
            lst = list(need.get(q, ()))
            for i, kp, _r in placed:
                kid_off[i, q] = len(lst)
                lst += [("s", i, a) for a in kp.imp.get(q, ())]
            if lst:
                imp[q] = lst
        words = list(own_words)
        kid_w: dict[int, int] = {}
        for i, kp, _r in placed:
            kid_w[i] = len(words)
            words += [("s", i, a) for a in kp.words]
        args_l = sorted(args)
        argpos = {a: k for k, a in enumerate(args_l)}
        base, off = {}, len(args_l)
        for q in sorted(imp):
            base[q] = off
            off += len(imp[q])
        wbase = off
        ins = off + VB * len(words)
        cols = n
        if cols == 0:
            raise NotImplementedError(f"Definition {I.e}: an empty part in stage {s}")
        pos = {q: {a: k for k, a in enumerate(need.get(q, ()))} for q in imp}

        def R(r: tuple) -> int | None:
            t = r[0]
            if t == "n":
                return r[1]
            if t == "k":
                return cols + ins
            if t == "a":
                return cols + argpos[r[1]]
            if t == "f":
                return cols + base[r[1]] + pos[r[1]][r[2]]
            if t == "w":
                return cols + wbase + VB * r[1] + r[2]
            return None

        def side(rs: list) -> tuple:
            return T.canon([x for x in map(R, rs) if x is not None])

        out_rows = tuple((c, side(a), side(b)) for c, a, b in rows)
        runs = []
        for i, kp, rs in placed:
            r = [R(x) for x in rs]
            if None in r:
                raise NotImplementedError(f"Definition {I.e}: a copy bound to a zero column in stage {s}")
            for q in sorted(kp.imp):
                r += range(cols + base[q] + kid_off[i, q], cols + base[q] + kid_off[i, q] + len(kp.imp[q]))
            r += range(cols + wbase + VB * kid_w[i], cols + wbase + VB * (kid_w[i] + len(kp.words)))
            runs.append((pr.at[i][0], kp, np.asarray(r, np.int64)))
        defs = self.defs.setdefault(s, [])
        pr.index, pr.cols, pr.args, pr.imp, pr.words = len(defs), cols, args_l, imp, words
        defs.append(T.Def(cols, ins, out_rows, tuple(_runs(runs, cols))))
        self.projs[key] = pr
        return pr


def _groups(g: list[str]) -> tuple[int, list[int], list[int]]:
    """The unit's input words, input port widths and output port widths from `struct_lower.lower`'s header."""
    v = [int(x) for x in g]
    assert v[0] == 1 and v[1] == 0, "one input group at column 0"
    n_in = v[3]
    widths = v[4:4 + n_in]
    at = 4 + n_in
    assert v[at] == 1, "one output group"
    n_out = v[at + 3]
    return v[2], widths, v[at + 4:at + 4 + n_out]


def split(t: T.Tmpl, sites: dict[int, list], name: str = "unit", late: bool = False) -> Split | None:
    """``t`` (`struct_lower.lower`'s with ``slots``, ``sites`` each Definition's `Spec.sites`) split at its slot reads (the
    module docstring), each row in the stage of its depth or, ``late``, as late as what reads it allows (`_Split.plan`);
    None when it has none."""
    S = _Split(t, sites)
    r0 = t.defs[t.root]
    _iw, in_widths, ret_widths = _groups(t.groups)
    nin = sum(in_widths)
    if len(r0.calls) != 1:
        return None
    c0 = r0.calls[0]
    bound0 = _bound(c0, 0)
    if any(x >= nin for x in bound0):
        raise NotImplementedError(f"{name}: the root's call binds a column past the inputs")
    top = S.info(c0.e, bytes(len(bound0)))
    top_cols = t.defs[c0.e].cols
    outs = [(col, a) for col, a, _b in r0.rows if col >= nin]

    def rdepth(x: int) -> int:
        return S.dcol(top, x - c0.base) if c0.base <= x < c0.base + top_cols else 0

    D = max([*top.stages, *(rdepth(x) for _c, a in outs for x in a)], default=0)
    if D == 0:
        return None
    if late:
        top = S.plan(top, {x - c0.base: D for _c, a in outs for x in a if c0.base <= x < c0.base + top_cols})
    parts = [S.proj(top, s) if s in top.stages else None for s in range(D + 1)]
    # each later stage's root inputs: per depth p its addresses (("in", c) a unit input, ("t", a) the top's), then words
    needs: list[dict[int, list]] = []
    words: list[list] = []
    for s in range(D + 1):
        pt = parts[s]
        nd: dict[int, dict] = {}
        if s and pt is not None:
            nd.setdefault(0, {}).update({("in", bound0[a]): None for a in pt.args})
        if pt is not None:
            for q, lst in pt.imp.items():
                nd.setdefault(q, {}).update({("t", a): None for a in lst})
        if s == D:
            for _c, a in outs:
                for x in a:
                    if x < nin and s:
                        nd.setdefault(0, {})[("in", x)] = None
                    elif c0.base <= x < c0.base + top_cols and rdepth(x) < s:
                        nd.setdefault(rdepth(x), {})[("t", S.addr(top, x - c0.base))] = None
        needs.append({q: list(v) for q, v in sorted(nd.items()) if v})
        words.append([("t", a) for a in pt.words] if pt is not None else [])
    gid: list[list[int]] = []
    lookups: list[tuple[str, int]] = []
    for s in range(D + 1):
        gid.append(list(range(len(lookups), len(lookups) + len(words[s]))))
        lookups += [(S.site_of(top, a[1]).table, s - 1) for a in words[s]]

    nets, ins, outs_t, in_w, out_w = [], [], [], [], []
    for s in range(D + 1):
        pt = parts[s]
        if s == 0:
            tags_in, widths, nin_s = None, list(in_widths), nin
            loc: dict = {("in", c): c for c in range(nin)}
            wat = nin
        else:
            tags_in, widths, loc, at = [], [], {}, 0
            for q, lst in needs[s].items():
                tags_in.append(("from", q))
                widths.append(len(lst))
                loc.update({a: at + k for k, a in enumerate(lst)})
                at += len(lst)
            wat = at
            for g in gid[s]:
                tags_in.append(("lookup", g))
                widths.append(VB)
                at += VB
            nin_s = at
        iw = ceil_pow2(max(1, -(-nin_s // IL.WORD)))
        base = iw * IL.WORD
        end = base + (pt.cols if pt is not None else 0)

        def here(a: tuple) -> int:
            if a[0] == "in":
                return a[1]
            return base + S.at_addr(pt, a[1])

        out_rows, tags_out, wo = [], [], []
        for t2 in range(s + 1, D + 1):
            lst = needs[t2].get(s)
            if lst:
                tags_out.append(("to", t2))
                wo.append(len(lst))
                out_rows += [((here(a),), False) for a in lst]
        if s + 1 <= D:
            for g, a in zip(gid[s + 1], words[s + 1]):
                site = S.site_of(top, a[1])
                tags_out.append(("index", g))
                wo.append(len(site.idx))
                out_rows += [((here(("t", _index_addr(a[1], b))),), False) for b in range(len(site.idx))]
        if s == D:
            tags_out += [("ret", o) for o in range(len(ret_widths))]
            wo += list(ret_widths)
            for _c, a in outs:
                refs, k = [], False
                for x in a:
                    if x >= r0.cols:
                        k = not k
                    elif x < nin:
                        refs.append(x if s == 0 else loc[("in", x)])
                    elif rdepth(x) == s:
                        refs.append(base + S.col_in(pt, top, x - c0.base))
                    else:
                        refs.append(loc[("t", S.addr(top, x - c0.base))])
                out_rows.append((tuple(refs), k))
        nout = sum(wo)
        ow = ceil_pow2(max(1, -(-nout // IL.WORD)))
        o_word = -(-end // IL.WORD)
        o_word = -(-o_word // ow) * ow
        o_at = o_word * IL.WORD
        cols = (o_word + ow) * IL.WORD + 1
        rows = [(i, (i,), (i,)) for i in range(nin_s)]
        rows += [(o_at + j, T.canon(list(refs) + ([cols] if k else [])), (cols,)) for j, (refs, k) in enumerate(out_rows)]
        calls = []
        if pt is not None:
            if s == 0:
                r = [bound0[a] for a in pt.args]
            else:
                r = [loc[("in", bound0[a])] for a in pt.args]
            for q in sorted(pt.imp):
                r += [loc[("t", a)] for a in pt.imp[q]]
            for k in range(len(pt.words)):
                r += range(wat + VB * k, wat + VB * (k + 1))
            calls.append(T.Call(pt.index, 1, base, pt.cols, SL._segs(r, [0] * len(r), cols)))
        defs = list(S.defs.get(s, []))
        defs.append(T.Def(cols, 0, tuple(rows), tuple(calls)))
        groups = [str(x) for x in (1, 0, iw, len(widths), *widths, 1, o_word, ow, len(wo), *wo)]
        nets.append(T.pack(T.Tmpl(f"{name}/{net_name(s)}", len(defs) - 1, groups, defs)))
        ins.append(tags_in)
        outs_t.append(tags_out)
        in_w.append(widths)
        out_w.append(wo)
    local: dict[int, int] = {}
    seen: dict[str, int] = {}
    for g, (table, _s) in enumerate(lookups):
        local[g] = seen.get(table, 0)
        seen[table] = local[g] + 1
    wires = []
    for s in range(1, D + 1):
        for j, tag in enumerate(ins[s]):
            src = ([net_name(tag[1]), 0, outs_t[tag[1]].index(("to", s))] if tag[0] == "from"
                   else [f"lookup:{lookups[tag[1]][0]}", local[tag[1]], 0])
            wires.append([net_name(s), 0, j, *src])
    for g, (table, s) in enumerate(lookups):
        wires.append([f"lookup:{table}", local[g], 0, net_name(s), 0, outs_t[s].index(("index", g))])
    return Split(nets, ins, outs_t, in_w, out_w, lookups, wires)


def _index_addr(a: tuple, b: int) -> tuple:
    """Of a slot word's address (``("w", k)`` under copies ``("s", i, ..)``), the address of its read's index bit ``b``."""
    return ("i", a[1], b) if a[0] == "w" else ("s", a[1], _index_addr(a[2], b))


def lower(target, name: str = "unit", ports: SL.Ports | None = None, slots: dict | None = None,
          **kw: Any) -> tuple[Split | None, T.Tmpl, Any]:
    """``target`` lowered structurally with its reads of ``slots``' tables (default `SLOTS`) in lookup slots: the split
    (None: it reads no such table, and the template is the unit), the template it was split from, and the lowerer."""
    tm, lw, _top = SL.lower(target, name=name, ports=ports, slots=SLOTS if slots is None else slots, **kw)
    sites = {sp.index: sp.sites for sp in lw.order if sp.sites}
    early = split(tm, sites, name)
    if early is None:
        return None, tm, lw
    late = split(tm, sites, name, late=True)
    return (late if boundary(late) < boundary(early) else early), tm, lw


def boundary(sp: Split) -> int:
    """The bits a split's nets pass to later nets: what its stages read from earlier nets, its slots' words aside."""
    return sum(w for tags, ws in zip(sp.ins[1:], sp.in_widths[1:]) for tag, w in zip(tags, ws) if tag[0] == "from")


def _lookup(table: str, idx: np.ndarray) -> np.ndarray:
    from . import gf2

    words = np.asarray(gf2.table(table), np.uint32)
    n, W = idx.shape
    bits = T.unslice_lanes(idx, W * 64).astype(np.int64)
    at = (bits << np.arange(n, dtype=np.int64)).sum(axis=1)
    v = words[at] & np.uint32((1 << VB) - 1)
    return T.slice_lanes(((v[:, None] >> np.arange(VB, dtype=np.uint32)) & 1).astype(np.uint8))


def evaluate(sp: Split, inputs: np.ndarray) -> np.ndarray:
    """The unit's output port bits, bit-sliced (`tmpl.evaluate`'s lanes), from its input port bits: each net evaluated from
    its template, in order, each slot word looked up from its pinned table at the index its net returns."""
    produced: list[list[np.ndarray]] = []
    words: dict[int, np.ndarray] = {}
    for s, tm in enumerate(sp.nets):
        if s == 0:
            x = np.asarray(inputs, np.uint64)
        else:
            x = np.concatenate([produced[t[1]][sp.outs[t[1]].index(("to", s))] if t[0] == "from" else words[t[1]]
                                for t in sp.ins[s]])
        out = T.evaluate(tm, x)
        ports = np.split(out, np.cumsum(sp.out_widths[s])[:-1])
        produced.append(ports)
        for p, tag in zip(ports, sp.outs[s]):
            if tag[0] == "index":
                words[tag[1]] = _lookup(sp.lookups[tag[1]][0], p)
    last = len(sp.nets) - 1
    return np.concatenate([p for p, tag in zip(produced[last], sp.outs[last]) if tag[0] == "ret"])


def check(target, seed: int) -> dict:
    """``target``'s split evaluated on 64 random inputs made from ``seed`` (`evaluate`), against the flat lowering's
    outputs (`forms.Circuit.evaluate`)."""
    from verity.circuits.boolean.forms import Circuit

    sp, _tm, _lw = lower(target)
    _prog, root = IL.standalone(target)
    args = list(dict.fromkeys(root.args))
    widths = [IL._width(root.circuit.gate(a).prim) for a in args]
    lanes = np.random.default_rng(seed).integers(0, 1 << 64, size=sum(widths), dtype=np.uint64)
    got = [int(x) for x in evaluate(sp, lanes[:, None])[:, 0]]
    C = Circuit()
    words = IL.lower(C, target, [C.input(f"in{a}", w) for a, w in zip(args, widths)])
    for j, w in enumerate(words):
        C.output(f"o{j}", w)
    at = np.cumsum([0, *widths])
    want, _fail = C.evaluate({f"in{a}": [int(x) for x in lanes[at[i]:at[i + 1]]] for i, a in enumerate(args)}, 64)
    want = [x for j in range(len(words)) for x in want[f"o{j}"]]
    return {"nets": len(sp.nets), "lookups": len(sp.lookups), "outputs": len(want),
            "mismatched_bits": sum(x != y for x, y in zip(got, want)) + abs(len(got) - len(want))}


#: a split on disk (`write`): ``<net>.tmpl`` a net, and ``slots.json`` its nets in order, port tags and widths, slots and wires
FORMAT = "struct-slots/v1"


def write(sp: Split, d: Path) -> None:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    names = [net_name(s) for s in range(len(sp.nets))]
    for nm, tm in zip(names, sp.nets):
        (d / f"{nm}.tmpl").write_text(T.write(tm))
    (d / "slots.json").write_text(json.dumps({"format": FORMAT, "nets": names, "ins": sp.ins, "outs": sp.outs,
                                              "in_widths": sp.in_widths, "out_widths": sp.out_widths,
                                              "lookups": sp.lookups, "wires": sp.wires}) + "\n")


def read(d: Path) -> Split:
    d = Path(d)
    m = json.loads((d / "slots.json").read_text())
    if m.get("format") != FORMAT:
        raise ValueError(f"{d}: slots.json is not {FORMAT}")

    def tags(ts):
        return None if ts is None else [tuple(t) for t in ts]

    return Split([T.parse((d / f"{nm}.tmpl").read_text()) for nm in m["nets"]], [tags(t) for t in m["ins"]],
                 [tags(t) for t in m["outs"]], m["in_widths"], m["out_widths"], [tuple(x) for x in m["lookups"]], m["wires"])


def stats(sp: Split) -> dict:
    """Each net's template rows, references and calls, and its rows unrolled (`tmpl.totals`); the slot words and wires."""
    nets = []
    for tm in sp.nets:
        nets.append({"name": tm.name, "rows": sum(len(d.rows) for d in tm.defs),
                     "refs": sum(len(a) + len(b) for d in tm.defs for _c, a, b in d.rows),
                     "calls": sum(len(d.calls) for d in tm.defs), "defs": len(tm.defs),
                     "unrolled_rows": T.totals(tm)["rows"], "root_cols": tm.defs[tm.root].cols})
    return {"nets": nets, "lookups": len(sp.lookups), "wires": len(sp.wires), "boundary_bits": boundary(sp),
            "tables": sorted({t for t, _s in sp.lookups})}

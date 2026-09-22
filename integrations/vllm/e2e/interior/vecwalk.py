"""E2 (SCALE wave 2026-09-12) -- member-vectorised evaluation of composite ``batch`` nodes for E1's ``interior.producer``.

MEASURED on E8's 32x16 rows job (cProfile, pod `sc-e2-prof`): 75 % of the evaluator's time is the per-gate Python operand
resolution (``FastArgs.res`` through the ref/scope chain: 33 resolves per ``AmpereBF16TcDot16`` gate), ~20 % the exact-rational
numerics (``tc_dot`` over ``Term`` objects), the rest ``Gate`` tuples / dict traffic; 95 % of all gates sit in composite ``batch``
nodes whose members are structurally identical (``Gemm`` = ``batch`` of ``GemmCoordinate`` over the N output rows).

This module evaluates such a node for ALL members at once:

1. **Symbolic template.**  The member index ``m`` is kept symbolic as a linear form ``c0 + c1*m`` (a ``Lin``).  ``sym_resolve`` /
   ``sym_param_leaf`` are ``verity.ir.layout.resolve`` / ``param_leaf`` line by line with ``Lin`` arithmetic; every step that would
   need a non-linear function of ``m`` (an ``Explicit``/``Concat`` ref indexed by ``m``, a ``Strided`` ref whose inner run does not
   divide the stride, ``take`` on a non-leading axis, a scan indexed by ``m``) raises ``NotVectorizable`` and the node falls back to
   the per-gate walk.  The template lists every gate of ONE member as ``(gate Lin, primitive, operand Lins)`` in gate order.
2. **Check.**  The template instantiated at ``m in {0, 1, n-1}`` is compared, gate by gate (index, primitive, operand gate indices),
   with the producer's own concrete walk of those members (``CallWalk._scope`` / ``FastArgs``): a mismatch aborts the run.
3. **Evaluate.**  Every template gate is evaluated over the ``n`` members as one numpy column: operands are internal columns, one
   shared transcript word, or a gather ``c0 + c1*arange(n)`` from the transcript (weights / prompt straight from their arrays).
   ``AmpereBF16TcDot16`` (36 of the 38 gates of a K=576 ``GemmCoordinate``), ``F2fpBf16`` and the ``Const*`` family have exact
   integer-numpy twins (``vec_tc_dot16`` is ``veritor.core.silicon.tc_dot`` on int64 arrays; every other primitive is evaluated
   member by member with its own ``evaluate`` -- same function, same operands, just without the resolution).  Non-finite tensor-core
   operands (outside the silicon model) are routed to the scalar ``_tc_dot16_total`` per member.

Same gates, same values, same ``runs`` (in gate order, as ``RunBlock``s the row writer expands with numpy) -- the equivalence gate
is E8's rows digest (byte-identical JSONL) plus the 15.4 M boundary words the producer compares anyway.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from verity.ir.defs import PrimitiveDefinition
from verity.ir.layout import WIDE_FAN_IN, Scope
from verity.ir.refs import Affine, Concat, Strided

Lin = tuple[int, int]  # c0 + c1 * m


class NotVectorizable(Exception):
    """The node needs a non-linear function of the member index somewhere: use the per-gate walk.  `limit`, when set, is the
    number of leading members `[0, limit)` for which the linear form WOULD have resolved (a `param_leaves` range splits there)."""

    def __init__(self, msg: str, limit: int | None = None) -> None:
        super().__init__(msg)
        self.limit = limit


def _prefix(c0: int, c1: int, lo_ok: int, hi_ok: int) -> int:
    """Members `m = 0, 1, ...` for which `c0 + c1 m` stays inside `[lo_ok, hi_ok]` (which contains `c0`); `c1 != 0`."""
    return -(-(hi_ok - c0 + 1) // c1) if c1 > 0 else -(-(c0 - lo_ok + 1) // -c1)


def _lin(x: Any) -> Lin:
    return x if isinstance(x, tuple) else (int(x), 0)


def _add(a: Lin, b: Lin) -> Lin:
    return (a[0] + b[0], a[1] + b[1])


def _mul(a: Lin, k: int) -> Lin:
    return (a[0] * k, a[1] * k)


def _at(lin: Lin, m: int) -> int:
    return lin[0] + lin[1] * m


class SymScope:
    """`layout.Scope` whose offset and member may be linear forms in the vectorised member index.  A top scope (`parent is None`)
    with `param_starts` set is a FLAT template's function: its parameter leaves resolve to the codes `PARAM + start[pi] + leaf`."""

    __slots__ = ("spec", "offset", "parent", "node_idx", "member", "param_starts")

    def __init__(self, spec, offset: Lin, parent, node_idx: int, member: Lin, param_starts: tuple[int, ...] | None = None) -> None:
        self.spec, self.offset, self.parent, self.node_idx, self.member = spec, offset, parent, node_idx, member
        self.param_starts = param_starts


PARAM = 1 << 60  # operand codes >= PARAM are parameter leaves of a flat template's function (never a gate index)


def _ref_at(refs, j: Lin, n: int) -> tuple[str, int, Lin]:
    """`refs[j]` for a linear index `j` valid for every member `m in [0, n)`: `(space, idx, leaf Lin)`."""
    c0, c1 = j
    if c1 == 0:
        s, i, leaf = refs[c0]
        return s, i, (leaf, 0)
    lo, hi = min(c0, c0 + c1 * (n - 1)), max(c0, c0 + c1 * (n - 1))
    if lo < 0 or hi >= refs.count:
        raise NotVectorizable("index out of range for some member")
    if isinstance(refs, Affine):
        return refs.space, refs.idx, (refs.base + c0, c1)
    if isinstance(refs, Strided):
        if c1 % refs.inner == 0:  # (c0 + c1 m) // inner = c0 // inner + (c1 // inner) m exactly; the remainder is constant
            q0, r0 = divmod(c0, refs.inner)
            return refs.space, refs.idx, (refs.base + q0 * refs.outer_stride + r0 * refs.inner_stride, (c1 // refs.inner) * refs.outer_stride)
        raise NotVectorizable("strided ref not aligned to the member stride")
    if isinstance(refs, Concat):
        k = bisect.bisect_right(refs.starts, lo) - 1
        part = refs.parts[k]
        if hi < refs.starts[k] + part.count:
            return _ref_at(part, (c0 - refs.starts[k], c1), n)
        k0 = bisect.bisect_right(refs.starts, c0) - 1
        raise NotVectorizable("concat ref spans parts across members", _prefix(c0, c1, refs.starts[k0], refs.starts[k0] + refs.parts[k0].count - 1))
    raise NotVectorizable(f"{type(refs).__name__} ref indexed by the member")


def _divmod_lin(leaf: Lin, L: int, n: int) -> tuple[Lin, Lin]:
    """`divmod(leaf, L)` as linear forms over the `n` members: the quotient (batch member) and remainder (leaf within it)."""
    c0, c1 = leaf
    if c1 % L == 0:  # the quotient steps with the member, the remainder is constant
        q0, r0 = divmod(c0, L)
        return (q0, c1 // L), (r0, 0)
    lo, hi = min(c0, c0 + c1 * (n - 1)), max(c0, c0 + c1 * (n - 1))
    if lo // L == hi // L:  # every member's leaf lies in the same batch member; the remainder steps with the member
        q0 = lo // L
        return (q0, 0), (c0 - q0 * L, c1)
    q0 = c0 // L
    raise NotVectorizable("batch leaf not aligned to the member stride", _prefix(c0, c1, q0 * L, q0 * L + L - 1))


def sym_resolve(scope, space: str, idx: int, leaf: Lin, n: int) -> Lin:
    """`layout.resolve(scope, (space, idx, leaf))` with linear forms (`scope` is a `SymScope` or a concrete `Scope`)."""
    while True:
        if space == "p":
            return sym_param_leaf(scope, idx, leaf, n)
        body = scope.spec.body
        node = body.nodes[idx]
        node_off = _add(_lin(scope.offset), (body.offsets[idx], 0))
        if node.form == "primitive":
            return node_off
        fn = node.fn
        if node.form == "call":
            scope = SymScope(fn, node_off, scope, idx, (0, 0))
            space, idx, leaf = _ref_at(fn.body.ret, leaf, n)
            continue
        if node.form == "batch":
            m, j = _divmod_lin(leaf, fn.out_leaves, n)
            if isinstance(fn, PrimitiveDefinition):
                return _add(node_off, m)
            scope = SymScope(fn, _add(node_off, _mul(m, fn.gates)), scope, idx, m)
            space, idx, leaf = _ref_at(fn.body.ret, j, n)
            continue
        # scan: output is (carry_n, Array<n, Y>); the iteration index must be concrete
        if leaf[1] != 0:
            raise NotVectorizable("scan output indexed by the member")
        c = node.out.fields[0].leaves
        if leaf[0] < c:
            m_, j = node.n - 1, leaf[0]
        else:
            y = node.out.fields[1].elem.leaves
            m_, r = divmod(leaf[0] - c, y)
            j = c + r
        scope = SymScope(fn, _add(node_off, (m_ * fn.gates, 0)), scope, idx, (m_, 0))
        space, idx, leaf = _ref_at(fn.body.ret, (j, 0), n)


def sym_param_leaf(scope, pi: int, leaf: Lin, n: int) -> Lin:
    """`layout.param_leaf(scope, pi, leaf)` with linear forms."""
    parent = scope.parent
    if parent is None:
        starts = getattr(scope, "param_starts", None)
        if starts is None:
            raise NotVectorizable("root parameter")
        return _add((PARAM + starts[pi], 0), leaf)
    node = parent.spec.body.nodes[scope.node_idx]
    member = _lin(scope.member)
    if node.form == "call":
        return sym_resolve(parent, *_ref_at(node.args[pi].refs, leaf, n), n)
    if node.form == "batch":
        ax = node.axes[pi]
        a = node.args[pi]
        if ax is None:
            return sym_resolve(parent, *_ref_at(a.refs, leaf, n), n)
        if ax == 0:
            L = a.type.elem.leaves
            return sym_resolve(parent, *_ref_at(a.refs, _add(_mul(member, L), leaf), n), n)
        if member[1]:
            raise NotVectorizable("take on a non-leading axis by the member")
        return sym_resolve(parent, *_ref_at(a.take(ax, member[0]).refs, leaf, n), n)
    # scan: params are (carry, xs..., shared...)
    if member[1]:
        raise NotVectorizable("scan iteration indexed by the member")
    m = member[0]
    if pi == 0:
        if m == 0:
            return sym_resolve(parent, *_ref_at(node.args[0].refs, leaf, n), n)
        prev = SymScope(scope.spec, _add(_lin(scope.offset), (-scope.spec.gates, 0)), parent, scope.node_idx, (m - 1, 0))
        return sym_resolve(prev, *_ref_at(scope.spec.body.ret, leaf, n), n)
    if pi <= node.n_xs:
        a = node.args[pi]
        L = a.type.elem.leaves
        return sym_resolve(parent, *_ref_at(a.refs, _add((m * L, 0), leaf), n), n)
    return sym_resolve(parent, *_ref_at(node.args[pi].refs, leaf, n), n)


# ---------------------------------------------------------------------------------------------------------------------------------
# the template of one member
# ---------------------------------------------------------------------------------------------------------------------------------


class RunBlock:
    """`n` members' runs at once: run `r` of member `m` starts at `starts[r][0] + starts[r][1] * m`."""

    __slots__ = ("n", "runs")

    def __init__(self, n: int, runs: list[tuple[Lin, int, str, int]]) -> None:
        self.n, self.runs = n, runs

    def expand(self) -> Iterable[tuple[int, int, str, int]]:
        for m in range(self.n):
            for lin, count, kind, width in self.runs:
                yield (_at(lin, m), count, kind, width)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        """(starts, counts, kinds, widths) of every run of every member in gate order (members are contiguous, runs in member order)."""
        c0 = np.array([r[0][0] for r in self.runs], dtype=np.int64)
        c1 = np.array([r[0][1] for r in self.runs], dtype=np.int64)
        m = np.arange(self.n, dtype=np.int64)
        starts = (c0[None, :] + c1[None, :] * m[:, None]).ravel()
        counts = np.tile(np.array([r[1] for r in self.runs], dtype=np.int64), self.n)
        kinds = [r[2] for r in self.runs] * self.n
        widths = np.tile(np.array([r[3] for r in self.runs], dtype=np.int64), self.n)
        return starts, counts, kinds, widths


class Template:
    """Every gate of one member of a composite `batch` node, with the member index symbolic."""

    def __init__(self, sc, k: int, node, node_off: int) -> None:
        fn = node.fn
        self.n, self.g, self.base = node.n, fn.gates, node_off
        self.gates: list[tuple[Lin, PrimitiveDefinition, list[Lin]]] = []
        self.runs: list[tuple[Lin, int, str, int]] = []
        member = SymScope(fn, (node_off, fn.gates), sc, k, (0, 1))
        self._walk(member)
        if len(self.gates) != self.g:
            raise NotVectorizable(f"template has {len(self.gates)} gates, member has {self.g}")
        for lin, prim, _ in self.gates:
            if prim.ret.w > 64:
                raise NotVectorizable(f"{prim.id} is wider than 64 bits")
            if lin[1] != self.g or not (node_off <= lin[0] < node_off + self.g):
                raise NotVectorizable("template gate outside its member")

    def _args(self, sc: SymScope, node, m: int) -> list[Lin]:
        n = self.n
        if node.form == "primitive":
            return [sym_resolve(sc, *_ref_at(a.refs, (j, 0), n), n) for a in node.args for j in range(a.refs.count)]
        out: list[Lin] = []
        for a, ax in zip(node.args, node.axes, strict=True):
            if ax is None:
                out.extend(sym_resolve(sc, *_ref_at(a.refs, (j, 0), n), n) for j in range(a.refs.count))
            elif ax == 0:
                L = a.type.elem.leaves
                out.extend(sym_resolve(sc, *_ref_at(a.refs, (m * L + j, 0), n), n) for j in range(L))
            else:
                refs = a.take(ax, m).refs
                out.extend(sym_resolve(sc, *_ref_at(refs, (j, 0), n), n) for j in range(refs.count))
        return out

    def _walk(self, sc: SymScope) -> None:
        body = sc.spec.body
        for k, node in enumerate(body.nodes):
            node_off = _add(sc.offset, (body.offsets[k], 0))
            fn = node.fn
            if isinstance(fn, PrimitiveDefinition):
                if sum(p.leaves for _, p in fn.params) > WIDE_FAN_IN:
                    raise NotVectorizable(f"wide fan-in {fn.id}")
                if node.form == "primitive":
                    self.runs.append((node_off, 1, f"{fn.id}|primitive", fn.ret.w))
                    self.gates.append((node_off, fn, self._args(sc, node, 0)))
                else:
                    self.runs.append((node_off, node.n, f"{fn.id}|{node.form}", fn.ret.w))
                    for m in range(node.n):
                        self.gates.append((_add(node_off, (m, 0)), fn, self._args(sc, node, m)))
            elif node.form == "call":
                self._walk(SymScope(fn, node_off, sc, k, (0, 0)))
            else:
                g = fn.gates
                for m in range(node.n):
                    self._walk(SymScope(fn, _add(node_off, (m * g, 0)), sc, k, (m, 0)))

    def check(self, m: int, concrete: Iterable[Any]) -> None:
        """Compare the template at member `m` with the producer's concrete gate walk of that member (`Gate` objects)."""
        for (lin, prim, ops), gv in zip(self.gates, concrete, strict=True):
            if gv.index != _at(lin, m) or gv.prim is not prim or tuple(gv.operands) != tuple(_at(o, m) for o in ops):
                raise NotVectorizable(f"template != concrete walk at member {m}, gate {gv.index} ({prim.id})")

    def run_block(self) -> RunBlock:
        return RunBlock(self.n, self.runs)

    # -- evaluation ---------------------------------------------------------------------------------------------------------------
    def evaluate(self, tr, vec: dict[str, Callable[..., np.ndarray]], stats: dict | None = None) -> None:
        """Evaluate all `n` members into `tr` (every gate of the node)."""
        n, g, base = self.n, self.g, self.base
        m_arr = np.arange(n, dtype=np.int64)
        cols: dict[int, np.ndarray] = {}
        gather = getattr(tr, "gather", None)

        def col(op: Lin) -> np.ndarray:
            c0, c1 = op
            if c1 == g and base <= c0 < base + g:
                return cols[c0 - base]
            if c1 == 0:
                return np.full(n, tr[c0], dtype=np.uint64)
            gates = c0 + c1 * m_arr
            if gather is not None:
                return gather(gates)
            return np.array([tr[x] for x in gates.tolist()], dtype=np.uint64)

        for lin, prim, ops in self.gates:
            f = vec.get(prim.id) or (vec.get("Const") if getattr(prim, "family", None) == "Const" else None)
            if f is not None:
                vals = f(n, prim, *[col(o) for o in ops])
                if stats is not None:
                    stats["vec_gates"] += n
            else:
                ev = prim.evaluate
                if ops:
                    vals = np.fromiter((ev(*t) for t in zip(*[col(o).tolist() for o in ops], strict=True)), dtype=np.uint64, count=n)
                else:
                    vals = np.full(n, ev(), dtype=np.uint64)
                if stats is not None:
                    stats["scalar_gates"] += n
            cols[lin[0] - base] = vals
        # write back: gate (j, m) = base + j + g*m
        for j in range(g):
            tr.update(zip(range(base + j, base + j + g * n, g), cols[j].tolist(), strict=True))

    def evaluate_rel(self, vals: np.ndarray, plv: np.ndarray, vec: dict[str, Callable[..., np.ndarray]], stats: dict | None = None) -> None:
        """`evaluate` for a member template inside a `RootTemplate`: gate offsets relative to the call (`vals`), parameter leaves'
        words in `plv` (codes `PARAM + leaf`), results into `vals`."""
        n, g, base = self.n, self.g, self.base
        m_arr = np.arange(n, dtype=np.int64)
        cols: dict[int, np.ndarray] = {}
        P = PARAM

        def col(op: Lin) -> np.ndarray:
            c0, c1 = op
            if c1 == g and base <= c0 < base + g:
                return cols[c0 - base]
            if c0 >= P:
                return np.full(n, plv[c0 - P], dtype=np.uint64) if c1 == 0 else plv[(c0 - P) + c1 * m_arr]
            if c1 == 0:
                return np.full(n, vals[c0], dtype=np.uint64)
            return vals[c0 + c1 * m_arr]

        for lin, prim, ops in self.gates:
            f = vec.get(prim.id) or (vec.get("Const") if getattr(prim, "family", None) == "Const" else None)
            if f is not None:
                v = f(n, prim, *[col(o) for o in ops])
                if stats is not None:
                    stats["vec_gates"] += n
            else:
                ev = prim.evaluate
                if ops:
                    v = np.fromiter((ev(*t) for t in zip(*[col(o).tolist() for o in ops], strict=True)), dtype=np.uint64, count=n)
                else:
                    v = np.full(n, ev(), dtype=np.uint64)
                if stats is not None:
                    stats["scalar_gates"] += n
            cols[lin[0] - base] = v
        for j in range(g):
            vals[base + j:base + j + g * n:g] = cols[j]


# ---------------------------------------------------------------------------------------------------------------------------------
# flat template of a whole function (E2 SCALE follow-up: the attention calls -- 3-member head batches over unrolled per-key
# nodes, nothing for the member template to vectorise)
# ---------------------------------------------------------------------------------------------------------------------------------


def _abs_op(op: Lin, m: int, base: int, pl: list[int]) -> int:
    c0, c1 = op
    if c0 >= PARAM:
        return pl[c0 - PARAM + c1 * m]
    return base + c0 + c1 * m


def _schedule(gates: list[tuple[int, PrimitiveDefinition, tuple[int, ...]]]) -> list[tuple[PrimitiveDefinition, np.ndarray, np.ndarray]]:
    """(level, primitive) groups of flat gates in level order: a gate is one level above its deepest operand IN THIS LIST; operands
    outside it (earlier segments, parameter leaves) are known before the segment starts."""
    level: dict[int, int] = {}
    groups: dict[tuple[int, str], tuple[PrimitiveDefinition, list[int], list[tuple[int, ...]]]] = {}
    for rel, prim, ops in gates:
        lv = 0
        for c in ops:
            if c < PARAM:
                lc = level.get(c)
                if lc is not None and lc >= lv:
                    lv = lc + 1
        level[rel] = lv
        key = (lv, prim.id)
        grp = groups.get(key)
        if grp is None:
            grp = groups[key] = (prim, [], [])
        grp[1].append(rel)
        grp[2].append(ops)
    out = []
    for key in sorted(groups, key=lambda kv: (kv[0], groups[kv][1][0])):
        prim, idx, ops = groups[key]
        out.append((prim, np.array(idx, dtype=np.int64), np.array(ops, dtype=np.int64).reshape(len(idx), len(ops[0]))))
    return out


SMALL_GROUP = 4  # (level, primitive) groups below this many gates (the argmax chain of `final`: one gate per level) skip numpy


def _eval_flat(schedule, vals: np.ndarray, plv: np.ndarray, vec: dict[str, Callable[..., np.ndarray]], stats: dict | None) -> None:
    P = PARAM
    small = 0
    for prim, gidx, ops in schedule:
        m = len(gidx)
        if m < SMALL_GROUP:
            ev = prim.evaluate
            for i in range(m):
                args = [int(plv[c - P]) if c >= P else int(vals[c]) for c in ops[i].tolist()]
                vals[gidx[i]] = ev(*args)
            small += m
            continue
        cols = []
        for j in range(ops.shape[1]):
            code = ops[:, j]
            ext = code >= P
            if not ext.any():
                cols.append(vals[code])
            elif ext.all():
                cols.append(plv[code - P])
            else:
                col = np.empty(len(code), dtype=np.uint64)
                col[~ext] = vals[code[~ext]]
                col[ext] = plv[code[ext] - P]
                cols.append(col)
        f = vec.get(prim.id) or (vec.get("Const") if getattr(prim, "family", None) == "Const" else None)
        m = len(gidx)
        if f is not None:
            out = f(m, prim, *cols)
            if stats is not None:
                stats["vec_gates"] += m
        else:
            ev = prim.evaluate
            if cols:
                out = np.fromiter((ev(*t) for t in zip(*[c.tolist() for c in cols], strict=True)), dtype=np.uint64, count=m)
            else:
                out = np.full(m, ev(), dtype=np.uint64)
            if stats is not None:
                stats["scalar_gates"] += m
        vals[gidx] = out
    if stats is not None and small:
        stats["scalar_gates"] += small


class FlatSeg:
    """A maximal run of gates outside any member-vectorised node, level-scheduled."""

    __slots__ = ("lo", "hi", "schedule", "runs")

    def __init__(self, gates: list[tuple[Lin, PrimitiveDefinition, list[Lin]]], runs: list[tuple[Lin, int, str, int]]) -> None:
        self.lo, self.hi = gates[0][0][0], gates[-1][0][0] + 1
        flat = []
        for i, (lin, prim, ops) in enumerate(gates):
            rel = lin[0]
            if lin[1] != 0 or rel != self.lo + i:
                raise NotVectorizable("flat gates not contiguous / in order")
            if prim.ret.w > 64:
                raise NotVectorizable(f"{prim.id} is wider than 64 bits")
            for c0, c1 in ops:
                if c1 != 0 or (rel <= c0 < PARAM):
                    raise NotVectorizable("flat operand is not an earlier gate or a parameter leaf")
            flat.append((rel, prim, tuple(c0 for c0, _ in ops)))
        self.schedule = _schedule(flat)
        self.runs = runs


class MemberSeg:
    """A composite `batch` node of >= `min_members` members as a member template (relative to the root function)."""

    __slots__ = ("lo", "hi", "tpl")

    def __init__(self, tpl: Template) -> None:
        self.tpl = tpl
        self.lo, self.hi = tpl.base, tpl.base + tpl.g * tpl.n


class RootTemplate(Template):
    """Every gate of one FUNCTION (a root call's) in gate order, RELATIVE to the call and built once per function id: operands are
    coded as an internal gate offset (`< PARAM`) or a parameter leaf (`PARAM + flat leaf index`), so the template of `LayerPost`,
    `LayerPre`, `Attention{T}` ... is built and checked once and re-used by every call of it (before: the member templates were
    rebuilt and re-checked at every root call).  The gates are a sequence of SEGMENTS in gate order: composite `batch` nodes of
    >= `min_members` members are member templates (`Template`, the GEMMs); everything else (RMSNorm chains, the attention heads'
    per-key nodes) forms flat segments, level-scheduled: the gates of one primitive whose operands are all known are one numpy
    column, evaluated by the primitive's exact twin or, without a twin, by its own `evaluate` gate by gate -- same function, same
    operands, no per-gate resolution.  Per call: the parameter leaves' gate indices (`param_leaves`), one gather of their words,
    the segments in order into one values array, one write-back.  `check` compares the template instantiated at a call with the
    producer's concrete walk of that call (gate index, primitive, operands)."""

    def __init__(self, fn, min_members: int) -> None:
        self.fn = fn
        self.n, self.g, self.base = 1, fn.gates, 0
        self.min_members = min_members
        self.gates = []
        self.runs = []
        self.segments: list[FlatSeg | MemberSeg] = []
        self.fallbacks: list[str] = []
        starts: list[int] = []
        acc = 0
        for _, t in fn.body.params:
            starts.append(acc)
            acc += t.leaves
        self.n_leaves = acc
        self._walk(SymScope(fn, (0, 0), None, -1, (0, 0), tuple(starts)))
        self._flush()
        pos = 0
        for seg in self.segments:
            if seg.lo != pos:
                raise NotVectorizable("root template segments not contiguous")
            pos = seg.hi
        if pos != self.g:
            raise NotVectorizable(f"root template covers {pos} gates, function has {self.g}")

    def _flush(self) -> None:
        if self.gates:
            self.segments.append(FlatSeg(self.gates, self.runs))
            self.gates, self.runs = [], []

    def _walk(self, sc: SymScope) -> None:
        body = sc.spec.body
        for k, node in enumerate(body.nodes):
            node_off = _add(sc.offset, (body.offsets[k], 0))
            fn = node.fn
            if isinstance(fn, PrimitiveDefinition):
                if sum(p.leaves for _, p in fn.params) > WIDE_FAN_IN:
                    raise NotVectorizable(f"wide fan-in {fn.id}")
                if node.form == "primitive":
                    self.runs.append((node_off, 1, f"{fn.id}|primitive", fn.ret.w))
                    self.gates.append((node_off, fn, self._args(sc, node, 0)))
                else:
                    self.runs.append((node_off, node.n, f"{fn.id}|{node.form}", fn.ret.w))
                    for m in range(node.n):
                        self.gates.append((_add(node_off, (m, 0)), fn, self._args(sc, node, m)))
            elif node.form == "call":
                self._walk(SymScope(fn, node_off, sc, k, (0, 0)))
            else:
                if node.form == "batch" and node.n >= self.min_members:
                    try:
                        tpl = Template(sc, k, node, node_off[0])
                    except NotVectorizable as e:
                        self.fallbacks.append(f"{fn.id} x{node.n}: {e}")
                    else:
                        self._flush()
                        self.segments.append(MemberSeg(tpl))
                        continue
                g = fn.gates
                for m in range(node.n):
                    self._walk(SymScope(fn, _add(node_off, (m * g, 0)), sc, k, (m, 0)))

    def member_los(self) -> set[int]:
        return {seg.lo for seg in self.segments if isinstance(seg, MemberSeg)}

    def check(self, base: int, pl: list[int], concrete: Iterable[Any], keep=None) -> None:  # type: ignore[override]
        """Template at a call with lower gate `base` and parameter leaves `pl` vs the producer's concrete gate walk of that call
        (`concrete`: every gate, or -- `keep` given -- every flat gate and the members `keep(n)` of each member template, as
        `CallWalk._scope_selective` enumerates them)."""
        it = iter(concrete)
        for seg in self.segments:
            if isinstance(seg, FlatSeg):
                exp = {}
                for prim, gidx, ops in seg.schedule:
                    for rel, row in zip(gidx.tolist(), ops.tolist(), strict=True):
                        exp[rel] = (prim, row)
                for rel in range(seg.lo, seg.hi):
                    prim, row = exp[rel]
                    gv = next(it)
                    if gv.index != base + rel or gv.prim is not prim or \
                            tuple(gv.operands) != tuple(pl[c - PARAM] if c >= PARAM else base + c for c in row):
                        raise NotVectorizable(f"root template != concrete walk at gate {gv.index} ({prim.id})")
            else:
                tpl = seg.tpl
                for m in (range(tpl.n) if keep is None else keep(tpl.n)):
                    for lin, prim, ops in tpl.gates:
                        gv = next(it)
                        if gv.index != base + _at(lin, m) or gv.prim is not prim or \
                                tuple(gv.operands) != tuple(_abs_op(o, m, base, pl) for o in ops):
                            raise NotVectorizable(f"root template != concrete walk at gate {gv.index} ({prim.id}, member {m})")
        if next(it, None) is not None:
            raise NotVectorizable("concrete walk has more gates than the root template")

    def run_blocks(self, base: int) -> list[RunBlock]:
        out = []
        for seg in self.segments:
            if isinstance(seg, FlatSeg):
                out.append(RunBlock(1, [((base + lin[0], 0), count, kind, width) for lin, count, kind, width in seg.runs]))
            else:
                out.append(RunBlock(seg.tpl.n, [((base + lin[0], lin[1]), count, kind, width) for lin, count, kind, width in seg.tpl.runs]))
        return out

    def evaluate(self, base: int, pl: np.ndarray, tr, vec: dict[str, Callable[..., np.ndarray]], stats: dict | None = None) -> None:  # type: ignore[override]
        """Evaluate one call of the function (gates `[base, base + g)`, parameter leaves at gate indices `pl`) into `tr`."""
        gather = getattr(tr, "gather", None)
        plv = gather(pl) if gather is not None else np.array([tr[x] for x in pl.tolist()], dtype=np.uint64)
        vals = np.zeros(self.g, dtype=np.uint64)
        for seg in self.segments:
            if isinstance(seg, FlatSeg):
                _eval_flat(seg.schedule, vals, plv, vec, stats)
            else:
                seg.tpl.evaluate_rel(vals, plv, vec, stats)
                if stats is not None:
                    stats["vec_nodes"] += 1
        add_block = getattr(tr, "add_block", None)
        if add_block is not None:
            add_block(base, vals)  # one call block (StreamTranscript); dropped to its returns after the rows are written
        else:
            tr.update(zip(range(base, base + self.g), vals.tolist(), strict=True))


def check_members(n: int) -> list[int]:
    """The members of a member template the root template's check compares with the concrete walk: 0, 1 and n-1 (the member
    template is symbolic in the member index; three members pin the linear forms, as `Template.check` did per node)."""
    return sorted({0, min(1, n - 1), n - 1})


_FLAT_ELIGIBLE: dict[str, bool] = {}
_FLAT: dict[str, RootTemplate | None] = {}
FLAT_CHECKED: set[str] = set()
FLAT_MAX_GATES = 4_000_000


def flat_eligible(fn, min_members: int) -> bool:
    """A function with no wide-fan-in primitive and at most `FLAT_MAX_GATES` gates: the root template's domain."""
    if fn.gates > FLAT_MAX_GATES:
        return False

    def ok(f) -> bool:
        key = f.id
        if key in _FLAT_ELIGIBLE:
            return _FLAT_ELIGIBLE[key]
        _FLAT_ELIGIBLE[key] = True
        r = True
        for node in f.body.nodes:
            nf = node.fn
            if isinstance(nf, PrimitiveDefinition):
                if nf.ret.w > 64 or sum(p.leaves for _, p in nf.params) > WIDE_FAN_IN:
                    r = False
                    break
            elif not ok(nf):
                r = False
                break
        _FLAT_ELIGIBLE[key] = r
        return r

    return ok(fn)


def flat_template(fn, min_members: int, stats: dict | None = None) -> RootTemplate | None:
    """The cached root template of `fn`, or None (not eligible / not buildable -- the per-node walk applies)."""
    key = fn.id
    if key in _FLAT:
        return _FLAT[key]
    ft = None
    if flat_eligible(fn, min_members):
        try:
            ft = RootTemplate(fn, min_members)
            if stats is not None and ft.fallbacks:
                stats["not_vectorizable"].extend(f"root {key}: {x}" for x in ft.fallbacks[:4])
        except NotVectorizable as e:
            if stats is not None:
                stats["not_vectorizable"].append(f"flat {key}: {e}")
    _FLAT[key] = ft
    return ft


def param_leaves(call) -> np.ndarray:
    """Gate indices of every parameter leaf of `call` in signature order (== `FunctionCall.args`), a linear run per parameter when
    the references allow it, `layout.param_leaf` leaf by leaf otherwise."""
    from verity.ir.layout import param_leaf
    sc = call.scope()
    params = sc.spec.body.params
    out = np.empty(sum(t.leaves for _, t in params), dtype=np.int64)
    pos = 0
    for pi, (_, t) in enumerate(params):
        # leaves [lo, hi) of parameter pi: one linear run when the references allow it, else split where the resolver says the
        # linear form stops holding (`NotVectorizable.limit`: a `Concat` of per-position parts -- the K/V caches -- resolves part by
        # part, a batch's output member by member), single leaves through `param_leaf`.  Iterative, no exception outlives its
        # handler (a recursive form retained the frames -- and this array -- across calls).
        work = [(0, t.leaves)]
        while work:
            lo, hi = work.pop()
            n = hi - lo
            limit = None
            try:
                c0, c1 = sym_param_leaf(sc, pi, (lo, 1), n)
            except NotVectorizable as e:
                limit = e.limit
                ok = False
            else:
                ok = True
            if ok:
                out[pos + lo:pos + hi] = c0 + c1 * np.arange(n, dtype=np.int64)
            elif n <= 4:
                for j in range(lo, hi):
                    out[pos + j] = param_leaf(sc, pi, j)
            else:
                mid = lo + (limit if limit is not None and 0 < limit < n else n // 2)
                work.append((mid, hi))
                work.append((lo, mid))
        pos += t.leaves
    return out


def concrete_gates_of_call(walk) -> Iterable[Any]:
    """The producer's own gate enumeration of the whole call (`CallWalk._scope`), runs discarded."""
    saved, walk.runs = walk.runs, []
    try:
        yield from walk._scope(walk.call.scope())
    finally:
        walk.runs = saved


# ---------------------------------------------------------------------------------------------------------------------------------
# exact numpy twins of the hot primitives
# ---------------------------------------------------------------------------------------------------------------------------------

_MIN_NORMAL_EXP = -126
_MAX_EXP = 127
_WIDTH = 25  # Ampere adder
_ZERO_EXP = -132
_RESCALE = _WIDTH - 24


def _bit_length(x: np.ndarray) -> np.ndarray:
    """bit_length of non-negative int64 < 2**53 (exact through frexp)."""
    return np.frexp(x.astype(np.float64))[1].astype(np.int64)


def _shr(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """x >> s for non-negative x with arbitrary non-negative s (numpy leaves s >= 64 undefined)."""
    return np.where(s >= 63, 0, x >> np.clip(s, 0, 63))


def _group_sum(exps: list[np.ndarray], mags: list[np.ndarray], negs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`silicon.group_sum` over columns of (exponent, magnitude, negative) terms; magnitude 0 = the unsigned zero."""
    n = mags[0].shape[0]
    max_exp = np.full(n, _ZERO_EXP, dtype=np.int64)
    for e, m in zip(exps, mags, strict=True):
        max_exp = np.where(m != 0, np.maximum(max_exp, e), max_exp)
    total = np.zeros(n, dtype=np.int64)
    for e, m, ng in zip(exps, mags, negs, strict=True):
        scaled = m << _RESCALE
        aligned = _shr(scaled, max_exp - e)
        total += np.where(ng, -aligned, aligned)
    neg = total < 0
    mag = np.abs(total)
    bl = _bit_length(mag)
    exp = max_exp + bl - _WIDTH
    mag = np.where(bl > _WIDTH, _shr(mag, bl - _WIDTH), mag << np.maximum(_WIDTH - bl, 0))
    below = exp < _MIN_NORMAL_EXP
    mag = np.where(below, _shr(mag, _MIN_NORMAL_EXP - exp), mag)
    exp = np.where(below, _MIN_NORMAL_EXP, exp)
    mag = mag >> _RESCALE
    return exp, mag, neg


def _pack_fp32(exp: np.ndarray, mag: np.ndarray, neg: np.ndarray) -> np.ndarray:
    sign = neg.astype(np.int64) << 31
    ef = exp + 127 - ((mag & (1 << 23)) == 0).astype(np.int64)
    out = sign | (ef << 23) | (mag & 0x7FFFFF)
    out = np.where(exp > _MAX_EXP, sign | 0x7F800000, out)
    return np.where(mag == 0, 0, out)


def vec_tc_dot16(n: int, prim, acc: np.ndarray, *ab: np.ndarray) -> np.ndarray:
    """`AmpereBF16TcDot16` over `n` members: `silicon.tc_dot` (groups (8, 8), width 25, floor -132) on int64 columns; members with a
    non-finite operand or accumulator take the scalar extension path (`prims._tc_dot16_total`)."""
    a, b = [x.astype(np.int64) for x in ab[:16]], [x.astype(np.int64) for x in ab[16:32]]
    acc = acc.astype(np.int64)
    nonfinite = (acc & 0x7F800000) == 0x7F800000
    for x in a + b:
        nonfinite |= (x & 0x7F80) == 0x7F80
    # accumulator term
    ef = (acc >> 23) & 0xFF
    man = acc & 0x7FFFFF
    acc_neg = (acc >> 31) == 1
    acc_exp = np.where(ef == 0, _MIN_NORMAL_EXP, ef - 127)
    acc_mag = np.where(ef == 0, man, man | (1 << 23))
    # products
    p_exp, p_mag, p_neg = [], [], []
    for x, y in zip(a, b, strict=True):
        ea, eb = (x >> 7) & 0xFF, (y >> 7) & 0xFF
        sa = (x & 0x7F) | np.where(ea != 0, 0x80, 0)
        sb = (y & 0x7F) | np.where(eb != 0, 0x80, 0)
        p_mag.append((sa * sb) << 9)
        p_exp.append(np.maximum(ea, 1) + np.maximum(eb, 1) - 254)
        p_neg.append((((x ^ y) >> 15) & 1) == 1)
    e1, m1, n1 = _group_sum([acc_exp] + p_exp[:8], [acc_mag] + p_mag[:8], [acc_neg] + p_neg[:8])
    sticky = (m1 != 0) & (e1 > _MAX_EXP)
    e2, m2, n2 = _group_sum([e1] + p_exp[8:], [m1] + p_mag[8:], [n1] + p_neg[8:])
    out = np.where(sticky, _pack_fp32(e1, m1, n1), _pack_fp32(e2, m2, n2))
    if nonfinite.any():
        from verity_vllm.registry.prims import _tc_dot16_total
        idx = np.nonzero(nonfinite)[0]
        for i in idx.tolist():
            out[i] = _tc_dot16_total(int(acc[i]), tuple(int(x[i]) for x in a), tuple(int(y[i]) for y in b))
    return out.astype(np.uint64)


def vec_f2fp_bf16(n: int, prim, a: np.ndarray) -> np.ndarray:
    """`F2fpBf16`: RNE f32 -> bf16, every NaN -> 0x7FFF."""
    u = a.astype(np.int64) & 0xFFFFFFFF
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0)
    lsb = (u >> 16) & 1
    return np.where(nan, 0x7FFF, ((u + 0x7FFF + lsb) >> 16) & 0xFFFF).astype(np.uint64)


def vec_const(n: int, prim) -> np.ndarray:
    return np.full(n, prim.evaluate(), dtype=np.uint64)


# -- the FA2 fast-math f32 family (attention): `prims.f32_add_bits` / `f32_mul_bits` ARE numpy float32 arithmetic on one word;
#    the twins are the same arithmetic on a column, `ftz_bits` as a mask, `f32_max_bits` as a select, the MUFU models on arrays
#    (`fa2_model.mufu_ex2` / `mufu_rcp` take arrays; `prims._scalar` wraps them for one word).

def _ftz(u: np.ndarray) -> np.ndarray:
    """`prims.ftz_bits` on an int64 column of f32 encodings."""
    u = u & 0xFFFFFFFF
    sub = ((u & 0x7F800000) == 0) & ((u & 0x7FFFFF) != 0)
    return np.where(sub, u & 0x80000000, u)


def _f32(u: np.ndarray) -> np.ndarray:
    return (u & 0xFFFFFFFF).astype(np.uint32).view(np.float32)


def _bits(f: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(f, dtype=np.float32).view(np.uint32).astype(np.int64)


def vec_f32_add_ftz(n: int, prim, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ua, ub = _ftz(a.astype(np.int64)), _ftz(b.astype(np.int64))
    with np.errstate(all="ignore"):
        return _ftz(_bits(_f32(ua) + _f32(ub))).astype(np.uint64)


def vec_f32_mul_ftz(n: int, prim, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ua, ub = _ftz(a.astype(np.int64)), _ftz(b.astype(np.int64))
    with np.errstate(all="ignore"):
        return _ftz(_bits(_f32(ua) * _f32(ub))).astype(np.uint64)


def vec_f32_max(n: int, prim, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ua, ub = _ftz(a.astype(np.int64)), _ftz(b.astype(np.int64))
    with np.errstate(all="ignore"):
        return np.where(_f32(ua) > _f32(ub), ua, ub).astype(np.uint64)


def vec_guard_neg_inf_zero(n: int, prim, m: np.ndarray) -> np.ndarray:
    u = m.astype(np.int64)
    return np.where((u & 0xFFFFFFFF) == 0xFF800000, 0, u).astype(np.uint64)


def vec_mufu_ex2_ftz(n: int, prim, x: np.ndarray) -> np.ndarray:
    from verity_vllm.registry.prims import _fa2
    fa2, t = _fa2()
    return _bits(fa2.mufu_ex2(_f32(x.astype(np.int64)), t)).astype(np.uint64)


def vec_mufu_rcp_ftz(n: int, prim, x: np.ndarray) -> np.ndarray:
    from verity_vllm.registry.prims import _fa2
    fa2, t = _fa2()
    return _bits(fa2.mufu_rcp(_f32(x.astype(np.int64)), t)).astype(np.uint64)


def vec_fa2_inv_sum(n: int, prim, s: np.ndarray) -> np.ndarray:
    u = s.astype(np.int64) & 0xFFFFFFFF
    one = ((u & 0x7FFFFFFF) == 0) | (((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0))
    return np.where(one, 0x3F800000, vec_mufu_rcp_ftz(n, prim, u).astype(np.int64)).astype(np.uint64)


# -- the IEEE f32 / bf16 element-wise family (RMSNorm chains, residual adds, RoPE): `prims.f32_add_bits` & co. are numpy float32
#    arithmetic on one word; the twins are the same arithmetic on a column.  The fused multiply-add is exact rational in `prims`;
#    the twin computes RN64(a*b + c) (the f32 product is exact in f64) and casts to f32, which equals the single rounding RN32(a*b + c)
#    unless the f64 sum was inexact AND landed exactly on an f32 midpoint (or the result is in the f32 subnormal / overflow range,
#    or an operand is non-finite): those lanes -- rare, detected exactly with a TwoSum error term -- take `prims.f32_fma_bits`.

_BF16_NAN_TORCH = 0x7FC0
_BF16_NAN_F2FP = 0x7FFF


def _bf16_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.int64) & 0xFFFF) << 16


def _f32_to_bf16_rn(u: np.ndarray, nan: int) -> np.ndarray:
    u = u & 0xFFFFFFFF
    isnan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0)
    lsb = (u >> 16) & 1
    return np.where(isnan, nan, ((u + 0x7FFF + lsb) >> 16) & 0xFFFF)


def vec_bf16_to_f32(n: int, prim, a: np.ndarray) -> np.ndarray:
    return _bf16_to_f32(a).astype(np.uint64)


def vec_f32_add(n: int, prim, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return _bits(_f32(a.astype(np.int64)) + _f32(b.astype(np.int64))).astype(np.uint64)


def vec_f32_mul(n: int, prim, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return _bits(_f32(a.astype(np.int64)) * _f32(b.astype(np.int64))).astype(np.uint64)


def vec_f32_div(n: int, prim, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return _bits(_f32(a.astype(np.int64)) / _f32(b.astype(np.int64))).astype(np.uint64)


def _fma_bits(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """RN32(a*b + c), one rounding, on int64 columns of f32 encodings (see the family comment)."""
    fa, fb, fc = _f32(a).astype(np.float64), _f32(b).astype(np.float64), _f32(c).astype(np.float64)
    with np.errstate(all="ignore"):
        p = fa * fb  # exact: 24 + 24 bits < 53
        s = p + fc
        bb = s - p
        err = (p - (s - bb)) + (fc - bb)  # TwoSum: s + err == p + fc exactly (finite operands)
        r = s.astype(np.float32)
        finite = np.isfinite(fa) & np.isfinite(fb) & np.isfinite(fc) & np.isfinite(s)
        mid32 = (s.view(np.int64) & ((1 << 29) - 1)) == (1 << 28)  # RN64(x) exactly half-way between two f32 neighbours
        mag = np.abs(s)
        hazard = ~finite | ((err != 0) & mid32) | ((mag != 0) & (mag < 2.0 ** -125)) | (mag >= 2.0 ** 127)
    out = _bits(r)
    if hazard.any():
        from verity_vllm.registry.prims import f32_fma_bits
        for i in np.nonzero(hazard)[0].tolist():
            out[i] = f32_fma_bits(int(a[i]), int(b[i]), int(c[i]))
    return out


def vec_f32_fma(n: int, prim, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return _fma_bits(a.astype(np.int64) & 0xFFFFFFFF, b.astype(np.int64) & 0xFFFFFFFF, c.astype(np.int64) & 0xFFFFFFFF).astype(np.uint64)


def vec_f32_fma_ftz(n: int, prim, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return _ftz(_fma_bits(_ftz(a.astype(np.int64)), _ftz(b.astype(np.int64)), _ftz(c.astype(np.int64)))).astype(np.uint64)


def vec_f32_fma_sub_ftz(n: int, prim, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return _ftz(_fma_bits(_ftz(a.astype(np.int64)), _ftz(b.astype(np.int64)), _ftz(c.astype(np.int64)) ^ 0x80000000)).astype(np.uint64)


def vec_bf16_add_f2fp(n: int, prim, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        s = _bits(_f32(_bf16_to_f32(a)) + _f32(_bf16_to_f32(b)))
    return _f32_to_bf16_rn(s, _BF16_NAN_F2FP).astype(np.uint64)


def vec_bf16_mul_f32(n: int, prim, a: np.ndarray, s: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        p = _bits(_f32(_bf16_to_f32(a)) * _f32(s.astype(np.int64)))
    return _f32_to_bf16_rn(p, _BF16_NAN_F2FP).astype(np.uint64)


def vec_bf16_mul_bf16(n: int, prim, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        p = _bits(_f32(_bf16_to_f32(a)) * _f32(_bf16_to_f32(b)))
    return _f32_to_bf16_rn(p, _BF16_NAN_F2FP).astype(np.uint64)


def vec_rope_out(n: int, prim, x: np.ndarray, y: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        t = _bits(_f32(_bf16_to_f32(y)) * _f32(_bf16_to_f32(s)))
    o = _fma_bits(_bf16_to_f32(x), _bf16_to_f32(c), t ^ 0x80000000)
    return _f32_to_bf16_rn(o, _BF16_NAN_TORCH).astype(np.uint64)


def vec_rope_out_add(n: int, prim, x: np.ndarray, y: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        t = _bits(_f32(_bf16_to_f32(x)) * _f32(_bf16_to_f32(s)))
    o = _fma_bits(_bf16_to_f32(y), _bf16_to_f32(c), t)
    return _f32_to_bf16_rn(o, _BF16_NAN_TORCH).astype(np.uint64)


def vec_i32_add(n: int, prim, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a.astype(np.int64) + b.astype(np.int64)) & 0xFFFFFFFF).astype(np.uint64)


def vec_select(n: int, prim, c: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where((c.astype(np.int64) & 1) == 1, a, b).astype(np.uint64)


def vec_bf16_gt_strict(n: int, prim, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return (_f32(_bf16_to_f32(a)) > _f32(_bf16_to_f32(b))).astype(np.uint64)


VEC_EVALUATORS: dict[str, Callable[..., np.ndarray]] = {
    "AmpereBF16TcDot16_v1": vec_tc_dot16,
    "F2fpBf16_v1": vec_f2fp_bf16,
    "Const": vec_const,
    "F32AddFtz_v1": vec_f32_add_ftz,
    "F32MulFtz_v1": vec_f32_mul_ftz,
    "F32Max_v1": vec_f32_max,
    "GuardNegInfZero_v1": vec_guard_neg_inf_zero,
    "MufuEx2Ftz_v1": vec_mufu_ex2_ftz,
    "MufuRcpFtz_v1": vec_mufu_rcp_ftz,
    "Fa2InvSum_v1": vec_fa2_inv_sum,
}


def evaluators() -> dict[str, Callable[..., np.ndarray]]:
    """The twins keyed by primitive id (ids carry the version: `AmpereBF16TcDot16_v1`); resolved against the registry at import."""
    from verity_vllm.registry import prims as P
    out = {"Const": vec_const}
    out[P.AmpereBF16TcDot16.id] = vec_tc_dot16
    out[P.F2fpBf16.id] = vec_f2fp_bf16
    out[P.F32AddFtz.id] = vec_f32_add_ftz
    out[P.F32MulFtz.id] = vec_f32_mul_ftz
    out[P.F32Max.id] = vec_f32_max
    out[P.GuardNegInfZero.id] = vec_guard_neg_inf_zero
    out[P.MufuEx2Ftz.id] = vec_mufu_ex2_ftz
    out[P.MufuRcpFtz.id] = vec_mufu_rcp_ftz
    out[P.Fa2InvSum.id] = vec_fa2_inv_sum
    out[P.Bf16ToF32.id] = vec_bf16_to_f32
    out[P.F32Add.id] = vec_f32_add
    out[P.F32Mul.id] = vec_f32_mul
    out[P.F32Div.id] = vec_f32_div
    out[P.F32Fma.id] = vec_f32_fma
    out[P.F32FmaFtz.id] = vec_f32_fma_ftz
    out[P.F32FmaSubFtz.id] = vec_f32_fma_sub_ftz
    out[P.Bf16AddF2fp.id] = vec_bf16_add_f2fp
    out[P.Bf16MulF32.id] = vec_bf16_mul_f32
    out[P.Bf16MulBf16.id] = vec_bf16_mul_bf16
    out[P.RopeOut.id] = vec_rope_out
    out[P.RopeOutAdd.id] = vec_rope_out_add
    out[P.I32Add.id] = vec_i32_add
    out[P.SelectBf16.id] = vec_select
    out[P.SelectI32.id] = vec_select
    out[P.Bf16GtStrict.id] = vec_bf16_gt_strict
    return out


def concrete_gates_of_member(walk, sc: Scope, k: int, node, node_off: int, m: int) -> Iterable[Any]:
    """The producer's own gate enumeration of member `m` of the node (`CallWalk._scope`), runs discarded."""
    fn = node.fn
    saved, walk.runs = walk.runs, []
    try:
        yield from walk._scope(Scope(fn, node_off + m * fn.gates, sc, k, m))
    finally:
        walk.runs = saved

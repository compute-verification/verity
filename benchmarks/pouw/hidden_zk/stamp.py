"""The flat unit's lowering (`class_statement.unit_lowering`, inline reads) with each distinct sub-circuit built once: the same
rows, columns and netlist bytes, in memory proportional to the IR walk rather than to the unit's ANDs.

`ir_lower.lower_gates` builds a unit call by call: an activation of a Boolean Definition by its builder, any other primitive
by its piece. What one call commits depends only on its builder and on how its operand forms share bits: `forms.Circuit.AND`
reads only which forms are constant or equal, `commit` only each form's size. So each (builder, sharing pattern) is built
once, a *template*, on a scratch circuit whose first bits are the call's operand bits ranked by index, and every call is
recorded as its template, the index of its first AND and its operand bits (its *bind*). Its ANDs are the template's, renamed,
in the library's creation order.

The layout (`ir_lower.layout`, flock-ir-unit/v2) numbers the input bits first, then the live ANDs in creation order, so a
column is increasing in the bit's index: a template's rows, renamed, stay sorted, and a call's rows are its template's text
with its own column numbers. Liveness is `forms.Circuit.live`'s, backward over the calls: what a later call reads of an
earlier one is its output forms' bits, so a call's live ANDs are a closure of its template's, one per pattern of its output
bits that later rows (or the unit's outputs) read.

A unit with a table read, a hint or an assertion is not stamped (`Unstampable`): `fast_stage` lowers it with the library."""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import forms
from verity.ml.circuits.boolean.trace import _flat_out
from verity.ml.circuits.layout import param_leaf, resolve
from verity.ml.circuits.types import Array, Tuple, Value
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL

W = IL.WORD
#: live ANDs per rendering task (a call is never split): a task's text is about this many rows
TASK_ANDS = 1 << 16
#: rendered tasks held ahead of the writer, per worker
AHEAD = 2


class Unstampable(Exception):
    """A unit the stamp does not lay out: a template made a table read, a hint or an assertion."""


class Template:
    """One (builder, sharing pattern)'s circuit: ``m`` operand bits, then ``n`` ANDs as ``rows`` ``(A bits, A constant, B bits,
    B constant)`` over local bits (below ``m`` an operand's rank, else AND ``j`` at ``m + j``), its output forms, and ``vis``,
    the ANDs its outputs read: all a later call can read of it."""

    __slots__ = ("id", "label", "m", "n", "rows", "outs", "vis", "closures")

    def __init__(self, tid: int, label: str, m: int, S: forms.Circuit, outs: list):
        if S.lookups or S.asserts or any(k != "and" for k in S.kind[m:]):
            raise Unstampable(f"{label}'s template makes {sorted(set(S.kind[m:]) - {'and'}) or 'assertions or reads'}")
        self.id, self.label, self.m, self.n = tid, label, m, len(S.kind) - m
        self.rows = [(tuple(sorted(a.s)), a.c, tuple(sorted(b.s)), b.c) for a, b in S.data[m:]]
        self.outs = [(tuple(sorted(f.s)), f.c) for f in outs]
        self.vis = np.array(sorted({v - m for s, _ in self.outs for v in s if v >= m}), dtype=np.int64)
        self.closures: dict[bytes, Closure] = {}


class Closure:
    """A template's live ANDs when later rows read ``ext`` of its ``vis``: ``mask`` and ``rank`` (each live AND's place among
    them, -1 dead), and ``need``, the operand ranks the live rows read."""

    __slots__ = ("id", "t", "mask", "rank", "count", "need", "prog")

    def __init__(self, cid: int, t: Template, ext: bytes):
        live = bytearray(t.n)
        stack = [int(j) for j, e in zip(t.vis, ext) if e]
        need: set[int] = set()
        rows, m = t.rows, t.m
        while stack:
            j = stack.pop()
            if live[j]:
                continue
            live[j] = 1
            a, _, b, _ = rows[j]
            for v in a + b:
                if v >= m:
                    if not live[v - m]:
                        stack.append(v - m)
                else:
                    need.add(v)
        self.id, self.t = cid, t
        self.mask = np.frombuffer(bytes(live), dtype=np.uint8).astype(bool)
        self.count = int(self.mask.sum())
        self.rank = np.where(self.mask, np.cumsum(self.mask) - 1, -1).astype(np.int64)
        self.need = np.array(sorted(need), dtype=np.int64)
        self.prog = None

    def program(self, const: int):
        """The live rows' text as a %-format of their columns, and per column ``(operand rank or -1, offset from the call's
        first column)``."""
        if self.prog is None:
            t, m, rank = self.t, self.t.m, self.rank
            text, ops = [], []
            for j in np.flatnonzero(self.mask):
                a, ac, b, bc = t.rows[j]
                tok = []
                for s, c in ((a, ac), (b, bc)):
                    tok.append(str(len(s) + c))
                    for v in s:
                        tok.append("%d")
                        ops.append(v if v < m else -1 - int(rank[v - m]))
                    if c:
                        tok.append(str(const))
                text.append(" ".join(tok))
            ops = np.array(ops, dtype=np.int64)
            isb = ops >= 0
            self.prog = ("\n".join(text) + "\n" if text else "", isb, ops[isb], -1 - ops[~isb])
        return self.prog


class Stamp:
    """The unit circuit as calls of templates: `forms.Circuit`'s ``input`` and ``output``, and `lower` in place of
    `ir_lower.lower`."""

    def __init__(self):
        self.cse = None
        self.inputs: dict[str, list] = {}
        self.outputs: dict[str, list] = {}
        self.nv = 0
        self.templates: list[Template] = []
        self.keys: dict[tuple, Template] = {}
        self.tids: list[int] = []
        self.bases: list[int] = []
        self.binds: list[np.ndarray] = []
        self.calls = 0

    def input(self, name: str, n: int) -> list:
        bits = [forms.L(frozenset((self.nv + i,)), 0) for i in range(n)]
        self.nv += n
        self.inputs[name] = bits
        return bits

    def output(self, name: str, bits) -> None:
        self.outputs[name] = bits

    # --- the walk --------------------------------------------------------------------------------------------------------
    def call(self, key0, groups, fn, cse: bool) -> list:
        """``fn(C, *groups)``'s output forms, ``groups`` lists of this circuit's forms: its template's, renamed."""
        self.calls += 1
        vs: set[int] = set()
        for g in groups:
            for f in g:
                vs.update(f.s)
        V = sorted(vs)
        rank = dict(zip(V, range(len(V))))
        enc = [key0]
        for g in groups:
            enc.append(len(g))
            for f in g:
                s = f.s
                if len(s) == 1 and not f.c:
                    for v in s:
                        enc.append(rank[v])
                else:
                    enc.append((f.c, *sorted([rank[v] for v in s])))
        key = tuple(enc)
        t = self.keys.get(key)
        if t is None:
            S = forms.Circuit()
            if cse:
                S.cse = {}
            for r in range(len(V)):
                S._new("in", r)
            loc = [[f if not f.s else forms.L(frozenset([rank[v] for v in f.s]), f.c) for f in g] for g in groups]
            label = key0[1] if key0[0] == "piece" else getattr(key0[1], "id", str(key0[1]))
            t = self.keys[key] = Template(len(self.templates), str(label), len(V), S, fn(S, *loc))
            self.templates.append(t)
        m, base = t.m, self.nv
        if t.n:
            self.tids.append(t.id)
            self.bases.append(base)
            self.binds.append(np.array(V, dtype=np.int64))
            self.nv += t.n
        out = []
        for s, c in t.outs:
            if not s:
                out.append(forms.ONE if c else forms.ZERO)
            else:
                out.append(forms.L(frozenset([V[v] if v < m else base + v - m for v in s]), c))
        return out

    def _activation(self, orig):
        def act(C, s, build, wires, pieces):
            if C is not self:
                return orig(C, s, build, wires, pieces)
            params, ret = s.spec.signature()
            leaves = [[param_leaf(s, pi, j) for j in range(t.leaves)] for pi, (_, t) in enumerate(params)]
            if any(o not in wires for ls in leaves for o in ls):
                return False
            els = []
            for (_, t), ls in zip(params, leaves):
                at = 0
                for w in BL.elements(t):
                    els.append([wires[o][0] for o in ls[at:at + w]])
                    at += w
            wrap = getattr(pieces, "wrap", None)
            fn = wrap(s.spec.id, build) if wrap else build
            outs = self.call(("act", s.spec), els, lambda S, *e: _flat_out(fn(S, *e), ret), True)
            lo, hi = s.offset, s.offset + s.spec.gates
            for r, f in zip(s.spec.body.ret, outs):
                j = resolve(s, r)
                if lo <= j < hi:
                    wires.setdefault(j, [f])
            return True
        return act

    @contextmanager
    def _walking(self):
        orig = IL._lower_activation
        IL._lower_activation = self._activation(orig)
        try:
            yield
        finally:
            IL._lower_activation = orig

    def lower(self, target, inputs) -> list[list]:
        """`ir_lower.lower` on this circuit."""
        from verity.ml.circuits.defs import PrimitiveDefinition
        from verity.ml.circuits.evaluation.reference import resolve as resolve_target

        pieces = _Pieces(self)
        t = resolve_target(target)
        with self._walking():
            if isinstance(t, PrimitiveDefinition):
                wires = IL.lower_gates(self, [IL._PrimGate(t, len(inputs))], {-(i + 1): list(w) for i, w in enumerate(inputs)}, pieces)
                return [wires[0]]
            _prog, call = IL.standalone(t)
            wires = dict(zip(call.args, inputs))
            IL.lower_gates(self, call, wires, pieces)
            return [wires[r] for r in call.returns]


class _Pieces:
    """`ir_lower.PIECES` for the walk: a piece called on the stamp is a call of its template."""

    def __init__(self, st: Stamp):
        self.st, self.wrap, self.fns = st, IL.PIECES.wrap, {}

    def get(self, key, default=None):
        fn = self.fns.get(key)
        if fn is None:
            piece = IL.PIECES.get(key)
            if piece is None:
                return default
            st = self.st
            fn = self.fns[key] = lambda C, *ops: st.call(("piece", key), ops, piece, False) if C is st else piece(C, *ops)
        return fn


# --- the layout ----------------------------------------------------------------------------------------------------------

class StampLayout(IL.Layout):
    """`ir_lower.Layout` of a stamped unit: its groups, constant column and counts, and its netlist rendered from the calls
    (`netlist_pieces`); ``rows_a`` / ``rows_b`` are built on first read only (`materialize`)."""

    def __init__(self, st: Stamp, in_names, out_names):
        t0 = time.time()
        self.st = st
        in_bits = [len(st.inputs[nm]) for nm in in_names]
        self.n_in = sum(in_bits)
        if [v for nm in in_names for w in st.inputs[nm] for v in w.s] != list(range(self.n_in)):
            raise Unstampable("the input bits are not the first bits in port order")
        words = IL.ceil_pow2(max(1, -(-self.n_in // W)))
        self.first = words * W
        self.in_groups = [(0, words, in_bits)]
        self._live(in_names, out_names)
        n = self.first + self.n_live
        out_bits = [len(st.outputs[nm]) for nm in out_names]
        self.out_names = list(out_names)
        owords = IL.ceil_pow2(max(1, -(-sum(out_bits) // W)))
        self.out_at = -(-n // (W * owords)) * (W * owords)
        self.out_groups = [(self.out_at // W, owords, out_bits)]
        self.const = self.out_at + owords * W
        self._bind_cols()
        self.out_rows = [self.form(w) for nm in out_names for w in st.outputs[nm]]
        nnz_a = self.n_in + self.nnz[0] + sum(map(len, self.out_rows)) + 1
        nnz_b = self.n_in + self.nnz[1] + len(self.out_rows) + 1
        self.counts = {"ands": self.n_live, "asserts": 0, "rows": self.const + 1, "in_words": self.in_words,
                       "out_words": self.out_words, "nnz_a": nnz_a, "nnz_b": nnz_b, "reads": 0, "read_rows": 0, "read_nnz_a": 0}
        self.seconds = {"live_cols": time.time() - t0}
        self._rows = None

    @property
    def useful(self) -> int:
        return self.const + 1

    def _live(self, in_names, out_names) -> None:
        st = self.st
        live = np.zeros(st.nv, dtype=np.uint8)
        for nm in out_names:
            for w in st.outputs[nm]:
                for v in w.s:
                    live[v] = 1
        T, closures = st.templates, []
        cid = [0] * len(st.tids)
        for i in range(len(st.tids) - 1, -1, -1):
            t = T[st.tids[i]]
            ext = live[st.bases[i] + t.vis].tobytes()
            cl = t.closures.get(ext)
            if cl is None:
                cl = t.closures[ext] = Closure(len(closures), t, ext)
                closures.append(cl)
            cid[i] = cl.id
            if len(cl.need):
                live[st.binds[i][cl.need]] = 1
        del live
        self.closures = closures
        self.cid = np.array(cid, dtype=np.int64)
        counts = np.array([c.count for c in closures], dtype=np.int64)[self.cid] if cid else np.zeros(0, np.int64)
        self.n_live = int(counts.sum())
        self.colbase = self.first + np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64) if cid else counts
        nz = [[0, 0] for _ in closures]
        for c in closures:
            for j in np.flatnonzero(c.mask):
                a, ac, b, bc = c.t.rows[j]
                nz[c.id][0] += len(a) + ac
                nz[c.id][1] += len(b) + bc
        nz = np.array(nz, dtype=np.int64).reshape(-1, 2)
        self.nnz = nz[self.cid].sum(axis=0).tolist() if cid else [0, 0]

    def _bind_cols(self) -> None:
        """Each call's operand bits as columns (``bindcol``, ``off``): an input bit's is its index, an AND's its call's first
        column plus its rank among the call's live ANDs (a dead one's is never read)."""
        st = self.st
        base = np.array(st.bases, dtype=np.int64)
        lens = np.array([len(b) for b in st.binds], dtype=np.int64)
        self.off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        ranks = [c.rank for c in self.closures]
        roff = np.concatenate([[0], np.cumsum([len(r) for r in ranks])]).astype(np.int64)
        R = np.concatenate(ranks) if ranks else np.zeros(0, np.int64)
        self._base, self._R, self._roff = base, R, roff
        out = np.empty(int(self.off[-1]), dtype=np.int64)
        for i0 in range(0, len(st.binds), 1 << 16):
            i1 = min(len(st.binds), i0 + (1 << 16))
            out[self.off[i0]:self.off[i1]] = self._colmap(np.concatenate(st.binds[i0:i1]))
        self.bindcol = out

    def _colmap(self, B: np.ndarray) -> np.ndarray:
        base, R, roff = self._base, self._R, self._roff
        col = B.copy()
        a = B >= self.n_in
        if a.any():
            Ba = B[a]
            own = np.searchsorted(base, Ba, side="right") - 1
            col[a] = self.colbase[own] + R[roff[self.cid[own]] + (Ba - base[own])]
        return col

    def form(self, f) -> list[int]:
        """An output form's copy-row A columns (`ir_lower.layout`'s ``form``, sorted, the constant last)."""
        r = sorted(self._colmap(np.array(sorted(f.s), dtype=np.int64)).tolist()) if f.s else []
        return r + [self.const] if f.c & 1 else r

    # --- the netlist ---------------------------------------------------------------------------------------------------------
    def head(self, name: str) -> str:
        head = [IL.FORMAT, name, self.useful, self.const]
        for groups in (self.in_groups, self.out_groups):
            head.append(len(groups))
            for c, words, bits in groups:
                head += [c, words, len(bits), *bits]
        return " ".join(map(str, head)) + "\n"

    def _calls_text(self, lo: int, hi: int) -> bytes:
        st, cl, cid, colbase, bindcol, off = self.st, self.closures, self.cid, self.colbase, self.bindcol, self.off
        out = []
        for i in range(lo, hi):
            fmt, isb, bidx, loff = cl[cid[i]].program(self.const)
            if not fmt:
                continue
            vals = np.empty(len(isb), dtype=np.int64)
            vals[isb] = bindcol[off[i] + bidx]
            vals[~isb] = colbase[i] + loff
            out.append(fmt % tuple(vals.tolist()))
        return "".join(out).encode()

    def _tasks(self) -> list[tuple[int, int]]:
        """The recorded calls in runs of about `TASK_ANDS` live ANDs."""
        n = len(self.st.tids)
        if not n:
            return []
        cum = np.cumsum(np.array([c.count for c in self.closures], dtype=np.int64)[self.cid])
        cuts = np.searchsorted(cum, np.arange(TASK_ANDS, int(cum[-1]), TASK_ANDS), side="left") + 1
        bounds = np.unique(np.concatenate([[0], cuts, [n]]))
        return list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))

    def _tail_text(self) -> str:
        rows = []
        rows += ["0 0"] * (self.out_at - self.first - self.n_live)
        for r in self.out_rows:
            rows.append(" ".join(map(str, [len(r), *r, 1, self.const])))
        rows += ["0 0"] * (self.const - self.out_at - len(self.out_rows))
        rows.append(f"1 {self.const} 1 {self.const}")
        return "\n".join(rows) + "\n"

    def netlist_pieces(self, name: str):
        """`ir_lower.netlist(self, name)`'s bytes, in order, the calls' rows rendered by worker processes."""
        t0 = time.time()
        yield self.head(name).encode()
        rows = [f"1 {i} 1 {i}" for i in range(self.n_in)] + ["0 0"] * (self.first - self.n_in)
        for k in range(0, len(rows), 1 << 16):
            yield ("\n".join(rows[k:k + (1 << 16)]) + "\n").encode()
        tasks = self._tasks()
        procs = _procs()
        if procs <= 1 or len(tasks) <= 1:
            for lo, hi in tasks:
                yield self._calls_text(lo, hi)
        else:
            import multiprocessing as mp
            from collections import deque

            _WORKER[:] = [self]
            try:
                with mp.get_context("fork").Pool(procs) as pool:
                    todo, ahead = iter(tasks), deque()
                    for task in todo:
                        ahead.append(pool.apply_async(_render, (task,)))
                        if len(ahead) >= AHEAD * procs:
                            yield ahead.popleft().get()
                    while ahead:
                        yield ahead.popleft().get()
            finally:
                _WORKER.clear()
        yield self._tail_text().encode()
        print(f"STAMP-NETLIST {name}: procs={procs} tasks={len(tasks)} seconds={time.time() - t0:.1f} {_mem()}", file=sys.stderr,
              flush=True)

    # --- the rows, built only on request ----------------------------------------------------------------------------------
    def materialize(self) -> tuple[list, list]:
        if self._rows is None:
            ra: list = [[i] for i in range(self.n_in)] + [[] for _ in range(self.first - self.n_in)]
            rb: list = [list(r) for r in ra]
            for i in range(len(self.st.tids)):
                c = self.closures[self.cid[i]]
                fmt, isb, bidx, loff = c.program(self.const)
                vals = np.empty(len(isb), dtype=np.int64)
                vals[isb] = self.bindcol[self.off[i] + bidx]
                vals[~isb] = self.colbase[i] + loff
                for line in (fmt % tuple(vals.tolist())).splitlines():
                    x = list(map(int, line.split()))
                    na = x[0]
                    ra.append(x[1:1 + na])
                    rb.append(x[2 + na:])
            ra += [[] for _ in range(self.out_at - len(ra))]
            rb += [[] for _ in range(self.out_at - len(rb))]
            for r in self.out_rows:
                ra.append(list(r))
                rb.append([self.const])
            ra += [[] for _ in range(self.const - len(ra))]
            rb += [[] for _ in range(self.const - len(rb))]
            ra.append([self.const])
            rb.append([self.const])
            self._rows = (ra, rb)
        return self._rows

    @property
    def rows_a(self):
        return self.materialize()[0]

    @property
    def rows_b(self):
        return self.materialize()[1]

    # --- one lane's values -------------------------------------------------------------------------------------------------
    def evaluate1(self, in_bits: np.ndarray) -> list[list[int]]:
        """Every output port's bits on one lane (``in_bits`` the input bits, 0/1), call by call, each template's values for
        an operand pattern computed once."""
        st, T = self.st, self.st.templates
        val = np.zeros(st.nv, dtype=np.uint8)
        val[:self.n_in] = in_bits
        memo: dict[tuple[int, bytes], np.ndarray] = {}
        for i in range(len(st.tids)):
            t = T[st.tids[i]]
            x = val[st.binds[i]]
            k = (t.id, x.tobytes())
            y = memo.get(k)
            if y is None:
                y = memo[k] = _eval_template(t, x)
            val[st.bases[i] + t.vis] = y
        out = []
        for nm in self.out_names:
            bits = []
            for f in st.outputs[nm]:
                b = f.c
                for v in f.s:
                    b ^= int(val[v])
                bits.append(b)
            out.append(bits)
        return out


def _eval_template(t: Template, x: np.ndarray) -> np.ndarray:
    lv = x.tolist() + [0] * t.n
    m = t.m
    for j, (a, ac, b, bc) in enumerate(t.rows):
        p = ac
        for v in a:
            p ^= lv[v]
        q = bc
        for v in b:
            q ^= lv[v]
        lv[m + j] = p & q
    return np.array([lv[m + int(j)] for j in t.vis], dtype=np.uint8)


_WORKER: list = []


def _render(task: tuple[int, int]) -> bytes:
    return _WORKER[0]._calls_text(*task)


def _procs() -> int:
    n = os.environ.get("STAMP_PROCS")
    return int(n) if n else len(os.sched_getaffinity(0))


class StampLowering(IL.Lowering):
    """`ir_lower.Lowering` of a stamped unit: one lane is evaluated call by call (`StampLayout.evaluate1`; `circuit.compose`'s
    all-zero instance), more lanes on the materialized rows."""

    def evaluate(self, flat_inputs, cuts=None, flip=None):
        flat_inputs = np.asarray(flat_inputs, dtype=np.uint64)
        if flat_inputs.shape[0] != 1 or flip is not None or self.units.cut:
            return super().evaluate(flat_inputs, cuts, flip)
        t0, lay = time.time(), self.layout
        ports = self.unit_inputs(flat_inputs, cuts)
        bits = np.concatenate([(np.asarray(p, dtype=np.uint64)[0] >> np.arange(b, dtype=np.uint64)) & np.uint64(1)
                               for p, b in zip(ports, lay.in_bits)]).astype(np.uint8)
        outs = lay.evaluate1(bits)
        print(f"STAMP-EVAL {self.name}: one lane, seconds={time.time() - t0:.1f} {_mem()}", file=sys.stderr, flush=True)
        words = [sum(b << i for i, b in enumerate(o)) for o in outs]
        res = np.zeros((1, len(words)), dtype=np.uint64)
        for j, w in enumerate(words):
            k, p = self.units.out_src[0][j]
            res[0, p] = w
        return res, np.zeros((1, 0), dtype=np.uint64), 1


def applies(definition, name, in_rows, public=(), structural=False, reads="inline", stage_ands="auto") -> bool:
    return not structural and reads == "inline"


def unit_lowering(definition, name: str, in_rows, public=(), structural: bool = False, reads: str = "inline",
                  stage_ands="auto") -> tuple[IL.Lowering, list[int]]:
    """`class_statement.unit_lowering` of a flat unit with inline reads, stamped; `Unstampable` where it can't be."""
    t0, start_mem = time.time(), _mem()
    ports, out_words, row_words = CS.unit_ports(definition, name, in_rows, public)
    n_words, n_out = len(ports.ins), len(ports.outs)
    units = IL.Units(target=None, in_widths=[16] * n_words, out_widths=[16] * n_out, in_src=[[("leaf", p) for p in range(n_words)]],
                     out_src=[[("ret", o) for o in range(n_out)]], signature=(definition.id,), gates=[], arg_gates=[], ret_gates=[])
    sig = (tuple((r, Array(w, Value(16))) for r, w in zip([*public, *(r for r, _ps in in_rows)], row_words)),
           Tuple(tuple(Array(w, Value(16)) for w in out_words)))
    unit = SimpleNamespace(id=name, signature=lambda: sig)
    st = Stamp()
    in_bits = [b for j, w in enumerate(ports.ins) for b in st.input(f"in{j}", w)]
    outs = [b for w in st.lower(definition, [[in_bits[i] for i in lf] for lf in ports.leaves]) for b in w]
    bits = [forms.ZERO if i is None else outs[i] for i in ports.bits]
    at = 0
    for j, w in enumerate(ports.outs):
        st.output(f"out{j}", bits[at:at + w])
        at += w
    walk, walk_mem = time.time() - t0, _mem()
    lay = StampLayout(st, [f"in{j}" for j in range(n_words)], [f"out{o}" for o in range(n_out)])
    lay.seconds["walk"] = walk
    print(f"STAMP {name}: calls={st.calls} recorded={len(st.tids)} templates={len(st.templates)} closures={len(lay.closures)} "
          f"bits={st.nv} live_ands={lay.n_live} rows={lay.useful} walk_s={walk:.1f} live_cols_s={lay.seconds['live_cols']:.1f} "
          f"at start {start_mem}, after walk {walk_mem}, after layout {_mem()}", file=sys.stderr, flush=True)
    print(f"STAMP-TOP {name}: {_top(st, lay)}", file=sys.stderr, flush=True)
    return StampLowering(name, unit, units, None, lay), out_words


def _mem() -> str:
    import resource
    with open("/proc/self/statm") as f:
        rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return f"rss_gib={rss / 2 ** 30:.2f} peak_gib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.2f}"


def _top(st: Stamp, lay: StampLayout, n: int = 12) -> str:
    """The builders whose calls make the most live ANDs: each one's templates, recorded calls and live ANDs."""
    if not st.tids:
        return ""
    counts = np.array([c.count for c in lay.closures], dtype=np.int64)[lay.cid]
    tids = np.asarray(st.tids, dtype=np.int64)
    by_t = np.bincount(tids, weights=counts, minlength=len(st.templates))
    calls_t = np.bincount(tids, minlength=len(st.templates))
    agg: dict[str, list[int]] = {}
    for t in st.templates:
        a = agg.setdefault(t.label, [0, 0, 0])
        a[0] += 1
        a[1] += int(calls_t[t.id])
        a[2] += int(by_t[t.id])
    rank = sorted(agg.items(), key=lambda kv: -kv[1][2])[:n]
    return "; ".join(f"{lb} templates={a[0]} calls={a[1]} live={a[2]}" for lb, a in rank)

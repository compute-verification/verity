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

import gc
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
TASK_ANDS = 1 << 14
#: rendered tasks held ahead of the writer, per thread
AHEAD = 2
#: tokens converted to text at once: about 70 bytes each while converting
ASCII_TOKENS = 1 << 20
#: walk calls between collections of the young generation (every 16th also the next): `fast_stage` turns automatic
#: collection off, and the IR's traced forms are reference cycles (a `trace._Set` and its circuit's table)
GC_CALLS = 1 << 8
#: a template of more than one call keeps its rows as Python lists (for its evaluations) up to this many columns
KEEP_COLS = 1 << 20


class Unstampable(Exception):
    """A unit the stamp does not lay out: a template made a table read, a hint or an assertion."""


class Template:
    """One (builder, sharing pattern)'s circuit: ``m`` operand bits, then ``n`` ANDs over local bits (below ``m`` an operand's
    rank, else AND ``j`` at ``m + j``), AND ``j``'s A bits ``cols[ptr[2 j]:ptr[2 j + 1]]`` and B bits
    ``cols[ptr[2 j + 1]:ptr[2 j + 2]]``, each sorted, and their constants ``cst[2 j]``, ``cst[2 j + 1]``; its output forms, and
    ``vis``, the ANDs its outputs read: all a later call can read of it. ``calls`` is set once the walk ends."""

    __slots__ = ("id", "label", "m", "n", "cols", "ptr", "cst", "outs", "vis", "closures", "build_s", "calls", "_lists")

    def __init__(self, tid: int, label: str, m: int, S: forms.Circuit, outs: list):
        if S.lookups or S.asserts or any(k != "and" for k in S.kind[m:]):
            raise Unstampable(f"{label}'s template makes {sorted(set(S.kind[m:]) - {'and'}) or 'assertions or reads'}")
        self.id, self.label, self.m, self.n = tid, label, m, len(S.kind) - m
        flat: list[int] = []
        lens, cst = [], []
        for a, b in S.data[m:]:
            for f in (a, b):
                s = sorted(f.s)
                flat += s
                lens.append(len(s))
                cst.append(f.c & 1)
        self.cols = np.array(flat, dtype=np.int32)
        self.ptr = np.concatenate([[0], np.cumsum(np.array(lens, dtype=np.int64))]).astype(np.int64)
        self.cst = np.array(cst, dtype=np.uint8)
        self.outs = [(tuple(sorted(f.s)), f.c) for f in outs]
        self.vis = np.array(sorted({v - m for s, _ in self.outs for v in s if v >= m}), dtype=np.int64)
        self.closures: dict[bytes, Closure] = {}
        self.build_s, self.calls, self._lists = 0.0, 0, None

    def lists(self) -> tuple[list, list, list]:
        """``cols``, ``ptr`` and ``cst`` as lists, kept for a template of more than one call whose columns are few."""
        if self._lists is not None:
            return self._lists
        ls = (self.cols.tolist(), self.ptr.tolist(), self.cst.tolist())
        if self.calls > 1 and len(self.cols) <= KEEP_COLS:
            self._lists = ls
        return ls


class Closure:
    """A template's live ANDs when later rows read ``ext`` of its ``vis``: ``mask`` and ``rank`` (each live AND's place among
    them, -1 dead), and ``need``, the operand ranks the live rows read. ``uses`` is set once liveness ends."""

    __slots__ = ("id", "t", "mask", "rank", "count", "need", "prog", "uses")

    def __init__(self, cid: int, t: Template, ext: bytes):
        cols, ptr, _ = t.lists()
        live = bytearray(t.n)
        stack = [int(j) for j, e in zip(t.vis, ext) if e]
        need: set[int] = set()
        m = t.m
        while stack:
            j = stack.pop()
            if live[j]:
                continue
            live[j] = 1
            for v in cols[ptr[2 * j]:ptr[2 * j + 2]]:
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
        self.prog, self.uses = None, 0

    def program(self, const: int):
        """The live rows as tokens, each row ``na A.. nb B..`` (a count, then its columns, the constant column ``const`` last
        where the side has a constant): ``fixed`` (the counts and constants; a column's token 0), ``sep`` (the byte after each
        token, the row's last a newline), and the columns' places in ``fixed``, those of an operand bit with its rank and those
        of the call's own ANDs with their offset from its first column. Kept for a closure of more than one call."""
        if self.prog is not None:
            return self.prog
        t, m = self.t, self.t.m
        J = np.flatnonzero(self.mask)
        s0, s1, s2 = t.ptr[2 * J], t.ptr[2 * J + 1], t.ptr[2 * J + 2]
        a, b = s1 - s0, s2 - s1
        x, y = t.cst[2 * J].astype(np.int64), t.cst[2 * J + 1].astype(np.int64)
        per = 2 + a + x + b + y
        R = np.cumsum(per) - per
        fixed = np.zeros(int(per.sum()), dtype=np.int64)
        fixed[R] = a + x
        fixed[R + 1 + a + x] = b + y
        fixed[(R + 1 + a)[x == 1]] = const
        fixed[(R + 2 + a + x + b)[y == 1]] = const
        sep = np.full(len(fixed), 32, dtype=np.uint8)
        sep[R + per - 1] = 10
        tot = a + b
        e = np.arange(int(tot.sum()), dtype=np.int64) - np.repeat(np.cumsum(tot) - tot, tot)
        v = t.cols[np.repeat(s0, tot) + e].astype(np.int64)
        ops = np.where(v < m, v, -1 - self.rank[np.where(v >= m, v - m, 0)])
        at = np.repeat(R + 1, tot) + e + np.where(e >= np.repeat(a, tot), 1 + np.repeat(x, tot), 0)
        isb = ops >= 0
        prog = (fixed, sep, at[isb], ops[isb], at[~isb], -1 - ops[~isb])
        if self.uses > 1:
            self.prog = prog
        return prog


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
        self.counting = False
        self.count_s, self.kept = 0.0, None

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
        if self.counting:
            raise _Builds(getattr(key0[1], "id", key0[1]))
        self.calls += 1
        if not self.calls % GC_CALLS:
            if self.calls % (GC_CALLS << 4):
                gc.collect(0)
            else:
                gc.collect(1)
                _trim()
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
            tb = time.time()
            S = forms.Circuit()
            if cse:
                S.cse = {}
            for r in range(len(V)):
                S._new("in", r)
            loc = [[f if not f.s else forms.L(frozenset([rank[v] for v in f.s]), f.c) for f in g] for g in groups]
            label = key0[1] if key0[0] == "piece" else getattr(key0[1], "id", str(key0[1]))
            t = self.keys[key] = Template(len(self.templates), str(label), len(V), S, fn(S, *loc))
            t.build_s = time.time() - tb
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
            if self.counting:
                for ls in leaves:
                    for o in ls:
                        wires[o]
                lo, hi = s.offset, s.offset + s.spec.gates
                for r in s.spec.body.ret:
                    j = resolve(s, r)
                    if lo <= j < hi:
                        wires.setdefault(j, _BIT)
                return True
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
            left = self._reads(call, inputs, pieces)
            wires = dict(zip(call.args, inputs)) if left is None else _Wires(zip(call.args, inputs), left)
            IL.lower_gates(self, call, wires, pieces)
            outs = [wires[r] for r in call.returns]
            if left is not None:
                self.kept = wires.most
            return outs

    def _reads(self, call, inputs, pieces) -> dict[int, int] | None:
        """How many times the walk reads each gate's word, from a walk on constant words (no piece or builder runs), or None
        where that walk fails: the walk then keeps every word, as `ir_lower.lower` does. The walk's reads don't depend on its
        words' values, so a count that is off can only make the real walk raise."""
        t0 = time.time()
        reads = _Reads(zip(call.args, [[forms.ZERO] * len(w) for w in inputs]))
        self.counting = True
        try:
            IL.lower_gates(self, call, reads, pieces)
            for r in call.returns:
                reads[r]
        except Exception as e:  # noqa: BLE001
            print(f"STAMP: the walk keeps every word: its counting walk raised {type(e).__name__}: {e}", file=sys.stderr,
                  flush=True)
            return None
        finally:
            self.counting = False
            self.count_s = time.time() - t0
        return reads.n


_BIT = [forms.ZERO]


class _Builds(Exception):
    """The counting walk reached a piece or a builder."""


class _Reads(dict):
    """The counting walk's words, counting each gate's reads (``n``)."""

    def __init__(self, items):
        super().__init__(items)
        self.n: dict[int, int] = {}

    def __getitem__(self, g):
        w = dict.__getitem__(self, g)
        self.n[g] = self.n.get(g, 0) + 1
        return w


class _Wires(dict):
    """The walk's words, each dropped at its last read (``left``: the reads still to come); a word never read is not kept.
    ``most``: the most words held at once."""

    def __init__(self, items, left: dict[int, int]):
        super().__init__(items)
        self.left, self.most = left, len(self)

    def __setitem__(self, g, w):
        if g in self.left:
            dict.__setitem__(self, g, w)
            if len(self) > self.most:
                self.most = len(self)

    def setdefault(self, g, w):
        if g not in self.left:
            return w
        if g not in self:
            self[g] = w
        return dict.__getitem__(self, g)

    def __getitem__(self, g):
        w = dict.__getitem__(self, g)
        k = self.left[g] - 1
        if k:
            self.left[g] = k
        else:
            del self.left[g]
            dict.__delitem__(self, g)
        return w


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
        for t, k in zip(T, np.bincount(np.asarray(st.tids, dtype=np.int64), minlength=len(T)).tolist()):
            t.calls = k
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
        for c, k in zip(closures, np.bincount(self.cid, minlength=len(closures)).tolist()):
            c.uses = k
        counts = np.array([c.count for c in closures], dtype=np.int64)[self.cid] if cid else np.zeros(0, np.int64)
        self.n_live = int(counts.sum())
        self.colbase = self.first + np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64) if cid else counts
        nz = np.zeros((len(closures), 2), dtype=np.int64)
        for c in closures:
            per = np.diff(c.t.ptr) + c.t.cst
            nz[c.id] = per[0::2][c.mask].sum(), per[1::2][c.mask].sum()
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

    def _tokens(self, i: int):
        """Call ``i``'s rows as tokens and separators (`Closure.program`), or None for a call with no live AND."""
        fixed, sep, pb, bidx, pl, loff = self.closures[self.cid[i]].program(self.const)
        if not len(fixed):
            return None
        tok = fixed.copy()
        tok[pb] = self.bindcol[self.off[i] + bidx]
        tok[pl] = self.colbase[i] + loff
        return tok, sep

    def _calls_text(self, lo: int, hi: int) -> bytes:
        toks, seps = [], []
        for i in range(lo, hi):
            r = self._tokens(i)
            if r is not None:
                toks.append(r[0])
                seps.append(r[1])
        return _ascii(np.concatenate(toks), np.concatenate(seps)) if toks else b""

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
        """`ir_lower.netlist(self, name)`'s bytes, in order, the calls' rows rendered by threads (numpy's loops release the
        GIL; worker processes would copy the walk's heap as they allocate in it)."""
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
            from collections import deque
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(procs) as pool:
                ahead: deque = deque()
                for task in tasks:
                    ahead.append(pool.submit(self._calls_text, *task))
                    if len(ahead) >= AHEAD * procs:
                        yield ahead.popleft().result()
                while ahead:
                    yield ahead.popleft().result()
        _trim()
        yield self._tail_text().encode()
        print(f"STAMP-NETLIST {name}: threads={procs} tasks={len(tasks)} seconds={time.time() - t0:.1f} {_mem()}", file=sys.stderr,
              flush=True)

    # --- the rows, built only on request ----------------------------------------------------------------------------------
    def materialize(self) -> tuple[list, list]:
        if self._rows is None:
            ra: list = [[i] for i in range(self.n_in)] + [[] for _ in range(self.first - self.n_in)]
            rb: list = [list(r) for r in ra]
            for i in range(len(self.st.tids)):
                r = self._tokens(i)
                if r is None:
                    continue
                x, s = r[0].tolist(), 0
                for e in np.flatnonzero(r[1] == 10).tolist():
                    na = x[s]
                    ra.append(x[s + 1:s + 1 + na])
                    rb.append(x[s + 2 + na:e + 1])
                    s = e + 1
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
    cols, ptr, cst = t.lists()
    lv = x.tolist() + [0] * t.n
    m = t.m
    for j in range(t.n):
        p = cst[2 * j]
        for v in cols[ptr[2 * j]:ptr[2 * j + 1]]:
            p ^= lv[v]
        q = cst[2 * j + 1]
        for v in cols[ptr[2 * j + 1]:ptr[2 * j + 2]]:
            q ^= lv[v]
        lv[m + j] = p & q
    return np.array([lv[m + int(j)] for j in t.vis], dtype=np.uint8)


_D4 = np.frombuffer("".join(f"{i:04d}" for i in range(10 ** 4)).encode(), dtype=np.uint32)
_P10 = 10 ** np.arange(1, 12, dtype=np.int64)
#: by a token's digits less one: which of its 16 bytes (12 digits, its separator, 3 unused) the text keeps
_KEEP = np.array([[11 - q <= j <= 12 for j in range(16)] for q in range(12)])


def _ascii(tok: np.ndarray, sep: np.ndarray) -> bytes:
    """Each of ``tok`` (in [0, 10^12)) in decimal, followed by its ``sep`` byte."""
    if len(tok) and int(tok.max()) >= 10 ** 12:
        raise ValueError("a column of 13 digits")
    out = []
    for i in range(0, len(tok), ASCII_TOKENS):
        t = tok[i:i + ASCII_TOKENS]
        D = np.empty((len(t), 4), dtype=np.uint32)
        D[:, 0] = _D4[t // 10 ** 8]
        D[:, 1] = _D4[t // 10 ** 4 % 10 ** 4]
        D[:, 2] = _D4[t % 10 ** 4]
        B = D.view(np.uint8).reshape(len(t), 16)
        B[:, 12] = sep[i:i + ASCII_TOKENS]
        out.append(B[_KEEP[np.searchsorted(_P10, t, side="right")]].tobytes())
    return b"".join(out)


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
    gc.collect()
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
    _trim()
    walk, walk_mem = time.time() - t0, _mem()
    lay = StampLayout(st, [f"in{j}" for j in range(n_words)], [f"out{o}" for o in range(n_out)])
    lay.seconds["walk"] = walk
    _trim()
    print(f"STAMP {name}: calls={st.calls} recorded={len(st.tids)} templates={len(st.templates)} closures={len(lay.closures)} "
          f"bits={st.nv} live_ands={lay.n_live} rows={lay.useful} walk_s={walk:.1f} count_s={st.count_s:.1f} words_kept={st.kept} "
          f"live_cols_s={lay.seconds['live_cols']:.1f} "
          f"at start {start_mem}, after walk {walk_mem}, after layout {_mem()}", file=sys.stderr, flush=True)
    print(f"STAMP-TOP {name}: {_top(st, lay)}", file=sys.stderr, flush=True)
    return StampLowering(name, unit, units, None, lay), out_words


def _trim() -> None:
    """Return the C heap's free pages: the walk frees most of what it allocates, between blocks it keeps, so glibc's heap
    stays at the walk's high-water mark, several times what is in use, unless trimmed."""
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def _libc():
    import ctypes

    try:
        libc = ctypes.CDLL(None)
        return libc if hasattr(libc, "malloc_trim") else None
    except OSError:
        return None


_LIBC = _libc()


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
        a = agg.setdefault(t.label, [0, 0, 0, 0, 0.0])
        a[0] += 1
        a[1] += int(calls_t[t.id])
        a[2] += int(by_t[t.id])
        a[3] += t.n
        a[4] += t.build_s
    rank = sorted(agg.items(), key=lambda kv: -kv[1][2])[:n]
    built = sum(t.n for t in st.templates)
    return (f"built={built} build_s={sum(t.build_s for t in st.templates):.1f}; "
            + "; ".join(f"{lb} templates={a[0]} built={a[3]} build_s={a[4]:.1f} calls={a[1]} live={a[2]}" for lb, a in rank))

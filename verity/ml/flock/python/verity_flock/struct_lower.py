"""A structural lowering: each specialized Definition lowered once per pattern of its arguments, its calls,
batch members and scan iterations kept as calls, written as ``flock-ir-tmpl/v2`` (:mod:`verity_flock.tmpl`) instead of
:func:`ir_lower.lower`'s one walk over the unrolled gates. An opt-in mode: only a structural unit stager
(`class_statement.UnitStager`'s ``structural``, on `unit_lowering`'s ports, :class:`Ports`) and circuit-check's
``c-flock:struct_lower`` realization call it, and no existing circuit's lowering or digest changes.

A Definition's argument bit is a constant or a linear form over the caller's committed bits, as the unrolled lowering
carries it. The callee is lowered once per *pattern* of its arguments: which argument positions name one gate (a caller
may pass a gate twice, and the callee's body reads that), each constant bit and its value, and each form as
the set of distinct caller bits (the callee's inputs, numbered in the caller's column order, so a call binds a few runs
of consecutive references) it sums plus its constant, so the
constant folding and the free XORs the unrolled lowering does across a call boundary are kept. A form over more than
``support`` bits is committed by the callee (one row, ``A = form, B = 1``); a returned bit is a constant, or a form over
the callee's inputs and its own columns, committed by the callee past ``support`` too, and (``gather``) whenever it
reads two bits one of which is its own column; a returned column nothing else in the callee reads is laid out last, in
return order, so a caller binds a copy's returns as one run. Primitive gates are lowered by
:data:`ir_lower.PIECES` and a Boolean Definition by its builder, as :func:`ir_lower.lower_gates` lowers them; the members
of a batched primitive node are calls too. So a Definition's template rows are its own gates, never its callees', and
consecutive copies of one pattern whose inputs step by a fixed amount per copy are one ``CALL`` line, its runs of
segments a fixed stride apart one two-level segment each (:func:`tmpl.pack`).

Folding constants in costs a specialization per distinct constant argument (a counter passed as a constant makes one per
call), so a Definition of at least ``fold_gates`` gates folds only its first ``fold_cap`` patterns with constants; its
later calls share one specialization, lowered on the Definition's standalone call, in which every argument leaf is an
input (a caller names equal constant bits by one gate, so the gates an argument repeats would spell its value) and each
constant bit is bound to the unit's constant or to the caller's own column 0, which the caller then leaves unwritten.

What the unit loses against the unrolled lowering is counted, not hidden: the commits, and the gates of a callee's
returns that no caller reads (a callee is lowered without its callers' liveness). :func:`unrolled_rows` gives the
template's rows unrolled (``tmpl.unroll``) and :func:`evaluate` evaluates them, which :func:`check` holds to the unrolled
lowering's evaluation on random inputs.

    python -m verity_flock.struct_lower K [R C] [--support S] [--fold-cap N] [--first-appearance] [--scattered-returns]
    [--out UNIT.tmpl] [--flat | --check]

the H1 tile unit (``benchmarks/pouw/h1_prove/h1unit.Shape``): prints ``STRUCT {json}`` (the unrolled totals, multiplied
through the calls, and the template's size) and writes the template, with each Definition's name in
``UNIT.tmpl.names.json``; ``--flat`` counts the unrolled lowering instead, and
``--check`` runs :func:`check`.
"""
from __future__ import annotations

import json
import sys
import time
from array import array
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from verity.ml.circuits.boolean.forms import ONE, Circuit, L
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.layout import FunctionCall, Region

from . import ir_lower as IL
from . import tmpl as T
from .fp import ceil_pow2

SUPPORT = 8
#: a Definition of at least FOLD_GATES gates gets FOLD_CAP specializations with its constant arguments folded in; its
#: later calls share one in which each constant argument bit is an input
FOLD_CAP, FOLD_GATES = 4, 1000
ZERO, ONE_REF = -1, -2  # a caller "bit" an input is bound to: the caller's zero column, or the unit's constant
#: a call variable's data is one int, its copy's index above COL_BITS and its column in that copy below (not a tuple of two
#: ints, which cost a 32 × 32 tile's 15.8M call variables a gigabyte); a copy's labels are 32-bit
COL_BITS = 32


@dataclass
class Spec:
    """One lowered specialization: its inputs, own columns, rows ``(col, A, B)`` and calls (``tmpl.Call``) in template
    references; per returned bit ``("c", v)`` or ``("f", c, terms)`` with each term ``("l", input)`` or ``("v", column)``;
    and its placed copies' keys, for the unrolled totals."""

    key: tuple
    ins: int = 0
    cols: int = 0
    rows: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    rets: list = field(default_factory=list)
    commits: int = 0
    refs: int = 0
    copies: list = field(default_factory=list)
    index: int = -1


@dataclass(slots=True)
class _Block:
    pos: int
    key: tuple
    labels: array
    base: int = 0


def _live(C: Circuit, seeds) -> set:
    """The ANDs of ``C.live`` from the variables ``seeds`` (a specialization's read labels and returns) rather than
    outputs: a specialization lays out only its ANDs, so no other variable is held."""
    seen: set = set()
    stack = list(seeds)
    kind, data = C.kind, C.data
    while stack:
        v = stack.pop()
        if v in seen or kind[v] != "and":
            continue
        seen.add(v)
        a, b = data[v]
        stack.extend(a.s)
        stack.extend(b.s)
    return seen


def _last_reads(circuit, sc, body) -> dict[tuple[int, int], list[int]]:
    """Per body node and member (a batch's or scan's activation), the gates whose last reader it is: the lowering drops
    their wires after it, so a Definition whose body is a long loop, unrolled or a scan, holds the words its later members
    still read, not every word it computed."""
    last: dict[int, tuple[int, int]] = {}
    for k, node in enumerate(body.nodes):
        if isinstance(node.fn, PrimitiveDefinition) and node.form == "primitive":
            lo = sc.offset + body.offsets[k]
            for g in range(lo, lo + node.gates):
                for o in circuit.gate(g).operands:
                    last[o] = (k, 0)
            continue
        for m in range(1 if node.form == "call" else node.n):
            for a in FunctionCall(circuit, sc, k, m).args:
                last[a] = (k, m)
    out: dict[tuple[int, int], list[int]] = {}
    for g, km in last.items():
        out.setdefault(km, []).append(g)
    return out


def _column_order(C: Circuit, v: int) -> tuple:
    """Where caller bit ``v`` lands among the caller's references (``_Lowerer._finish``'s layout): its own columns in
    creation order, a copy's columns where the copy was placed, then its inputs."""
    k = C.kind[v]
    if k == "arg":
        return (1, C.data[v], 0, 0)
    if k == "call":
        d = C.data[v]
        return (0, C.blocks[d >> COL_BITS].pos, 0, d & ((1 << COL_BITS) - 1))
    return (0, v, 1, 0)


def _pattern(words: list[list[L]], fold: bool = True, order=None) -> tuple[tuple, list[int]]:
    """The argument bits' pattern and the caller bits its inputs stand for: the inputs numbered in the caller's column
    ``order`` (so a call binds few runs of consecutive references), else by first appearance; not ``fold``: each
    constant bit an input of its own, after them, bound to ``ZERO`` or ``ONE_REF``."""
    vs = list(dict.fromkeys(v for w in words for b in w for v in sorted(b.s)))
    if order is not None:
        vs.sort(key=order)
    labels = {v: i for i, v in enumerate(vs)}
    out: list[int] = list(vs)
    pat = []
    for w in words:
        for b in w:
            if b.is_const():
                if fold:
                    pat.append(("c", b.c))
                else:
                    pat.append(((len(out),), 0))
                    out.append(ONE_REF if b.c else ZERO)
                continue
            pat.append((tuple(sorted(labels[v] for v in b.s)), b.c))
    return tuple(pat), out


class _Lowerer:
    def __init__(self, circuit, support: int = SUPPORT, fold_cap: int = FOLD_CAP, fold_gates: int = FOLD_GATES,
                 column_order: bool = True, gather: bool = True):
        self.circuit = circuit
        self.in_column_order = column_order
        self.gather = gather
        self.support = support
        self.fold_cap, self.fold_gates = fold_cap, fold_gates
        self.folded: dict[str, int] = {}
        self.specs: dict[tuple, Spec] = {}
        self.order: list[Spec] = []
        self.totals: dict[tuple, dict] = {}

    def call(self, C: Circuit, fc: FunctionCall, wires: dict) -> tuple:
        """Lower call ``fc`` of the Definition being lowered on C: its specialization once per pattern, one placed copy,
        its returns' words set."""
        first: dict[int, int] = {}
        shape = tuple(first.setdefault(a, len(first)) for a in fc.args)
        words = [wires[a] for a in first]
        order = partial(_column_order, C) if self.in_column_order else None
        pat, label_vars = _pattern(words, order=order)
        key = (fc.fn.id, shape, pat)
        sp = self.specs.get(key)
        if sp is None and not fc.is_primitive and fc.fn.gates >= self.fold_gates and any(p[0] == "c" for p in pat):
            if self.folded.get(fc.fn.id, 0) < self.fold_cap:
                self.folded[fc.fn.id] = self.folded.get(fc.fn.id, 0) + 1
            else:
                pat, label_vars = _pattern([wires[a] for a in fc.args], fold=False, order=order)
                key = (fc.fn.id, None, pat)
                sp = self.specs.get(key) or self._generic(fc, key, pat, len(label_vars))
        sp = sp or self._lower(fc, key, pat, len(label_vars))
        key = sp.key
        if sp.cols >> COL_BITS or len(C.kind) >> 31:
            raise NotImplementedError(f"{fc.id}: a specialization or a body of 2^{COL_BITS} columns or 2^31 variables")
        blk = _Block(len(C.kind), key, array("i", label_vars))
        C.blocks.append(blk)
        call_at = (len(C.blocks) - 1) << COL_BITS
        col_var: dict[int, L] = {}
        bits = iter(sp.rets)
        for r in fc.returns:
            if r in wires and not (fc.lo <= r < fc.hi):
                for _ in wires[r]:
                    next(bits)
                continue
            out = []
            for _ in range(IL._width(self.circuit.gate(r).prim)):
                t = next(bits)
                if t[0] == "c":
                    out.append(L(frozenset(), t[1]))
                    continue
                s: frozenset = frozenset()
                c = t[1]
                for kind, x in t[2]:
                    if kind == "l" and label_vars[x] < 0:
                        c ^= label_vars[x] == ONE_REF
                    elif kind == "l":
                        s = s ^ frozenset((label_vars[x],))
                    else:
                        if x not in col_var:
                            col_var[x] = C._new("call", call_at | x)
                        s = s ^ col_var[x].s
                out.append(L(s, c))
            wires.setdefault(r, out)
        kind = C.kind
        C.used.update(v for v in label_vars if v >= 0 and kind[v] == "and")
        C.zero = getattr(C, "zero", False) or ZERO in label_vars
        return key

    def _generic(self, fc: FunctionCall, key: tuple, pat: tuple, ins: int) -> Spec:
        """``fc``'s Definition lowered on its standalone call, whose argument leaves are distinct gates: a caller binds
        equal constant bits to one gate, so only there is each argument leaf an input of its own."""
        _prog, root = IL.standalone(fc.fn)
        assert len(set(root.args)) == len(root.args) == len(fc.args), f"{fc.id}: a standalone argument leaf repeats"
        outer, self.circuit = self.circuit, root.circuit
        try:
            return self._lower(root, key, pat, ins)
        finally:
            self.circuit = outer

    def _lower(self, fc: FunctionCall, key: tuple, pat: tuple, ins: int) -> Spec:
        C = Circuit()
        C.used, C.blocks = set(), []
        fresh = [C._new("arg", i) for i in range(ins)]
        it = iter(pat)
        wires: dict[int, list] = {}
        for a in dict.fromkeys(fc.args):
            w = []
            for _ in range(IL._width(self.circuit.gate(a).prim)):
                p = next(it)
                if p[0] == "c":
                    w.append(L(frozenset(), p[1]))
                    continue
                f = L(frozenset(), p[1])
                for i in p[0]:
                    f = f ^ fresh[i]
                w.append(C.commit([f])[0] if len(f.s) > self.support else f)
            wires[a] = w
        if fc.is_primitive:
            args = [wires[a] for a in fc.args]
            got = IL.lower_gates(C, [IL._PrimGate(fc.fn, len(args))], {-(i + 1): list(w) for i, w in enumerate(args)})
            wires[fc.lo] = got[0]
        else:
            sc = fc.scope()
            d = getattr(sc.spec, "defn", sc.spec)
            build = getattr(d, "build", None)
            if callable(build):
                if getattr(d, "statics", ()):
                    build = partial(lambda S, C, *e: d.build(C, S, *e), sc.spec.S())
                if not IL._lower_activation(C, sc, build, wires, IL.PIECES):
                    raise KeyError(f"{fc.id}: a parameter leaf has no wire")
            else:
                body = sc.spec.body
                keep = set(fc.returns)
                drop = _last_reads(self.circuit, sc, body)
                for k, node in enumerate(body.nodes):
                    if isinstance(node.fn, PrimitiveDefinition) and node.form == "primitive":
                        lo = sc.offset + body.offsets[k]
                        IL.lower_gates(C, Region(self.circuit, lo, lo + node.gates), wires)
                        members = 0
                    else:
                        members = 1 if node.form == "call" else node.n
                    for m in range(max(members, 1)):
                        if members:
                            self.call(C, FunctionCall(self.circuit, sc, k, m), wires)
                        for g in drop.pop((k, m), ()):
                            if g not in keep:
                                wires.pop(g, None)
        if C.asserts or C.lookups:
            raise NotImplementedError(f"{fc.id}: assertions and table reads are not laid out by the structural lowering yet")
        rets = []
        for r in fc.returns:
            for b in wires[r]:
                if b.is_const():
                    rets.append(("c", b.c))
                    continue
                if len(b.s) > self.support or (self.gather and len(b.s) > 1 and any(C.kind[v] != "arg" for v in b.s)):
                    b = C.commit([b])[0]
                rets.append(("f", b.c, b.s))
                C.used.update(v for v in b.s if C.kind[v] == "and")
        sp = self._finish(C, key, ins, rets)
        self.specs[key] = sp
        sp.index = len(self.order)
        self.order.append(sp)
        return sp

    @staticmethod
    def _returned_last(C: Circuit, live, rets: list) -> list[int]:
        """The returned own columns nothing else in the specialization reads, in return order: placed last, so a caller
        binds a copy's returns as one run."""
        read: set[int] = set()
        for v in live:
            if C.kind[v] == "and":
                a, b = C.data[v]
                read |= a.s | b.s
        kind = C.kind
        for blk in C.blocks:
            read.update(x for x in blk.labels if x >= 0 and kind[x] == "and")
        out: dict[int, None] = {}
        for t in rets:
            if t[0] == "f" and len(t[2]) == 1:
                (v,) = t[2]
                if C.kind[v] == "and" and v not in read:
                    out.setdefault(v)
        return list(out)

    def _finish(self, C: Circuit, key: tuple, ins: int, rets: list) -> Spec:
        """Lay out a lowered specialization: its live ANDs and its copies in creation order, then its rows and its copies'
        runs in template references."""
        live = _live(C, C.used)
        sp = Spec(key, ins)
        col: dict[int, int] = {}
        blocks = [b for b in C.blocks if self.specs[b.key].cols]
        last = self._returned_last(C, live, rets) if self.gather else []
        at_end = set(last)
        bi, n = 0, int(getattr(C, "zero", False))
        for v in range(len(C.kind)):
            while bi < len(blocks) and blocks[bi].pos <= v:
                blocks[bi].base = n
                n += self.specs[blocks[bi].key].cols
                bi += 1
            if C.kind[v] == "and" and v in live and v not in at_end:
                col[v] = n
                n += 1
        for b in blocks[bi:]:
            b.base = n
            n += self.specs[b.key].cols
        for v in last:
            col[v] = n
            n += 1
        sp.cols = n
        K = sp.cols + ins
        kind, data, cblocks = C.kind, C.data, C.blocks
        mask = (1 << COL_BITS) - 1

        def ref(v: int) -> int:
            if v < 0:
                return 0 if v == ZERO else K
            k = kind[v]
            if k == "call":
                d = data[v]
                return cblocks[d >> COL_BITS].base + (d & mask)
            return sp.cols + data[v] if k == "arg" else col[v]

        def refs(f: L) -> tuple:
            return tuple(sorted(map(ref, f.s))) + ((K,) if f.c else ())

        for v in sorted(col):
            if C.kind[v] != "and":
                continue
            a, b = C.data[v]
            sp.rows.append((col[v], refs(a), refs(b)))
            sp.refs += len(a.s) + a.c + len(b.s) + b.c
            sp.commits += b is ONE
        sp.rets = [t if t[0] == "c" else ("f", t[1], tuple(("l", C.data[v]) if C.kind[v] == "arg" else ("v", ref(v))
                                                                for v in sorted(t[2]))) for t in rets]
        sp.copies = [b.key for b in C.blocks]
        sp.calls = _runs(blocks, lambda b: np.fromiter(map(ref, b.labels), np.int64, len(b.labels)), self.specs, sp.cols)
        return sp

    def total(self, key: tuple) -> dict:
        t = self.totals.get(key)
        if t is None:
            sp = self.specs[key]
            t = {"rows": len(sp.rows), "commits": sp.commits, "refs": sp.refs, "copies": len(sp.copies)}
            for c in sp.copies:
                for x, y in self.total(c).items():
                    t[x] += y
            self.totals[key] = t
        return t


def _segs(r0: list[int], step: list[int], cols: int) -> tuple:
    """A copy's input references ``r0`` stepping by ``step`` per copy as one-level segments ``(len, at, step, 1, 0)``: runs
    of consecutive references with one step, each wholly among the caller's own columns or wholly past them."""
    out: list[list[int]] = []
    for t, (r, s) in enumerate(zip(r0, step)):
        if out and r == out[-1][1] + out[-1][0] and s == out[-1][2] and (r < cols) == (out[-1][1] < cols):
            out[-1][0] += 1
        else:
            out.append([1, r, s, 1, 0])
    return tuple(map(tuple, out))


def _runs(blocks: list[_Block], refs, specs: dict, cols: int) -> list[T.Call]:
    """Consecutive copies of one specialization, side by side, whose input references step by one fixed vector per copy,
    each staying among the caller's own columns or past them, as one call each. ``refs(block)`` is a copy's input
    references, read once per copy, so only one run's are held."""
    calls = []
    i = 0
    nxt = refs(blocks[0]) if blocks else None
    while i < len(blocks):
        base, key, r0 = blocks[i].base, blocks[i].key, nxt
        own0 = r0 < cols
        sp = specs[key]
        j = i + 1
        step = None
        nxt = None
        while j < len(blocks):
            b = blocks[j]
            if b.key != key or b.base != base + (j - i) * sp.cols:
                nxt = None
                break
            rj = refs(b)
            nxt = rj
            if not np.array_equal(rj < cols, own0):
                break
            d = rj - r0
            if step is None:
                if (d < 0).any():
                    break
                step = d
            elif not np.array_equal(d, (j - i) * step):
                break
            nxt = None
            j += 1
        calls.append(T.Call(sp.index, j - i, base, sp.cols, _segs(r0.tolist(), (step if step is not None else np.zeros(len(r0), np.int64)).tolist(), cols)))
        i = j
        if nxt is None and i < len(blocks):
            nxt = refs(blocks[i])
    return calls


@dataclass(frozen=True)
class Ports:
    """A unit's ports, as `class_statement.unit_lowering` lays a Definition on rows: the input ports' widths (one input
    group), each argument leaf's bits among their concatenated bits; the output ports' widths (one output group), each
    output bit the returned bit it copies (the returns' leaves concatenated) or None, a zero bit."""

    ins: tuple[int, ...]
    leaves: tuple[tuple[int, ...], ...]
    outs: tuple[int, ...]
    bits: tuple[int | None, ...]


def lower(target, support: int = SUPPORT, name: str = "unit", fold_cap: int = FOLD_CAP,
          fold_gates: int = FOLD_GATES, column_order: bool = True, gather: bool = True,
          ports: Ports | None = None) -> tuple[T.Tmpl, _Lowerer, Spec]:
    """``target`` (a bound composite) as a templated unit: one input group, its root's copy, one output group, the constant
    last (``ir_lower.layout``'s order). The input group is the argument leaves and the output group one port of the return
    leaves, or ``ports``' ports."""
    sys.setrecursionlimit(100000)
    _prog, root = IL.standalone(target)
    lw = _Lowerer(root.circuit, support, fold_cap, fold_gates, column_order, gather)
    args = list(dict.fromkeys(root.args))
    widths = [IL._width(root.circuit.gate(a).prim) for a in args]
    C = Circuit()
    C.used, C.blocks = set(), []
    if ports is None:
        wires = {a: C.input(f"in{a}", w) for a, w in zip(args, widths)}
    else:
        if len(ports.leaves) != len(args) or [len(x) for x in ports.leaves] != widths:
            raise ValueError(f"{name}: the ports' leaves are not the arguments' {widths}")
        bits = [b for j, w in enumerate(ports.ins) for b in C.input(f"in{j}", w)]
        wires = {a: [bits[i] for i in x] for a, x in zip(args, ports.leaves)}
        widths = list(ports.ins)
    top = lw.call(C, root, wires)
    sp = lw.specs[top]
    outs = [b for r in root.returns for b in wires[r]]
    if ports is not None:
        if len(ports.bits) != sum(ports.outs) or {i for i in ports.bits if i is not None} != set(range(len(outs))):
            raise ValueError(f"{name}: the output ports do not copy each returned bit")
        outs = [L(frozenset(), 0) if i is None else outs[i] for i in ports.bits]
    nin = sum(widths)
    in_words = ceil_pow2(max(1, -(-nin // IL.WORD)))
    in_cols = in_words * IL.WORD
    base = in_cols
    col_of_call = {}
    for v in range(len(C.kind)):
        if C.kind[v] == "call":
            col_of_call[v] = base + (C.data[v] & ((1 << COL_BITS) - 1))
    end = base + sp.cols
    out_word = -(-end // IL.WORD)
    out_words = ceil_pow2(max(1, -(-len(outs) // IL.WORD)))
    out_word = -(-out_word // out_words) * out_words
    out_at = out_word * IL.WORD
    const = (out_word + out_words) * IL.WORD
    cols = const + 1

    def ref(v):
        return v if C.kind[v] == "in" else col_of_call[v]

    rows = [(i, (i,), (i,)) for i in range(nin)]
    for j, b in enumerate(outs):
        rows.append((out_at + j, tuple(sorted(map(ref, b.s))) + ((cols,) if b.c else ()), (cols,)))
    blk = C.blocks[0]
    assert min(blk.labels, default=0) >= 0, "the root's call binds a constant"
    calls = [T.Call(sp.index, 1, base, sp.cols, _segs([ref(v) for v in blk.labels], [0] * len(blk.labels), cols))] if sp.cols else []
    out_widths = (len(outs),) if ports is None else ports.outs
    groups = [str(x) for x in (1, 0, in_words, len(widths), *widths, 1, out_word, out_words, len(out_widths), *out_widths)]
    defs = [T.Def(s.cols, s.ins, tuple(s.rows), tuple(s.calls)) for s in lw.order]
    defs.append(T.Def(cols, 0, tuple(rows), tuple(calls)))
    t = T.Tmpl(name, len(defs) - 1, groups, defs)
    lw.segs_one_level = sum(len(c.segs) for d in t.defs for c in d.calls)
    return T.pack(t), lw, sp


def evaluate(flat: T.Flat, inputs: list[int], nin: int) -> list[int]:
    """The unrolled unit's column values, each a T-lane int: input column ``i`` is ``inputs[i]``, each later row
    ``(A·z)(B·z)``, in column order; the constant's lanes all one."""
    k = flat.const
    n = len(flat.na)
    mask = (1 << 64) - 1
    z = [0] * n
    z[k] = mask
    for i in range(nin):
        z[i] = inputs[i]
    for i in range(nin, n):
        if i == k:
            continue
        a, b = flat.row(i)
        if not a and not b:
            continue
        assert all(c < i or c == k for c in a + b), f"row {i} reads a later column"
        x = y = 0
        for c in a:
            x ^= z[c]
        for c in b:
            y ^= z[c]
        z[i] = x & y
    return z


def check(target, support: int = SUPPORT, seed: int = 0, **fold) -> dict:
    """The templated unit unrolled and evaluated on 64 random inputs, against the unrolled lowering's outputs."""
    t, _lw, _sp = lower(target, support, **fold)
    flat = T.unroll(t)
    _prog, root = IL.standalone(target)
    args = list(dict.fromkeys(root.args))
    widths = [IL._width(root.circuit.gate(a).prim) for a in args]
    rng = np.random.default_rng(seed)
    lanes = [int(x) for x in rng.integers(0, 1 << 64, size=sum(widths), dtype=np.uint64)]
    z = evaluate(flat, lanes, sum(widths))
    g = t.groups
    nin_groups = int(g[0])
    at = 1 + nin_groups * 3 + sum(int(g[1 + 3 * i + 2]) for i in range(nin_groups))
    out_col = int(g[at + 1]) * IL.WORD
    C = Circuit()
    ins, at_bit = {}, 0
    for a, w in zip(args, widths):
        ins[f"in{a}"] = lanes[at_bit:at_bit + w]
        at_bit += w
    words = IL.lower(C, target, [C.input(f"in{a}", w) for a, w in zip(args, widths)])
    for j, w in enumerate(words):
        C.output(f"o{j}", w)
    want, _fail = C.evaluate(ins, 64)
    want = [x for j in range(len(words)) for x in want[f"o{j}"]]
    got = z[out_col:out_col + len(want)]
    return {"outputs": len(want), "mismatched_bits": sum(x != y for x, y in zip(got, want)), "rows": len(flat.na)}


def stats(t: T.Tmpl, lw: _Lowerer, top: Spec, text_bytes: int | None = None) -> dict:
    tot = lw.total(top.key)
    by_fn: dict[str, int] = {}
    for k in lw.specs:
        by_fn[k[0]] = by_fn.get(k[0], 0) + 1
    rows = sum(len(d.rows) for d in t.defs)
    segs: dict[str, list[int]] = {}
    for d in t.defs:
        for c in d.calls:
            s = segs.setdefault(lw.order[c.e].key[0].split("{")[0], [0, 0])
            s[0] += 1
            s[1] += len(c.segs)
    top_segs = sorted(segs.items(), key=lambda kv: -kv[1][1])[:8]
    return {"segs_by_callee": {f: {"calls": n, "segs": m} for f, (n, m) in top_segs},
            "unrolled": tot, "template": {"defs": len(t.defs), "definitions": len(by_fn), "rows": rows,
                                          "refs": sum(len(a) + len(b) for d in t.defs for _, a, b in d.rows),
                                          "calls": sum(len(d.calls) for d in t.defs),
                                          "segs": sum(len(c.segs) for d in t.defs for c in d.calls),
                                          "segs_one_level": getattr(lw, "segs_one_level", None),
                                          "copies": sum(len(s.copies) for s in lw.order), "bytes": text_bytes,
                                          "root_cols": t.defs[t.root].cols},
            "patterns": {f: n for f, n in sorted(by_fn.items(), key=lambda kv: -kv[1])[:8]}}


def flat(target) -> dict:
    """``target``'s unrolled lowering (:func:`ir_lower.lower`), counted as :func:`stats` counts the template's rows."""
    _prog, root = IL.standalone(target)
    C = Circuit()
    outs = IL.lower(C, target, [C.input(f"in{a}", IL._width(root.circuit.gate(a).prim)) for a in dict.fromkeys(root.args)])
    C.output("used", [b for w in outs for b in w])
    t = {"rows": 0, "commits": 0, "refs": 0, "lookups": 0, "asserts": len(C.asserts)}
    for v in C.live():
        if C.kind[v] == "and":
            a, b = C.data[v]
            t["rows"] += 1
            t["refs"] += len(a.s) + a.c + len(b.s) + b.c
            t["commits"] += b is ONE
        elif C.kind[v] == "lookup":
            t["lookups"] += 1
    return {"flat": t}


def main(argv: list[str]) -> int:
    import argparse
    import resource
    from pathlib import Path

    ap = argparse.ArgumentParser(prog="struct_lower")
    ap.add_argument("shape", nargs="+", type=int)
    ap.add_argument("--support", type=int, default=SUPPORT)
    ap.add_argument("--fold-cap", type=int, default=FOLD_CAP)
    ap.add_argument("--first-appearance", action="store_true", help="number a callee's inputs as its arguments name them")
    ap.add_argument("--scattered-returns", action="store_true", help="return forms over a callee's columns where they lie")
    ap.add_argument("--out")
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    h1 = Path(__file__).resolve().parents[5] / "benchmarks" / "pouw" / "h1_prove"
    sys.path[:0] = [str(h1), str(h1.parent / "hidden_zk")]
    import h1unit as U

    shape = U.Shape(*a.shape)
    t0 = time.time()
    if a.flat:
        out = flat(shape.unit())
    elif a.check:
        out = {"check": check(shape.unit(), a.support, fold_cap=a.fold_cap, column_order=not a.first_appearance,
                             gather=not a.scattered_returns)}
    else:
        t, lw, top = lower(shape.unit(), a.support, f"h1-struct-{shape.k}", a.fold_cap, column_order=not a.first_appearance,
                           gather=not a.scattered_returns)
        size = None
        if a.out:
            text = T.write(t)
            Path(a.out).write_text(text)
            size = len(text)
            Path(a.out + ".names.json").write_text(json.dumps([s.key[0] for s in lw.order] + ["(unit)"]))
        out = stats(t, lw, top, size) | {"support": a.support, "fold_cap": a.fold_cap, "column_order": not a.first_appearance,
                                               "gather": not a.scattered_returns}
    out |= {"k": shape.k, "tile": [shape.r, shape.c], "seconds": round(time.time() - t0, 1),
            "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    print("STRUCT " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

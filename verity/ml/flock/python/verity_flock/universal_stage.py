"""`UniversalUnit_v1{G, N_IN, N_OUT}`'s class statement at the size of a real private circuit: the files
`class_statement.stage` writes for the unit's one class (``circuit.txt``, ``inst-n.bin``, ``pub-n.bin``, words packed), byte for
byte, from arrays rather than Python objects per gate (``pod/rec_inner.py universal --fast``).

Today's path traces the unit into `forms` objects (`trace.trace`), cuts it into units (`units.PartitionUnits`), lowers the
unit's gates into GF(2) forms (`class_statement.class_lowering`), lays them out (`ir_lower.layout`) and writes the layout as
text, each a Python object per gate or per column: at G = 8090 (67M ANDs) hours and hundreds of GB. This path writes the
same statement from the unit's structure:

* **The gate circuit** (`trace`): `universal.build` traced as `trace.trace` emits it, as arrays (kind, operand wires, depth):
  each gather's multiplexer levels, the gate's tail, every form emitted when an AND first reads it, the gather's first-level
  differences shared across trees, a gate's wire written when a later tree first reads it. `TR.Traced` of it is the one
  `universal.traced` builds (checked by the tests), and the Program's descriptor (its digest, the partition object's) is
  core's own, encoded from it (`program`).
* **The unit** (`Unit`): the gates in canonical order (`trace.emit`'s batches: depth, kind, trace order), the unit's inputs in
  first-read order and its sources (`class_statement.sources`), its form's SHA-512 (the shape) and SHA-256 (the shims'
  content), hashed in chunks without holding the form.
* **The layout**: `ir_lower.layout`'s rows of `class_statement.class_lowering`'s circuit, each AND's operands as the sorted
  columns of their GF(2) forms (symmetric differences, batch by batch in canonical order), written as ``flock-ir-unit/v2``
  text by vectorized formatting, and run on the instances (bit-sliced, batch by batch) as `ir_lower.evaluate_layout` runs it.
* **The statement**: `circuit.compose` and `circuit.write` themselves, on a `Lowering` whose layout is this one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from verity.ml.private_circuits import universal as U

from . import circuit as FC
from . import class_statement as CS
from . import ir_lower as IL

AND, XOR = 0, 1
_KIND_BITS = 29
#: gates per vectorized step of the canonical-order passes (a depth batch is split into steps of at most this many)
STEP = 1 << 22


# --- the gate circuit --------------------------------------------------------------------------------------------------------

class _Emitter:
    def __init__(self, first: int, cap: int = 1 << 16):
        self.first, self.n = first, 0
        self.kind = np.empty(cap, np.uint8)
        self.a = np.empty(cap, np.int64)
        self.b = np.empty(cap, np.int64)
        self.depth = np.empty(cap, np.int32)

    def alloc(self, m: int) -> int:
        if self.n + m > len(self.kind):
            cap = len(self.kind)
            while cap < self.n + m:
                cap *= 2
            for k in ("kind", "a", "b", "depth"):
                old = getattr(self, k)
                new = np.empty(cap, old.dtype)
                new[:self.n] = old[:self.n]
                setattr(self, k, new)
        base = self.first + self.n
        self.n += m
        return base

    def _d(self, w):
        w = np.asarray(w, np.int64)
        return np.where(w >= self.first, self.depth[np.maximum(w - self.first, 0)], 0)

    def put(self, ids, kind: int, a, b) -> None:
        i = np.asarray(ids, np.int64) - self.first
        self.kind[i] = kind
        self.a[i] = a
        self.b[i] = b
        self.depth[i] = 1 + np.maximum(self._d(a), self._d(b))

    def one(self, kind: int, a: int, b: int) -> int:
        w = self.alloc(1)
        self.put([w], kind, [a], [b])
        return w


@dataclass
class Gates:
    """`universal.traced`'s gate circuit as arrays: gate ``i`` (wire ``first + i``) is ``kind[i]`` (`AND`, `XOR`) of wires
    ``a[i]``, ``b[i]``; wires ``2 ..`` are the input bits in leaf order; ``outputs`` each output's wire."""

    G: int
    N_IN: int
    N_OUT: int
    first: int
    kind: np.ndarray
    a: np.ndarray
    b: np.ndarray
    depth: np.ndarray
    outputs: np.ndarray

    def traced(self):
        """The `trace.Traced` `universal.traced` returns (for `program`; a Python tuple per gate)."""
        from verity.ml.circuits.boolean import trace as TR

        names = ("and", "xor")
        gates = list(zip(map(names.__getitem__, self.kind.tolist()), zip(self.a.tolist(), self.b.tolist())))
        counts = {"and": int(np.count_nonzero(self.kind == AND)), "xor": int(np.count_nonzero(self.kind == XOR)), "not": 0}
        return TR.Traced(self.first - 2, gates, [int(o) for o in self.outputs], counts)


def trace(G: int, N_IN: int, N_OUT: int) -> Gates:
    """The unit's gate circuit as `trace.trace` emits `universal.build`'s: every AND in build order, each preceded by the forms
    it is the first to read (a gather level's partial results, a first-level difference of two wires, a gate's wire), then the
    outputs."""
    U._check_class(G, N_IN, N_OUT)
    K = U.index_bits(G, N_IN)
    op0, src0 = 2, 2 + 2 * G
    out0 = src0 + 2 * G * K
    x0 = out0 + N_OUT * K
    first = x0 + N_IN
    E = _Emitter(first)
    W_max = N_IN + G
    leaf = np.full(W_max, -1, np.int64)
    leaf[:N_IN] = x0 + np.arange(N_IN)
    mk_s = np.zeros(G, np.int64)
    mk_tq = np.zeros(G, np.int64)
    d1 = np.full(W_max // 2 + 1, -1, np.int64)
    n_d1 = 0

    def leaf_wire(j: int) -> int:
        if leaf[j] < 0:
            h = j - N_IN
            m = E.one(XOR, int(mk_s[h]), int(mk_tq[h]))
            leaf[j] = E.one(XOR, m, op0 + 2 * h + 1)
        return int(leaf[j])

    def tree(W: int, bit0: int) -> tuple[int, int]:
        nonlocal n_d1
        p, f = W // 2, W % 2
        s = bit0
        A = np.empty(p + f, np.int64)
        old = min(p, n_d1)
        if old:
            base = E.alloc(old)
            ids = base + np.arange(old)
            E.put(ids, AND, s, d1[:old])
            A[:old] = ids
        for i in range(old, p):
            leaf_wire(2 * i)
            leaf_wire(2 * i + 1)
            d = E.one(XOR, int(leaf[2 * i + 1]), int(leaf[2 * i]))
            d1[i] = d
            n_d1 = i + 1
            A[i] = E.one(AND, s, d)
        if f:
            A[p] = E.one(AND, s, leaf_wire(W - 1))
        Y = np.concatenate([leaf[0:2 * p:2], leaf[W - 1:W] if f else leaf[:0]])
        for lv in range(1, K):
            s = bit0 + lv
            c = len(Y)
            p, f = c // 2, c % 2
            base = E.alloc(4 * p + 2 * f)
            yi = base + 4 * np.arange(p)
            E.put(yi, XOR, Y[0:2 * p:2], A[0:2 * p:2])
            E.put(yi + 1, XOR, Y[1:2 * p:2], A[1:2 * p:2])
            E.put(yi + 2, XOR, yi + 1, yi)
            E.put(yi + 3, AND, s, yi + 2)
            Yn, An = yi, yi + 3
            if f:
                yf = base + 4 * p
                E.put([yf], XOR, Y[c - 1:c], A[c - 1:c])
                E.put([yf + 1], AND, [s], [yf])
                Yn, An = np.append(Yn, yf), np.append(An, yf + 1)
            Y, A = Yn, An
        assert len(Y) == 1
        return int(Y[0]), int(A[0])

    for g in range(G):
        W = N_IN + g
        ya, aa = tree(W, src0 + 2 * g * K)
        yb, ab_ = tree(W, src0 + (2 * g + 1) * K)
        a = E.one(XOR, ya, aa)
        b = E.one(XOR, yb, ab_)
        ab = E.one(AND, a, b)
        s = E.one(XOR, a, b)
        abs_ = E.one(XOR, ab, s)
        mk_s[g], mk_tq[g] = s, E.one(AND, op0 + 2 * g, abs_)
    tops = [tree(W_max, out0 + j * K) for j in range(N_OUT)]
    outputs = np.asarray([E.one(XOR, y, a) for y, a in tops], np.int64)
    n = E.n
    return Gates(G, N_IN, N_OUT, first, E.kind[:n].copy(), E.a[:n].copy(), E.b[:n].copy(), E.depth[:n].copy(), outputs)


def program(gates: Gates):
    """(the one-Call Program of the unit, its ``Q_template_instance`` query), core's own, with `universal.traced` answered
    from ``gates``: the Program's descriptor (its digest and the partition object's) is encoded from the emitted body."""
    from verity.ml.circuits import partition_object as PO
    from verity.ml.circuits.defs import bind

    from . import partition_units as PU

    T = gates.traced()
    key = (gates.G, gates.N_IN, gates.N_OUT)
    orig = U.traced
    U.traced = lambda *k: T if k == key else orig(*k)  # type: ignore[assignment]
    try:
        fn = bind(U.UniversalUnit, G=gates.G, N_IN=gates.N_IN, N_OUT=gates.N_OUT)
        prog = PU.one_call(fn)
        query = PO.template_instance_query(fn)
        built = PO.build(prog, query)
        from verity.ml.circuits.codec import program_digest

        digest = program_digest(prog)
    finally:
        U.traced = orig  # type: ignore[assignment]
    return prog, query, built, digest


# --- vectorized text ---------------------------------------------------------------------------------------------------------

def _digits(v: np.ndarray) -> np.ndarray:
    nd = np.ones(v.shape, np.int64)
    t = 10
    while True:
        m = v >= t
        if not m.any():
            return nd
        nd += m
        t *= 10


def _render(pieces: list) -> bytes:
    """Items (one per element of the pieces' arrays), each the concatenation of its pieces: ``("num", ints)`` a non-negative
    integer in decimal, ``("lit", [bytes, ...], choice)`` the literal ``choice`` picks (an array, or one index for all)."""
    m = next(len(p[1]) if p[0] == "num" else len(p[2]) for p in pieces if p[0] == "num" or not np.isscalar(p[2]))
    lens = np.empty((m, len(pieces)), np.int64)
    for j, p in enumerate(pieces):
        if p[0] == "num":
            lens[:, j] = _digits(np.asarray(p[1], np.int64))
        else:
            lens[:, j] = np.asarray([len(x) for x in p[1]], np.int64)[p[2]]
    flat = lens.ravel()
    ends = np.cumsum(flat)
    if not len(ends):
        return b""
    starts = (ends - flat).reshape(m, len(pieces))
    buf = np.empty(int(ends[-1]), np.uint8)
    for j, p in enumerate(pieces):
        st = starts[:, j]
        if p[0] == "num":
            v = np.asarray(p[1], np.int64).copy()
            nd = lens[:, j]
            at = st + nd - 1
            for k in range(int(nd.max())):
                sel = nd > k
                buf[at[sel] - k] = 48 + (v[sel] % 10)
                v //= 10
        else:
            choice = np.broadcast_to(np.asarray(p[2]), (m,))
            for c, lit in enumerate(p[1]):
                sel = st[choice == c]
                for t, ch in enumerate(lit):
                    buf[sel + t] = ch
    return buf.tobytes()


def _lines(lens_a: np.ndarray, cols_a: np.ndarray, lens_b: np.ndarray, cols_b: np.ndarray) -> bytes:
    """Rows as ``flock-ir-unit/v2`` lines ``|A| A.. |B| B..``."""
    r = len(lens_a)
    per = 2 + lens_a + lens_b
    rs = np.cumsum(per) - per
    flat = np.empty(int(per.sum()), np.int64)
    flat[rs] = lens_a
    oa = np.cumsum(lens_a) - lens_a
    flat[np.repeat(rs + 1 - oa, lens_a) + np.arange(len(cols_a))] = cols_a
    flat[rs + 1 + lens_a] = lens_b
    ob = np.cumsum(lens_b) - lens_b
    flat[np.repeat(rs + 2 + lens_a - ob, lens_b) + np.arange(len(cols_b))] = cols_b
    sep = np.zeros(len(flat), np.int64)
    sep[rs + per - 1] = 1
    return _render([("num", flat), ("lit", [b" ", b"\n"], sep)]) if r else b""


# --- the unit ----------------------------------------------------------------------------------------------------------------

class _Store:
    """Each wire's GF(2) form as sorted unit columns, append-only."""

    def __init__(self, wires: int, cap: int = 1 << 20):
        self.start = np.zeros(wires, np.int64)
        self.len = np.zeros(wires, np.int64)
        self.cols = np.empty(cap, np.int32)
        self.n = 0

    def put(self, wires: np.ndarray, lens: np.ndarray, cols: np.ndarray) -> None:
        if self.n + len(cols) > len(self.cols):
            cap = len(self.cols)
            while cap < self.n + len(cols):
                cap *= 2
            new = np.empty(cap, np.int32)
            new[:self.n] = self.cols[:self.n]
            self.cols = new
        self.start[wires] = self.n + np.cumsum(lens) - lens
        self.len[wires] = lens
        self.cols[self.n:self.n + len(cols)] = cols
        self.n += len(cols)

    def get(self, wires: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        st, ln = self.start[wires], self.len[wires]
        off = np.cumsum(ln) - ln
        return ln, self.cols[np.repeat(st - off, ln) + np.arange(int(ln.sum()))]


def _xor_reduce(z: np.ndarray, lens: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if not len(lens):
        return z[:0]
    if (lens == 0).any():
        raise AssertionError("an empty form reaches a gate")
    return np.bitwise_xor.reduceat(z[cols], np.cumsum(lens) - lens, axis=0)


class FastLayout:
    """`ir_lower.Layout`'s fields `circuit.compose` and `class_statement.stage` read, and the layout's ``flock-ir-unit/v2``
    body (`ir_lower.netlist` without ``LEAVES``) as bytes."""

    def __init__(self, useful, const, in_groups, out_groups, counts, body: bytes):
        self.useful, self.const, self.in_groups, self.out_groups, self.counts, self.body = useful, const, in_groups, out_groups, counts, body

    @property
    def in_words(self) -> int:
        col, words, _ = self.in_groups[-1]
        return col + words


class FastLowering(IL.Lowering):
    """The class lowering `class_statement.class_lowering` builds, its layout a `FastLayout`: its text is the layout's body and
    its ``LEAVES``; it evaluates the instances it was built on (and the all-zero instance, which every row maps to zero: no
    row reads the constant column)."""

    runs: dict

    @property
    def text(self) -> str:
        return (self.layout.body + b"LEAVES " + json.dumps(self.leaves, sort_keys=True, separators=(",", ":")).encode() + b"\n").decode()

    @property
    def sha256(self) -> str:
        h = hashlib.sha256(self.layout.body)
        h.update(b"LEAVES " + json.dumps(self.leaves, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        return h.hexdigest()

    def evaluate(self, rows, cuts=None, flip=None):
        rows = np.asarray(rows, np.uint64)
        if flip is not None or cuts is not None:
            raise NotImplementedError("the fast lowering evaluates its own instances only")
        w_out = self.definition.w_out
        if not rows.any():
            return np.zeros((len(rows), w_out), np.uint64), None, (1 << len(rows)) - 1
        hit = self.runs.get(rows.tobytes())
        if hit is None:
            raise NotImplementedError("the fast lowering evaluates its own instances only")
        return hit.copy(), None, (1 << len(rows)) - 1


@dataclass
class Unit:
    """The unit's class: its gates in canonical order and its interface, as `units.PartitionUnits` derives them."""

    gates: Gates
    order: np.ndarray          # canonical position -> trace gate
    inputs: np.ndarray         # unit input port -> input wire
    outputs: np.ndarray        # unit output (ascending canonical position) -> output index of the unit
    out_pos: np.ndarray        # their canonical positions
    groups: list[list[int]]    # `class_statement.sources`: per row port, its unit inputs in order
    shape: str
    content_sha256: str        # SHA-256 of the form + "/packed" (`class_statement._shims`)

    @property
    def leaf_of_input(self) -> np.ndarray:
        return self.inputs - 2


def unit(gates: Gates) -> Unit:
    first = gates.first
    N = len(gates.kind)
    idx = np.arange(N, dtype=np.int64)
    if N >> _KIND_BITS:
        raise ValueError(f"{N} gates: past the sort key's {_KIND_BITS} index bits")
    key = (gates.depth.astype(np.int64) << (_KIND_BITS + 1)) | (gates.kind.astype(np.int64) << _KIND_BITS) | idx
    key.sort()
    order = key & ((1 << _KIND_BITS) - 1)
    del key
    pos = np.empty(N, np.int64)
    pos[order] = idx
    fo = np.empty(2 * N, np.int64)
    fo[0::2] = gates.a[order]
    fo[1::2] = gates.b[order]
    ins = fo < first
    wires, at = np.unique(fo[ins], return_index=True)
    inputs = wires[np.argsort(at)]
    port = np.full(first, -1, np.int64)
    port[inputs] = np.arange(len(inputs))
    G, K, N_OUT = gates.G, U.index_bits(gates.G, gates.N_IN), gates.N_OUT
    bounds = np.asarray([2 + 2 * G, 2 + 2 * G + 2 * G * K, 2 + 2 * G + 2 * G * K + N_OUT * K])
    param = np.searchsorted(bounds, inputs, side="right")
    groups: dict[int, list[int]] = {}
    for t, p in enumerate(param.tolist()):
        groups.setdefault(p, []).append(t)
    out_pos = np.sort(pos[gates.outputs - first])
    outputs = np.argsort(pos[gates.outputs - first])
    sha512 = hashlib.sha512(b"verity/partition/v2/shape\0")
    sha256 = hashlib.sha256()

    def feed(b: bytes) -> None:
        sha512.update(b)
        sha256.update(b)

    feed(b'{"gates":[')
    prims = [b'["And2_v1",[["', b'["Xor2_v1",[["']
    for lo in range(0, N, STEP):
        hi = min(N, lo + STEP)
        t = order[lo:hi]
        refs = []
        for ops in (gates.a[t], gates.b[t]):
            g = ops >= first
            refs.append((g.astype(np.int64), np.where(g, pos[np.maximum(ops - first, 0)], port[np.minimum(ops, first - 1)])))
        last = np.zeros(hi - lo, np.int64)
        if hi == N:
            last[-1] = 1
        feed(_render([("lit", prims, gates.kind[t].astype(np.int64)), ("lit", [b'i",', b'g",'], refs[0][0]), ("num", refs[0][1]),
                      ("lit", [b'],["i",', b'],["g",'], refs[1][0]), ("num", refs[1][1]), ("lit", [b"]]],", b"]]]"], last)]))
    feed(b'],"in":[' + b",".join([b"1"] * len(inputs)) + b'],"out":[' + ",".join(map(str, out_pos.tolist())).encode() + b"]}")
    sha256.update(b"/packed")
    return Unit(gates, order, inputs, outputs, out_pos, list(groups.values()), sha512.hexdigest(), sha256.hexdigest())


def _port_layout(u: Unit) -> tuple[list[int], np.ndarray]:
    """Each row port's words (`class_statement.class_lowering`: packed slots, padded to whole SHA-512 blocks), and each unit
    input's column (its leaf's bit)."""
    col = np.empty(len(u.inputs), np.int64)
    port_words, base = [], 0
    for g in u.groups:
        r = np.arange(len(g))
        col[g] = 16 * (base + r // 16) + r % 16
        used = -(-len(g) // 16)
        port_words.append(CS._pad(used, CS.BLOCK_WORDS))
        base += port_words[-1]
    return port_words, col


def rows_of(u: Unit, net: U.Netlist, xs: Sequence[Sequence[int]], port_words: Sequence[int]) -> np.ndarray:
    """`class_statement.pack`'s ``rows``: every instance's input words, all row ports side by side, u16."""
    leaf = u.leaf_of_input
    rows = np.zeros((len(xs), sum(port_words)), np.uint16)
    _, col = _port_layout(u)
    for t, x in enumerate(xs):
        bits = np.asarray([b for part in U.arguments(net, x) for b in part], np.uint16)
        np.bitwise_or.at(rows[t], col // 16, (bits[leaf] << (col % 16)).astype(np.uint16))
    return rows


def lower(u: Unit, name: str, lanes: Sequence[np.ndarray]) -> tuple[FastLowering, list[np.ndarray]]:
    """The class lowering and its layout, and each lane set's outputs as the laid-out rows compute them (``lanes``: `rows_of`'s
    rows per netlist; outputs ``(n, w_out)`` uint64, in the unit's output order)."""
    gates, first = u.gates, u.gates.first
    N = len(gates.kind)
    port_words, in_col = _port_layout(u)
    w_in, w_out = sum(port_words), len(u.outputs)
    W = IL.WORD
    in_words = IL.ceil_pow2(max(1, -(-16 * w_in // W)))
    base = in_words * W
    st = _Store(first + N)
    st.put(u.inputs, np.ones(len(u.inputs), np.int64), in_col.astype(np.int32))
    sizes = [len(r) for r in lanes]
    allrows = np.concatenate(lanes, axis=0) if lanes else np.zeros((0, w_in), np.uint16)
    T = len(allrows)
    L = max(1, -(-T // 64))
    n_ands = int(np.count_nonzero(gates.kind == AND))
    words_out = IL.ceil_pow2(max(1, -(-16 * w_out // W)))
    out_at = -(-(base + n_ands) // (W * words_out)) * (W * words_out)
    const = out_at + W * words_out
    z = np.zeros((const + 1, L), np.uint64)
    lane_bits = np.zeros((16 * w_in, L), np.uint64)
    for t in range(T):
        w, bit = t // 64, np.uint64(t % 64)
        v = allrows[t].astype(np.uint64)
        for b in range(16):
            lane_bits[b::16, w] |= ((v >> np.uint64(b)) & np.uint64(1)) << bit
    z[:16 * w_in] = lane_bits
    key = (gates.depth[u.order].astype(np.int64) << 1) | gates.kind[u.order].astype(np.int64)
    cuts = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [N]])
    chunks = []
    nnz_a = nnz_b = 0
    k = 0
    for s0, e0 in zip(starts.tolist(), ends.tolist()):
        for lo in range(s0, e0, STEP):
            hi = min(e0, lo + STEP)
            t = u.order[lo:hi]
            wires = first + t
            la, ca = st.get(gates.a[t])
            lb, cb = st.get(gates.b[t])
            m = hi - lo
            if gates.kind[t[0]] == AND:
                cols = base + k + np.arange(m)
                k += m
                st.put(wires, np.ones(m, np.int64), cols.astype(np.int32))
                z[cols] = _xor_reduce(z, la, ca) & _xor_reduce(z, lb, cb)
                chunks.append((la, ca, lb, cb))
                nnz_a += int(la.sum())
                nnz_b += int(lb.sum())
            else:
                seg = np.concatenate([np.repeat(np.arange(m, dtype=np.int64), la), np.repeat(np.arange(m, dtype=np.int64), lb)])
                kk = (seg << 32) | np.concatenate([ca, cb]).astype(np.int64)
                kk.sort()
                keep = np.ones(len(kk), bool)
                dup = kk[1:] == kk[:-1]
                keep[1:] &= ~dup
                keep[:-1] &= ~dup
                kk = kk[keep]
                st.put(wires, np.bincount(kk >> 32, minlength=m).astype(np.int64), (kk & 0xFFFFFFFF).astype(np.int32))
    assert k == n_ands
    lo_, fo = st.get(gates.outputs[u.outputs])
    zo = _xor_reduce(z, lo_, fo)
    z[out_at + 16 * np.arange(w_out)] = zo
    outs, at = [], 0
    for n in sizes:
        o = np.zeros((n, w_out), np.uint64)
        for t in range(n):
            w, bit = (at + t) // 64, np.uint64((at + t) % 64)
            o[t] = (zo[:, w] >> bit) & np.uint64(1)
        outs.append(o)
        at += n
    in_groups = [(0, in_words, [16] * w_in)]
    out_groups = [(out_at // W, words_out, [16] * w_out)]
    head = [IL.FORMAT, name, const + 1, const]
    for grp in (in_groups, out_groups):
        head.append(len(grp))
        for c, words, bits in grp:
            head += [c, words, len(bits), *bits]
    parts = [" ".join(map(str, head)).encode() + b"\n"]
    ic = np.arange(16 * w_in, dtype=np.int64)
    ones = np.ones(16 * w_in, np.int64)
    parts.append(_lines(ones, ic, ones, ic))
    parts.append(b"0 0\n" * (base - 16 * w_in))
    parts += [_lines(*c) for c in chunks]
    parts.append(b"0 0\n" * (out_at - base - n_ands))
    out_lines = np.zeros(16 * w_out, np.int64)
    out_lines[::16] = lo_
    oc = np.zeros(int(lo_.sum()), np.int64)
    oc[:] = fo
    parts.append(_lines(out_lines, oc, np.ones(16 * w_out, np.int64), np.full(16 * w_out, const, np.int64)))
    parts.append(b"0 0\n" * (W * words_out - 16 * w_out))
    parts.append(f"1 {const} 1 {const}\n".encode())
    body = b"".join(parts)
    counts = {"ands": n_ands, "asserts": 0, "rows": const + 1, "in_words": in_words, "out_words": words_out,
              "nnz_a": 16 * w_in + nnz_a + int(lo_.sum()) + 1, "nnz_b": 16 * w_in + nnz_b + 16 * w_out + 1, "reads": 0,
              "read_rows": 0, "read_nnz_a": 0}
    lay = FastLayout(const + 1, const, in_groups, out_groups, counts, body)
    units = IL.Units(target=None, in_widths=[16] * w_in, out_widths=[16] * w_out, in_src=[[("leaf", p) for p in range(w_in)]],
                     out_src=[[("ret", p) for p in range(w_out)]], signature=(u.shape,), gates=[], arg_gates=[], ret_gates=[])
    low = FastLowering(name, CS.ClassDefinition(name, tuple(port_words), w_out), units, None, lay)
    low.runs = {r.astype(np.uint64).tobytes(): o for r, o in zip(lanes, outs)}
    return low, outs


# --- the statement -----------------------------------------------------------------------------------------------------------

def _expanded_with(lay_type):
    orig = FC._expanded

    def expanded(lay, name):
        return lay.body.decode() if isinstance(lay, lay_type) else orig(lay, name)

    return orig, expanded


def stage(gates: Gates, u: Unit, partition: str, program_sha512: str, nets: Sequence[tuple[str, U.Netlist]],
          xs: Sequence[Sequence[int]], out: Path, salt_key: bytes) -> dict:
    """`class_statement.stage`'s files for each ``(name, netlist)`` of ``nets`` on the instances ``xs`` into ``out/name``
    (packed words, `class_statement.LEAF_SCHEME`), one composed circuit for all; each name's record as `stage` returns it,
    with its outputs (``outputs``: per instance, `universal.interpret`'s order) as the laid-out rows compute them."""
    name = u.shape[:16] + "-packed"
    port_words, _ = _port_layout(u)
    lanes = [rows_of(u, net, xs, port_words) for _, net in nets]
    low, outs = lower(u, f"unit-class/{name}", lanes)
    sub_name = name
    content = u.content_sha256
    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
    sub = SimpleNamespace(id=f"unit-class/{sub_name}", template=SimpleNamespace(id="unit-class"),
                          inputs=[port(f"p{i}") for i in range(len(u.groups))], outputs=[port("out")],
                          program_digests=lambda statics: {f"unit-class/{sub_name}": content})
    iset = SimpleNamespace(subcircuit=sub, name=f"unit-class-{sub_name}", content_digest=content)
    orig, expanded = _expanded_with(FastLayout)
    FC._expanded = expanded
    try:
        text = FC.compose(low, sub, partition=partition, program=program_sha512, port_bits=None, seg_ports=frozenset(),
                          seg_out_rows=(), v2_out_rows=(), links=None, definition=None, leaf_scheme=CS.LEAF_SCHEME)
    finally:
        FC._expanded = orig
    meta = json.loads(text.split("\n", 2)[1][5:])
    per_instance = (1 << meta["k_log"]) // meta["vus_per_block"]
    circuit_sha512, class_sha512 = FC.digest(text), FC.class_digest(text)
    res = {}
    for (nm, net), rows, got in zip(nets, lanes, outs):
        n = len(rows)
        want = np.asarray([U.interpret(net, x) for x in xs], np.uint64).reshape(n, -1)[:, u.outputs]
        if not np.array_equal(got, want):
            raise AssertionError(f"{nm}: the laid-out unit rows disagree with universal.interpret")
        if n * per_instance > CS._CTX.get("max_statement_bits", CS.MAX_STATEMENT_BITS):
            raise ValueError(f"{nm}: {n} instances are past the statement's cap (class_statement.stage would cap them)")
        outw = want.astype(np.uint16)
        d = out / nm
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "circuit.txt", "w") as f:
            f.write(text)
        per_port, b = {}, 0
        tables, refs = {}, {}
        for i, (g, wp) in enumerate(zip(u.groups, port_words)):
            per_port[f"p{i}"] = rows[:, b:b + wp]
            b += wp
            x_port = int(np.searchsorted(np.asarray([2 + 2 * gates.G, 2 + 2 * gates.G + 2 * gates.G * U.index_bits(gates.G, gates.N_IN),
                                                     gates.first - gates.N_IN]), u.inputs[g[0]], side="right")) == 3
            at = list(range(n)) if x_port else [0]
            tables[f"p{i}"], refs[f"p{i}"] = per_port[f"p{i}"][at], np.arange(n) if x_port else np.zeros(n, np.int64)
        head = FC.write(text, iset, 0, n, d / f"inst-{n}.bin", d / f"pub-{n}.bin", salt_key, rows=per_port, outs={"out": outw},
                        share_rows=True, tables=tables, refs=refs, salts=None, indices=None)
        res[nm] = {"n": n, "unit_nnz": low.layout.counts["nnz_a"] + low.layout.counts["nnz_b"], "circuit_bytes": len(text),
                   "w_in": low.definition.w_in, "w_out": low.definition.w_out, "unit_rows": low.layout.useful,
                   "ports": list(low.definition.port_words), "shared_rows": head.get("shared_rows"), "ands": low.layout.counts["ands"],
                   "k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"], "circuit_sha512": circuit_sha512,
                   "class_sha512": class_sha512, "packed": True, "outputs": want[:, np.argsort(u.outputs)].astype(int).tolist()}
    return res

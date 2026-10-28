"""A unit circuit and its layout without a Python object per gate, for `fast_stage`: the same rows, netlist text and values as
`fast_stage.ands_layout` and `ir_lower.evaluate_layout`, in a few bytes of memory per AND.

    - `CompactCircuit` (`class_statement._UnitCircuit` under fast_stage) spools each AND's operand bits to disk as int32, in
      blocks of `FLUSH` entries, and keeps per gate its kind and its value on the all-zero instance, one byte each
    - `layout` finds the live gates in one pass over the blocks, last first (an AND only reads earlier gates, so a block's
      dead rows are those its later blocks and its own live rows never read), and numbers the live ANDs' columns
    - a `CompactLayout`'s netlist is rendered from the spool a block at a time, with numpy, each time it is read: `fast_stage`
      hashes it once when `compose` asks for the unit's sha256, and writes it once with the circuit, never to a scratch copy
    - `evaluate` on the all-zero instance (`compose`'s dummy) reads the values kept per gate; any other instance runs the rows
      block by block, as `ir_lower.evaluate_layout` does
    - `lower` is `ir_lower.lower` dropping each wire at its last read: the same gates in the same order, holding only the
      words still to be read

A circuit with a gate other than inputs and ANDs, or an input after its first AND, is laid out by the library: its gates are
read back from the spool as `_UnitCircuit` held them."""
from __future__ import annotations

import atexit
import bisect
import os
import shutil
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from operator import xor
from pathlib import Path

import numpy as np

from verity.ml.circuits.boolean import forms
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.evaluation.reference import resolve as resolve_target
from verity.ml.circuits.layout import param_leaf, resolve
from verity_flock import ir_lower as IL

FLUSH = 1 << 21
ROWS = 1 << 16
KINDS = ("in", "and", "hint", "lookup")
_CODE = {k: i for i, k in enumerate(KINDS)}
_PAIRS = np.frombuffer(b"".join(b"%02d" % i for i in range(100)), dtype=np.uint16)
_POW10 = [np.uint32(10 ** i) for i in range(1, 10)]
#: per digit count, which of a token's 12 rendered bytes (10 digits, its separator, one spare) it keeps
_KEEP = np.array([[j >= 10 - d for j in range(10)] + [True, False] for d in range(11)], dtype=bool)
_root: list[Path] = []
_spools: list["Spool"] = []


def set_root(path: Path) -> None:
    """Spools go under ``path`` (else the system's temporary directory)."""
    _root[:] = [Path(path)]


def cleanup() -> None:
    """Delete every spool this process made."""
    for s in _spools:
        s.remove()
    _spools.clear()


atexit.register(cleanup)


class _Wires(dict):
    """`ir_lower.lower_gates`' wires, each word dropped at its last read: ``left`` holds a key's reads still to come, and a key it
    doesn't hold is never dropped."""

    __slots__ = ("left",)

    def __getitem__(self, k):
        w = dict.__getitem__(self, k)
        n = self.left.get(k)
        if n is not None:
            if n == 1:
                del self.left[k]
                dict.__delitem__(self, k)
            else:
                self.left[k] = n - 1
        return w


def _reads(call, present: set) -> dict:
    """Each wire's reads in `ir_lower.lower_gates` over ``call`` (a region) from the wires ``present``, walked as it walks: an
    activation held whole when its gates are in the region and every parameter leaf is wired (reading each leaf once,
    wiring the returns inside it), else each gate reading its operands."""
    i, end, gate = call.lo, call.hi, call.circuit.gate
    left: dict = {}
    whole: dict = {}
    while i < end:
        g = gate(i)
        act = IL._boolean_activation(g)
        if act is not None:
            s = act[0]
            lo, hi = s.offset, s.offset + s.spec.gates
            if lo not in whole:
                whole[lo] = call.lo <= lo and hi <= call.hi
            if whole[lo]:
                params, _ = s.spec.signature()
                leaves = [param_leaf(s, pi, j) for pi, (_, t) in enumerate(params) for j in range(t.leaves)]
                whole[lo] = all(o in present for o in leaves)
                if whole[lo]:
                    for o in leaves:
                        left[o] = left.get(o, 0) + 1
                    for r in s.spec.body.ret:
                        j = resolve(s, r)
                        if lo <= j < hi:
                            present.add(j)
                    i += hi - lo
                    continue
        i += 1
        if getattr(g.prim, "family", None) == "Input":
            continue
        for o in g.operands:
            left[o] = left.get(o, 0) + 1
        present.add(g.index)
    return left


def lower(C, target, inputs) -> list[list]:
    """`ir_lower.lower`, each wire dropped once `lower_gates` has read it for the last time (`_reads`; a read past that is a
    KeyError, never other gates). A row's chain of residue adds otherwise keeps every partial sum's forms to the end."""
    t = resolve_target(target)
    if isinstance(t, PrimitiveDefinition):
        return [IL.lower_gates(C, [IL._PrimGate(t, len(inputs))], {-(i + 1): list(w) for i, w in enumerate(inputs)})[0]]
    _prog, call = IL.standalone(t)
    wires = _Wires(zip(call.args, inputs))
    left = _reads(call, set(wires))
    for r in call.returns:
        left.pop(r, None)
    wires.left = left
    IL.lower_gates(C, call, wires)
    return [dict.__getitem__(wires, r) for r in call.returns]


class Spool:
    """The ANDs' operands on disk, in blocks: ``ops`` (int32, each operand's bits), ``lens`` (int32, each AND's two operand
    sizes) and ``consts`` (uint8, bit 0 A's constant and bit 1 B's); ``blocks`` holds each block's first and end AND and first
    and end operand bit."""

    def __init__(self):
        root = _root[0] if _root else None
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix=".spool-", dir=root))
        self.paths = {k: self.dir / f"{k}.bin" for k in ("ops", "lens", "consts")}
        self.files = {k: open(p, "wb") for k, p in self.paths.items()}
        self.blocks: list[tuple[int, int, int, int]] = []
        self.ands = self.bits = 0
        _spools.append(self)

    def append(self, ops: np.ndarray, lens: np.ndarray, consts: np.ndarray) -> None:
        m = len(consts)
        self.blocks.append((self.ands, self.ands + m, self.bits, self.bits + len(ops)))
        for k, a in (("ops", ops), ("lens", lens), ("consts", consts)):
            a.tofile(self.files[k])
        self.ands += m
        self.bits += len(ops)

    def close(self) -> None:
        for f in self.files.values():
            f.close()

    def read(self, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, hi, o0, o1 = self.blocks[b]
        return (np.fromfile(self.paths["ops"], np.int32, o1 - o0, offset=4 * o0),
                np.fromfile(self.paths["lens"], np.int32, 2 * (hi - lo), offset=8 * lo),
                np.fromfile(self.paths["consts"], np.uint8, hi - lo, offset=lo))

    def remove(self) -> None:
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)


class CompactCircuit(forms.Circuit):
    """`class_statement._UnitCircuit` without its per-gate lists: ``kind``, ``data`` and ``phase_of`` are read back from the
    spool on first read (after which the circuit keeps them as `_UnitCircuit` does)."""

    def __init__(self, cse: bool = False):
        self.lookups = []
        self.cse = {} if cse else None
        self.inputs, self.outputs, self.asserts = {}, {}, []
        self.phase = "misc"
        self._n = 0
        self._kinds = bytearray()
        self._other: dict[int, object] = {}
        self._phases: list[tuple[int, str]] = []
        self._zero = bytearray()
        self._ops: list[int] = []
        self._lens: list[int] = []
        self._consts = bytearray()
        self._first_and = None
        self._late_input = False
        self._spool: Spool | None = None
        self._plain = None

    def _new(self, kind, data):
        if self._plain is not None:
            if kind == "and":
                a, b = data
                data = (forms.L(tuple(a.s), a.c), forms.L(tuple(b.s), b.c))
            for lst, x in zip(self._plain, (kind, data, self.phase)):
                lst.append(x)
            self._n += 1
            return forms.L(frozenset((self._n - 1,)), 0)
        v = self._n
        self._n = v + 1
        if not self._phases or self._phases[-1][1] is not self.phase:
            self._phases.append((v, self.phase))
        if kind == "and":
            a, b = data
            sa, sb = a.s, b.s
            ops = self._ops
            ops.extend(sa)
            ops.extend(sb)
            self._lens += (len(sa), len(sb))
            self._consts.append((a.c & 1) | (b.c & 1) << 1)
            z = self._zero
            get = z.__getitem__
            z.append((a.c + sum(map(get, sa))) & (b.c + sum(map(get, sb))) & 1)
            self._kinds.append(1)
            if self._first_and is None:
                self._first_and = v
            if len(ops) >= FLUSH:
                self._flush()
        else:
            self._kinds.append(_CODE[kind])
            self._other[v] = data
            self._zero.append(0)
            if kind == "in" and self._first_and is not None:
                self._late_input = True
        return forms.L(frozenset((v,)), 0)

    def _flush(self) -> None:
        if self._consts:
            if self._n >= 1 << 31:
                raise OverflowError("compact_stage: a unit of 2^31 gates or more")
            if self._spool is None:
                self._spool = Spool()
            self._spool.append(np.fromiter(self._ops, np.int32, len(self._ops)), np.array(self._lens, dtype=np.int32),
                               np.frombuffer(bytes(self._consts), dtype=np.uint8))
        self._ops, self._lens, self._consts = [], [], bytearray()

    def compact(self) -> bool:
        """Whether `layout` lays this circuit out: inputs, then ANDs, and nothing read back."""
        return self._plain is None and not self._late_input and set(self._kinds) <= {0, 1}

    def _materialize(self) -> tuple[list, list, list]:
        if self._plain is None:
            self._flush()
            kind = [KINDS[k] for k in self._kinds]
            data = [self._other.get(v) for v in range(self._n)]
            ands = [v for v, k in enumerate(self._kinds) if k == 1]
            if self._spool is not None:
                self._spool.close()
                for b in range(len(self._spool.blocks)):
                    ops, lens, consts = self._spool.read(b)
                    lo = self._spool.blocks[b][0]
                    ops, ends = ops.tolist(), np.cumsum(lens).tolist()
                    at = 0
                    for j in range(len(consts)):
                        ea, eb, c = ends[2 * j], ends[2 * j + 1], int(consts[j])
                        data[ands[lo + j]] = (forms.L(tuple(ops[at:ea]), c & 1), forms.L(tuple(ops[ea:eb]), c >> 1))
                        at = eb
                self._spool.remove()
                self._spool = None
            runs = self._phases + [(self._n, None)]
            phase_of = [p for (v, p), (w, _) in zip(runs, runs[1:]) for _ in range(w - v)]
            self._plain = (kind, data, phase_of)
            self._kinds = self._zero = None
        return self._plain

    kind = property(lambda self: self._materialize()[0], lambda self, v: self._set(0, v))
    data = property(lambda self: self._materialize()[1], lambda self, v: self._set(1, v))
    phase_of = property(lambda self: self._materialize()[2], lambda self, v: self._set(2, v))

    def _set(self, i: int, value) -> None:
        plain = list(self._materialize())
        plain[i] = value
        self._plain = tuple(plain)


def _aligned(n: int, nbits: int) -> tuple[int, int]:
    W = IL.WORD
    words = IL.ceil_pow2(max(1, -(-nbits // W)))
    return -(-n // (W * words)) * (W * words), words


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """The concatenated index ranges ``[starts[i], starts[i] + lens[i])``."""
    return np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(int(lens.sum()))


def _live_block(ops: np.ndarray, lens: np.ndarray, g0: int, alive: np.ndarray) -> np.ndarray:
    """Block's live ANDs (gates ``g0 ..``) given ``alive``'s marks from the later blocks and the outputs, marking the earlier
    gates its live ANDs read."""
    m = len(lens) // 2
    tot = lens[0::2] + lens[1::2]
    owner = np.repeat(np.arange(m, dtype=np.int32), tot)
    intra = ops >= g0
    t, ot = ops[intra] - g0, owner[intra]
    cnt = np.bincount(t, minlength=m)
    ext = alive[g0:g0 + m] != 0
    live = ext | (cnt > 0)
    dead = np.flatnonzero(~live)
    if dead.size:
        starts = np.concatenate([[0], np.cumsum(np.bincount(ot, minlength=m))])
        while dead.size:
            tt = t[_ranges(starts[dead], starts[dead + 1] - starts[dead])]
            if not tt.size:
                break
            cnt -= np.bincount(tt, minlength=m)
            cand = np.unique(tt)
            dead = cand[(cnt[cand] == 0) & live[cand] & ~ext[cand]]
            live[dead] = False
    alive[g0:g0 + m] = live
    alive[ops[~intra & np.repeat(live, tot)]] = 1
    return live


class _Rows:
    """A `CompactLayout`'s ``rows_a`` (``side`` 0) or ``rows_b`` (1), read off the spool."""

    def __init__(self, lay: "CompactLayout", side: int):
        self.lay, self.side = lay, side

    def __len__(self) -> int:
        return self.lay.n_rows

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        lay = self.lay
        i = i + len(self) if i < 0 else i
        if not 0 <= i < len(self):
            raise IndexError(i)
        if i < lay.first:
            return [i] if lay.head_in[i] else []
        k = i - lay.first
        if k < lay.n_alive:
            b = bisect.bisect_right(lay.live_lo, k) - 1
            return lay.block_rows(b)[self.side][k - lay.live_lo[b]]
        return lay.tail[self.side][k - lay.n_alive]

    def __iter__(self):
        lay = self.lay
        for i in range(lay.first):
            yield [i] if lay.head_in[i] else []
        for b in range(len(lay.spool.blocks)):
            yield from lay.block_rows(b)[self.side]
        yield from lay.tail[self.side]


class CompactLayout(IL.Layout):
    """`fast_stage.ands_layout`'s layout of a `CompactCircuit`: the input region's copy rows (``head_in``), the live ANDs' rows
    in the spool's blocks, then the assert, output and constant rows (``tail``, lists as the library's)."""

    def __init__(self, C: CompactCircuit, spool: Spool, alive: np.ndarray, col: np.ndarray, live_lo: list[int], first: int,
                 head_in: np.ndarray, tail: tuple[list, list], const: int, in_groups, out_groups, counts: dict):
        self.spool, self.alive, self.col, self.live_lo = spool, alive, col, live_lo
        self.n_in, self.first, self.head_in, self.tail = len(alive) - spool.ands, first, head_in, tail
        self.n_alive = live_lo[-1] if live_lo else 0
        self.n_rows = first + self.n_alive + len(tail[0])
        self.zero, self.outputs, self.asserts = C._zero, C.outputs, C.asserts
        self._cache: tuple[int, tuple] | None = None
        super().__init__(_Rows(self, 0), _Rows(self, 1), const, in_groups, out_groups, counts)

    def _block(self, b: int):
        """Block ``b``'s live rows: each operand's columns sorted (``cols``), the operands' sizes (``seg``, A then B per row)
        and constant flags (``cf``)."""
        ops, lens, consts = self.spool.read(b)
        lo = self.spool.blocks[b][0]
        live = self.alive[self.n_in + lo:self.n_in + lo + len(consts)].view(bool)
        seg_live = np.repeat(live, 2)
        seg = lens[seg_live].astype(np.int64)
        cols = self.col[ops[np.repeat(seg_live, lens)]].astype(np.int64)
        key = np.repeat(np.arange(len(seg), dtype=np.int64) << 32, seg) | cols
        key.sort()
        cols = key & 0xFFFFFFFF
        c = consts[live]
        cf = np.empty(len(seg), dtype=np.int64)
        cf[0::2], cf[1::2] = c & 1, c >> 1
        return cols, seg, cf

    def block_rows(self, b: int) -> tuple[list, list]:
        if self._cache is None or self._cache[0] != b:
            cols, seg, cf = self._block(b)
            parts = np.split(cols, np.cumsum(seg)[:-1]) if len(seg) else []
            rows = [p.tolist() + ([self.const] if f else []) for p, f in zip(parts, cf.tolist())]
            self._cache = (b, (rows[0::2], rows[1::2]))
        return self._cache[1]

    def block_text(self, b: int) -> np.ndarray:
        """Block ``b``'s netlist lines (``|A| A.. |B| B..``), as bytes."""
        cols, seg, cf = self._block(b)
        if not len(seg):
            return np.zeros(0, dtype=np.uint8)
        ntok = 1 + seg + cf
        start = np.cumsum(ntok) - ntok
        tok = np.empty(int(ntok.sum()), dtype=np.int64)
        tok[start] = seg + cf
        tok[_ranges(start + 1, seg)] = cols
        tok[(start + 1 + seg)[cf == 1]] = self.const
        return render(tok, (start + ntok - 1)[1::2])

    def pieces(self, name: str, workers: int | None = None):
        """``ir_lower.netlist(lay, name)``'s text, as bytes in pieces."""
        head = [IL.FORMAT, name, self.n_rows, self.const]
        for groups in (self.in_groups, self.out_groups):
            head.append(len(groups))
            for c, words, bits in groups:
                head += [c, words, len(bits), *bits]
        yield (" ".join(map(str, head)) + "\n").encode()
        for k in range(0, self.first, ROWS):
            yield ("\n".join(f"1 {i} 1 {i}" if h else "0 0" for i, h in zip(range(k, k + ROWS), self.head_in[k:k + ROWS].tolist()))
                   + "\n").encode()
        n = len(self.spool.blocks)
        workers = workers or min(4, os.cpu_count() or 1)
        with ThreadPoolExecutor(workers) as pool:
            ahead = deque(pool.submit(self.block_text, b) for b in range(min(n, 2 * workers)))
            for b in range(len(ahead), n + len(ahead)):
                text = ahead.popleft().result()
                if b < n:
                    ahead.append(pool.submit(self.block_text, b))
                yield memoryview(text)
        ta, tb = self.tail
        for k in range(0, len(ta), ROWS):
            yield ("\n".join(" ".join(map(str, [len(a), *a, len(b), *b])) for a, b in zip(ta[k:k + ROWS], tb[k:k + ROWS]))
                   + "\n").encode()

    def evaluate(self, ports, T: int, flip=None):
        """`ir_lower.evaluate_layout`."""
        mask = (1 << T) - 1
        if flip is None and not any(np.any(np.asarray(p, dtype=np.uint64)) for p in ports):
            return self._zero_instance(T, mask)
        z = [0] * self.n_rows
        for at, width, words in zip(self.in_port_cols, self.in_bits, ports):
            z[at:at + width] = IL.slice_words(words, width)
        z[self.const] = mask
        get = z.__getitem__
        ev = lambda r: reduce(xor, map(get, r), 0)  # noqa: E731
        i = self.first
        for b in range(len(self.spool.blocks)):
            for a, bb in zip(*self.block_rows(b)):
                z[i] = ev(a) & ev(bb)
                i += 1
        for a, bb in zip(*self.tail):
            if i != self.const:
                z[i] = ev(a) & ev(bb)
            i += 1
        if flip is not None:
            z[flip[0]] ^= 1 << flip[1]
        ok = mask
        for a, bb, zi in zip(self.rows_a, self.rows_b, z):
            ok &= ~((ev(a) & ev(bb)) ^ zi) & mask
        outs = [IL.unslice_words(z[at:at + nb], T) for at, nb in zip(self.out_port_cols, self.out_bits)]
        return outs, ok

    def _zero_instance(self, T: int, mask: int):
        """The all-zero instance: every gate's value is the one `CompactCircuit` kept, an output row's its form's, and an
        assert row holds when its form is zero."""
        get = self.zero.__getitem__
        val = lambda w: (w.c + sum(map(get, w.s))) & 1  # noqa: E731
        ok = mask
        for _, w in self.asserts:
            if val(w):
                ok = 0
        names = [nm for g in self.out_names for nm in g]
        outs = [IL.unslice_words([mask * val(w) for w in self.outputs[nm]], T) for nm in names]
        return outs, ok


def render(tok: np.ndarray, last: np.ndarray) -> np.ndarray:
    """The non-negative ``tok`` (each below 2^32) in decimal, each followed by a space, or a newline for the tokens at
    ``last``, as bytes."""
    v = tok.astype(np.uint32)
    d = np.ones(len(v), dtype=np.uint8)
    for p in _POW10:
        d += v >= p
    out = np.empty((len(v), 12), dtype=np.uint8)
    pairs = out.view(np.uint16)
    for j in range(4, -1, -1):
        q = v // np.uint32(100)
        pairs[:, j] = _PAIRS[v - q * np.uint32(100)]
        v = q
    out[:, 10] = 32
    out[last, 10] = 10
    return out[_KEEP[d]]


def layout(C: CompactCircuit, in_names, out_names, in_split=None, out_split=None) -> CompactLayout:
    """`fast_stage.ands_layout(C, ...)` of a circuit `CompactCircuit.compact` accepts."""
    W = IL.WORD
    C._flush()
    spool = C._spool or Spool()
    C._spool = None
    spool.close()
    n = C._n
    n_in = n - spool.ands
    alive = np.zeros(n, dtype=np.uint8)
    alive[:n_in] = 1
    for w in [w for bits in C.outputs.values() for w in bits] + [w for _, w in C.asserts]:
        alive[list(w.s)] = 1
    nnz = [0, 0]
    for b in range(len(spool.blocks) - 1, -1, -1):
        ops, lens, consts = spool.read(b)
        live = _live_block(ops, lens, n_in + spool.blocks[b][0], alive)
        nnz[0] += int(lens[0::2][live].sum()) + int(np.count_nonzero(consts[live] & 1))
        nnz[1] += int(lens[1::2][live].sum()) + int(np.count_nonzero(consts[live] & 2))
    col = np.full(n, -1, dtype=np.int32)
    in_groups, at = [], 0
    for g in IL._groups(in_names, in_split):
        bits = [len(C.inputs[nm]) for nm in g]
        at, words = _aligned(at, sum(bits))
        i = at
        for nm in g:
            for w in C.inputs[nm]:
                (v,) = tuple(w.s)
                col[v] = i
                i += 1
        at += words * W
        in_groups.append((at // W - words, words, bits))
    first = at
    head_in = np.zeros(first, dtype=bool)
    head_in[col[:n_in][col[:n_in] >= 0]] = True
    if not head_in.sum() == n_in:
        raise ValueError("compact_stage: an input outside the input ports")
    np.cumsum(alive[n_in:], dtype=np.int32, out=col[n_in:])
    col[n_in:] += first - 1
    live_lo = [0]
    for lo, hi, _, _ in spool.blocks:
        live_lo.append(int(col[n_in + hi - 1]) - first + 1 if hi > lo else live_lo[-1])
    n_alive = live_lo[-1]
    nn = first + n_alive + len(C.asserts)
    out_at = []
    for g in IL._groups(out_names, out_split):
        bits = [len(C.outputs[nm]) for nm in g]
        a, words = _aligned(nn, sum(bits))
        out_at.append((g, a, words, bits))
        nn = a + words * W
    const = nn
    ta: list = []
    tb: list = []

    def form(Lf) -> list:
        r = sorted(int(col[v]) for v in Lf.s)
        if Lf.c & 1:
            r.append(const)
        return r

    row = first + n_alive
    for _, Lf in C.asserts:
        ta.append(sorted(form(Lf) + [row + len(ta)]))
        tb.append([const])
    out_groups = []
    for g, a, words, bits in out_at:
        while row + len(ta) < a:
            ta.append([])
            tb.append([])
        for nm in g:
            for w in C.outputs[nm]:
                ta.append(form(w))
                tb.append([const])
        while row + len(ta) < a + words * W:
            ta.append([])
            tb.append([])
        out_groups.append((a // W, words, bits))
    assert row + len(ta) == const
    ta.append([const])
    tb.append([const])
    nnz[0] += n_in + sum(map(len, ta))
    nnz[1] += n_in + sum(map(len, tb))
    lay = CompactLayout(C, spool, alive, col, live_lo, first, head_in, (ta, tb), const, in_groups, out_groups, {})
    lay.out_names = [g for g, *_ in out_at]
    lay.counts = {"ands": n_alive, "asserts": len(C.asserts), "rows": lay.n_rows, "in_words": lay.in_words,
                  "out_words": lay.out_words, "nnz_a": nnz[0], "nnz_b": nnz[1], "reads": 0, "read_rows": 0, "read_nnz_a": 0}
    C._kinds = None
    return lay

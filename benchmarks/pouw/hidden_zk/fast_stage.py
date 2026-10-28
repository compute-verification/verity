"""`class_statement.stage_unit` in less time and memory, writing the same bytes: `install(scratch)` patches the library for
one process (stage.py's); `composed()`, `size()` and `release()` hand stage.py the circuit it composed, so it never reads
the circuit back.

    - the cycle collector is off while a unit's gates are lowered (`ir_lower.lower_gates`): they make no cyclic garbage (a
      K=256 row unit's 30k collections freed 325 objects), and its collections were about 30% of the time at K=256, more
      at K=4096. It stays on elsewhere: building the Programs and tracing the IR leave cyclic garbage, 0.9 GiB at a lane
    - the unit circuit's ANDs keep their operand forms as tuples, not frozensets, and drop them once the unit is laid out:
      nothing reads them after `ir_lower.layout`
    - the unit's netlist (`ir_lower.netlist`'s text) is written once, to a file under ``scratch``, hashed as it is written:
      `compose` builds it as one string, which `Lowering.sha256_of` hashes with the LEAVES and CUT lines. `compose` gets
      in its place a string of as many newlines (and `sha256_of` the file's hash), and the circuit text it returns stands
      for that text with the file in the newlines' place: its digests (`circuit.digest`, `circuit.class_digest`) and
      circuit.txt are streamed from the parts, both digests in the pass that writes the file
    - any other text's SHA-512 / SHA-256 goes in pieces of 64 Mi characters, each computed once
    - the unit's layout (`ands_layout`) sorts each form's columns in one call, and `ir_lower.evaluate_layout` XORs a row's
      columns with `functools.reduce`: the same rows and values
    - a flat unit of inputs and ANDs (`unit_lowering`'s inline, non-structural one) is built as a `StreamCircuit`, which
      keeps no AND in memory: each goes to a spool under ``scratch`` as it is made, with its value bit on the all-zero
      instance (all `compose`'s dummy reads), and `StreamLayout` lays the unit out from the spool (`ands_layout`'s rows: a
      backward pass for liveness, then the netlist written block by block). It holds two bytes a bit while it lays out,
      none after; a unit that reads a table, makes a hint or takes an input after an AND is built again the library's way

stage.py runs the library unpatched under `HZ_STAGE=library`; both write the same bytes."""
from __future__ import annotations

import atexit
import gc
import hashlib
import inspect
import io
import json
import shutil
import sys
import tempfile
from array import array
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from operator import xor
from pathlib import Path

import numpy as np

from verity.ml.circuits.boolean import forms
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL

CHUNK = 1 << 26
ROWS = 1 << 16
#: the ANDs a spool block holds: its operand bits are about 140 MB as int64
BLOCK = 1 << 19
_scratch: list[Path] = []
_spools: list[Path] = []
_memo: dict[int, tuple[str, dict]] = {}
_unit_lays: dict[int, object] = {}
_units: dict[tuple[int, str], dict] = {}
_streams: dict[int, dict] = {}
_in_unit: list = []
_lowering: list = []
_composed: list = []


def _update(h, text: str, start: int = 0, stop: int | None = None):
    stop = len(text) if stop is None else stop
    for i in range(start, stop, CHUNK):
        h.update(text[i:min(i + CHUNK, stop)].encode())
    return h


def _of(text: str) -> dict:
    e = _memo.get(id(text))
    if e is None or e[0] is not text:
        e = _memo[id(text)] = (text, {})
    return e[1]


def _class_head(text: str) -> tuple[bytes, int]:
    """`circuit.class_digest`'s first two lines (META without its binding keys), and where the rest starts."""
    i = text.index("\n")
    j = text.index("\n", i + 1)
    meta = {k: v for k, v in json.loads(text[i + 6:j]).items() if k not in FC.BINDING_KEYS}
    return (text[:i] + "\nMETA " + json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n").encode(), j + 1


def _stream(text: str, out=None) -> dict:
    """A composed text's file digest and class digest, in one pass over its parts, writing them to ``out`` too."""
    s = _streams[id(text)]
    head, j = _class_head(text)
    h_file, h_class = hashlib.sha512(), hashlib.sha512(head)
    with ThreadPoolExecutor(2) as pool:
        def put(b: bytes, file: bool = True, cls: bool = True):
            jobs = [pool.submit(h.update, b) for h, on in ((h_file, file), (h_class, cls)) if on]
            if out is not None:
                out.write(b)
            for f in jobs:
                f.result()

        put(text[:j].encode(), cls=False)
        put(text[j:s["at"]].encode())
        with open(s["path"], "rb") as f:
            while b := f.read(CHUNK):
                put(b)
        put(text[s["at"] + s["newlines"]:].encode())
    return {"sha512": h_file.hexdigest(), "class": h_class.hexdigest()}


def digest(text: str) -> str:
    """`circuit.digest`."""
    m = _of(text)
    if "sha512" not in m:
        if id(text) in _streams:
            m.update(_stream(text))
        else:
            m["sha512"] = _update(hashlib.sha512(), text).hexdigest()
    return m["sha512"]


def class_digest(text: str) -> str:
    """`circuit.class_digest`."""
    m = _of(text)
    if "class" not in m:
        if id(text) in _streams:
            m.update(_stream(text))
        else:
            head, j = _class_head(text)
            m["class"] = _update(hashlib.sha512(head), text, j).hexdigest()
    return m["class"]


def sha256(text: str) -> str:
    """`ir_lower.digest`."""
    return _update(hashlib.sha256(), text).hexdigest()


def _netlist(lay, name: str):
    """`ir_lower.netlist(lay, name)`'s text in pieces."""
    if isinstance(lay, StreamLayout):
        yield from lay.pieces(name)
        return
    head = [IL.FORMAT, name, lay.useful, lay.const]
    for groups in (lay.in_groups, lay.out_groups):
        head.append(len(groups))
        for c, words, bits in groups:
            head += [c, words, len(bits), *bits]
    yield " ".join(map(str, head)) + "\n"
    cols, rows = IL._cols_of, []
    for a, b in zip(lay.rows_a, lay.rows_b):
        rows.append(" ".join(map(str, [len(a), *cols(a), len(b), *cols(b)])))
        if len(rows) == ROWS:
            yield "\n".join(rows) + "\n"
            rows = []
    if rows:
        yield "\n".join(rows) + "\n"


def _unit(lay, name: str) -> dict:
    """The unit's expanded netlist, written once to a scratch file: its path, newlines, bytes and sha256 state."""
    e = _units.get((id(lay), name))
    if e is None or e["lay"] is not lay:
        path = _scratch[0] / f".unit-{len(_units)}.netlist"
        h, n, size = hashlib.sha256(), 0, 0
        with open(path, "wb") as f:
            for piece in _netlist(lay, name):
                b = piece.encode()
                h.update(b)
                f.write(b)
                n += piece.count("\n")
                size += len(b)
        e = _units[(id(lay), name)] = {"lay": lay, "path": path, "newlines": n, "bytes": size, "sha256": h}
    return e


def _lowering_sha256(self) -> str:
    d = self.__dict__.get("_sha256")
    if d is None:
        tail = "LEAVES " + json.dumps(self.leaves, sort_keys=True, separators=(",", ":")) + "\n"
        cut = self.cut
        if cut is not None:
            tail += "CUT " + json.dumps(cut, sort_keys=True, separators=(",", ":")) + "\n"
        if id(self.layout) in _unit_lays:
            h = _unit(self.layout, self.name)["sha256"].copy()
        else:
            h = _update(hashlib.sha256(), FC._expanded(self.layout, self.name))
        h.update(tail.encode())
        d = self.__dict__["_sha256"] = h.hexdigest()
    return d


def _lowering_sha256_of(self, expanded: str) -> str:
    """`ir_lower.Lowering.sha256_of`, whose ``expanded`` is the newlines `compose` got in the unit netlist's place."""
    return self.sha256


def ands_layout(C, in_names, out_names, in_split=None, out_split=None):
    """`ir_lower.layout` of a circuit of inputs and ANDs only: the same rows, each form's columns sorted in one call (every
    form's bits are distinct and laid out, so the library's parity pass keeps them all) and the constant column, the last,
    appended."""
    W = IL.WORD
    live = C.live()
    col = [-1] * len(C.kind)
    rows_a: list = []
    rows_b: list = []

    def aligned(n: int, nbits: int) -> tuple[int, int]:
        words = IL.ceil_pow2(max(1, -(-nbits // W)))
        return -(-n // (W * words)) * (W * words), words

    def pad(to: int) -> None:
        rows_a.extend([] for _ in range(to - len(rows_a)))
        rows_b.extend([] for _ in range(to - len(rows_b)))

    in_groups = []
    for g in IL._groups(in_names, in_split):
        bits = [len(C.inputs[nm]) for nm in g]
        at, words = aligned(len(rows_a), sum(bits))
        pad(at)
        for nm in g:
            for w in C.inputs[nm]:
                (v,) = tuple(w.s)
                col[v] = i = len(rows_a)
                rows_a.append([i])
                rows_b.append([i])
        pad(at + words * W)
        in_groups.append((at // W, words, bits))
    ands = [v for v, kind in enumerate(C.kind) if kind == "and" and v in live]
    first = len(rows_a)
    for i, v in enumerate(ands, first):
        col[v] = i
    n = first + len(ands) + len(C.asserts)
    out_at = []
    for g in IL._groups(out_names, out_split):
        bits = [len(C.outputs[nm]) for nm in g]
        at, words = aligned(n, sum(bits))
        out_at.append((g, at, words, bits))
        n = at + words * W
    const = n
    get = col.__getitem__

    def form(Lf) -> list:
        r = sorted(map(get, Lf.s))
        if Lf.c & 1:
            r.append(const)
        return r

    data = C.data
    for v in ands:
        a, b = data[v]
        rows_a.append(form(a))
        rows_b.append(form(b))
    for _, Lf in C.asserts:
        rows_a.append(sorted(form(Lf) + [len(rows_a)]))
        rows_b.append([const])
    out_groups = []
    for g, at, words, bits in out_at:
        pad(at)
        for nm in g:
            for w in C.outputs[nm]:
                rows_a.append(form(w))
                rows_b.append([const])
        pad(at + words * W)
        out_groups.append((at // W, words, bits))
    assert len(rows_a) == const
    rows_a.append([const])
    rows_b.append([const])
    lay = IL.Layout(rows_a, rows_b, const, in_groups, out_groups, {})
    lay.counts = {"ands": len(ands), "asserts": len(C.asserts), "rows": len(rows_a), "in_words": lay.in_words,
                  "out_words": lay.out_words, "nnz_a": sum(map(len, rows_a)), "nnz_b": sum(map(len, rows_b)), "reads": 0,
                  "read_rows": 0, "read_nnz_a": 0}
    return lay


class _NotStreamed(Exception):
    """A unit `StreamCircuit` doesn't lay out: one with a table read or a hint, an input after an AND, or an input no port
    lays out."""


class _Spool:
    """A unit's ANDs in creation order, in two files under a directory of ``scratch``: per AND its sides' heads
    ``2 |s| + c`` (int32 pairs) and its sides' bits, side A's then side B's (int32), in blocks of `BLOCK` ANDs."""

    def __init__(self, scratch: Path):
        self.dir = Path(tempfile.mkdtemp(prefix=".spool-", dir=scratch))
        _spools.append(self.dir)
        self.files = [open(self.dir / "heads", "wb"), open(self.dir / "ids", "wb")]
        self.heads, self.ids = array("i"), array("i")
        self.blocks: list[tuple[int, int, int, int]] = []   # (first AND, ANDs, first bit, bits)
        self.ands = self.bits = 0

    def flush(self) -> None:
        n, m = len(self.heads) // 2, len(self.ids)
        if n:
            self.heads.tofile(self.files[0])
            self.ids.tofile(self.files[1])
            self.blocks.append((self.ands, n, self.bits, m))
            self.ands, self.bits = self.ands + n, self.bits + m
            self.heads, self.ids = array("i"), array("i")

    def close(self) -> None:
        self.flush()
        for f in self.files:
            f.close()

    def block(self, i: int) -> tuple[int, np.ndarray, np.ndarray]:
        """Block i's first AND, its heads ``(ANDs, 2)`` and its bits."""
        a0, n, b0, m = self.blocks[i]
        heads = np.fromfile(self.dir / "heads", np.int32, 2 * n, offset=8 * a0).reshape(n, 2)
        return a0, heads, np.fromfile(self.dir / "ids", np.int32, m, offset=4 * b0)


class StreamCircuit(forms.Circuit):
    """`class_statement.unit_circuit`'s circuit for `unit_lowering`'s flat unit, keeping no AND: each goes to a `_Spool` as
    it is made, and its value on the all-zero instance to ``zero`` (a byte a bit). It has no ``kind`` or ``data``, so code
    that reads them fails instead of reading nothing."""

    def __init__(self, cse: bool = False):
        super().__init__(cse)
        del self.kind, self.data, self.phase_of
        self.n = self.n_in = 0
        self.zero = bytearray()
        self.spool = _Spool(_scratch[0])

    def _new(self, kind, data):
        v, z = self.n, self.zero
        if kind == "and":
            a, b = data
            x = a.c
            for u in a.s:
                x ^= z[u]
            if x:
                x = b.c
                for u in b.s:
                    x ^= z[u]
            z.append(x)
            sp = self.spool
            sp.heads.append(2 * len(a.s) + a.c)
            sp.heads.append(2 * len(b.s) + b.c)
            sp.ids.extend(a.s)
            sp.ids.extend(b.s)
            if len(sp.heads) >= 2 * BLOCK:
                sp.flush()
        elif kind == "in" and v == self.n_in:
            self.n_in += 1
            z.append(0)
        else:
            raise _NotStreamed(f"a {kind} bit at {v}")
        self.n = v + 1
        return forms.L(frozenset((v,)), 0)


class StreamLayout(IL.Layout):
    """`ands_layout` of a `StreamCircuit`, from its spool: the port groups, ``const``, ``useful`` and ``counts`` at once,
    the netlist in pieces (`pieces`, block by block) and the all-zero instance's outputs and verdict (`zero`). A dead AND
    is one no output, assertion or live AND reads (one backward pass), and an AND's column is its rank among the live
    ones: every form's bits are distinct and its columns ascend with its bits, so sorting a form's columns sorts its
    bits. ``rows_a`` and ``rows_b`` are built, from the netlist, only for a reader of them (a templated statement of the
    flat unit, an evaluation of other instances)."""

    def __init__(self, C: StreamCircuit, in_names, out_names, in_split=None, out_split=None):
        W = IL.WORD
        C.spool.close()
        self.spool, self.n_in, n = C.spool, C.n_in, C.n

        def aligned(at: int, nbits: int) -> tuple[int, int]:
            words = IL.ceil_pow2(max(1, -(-nbits // W)))
            return -(-at // (W * words)) * (W * words), words

        self.in_rows = bytearray()
        self.in_col = np.full(self.n_in, -1, np.int64)
        self.in_groups = []
        for g in IL._groups(in_names, in_split):
            bits = [len(C.inputs[nm]) for nm in g]
            at, words = aligned(len(self.in_rows), sum(bits))
            self.in_rows.extend(bytes(at - len(self.in_rows)))
            for nm in g:
                for w in C.inputs[nm]:
                    (v,) = tuple(w.s)
                    self.in_col[v] = len(self.in_rows)
                    self.in_rows.append(1)
            self.in_rows.extend(bytes(at + words * W - len(self.in_rows)))
            self.in_groups.append((at // W, words, bits))
        if (self.in_col < 0).any():
            raise _NotStreamed("an input no port lays out")
        self.in_ident = bool((self.in_col == np.arange(self.n_in)).all())
        self.first = len(self.in_rows)

        live = bytearray(n)
        live[:self.n_in] = b"\x01" * self.n_in
        for w in [w for bits in C.outputs.values() for w in bits] + [w for _, w in C.asserts]:
            for u in w.s:
                live[u] = 1
        nnz_a = nnz_b = 0
        for i in reversed(range(len(self.spool.blocks))):
            a0, heads, ids = self.spool.block(i)
            v0, ends = self.n_in + a0, np.concatenate([[0], np.cumsum((heads >> 1).sum(axis=1))])
            # `ROWS` ANDs' bits at a time as Python ints (36 bytes a bit): a whole block's, at a lane's 46 bits an AND at
            # K = 448 and more as K grows, are about 0.9 GB
            for j1 in range(len(heads), 0, -ROWS):
                j0 = max(0, j1 - ROWS)
                e, x = (ends[j0:j1 + 1] - ends[j0]).tolist(), ids[ends[j0]:ends[j1]].tolist()
                for j in range(j1 - j0 - 1, -1, -1):
                    if live[v0 + j0 + j]:
                        for u in x[e[j]:e[j + 1]]:
                            live[u] = 1
            m = np.frombuffer(live, np.uint8, len(heads), v0).astype(bool)
            side = (heads >> 1) + (heads & 1)
            nnz_a, nnz_b = nnz_a + int(side[m, 0].sum()), nnz_b + int(side[m, 1].sum())
        lv = np.frombuffer(live, np.uint8)
        self.dead = np.flatnonzero(lv[self.n_in:] == 0) + self.n_in
        del lv, live
        n_live = n - self.n_in - len(self.dead)

        j = self.first + n_live + len(C.asserts)
        out_at = []
        for g in IL._groups(out_names, out_split):
            bits = [len(C.outputs[nm]) for nm in g]
            at, words = aligned(j, sum(bits))
            out_at.append((g, at, words, bits))
            j = at + words * W
        self.const = const = j
        self._useful = const + 1
        self.out_groups = [(at // W, words, bits) for _g, at, words, bits in out_at]

        z = C.zero

        def value(w) -> int:
            x = w.c
            for u in w.s:
                x ^= z[u]
            return x

        def form(w) -> list[int]:
            cols = sorted(self._col(np.fromiter(w.s, np.int64, len(w.s))).tolist())
            return cols + [const] if w.c & 1 else cols

        self.tail, j = [], self.first + n_live
        for _, w in C.asserts:
            a = form(w)
            a.insert(len(a) - (w.c & 1), j)
            self.tail.append(a)
            j += 1
        self.zero_ok = not any(value(w) for _, w in C.asserts)
        self.zero_outs = []
        for g, at, words, bits in out_at:
            self.tail.extend([None] * (at - j))
            for nm in g:
                self.tail.extend(form(w) for w in C.outputs[nm])
                self.zero_outs.append([value(w) for w in C.outputs[nm]])
            j = at + words * W
            self.tail.extend([None] * (j - at - sum(bits)))
        C.zero = None
        assert self.first + n_live + len(self.tail) == const
        full = [a for a in self.tail if a is not None]
        self.counts = {"ands": n_live, "asserts": len(C.asserts), "rows": self._useful, "in_words": self.in_words,
                       "out_words": self.out_words, "nnz_a": self.n_in + nnz_a + sum(map(len, full)) + 1,
                       "nnz_b": self.n_in + nnz_b + len(full) + 1, "reads": 0, "read_rows": 0, "read_nnz_a": 0}
        self._rows_ab = None

    @property
    def useful(self) -> int:
        return self._useful

    @property
    def rows_a(self) -> list:
        return self._rows()[0]

    @property
    def rows_b(self) -> list:
        return self._rows()[1]

    def _rows(self) -> tuple[list, list]:
        if self._rows_ab is None:
            ra, rb = [], []
            pieces = self.pieces("")
            next(pieces)
            for piece in pieces:
                for line in piece.splitlines():
                    r = list(map(int, line.split()))
                    ra.append(r[1:1 + r[0]])
                    rb.append(r[2 + r[0]:])
            self._rows_ab = ra, rb
        return self._rows_ab

    def _col(self, x: np.ndarray) -> np.ndarray:
        """The columns of live bits ``x``."""
        x = x.astype(np.int64)
        out = np.empty_like(x)
        a = x >= self.n_in
        xa = x[a]
        out[a] = self.first + (xa - self.n_in) - np.searchsorted(self.dead, xa)
        out[~a] = x[~a] if self.in_ident else self.in_col[x[~a]]
        return out

    def pieces(self, name: str):
        """`ir_lower.netlist(self, name)`'s text in pieces of at most `ROWS` rows."""
        head = [IL.FORMAT, name, self.useful, self.const]
        for groups in (self.in_groups, self.out_groups):
            head.append(len(groups))
            for c, words, bits in groups:
                head += [c, words, len(bits), *bits]
        yield " ".join(map(str, head)) + "\n"
        for k in range(0, len(self.in_rows), ROWS):
            yield "".join(f"1 {i} 1 {i}\n" if self.in_rows[i] else "0 0\n" for i in range(k, min(k + ROWS, len(self.in_rows))))
        for i in range(len(self.spool.blocks)):
            yield from self._and_rows(i)
        C = self.const
        for k in range(0, len(self.tail), ROWS):
            yield "".join("0 0\n" if a is None else " ".join(map(str, [len(a), *a, 1, C])) + "\n" for a in self.tail[k:k + ROWS])
        yield f"1 {C} 1 {C}\n"

    def _and_rows(self, i: int):
        """Spool block i's live ANDs' rows, `ROWS` at a time: each side ``|cols| cols..`` with its columns ascending and the
        constant column last when its form's constant is 1."""
        a0, heads, ids = self.spool.block(i)
        v0, n = self.n_in + a0, len(heads)
        keep = np.ones(n, bool)
        lo, hi = np.searchsorted(self.dead, [v0, v0 + n])
        keep[self.dead[lo:hi] - v0] = False
        lens = heads >> 1
        ends = np.concatenate([[0], np.cumsum(lens.sum(axis=1))])
        for j0 in range(0, n, ROWS):
            j1 = min(j0 + ROWS, n)
            k, h, x = keep[j0:j1], lens[j0:j1], ids[ends[j0]:ends[j1]]
            c = heads[j0:j1] & 1
            if not k.all():
                x, h, c = x[np.repeat(k, h.sum(axis=1))], h[k], c[k]
            if not len(h):
                continue
            seglen, segc = h.reshape(-1).astype(np.int64), c.reshape(-1).astype(np.int64)
            seg = np.repeat(np.arange(len(seglen), dtype=np.int64), seglen)
            key = seg * (self.const + 1) + self._col(x)
            key.sort()
            cols = key - seg * (self.const + 1)
            segtok = 1 + seglen + segc
            rowtok = segtok[0::2] + segtok[1::2] + 1
            rstart = np.cumsum(rowtok) - rowtok
            sstart = np.empty(len(seglen), np.int64)
            sstart[0::2], sstart[1::2] = rstart, rstart + segtok[0::2]
            out = np.empty(int(rowtok.sum()), np.int64)
            out[sstart] = seglen + segc
            out[np.repeat(sstart + 1 - (np.cumsum(seglen) - seglen), seglen) + np.arange(len(cols))] = cols
            out[(sstart + 1 + seglen)[segc == 1]] = self.const
            out[rstart + rowtok - 1] = -1
            yield (" ".join(map(str, out.tolist())) + " ").replace(" -1 ", "\n")

    def zero(self, ports, T: int, flip=None):
        """`ir_lower.evaluate_layout` on ports all zero and no flip, from the all-zero instance's outputs and assertions;
        else None."""
        if flip is not None or any(np.any(np.asarray(p)) for p in ports):
            return None
        mask = (1 << T) - 1
        return [IL.unslice_words([mask if b else 0 for b in bits], T) for bits in self.zero_outs], mask if self.zero_ok else 0


def evaluate_layout(lay, ports, T, flip=None):
    """`ir_lower.evaluate_layout`."""
    if isinstance(lay, StreamLayout) and (got := lay.zero(ports, T, flip)) is not None:
        return got
    mask = (1 << T) - 1
    z = [0] * lay.useful
    for at, width, words in zip(lay.in_port_cols, lay.in_bits, ports):
        z[at:at + width] = IL.slice_words(words, width)
    z[lay.const] = mask
    get, cols, ra, rb = z.__getitem__, IL._cols_of, lay.rows_a, lay.rows_b
    for i in range(lay.in_words * IL.WORD, lay.useful):
        if i != lay.const:
            z[i] = reduce(xor, map(get, cols(ra[i])), 0) & reduce(xor, map(get, cols(rb[i])), 0)
    if flip is not None:
        z[flip[0]] ^= 1 << flip[1]
    ok = mask
    for i in range(lay.useful):
        ok &= ~((reduce(xor, map(get, cols(ra[i])), 0) & reduce(xor, map(get, cols(rb[i])), 0)) ^ z[i]) & mask
    outs = [IL.unslice_words(z[at:at + nb], T) for at, nb in zip(lay.out_port_cols, lay.out_bits)]
    return outs, ok


def install(scratch: Path) -> None:
    """Patch the library for this process; the unit's netlist goes under ``scratch`` until `release`. The slot circuits'
    caches (`cache_slot_circuits`) go first, so these wrap them."""
    scratch.mkdir(parents=True, exist_ok=True)
    _scratch[:] = [scratch]
    CS.cache_slot_circuits()
    new, lower, layout, expanded = forms.Circuit._new, IL.lower, IL.layout, FC._expanded
    lower_gates = IL.lower_gates
    unit_lowering, unit_circuit, compose, write_text = CS.unit_lowering, CS.unit_circuit, FC.compose, Path.write_text
    params = inspect.signature(unit_lowering)
    atexit.register(_drop_spools)

    def new_(self, kind, data):
        if kind == "and" and _lowering and self is _lowering[-1]:
            a, b = data
            data = (forms.L(tuple(a.s), a.c), forms.L(tuple(b.s), b.c))
        return new(self, kind, data)

    def unit_lowering_(*a, **k):
        arg = params.bind(*a, **k)
        arg.apply_defaults()
        _in_unit.append(not arg.arguments["structural"] and arg.arguments["reads"] == "inline")
        try:
            try:
                low, out_words = unit_lowering(*a, **k)
            except _NotStreamed as e:
                print(f"fast_stage: {arg.arguments['name']}: built again the library's way ({e})", file=sys.stderr, flush=True)
                shutil.rmtree(_spools.pop(), ignore_errors=True)
                _in_unit[-1] = False
                low, out_words = unit_lowering(*a, **k)
        finally:
            _in_unit.pop()
        if not hasattr(low, "template"):
            _unit_lays[id(low.layout)] = low.layout
        return low, out_words

    def unit_circuit_(definition, ports, cls=forms.Circuit):
        if _in_unit and _in_unit[-1] and cls is CS._UnitCircuit:
            cls = StreamCircuit
        return unit_circuit(definition, ports, cls)

    def lower_(C, *a, **k):
        if not _in_unit:
            return lower(C, *a, **k)
        C._compact = True
        _lowering.append(C)
        try:
            return lower(C, *a, **k)
        finally:
            _lowering.pop()

    def lower_gates_(C, *a, **k):
        if not (_lowering and C is _lowering[-1] and gc.isenabled()):
            return lower_gates(C, *a, **k)
        gc.disable()
        try:
            return lower_gates(C, *a, **k)
        finally:
            gc.enable()

    def layout_(C, *a, **k):
        if isinstance(C, StreamCircuit):
            return StreamLayout(C, *a, **k)
        compact = getattr(C, "_compact", False)
        lay = (ands_layout if compact and set(C.kind) <= {"in", "and"} else layout)(C, *a, **k)
        if compact:
            C.data = [None if kind == "and" else d for kind, d in zip(C.kind, C.data)]
        return lay

    def expanded_(lay, name):
        if _unit_lays.get(id(lay)) is lay:
            return "\n" * _unit(lay, name)["newlines"]
        return expanded(lay, name)

    def compose_(low, *a, **k):
        text = compose(low, *a, **k)
        e = _units.get((id(low.layout), low.name))
        if e is not None and text.startswith(f"{FC.FORMAT} "):
            j = text.index("\n", text.index("\n") + 1) + 1
            mark = f"CIRCUIT unit {e['newlines']}\n"
            at = j + len(mark)
            if text[j:at] != mark or text[at:at + e["newlines"]] != "\n" * e["newlines"] or not text.startswith("CIRCUIT ", at + e["newlines"]):
                raise AssertionError("compose: the unit's netlist is not where fast_stage put it")
            _streams[id(text)] = {"text": text, "at": at, "newlines": e["newlines"], "path": e["path"],
                                  "bytes": len(text) - e["newlines"] + e["bytes"]}
        _composed[:] = [text]
        return text

    def write_text_(self, data, encoding=None, errors=None, newline=None):
        if isinstance(data, str) and id(data) in _streams and _streams[id(data)]["text"] is data:
            if (encoding or "utf-8").lower().replace("_", "-") not in ("utf-8", "utf8", "locale") or newline not in (None, "\n"):
                raise ValueError("fast_stage: a streamed circuit is written as UTF-8 with \\n newlines")
            with open(self, "wb") as f:
                _of(data).update(_stream(data, f))
            return _streams[id(data)]["bytes"]
        if not isinstance(data, str) or len(data) <= CHUNK:
            return write_text(self, data, encoding, errors, newline)
        with self.open(mode="w", encoding=io.text_encoding(encoding), errors=errors, newline=newline) as f:
            for i in range(0, len(data), CHUNK):
                f.write(data[i:i + CHUNK])
        return len(data)

    forms.Circuit._new = new_
    CS.unit_lowering, CS.unit_circuit = unit_lowering_, unit_circuit_
    IL.lower, IL.lower_gates, IL.layout, FC._expanded, FC.compose = lower_, lower_gates_, layout_, expanded_, compose_
    Path.write_text = write_text_
    IL.digest, FC.digest, FC.class_digest = sha256, digest, class_digest
    # a structural unit (`class_statement.TmplLowering`) has no netlist: its unit_sha256 stays the library's, over its template
    CS.TmplLowering.sha256_of = IL.Lowering.sha256_of
    IL.Lowering.sha256, IL.Lowering.sha256_of = property(_lowering_sha256), _lowering_sha256_of
    IL.evaluate_layout = evaluate_layout


def composed() -> str:
    """The circuit text `circuit.compose` returned last (a streamed one stands for its file)."""
    return _composed[-1]


def size(text: str) -> int:
    """The length of the circuit ``text`` stands for."""
    s = _streams.get(id(text))
    return s["bytes"] if s is not None and s["text"] is text else len(text)


def _drop_spools() -> None:
    while _spools:
        shutil.rmtree(_spools.pop(), ignore_errors=True)


def release() -> None:
    """Drop the composed text, its digests, the unit's scratch netlist and its spool."""
    for e in _units.values():
        e["path"].unlink(missing_ok=True)
    for d in (_units, _streams, _memo, _unit_lays):
        d.clear()
    _composed.clear()
    _drop_spools()

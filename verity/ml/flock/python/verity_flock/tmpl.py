"""``flock-ir-tmpl/v1`` (``Flock.Tmpl`` in the Lean verifier, PRF-133): a flat ``flock-ir-unit/v2`` unit written as
Definitions and placed copies, so its text is O(description) where the flat form is O(unrolled).

:func:`emit` reads the structure off the flat unit itself, whatever lowering wrote it. It works level by level: at each
level it finds runs of identical consecutive blocks of items (rows at level 0, then rows and the calls the levels below
made), makes copy 0 of a run a Definition, and replaces the run with one call. A block repeats when each of its items
matches copy 0's item shifted by the block's width, a reference inside the copy moving with it and one outside moving
by a fixed step per copy. A reference outside becomes one of the Definition's inputs, bound by the call to a range of
the caller's columns that steps the same way. Definitions with the same content are one Definition. The format's
statement is in ``Flock/Tmpl.lean``; :func:`unroll` mirrors its ``unroll`` (substituted inputs, rows sorted and reduced
over GF(2)). The emitter's output unrolls to its input by construction; ``benchmarks/verifier_cost/flock_tmpl`` holds
the Lean unroll to it byte for byte.

The unit is held as arrays (each row's A and B as one run of int32 columns), since the H1 tile's flat unit at K = 4096 has
about 3·10⁹ of them.

    python -m verity_flock.tmpl CIRCUIT OUT.tmpl [--name NET] [--check]

reads the circuit file's largest block (or ``NET``), writes its templated form and prints its statistics as JSON;
``--check`` unrolls it in Python and compares (small units).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "flock-ir-tmpl/v1"
FLAT = "flock-ir-unit/v2"
_CHUNK = 1 << 20


@dataclass
class Flat:
    """A ``flock-ir-unit/v2`` unit: its header's tokens; row ``i``'s A then B columns, ``refs[ptr[i]:ptr[i + 1]]``, of
    which the first ``na[i]`` are A's; its ``LEAVES`` line."""

    head: list[str]
    ptr: np.ndarray
    refs: np.ndarray
    na: np.ndarray
    leaves: str | None = None

    @property
    def name(self) -> str:
        return self.head[1]

    @property
    def useful(self) -> int:
        return int(self.head[2])

    @property
    def const(self) -> int:
        return int(self.head[3])

    def row(self, i: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        r = self.refs[self.ptr[i]:self.ptr[i + 1]].tolist()
        return tuple(r[:self.na[i]]), tuple(r[self.na[i]:])


def _rows_of(lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows ``na a… nb b…``: per row its reference count and A's, and the references."""
    tok = np.fromstring(" ".join(lines), dtype=np.int64, sep=" ")
    cnt = np.fromiter((ln.count(" ") + 1 for ln in lines), np.int64, len(lines))
    ls = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
    na = tok[ls]
    nbp = ls + 1 + na
    nb = tok[nbp]
    assert (cnt == 2 + na + nb).all(), f"{FLAT}: a malformed row"
    keep = np.ones(len(tok), bool)
    keep[ls] = False
    keep[nbp] = False
    return na + nb, na, tok[keep].astype(np.int32)


def _flat(head: list[str], chunks, leaves: str | None) -> Flat:
    lens = np.concatenate([c[0] for c in chunks]) if chunks else np.zeros(0, np.int64)
    ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    na = np.concatenate([c[1] for c in chunks]).astype(np.int32) if chunks else np.zeros(0, np.int32)
    refs = np.concatenate([c[2] for c in chunks]) if chunks else np.zeros(0, np.int32)
    return Flat(head, ptr, refs, na, leaves)


def parse_flat(lines) -> Flat:
    lines = list(lines)
    head = lines[0].split()
    assert head[0] == FLAT, f"not a {FLAT} unit"
    n = int(head[2])
    rest = [ln for ln in lines[1 + n:] if ln.strip()]
    leaves = rest[0] if rest and rest[0].startswith("LEAVES ") else None
    assert len(rest) == (leaves is not None), f"{FLAT}: lines past the rows and LEAVES"
    chunks = [_rows_of(lines[1 + a:1 + min(n, a + _CHUNK)]) for a in range(0, n, _CHUNK)]
    return _flat(head, chunks, leaves)


def read_flat(path, name: str | None = None) -> tuple[Flat, int]:
    """A circuit file's ``KEYWORD name n`` block named ``name`` (else its largest) as a unit, read a chunk of rows at a
    time, and its text's bytes."""
    with open(path) as fh:
        fh.readline()
        fh.readline()
        blocks = []
        while line := fh.readline():
            _, nm, n = line.split()
            blocks.append((nm, int(n), fh.tell()))
            for _ in range(int(n)):
                fh.readline()
        pick = [b for b in blocks if b[0] == name] if name is not None else sorted(blocks, key=lambda b: b[1])[-1:]
        assert pick, f"{path}: no block {name or ''}"
        nm, n, at = pick[0]
        fh.seek(at)
        first = fh.readline()
        head, size = first.split(), len(first)
        rows = int(head[2])
        chunks = []
        for a in range(0, rows, _CHUNK):
            lines = [fh.readline() for _ in range(min(_CHUNK, rows - a))]
            size += sum(map(len, lines))
            chunks.append(_rows_of([ln.rstrip("\n") for ln in lines]))
        rest = [fh.readline() for _ in range(n - 1 - rows)]
        size += sum(map(len, rest))
        leaves = rest[0].rstrip("\n") if rest and rest[0].startswith("LEAVES ") else None
        assert len(rest) == (leaves is not None), f"{nm}: lines past the rows and LEAVES"
    return _flat(head, chunks, leaves), size


def flat_lines(f: Flat) -> list[str]:
    out = [" ".join(f.head)]
    for i in range(len(f.na)):
        a, b = f.row(i)
        out.append(" ".join(map(str, (len(a), *a, len(b), *b))))
    return out + ([f.leaves] if f.leaves is not None else [])


@dataclass(frozen=True)
class Call:
    e: int
    count: int
    base: int
    stride: int
    segs: tuple[tuple[int, int, int], ...]  # (len, at, step)


@dataclass(frozen=True)
class Def:
    cols: int
    ins: int
    rows: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]
    calls: tuple[Call, ...]


@dataclass
class Tmpl:
    name: str
    root: int
    groups: list[str]  # the flat header's port groups, as its tokens
    defs: list[Def]
    leaves: str | None = None


def write(t: Tmpl) -> str:
    out = [" ".join([FORMAT, t.name, str(t.root), *t.groups])]
    for d in t.defs:
        out.append(f"DEF {d.cols} {d.ins} {len(d.rows)} {len(d.calls)}")
        out += [" ".join(map(str, (c, len(a), *a, len(b), *b))) for c, a, b in d.rows]
        out += [" ".join(map(str, ("CALL", c.e, c.count, c.base, c.stride, len(c.segs), *(x for s in c.segs for x in s))))
                for c in d.calls]
    if t.leaves is not None:
        out.append(t.leaves)
    return "\n".join(out) + "\n"


def parse(text: str) -> Tmpl:
    lines = text.splitlines()
    h = lines[0].split()
    assert h[0] == FORMAT
    defs, leaves, at = [], None, 1
    while at < len(lines):
        if at + 1 == len(lines) and lines[at].startswith("LEAVES "):
            leaves = lines[at]
            break
        _, cols, ins, nr, nc = lines[at].split()
        rows, calls = [], []
        for ln in lines[at + 1:at + 1 + int(nr)]:
            v = list(map(int, ln.split()))
            na = v[1]
            rows.append((v[0], tuple(v[2:2 + na]), tuple(v[3 + na:])))
        for ln in lines[at + 1 + int(nr):at + 1 + int(nr) + int(nc)]:
            v = list(map(int, ln.split()[1:]))
            calls.append(Call(v[0], v[1], v[2], v[3], tuple(tuple(v[5 + 3 * i:8 + 3 * i]) for i in range(v[4]))))
        defs.append(Def(int(cols), int(ins), tuple(rows), tuple(calls)))
        at += 1 + int(nr) + int(nc)
    return Tmpl(h[1], int(h[2]), h[3:], defs, leaves)


def canon(xs) -> tuple[int, ...]:
    """A row as ``flock-ir-unit/v2`` writes it: a sum over GF(2), ascending, each column named an odd number of times."""
    par: dict[int, int] = {}
    for c in xs:
        par[c] = par.get(c, 0) ^ 1
    return tuple(sorted(c for c, p in par.items() if p))


def unroll(t: Tmpl) -> Flat:
    """``Tmpl.unroll``'s rows (without ``Net.ofRows``'s checks): the root at 0, its last column the constant."""
    n = t.defs[t.root].cols
    k = n - 1
    rows: list = [None] * n

    def place(d: int, o: int, inp: list[int]) -> None:
        df = t.defs[d]
        cols, ins = df.cols, df.ins

        def g(x):
            return o + x if x < cols else inp[x - cols] if x < cols + ins else k

        for col, a, b in df.rows:
            assert rows[o + col] is None, f"column {o + col} written twice"
            rows[o + col] = (canon(map(g, a)), canon(map(g, b)))
        for c in df.calls:
            for j in range(c.count):
                place(c.e, o + c.base + j * c.stride, [g(at + j * st + i) for ln, at, st in c.segs for i in range(ln)])

    place(t.root, 0, [])
    assert rows[k] is None, "the constant is written by a row"
    rows[k] = ((k,), (k,))
    rows = [r if r is not None else ((), ()) for r in rows]
    lens = np.array([len(a) + len(b) for a, b in rows], np.int64)
    na = np.array([len(a) for a, _ in rows], np.int64)
    refs = np.array([c for a, b in rows for c in a + b], np.int32)
    return _flat([FLAT, t.name, str(n), str(k), *t.groups], [(lens, na, refs)], t.leaves)


# --- the emitter ------------------------------------------------------------------------------------------------------

_ROW = 1 << 62  # a shape id at least this is a call's, `_Emitter.cshape[id - _ROW]`; below, a row's `na << 31 | nb`
_B = np.uint64(0x9E3779B97F4A7C15)


@dataclass
class _Level:
    """One level's items, in column order, tiling the unit: per item its first column, its width, its shape id and its
    references (a row's A and B, a call's segments' first columns; all global) ``refs[ptr[i]:ptr[i + 1]]``."""

    start: np.ndarray
    width: np.ndarray
    shape: np.ndarray
    ptr: np.ndarray
    refs: np.ndarray


class _Emitter:
    def __init__(self, flat: Flat, gram: int = 6, near: int = 64, gap: int = 4, max_period: int = 1 << 16):
        self.flat, self.k = flat, flat.const
        self.gram, self.near, self.gap, self.max_period = gram, near, gap, max_period
        self.defs: list[Def] = []
        self.ids: dict[Def, int] = {}
        self.cshape: list[tuple] = []  # (callee, count, stride, segment lengths, segment steps)
        self.cids: dict[tuple, int] = {}

    def _def(self, d: Def) -> int:
        e = self.ids.get(d)
        if e is None:
            e = self.ids[d] = len(self.defs)
            self.defs.append(d)
        return e

    def _cshape(self, s: tuple) -> int:
        i = self.cids.get(s)
        if i is None:
            i = self.cids[s] = len(self.cshape)
            self.cshape.append(s)
        return _ROW + i

    def _spans(self, sid: int, nrefs: int) -> list[int]:
        if sid < _ROW:
            return [1] * nrefs
        _, count, _, lens, steps = self.cshape[sid - _ROW]
        return [(count - 1) * s + ln for ln, s in zip(lens, steps)]

    def _candidates(self, lv: _Level) -> list[int]:
        """Per item, the next item whose ``gram`` items have the same signatures as its (each item's shape and its
        references, a reference as its distance back if within ``near``), or -1."""
        n, k = len(lv.start), self.k
        lens = np.diff(lv.ptr)
        mx = int(lens.max()) if n else 0
        pw = np.cumprod(np.full(mx + 1, _B, np.uint64))
        sig = np.empty(n, np.uint64)
        for a in range(0, n, _CHUNK):
            b = min(n, a + _CHUNK)
            p0, p1 = int(lv.ptr[a]), int(lv.ptr[b])
            ln = lens[a:b]
            r = lv.refs[p0:p1].astype(np.int64)
            d = np.repeat(lv.start[a:b], ln) - r
            code = np.where(r == k, -1, np.where((d >= 0) & (d <= self.near), d, -2)).astype(np.uint64)
            pos = np.arange(p1 - p0) - np.repeat(lv.ptr[a:b] - p0, ln)
            cs = np.concatenate([[np.uint64(0)], np.cumsum(code * pw[pos], dtype=np.uint64)])
            h = cs[lv.ptr[a + 1:b + 1] - p0] - cs[lv.ptr[a:b] - p0]
            sig[a:b] = (h ^ (lv.shape[a:b].astype(np.uint64) * _B)) * _B + ln.astype(np.uint64)
        g = np.zeros(n, np.uint64)
        for t in range(min(self.gram, n)):
            g[:n - t] += sig[t:] * pw[t]
        order = np.argsort(g, kind="stable")
        gs = g[order]
        nxt = np.full(n, -1, np.int64)
        same = gs[1:] == gs[:-1]
        nxt[order[:-1][same]] = order[1:][same]
        return nxt.tolist()

    def _run(self, lv: _Level, i0: int, P: int, limit: int) -> tuple[int, np.ndarray | None]:
        """How many copies of items ``i0 … i0 + P − 1`` follow one another from ``i0`` (to at most ``limit``), and copy 0's
        references' steps per copy; 0 if fewer than two."""
        if i0 + 2 * P > limit:
            return 0, None
        sh, ptr = lv.shape, lv.ptr
        s0 = sh[i0:i0 + P]
        if not np.array_equal(s0, sh[i0 + P:i0 + 2 * P]):
            return 0, None
        r0 = lv.refs[ptr[i0]:ptr[i0 + P]].astype(np.int64)
        r1 = lv.refs[ptr[i0 + P]:ptr[i0 + 2 * P]].astype(np.int64)
        D = r1 - r0
        if (D < 0).any() or ((r0 == self.k) != (r1 == self.k)).any():
            return 0, None
        c = len(r0)
        m, grow = 2, 2
        while True:
            M = min(m + grow, (limit - i0) // P, m + max(1, (1 << 24) // max(1, c + P)))
            if M <= m:
                break
            ok = (sh[i0 + m * P:i0 + M * P].reshape(M - m, P) == s0).all(axis=1)
            nS = M - m if ok.all() else int(np.argmin(ok))
            nR = 0
            if nS:
                rb = lv.refs[ptr[i0 + m * P]:ptr[i0 + (m + nS) * P]].astype(np.int64).reshape(nS, c)
                okR = (rb == r0 + np.arange(m, m + nS, dtype=np.int64)[:, None] * D).all(axis=1)
                nR = nS if okR.all() else int(np.argmin(okR))
            m += nR
            if nR < M - (m - nR):
                break
            grow *= 2
        return m, D

    def _make(self, lv: _Level, i0: int, P: int, D: np.ndarray) -> tuple[int, list[tuple[int, int, int]]]:
        """Copy 0 of a run as a Definition, and the run's input blocks ``(lo, hi, step)`` in global columns."""
        k, ptr = self.k, lv.ptr
        o0 = int(lv.start[i0])
        L = int(lv.start[i0 + P]) - o0
        items = []
        ext: dict[int, list[tuple[int, int]]] = {}
        p0 = int(ptr[i0])
        for x in range(P):
            u = i0 + x
            a, b = int(ptr[u]) - p0, int(ptr[u + 1]) - p0
            sid = int(lv.shape[u])
            refs = lv.refs[ptr[u]:ptr[u + 1]].tolist()
            steps = D[a:b].tolist()
            spans = self._spans(sid, len(refs))
            cls = []
            for r, s, n in zip(refs, steps, spans):
                if sid < _ROW and r == k:
                    cls.append(0)
                elif o0 <= r and r + n <= o0 + L and s == L:
                    cls.append(1)
                else:
                    cls.append(2)
                    ext.setdefault(s, []).append((r, r + n))
            items.append((u, sid, refs, steps, spans, cls))
        blocks: list[tuple[int, int, int]] = []
        for s in sorted(ext):
            cur = None
            for lo, hi in sorted(ext[s]):
                if cur is not None and lo <= cur[1] + self.gap:
                    cur[1] = max(cur[1], hi)
                else:
                    if cur is not None:
                        blocks.append((cur[0], cur[1], s))
                    cur = [lo, hi]
            blocks.append((cur[0], cur[1], s))
        blocks.sort()
        base, at = {}, 0
        by_step: dict[int, list[tuple[int, int, int]]] = {}
        for bl in blocks:
            base[bl] = at
            at += bl[1] - bl[0]
            by_step.setdefault(bl[2], []).append(bl)
        ins = at

        def local(r: int, n: int, s: int, c: int) -> int:
            if c == 0:
                return L + ins
            if c == 1:
                return r - o0
            for bl in by_step[s]:
                if bl[0] <= r and r + n <= bl[1]:
                    return L + base[bl] + r - bl[0]
            raise AssertionError("a reference outside every block")

        rows, calls = [], []
        for u, sid, refs, steps, spans, cls in items:
            col = int(lv.start[u]) - o0
            loc = [local(r, n, s, c) for r, s, n, c in zip(refs, steps, spans, cls)]
            if sid < _ROW:
                if loc:
                    na = sid >> 31
                    rows.append((col, tuple(loc[:na]), tuple(loc[na:])))
            else:
                e, count, stride, lens, ssteps = self.cshape[sid - _ROW]
                calls.append(Call(e, count, col, stride, tuple(zip(lens, loc, ssteps))))
        return self._def(Def(L, ins, tuple(rows), tuple(calls))), blocks

    def level(self, lv: _Level) -> tuple[_Level, int]:
        n = len(lv.start)
        nxt = self._candidates(lv)
        shape = lv.shape.tolist()
        limit = n - 1  # the constant's row, last, stays the root's
        keep, calls, runs, i = [], [], 0, 0
        kept = 0
        while i < n:
            best = (0, 0, None)
            j = nxt[i] if i < limit else -1
            for _ in range(2):
                if j <= i or j - i > self.max_period:
                    break
                if shape[i] == shape[j]:
                    m, D = self._run(lv, i, j - i, limit)
                    if m >= 2 and m * (j - i) > best[0] * best[1]:
                        best = (m, j - i, D)
                j = nxt[j]
            m, P, D = best
            if m >= 2:
                if kept < i:
                    keep.append((kept, i))
                e, blocks = self._make(lv, i, P, D)
                L = int(lv.start[i + P] - lv.start[i])
                sid = self._cshape((e, m, L, tuple(hi - lo for lo, hi, _ in blocks), tuple(s for *_, s in blocks)))
                calls.append((len(keep), int(lv.start[i]), m * L, sid, [lo for lo, _, _ in blocks]))
                keep.append(None)
                runs += 1
                i += m * P
                kept = i
            else:
                i += 1
        if kept < n:
            keep.append((kept, n))
        if not runs:
            return lv, 0
        st, wd, sh, lens, refs = [], [], [], [], []
        ci = 0
        for piece in keep:
            if piece is None:
                _, s0, w, sid, rf = calls[ci]
                ci += 1
                st.append(np.array([s0]))
                wd.append(np.array([w]))
                sh.append(np.array([sid]))
                lens.append(np.array([len(rf)]))
                refs.append(np.array(rf, np.int32))
            else:
                a, b = piece
                st.append(lv.start[a:b])
                wd.append(lv.width[a:b])
                sh.append(lv.shape[a:b])
                lens.append(np.diff(lv.ptr[a:b + 1]))
                refs.append(lv.refs[lv.ptr[a]:lv.ptr[b]])
        ln = np.concatenate(lens).astype(np.int64)
        out = _Level(np.concatenate(st).astype(np.int64), np.concatenate(wd).astype(np.int64),
                     np.concatenate(sh).astype(np.int64), np.concatenate([[0], np.cumsum(ln)]).astype(np.int64),
                     np.concatenate(refs).astype(np.int32))
        return out, runs

    def emit(self, max_levels: int = 64) -> tuple[Tmpl, dict]:
        f = self.flat
        n = f.useful
        nb = np.diff(f.ptr) - f.na
        lv = _Level(np.arange(n, dtype=np.int64), np.ones(n, np.int64),
                    (f.na.astype(np.int64) << 31) | nb.astype(np.int64), f.ptr, f.refs)
        levels = []
        for _ in range(max_levels):
            t0 = time.time()
            items = len(lv.start)
            lv, runs = self.level(lv)
            levels.append({"items": items, "runs": runs, "out": len(lv.start), "defs": len(self.defs),
                           "seconds": round(time.time() - t0, 1)})
            if runs == 0:
                break
        rows, calls = [], []
        for i in range(len(lv.start)):
            sid, s0 = int(lv.shape[i]), int(lv.start[i])
            refs = tuple(lv.refs[lv.ptr[i]:lv.ptr[i + 1]].tolist())
            if sid < _ROW:
                if s0 == self.k or not refs:
                    continue
                na = sid >> 31
                rows.append((s0, refs[:na], refs[na:]))
            else:
                e, count, stride, lens, ss = self.cshape[sid - _ROW]
                calls.append(Call(e, count, s0, stride, tuple(zip(lens, refs, ss))))
        root = self._def(Def(n, 0, tuple(rows), tuple(calls)))
        t = Tmpl(f.name, root, f.head[4:], self.defs, f.leaves)
        uses: dict[int, int] = {}
        for d in self.defs:
            for c in d.calls:
                uses[c.e] = uses.get(c.e, 0) + 1
        stats = {"flat_rows": n, "flat_refs": int(len(f.refs)), "levels": levels, "defs": len(self.defs),
                 "def_rows": sum(len(d.rows) for d in self.defs), "def_refs": sum(len(a) + len(b) for d in self.defs
                                                                                 for _, a, b in d.rows),
                 "def_calls": sum(len(d.calls) for d in self.defs),
                 "segs": sum(len(c.segs) for d in self.defs for c in d.calls), "root_rows": len(rows),
                 "root_refs": sum(len(a) + len(b) for _, a, b in rows), "root_calls": len(calls),
                 "shared_defs": sum(1 for v in uses.values() if v > 1)}
        return t, stats


def emit(flat: Flat, **kw) -> tuple[Tmpl, dict]:
    """``flat``'s templated form, and its statistics: per level its items, runs and Definitions so far, and the totals."""
    return _Emitter(flat, **kw).emit()


# --- aligned slots ----------------------------------------------------------------------------------------------------

WORD = 128


def _pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _up(x: int, a: int) -> int:
    return -(-x // a) * a


def _groups(tok: list[str]) -> tuple[list[tuple[int, int, list[int]]], list[tuple[int, int, list[int]]]]:
    v, at, out = list(map(int, tok)), 0, []
    for _ in range(2):
        gs = []
        n, at = v[at], at + 1
        for _ in range(n):
            col, words, nb = v[at:at + 3]
            gs.append((col, words, v[at + 3:at + 3 + nb]))
            at += 3 + nb
        out.append(gs)
    return out[0], out[1]


def _group_tokens(gs) -> list[str]:
    return [str(len(gs))] + [str(x) for col, words, bits in gs for x in (col, words, len(bits), *bits)]


class NotAligned(ValueError):
    pass


def align(t: Tmpl, flat: Flat) -> tuple[Tmpl, Flat, dict]:
    """``t`` with its repeated blocks in aligned power-of-two slots, and ``flat`` padded to match: so the fold over each run
    of copies splits into the copy's own columns and the slot index (``eq(r, o + j·2^s + x)``, ``o`` a multiple of
    ``2^s``). Each Definition is laid out once, its rows and unwritten columns in their order, each run of two or more
    copies at a stride of its callee's size rounded up to a power of two and a start aligned to it, a single copy at its
    callee's alignment. The unit's input rows keep their columns and each output group its width, at a start aligned to
    its words. The padding is empty rows (forced zeros); the map from old to new columns is increasing, so every row
    still reads earlier columns, and A and B keep their entries. ``flat`` is ``unroll(t)``."""
    defs = t.defs
    ins_g, outs_g = _groups(t.groups)
    in_end = (ins_g[-1][0] + ins_g[-1][1]) * WORD if ins_g else 0
    out_rng = [(c * WORD, (c + w) * WORD) for c, w, _ in outs_g]
    maps: list[np.ndarray] = [np.zeros(0, np.int64)] * len(defs)
    size, align_of = [0] * len(defs), [1] * len(defs)
    place: list[list[tuple[int, int]]] = [[] for _ in defs]  # per call: (new base, new stride)
    st = {"runs": 0, "nopad_runs": 0}
    for d, D in enumerate(defs):
        root = d == t.root
        L = D.cols
        m = np.empty(L, np.int64)
        pos, x, al = 0, 0, 1
        points = sorted(lo for lo, _ in out_rng) if root else []
        done: set[int] = set()

        def seq(x0: int, x1: int) -> None:
            nonlocal pos
            for p in points:
                if x0 <= p <= x1 and p not in done:
                    m[x0:p] = np.arange(pos, pos + p - x0)
                    pos += p - x0
                    x0 = p
                    words = next(w for c, w, _ in outs_g if c * WORD == p)
                    pos = _up(pos, words * WORD)
                    done.add(p)
            m[x0:x1] = np.arange(pos, pos + x1 - x0)
            pos += x1 - x0

        order = sorted(range(len(D.calls)), key=lambda i: D.calls[i].base)
        newc: dict[int, tuple[int, int]] = {}
        for ci in order:
            c = D.calls[ci]
            e = defs[c.e]
            if c.base < x:
                raise NotAligned(f"definition {d}: calls overlap")
            seq(x, c.base)
            if c.count > 1 and c.stride != e.cols:
                raise NotAligned(f"definition {d}: a call's copies have gaps")
            width = (c.count - 1) * c.stride + e.cols
            inside = [r for r in out_rng if r[0] <= c.base < r[1]]
            if root and (c.base < in_end or inside):
                if (c.base < in_end and c.base + width > in_end) or (inside and c.base + width > inside[0][1]):
                    raise NotAligned("a call crosses the end of the input rows or of an output group")
                if size[c.e] != e.cols:
                    raise NotAligned("a call in the input rows or an output group has padding inside")
                nb, S = pos, c.stride
                st["nopad_runs"] += c.count > 1
            elif c.count > 1:
                S = _pow2(size[c.e])
                nb = _up(pos, S)
                al = max(al, S)
            else:
                S = size[c.e]
                nb = _up(pos, align_of[c.e])
                al = max(al, align_of[c.e])
            st["runs"] += c.count > 1
            j = np.arange(c.count, dtype=np.int64)[:, None]
            m[c.base + j * c.stride + np.arange(e.cols)[None, :]] = nb + j * S + maps[c.e][None, :]
            newc[ci] = (nb, S)
            pos = nb + (c.count - 1) * S + size[c.e]
            x = c.base + width
        seq(x, L)
        maps[d], size[d], align_of[d] = m, pos, al
        place[d] = [newc[i] for i in range(len(D.calls))]
    phi = maps[t.root]
    if not (np.diff(phi) > 0).all():
        raise NotAligned("the column map is not increasing")
    if in_end and not (phi[:in_end] == np.arange(in_end)).all():
        raise NotAligned("the input rows moved")

    split = 0
    new_defs = []
    for d, D in enumerate(defs):
        m, L, L2 = maps[d], D.cols, size[d]

        def loc(v: int) -> int:
            return int(m[v]) if v < L else v - L + L2

        def locv(v: np.ndarray) -> np.ndarray:
            return np.where(v < L, m[np.minimum(v, L - 1)], v - L + L2)

        rows = tuple((int(m[col]), tuple(map(loc, a)), tuple(map(loc, b))) for col, a, b in D.rows)

        def bind(c: Call, j0: int, count: int):
            """Copies ``j0 … j0 + count − 1``'s bindings as segments in new columns, or None if one is not affine."""
            j = np.arange(count, dtype=np.int64)[:, None]
            segs = []
            for ln, at, sp in c.segs:
                if ln == 0:
                    segs.append((0, at, sp))
                    continue
                W = locv(at + (j0 + j) * sp + np.arange(ln, dtype=np.int64)[None, :])
                s2 = int(W[1, 0] - W[0, 0]) if count > 1 else 0
                if s2 >= 0 and (W == W[0, 0] + j * s2 + np.arange(ln)[None, :]).all():
                    segs.append((ln, int(W[0, 0]), s2))
                    continue
                for q in range(ln):
                    w = W[:, q]
                    s2 = int(w[1] - w[0]) if count > 1 else 0
                    if s2 < 0 or not (w == w[0] + j[:, 0] * s2).all():
                        return None
                    segs.append((1, int(w[0]), s2))
            return tuple(segs)

        calls = []
        for c, (nb, S) in zip(D.calls, place[d]):
            segs = bind(c, 0, c.count)
            if segs is not None:
                calls.append(Call(c.e, c.count, nb, S, segs))
                continue
            # a chained run (copy j reads copy j − 1) whose copy 0 reads what precedes the run: copy 0 on its own
            rest = bind(c, 1, c.count - 1) if c.count > 2 else None
            if rest is not None:
                st["peeled"] = st.get("peeled", 0) + 1
                calls.append(Call(c.e, 1, nb, size[c.e], bind(c, 0, 1)))
                calls.append(Call(c.e, c.count - 1, nb + S, S, rest))
                continue
            split += 1
            for jj in range(c.count):
                one = tuple((1, int(v), 0) for ln, at, sp in c.segs for v in locv(at + jj * sp + np.arange(ln)))
                calls.append(Call(c.e, 1, nb + jj * S, size[c.e], one))
        new_defs.append(Def(L2, D.ins, rows, tuple(calls)))

    n2 = size[t.root]
    outs2 = [(int(phi[c * WORD]) // WORD, w, bits) for c, w, bits in outs_g]
    groups2 = _group_tokens(ins_g) + _group_tokens(outs2)
    t2 = Tmpl(t.name, t.root, groups2, new_defs, t.leaves)
    lens = np.diff(flat.ptr)
    lens2 = np.zeros(n2, np.int64)
    lens2[phi] = lens
    na2 = np.zeros(n2, np.int32)
    na2[phi] = flat.na
    f2 = Flat([FLAT, flat.name, str(n2), str(n2 - 1), *groups2], np.concatenate([[0], np.cumsum(lens2)]).astype(np.int64),
              phi[flat.refs].astype(np.int32), na2, flat.leaves)
    pow2_runs = sum(1 for D in new_defs for c in D.calls if c.count > 1 and c.stride & (c.stride - 1) == 0
                    and c.base % c.stride == 0)
    n = flat.useful
    st |= {"useful": n, "useful_aligned": n2, "unit_log": max(7, (n - 1).bit_length()),
           "unit_log_aligned": max(7, (n2 - 1).bit_length()), "pow2_aligned_runs": pow2_runs, "split_calls": split}
    return t2, f2, st


def write_flat(f: Flat, path, block: str = "NET") -> None:
    """``f`` as a circuit file holding only its block (``flock_tmpl``'s diff reads it), a chunk of rows at a time."""
    with open(path, "w") as fh:
        n = len(f.na)
        fh.write(f"flock-ir-unit-only {f.name}\nMETA {{}}\n{block} {f.name} {1 + n + (f.leaves is not None)}\n")
        fh.write(" ".join(f.head) + "\n")
        for a in range(0, n, _CHUNK):
            b = min(n, a + _CHUNK)
            out = []
            for i in range(a, b):
                r = f.refs[f.ptr[i]:f.ptr[i + 1]].tolist()
                k = int(f.na[i])
                out.append(" ".join(map(str, (k, *r[:k], len(r) - k, *r[k:]))))
            fh.write("\n".join(out) + "\n")
        if f.leaves is not None:
            fh.write(f.leaves + "\n")


def main(argv: list[str]) -> int:
    import argparse
    import resource

    ap = argparse.ArgumentParser(prog="python -m verity_flock.tmpl")
    ap.add_argument("circuit")
    ap.add_argument("out")
    ap.add_argument("--name")
    ap.add_argument("--check", action="store_true", help="unroll the output in Python and compare (small units)")
    ap.add_argument("--align", metavar="TMPL", help="also write the template with aligned power-of-two slots (`align`)")
    ap.add_argument("--align-flat", metavar="CIRCUIT", help="and the padded flat unit it unrolls to, as a one-block file")
    a = ap.parse_args(argv)
    t0 = time.time()
    flat, size = read_flat(a.circuit, a.name)
    t1 = time.time()
    t, stats = emit(flat)
    text = write(t)
    Path(a.out).write_text(text)
    stats |= {"name": flat.name, "flat_bytes": size, "tmpl_bytes": len(text.encode()), "tmpl_lines": text.count("\n"),
              "read_seconds": round(t1 - t0, 1), "emit_seconds": round(time.time() - t1, 1),
              "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    if a.check:
        stats["check"] = _check(text, flat)
    if a.align:
        t3 = time.time()
        t2, f2, ast = align(t, flat)
        text2 = write(t2)
        Path(a.align).write_text(text2)
        if a.align_flat:
            write_flat(f2, a.align_flat)
        stats["aligned"] = ast | {"tmpl_bytes": len(text2.encode()), "seconds": round(time.time() - t3, 1)}
        if a.check:
            stats["aligned"]["check"] = _check(text2, f2)
    print("TMPL " + json.dumps(stats), flush=True)
    ok = all(s.get("check", {}).get("equal", True) for s in (stats, stats.get("aligned", {})))
    return 0 if ok else 1


def _check(text: str, flat: Flat) -> dict:
    t0 = time.time()
    got = unroll(parse(text))
    equal = (got.head == flat.head and got.leaves == flat.leaves and np.array_equal(got.ptr, flat.ptr)
             and np.array_equal(got.na, flat.na) and np.array_equal(got.refs, flat.refs))
    return {"equal": bool(equal), "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

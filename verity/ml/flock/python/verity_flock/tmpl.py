"""``flock-ir-tmpl/v2`` (``Flock.Tmpl`` in the Lean verifier): a flat ``flock-ir-unit/v2`` unit written as
Definitions and placed copies, so its text is O(description) where the flat form is O(unrolled).

:func:`emit` reads the structure off the flat unit itself, whatever lowering wrote it. It works level by level: at each
level it finds runs of identical consecutive blocks of items (rows at level 0, then rows and the calls the levels below
made), makes copy 0 of a run a Definition, and replaces the run with one call. A block repeats when each of its items
matches copy 0's item shifted by the block's width, a reference inside the copy moving with it and one outside moving
by a fixed step per copy. A reference outside becomes one of the Definition's inputs, bound by the call to a range of
the caller's columns that steps the same way. Definitions with the same content are one Definition. :func:`pack` then
writes each run of a call's segments one fixed stride apart as one two-level segment. The format's
statement is in ``Flock/Tmpl.lean``; :func:`unroll` mirrors its ``unroll`` (substituted inputs, rows sorted and reduced
over GF(2)). The emitter's output unrolls to its input by construction; ``benchmarks/verifier_cost/flock_tmpl`` holds
the Lean unroll to it byte for byte.

:func:`layout` mirrors ``Tmpl.layoutM`` (``Flock/TmplLayout.lean``): layout (iv) under a mode (:func:`mode_of`), where
the verifier commits each of the unrolled unit's columns. The templated statement (`circuit.compose`'s ``layout``) puts
the unit's range at the slot of the laid-out size, the root's ``Lay.size``.

The unit is held as arrays (each row's A and B as one run of int32 columns), since the H1 tile's flat unit at K = 4096 has
about 3·10⁹ of them.

    python -m verity_flock.tmpl CIRCUIT OUT.tmpl [--name NET] [--check]

reads the circuit file's largest block (or ``NET``), writes its templated form and prints its statistics as JSON;
``--check`` unrolls it in Python and compares (small units).
"""
from __future__ import annotations

import bisect
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "flock-ir-tmpl/v2"
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


def of_layout(lay, name: str) -> Flat:
    """A laid-out unit (`ir_lower.Layout`) as the unit `ir_lower.netlist` writes of it (no ``LEAVES`` line), read off its
    rows rather than its text."""
    head = [FLAT, name, str(lay.useful), str(lay.const)]
    for groups in (lay.in_groups, lay.out_groups):
        head.append(str(len(groups)))
        for col, words, bits in groups:
            head += [str(col), str(words), str(len(bits)), *map(str, bits)]
    na = np.fromiter((len(a) for a in lay.rows_a), np.int64, len(lay.rows_a))
    lens = na + np.fromiter((len(b) for b in lay.rows_b), np.int64, len(lay.rows_b))
    refs = np.fromiter((c for a, b in zip(lay.rows_a, lay.rows_b) for r in (a, b) for c in r), np.int32, int(lens.sum()))
    return _flat(head, [(lens, na, refs)], None)


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
    segs: tuple[tuple[int, int, int, int, int], ...]  # (len, at, step, reps, rstride)


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
            calls.append(Call(v[0], v[1], v[2], v[3], tuple(tuple(v[5 + 5 * i:10 + 5 * i]) for i in range(v[4]))))
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
                place(c.e, o + c.base + j * c.stride, [g(at + r * rs + j * st + i) for ln, at, st, reps, rs in c.segs
                                                       for r in range(reps) for i in range(ln)])

    place(t.root, 0, [])
    assert rows[k] is None, "the constant is written by a row"
    rows[k] = ((k,), (k,))
    rows = [r if r is not None else ((), ()) for r in rows]
    lens = np.array([len(a) + len(b) for a, b in rows], np.int64)
    na = np.array([len(a) for a, _ in rows], np.int64)
    refs = np.array([c for a, b in rows for c in a + b], np.int32)
    return _flat([FLAT, t.name, str(n), str(k), *t.groups], [(lens, na, refs)], t.leaves)


# --- the row order, checked on the description (``Tmpl.ordered``) -------------------------------------------------------

WORD = 128


def _ports(t: Tmpl) -> tuple[int, list[tuple[int, int]]]:
    """The input rows' end and each input group's port bits ``[lo, hi)``, from the header's tokens."""
    g, out, end = t.groups, [], 0
    at = 1
    for _ in range(int(g[0])):
        col, words, nb = int(g[at]), int(g[at + 1]), int(g[at + 2])
        out.append((col * WORD, col * WORD + sum(map(int, g[at + 3:at + 3 + nb]))))
        end = (col + words) * WORD
        at += 3 + nb
    return end, out


def _hi(s: tuple, count: int) -> int:
    """``Seg.hi``: past every reference segment ``s`` binds over ``count`` copies."""
    ln, at, st, reps, rs = s
    return at + max(reps - 1, 0) * rs + max(count - 1, 0) * st + ln


def _seg_ordered(cols: int, base: int, stride: int, count: int, s: tuple) -> bool:
    ln, at, st, reps, rs = s
    hi = _hi(s, count)
    return ln == 0 or cols <= at or (hi <= cols and at + max(reps - 1, 0) * rs + ln <= base
                                     and hi <= base + max(count - 1, 0) * stride)


def _def_ordered(F: list[bool], d: Def) -> bool:
    return (all(d.cols <= v or v < col for col, a, b in d.rows for v in a + b)
            and all((F[c.e] if c.e < len(F) else False)
                    and all(_seg_ordered(d.cols, c.base, c.stride, c.count, s) for s in c.segs) for c in d.calls))


def _fits(t: Tmpl, d: Def) -> bool:
    return (all(col < d.cols for col, _, _ in d.rows)
            and all(c.base + max(c.count - 1, 0) * c.stride + t.defs[c.e].cols <= d.cols for c in d.calls))


def ordered(t: Tmpl) -> dict:
    """``Tmpl.ordered``'s verdict and, for a refusal, what it refused: Definitions that don't fit their columns, the
    root's rows reading a later column or input padding, its calls straddling the input rows' end or of an unordered
    Definition, and its segments (``late``: an own column at or past a copy's first; ``padding``: an input column that
    is not a port bit of one group)."""
    F: list[bool] = []
    for d in t.defs:
        F.append(_def_ordered(F, d))
    ic, ports = _ports(t)
    span = lambda lo, hi: any(p0 <= lo and hi <= p1 for p0, p1 in ports)  # noqa: E731
    r = t.defs[t.root]
    cols = r.cols
    out = {"unfit": sum(not _fits(t, d) for d in t.defs), "defs_unordered": F.count(False), "root_rows": 0,
           "root_calls_straddle": 0, "root_callee_unordered": 0, "late": 0, "padding": 0}
    for col, a, b in r.rows:
        if col >= ic and not all(cols <= v + 1 or (v < col and (ic <= v or span(v, v + 1))) for v in a + b):
            out["root_rows"] += 1
    for c in r.calls:
        if c.base + max(c.count - 1, 0) * c.stride + t.defs[c.e].cols <= ic:
            continue
        if c.base < ic:
            out["root_calls_straddle"] += 1
            continue
        if not F[c.e]:
            out["root_callee_unordered"] += 1
        for s in c.segs:
            ln, at = s[0], s[1]
            hi = _hi(s, c.count)
            if ln == 1 and at + 1 == cols and hi <= cols:
                continue
            if not _seg_ordered(cols, c.base, c.stride, c.count, s):
                out["late"] += 1
            elif not (ic <= at or span(at, hi)):
                out["padding"] += 1
    out["ok"] = (not out["unfit"] and not any(out[k] for k in ("root_rows", "root_calls_straddle",
                                                                "root_callee_unordered", "late", "padding"))
                 and all(_def_ordered(F, d) for d, f in zip(t.defs, F) if f))
    return out


def _cut(segs, cuts: list[int]) -> tuple[tuple[int, int, int, int, int], ...]:
    """One-level segments of a single copy, or stepping by 0, split at ``cuts``, so none straddles one."""
    out = []
    for ln, at, st, reps, _rs in segs:
        assert reps == 1, "legalize takes one-level segments"
        x = at
        for c in sorted(set(cuts)):
            if x < c < at + ln:
                out.append((c - x, x, st, 1, 0))
                x = c
        if at + ln > x or ln == 0:
            out.append((at + ln - x, x, st, 1, 0))
    return tuple(out)


def legalize(t: Tmpl) -> tuple[Tmpl, dict]:
    """``t`` with every segment that straddles what ``ordered`` tells apart split: a Definition's own columns and its
    inputs, and in the root the input rows' end, an input group's port bits and the constant. A segment of a run of
    copies that steps can't be split for every copy at once, so its run becomes single copies first. A root run
    straddling the input rows' end becomes the run below it, the copy across it and the run above. It unrolls to the same
    rows."""
    ic, ports = _ports(t)
    st = {"split_segs": 0, "split_runs": 0, "single_calls": 0}
    defs = list(t.defs)

    def sub(c: Call, a: int, b: int) -> Call:
        return Call(c.e, b - a, c.base + a * c.stride, c.stride,
                    tuple((ln, at + a * step, step if b - a > 1 else 0, reps, rs) for ln, at, step, reps, rs in c.segs))

    for d, D in enumerate(defs):
        root = d == t.root
        cuts = [D.cols] + ([ic, D.cols - 1] + [x for p in ports for x in p] if root else [])

        def straddles(c: Call, s) -> bool:
            ln, at = s[0], s[1]
            hi = _hi(s, c.count)
            return ln > 0 and any(at < x < hi for x in cuts)

        def pieces(c: Call) -> list[Call]:
            w = defs[c.e].cols
            if c.count > 1 and root and c.base < ic < c.base + (c.count - 1) * c.stride + w:
                j1 = (ic - c.base - w) // c.stride + 1 if c.base + w <= ic else 0
                j2 = -(-(ic - c.base) // c.stride)
                st["split_runs"] += 1
                st["single_calls"] += j2 - j1
                return ((pieces(sub(c, 0, j1)) if j1 else []) + [sub(c, j, j + 1) for j in range(j1, j2)]
                        + (pieces(sub(c, j2, c.count)) if j2 < c.count else []))
            if c.count > 1 and any(straddles(c, s) and s[2] for s in c.segs):
                st["single_calls"] += c.count
                return [sub(c, j, j + 1) for j in range(c.count)]
            return [c]

        calls = []
        for c in D.calls:
            for p in pieces(c):
                segs = _cut(p.segs, cuts) if any(straddles(p, s) for s in p.segs) else p.segs
                st["split_segs"] += len(segs) - len(p.segs)
                calls.append(Call(p.e, p.count, p.base, p.stride, segs))
        defs[d] = Def(D.cols, D.ins, D.rows, tuple(calls))
    return Tmpl(t.name, t.root, t.groups, defs, t.leaves), st


def _seg_ok(t: Tmpl, ic: int, ports: list[tuple[int, int]], d: int, c: Call, s: tuple) -> bool:
    """Whether ``Tmpl.ordered`` takes segment ``s`` of call ``c`` in Definition ``d``."""
    D = t.defs[d]
    if d != t.root:
        return _seg_ordered(D.cols, c.base, c.stride, c.count, s)
    if c.base + max(c.count - 1, 0) * c.stride + t.defs[c.e].cols <= ic:
        return True
    ln, at, hi = s[0], s[1], _hi(s, c.count)
    return ((ln == 1 and at + 1 == D.cols and hi <= D.cols)
            or (_seg_ordered(D.cols, c.base, c.stride, c.count, s)
                and (ic <= at or any(p0 <= at and hi <= p1 for p0, p1 in ports))))


def pack(t: Tmpl) -> Tmpl:
    """``t`` with each run of a call's consecutive segments of one length and step, their first references one fixed
    stride apart, as one two-level segment, wherever ``Tmpl.ordered`` takes the merged segment. It binds the same
    references in the same order, so it unrolls to the same rows."""
    ic, ports = _ports(t)
    defs = []
    for d, D in enumerate(t.defs):
        calls = []
        for c in D.calls:
            out: list[tuple] = []
            for s in c.segs:
                if out and s[3] == 1 and s[0] > 0:
                    ln, at, st, R, RS = out[-1]
                    rs = s[1] - at if R == 1 else RS
                    if s[0] == ln and s[2] == st and rs >= 0 and s[1] == at + R * rs:
                        m = (ln, at, st, R + 1, rs)
                        if _seg_ok(t, ic, ports, d, c, m):
                            out[-1] = m
                            continue
                out.append(tuple(s))
            calls.append(Call(c.e, c.count, c.base, c.stride, tuple(out)))
        defs.append(Def(D.cols, D.ins, D.rows, tuple(calls)))
    return Tmpl(t.name, t.root, t.groups, defs, t.leaves)


# --- layout (iv): where the unrolled unit's columns are committed (``Flock.TmplLayout``) -----------------------------------

def mode_of(name: str) -> tuple[str, int]:
    """``Tmpl.modeOf``: ``iv`` (lenient), ``iv/strict``, ``iv/hybrid`` (``T = 0``) or ``iv/hybrid/T``, as ``(kind, T)``."""
    if name in ("iv", "iv/strict", "iv/hybrid"):
        return {"iv": "lenient", "iv/strict": "strict", "iv/hybrid": "hybrid"}[name], 0
    if name.startswith("iv/hybrid/") and name[10:].isdigit():
        return "hybrid", int(name[10:])
    raise ValueError(f"{name!r} is not a layout mode (iv, iv/strict, iv/hybrid, iv/hybrid/T)")


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _up(x: int, a: int) -> int:
    return -(-x // a) * a


@dataclass(frozen=True)
class Lay:
    """``Tmpl.Lay``: a Definition's laid-out size and alignment, its own columns' runs ``(x0, x1, p0)`` (column ``x`` of
    ``[x0, x1)`` at ``p0 + x − x0``), each call's place ``(nb, cs, ms)`` (copy ``j``'s column at callee position ``p`` at
    ``nb + j·cs + p·ms``), whether every column is at its own index, and the alignment its copies are placed at."""

    size: int
    align: int
    runs: tuple[tuple[int, int, int], ...]
    places: tuple[tuple[int, int, int], ...]
    ident: bool
    palign: int


def _out_groups(t: Tmpl) -> list[tuple[int, int]]:
    """Each output group's first column and its width in columns, from the header's tokens."""
    g, at, out = t.groups, 1, []
    for _ in range(int(g[0])):
        at += 3 + int(g[at + 2])
    n, at = int(g[at]), at + 1
    for _ in range(n):
        out.append((int(g[at]) * WORD, int(g[at + 1]) * WORD))
        at += 3 + int(g[at + 2])
    return out


def _lay_def(t: Tmpl, lays: list[Lay], kind: str, T: int, paths: list[int], ic: int, d: int) -> Lay:
    """``Tmpl.layDefM``: Definition ``d`` laid out from its callees' layouts."""
    D, root = t.defs[d], d == t.root
    outs = sorted(_out_groups(t)) if root else []
    pos, x, al, ident = 0, 0, 1, True
    runs: list[tuple[int, int, int]] = []
    places: list[tuple[int, int, int]] = []
    done: set[int] = set()

    def own(x0: int, hi: int) -> None:
        nonlocal pos
        for p, w in outs:
            if x0 <= p <= hi and p not in done:
                if x0 < p:
                    runs.append((x0, p, pos))
                pos = _up(pos + (p - x0), w)
                x0 = p
                done.add(p)
        if x0 < hi:
            runs.append((x0, hi, pos))
        pos += hi - x0

    for c in D.calls:
        Le, E = t.defs[c.e].cols, lays[c.e]
        if c.base < x:
            raise ValueError(f"layout: definition {d}: its calls are not in order of their columns")
        if c.count > 1 and c.stride != Le:
            raise ValueError(f"layout: definition {d}: a call's copies are not side by side")
        own(x, c.base)
        in_rows = root and c.base < ic
        if in_rows or any(p <= c.base < p + w for p, w in outs):
            if E.size != Le or (in_rows and not E.ident):
                raise ValueError(f"layout: definition {d}: a call in the input rows or an output group to a definition laid out apart")
            places.append((pos, Le, 1))
            pos += c.count * Le
        elif c.count == 1:
            nb = _up(pos, E.palign)
            al = max(al, E.palign)
            places.append((nb, Le, 1))
            pos = nb + E.size
        else:
            M = _next_pow2(c.count)
            nb = _up(pos, M * E.palign)
            al = max(al, M * E.palign)
            places.append((nb, 1, M))
            pos = nb + E.size * M
        ident = ident and E.ident and places[-1][0] == c.base and (c.count == 1 or places[-1][2] == 1)
        x = c.base + c.count * Le
    own(x, D.cols)
    ident = ident and pos == D.cols and all(a == p for a, _b, p in runs)
    whole = kind == "strict" or (kind == "hybrid" and paths[d] > 1)
    palign = max(al, _next_pow2(pos) >> (T if kind == "hybrid" else 0)) if whole else al
    return Lay(pos, al, tuple(runs), tuple(places), ident, palign)


def pos_of(t: Tmpl, lays: list[Lay]):
    """``Tmpl.posOf``: the root's column ``x`` at its committed position, through the call tree, as a function of ``x``."""
    bases = [[c.base for c in D.calls] for D in t.defs]
    starts = [[a for a, _b, _p in L.runs] for L in lays]

    def at(x: int) -> int:
        d, A, B = t.root, 0, 1
        for _ in range(len(t.defs) + 1):
            D, L = t.defs[d], lays[d]
            i = bisect.bisect_right(bases[d], x)
            if i and x < D.calls[i - 1].base + D.calls[i - 1].count * t.defs[D.calls[i - 1].e].cols:
                c = D.calls[i - 1]
                j, x = divmod(x - c.base, t.defs[c.e].cols)
                nb, cs, ms = L.places[i - 1]
                A, B, d = A + B * (nb + j * cs), B * ms, c.e
                continue
            r = bisect.bisect_right(starts[d], x)
            if r and x < L.runs[r - 1][1]:
                a, _b, p = L.runs[r - 1]
                return A + B * (p + x - a)
            break
        return A + B * x

    return at


def layout(t: Tmpl, mode: str) -> list[Lay]:
    """``Tmpl.layoutM``: every Definition's layout (iv) under ``mode`` (:func:`mode_of`), callees first, refused unless the
    input rows keep their columns, each output group starts at a multiple of its width with its columns in order, and the
    constant is the layout's last column. The root's ``size`` is the committed unit's: the constant's position plus one."""
    kind, T = mode_of(mode)
    ic, _ = _ports(t)
    paths = [0] * len(t.defs)
    paths[t.root] = 1
    for d in range(len(t.defs) - 1, -1, -1):
        for c in t.defs[d].calls:
            paths[c.e] += paths[d]
    lays: list[Lay] = []
    for d in range(len(t.defs)):
        lays.append(_lay_def(t, lays, kind, T, paths, ic, d))
    R, L = t.defs[t.root], lays[t.root]
    if not (all(ic <= a or a == p for a, _b, p in L.runs)
            and all(ic <= c.base or (nb == c.base and lays[c.e].ident) for c, (nb, _cs, _ms) in zip(R.calls, L.places))):
        raise ValueError("layout: the input rows do not keep their columns")
    at = pos_of(t, lays)
    for col, w in _out_groups(t):
        p = at(col)
        if p % w or any(at(col + i) != p + i for i in range(w)):
            raise ValueError(f"layout: output group at word {col // WORD} does not keep its columns at a multiple of its width")
    if at(R.cols - 1) + 1 != L.size:
        raise ValueError("layout: the constant is not the layout's last column")
    return lays


# --- evaluation from the template, never unrolled ------------------------------------------------------------------------

_ALL = np.uint64(0xFFFFFFFFFFFFFFFF)
#: a frame's bytes past which `_Evaluator` splits its lanes and chunks a run's copies
BUDGET = 1 << 28


def out_ports(t: Tmpl) -> list[tuple[int, int]]:
    """Each output group's port bits ``[lo, hi)``, from the header's tokens."""
    g, at, out = t.groups, 1, []
    for _ in range(int(g[0])):
        at += 3 + int(g[at + 2])
    for _ in range(int(g[at])):
        col, nb = int(g[at + 1]), int(g[at + 3])
        out.append((col * WORD, col * WORD + sum(map(int, g[at + 4:at + 4 + nb]))))
        at += 3 + nb
    return out


def _bindings(c: Call) -> tuple[np.ndarray, np.ndarray]:
    """Call ``c``'s references for copy 0, in its callee's input order, and their step per copy: copy ``j`` binds
    ``p0 + j·step`` (``unroll``'s ``at + r·rstride + j·step + i``)."""
    p0, st = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for ln, at, step, reps, rs in c.segs:
        p0.append((at + rs * np.arange(reps, dtype=np.int64)[:, None] + np.arange(ln, dtype=np.int64)[None, :]).reshape(-1))
        st.append(np.full(ln * reps, step, np.int64))
    return np.concatenate(p0), np.concatenate(st)


def _pow2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


@dataclass
class _Plan:
    """How one Definition is evaluated: the own columns its frame holds (``keep``, ascending; frame row ``i`` is
    ``keep[i]``, then the inputs, the constant's all-ones and a zero row), and its steps by level, each ``("rows", dst, A,
    B)`` (frame rows, A and B padded with the zero row) or ``("call", c, p0, step, chained, need)``."""

    cols: int
    ins: int
    keep: np.ndarray
    steps: list
    given: tuple = ()

    @property
    def frame(self) -> int:
        return len(self.keep) + self.ins + 2

    def at(self, refs: np.ndarray) -> np.ndarray:
        """Template references as frame rows: an own column its slot in ``keep``, an input after them, the constant the
        all-ones row."""
        refs = np.asarray(refs, np.int64)
        K = len(self.keep)
        out = np.empty(refs.shape, np.int64)
        own = refs < self.cols
        pos = np.searchsorted(self.keep, refs[own])
        if pos.size and (pos.max() >= K or not np.array_equal(self.keep[pos], refs[own])):
            raise ValueError("evaluate: a reference to an own column the frame does not hold")
        out[own] = pos
        if (refs > self.cols + self.ins).any():
            raise ValueError("evaluate: a reference past the constant")
        out[~own] = K + refs[~own] - self.cols
        return out


class _Evaluator:
    """`evaluate`'s walk: each Definition planned once, from the columns its callers read of it (`_plan`), and evaluated
    once per call path, every copy of a run at once (a chained run, whose copies read earlier copies, one copy at a time),
    each lane a word of 64 instances: memory is the frames along one call path, each at most ``budget`` bytes a lane
    chunk, never the unrolled unit."""

    def __init__(self, t: Tmpl, out_cols: np.ndarray, given: np.ndarray, budget: int = BUDGET):
        self.t, self.budget = t, budget
        need: list[list[np.ndarray]] = [[] for _ in t.defs]
        need[t.root].append(np.asarray(out_cols, np.int64))
        self.need: list[np.ndarray | None] = [None] * len(t.defs)
        self.plans: list[_Plan | None] = [None] * len(t.defs)
        self.given = np.asarray(given, np.int64)
        self.given_set = set(self.given.tolist())
        for d in range(len(t.defs) - 1, -1, -1):
            if not need[d]:
                continue
            for c in t.defs[d].calls:
                if c.e >= d:
                    raise ValueError(f"evaluate: definition {d} calls {c.e}, not an earlier definition")
            self.need[d] = np.unique(np.concatenate(need[d]))
            self.plans[d] = self._plan(d, self.need[d], need)

    def _plan(self, d: int, want: np.ndarray, need: list[list[np.ndarray]]) -> _Plan:
        D = self.t.defs[d]
        cols, ins = D.cols, D.ins
        given = self.given if d == self.t.root else np.zeros(0, np.int64)
        rows = [r for r in D.rows if not (d == self.t.root and r[0] in self.given_set)]
        binds = [_bindings(c) for c in D.calls]
        spans = [(c.base, c.base + c.count * c.stride) for c in D.calls]
        for c in D.calls:
            if c.count > 1 and c.stride < self.t.defs[c.e].cols:
                raise ValueError(f"evaluate: definition {d}: a call's copies overlap")
        # candidate own columns: everything wanted, given or read, so liveness is marks on one sorted array
        reads = [want, given, np.fromiter((x for _c, a, b in rows for x in a + b if x < cols), np.int64)]
        reads.append(np.fromiter((c for c, _a, _b in rows), np.int64, len(rows)))
        for c, (p0, st) in zip(D.calls, binds):
            R = (p0[None, :] + np.arange(c.count, dtype=np.int64)[:, None] * st[None, :]).reshape(-1)
            reads.append(np.unique(R[R < cols]))
        cand = np.unique(np.concatenate(reads))
        live = np.zeros(len(cand), bool)
        live[np.searchsorted(cand, want)] = True

        def pos(x) -> np.ndarray:
            return np.searchsorted(cand, x)

        # backward liveness over the rows and calls in column order (every read is of an earlier column)
        items = sorted([(r[0], 0, i) for i, r in enumerate(rows)] + [(s[0], 1, k) for k, s in enumerate(spans)])
        row_pos = pos(np.fromiter((r[0] for r in rows), np.int64, len(rows)))
        evaluated: set[tuple[int, int]] = set()
        call_need: dict[int, np.ndarray] = {}
        for _col, kind, i in reversed(items):
            if kind == 0:
                if not live[row_pos[i]]:
                    continue
                evaluated.add((0, i))
                _c, a, b = rows[i]
                own = [x for x in a + b if x < cols]
                if own:
                    live[pos(np.asarray(own, np.int64))] = True
                continue
            c, (lo, hi) = D.calls[i], spans[i]
            a, z = np.searchsorted(cand, [lo, hi])
            if not live[a:z].any():
                continue
            evaluated.add((1, i))
            p0, st = binds[i]
            R = (p0[None, :] + np.arange(c.count, dtype=np.int64)[:, None] * st[None, :]).reshape(-1)
            R = R[R < cols]
            if R.size:
                live[pos(R)] = True
            got = cand[a:z][live[a:z]]
            off = (got - lo) % c.stride if c.count > 1 else got - lo
            Le = self.t.defs[c.e].cols
            if (off >= Le).any():
                raise ValueError(f"evaluate: definition {d} reads a column between a call's copies")
            call_need[i] = np.unique(off)
            need[c.e].append(call_need[i])
        keep = cand[live]
        plan = _Plan(cols, ins, keep, [])
        held = np.isin(given, keep)
        plan.given = (np.flatnonzero(held), np.searchsorted(keep, given[held]))
        for col, a, b in rows:
            if any(col <= x < cols for x in a + b):
                raise ValueError(f"evaluate: definition {d}: row {col} reads a later column")
        for c, (p0, st) in zip(D.calls, binds):
            js = np.arange(c.count, dtype=np.int64)[:, None]
            R = p0[None, :] + js * st[None, :]
            if ((R < cols) & (R >= c.base + js * c.stride)).any():
                raise ValueError(f"evaluate: definition {d}: a call's copy reads a column at or past its own")
        # levels: a row or call one past the latest item it reads; a chained run's copies read their own span
        level = np.zeros(len(keep), np.int64)
        K = len(keep)
        by_level: dict[int, list] = {}
        for _col, kind, i in items:
            if (kind, i) not in evaluated:
                continue
            if kind == 0:
                col, a, b = rows[i]
                own = [x for x in a + b if x < cols]
                lv = 1 + (int(level[np.searchsorted(keep, own)].max()) if own else 0)
                level[np.searchsorted(keep, col)] = lv
                by_level.setdefault(lv, []).append((0, i))
                continue
            c, (lo, hi) = D.calls[i], spans[i]
            p0, st = binds[i]
            R = (p0[None, :] + np.arange(c.count, dtype=np.int64)[:, None] * st[None, :]).reshape(-1)
            chained = bool(((R >= lo) & (R < hi)).any())
            R = R[(R < cols) & ~((R >= lo) & (R < hi))]
            lv = 1 + (int(level[np.searchsorted(keep, R)].max()) if R.size else 0)
            a, z = np.searchsorted(keep, [lo, hi])
            level[a:z] = lv
            by_level.setdefault(lv, []).append((1, i, chained))
        for lv in sorted(by_level):
            groups: dict[tuple[int, int], list] = {}
            for it in by_level[lv]:
                if it[0] == 1:
                    i = it[1]
                    p0, st = binds[i]
                    plan.steps.append(("call", D.calls[i], p0, st, it[2], call_need[i]))
                    continue
                col, a, b = rows[it[1]]
                groups.setdefault((_pow2(len(a)), _pow2(len(b))), []).append((col, a, b))
            for (pa, pb), rs in groups.items():
                zero = K + ins + 1
                A = np.full((len(rs), pa), zero, np.int64)
                B = np.full((len(rs), pb), zero, np.int64)
                for j, (_col, a, b) in enumerate(rs):
                    A[j, :len(a)] = plan.at(np.asarray(a, np.int64))
                    B[j, :len(b)] = plan.at(np.asarray(b, np.int64))
                dst = np.searchsorted(keep, np.fromiter((r[0] for r in rs), np.int64, len(rs)))
                plan.steps.append(("rows", dst, A.astype(np.int32), B.astype(np.int32)))
        return plan

    def run(self, d: int, X: np.ndarray, given: np.ndarray | None = None) -> np.ndarray:
        """Definition ``d``'s wanted columns (``need[d]``) on its inputs ``X`` ``(ins, lanes)``; the root's ``given`` columns
        ``(given, lanes)``."""
        p = self.plans[d]
        L = X.shape[1]
        if L > 1 and p.frame * L * 8 > self.budget:
            h = L // 2
            return np.concatenate([self.run(d, X[:, :h], None if given is None else given[:, :h]),
                                   self.run(d, X[:, h:], None if given is None else given[:, h:])], axis=1)
        K = len(p.keep)
        V = np.zeros((p.frame, L), np.uint64)
        V[K:K + p.ins] = X
        V[K + p.ins] = _ALL
        if given is not None:
            gi, gs = p.given
            V[gs] = given[gi]
        for s in p.steps:
            if s[0] == "rows":
                _, dst, A, B = s
                per = max(1, self.budget // (8 * L * (A.shape[1] + B.shape[1])))
                for r0 in range(0, len(dst), per):
                    a = np.bitwise_xor.reduce(V[A[r0:r0 + per]], axis=1)
                    a &= np.bitwise_xor.reduce(V[B[r0:r0 + per]], axis=1)
                    V[dst[r0:r0 + per]] = a
                continue
            _, c, p0, st, chained, want = s
            e = self.plans[c.e]
            cc = 1 if chained else max(1, min(c.count, self.budget // (8 * L * max(1, e.frame))))
            for j0 in range(0, c.count, cc):
                j1 = min(c.count, j0 + cc)
                js = np.arange(j0, j1, dtype=np.int64)
                Xe = V[p.at(p0[None, :] + js[:, None] * st[None, :])]  # (copies, ins_e, L)
                Y = self.run(c.e, Xe.transpose(1, 0, 2).reshape(e.ins, (j1 - j0) * L))
                if len(want) != len(self.need[c.e]):
                    Y = Y[np.searchsorted(self.need[c.e], want)]
                dst = c.base + js[:, None] * c.stride + want[None, :]  # (copies, want)
                Y = Y.reshape(len(want), j1 - j0, L).transpose(1, 0, 2)
                at = np.searchsorted(p.keep, dst)
                ok = at < K
                ok[ok] = p.keep[at[ok]] == dst[ok]
                V[at[ok]] = Y[ok]
        return V[np.searchsorted(p.keep, self.need[d])]


def evaluate(t: Tmpl, inputs: np.ndarray, out_cols=None, budget: int = BUDGET) -> np.ndarray:
    """The unit's columns ``out_cols`` (root columns; by default every output port bit, in order), bit-sliced, from the
    template itself: ``inputs`` ``(bits, lanes)`` uint64 are the input port bits' columns (every input group's, in order),
    each lane a word of 64 instances. Rows are evaluated a Definition at a time over a run's copies at once, so the walk is
    O(description) Python steps and its memory one call path's frames (`_Evaluator`), what ``unroll`` then evaluating
    would give (`struct_lower.evaluate`)."""
    _ic, ports = _ports(t)
    given = np.concatenate([np.arange(lo, hi, dtype=np.int64) for lo, hi in ports]) if ports else np.zeros(0, np.int64)
    inputs = np.asarray(inputs, np.uint64)
    if inputs.shape[0] != len(given):
        raise ValueError(f"evaluate: {inputs.shape[0]} input bits for the template's {len(given)}")
    if out_cols is None:
        out_cols = np.concatenate([np.arange(lo, hi, dtype=np.int64) for lo, hi in out_ports(t)])
    out_cols = np.asarray(out_cols, np.int64)
    ev = _Evaluator(t, out_cols, given, budget)
    got = ev.run(t.root, np.zeros((0, inputs.shape[1]), np.uint64), inputs)
    return got[np.searchsorted(ev.need[t.root], out_cols)]


def totals(t: Tmpl) -> dict:
    """The unrolled unit's rows and their A and B references, counted per Definition and multiplied through its calls:
    ``unroll``'s rows without the cancellations its ``canon`` makes where two of a row's references bind one column."""
    tot: list[tuple[int, int, int]] = []
    for d in t.defs:
        r, a, b = len(d.rows), sum(len(x[1]) for x in d.rows), sum(len(x[2]) for x in d.rows)
        for c in d.calls:
            r, a, b = r + c.count * tot[c.e][0], a + c.count * tot[c.e][1], b + c.count * tot[c.e][2]
        tot.append((r, a, b))
    rows, nnz_a, nnz_b = tot[t.root]
    return {"rows": rows, "nnz_a": nnz_a, "nnz_b": nnz_b}


def slice_lanes(bits: np.ndarray) -> np.ndarray:
    """``(n, bits)`` 0/1 per instance as ``(bits, ceil(n / 64))`` uint64 lanes, instance ``i`` bit ``i mod 64`` of word
    ``i // 64``."""
    bits = np.asarray(bits, np.uint8)
    n = bits.shape[0]
    W = -(-n // 64)
    pad = np.zeros((W * 64, bits.shape[1]), np.uint8)
    pad[:n] = bits
    return np.ascontiguousarray(np.packbits(pad.T, axis=1, bitorder="little")).view("<u8").astype(np.uint64)


def unslice_lanes(lanes: np.ndarray, n: int) -> np.ndarray:
    """`slice_lanes`' inverse: ``(bits, W)`` lanes as ``(n, bits)`` 0/1."""
    b = np.unpackbits(np.ascontiguousarray(np.asarray(lanes, "<u8")).view(np.uint8), axis=1, bitorder="little")
    return b[:, :n].T.copy()


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
        self.unaligned = 0  # candidate runs cut short or dropped by `_aligned`

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
        pw = np.cumprod(np.full(max(mx, self.gram) + 1, _B, np.uint64))
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

    def _aligned(self, lv: _Level, i0: int, P: int, m: int, D: np.ndarray) -> int:
        """How many of the run's ``m`` copies keep each reference of a copy, as a span of columns, wholly the copy's own
        (stepping with it), wholly before the copy or wholly after it, so that ``Tmpl.ordered`` reads the run's
        bindings as ordered; 0 if copy 0 doesn't."""
        o0 = int(lv.start[i0])
        L = int(lv.start[i0 + P]) - o0
        p0, p1 = int(lv.ptr[i0]), int(lv.ptr[i0 + P])
        r = lv.refs[p0:p1].astype(np.int64)
        s = D.astype(np.int64)
        n = np.ones(p1 - p0, np.int64)
        const = np.zeros(p1 - p0, bool)
        for u in range(i0, i0 + P):
            a, b = int(lv.ptr[u]) - p0, int(lv.ptr[u + 1]) - p0
            sid = int(lv.shape[u])
            if sid >= _ROW:
                n[a:b] = self._spans(sid, b - a)
            else:
                const[a:b] = r[a:b] == self.k
        own = (o0 <= r) & (r + n <= o0 + L) & (s == L)
        before, after = r + n <= o0, r >= o0 + L
        if (~(const | own | before | after)).any():
            return 0
        up = before & ~own & ~const & (s > L)
        if up.any():
            m = min(m, int(((o0 - r[up] - n[up]) // (s[up] - L)).min()) + 1)
        down = after & ~const & (s < L)
        if down.any():
            m = min(m, int(((r[down] - o0 - L) // (L - s[down])).min()) + 1)
        return m

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
                calls.append(Call(e, count, col, stride, tuple((ln, r, s, 1, 0) for ln, r, s in zip(lens, loc, ssteps))))
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
                    if m >= 2:
                        m2 = self._aligned(lv, i, j - i, m, D)
                        self.unaligned += m2 < m
                        m = m2
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
                calls.append(Call(e, count, s0, stride, tuple((ln, r, s, 1, 0) for ln, r, s in zip(lens, refs, ss))))
        root = self._def(Def(n, 0, tuple(rows), tuple(calls)))
        t, legal = legalize(Tmpl(f.name, root, f.head[4:], self.defs, f.leaves))
        segs1 = sum(len(c.segs) for d in t.defs for c in d.calls)
        t = pack(t)
        uses: dict[int, int] = {}
        for d in t.defs:
            for c in d.calls:
                uses[c.e] = uses.get(c.e, 0) + 1
        stats = {"flat_rows": n, "flat_refs": int(len(f.refs)), "levels": levels, "defs": len(t.defs),
                 "def_rows": sum(len(d.rows) for d in t.defs), "def_refs": sum(len(a) + len(b) for d in t.defs
                                                                                 for _, a, b in d.rows),
                 "def_calls": sum(len(d.calls) for d in t.defs),
                 "segs": sum(len(c.segs) for d in t.defs for c in d.calls), "segs_one_level": segs1,
                 "root_rows": len(rows),
                 "root_refs": sum(len(a) + len(b) for _, a, b in rows), "root_calls": len(calls),
                 "shared_defs": sum(1 for v in uses.values() if v > 1), "unaligned_runs": self.unaligned, "legalized": legal, "order": ordered(t)}
        return t, stats


def emit(flat: Flat, **kw) -> tuple[Tmpl, dict]:
    """``flat``'s templated form, and its statistics: per level its items, runs and Definitions so far, and the totals."""
    return _Emitter(flat, **kw).emit()


def main(argv: list[str]) -> int:
    import argparse
    import resource

    ap = argparse.ArgumentParser(prog="python -m verity_flock.tmpl")
    ap.add_argument("circuit")
    ap.add_argument("out")
    ap.add_argument("--name")
    ap.add_argument("--check", action="store_true", help="unroll the output in Python and compare (small units)")
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
        t2 = time.time()
        got = unroll(parse(text))
        equal = (got.head == flat.head and got.leaves == flat.leaves and np.array_equal(got.ptr, flat.ptr)
                 and np.array_equal(got.na, flat.na) and np.array_equal(got.refs, flat.refs))
        stats["check"] = {"equal": bool(equal), "seconds": round(time.time() - t2, 1)}
    print("TMPL " + json.dumps(stats), flush=True)
    return 0 if stats.get("check", {}).get("equal", True) and stats["order"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

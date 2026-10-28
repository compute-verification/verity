"""tmpl_layouts.py TMPL [--flat CIRCUIT | --unroll] [--modes M,...]: the sizes of layouts of a `flock-ir-tmpl/v2` unit, mined
(`verity_flock.tmpl`) or structural (`verity_flock.struct_lower`), as JSON lines.

* ``today``: the template's own layout, the flat unit as it is.
* ``pow2``: each run of copies at a stride of its callee's size rounded up to a power of two.
* ``interleaved``: a run of n ≥ 2 copies of a Definition of laid-out size L takes L·2^m rows, 2^m ≥ n, copy i's local
  row j at ``start + j·2^m + i``, the start aligned to 2^m times the Definition's own alignment, so the copy index is the
  low m bits and j the high ones. Applied recursively, a copy's local row j already holds its own inner runs' copy bits,
  so the outermost run's copy bits are the lowest. A single copy keeps its columns, at its Definition's alignment. A run
  whose copies read one another (a binding into the run's own columns, directly or through an input of the Definition
  that some instance binds to its own columns: chained, as a K loop's accumulator is) can't
  interleave and keep each row reading earlier columns, so it falls back to ``contiguous`` (its copies side by side at
  L, the carry fold's case) or ``pow2``. Aligning a region's start to 2^m times its Definition's own alignment cleans
  the copy bits, but the high bits are then the Definition's columns plus an offset that needn't be aligned, so its
  fold is redone per call (``call_paths``: the calls and own references of the call tree with each run counted once);
  ``strict`` aligns to 2^m times the Definition's size rounded up to a power of two, so each Definition folds once.
  Fallback ``interleave`` interleaves chained runs too: committed positions apart from the evaluation order, which
  stays the flat unit's, so its ``--flat`` late references are reported, not refused.

Every layout keeps the unit's input rows at their columns and each output group's width, at a start aligned to its
words, as `tmpl.align` does. Per layout: the unit's rows and its slot's log (`circuit._slot_log`; compose refuses a
unit that fills its slot exactly), the runs by kind with the rows they hold unrolled, and, for interleaved, each run's
bound inputs over all its copies by kind: shared (one column for every copy), lane (copy i's column is ``w0 + i·s``
with s a power of two and ``w0`` a multiple of ``s·2^m``, so the fold's sum over copies is eq on a bit field), shifted
(s a power of two, ``w0`` not aligned or s negative: adding a constant to the copy bits, a two-state carry), param (the
Definition's own input, bound by its caller), affine (steps evenly by a non-power of two: a multiply by a constant) or
irregular. A run is clean when every binding is shared, lane, shifted or param. ``--flat``
reads the flat unit (``--unroll``: the template's own unroll, for a structural template, which has no flat file) and
checks, under each layout's map, that the columns are distinct and every row reads only earlier columns (its own self
row for an input, and the constant, excepted). Option (iv) is ``interleaved`` with fallback ``interleave``, lenient
(``strict`` false) and strict. ``--modes`` picks layouts by name, ``today``, ``pow2``, ``interleaved/contiguous``,
``interleaved/pow2``, ``interleaved/contiguous/strict``, ``iv``, ``iv/strict``, ``iv/hybrid`` and ``iv/hybrid/T`` (hybrid
aligned to each power of two over 2^T, `layout`'s ``coarse``). ``memo``: the keys and reads of a fold memoized by
Definition, lane scale and start residue (`_memo`); ``gaps_by_def``: where the alignment rows go, with each Definition's
name from ``TMPL.names.json`` (`struct_lower` writes it) when there is one. ``--digest`` adds each map's digest,
which `flock-tmpl layout` (`Flock.TmplLayout`) prints for its own.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from verity_flock import tmpl as T

WORD = T.WORD


class NotAligned(ValueError):
    pass


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


def _seg_refs(s: tuple, j: np.ndarray) -> np.ndarray:
    """Segment ``s``'s references for the copies ``j`` (a column), run by run: copies × (reps · len)."""
    ln, at, sp, reps, rs = s
    r = (np.arange(reps, dtype=np.int64)[:, None] * rs + np.arange(ln, dtype=np.int64)[None, :]).ravel()
    return at + j * sp + r[None, :]


def _log(n: int) -> int:
    return max(0, (n - 1).bit_length())


def _slot_log(useful: int) -> int:
    return max(7, (useful - 1).bit_length())


def layout(t: T.Tmpl, mode: str, fallback: str = "contiguous", strict: bool = False,
           hybrid: bool = False, coarse: int = 0) -> tuple[np.ndarray, dict]:
    """The map from the unit's columns to the layout's, and the layout's statistics. ``strict``: each interleaved region
    and each single copy starts at a multiple of its Definition's size rounded up to a power of two (times 2^m), so the
    region's high bits are the Definition's own columns plus an aligned offset and its fold is computed once. ``hybrid``:
    strict for a Definition the call tree reaches on more than one path (a run counted once), lenient for the rest, which
    are folded once anyway; ``summary`` is then what a fold that summarizes each Definition once per lane scale reads.
    ``coarse`` = t aligns those to their power of two over 2^t instead (never below their own alignment), so a
    memoized fold (``memo``) keeps at most 2^t residues of each per lane scale."""
    defs = t.defs
    ins_g, outs_g = _groups(t.groups)
    in_end = (ins_g[-1][0] + ins_g[-1][1]) * WORD if ins_g else 0
    out_rng = [(c * WORD, (c + w) * WORD) for c, w, _ in outs_g]
    paths = [0] * len(defs)
    paths[t.root] = 1
    for d in range(len(defs) - 1, -1, -1):
        for c in defs[d].calls:
            paths[c.e] += paths[d]
    whole = [strict or (hybrid and paths[e] > 1) for e in range(len(defs))]
    urows = [0] * len(defs)
    maps: list[np.ndarray] = [np.zeros(0, np.int64)] * len(defs)
    size, align = [0] * len(defs), [1] * len(defs)
    rnd, gap = [0] * len(defs), [0] * len(defs)  # rows each definition adds: copies rounded up, alignment gaps
    kinds: list[list[tuple]] = [[] for _ in defs]  # per def: (call, kind, 2^m)
    place: list[list[tuple]] = [[] for _ in defs]  # per def: (call, kind, 2^m, stride, start)
    gaps: list[dict[int, int]] = [{} for _ in defs]  # per def: the alignment gap before each callee's placements
    alias = _aliases(t) if mode == "interleaved" else [{} for _ in defs]
    st = {"runs": 0, "fixed_runs": 0, "lanes_runs": 0, "chained_runs": 0, "rows_lanes": 0, "rows_chained": 0,
          "pow2_runs": 0, "rows_pow2": 0}
    for d, D in enumerate(defs):
        root = d == t.root
        L = D.cols
        urows[d] = len(D.rows) + sum(c.count * urows[c.e] for c in D.calls)
        m = np.empty(L, np.int64)
        pos, x, al, rd = 0, 0, 1, 0
        points = sorted(lo for lo, _ in out_rng) if root else []
        done: set[int] = set()

        def seq(x0: int, x1: int) -> None:
            nonlocal pos
            for p in points:
                if x0 <= p <= x1 and p not in done:
                    m[x0:p] = np.arange(pos, pos + p - x0)
                    pos += p - x0
                    x0 = p
                    pos = _up(pos, next(w for c, w, _ in outs_g if c * WORD == p) * WORD)
                    done.add(p)
            m[x0:x1] = np.arange(pos, pos + x1 - x0)
            pos += x1 - x0

        for c in sorted(D.calls, key=lambda c: c.base):
            e, Le = c.e, defs[c.e].cols
            if c.base < x:
                raise NotAligned(f"definition {d}: calls overlap")
            if c.count > 1 and c.stride != Le:
                raise NotAligned(f"definition {d}: a call's copies have gaps")
            seq(x, c.base)
            width = c.count * Le
            inside = [r for r in out_rng if r[0] <= c.base < r[1]]
            M = 1
            st["runs"] += c.count > 1
            if root and (c.base < in_end or inside):
                if size[e] != Le:
                    raise NotAligned("a call in the input rows or an output group has padding inside")
                kind, S, nb = "contig", Le, pos
                st["fixed_runs"] += c.count > 1
            elif mode == "today":
                kind, S, nb = "contig", size[e], pos
            elif c.count == 1:
                a = max(align[e], _pow2(size[e]) >> coarse) if whole[e] else align[e]
                kind, S, nb = "contig", size[e], _up(pos, a)
                al = max(al, a)
            elif mode == "interleaved" and (fallback == "interleave" or not _chained(c, Le, D, alias[d])):
                M = 1 << _log(c.count)
                a = M * (max(align[e], _pow2(size[e]) >> coarse) if whole[e] else align[e])
                kind, S, nb = "lanes", M, _up(pos, a)
                al = max(al, a)
                st["lanes_runs"] += 1
                st["rows_lanes"] += c.count * urows[e]
                if fallback == "interleave" and _chained(c, Le, D, alias[d]):
                    st["chained_runs"] += 1
                    st["rows_chained"] += c.count * urows[e]
            else:
                if mode == "pow2" or fallback == "pow2":
                    S = _pow2(size[e])
                    nb = _up(pos, S)
                    al = max(al, S)
                    st["pow2_runs"] += 1
                    st["rows_pow2"] += c.count * urows[e]
                else:
                    S, nb = size[e], _up(pos, align[e])
                    al = max(al, align[e])
                kind = "contig"
                if mode == "interleaved":
                    kind = "chained"
                    st["chained_runs"] += 1
                    st["rows_chained"] += c.count * urows[e]
            gaps[d][e] = gaps[d].get(e, 0) + nb - pos
            j = np.arange(c.count, dtype=np.int64)[:, None]
            cols = c.base + j * Le + np.arange(Le, dtype=np.int64)[None, :]
            if kind == "lanes":
                m[cols] = nb + maps[e][None, :] * M + j
                pos = nb + size[e] * M
                rd += (M - c.count) * size[e] + c.count * rnd[e]
            else:
                m[cols] = nb + j * S + maps[e][None, :]
                pos = nb + (c.count - 1) * S + size[e]
                rd += (c.count - 1) * (S - size[e]) + c.count * rnd[e]
            kinds[d].append((c, kind, M))
            place[d].append((c, kind, M, S, nb))
            x = c.base + width
        seq(x, L)
        maps[d], size[d], align[d], rnd[d] = m, pos, al, rd
        gap[d] = pos - L - rd
    n = size[t.root]
    own = [sum(len(a) + len(b) for _, a, b in D.rows) for D in defs]
    walk_n, walk_r = [0] * len(defs), [0] * len(defs)
    for d, D in enumerate(defs):
        walk_n[d] = sum(1 + walk_n[c.e] for c in D.calls)
        walk_r[d] = own[d] + sum(walk_r[c.e] for c in D.calls)
    out = {"mode": mode, "fallback": fallback if mode == "interleaved" else None, "strict": strict, "useful": n,
           "slot_log": _slot_log(n), "slack": round(1 - n / (1 << _slot_log(n)), 4), "rows_unrolled": urows[t.root],
           "rows_rounding": rnd[t.root], "rows_alignment": gap[t.root], **st,
           "template_refs": sum(own), "call_paths": {"calls": walk_n[t.root], "refs": walk_r[t.root]}}
    if strict or hybrid:
        scales: list[set[int]] = [set() for _ in defs]
        scales[t.root] = {1}
        for d in range(len(defs) - 1, -1, -1):
            for c, kind, M in kinds[d]:
                scales[c.e] |= {s * M for s in scales[d]}
        ctx = [len(s) for s in scales]
        out["summary"] = {"whole_defs": sum(whole), "contexts": sum(ctx),
                          "refs": sum(own[e] * ctx[e] for e in range(len(defs))),
                          "pairs": sum(ctx[d] * sum(c.count * defs[c.e].ins for c in D.calls) for d, D in enumerate(defs)),
                          "pairs_runs": sum(ctx[d] * sum(defs[c.e].ins for c in D.calls) for d, D in enumerate(defs))}
    inst = [0] * len(defs)
    inst[t.root] = 1
    for d in range(len(defs) - 1, -1, -1):
        for c in defs[d].calls:
            inst[c.e] += inst[d] * c.count
    pad = [0] * len(defs)
    for d in range(len(defs)):
        for e, g in gaps[d].items():
            pad[e] += inst[d] * g
    top = sorted(range(len(defs)), key=lambda e: -pad[e])[:12]
    out["gaps_by_def"] = [{"def": e, "rows": pad[e], "size": size[e], "pow2": _pow2(size[e]), "cols": defs[e].cols,
                           "whole": whole[e], "paths": paths[e], "instances": inst[e]} for e in top if pad[e]]
    if mode == "interleaved":
        g = [(max(align[e], _pow2(size[e]) >> coarse) if whole[e] else align[e]).bit_length() - 1 for e in range(len(defs))]
        out["memo"] = _memo(defs, t.root, place, size, own, maps, g)
    if mode != "today":
        out |= _bindings(defs, maps, size, kinds, urows)
    if mode == "interleaved":
        out["aliased_inputs"] = {"defs": sum(1 for a in alias if a), "inputs": sum(len(a) for a in alias)}
        if fallback == "contiguous":
            out["chained_split"] = _split(t, kinds, urows, size, alias, n)
    return maps[t.root], out


def _memo(defs, root: int, place, size, own, maps, g) -> dict:
    """What a fold memoized by (Definition, lane scale, its start mod B) reads, B its laid-out size rounded up to a power
    of two: below B a placed Definition's rows are its start's residue plus its own columns, one carry into the bits
    above, so its fold is one summary per key. A lane run's start is a multiple of 2^m (lenient aligns it to 2^m times
    its callee's alignment), so its copy bits stay clean and the callee's key is the start over 2^m, at scale + m.
    ``split``: the fold that keys by (Definition, scale) alone, its table at the starts' alignment, ``g`` (log2) or
    finer."""
    B = [_pow2(s) for s in size]
    keys: list[set[tuple[int, int]]] = [set() for _ in defs]
    keys[root] = {(0, 0)}
    unclean = 0
    for d in range(len(defs) - 1, -1, -1):
        for s, r in keys[d]:
            for c, kind, M, S, nb in place[d]:
                e = c.e
                if kind == "lanes":
                    q, z = divmod(r + nb, M)
                    unclean += z != 0
                    keys[e].add((s + M.bit_length() - 1, q % B[e]))
                else:
                    v = (r + nb + np.arange(c.count, dtype=np.int64) * S) % B[e]
                    keys[e].update((s, x) for x in v.tolist())
    n = [len(k) for k in keys]
    ctx = [len({s for s, _ in k}) for k in keys]
    beta = [b.bit_length() - 1 for b in B]
    # split: a (Definition, scale) whose starts are multiples of 2^a there is walked once into a table over its rows'
    # and references' positions above 2^a, and each placement combines the table, at most 4^(beta - a) entries; or,
    # where that isn't fewer (or it is placed once), each placement walks it directly: its own refs and its calls'
    # placements
    # placements; an input's reads are its (input, chunk) entries, paired copy by copy where a table is combined
    a_min: list[dict[int, int]] = [{} for _ in defs]
    for e, k in enumerate(keys):
        for s, r in k:
            a_min[e][s] = min(a_min[e].get(s, g[e]), ((r & -r).bit_length() - 1) if r else g[e], beta[e])
    npl: dict[tuple[int, int], int] = {}
    for d in range(len(defs)):
        for s in a_min[d]:
            for c, kind, M, S, nb in place[d]:
                k2 = (c.e, s + M.bit_length() - 1 if kind == "lanes" else s)
                npl[k2] = npl.get(k2, 0) + (1 if kind == "lanes" else c.count)
    cost: dict[tuple[int, int], int] = {}
    vin: dict[tuple[int, int], np.ndarray] = {}  # (input << 32 | chunk), its rows' input reads
    tabled: set[tuple[int, int]] = set()
    tables = table_refs = pairs = spans = 0
    for e in range(len(defs)):
        D = defs[e]
        L, K = D.cols, D.cols + D.ins
        rc = [(maps[e][col], x - L) for col, ra, rb in D.rows for x in ra + rb if L <= x < K]
        rpos = np.array([p for p, _ in rc], np.int64)
        rin = np.array([i for _, i in rc], np.int64)
        binds = []
        for c, kind, M, S, nb in place[e]:
            j = np.arange(c.count, dtype=np.int64)[:, None]
            vs = [_seg_refs(sg, j) for sg in c.segs if sg[0] * sg[3]]
            binds.append(np.concatenate(vs, axis=1) if vs else np.zeros((c.count, 0), np.int64))
        for s, a in a_min[e].items():
            direct, ent = own[e], [(rin << 32) | (rpos >> a)]
            for (c, kind, M, S, nb), bd in zip(place[e], binds):
                k2 = (c.e, s + M.bit_length() - 1 if kind == "lanes" else s)
                p = 1 if kind == "lanes" else c.count
                direct += p * cost[k2]
                v = vin[k2]
                if k2 in tabled:
                    pairs += c.count * len(v)
                if not len(v):
                    continue
                ac = a_min[k2[0]][k2[1]]
                ic, hc = v >> 32, v & 0xFFFFFFFF
                jj = np.arange(c.count, dtype=np.int64)[:, None]
                start = nb + jj if kind == "lanes" else nb + jj * S
                pos = start + (hc << ac)[None, :] * (M if kind == "lanes" else 1)
                spans += int(ac + (M.bit_length() - 1 if kind == "lanes" else 0) > a)
                w = bd[:, ic]
                ok = (w >= L) & (w < K)
                ent.append(((w[ok] - L) << 32) | (pos[ok] >> a))
            vin[(e, s)] = np.unique(np.concatenate(ent))
            comb = 4 ** (beta[e] - a)
            if npl.get((e, s), 1) > 1 and comb < direct:
                cost[(e, s)] = comb
                tabled.add((e, s))
                tables += 1
                table_refs += direct
            else:
                cost[(e, s)] = direct
    top = sorted(range(len(defs)), key=lambda e: -n[e] * own[e])[:8]
    return {"keys": sum(n), "contexts": sum(ctx), "max_keys": max(n), "unclean_lanes": unclean,
            "split": {"total": cost[(root, 0)] + table_refs + pairs, "tables": tables, "table_refs": table_refs,
                      "pairs": pairs, "coarser_children": spans, "placements": sum(npl.values()),
                      "max_t": max(beta[e] - a for e in range(len(defs)) for a in a_min[e].values())},
            "refs": sum(n[e] * own[e] for e in range(len(defs))),
            "pairs": sum(n[d] * sum(c.count * defs[c.e].ins for c in D.calls) for d, D in enumerate(defs)),
            "pairs_runs": sum(n[d] * sum(defs[c.e].ins for c in D.calls) for d, D in enumerate(defs)),
            "by_def": [{"def": e, "keys": n[e], "contexts": ctx[e], "own": own[e], "size": size[e]} for e in top]}


def _reads(c: T.Call, D: T.Def, al: dict[int, set[int]]):
    """Per segment of ``c``'s bindings, its offset among the callee's inputs and the caller's own columns copy j's input
    reads as (j, input, column) arrays: the column bound, or the columns an input of the caller it is bound to aliases."""
    j = np.arange(c.count, dtype=np.int64)[:, None]
    q0 = 0
    for s in c.segs:
        ln = s[0] * s[3]
        if ln:
            v = _seg_refs(s, j)
            jj, qq = np.nonzero(v < D.cols)
            parts = [(jj, qq, v[jj, qq])]
            if al:
                pj, pq = np.nonzero(np.isin(v - D.cols, np.fromiter(al, np.int64, len(al))))
                hits = [(a, b, x) for a, b in zip(pj.tolist(), pq.tolist()) for x in al[int(v[a, b]) - D.cols]]
                if hits:
                    parts.append(tuple(np.array(h, np.int64) for h in zip(*hits)))
            yield q0, tuple(np.concatenate(x) for x in zip(*parts))
        q0 += ln


def _aliases(t: T.Tmpl) -> list[dict[int, set[int]]]:
    """Per Definition, each input that some instance binds to one of that instance's own columns, and those columns
    (local). A Definition is laid out once, so a run inside it whose copies read such an input is chained."""
    defs = t.defs
    al: list[dict[int, set[int]]] = [{} for _ in defs]
    for d in range(len(defs) - 1, -1, -1):
        D = defs[d]
        for c in D.calls:
            Le = defs[c.e].cols
            for q0, (jj, qq, col) in _reads(c, D, al[d]):
                u = col - c.base - jj * Le
                ok = (u >= 0) & (u < Le)
                for q, x in zip((q0 + qq[ok]).tolist(), u[ok].tolist()):
                    al[c.e].setdefault(q, set()).add(x)
    return al


def _chained(c: T.Call, Le: int, D: T.Def, al: dict[int, set[int]]) -> bool:
    lo, hi = c.base, c.base + c.count * Le
    return any(((col >= lo) & (col < hi)).any() for _, (_, _, col) in _reads(c, D, al))


def _odd(x: int) -> int:
    return x >> ((x & -x).bit_length() - 1) if x else 0


def _cone(defs, e: int, bad_in: frozenset, urows, cache) -> tuple[int, np.ndarray]:
    """Definition ``e``'s rows (unrolled, per copy) that read the inputs ``bad_in`` transitively, and its columns among
    them; a call any of whose copies reads them is followed into its callee (the union over its copies)."""
    key = (e, bad_in)
    if key in cache:
        return cache[key]
    E = defs[e]
    L = E.cols
    bad = np.zeros(L + E.ins + 1, bool)
    bad[[L + q for q in bad_in]] = True
    items = sorted([(col, 0, a, b) for col, a, b in E.rows] + [(c.base, 1, c, None) for c in E.calls],
                   key=lambda x: x[0])
    n = 0
    for col, k, a, b in items:
        if k == 0:
            if bad[list(a)].any() or bad[list(b)].any():
                bad[col] = True
                n += 1
            continue
        c, Le = a, defs[a.e].cols
        j = np.arange(c.count, dtype=np.int64)[:, None]
        inner, q0 = set(), 0
        for s in c.segs:
            ln = s[0] * s[3]
            if ln:
                hit = bad[_seg_refs(s, j)].any(axis=0)
                inner |= set((q0 + np.nonzero(hit)[0]).tolist())
            q0 += ln
        if inner:
            ci, mask = _cone(defs, c.e, frozenset(inner), urows, cache)
            n += c.count * ci
            cols = c.base + j * Le + np.nonzero(mask)[0][None, :]
            bad[cols.ravel()] = True
    cache[key] = (n, bad[:L])
    return cache[key]


def _split(t: T.Tmpl, kinds, urows, size, alias, useful: int) -> dict:
    """An estimate (unrolled rows, no column map) of splitting each outermost chained run into its carried cone (the
    rows that read another copy, transitively) and the rest: the rest interleaved over 2^m lanes, the cone contiguous
    (the carry fold, at a stride of the cone's size) or in power-of-two slots. Also the strides' odd parts, which the
    carry fold's DP over the copy bits pays per run."""
    defs = t.defs
    free = [0] * len(defs)  # instances of each Definition outside every chained run
    free[t.root] = 1
    for d in range(len(defs) - 1, -1, -1):
        for c, kind, _ in kinds[d]:
            if kind != "chained":
                free[c.e] += free[d] * c.count
    cache: dict = {}
    out = dict.fromkeys(("runs", "cone_rows", "rest_rows", "rest_rounding", "cone_pow2_pad"), 0)
    q_full, q_cone = [], []
    for d, D in enumerate(defs):
        for c, kind, _ in kinds[d]:
            if kind != "chained":
                continue
            q_full.append(_odd(size[c.e]))
            if not free[d]:
                continue
            Le = defs[c.e].cols
            lo, hi = c.base, c.base + c.count * Le
            bad = set()
            for q0, (jj, qq, col) in _reads(c, D, alias[d]):
                inside = (col >= lo) & (col < hi)
                bad |= set((q0 + qq[inside]).tolist())
            cone, _ = _cone(defs, c.e, frozenset(bad), urows, cache)
            rest = urows[c.e] - cone
            M = 1 << _log(c.count)
            q_cone.append(_odd(cone))
            out["runs"] += free[d]
            out["cone_rows"] += free[d] * c.count * cone
            out["rest_rows"] += free[d] * c.count * rest
            out["rest_rounding"] += free[d] * (M - c.count) * rest
            out["cone_pow2_pad"] += free[d] * c.count * (_pow2(cone) - cone) if cone else 0
    est = useful + out["rest_rounding"]
    out |= {"useful_cone_contiguous": est, "slot_log_cone_contiguous": _slot_log(est),
            "useful_cone_pow2": est + out["cone_pow2_pad"], "slot_log_cone_pow2": _slot_log(est + out["cone_pow2_pad"]),
            "stride_odd_part": {"runs": len(q_full), "sum": sum(q_full), "max": max(q_full, default=0),
                                "median": sorted(q_full)[len(q_full) // 2] if q_full else 0},
            "cone_odd_part": {"sum": sum(q_cone), "max": max(q_cone, default=0),
                              "median": sorted(q_cone)[len(q_cone) // 2] if q_cone else 0}}
    return out


def _classify(v: np.ndarray, L: int, m: np.ndarray, M: int) -> dict:
    """Bindings ``v`` (copies × columns, the caller's) by kind, in the layout ``m``."""
    k = dict.fromkeys(("param", "lane", "shifted", "affine", "irregular"), 0)
    inp = v >= L
    k["param"] = int(inp.all(axis=0).sum())
    k["irregular"] = int((inp.any(axis=0) & ~inp.all(axis=0)).sum())
    v = v[:, ~inp.any(axis=0)]
    if v.shape[1] == 0:
        return k
    j = np.arange(v.shape[0], dtype=np.int64)[:, None]
    W = m[v]
    s = W[1] - W[0]
    a = np.abs(s)
    even = (W == W[0][None, :] + j * s[None, :]).all(axis=0)
    p2 = even & (a > 0) & ((a & (a - 1)) == 0)
    lane = p2 & (s > 0) & (W[0] % np.maximum(a * M, 1) == 0)
    k["lane"] += int(lane.sum())
    k["shifted"] += int((p2 & ~lane).sum())
    k["affine"] += int((even & ~p2).sum())
    k["irregular"] += int((~even).sum())
    return k


def _bindings(defs, maps, size, kinds, urows) -> dict:
    """Per kind of run (interleaved ``lanes``, ``contig``uous), its bindings by kind and its runs and rows: clean (every
    binding shared, lane, shifted or param), affine (a binding steps by a non-power of two), peeled (affine once copy 0
    is taken off, as `tmpl.align` peels a chained run) or irregular (it splits into single copies, as `tmpl.align`
    splits)."""
    out = {}
    for d, D in enumerate(defs):
        L, m = D.cols, maps[d]
        for c, kind, M in kinds[d]:
            if c.count < 2:
                continue
            o = out.setdefault(kind, {"bindings": dict.fromkeys(("shared", "param", "lane", "shifted", "affine",
                                                                  "irregular"), 0),
                                      **{x: {"runs": 0, "rows": 0} for x in ("clean", "affine", "peeled",
                                                                             "irregular")}})
            j = np.arange(c.count, dtype=np.int64)[:, None]
            vs = [_seg_refs(s, j) for s in c.segs if s[0] * s[3] and s[2]]
            o["bindings"]["shared"] += sum(s[0] * s[3] for s in c.segs if not s[2])
            v = np.concatenate(vs, axis=1) if vs else np.zeros((c.count, 0), np.int64)
            k = _classify(v, L, m, M)
            for x, n in k.items():
                o["bindings"][x] += n
            if k["irregular"]:
                cls = "irregular"
                if c.count > 2:
                    k1 = _classify(v[1:], L, m, M)
                    cls = "irregular" if k1["irregular"] else "peeled"
            else:
                cls = "affine" if k["affine"] else "clean"
            o[cls]["runs"] += 1
            o[cls]["rows"] += c.count * urows[c.e]
    return {"by_kind": out}


def order(phi: np.ndarray, flat: T.Flat) -> dict:
    """Whether ``phi`` keeps ``flat``'s columns distinct and each row reading earlier columns."""
    n = len(flat.na)
    distinct = len(np.unique(phi)) == n
    bad = selfs = 0
    late: list[tuple[int, int]] = []
    step = 1 << 22
    for a in range(0, n, step):
        b = min(n, a + step)
        lo, hi = int(flat.ptr[a]), int(flat.ptr[b])
        r = np.repeat(np.arange(a, b, dtype=np.int64), np.diff(flat.ptr[a:b + 1]))
        v = flat.refs[lo:hi].astype(np.int64)
        keep = (v != flat.const) & (v != r)
        selfs += int((v == r).sum())
        hit = keep & (phi[v] >= phi[r])
        bad += int(hit.sum())
        late += list(zip(r[hit][:8 - len(late)].tolist(), v[hit][:8 - len(late)].tolist()))
    return {"distinct": bool(distinct), "late_refs": bad, "self_refs": selfs,
            "constant_last": bool(phi[flat.const] == phi.max()), "late": late}


def path(t: T.Tmpl, x: int) -> list:
    """The unit's column ``x`` as the chain of calls it sits in: (definition, call base, count, callee cols, copy,
    segments) per level, then the definition and local column it is a row of."""
    out, d = [], t.root
    while True:
        D = t.defs[d]
        hit = next((c for c in D.calls if c.base <= x < c.base + c.count * t.defs[c.e].cols), None)
        if hit is None:
            out.append((d, x))
            return out
        Le = t.defs[hit.e].cols
        i, x = divmod(x - hit.base, Le)
        out.append((d, hit.base, hit.count, Le, i, [s for s in hit.segs][:6], D.cols))
        d = hit.e


def digest(phi: np.ndarray) -> str:
    """``h ← h · 1000003 + phi[x]`` over the columns, mod 2^64: `flock-tmpl layout`'s digest of `Tmpl.posMap`."""
    n = len(phi)
    with np.errstate(over="ignore"):
        pw = np.concatenate([np.ones(1, np.uint64), np.cumprod(np.full(max(n - 1, 0), 1000003, np.uint64))])
        return str(int((phi.astype(np.uint64) * pw[::-1]).sum(dtype=np.uint64)))


MODES = {"today": ("today", None, False), "pow2": ("pow2", None, False),
         "interleaved/contiguous": ("interleaved", "contiguous", False), "interleaved/pow2": ("interleaved", "pow2", False),
         "interleaved/contiguous/strict": ("interleaved", "contiguous", True), "iv": ("interleaved", "interleave", False),
         "iv/strict": ("interleaved", "interleave", True), "iv/hybrid": ("interleaved", "interleave", False)}


def main(argv: list[str]) -> int:
    import argparse
    import resource
    import time

    ap = argparse.ArgumentParser(prog="tmpl_layouts.py")
    ap.add_argument("tmpl")
    ap.add_argument("--flat", metavar="CIRCUIT", help="check each layout's order on the flat unit")
    ap.add_argument("--unroll", action="store_true", help="check each layout's order on the template's unroll")
    ap.add_argument("--modes", default=",".join(MODES), help="layouts, comma-separated: " + ", ".join(MODES))
    ap.add_argument("--digest", action="store_true", help="each map's digest, as `flock-tmpl layout` prints it")
    a = ap.parse_args(argv)
    t0 = time.time()
    t = T.parse(open(a.tmpl).read())
    print("PARSED " + json.dumps({"seconds": round(time.time() - t0, 1), "defs": len(t.defs)}), flush=True)
    flat = T.read_flat(a.flat)[0] if a.flat else T.unroll(t) if a.unroll else None
    names_at = a.tmpl + ".names.json"
    names = json.load(open(names_at)) if os.path.exists(names_at) else None
    ok = True
    for name in a.modes.split(","):
        base, _, tc = name.partition("/hybrid/")
        coarse = int(tc) if tc else 0
        hybrid = bool(tc) or name.endswith("/hybrid")
        mode, fb, strict = MODES[base + "/hybrid" if tc else name]
        t1 = time.time()
        phi, r = layout(t, mode, fb or "contiguous", strict, hybrid=hybrid, coarse=coarse)
        r["hybrid"], r["coarse"] = hybrid, coarse
        if names:
            for x in r.get("gaps_by_def", []) + r.get("memo", {}).get("by_def", []):
                x["name"] = names[x["def"]]
        if mode == "today":
            r["identity"] = bool((phi == np.arange(len(phi))).all())
            ok &= r["identity"]
        if flat is not None and mode != "today":
            r["order"] = order(phi, flat)
            r["order"]["late_paths"] = [(path(t, a), path(t, b)) for a, b in r["order"]["late"][:3]]
            ok &= r["order"]["distinct"] and (fb == "interleave" or r["order"]["late_refs"] == 0)
        if a.digest:
            r["digest"] = digest(phi)
        r |= {"seconds": round(time.time() - t1, 1),
              "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
        print("LAYOUT " + json.dumps(r), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

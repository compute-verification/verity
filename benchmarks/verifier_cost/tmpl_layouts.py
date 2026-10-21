"""tmpl_layouts.py TMPL [--flat CIRCUIT]: the sizes of layouts of a mined `flock-ir-tmpl/v1` unit (`verity_flock.tmpl`),
as JSON lines (PRF-133).

* ``today``: the template's own layout, the flat unit as it is.
* ``pow2``: `tmpl.align`'s, each run of copies at a stride of its callee's size rounded up to a power of two.
* ``interleaved``: a run of n ≥ 2 copies of a Definition of laid-out size L takes L·2^m rows, 2^m ≥ n, copy i's local
  row j at ``start + j·2^m + i``, the start aligned to 2^m times the Definition's own alignment, so the copy index is the
  low m bits and j the high ones. Applied recursively, a copy's local row j already holds its own inner runs' copy bits,
  so the outermost run's copy bits are the lowest. A single copy keeps its columns, at its Definition's alignment. A run
  whose copies read one another (a binding into the run's own columns: chained, as a K loop's accumulator is) can't
  interleave and keep each row reading earlier columns, so it falls back to ``contiguous`` (its copies side by side at
  L, the carry fold's case) or ``pow2``.

Every layout keeps the unit's input rows at their columns and each output group's width, at a start aligned to its
words, as `tmpl.align` does. Per layout: the unit's rows and its slot's log (`circuit._slot_log`; compose refuses a
unit that fills its slot exactly), the runs by kind with the rows they hold unrolled, and, for interleaved, each run's
bound inputs over all its copies by kind: shared (one column for every copy), lane (copy i's column is ``w0 + i·s``
with s a power of two and ``w0`` a multiple of ``s·2^m``, so the fold's sum over copies is eq on a bit field), param (the
Definition's own input, bound by its caller), affine (steps evenly, not in a clean field) or irregular. ``--flat``
reads the flat unit and checks, under the interleaved map, that the columns are distinct and every row reads only
earlier columns (its own self row for an input, and the constant, excepted).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from verity_flock import tmpl as T

WORD = T.WORD


def _log(n: int) -> int:
    return max(0, (n - 1).bit_length())


def _slot_log(useful: int) -> int:
    return max(7, (useful - 1).bit_length())


def layout(t: T.Tmpl, mode: str, fallback: str = "contiguous") -> tuple[np.ndarray, dict]:
    """The map from the unit's columns to the layout's, and the layout's statistics."""
    defs = t.defs
    ins_g, outs_g = T._groups(t.groups)
    in_end = (ins_g[-1][0] + ins_g[-1][1]) * WORD if ins_g else 0
    out_rng = [(c * WORD, (c + w) * WORD) for c, w, _ in outs_g]
    urows = [0] * len(defs)
    maps: list[np.ndarray] = [np.zeros(0, np.int64)] * len(defs)
    size, align = [0] * len(defs), [1] * len(defs)
    kinds: list[list[tuple]] = [[] for _ in defs]  # per def: (call, kind, 2^m)
    st = {"runs": 0, "fixed_runs": 0, "lanes_runs": 0, "chained_runs": 0, "rows_lanes": 0, "rows_chained": 0,
          "pow2_runs": 0, "rows_pow2": 0}
    for d, D in enumerate(defs):
        root = d == t.root
        L = D.cols
        urows[d] = len(D.rows) + sum(c.count * urows[c.e] for c in D.calls)
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
                    pos = T._up(pos, next(w for c, w, _ in outs_g if c * WORD == p) * WORD)
                    done.add(p)
            m[x0:x1] = np.arange(pos, pos + x1 - x0)
            pos += x1 - x0

        for c in sorted(D.calls, key=lambda c: c.base):
            e, Le = c.e, defs[c.e].cols
            if c.base < x:
                raise T.NotAligned(f"definition {d}: calls overlap")
            if c.count > 1 and c.stride != Le:
                raise T.NotAligned(f"definition {d}: a call's copies have gaps")
            seq(x, c.base)
            width = c.count * Le
            inside = [r for r in out_rng if r[0] <= c.base < r[1]]
            M = 1
            st["runs"] += c.count > 1
            if root and (c.base < in_end or inside):
                if size[e] != Le:
                    raise T.NotAligned("a call in the input rows or an output group has padding inside")
                kind, S, nb = "contig", Le, pos
                st["fixed_runs"] += c.count > 1
            elif mode == "today":
                kind, S, nb = "contig", size[e], pos
            elif c.count == 1:
                kind, S, nb = "contig", size[e], T._up(pos, align[e])
                al = max(al, align[e])
            elif mode == "interleaved" and not _chained(c, Le):
                M = 1 << _log(c.count)
                kind, S, nb = "lanes", M, T._up(pos, M * align[e])
                al = max(al, M * align[e])
                st["lanes_runs"] += 1
                st["rows_lanes"] += c.count * urows[e]
            else:
                if mode == "pow2" or fallback == "pow2":
                    S = T._pow2(size[e])
                    nb = T._up(pos, S)
                    al = max(al, S)
                    st["pow2_runs"] += 1
                    st["rows_pow2"] += c.count * urows[e]
                else:
                    S, nb = size[e], T._up(pos, align[e])
                    al = max(al, align[e])
                kind = "contig"
                if mode == "interleaved":
                    st["chained_runs"] += 1
                    st["rows_chained"] += c.count * urows[e]
            j = np.arange(c.count, dtype=np.int64)[:, None]
            cols = c.base + j * Le + np.arange(Le, dtype=np.int64)[None, :]
            if kind == "lanes":
                m[cols] = nb + maps[e][None, :] * M + j
                pos = nb + size[e] * M
            else:
                m[cols] = nb + j * S + maps[e][None, :]
                pos = nb + (c.count - 1) * S + size[e]
            kinds[d].append((c, kind, M))
            x = c.base + width
        seq(x, L)
        maps[d], size[d], align[d] = m, pos, al
    n = size[t.root]
    out = {"mode": mode, "fallback": fallback if mode == "interleaved" else None, "useful": n,
           "slot_log": _slot_log(n), "slack": round(1 - n / (1 << _slot_log(n)), 4), "rows_unrolled": urows[t.root],
           **st}
    if mode == "interleaved":
        out |= _bindings(defs, maps, size, kinds, urows)
    return maps[t.root], out


def _chained(c: T.Call, Le: int) -> bool:
    lo, hi = c.base, c.base + c.count * Le
    return any(ln and sp and at < hi and at + (c.count - 1) * sp + ln > lo for ln, at, sp in c.segs)


def _bindings(defs, maps, size, kinds, urows) -> dict:
    k = {"shared": 0, "lane": 0, "param": 0, "affine": 0, "irregular": 0}
    clean = {"runs": 0, "rows": 0}
    unclean = {"runs": 0, "rows": 0, "affine_runs": 0, "irregular_runs": 0}
    for d, D in enumerate(defs):
        L, L2, m = D.cols, size[d], maps[d]
        for c, kind, M in kinds[d]:
            if kind != "lanes":
                continue
            j = np.arange(c.count, dtype=np.int64)[:, None]
            aff = irr = False
            for ln, at, sp in c.segs:
                if ln == 0:
                    continue
                if sp == 0:
                    k["shared"] += ln
                    continue
                v = at + j * sp + np.arange(ln, dtype=np.int64)[None, :]
                inp = v >= L
                if inp.all():
                    k["param"] += ln
                    continue
                if inp.any():
                    k["irregular"] += ln
                    irr = True
                    continue
                W = m[v]
                s = W[1] - W[0]
                even = (W == W[0][None, :] + j * s[None, :]).all(axis=0) & (s >= 0)
                lane = even & (s > 0) & ((s & (s - 1)) == 0) & (W[0] % np.maximum(s * M, 1) == 0)
                k["lane"] += int(lane.sum())
                k["affine"] += int((even & ~lane).sum())
                k["irregular"] += int((~even).sum())
                aff |= bool((even & ~lane).any())
                irr |= bool((~even).any())
            rows = c.count * urows[c.e]
            if aff or irr:
                unclean["runs"] += 1
                unclean["rows"] += rows
                unclean["affine_runs"] += aff and not irr
                unclean["irregular_runs"] += irr
            else:
                clean["runs"] += 1
                clean["rows"] += rows
    return {"bindings": k, "clean_lanes": clean, "unclean_lanes": unclean}


def order(phi: np.ndarray, flat: T.Flat) -> dict:
    """Whether ``phi`` keeps ``flat``'s columns distinct and each row reading earlier columns."""
    n = len(flat.na)
    distinct = len(np.unique(phi)) == n
    bad = selfs = 0
    step = 1 << 22
    for a in range(0, n, step):
        b = min(n, a + step)
        lo, hi = int(flat.ptr[a]), int(flat.ptr[b])
        r = np.repeat(np.arange(a, b, dtype=np.int64), np.diff(flat.ptr[a:b + 1]))
        v = flat.refs[lo:hi].astype(np.int64)
        keep = (v != flat.const) & (v != r)
        selfs += int((v == r).sum())
        bad += int((phi[v[keep]] >= phi[r[keep]]).sum())
    return {"distinct": bool(distinct), "late_refs": bad, "self_refs": selfs,
            "constant_last": bool(phi[flat.const] == phi.max())}


def main(argv: list[str]) -> int:
    import argparse
    import resource
    import time

    ap = argparse.ArgumentParser(prog="tmpl_layouts.py")
    ap.add_argument("tmpl")
    ap.add_argument("--flat", metavar="CIRCUIT", help="check the interleaved layout's order on the flat unit")
    a = ap.parse_args(argv)
    t0 = time.time()
    t = T.parse(open(a.tmpl).read())
    print("PARSED " + json.dumps({"seconds": round(time.time() - t0, 1), "defs": len(t.defs)}), flush=True)
    flat = T.read_flat(a.flat)[0] if a.flat else None
    ok = True
    for mode, fb in (("today", None), ("pow2", None), ("interleaved", "contiguous"), ("interleaved", "pow2")):
        t1 = time.time()
        phi, r = layout(t, mode, fb or "contiguous")
        if mode == "today":
            r["identity"] = bool((phi == np.arange(len(phi))).all())
            ok &= r["identity"]
        if flat is not None and mode != "today":
            r["order"] = order(phi, flat)
            ok &= r["order"]["distinct"] and r["order"]["late_refs"] == 0
        r |= {"seconds": round(time.time() - t1, 1),
              "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
        print("LAYOUT " + json.dumps(r), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""v2-hot's first atoms: the widest exact blocks, as curves, and a block table judged on them.  CPU only, over hot.py's
bitsets (ttout-restatements.md §8: the corrected table `v2hot-blocks-corrected.json` and, at 19:55Z, "report widths,
not verdicts").

For each unit (hot.py's `units/<fam>@<u>`, chain --chain), start t in --starts and window length L from --min-len to T - t,
exactness is ANDed over atoms [t, t + L) and restricted to Π's admitted rows and columns.  On each side (`rows`: a line is
a row and its partners are columns; `cols`: the mirror) two greedy paths give, for every line count K, a set of K lines
and the partners they share, a lower bound on the widest K-line block (a block found exists; a miss is not a certificate):
- `count`: the K lines with the most exact partners (stable order; hot_search.py's R_r), every K up to the lines that
  have any;
- `isect`: from the line with the most partners, add K - 1 times the line that keeps the most common partners (ties to
  the first in count order), up to --isect-max lines or until they share fewer than --isect-min-width (32; every width
  and row count in the block tables is above it).
A task stores the curves as steps ([K, width] at the last K of each width; 0 past the last step, and past `isect_to`
unknown), so any block table applies afterwards.  Gated per task at (t, --min-len) and (t, --min-len + 3): every curve
is recounted from the row bitsets unpacked to bools (exit 4 on a mismatch).

--table (a list of {L, Wstar, blocks: [[M, N], ...]}) and --floors (widthfloor.py's {floors: {L: {width_floor}}}; each
Wstar must be its ⌈width_floor⌉) give the blocks; a block M × N at (t, L) is found in a unit when the M-line row curves
reach N columns or the N-line column curves reach M rows.  Summary: per (start, L, block) the widest columns found with
M rows over the units (`required` N, `margin` N - widest) and the most rows found with N columns, where, and whether any
unit has the block; per start the tightest pass; per (t, L) the units' envelope of each side's curves (`envelope.json`);
per unit hot.py's liveness and admission counts.

Out: `<out>/units/<unit>.t<t>.json` per unit and start (--per-start; else `<unit>.json`), `summary.json`, `envelope.json`.
Restartable: a task saves its progress to `<tag>.part.json` every minute and once --budget-min is spent, and resumes
after the saved plane; exit 99 while tasks remain, 0 after the summary.  --summary-only judges existing curves against
another table.
    python3 benchmarks/pouw/sm120_exact/hot_blocks.py --units-dir <hot.py's out> --table T.json --floors F.json --out DIR
    python3 benchmarks/pouw/sm120_exact/hot_blocks.py --selftest
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool, TimeoutError as PoolTimeout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
import run_result  # noqa: E402
from hot import FAMILIES, jdump  # noqa: E402
from hot_search import popc  # noqa: E402

clock = time.time


def load_table(table: Path, floors: Path) -> tuple[dict[int, list[tuple[int, int]]], dict]:
    rows = json.loads(table.read_text())
    fl = json.loads(floors.read_text())["floors"]
    blocks, bad = {}, []
    for r in rows:
        L, W = int(r["L"]), int(r["Wstar"])
        if str(L) not in fl or W != math.ceil(fl[str(L)]["width_floor"]) or any(N < W for _, N in r["blocks"]):
            bad.append(L)
        blocks[L] = [(int(M), int(N)) for M, N in r["blocks"]]
    if bad:
        sys.exit(f"{table}: Wstar is not ⌈width_floor⌉ of {floors}, or a block is narrower, at L = {bad}")
    ident = {p.name: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (table, floors)}
    return blocks, ident


def isect_greedy(bits: np.ndarray, cnt: np.ndarray, K: int) -> tuple[int, list[int]]:
    """bits: (lines, P / 64) uint64, cnt their popcounts.  From the line with most partners, add K - 1 times the line that
    keeps the most common partners (ties in count order).  Returns the common partners of the K lines and the lines.
    The reference for isect_path."""
    cand = np.flatnonzero(cnt > 0)
    if len(cand) < K:
        return 0, []
    cand = cand[np.argsort(-cnt[cand], kind="stable")]
    X = bits[cand]
    cur = X[0].copy()
    chosen = [int(cand[0])]
    alive = np.ones(len(cand), bool)
    alive[0] = False
    for step in range(K - 1):
        s = popc(X & cur).sum(1, dtype=np.int64)
        s[~alive] = -1
        j = int(np.argmax(s))
        if s[j] <= 0:
            return 0, chosen
        cur &= X[j]
        alive[j] = False
        chosen.append(int(cand[j]))
        if step % 32 == 31:
            keep = alive & (s > 0)
            if keep.sum() < 0.75 * len(keep):
                X, cand, alive = X[keep], cand[keep], alive[keep]
            if alive.sum() < K - 2 - step:
                return 0, chosen
    return int(popc(cur).sum()), chosen


def isect_path(bits: np.ndarray, cnt: np.ndarray, K: int, batch: int = 256, stop_below: int = 1
               ) -> tuple[np.ndarray, list[int]]:
    """isect_greedy's path, once to K lines: widths[k - 1] is the common partners of its first k lines (0 once they share
    fewer than `stop_below`), so isect_greedy(bits, cnt, k) = widths[k - 1] for every k <= K where that is at least
    `stop_below`, and the lines chosen.  Lazy: each line keeps
    an upper bound (its common partners when last counted, which only fall), and a step counts only the `batch` lines of
    highest bound (then first in order); it takes their best unless a line outside could beat or tie it from earlier in
    the order, when it counts them all."""
    widths = np.zeros(K, np.int64)
    cand = np.flatnonzero(cnt > 0)
    if K == 0 or len(cand) == 0:
        return widths, []
    cand = cand[np.argsort(-cnt[cand], kind="stable")]
    X = bits[cand]
    cur = X[0].copy()
    chosen = [int(cand[0])]
    widths[0] = int(cnt[cand[0]])
    ub = cnt[cand].astype(np.int64)
    alive = np.ones(len(cand), bool)
    alive[0], ub[0] = False, -1
    for step in range(1, K):
        j = -1
        n = len(ub)
        if n > batch:
            p = np.argpartition(ub * (n + 1) - np.arange(n), n - batch)
            top, rest = p[n - batch:], p[:n - batch]
            s = popc(X[top] & cur).sum(1, dtype=np.int64)
            s[~alive[top]] = -1
            ub[top] = s
            mx = int(s.max())
            if mx > 0:
                jt = int(top[s == mx].min())
                out = ub[rest]
                umax = int(out.max())
                if mx > umax or (mx == umax and jt < int(rest[out == mx].min())):
                    j = jt
        if j < 0:
            s = popc(X & cur).sum(1, dtype=np.int64)
            s[~alive] = -1
            ub = s
            j = int(np.argmax(s))
        w = int(ub[j])
        if w < max(stop_below, 1):
            break
        cur &= X[j]
        alive[j], ub[j] = False, -1
        chosen.append(int(cand[j]))
        widths[step] = w
        if step % 32 == 0:
            keep = alive & (ub > 0)
            if not keep.any():
                break
            if keep.sum() < 0.75 * len(keep):
                X, cand, alive, ub = X[keep], cand[keep], alive[keep], ub[keep]
    return widths, chosen


def count_curve(bits: np.ndarray, cnt: np.ndarray, chunk: int = 256) -> np.ndarray:
    """widths[k - 1]: the partners shared by the k lines with most partners (stable order), for k up to the lines with any;
    ANDed `chunk` lines at a time until they share none."""
    order = np.argsort(-cnt, kind="stable")
    nz = int((cnt > 0).sum())
    widths = np.zeros(nz, np.int64)
    acc = None
    for i in range(0, nz, chunk):
        blk = bits[order[i:min(i + chunk, nz)]]
        if acc is not None:
            blk[0] &= acc
        blk = np.bitwise_and.accumulate(blk, axis=0)
        widths[i:i + len(blk)] = popc(blk).sum(1, dtype=np.int64)
        acc = blk[-1]
        if widths[i + len(blk) - 1] == 0:
            break
    return widths


def steps(w: np.ndarray) -> list[list[int]]:
    """A non-increasing curve as [K, width] at the last K of each positive width."""
    w = np.asarray(w)[: int((np.asarray(w) > 0).sum())]
    if len(w) == 0:
        return []
    ends = np.flatnonzero(np.diff(w) != 0).tolist() + [len(w) - 1]
    return [[e + 1, int(w[e])] for e in ends]


def at(st: list[list[int]], K: int) -> int:
    """A step curve's width at K lines (0 past its last step)."""
    i = bisect.bisect_left(st, [K, -1])
    return st[i][1] if i < len(st) else 0


def envelope(a: np.ndarray, st: list[list[int]]) -> np.ndarray:
    """The pointwise maximum of two step curves (an (n, 2) array and a list of steps), as an (n, 2) array of steps."""
    b = np.array(st, np.int64).reshape(-1, 2)
    E = np.union1d(a[:, 0], b[:, 0])
    if len(E) == 0:
        return a
    best = np.zeros(len(E), np.int64)
    for c in (a, b):
        if len(c):
            i = np.searchsorted(c[:, 0], E, "left")
            best = np.maximum(best, np.where(i < len(c), c[np.minimum(i, len(c) - 1), 1], 0))
    keep = np.append(best[:-1] != best[1:], True) & (best > 0)
    return np.stack([E[keep], best[keep]], 1)


def unit_curves(task) -> dict:
    """One task: a unit's curves at the given starts, written to `<tag>.json`.  Its progress is saved to `<tag>.part.json`
    every minute and once `deadline` (epoch seconds) passes, when it returns early; a later call resumes after the saved
    plane (the ANDs up to it are replayed, not the searches)."""
    ud, chain, starts, min_len, isect_max, isect_min_width, out_dir, tag, deadline = task
    ud, out_dir = Path(ud), Path(out_dir)
    meta = json.loads((ud / "unit.json").read_text())
    m, n, T = meta["m"], meta["n"], meta["planes"]
    okA, okB = np.load(ud / "okA.npy"), np.load(ud / "okB.npy")
    RB = np.memmap(ud / f"{chain}.rows.u32", np.uint64, "r", shape=(T, m, n // 64))
    CB = np.memmap(ud / f"{chain}.cols.u32", np.uint64, "r", shape=(T, n, m // 64))
    maskB = np.packbits(okB, bitorder="little").view(np.uint64)
    maskA = np.packbits(okA, bitorder="little").view(np.uint64)
    gate_at = {(t, L) for t in starts for L in (min_len, min_len + 3)}
    recs, kept, secs = [], {}, {"and": 0.0, "count": 0.0, "isect": 0.0}
    cur: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    dead: dict[int, int] = {}
    part = out_dir / "units" / f"{tag}.part.json"
    s_done = min(starts) - 1
    if part.exists():
        p = json.loads(part.read_text())
        recs, s_done, secs = p["records"], p["s_done"], p["seconds"]
        kept = {(k[0], k[1], k[2]): v for k, v in p["kept"]}
        dead = {int(t): L for t, L in p["dead"].items()}
    resumed_at, saved = s_done, clock()

    def save(s: int) -> None:
        tmp = part.with_suffix(".tmp")
        tmp.write_text(jdump({"records": recs, "kept": [[list(k), v] for k, v in kept.items()], "dead": dead, "s_done": s,
                              "seconds": secs}))
        tmp.replace(part)

    for s in range(min(starts), T):
        if s > s_done and clock() > deadline:
            if s - 1 > resumed_at:
                save(s - 1)
            return {"unit": tag, "partial": s - 1}
        if s > s_done and clock() - saved > 60:
            save(s - 1)
            saved = clock()
        c0 = time.perf_counter()
        pr, pc = np.asarray(RB[s]), np.asarray(CB[s])
        for t in starts:
            if s < t or t in dead:
                continue
            if s == t:
                r = pr & maskB
                r[~okA] = 0
                c = pc & maskA
                c[~okB] = 0
                cur[t] = (r, c)
            else:
                cur[t][0][...] &= pr
                cur[t][1][...] &= pc
            L = s - t + 1
            if L < min_len or s <= s_done:
                continue
            r, c = cur[t]
            rc, cc = popc(r).sum(1, dtype=np.int64), popc(c).sum(1, dtype=np.int64)
            if int(rc.sum()) == 0:
                dead[t] = L
                continue
            rec = {"t": t, "L": L, "exact": int(rc.sum())}
            for side, bits, cnt in (("rows", r, rc), ("cols", c, cc)):
                c1 = time.perf_counter()
                wc = count_curve(bits, cnt)
                c2 = time.perf_counter()
                K = min(isect_max, len(wc))
                wi, path = isect_path(bits, cnt, K, stop_below=isect_min_width)
                secs["count"] += c2 - c1
                secs["isect"] += time.perf_counter() - c2
                rec[side] = {"lines": len(wc), "count": steps(wc), "isect": steps(wi), "isect_to": K}
                if (t, L) in gate_at:
                    kept[(t, L, side)] = path
            recs.append(rec)
        secs["and"] += time.perf_counter() - c0
        if len(dead) == len(starts):
            break
    secs["and"] -= secs["count"] + secs["isect"]
    gates = []
    for t, L in sorted(gate_at):
        if t + L > T:
            continue
        E = okA[:, None] & okB[None, :]
        for s in range(t, t + L):
            E &= np.unpackbits(np.asarray(RB[s]).view(np.uint8), axis=1, bitorder="little").astype(bool)
        rec = next((x for x in recs if x["t"] == t and x["L"] == L), None)
        bad = 0
        for side, F in (("rows", E), ("cols", np.ascontiguousarray(E.T))):
            cnt = F.sum(1)
            nz = int((cnt > 0).sum())
            want = np.logical_and.accumulate(F[np.argsort(-cnt, kind="stable")[:nz]], axis=0).sum(1)
            path = kept.get((t, L, side), [])
            wpath = np.logical_and.accumulate(F[path], axis=0).sum(1) if path else np.zeros(0, np.int64)
            if rec is None:
                bad += int(nz != 0)
                continue
            got = rec[side]
            bad += int(got["lines"] != nz) + sum(at(got["count"], k + 1) != int(x) for k, x in enumerate(want))
            bad += sum(at(got["isect"], k + 1) != int(x) for k, x in enumerate(wpath))
            bad += int(any(at(got["isect"], k) for k in range(len(path) + 1, got["isect_to"] + 1)))
        gates.append({"t": t, "L": L, "mismatches": bad})
    res = {"unit": ud.name, "family": meta["family"], "chain": chain, "m": m, "n": n, "T": T, "starts": starts,
           "min_len": min_len, "isect_max": isect_max, "isect_min_width": isect_min_width,
           "rows_admitted": meta.get("rows_admitted"),
           "rows_live": meta.get("rows_live"), "dead_from": {str(t): L for t, L in dead.items()},
           "seconds": {k: round(v, 1) for k, v in secs.items()}, "gates": gates, "records": recs}
    (out_dir / "units" / f"{tag}.json").write_text(jdump(res) + "\n")
    part.unlink(missing_ok=True)
    return {"unit": tag, "gates": gates, "seconds": res["seconds"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-dir", help="hot.py's --out")
    ap.add_argument("--table", type=Path, help="the block table: a list of {L, Wstar, blocks}")
    ap.add_argument("--floors", type=Path, help="widthfloor.py's floors, which the table's Wstar must match")
    ap.add_argument("--chain", default="hot")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--units", type=int, default=4)
    ap.add_argument("--starts", default=",".join(map(str, range(17))))
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--isect-max", type=int, default=8192, help="the intersection path's most lines")
    ap.add_argument("--isect-min-width", type=int, default=32, help="the intersection path stops below this many partners")
    ap.add_argument("--per-start", action="store_true", help="one task and one file per unit and start")
    ap.add_argument("--summary-only", action="store_true", help="judge the curves already in --out against --table")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--budget-min", type=float, default=5.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.table and args.floors):
        ap.error("--table and --floors are required")
    table, ident = load_table(args.table, args.floors)
    t0 = clock()
    out = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    (out / "units").mkdir(parents=True, exist_ok=True)
    starts = [int(x) for x in args.starts.split(",")]
    names = [f"{f}@{u}" for u in range(1, args.units + 1) for f in args.families.split(",") if f]
    tasks = ([(nm, [t], f"{nm}.t{t}") for nm in names for t in starts] if args.per_start else
             [(nm, starts, nm) for nm in names])
    fh = open(out / "log.jsonl", "a")
    if args.summary_only:
        write_summary(out, names, starts, fh, tasks, table, ident)
        return
    src = Path(args.units_dir) / "units"
    todo = [x for x in tasks if not (out / "units" / f"{x[2]}.json").exists()]
    bad, more = 0, 0
    deadline = t0 + 60 * args.budget_min
    with Pool(args.procs) as pool:
        pending = [pool.apply_async(unit_curves, ((str(src / nm), args.chain, ts, args.min_len, args.isect_max,
                                                   args.isect_min_width, str(out), tag, deadline),)) for nm, ts, tag in todo]
        for (_, _, nm), p in zip(todo, pending):
            left = deadline - clock()
            try:  # a task returns at the first plane past the deadline; the extra 4 minutes cover that plane's searches
                res = p.get(timeout=max(left, 1) + 240)
            except Exception as e:  # a task overran (or failed); what the tasks saved is kept
                fh.write(json.dumps({"stopped": nm, "why": type(e).__name__, "detail": str(e)[:300]}) + "\n")
                fh.flush()
                pool.terminate()
                sys.exit(99 if isinstance(e, PoolTimeout) else 1)
            if "partial" in res:
                more += 1
                fh.write(jdump({"saved": nm, "through_plane": res["partial"], "seconds": round(clock() - t0, 1)}) + "\n")
            else:
                bad += sum(g["mismatches"] for g in res["gates"])
                fh.write(jdump({"done": nm, "gates": res["gates"], "task_seconds": res["seconds"],
                                "seconds": round(clock() - t0, 1)}) + "\n")
            fh.flush()
    if bad:
        fh.write(json.dumps({"gate_mismatches": bad}) + "\n")
        sys.exit(4)
    if more:
        fh.write(json.dumps({"chunk_end": {"tasks_left": more, "seconds": round(clock() - t0, 1)}}) + "\n")
        sys.exit(99)
    write_summary(out, names, starts, fh, tasks, table, ident)


def write_summary(out: Path, names: list[str], starts: list[int], fh, tasks: list[tuple],
                  table: dict[int, list[tuple[int, int]]], ident: dict) -> None:
    best: dict[tuple, dict] = {}
    curves: dict[tuple, dict[str, np.ndarray]] = {}
    gates = 0
    liveness, seconds = {}, {"and": 0.0, "count": 0.0, "isect": 0.0}
    for nm, _, tag in tasks:
        d = json.loads((out / "units" / f"{tag}.json").read_text())
        gates += sum(g["mismatches"] for g in d["gates"])
        for k in seconds:
            seconds[k] += d["seconds"][k]
        liveness[nm] = {"rows_live": d.get("rows_live"), "rows_admitted": d.get("rows_admitted"), "m": d["m"], "n": d["n"]}
        fam = d["family"]
        byL = {(r["t"], r["L"]): r for r in d["records"]}
        for t in d["starts"]:
            for L in range(d["min_len"], d["T"] - t + 1):
                r = byL.get((t, L))
                if r is None and not L >= d["dead_from"].get(str(t), 10**9):
                    sys.exit(f"{tag}: no record at (t, L) = ({t}, {L})")
                if r is not None:
                    cv = curves.setdefault((t, L), {"rows": np.zeros((0, 2), np.int64), "cols": np.zeros((0, 2), np.int64)})
                    for side in cv:
                        cv[side] = envelope(envelope(cv[side], r[side]["count"]), r[side]["isect"])
                for M, N in table.get(L, []):
                    wc, wi = (at(r["rows"]["count"], M), at(r["rows"]["isect"], M)) if r else (0, 0)
                    rn = max(at(r["cols"]["count"], N), at(r["cols"]["isect"], N)) if r else 0
                    w = max(wc, wi)
                    b = best.setdefault((t, L, M, N), {"t": t, "L": L, "M": M, "required": N, "widest": -1, "family": None,
                                                        "unit": None, "by": None, "rows_at_N": -1, "rows_family": None,
                                                        "rows_unit": None, "found_in": []})
                    if w > b["widest"]:
                        b.update(widest=w, family=fam, unit=nm, by="isect" if wi > wc else "count")
                    if rn > b["rows_at_N"]:
                        b.update(rows_at_N=rn, rows_family=fam, rows_unit=nm)
                    if w >= N or rn >= M:
                        b["found_in"].append(nm)
    rows = sorted(best.values(), key=lambda b: (b["t"], b["L"], b["M"]))
    for b in rows:
        b["passed"] = not b["found_in"]
        b["margin"] = b["required"] - b["widest"]
    by_start = {}
    for t in starts:
        rs = [b for b in rows if b["t"] == t]
        tight = min((b for b in rs if b["passed"]), key=lambda b: b["margin"] / b["required"], default=None)
        by_start[str(t)] = {"judged": len(rs), "found": sum(not b["passed"] for b in rs), "tightest_pass": tight}
    fails = [b for b in rows if not b["passed"]]
    tight = min((b for b in rows if b["passed"]), key=lambda b: b["margin"] / b["required"], default=None)
    env = [{"t": t, "L": L, "rows": cv["rows"].tolist(), "cols": cv["cols"].tolist()} for (t, L), cv in sorted(curves.items())]
    (out / "envelope.json").write_text(jdump({"units": names, "curves": env}) + "\n")
    doc = {"tool": "pouw_sm120_exact_hot_blocks", "units": names, "starts": starts, "inputs": ident,
           "gate_mismatches": gates, "blocks_judged": len(rows), "blocks_found": len(fails), "tightest_pass": tight,
           "by_start": by_start, "cpu_seconds": {k: round(v) for k, v in seconds.items()}, "liveness": liveness, "rows": rows}
    (out / "summary.json").write_text(jdump(doc) + "\n")
    fh.write(jdump({"summary": {"gate_mismatches": gates, "judged": len(rows), "found": len(fails)}}) + "\n")
    run_result.write(out / "summary.json", gates == 0, "sm120-exact-hot-blocks",
                     f"{len(names)} units, starts {starts[0]}..{starts[-1]}: {len(fails)} of {len(rows)} (start, length, "
                     f"block) found by the greedy searches; gate mismatches {gates}",
                     [("units", len(names), "count"), ("blocks_judged", len(rows), "count"),
                      ("blocks_found", len(fails), "count"), ("gate_mismatches", gates, "count")])


def selftest() -> None:
    """On a 1,024² unit of 14 atoms (words exact with probability 0.97, a 300 × 500 block exact over atoms 2 .. 9, and
    runs of duplicate rows, so the greedy paths meet ties): isect_path against isect_greedy at every K, a task run
    straight and one stopped and resumed every few planes (the same records), its gates, and the planted block; and
    envelope against the pointwise maximum."""
    global clock
    import tempfile
    rng = np.random.default_rng(7)
    m = n = 1024
    T = 14
    E = rng.random((T, m, n)) < 0.97
    rows = rng.choice(np.setdiff1d(np.arange(m), [5, *range(600, 640), *range(700, 720)]), 300, replace=False)
    cols = rng.choice(np.setdiff1d(np.arange(n), [9]), 500, replace=False)
    for t in range(2, 10):
        E[t][np.ix_(rows, cols)] = True
    E[:, 600:640] = E[:, 600:601]
    E[:, 700:720] = E[:, rows[0]:rows[0] + 1]
    okA, okB = np.ones(m, bool), np.ones(n, bool)
    okA[5], okB[9] = False, False
    bad = []
    for t, L in ((2, 4), (2, 8), (0, 3)):
        W = np.logical_and.reduce(E[t:t + L], axis=0) & okA[:, None] & okB[None, :]
        for F in (W, W.T):
            bits = np.ascontiguousarray(np.packbits(F, axis=1, bitorder="little")).view(np.uint64)
            cnt = popc(bits).sum(1, dtype=np.int64)
            K = min(340, int((cnt > 0).sum()))
            ref = [isect_greedy(bits, cnt, k)[0] for k in range(1, K + 1)]
            for batch in (16, 256):
                widths, _ = isect_path(bits, cnt, K, batch)
                bad += [("isect", t, L, batch, k + 1) for k in range(K) if ref[k] != widths[k]]
    env, cs = np.zeros((0, 2), np.int64), []
    for _ in range(6):
        cs.append(steps(np.sort(rng.integers(0, 40, int(rng.integers(0, 30))))[::-1]))
        env = envelope(env, cs[-1])
    bad += [("envelope", k) for k in range(1, 32) if at(env.tolist(), k) != max(at(c, k) for c in cs)]
    with tempfile.TemporaryDirectory() as d:
        ud = Path(d) / "units" / "syn@1"
        ud.mkdir(parents=True)
        np.packbits(E, axis=2, bitorder="little").tofile(ud / "hot.rows.u32")
        np.packbits(E.transpose(0, 2, 1), axis=2, bitorder="little").tofile(ud / "hot.cols.u32")
        np.save(ud / "okA.npy", okA)
        np.save(ud / "okB.npy", okB)
        (ud / "unit.json").write_text(json.dumps({"m": m, "n": n, "planes": T, "family": "syn"}))
        res = {}
        for mode in ("straight", "resumed"):
            o = Path(d) / mode
            (o / "units").mkdir(parents=True)
            ticks = iter(range(10**6))
            clock = (lambda: 0.0) if mode == "straight" else (lambda: float(next(ticks) % 4 == 3) * 1e9)
            for _ in range(40):
                r = unit_curves((str(ud), "hot", [2], 4, 8192, 32, str(o), "syn@1.t2", 1.0))
                if "partial" not in r:
                    break
            res[mode] = json.loads((o / "units" / "syn@1.t2.json").read_text())
        clock = time.time
    for k in ("records", "gates", "dead_from"):
        bad += [("resume", k)] if res["straight"][k] != res["resumed"][k] else []
    bad += [("gate", g) for g in res["straight"]["gates"] if g["mismatches"]]
    byL = {r["L"]: r for r in res["straight"]["records"]}
    planted = max(at(byL[8]["rows"]["count"], 300), at(byL[8]["rows"]["isect"], 300))
    over = max(at(byL[9]["rows"]["count"], 300), at(byL[9]["rows"]["isect"], 300))
    bad += [("planted", planted, over)] if not (planted >= 500 and over < 500) else []
    print(f"selftest: {len(res['straight']['records'])} windows, gates {res['straight']['gates']}; planted 300 rows at 8 "
          f"atoms share {planted} columns, at 9 atoms {over}; mismatches {bad[:10]}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()

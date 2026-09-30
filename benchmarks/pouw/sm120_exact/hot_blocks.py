#!/usr/bin/env python3
"""The exact blocks to rule out at v2-hot's first atoms: clause (c) of `no-aligned-exact-region/sm120-unpromoted-hot` as
restated at 17:30Z (ttout-restatements.md §8, "v2-hot's first atoms", the table "Exact blocks to rule out").  CPU only, over
hot.py's bitsets.

For each unit (hot.py's `units/<fam>@<u>`, chain --chain), start t in --starts and window length L from --min-len to T - t,
exactness is ANDed over atoms [t, t + L) and restricted to Π's admitted rows and columns.  Each block M × N (rows × columns)
that governs L (L's own row of BLOCKS, or the next listed length's when L has none) asks whether some M rows are all exact on
N common columns.  Three greedy searches, each a lower bound (a block found exists; a miss is not a certificate):
- `count`: the columns exact with the M rows that have the most exact columns (stable order; hot_search.py's R_r);
- `isect`, run where `count` misses: from the row with the most exact columns, add M - 1 times the row that keeps the most
  common columns;
- `rows_n`: the rows exact with the N columns that have the most exact rows, which finds the block when it reaches M.
Past the last listed length no block governs; there `count` at 64 rows is recorded for reference.  Gated per unit: at
(first start, --min-len) and (last start, --min-len + 3) every value is recounted from the row bitsets unpacked to bools
(exit 4 on a mismatch).

Out: `<out>/units/<unit>.json` (one record per start, length and block), then `summary.json`: per (start, L, block) the
widest columns found with M rows over the units, where, and whether any search found the block.  Restartable: exit 99 once
--budget-min is spent while units remain, 0 after the summary.
    python3 benchmarks/pouw/sm120_exact/hot_blocks.py --units-dir <hot.py's out> --out DIR [--procs 32]
"""
from __future__ import annotations

import argparse
import json
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
from hot_search import popc, prefix_rows  # noqa: E402

BLOCKS = {
    6: [(288, 3982)], 8: [(448, 2487)], 9: [(144, 2047)], 12: [(192, 1261)], 14: [(256, 841)], 16: [(216, 841)],
    17: [(144, 1056), (1056, 634)], 18: [(96, 1344), (144, 634)], 21: [(96, 1152), (1152, 634)], 24: [(192, 634)],
    28: [(128, 960), (960, 519)], 30: [(128, 896), (896, 519)], 33: [(144, 528), (264, 314)],
    36: [(96, 672), (144, 314)], 40: [(64, 3840), (3840, 314)], 42: [(96, 576), (576, 314)],
    48: [(64, 2400), (2400, 314)], 50: [(64, 2304), (2304, 314)], 60: [(64, 1920), (1920, 314)],
    72: [(64, 1600), (1600, 64)], 75: [(64, 1536), (1536, 64)], 120: [(64, 1280), (1280, 64)],
}
LINES = sorted({x for bl in BLOCKS.values() for b in bl for x in b} | {64})


def governing(L: int) -> tuple[int | None, list[tuple[int, int]]]:
    nxt = [x for x in sorted(BLOCKS) if x >= L]
    return (nxt[0], BLOCKS[nxt[0]]) if nxt else (None, [])


def isect_greedy(bits: np.ndarray, cnt: np.ndarray, K: int) -> tuple[int, list[int]]:
    """bits: (lines, P / 64) uint64, cnt their popcounts.  From the line with most partners, add K - 1 times the line that
    keeps the most common partners (ties in count order).  Returns the common partners of the K lines and the lines."""
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


def unit_blocks(task) -> dict:
    ud, chain, starts, min_len, isect, out_dir = task
    ud, out_dir = Path(ud), Path(out_dir)
    meta = json.loads((ud / "unit.json").read_text())
    m, n, T = meta["m"], meta["n"], meta["planes"]
    okA, okB = np.load(ud / "okA.npy"), np.load(ud / "okB.npy")
    RB = np.memmap(ud / f"{chain}.rows.u32", np.uint64, "r", shape=(T, m, n // 64))
    CB = np.memmap(ud / f"{chain}.cols.u32", np.uint64, "r", shape=(T, n, m // 64))
    maskB = np.packbits(okB, bitorder="little").view(np.uint64)
    maskA = np.packbits(okA, bitorder="little").view(np.uint64)
    gate_at = {(starts[0], min_len), (starts[-1], min_len + 3)}
    recs, kept = [], {}
    cur: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    dead: dict[int, int] = {}
    for s in range(min(starts), T):
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
            if L < min_len:
                continue
            r, c = cur[t]
            rc, cc = popc(r).sum(1, dtype=np.int64), popc(c).sum(1, dtype=np.int64)
            wc = dict(zip(LINES, prefix_rows(r, rc, LINES).tolist()))
            wr = dict(zip(LINES, prefix_rows(c, cc, LINES).tolist()))
            src, bl = governing(L)
            if not bl:
                recs.append({"t": t, "L": L, "src": None, "M": 64, "N": None, "count": wc[64]})
            for M, N in bl:
                rec = {"t": t, "L": L, "src": src, "M": M, "N": N, "count": wc[M], "rows_n": wr[N], "isect": None}
                if isect and wc[M] < N and wr[N] < M:
                    rec["isect"], rows = isect_greedy(r, rc, M)
                    if (t, L) in gate_at:
                        kept[(t, L, M)] = rows
                recs.append(rec)
            if int(rc.sum()) == 0:
                dead[t] = L
        if len(dead) == len(starts):
            break
    for t, L0 in dead.items():
        for L in range(L0 + 1, T - t + 1):
            src, bl = governing(L)
            recs += [{"t": t, "L": L, "src": src, "M": M, "N": N, "count": 0, "rows_n": 0, "isect": 0 if isect else None}
                     for M, N in bl] or [{"t": t, "L": L, "src": None, "M": 64, "N": None, "count": 0}]
    gates = []
    for t, L in sorted(gate_at):
        if t + L > T:
            continue
        E = okA[:, None] & okB[None, :]
        for s in range(t, t + L):
            E &= np.unpackbits(np.asarray(RB[s]).view(np.uint8), axis=1, bitorder="little").astype(bool)
        bad = 0
        for rec in (x for x in recs if x["t"] == t and x["L"] == L and x["N"] is not None):
            for F, R, key in ((E, rec["M"], "count"), (E.T, rec["N"], "rows_n")):
                cnt = F.sum(1)
                order = np.argsort(-cnt, kind="stable")
                want = int(F[order[:R]].all(0).sum()) if R <= int((cnt > 0).sum()) else 0
                bad += int(rec[key] != want)
            if rec["isect"] is not None:
                rows = kept.get((t, L, rec["M"]), [])
                want = int(E[rows].all(0).sum()) if len(rows) == rec["M"] else 0
                bad += int(rec["isect"] != want)
        gates.append({"t": t, "L": L, "mismatches": bad})
    name = ud.name
    res = {"unit": name, "family": meta["family"], "chain": chain, "m": m, "n": n, "T": T, "starts": starts,
           "min_len": min_len, "gates": gates, "records": recs}
    (out_dir / "units" / f"{name}.json").write_text(jdump(res) + "\n")
    return {"unit": name, "gates": gates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-dir", required=True, help="hot.py's --out")
    ap.add_argument("--chain", default="hot")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--units", type=int, default=4)
    ap.add_argument("--starts", default="0,1,2,3,4,5")
    ap.add_argument("--min-len", type=int, default=6)
    ap.add_argument("--no-isect", action="store_true")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--budget-min", type=float, default=5.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    src = Path(args.units_dir) / "units"
    out = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    (out / "units").mkdir(parents=True, exist_ok=True)
    starts = [int(x) for x in args.starts.split(",")]
    names = [f"{f}@{u}" for u in range(1, args.units + 1) for f in args.families.split(",") if f]
    todo = [nm for nm in names if not (out / "units" / f"{nm}.json").exists()]
    fh = open(out / "log.jsonl", "a")
    bad = 0
    with Pool(args.procs) as pool:
        pending = [pool.apply_async(unit_blocks, ((str(src / nm), args.chain, starts, args.min_len, not args.no_isect,
                                                   str(out)),)) for nm in todo]
        for nm, p in zip(todo, pending):
            left = 60 * args.budget_min - (time.time() - t0)
            try:
                res = p.get(timeout=max(left, 1))
            except Exception as e:  # the budget ran out (or a unit failed); units written so far are kept
                fh.write(json.dumps({"stopped": nm, "why": type(e).__name__, "detail": str(e)[:300]}) + "\n")
                fh.flush()
                pool.terminate()
                sys.exit(99 if isinstance(e, PoolTimeout) else 1)
            bad += sum(g["mismatches"] for g in res["gates"])
            fh.write(jdump({"done": nm, "gates": res["gates"], "seconds": round(time.time() - t0, 1)}) + "\n")
            fh.flush()
    if bad:
        fh.write(json.dumps({"gate_mismatches": bad}) + "\n")
        sys.exit(4)
    write_summary(out, names, starts, fh)


def write_summary(out: Path, names: list[str], starts: list[int], fh) -> None:
    best: dict[tuple, dict] = {}
    gates = 0
    for nm in names:
        d = json.loads((out / "units" / f"{nm}.json").read_text())
        gates += sum(g["mismatches"] for g in d["gates"])
        for r in d["records"]:
            key = (r["t"], r["L"], r["M"], r["N"])
            w = max(r["count"], r.get("isect") or 0)
            found = r["N"] is not None and (w >= r["N"] or r["rows_n"] >= r["M"])
            b = best.setdefault(key, {"t": r["t"], "L": r["L"], "src": r["src"], "M": r["M"], "N": r["N"], "widest": -1,
                                      "unit": None, "by": None, "rows_n": 0, "found_in": []})
            if w > b["widest"]:
                b.update(widest=w, unit=nm, by="isect" if (r.get("isect") or 0) > r["count"] else "count")
            b["rows_n"] = max(b["rows_n"], r.get("rows_n") or 0)
            if found:
                b["found_in"].append(nm)
    rows = sorted(best.values(), key=lambda b: (b["L"], b["t"], b["M"]))
    for b in rows:
        b["passed"] = None if b["N"] is None else not b["found_in"]
        b["margin"] = None if b["N"] is None else b["N"] - b["widest"]
    judged = [b for b in rows if b["N"] is not None]
    fails = [b for b in judged if not b["passed"]]
    tight = min((b for b in judged if b["passed"]), key=lambda b: b["margin"] / b["N"], default=None)
    doc = {"tool": "pouw_sm120_exact_hot_blocks", "units": names, "starts": starts, "blocks": {str(k): v for k, v in BLOCKS.items()},
           "gate_mismatches": gates, "blocks_judged": len(judged), "blocks_found": len(fails), "tightest_pass": tight,
           "rows": rows}
    (out / "summary.json").write_text(jdump(doc) + "\n")
    fh.write(jdump({"summary": {"gate_mismatches": gates, "judged": len(judged), "found": len(fails)}}) + "\n")
    run_result.write(out / "summary.json", gates == 0, "sm120-exact-hot-blocks",
                     f"{len(names)} units, starts {starts}: {len(fails)} of {len(judged)} (start, length, block) found by "
                     f"the greedy searches; gate mismatches {gates}",
                     [("units", len(names), "count"), ("blocks_judged", len(judged), "count"),
                      ("blocks_found", len(fails), "count"), ("gate_mismatches", gates, "count")])


if __name__ == "__main__":
    main()

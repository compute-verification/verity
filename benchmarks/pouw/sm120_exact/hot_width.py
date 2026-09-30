#!/usr/bin/env python3
"""The widest exact width per window from v2-hot's H_i, for the pre-add closure's width floors W*(L) (bc-3006c44a,
theory-pearl-c-sm120.md §14, addendum 16:55Z; min-merge-search/widthfloor.py).  CPU only, over hot.py's bitsets.

For each unit (hot.py's `units/<fam>@<u>`, chain --chain), each start t in --starts and every window length L from
--min-len to T - t: AND the chain's per-atom exactness over atoms [t, t + L), restricted to Π's admitted rows and columns,
and for each count R in --lines (default 8 .. 2,048; an m16n8 atom is 16 rows by 8 columns):
- `Wc[R]`: the columns exact with all of the R rows with the most exact words (stable order), hot_search.py's R_r, a width
  in columns;
- `Wr[R]`: the rows exact with all of the R columns with the most exact words, hot_search.py's R_c, the same with the sides
  swapped.
The greedy choice of lines makes both lower bounds on the widest exact region with R lines.  One pass over the planes serves
every start.  Gated per unit: at (t, L) = (first start, --min-len) and (last start, --min-len + 3), both widths are recounted
from the row bitsets unpacked to bools (exit 4 on a mismatch).

Out: `<out>/units/<unit>.npz` (Wc, Wr: starts x (T + 1) x lines, -1 where not measured; total: exact words per (t, L)) and
`.json`, then `summary.npz` (the maxima over each family's units) and `summary.json`.  Restartable: exit 99 once
--budget-min is spent while units remain, 0 after the summary.
    python3 benchmarks/pouw/sm120_exact/hot_width.py --units-dir <hot.py's out> --out DIR [--procs 16]
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


def unit_widths(task) -> dict:
    ud, chain, starts, min_len, lines, out_dir = task
    ud, out_dir = Path(ud), Path(out_dir)
    meta = json.loads((ud / "unit.json").read_text())
    m, n, T = meta["m"], meta["n"], meta["planes"]
    okA, okB = np.load(ud / "okA.npy"), np.load(ud / "okB.npy")
    RB = np.memmap(ud / f"{chain}.rows.u32", np.uint64, "r", shape=(T, m, n // 64))
    CB = np.memmap(ud / f"{chain}.cols.u32", np.uint64, "r", shape=(T, n, m // 64))
    maskB = np.packbits(okB, bitorder="little").view(np.uint64)
    maskA = np.packbits(okA, bitorder="little").view(np.uint64)
    S = len(starts)
    Wc = np.full((S, T + 1, len(lines)), -1, np.int32)
    Wr = np.full_like(Wc, -1)
    total = np.zeros((S, T + 1), np.int64)
    cur: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    live = set(range(S))
    for s in range(min(starts), T):
        pr, pc = np.asarray(RB[s]), np.asarray(CB[s])
        for q in sorted(live):
            t = starts[q]
            if s < t:
                continue
            if s == t:
                r = pr & maskB
                r[~okA] = 0
                c = pc & maskA
                c[~okB] = 0
                cur[q] = (r, c)
            else:
                cur[q][0][...] &= pr
                cur[q][1][...] &= pc
            L = s - t + 1
            if L < min_len:
                continue
            r, c = cur[q]
            rc, cc = popc(r).sum(1, dtype=np.int64), popc(c).sum(1, dtype=np.int64)
            total[q, L] = int(rc.sum())
            Wc[q, L] = prefix_rows(r, rc, lines)
            Wr[q, L] = prefix_rows(c, cc, lines)
            if total[q, L] == 0:
                Wc[q, L + 1:] = 0
                Wr[q, L + 1:] = 0
                live.discard(q)
        if not live:
            break
    gates = []
    for q, L in ((0, min_len), (S - 1, min_len + 3)):
        t = starts[q]
        if t + L > T:
            continue
        E = okA[:, None] & okB[None, :]
        for s in range(t, t + L):
            E &= np.unpackbits(np.asarray(RB[s]).view(np.uint8), axis=1, bitorder="little").astype(bool)
        bad = 0
        for axis, got in ((1, Wc), (0, Wr)):
            F = E if axis == 1 else E.T
            cnt = F.sum(1)
            order = np.argsort(-cnt, kind="stable")
            nz = int((cnt > 0).sum())
            for i, R in enumerate(lines):
                want = int(F[order[:R]].all(0).sum()) if R <= nz else 0
                bad += int(got[q, L, i] != want)
        gates.append({"t": t, "L": L, "mismatches": bad})
    name = ud.name
    np.savez_compressed(out_dir / "units" / f"{name}.npz", Wc=Wc, Wr=Wr, total=total, starts=np.array(starts),
                        lines=np.array(lines))
    res = {"unit": name, "family": meta["family"], "chain": chain, "m": m, "n": n, "T": T, "starts": starts,
           "min_len": min_len, "lines": lines, "gates": gates,
           "widest": {str(t): {"Wc": Wc[q, min_len:].max(0).tolist(), "Wr": Wr[q, min_len:].max(0).tolist()}
                      for q, t in enumerate(starts)}}
    (out_dir / "units" / f"{name}.json").write_text(jdump(res, indent=1) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-dir", required=True, help="hot.py's --out")
    ap.add_argument("--chain", default="hot")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--units", type=int, default=4)
    ap.add_argument("--starts", default="0,1,2,3,4,5")
    ap.add_argument("--min-len", type=int, default=9)
    ap.add_argument("--lines", default="8,16,32,64,128,256,512,1024,2048")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--budget-min", type=float, default=5.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    src = Path(args.units_dir) / "units"
    out = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    (out / "units").mkdir(parents=True, exist_ok=True)
    starts = [int(x) for x in args.starts.split(",")]
    lines = [int(x) for x in args.lines.split(",")]
    names = [f"{f}@{u}" for u in range(1, args.units + 1) for f in args.families.split(",") if f]
    todo = [nm for nm in names if not (out / "units" / f"{nm}.json").exists()]
    fh = open(out / "log.jsonl", "a")
    bad = 0
    with Pool(args.procs) as pool:
        pending = [pool.apply_async(unit_widths, ((str(src / nm), args.chain, starts, args.min_len, lines, str(out)),))
                   for nm in todo]
        for nm, p in zip(todo, pending):
            left = 60 * args.budget_min - (time.time() - t0)
            try:
                res = p.get(timeout=max(left, 1))
            except Exception as e:  # the budget ran out (or a unit failed); units written so far are kept
                fh.write(json.dumps({"stopped": nm, "why": type(e).__name__}) + "\n")
                fh.flush()
                pool.terminate()
                sys.exit(99 if isinstance(e, PoolTimeout) else 1)
            bad += sum(g["mismatches"] for g in res["gates"])
            fh.write(jdump({"done": nm, "gates": res["gates"], "seconds": round(time.time() - t0, 1)}) + "\n")
            fh.flush()
    if bad:
        fh.write(json.dumps({"gate_mismatches": bad}) + "\n")
        sys.exit(4)
    write_summary(out, names, starts, lines, args.min_len, fh)


def write_summary(out: Path, names: list[str], starts: list[int], lines: list[int], min_len: int, fh) -> None:
    fams = sorted({nm.split("@")[0] for nm in names})
    arrs, gates = {}, 0
    for f in fams:
        us = [np.load(out / "units" / f"{nm}.npz") for nm in names if nm.split("@")[0] == f]
        arrs[f"{f}.Wc"] = np.max([u["Wc"] for u in us], axis=0)
        arrs[f"{f}.Wr"] = np.max([u["Wr"] for u in us], axis=0)
    for nm in names:
        gates += sum(g["mismatches"] for g in json.loads((out / "units" / f"{nm}.json").read_text())["gates"])
    np.savez_compressed(out / "summary.npz", starts=np.array(starts), lines=np.array(lines), **arrs)
    widest = {f: {str(t): {"Wc": arrs[f"{f}.Wc"][q, min_len:].max(0).tolist(), "Wr": arrs[f"{f}.Wr"][q, min_len:].max(0).tolist()}
                  for q, t in enumerate(starts)} for f in fams}
    doc = {"tool": "pouw_sm120_exact_hot_width", "units": names, "starts": starts, "lines": lines, "min_len": min_len,
           "gate_mismatches": gates, "widest_over_lengths": widest}
    (out / "summary.json").write_text(jdump(doc, indent=1) + "\n")
    R0 = min(lines)
    top = max(max(max(v["Wc"][lines.index(R0)], v["Wr"][lines.index(R0)]) for v in d.values()) for d in widest.values())
    fh.write(jdump({"summary": {"gate_mismatches": gates, f"widest_at_{R0}_lines": top}}) + "\n")
    run_result.write(out / "summary.json", gates == 0, "sm120-exact-hot-width",
                     f"{len(names)} units, starts {starts}, windows of {min_len}+ atoms: widest exact width with {R0} lines "
                     f"{top}; gate mismatches {gates}",
                     [("units", len(names), "count"), (f"widest_width_{R0}_lines", top, "count"),
                      ("gate_mismatches", gates, "count")])


if __name__ == "__main__":
    main()

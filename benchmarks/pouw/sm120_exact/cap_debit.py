#!/usr/bin/env python3
"""`v1-cap600`'s honest-completeness cost (the assessor, 12:47 AM PDT): how many honest tiles does a 1/600 debit cap reject
that 1/400 accepts?  CPU only.

Per item, one 64 x 64 x k tile: 64 admitted A rows against 64 admitted weight rows, both formed as v1 forms them
(pairing.py's `item_sides`: `form_s5` at delta 1, line norm 16, the domain rules), with v1's replayed debit
(`pearlc_census.debit_parts` at G = 4: identities, promotion identities and every joint skip inside one promotion group that
leaves the word's total unchanged).  A tile's debit is its flagged share of (word, atom) pairs; under a cap ρ an honest tile
whose debit exceeds ρ is rejected, so the cost of a cap is the share of honest tiles over it.  Items are `acts:<linear>` of
a pearlc_capture directory (--acts; real tokens and the model's weights) or `unit:<family>` (the census families).

    python benchmarks/pouw/sm120_exact/cap_debit.py --acts DIR [--linears a,b,...] [--workers N] [--out DIR]
    python benchmarks/pouw/sm120_exact/cap_debit.py --units gaussian,coherent-gaussian [--k 8192]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import pairing as PR  # noqa: E402
import pearlc_census as census  # noqa: E402
import run_result  # noqa: E402
from run import ATOM  # noqa: E402

CAPS = {"1/400": 1 / 400, "1/600": 1 / 600, "1/1000": 1 / 1000}
TILE = 64


def tile(item: str, k: int, acts: str | None, seed: int) -> dict:
    census.set_atom(ATOM)
    t0 = time.time()
    sides, k, info = PR.item_sides(item, k, TILE, Path(acts) if acts else None, seed)
    (a, oka), (b, okb) = sides["A"], sides["B"]
    a, b = a[oka][:TILE], b[okb][:TILE]
    rec = {"item": item, **info, "rows": int(len(a)), "cols": int(len(b))}
    if len(a) == 0 or len(b) == 0 or k % 128:
        rec["skipped"] = "no admitted rows" if k % 128 == 0 else f"k = {k} isn't a whole number of 4-atom groups"
        return rec
    ii, jj = np.meshgrid(np.arange(len(a)), np.arange(len(b)), indexing="ij")
    flags, eF, _ = census.debit_parts(a[ii.ravel()], b[jj.ravel()], 4)
    T = k // 32
    rec.update({"atoms": T, "debit": round(float(flags.mean()), 8), "debit_max_word": round(float(flags.sum(1).max()) / T, 6),
                "eF_share": round(float(eF.mean()) / T, 8), "seconds": round(time.time() - t0, 1)})
    rec["over"] = {name: bool(rec["debit"] > cap) for name, cap in CAPS.items()}
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=None, help="a pearlc_capture directory")
    ap.add_argument("--linears", default=None, help="comma-separated; default every linear in the manifest")
    ap.add_argument("--units", default=None, help="census families instead of a capture")
    ap.add_argument("--k", type=int, default=8192, help="k for --units")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.units:
        items = [f"unit:{f}" for f in args.units.split(",")]
    else:
        man = json.loads((Path(args.acts) / "manifest.json").read_text())
        names = args.linears.split(",") if args.linears else list(man["inputs"])
        items = [f"acts:{n}" for n in names]
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(items)))) as ex:
        recs = list(ex.map(tile, items, [args.k] * len(items), [args.acts] * len(items), [args.seed] * len(items)))
    done = [r for r in recs if "debit" in r]
    for r in recs:
        print(json.dumps({k: r[k] for k in ("item", "rows", "cols", "debit", "over", "skipped") if k in r}), flush=True)
    summary = {"tool": "pouw_sm120_cap_debit", "acts": args.acts, "tile": [TILE, TILE], "caps": CAPS, "tiles": recs,
               "tiles_measured": len(done),
               "debit_max": max((r["debit"] for r in done), default=None),
               "debit_quantiles": {q: float(np.quantile([r["debit"] for r in done], q)) for q in (0.5, 0.9, 0.99)} if done else {},
               "rejected_share": {name: (sum(r["over"][name] for r in done) / len(done) if done else None) for name in CAPS}}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("tiles_measured", "debit_max", "debit_quantiles", "rejected_share")}), flush=True)
    rj = summary["rejected_share"]
    run_result.write(path, bool(done), "sm120-cap-debit",
                     f"{len(done)} honest 64x64 tiles at v1's forming ({args.acts or args.units}); max tile debit "
                     f"{summary['debit_max']}; rejected at 1/400 {rj['1/400']}, at 1/600 {rj['1/600']}, at 1/1000 {rj['1/1000']}",
                     [("tiles_measured", len(done), "count"), ("debit_max", summary["debit_max"] or 0.0, "share"),
                      *((f"rejected_share_{n.replace('/', '_')}", v or 0.0, "share") for n, v in rj.items())])


if __name__ == "__main__":
    main()

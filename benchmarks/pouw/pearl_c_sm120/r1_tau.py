#!/usr/bin/env python3
"""The per-row cheap test's honest rejection rate (the store's docs/pouw/new-designs.md §3.6-§3.7, the red team's 7:48 AM
PDT point 3): over r1_tiles.py's outputs on model rows, the share of verifier tiles whose most-cheap row exceeds tau_row,
per class and tile size, with the tiles holding row 0 (the attention sink's position, when the capture starts there) apart.

  python r1_tau.py OUT_DIR... [--taus 0.2,0.25,0.3] [--json FILE]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--taus", default="0.2,0.25,0.3")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    taus = [float(t) for t in args.taus.split(",")]
    vals: dict = defaultdict(lambda: {"row0": [], "rest": []})
    for d in args.dirs:
        first = int((d / "first-row").read_text()) if (d / "first-row").exists() else 0
        for f in sorted(d.glob("L*.json")):
            r = json.loads(f.read_text())
            for T, v in r["tiles"].items():
                rc = np.asarray(v["per_tile"]["row_cheap"])
                key = (r["class"], int(T))
                if first == 0:
                    vals[key]["row0"] += rc[0].ravel().tolist()
                    vals[key]["rest"] += rc[1:].ravel().tolist()
                else:
                    vals[key]["rest"] += rc.ravel().tolist()
    out = []
    for (cls, T), v in sorted(vals.items()):
        for part in ("rest", "row0"):
            x = np.asarray(v[part])
            if not len(x):
                continue
            row = {"class": cls, "tile": T, "tiles": part, "count": int(len(x)), "max": float(x.max()),
                   "p99": float(np.quantile(x, 0.99)), **{f"over_{t}": float((x > t).mean()) for t in taus}}
            out.append(row)
            print(f"{cls:10s} T{T:<4d} {part:4s} tiles {len(x):6d}  max {x.max():.3f}  p99 {row['p99']:.3f}  "
                  + "  ".join(f">{t:g}: {row[f'over_{t}']:.4f}" for t in taus))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

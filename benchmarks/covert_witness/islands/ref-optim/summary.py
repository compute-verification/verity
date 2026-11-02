"""ref-optim's scoreboard: for each start, the lowest valid bpb and the fewest training FLOPs to each target
(start x (1 - 0.5%, 1%, 2%, 5%)) over every reference record's curve, from the launcher's ledger and the fetched run dirs.

    python benchmarks/covert_witness/islands/ref-optim/summary.py [--ledger ~/.research/covert-witness.jsonl] [--rows]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STARTS = {"m100": 1.0561, "m50": 1.1276, "s100": 1.1647, "s50": 1.2563}
CUTS = (0.005, 0.01, 0.02, 0.05)
RUNS = Path.home() / ".research" / "runs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ledger", type=Path, default=Path.home() / ".research" / "covert-witness.jsonl")
    ap.add_argument("--rows", action="store_true", help="every reference row, not only the best")
    a = ap.parse_args()
    rows = [json.loads(x) for x in a.ledger.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["class"] == "reference" and r.get("valid")]
    for start, base in STARTS.items():
        mine = [r for r in rows if r["start"] == start]
        if not mine:
            continue
        points = []
        for r in mine:
            w = RUNS / r["run"] / "out" / "witness.json"
            curve = json.loads(w.read_text())["curve"] if w.exists() else [{"train_flops": r["train_flops"], "bpb": r["bpb"]}]
            points += [(p["train_flops"], p["bpb"], r) for p in curve if isinstance(p["bpb"], float)]
        best = min(mine, key=lambda r: r["bpb"])
        print(f"{start} (start {base}): best {best['bpb']:.4f} ({(best['bpb'] / base - 1) * 100:+.2f}%) "
              f"{best['run']} {best['art']} [{best['note']}]")
        for c in CUTS:
            ok = [p for p in points if p[1] <= base * (1 - c)]
            if ok:
                f, b, r = min(ok, key=lambda p: p[0])
                print(f"  -{c * 100:g}% ({base * (1 - c):.4f}): {f:.3e} FLOPs, bpb {b:.4f}, {r['run']} [{r['note']}]")
            else:
                print(f"  -{c * 100:g}% ({base * (1 - c):.4f}): not reached")
        if a.rows:
            for r in sorted(mine, key=lambda r: r["bpb"]):
                print(f"    {r['bpb']:.4f} {r['train_flops']:.3e} {r['run']} [{r['note']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ref-optim's scoreboard: for each start, the lowest valid bpb and the fewest training FLOPs to each target
(start x (1 - 0.5%, 1%, 2%, 5%)) over every reference record's curve, from the launcher's ledger and the fetched run dirs.
Only records at the pinned measurement commit (`pinned.json`) count, each against the start bpb its witness carries.

    python benchmarks/covert_witness/islands/ref-optim/summary.py [--ledger ~/.research/covert-witness.jsonl] [--rows]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CUTS = (0.005, 0.01, 0.02, 0.05)
RUNS = Path.home() / ".research" / "runs"
PINNED = json.loads((Path(__file__).resolve().parents[2] / "pinned.json").read_text())
PIN = PINNED["commit"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ledger", type=Path, default=Path.home() / ".research" / "covert-witness.jsonl")
    ap.add_argument("--rows", action="store_true", help="every reference row, not only the best")
    ap.add_argument("--pin", default=PIN, help="another measurement commit (default: the pin)")
    a = ap.parse_args()
    rows = [json.loads(x) for x in a.ledger.read_text().splitlines() if x.strip()]
    pins = {a.pin, *PINNED.get("same_version", [])} if a.pin == PIN else {a.pin}
    rows = [r for r in rows if r["class"] == "reference" and r.get("valid") and r.get("measurement_commit") in pins]
    for start in sorted({r["start"] for r in rows}):
        mine, points, bases = [r for r in rows if r["start"] == start], [], []
        for r in mine:
            w = json.loads((RUNS / r["run"] / "out" / "witness.json").read_text())
            bases.append(w["start_bpb"] if isinstance(w.get("start_bpb"), float) else w["score"]["start_bpb"])
            points += [(p["train_flops"], p["bpb"], r) for p in w["curve"] if isinstance(p["bpb"], float)]
        base = statistics.median(bases)
        best = min(mine, key=lambda r: r["bpb"])
        print(f"{start} (start {base:.4f}): best {best['bpb']:.4f} ({(best['bpb'] / base - 1) * 100:+.2f}%) "
              f"{best['run']} {best['art'][4:16]} [{best['note']}]")
        for c in CUTS:
            ok = [p for p in points if p[1] <= base * (1 - c)]
            if ok:
                f, b, r = min(ok, key=lambda p: p[0])
                print(f"  -{c * 100:g}% ({base * (1 - c):.4f}): {f:.3e} FLOPs, bpb {b:.4f}, {r['art'][4:16]} [{r['note']}]")
            else:
                print(f"  -{c * 100:g}% ({base * (1 - c):.4f}): not reached")
        if a.rows:
            for f, b, r in sorted(points, key=lambda p: (p[2]["run"], p[0])):
                print(f"    {b:.4f} {f:.3e} {r['art'][4:16]} [{r['note'][:60]}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

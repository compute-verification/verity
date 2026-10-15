"""Render attacks.jsonl (attacks.cu's JSON lines) as the per-alphabet table: every gated route's median time against the
int8 GEMM's at each n, the gates, and the clocks seen while timing.

    python benchmarks/d4_attacks/table.py OUT/attacks.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def main(path: str) -> None:
    gates, summ, clocks, controls, unsupported = {}, defaultdict(dict), defaultdict(list), [], []
    cur = None
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        e = json.loads(line)
        ev = e["event"]
        if ev == "start":
            cur = (e["n"], e["alphabet"])
        elif ev == "gate":
            gates[(cur, e["route"])] = e
        elif ev == "control":
            controls.append((cur, e))
        elif ev == "unsupported":
            unsupported.append((cur, e))
        elif ev == "rep":
            clocks[cur].extend([e["clk0"], e["clk1"]])
        elif ev == "summary":
            summ[(e["n"], e["alphabet"])][e["route"]] = e
    print("| n | alphabet | route | gate (mismatches / checked) | median ms | ratio vs int8 | MAC/SM/clk @2092 |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(summ):
        for route, e in sorted(summ[key].items(), key=lambda kv: kv[1]["ratio_vs_int8"]):
            g = gates.get((key, route), {})
            gate = f"{g.get('mismatches', '?')} / {g.get('checked', '?')} vs {g.get('against', '?')}"
            print(f"| {key[0]} | {key[1]} | {route} | {gate} | {e['median_ms']:.4f} | {e['ratio_vs_int8']:.3f} | "
                  f"{e['macs_per_sm_clk_at_2092']:.0f} |")
    failed = [(k, r, g) for (k, r), g in gates.items() if g["mismatches"]]
    print("\nGates failed (route not timed):", ", ".join(f"{k[1]}@{k[0]} {r} {g['mismatches']}" for k, r, g in failed)
          or "none")
    print("Controls:", ", ".join(f"{k[1]}@{k[0]} rejected={c['rejected']} ({c['mismatches']})" for k, c in controls))
    if unsupported:
        print("Unsupported:", ", ".join(f"{k[1]}@{k[0]} {u['route']}" for k, u in unsupported))
    allc = [c for v in clocks.values() for c in v if c]
    if allc:
        print(f"SM clock over timed items: min {min(allc)} max {max(allc)} MHz")
    print("\n| alphabet | n | fastest exact route | time / int8 |")
    print("|---|---|---|---|")
    for key in sorted(summ, key=lambda k: (k[1], k[0])):
        best = min(summ[key].values(), key=lambda e: e["median_ms"])
        print(f"| {key[1]} | {key[0]} | {best['route']} | {best['ratio_vs_int8']:.3f} |")


if __name__ == "__main__":
    main(sys.argv[1])

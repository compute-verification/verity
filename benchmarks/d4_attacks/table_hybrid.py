"""Render hybrid.jsonl (hybrid.cu's JSON lines) as the per-alphabet table: each route's median time, the ratio to the
fastest baseline (int8 or FP8), the gates, the control, the FP4 core / correction split, and the clocks seen while timing.

    python benchmarks/d4_attacks/table_hybrid.py OUT/hybrid.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

BASELINES = ("int8", "fp8_e4m3")


def main(path: str) -> None:
    gates, summ, clocks, controls, splits, starts = {}, defaultdict(dict), defaultdict(list), [], {}, {}
    cur = None
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        e = json.loads(line)
        ev = e["event"]
        if ev == "start":
            cur = (e["n"], e["alphabet"])
            starts[cur] = e
        elif ev == "gate":
            gates[(cur, e["route"])] = e
        elif ev == "control":
            controls.append((cur, e))
        elif ev == "split":
            splits[(e["n"], e["alphabet"])] = e
        elif ev == "rep":
            clocks[cur].extend([e["clk0"], e["clk1"]])
        elif ev == "summary":
            summ[(e["n"], e["alphabet"])][e["route"]] = e

    print("| n | alphabet | δ | route | gate (mism/checked) | median ms | T/int8 | T/best-baseline |")
    print("|---|---|---|---|---|---|---|---|")
    for key in sorted(summ):
        st = starts.get(key, {})
        base = min((summ[key][b]["median_ms"] for b in BASELINES if b in summ[key]), default=None)
        for route, e in sorted(summ[key].items(), key=lambda kv: kv[1]["median_ms"]):
            g = gates.get((key, route), {})
            gate = f"{g.get('mismatches', '-')}/{g.get('checked', '-')}"
            best = f"{e['median_ms'] / base:.3f}" if base else "-"
            print(f"| {key[0]} | {key[1]} | {st.get('delta', 0):.4f} | {route} | {gate} | {e['median_ms']:.4f} | "
                  f"{e['ratio_vs_int8']:.3f} | {best} |")

    print("\n## FP4 core / correction split (standalone timings)")
    print("| n | alphabet | core ms | dense-corr ms | sparse-corr ms | dense method ms | core/dense-method |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(splits):
        s = splits[key]
        method = summ.get(key, {}).get("method_densefp4", {})
        mms = method.get("median_ms")
        frac = f"{s['core_ms'] / mms:.3f}" if mms else "-"
        mms_s = f"{mms:.4f}" if mms else "-"
        print(f"| {key[0]} | {key[1]} | {s['core_ms']:.4f} | {s['dense_corr_ms']:.4f} | {s['sparse_corr_ms']:.4f} | "
              f"{mms_s} | {frac} |")

    failed = [(k, r, g) for (k, r), g in gates.items() if g["mismatches"]]
    print("\nGates failed (route not timed):",
          ", ".join(f"{k[1]}@{k[0]} {r} {g['mismatches']}" for k, r, g in failed) or "none")
    print("Controls (core-only must be rejected):",
          ", ".join(f"{k[1]}@{k[0]} rejected={c['rejected']} ({c['mismatches']})" for k, c in controls))
    allc = [c for v in clocks.values() for c in v if c]
    if allc:
        print(f"SM clock over timed items: min {min(allc)} max {max(allc)} MHz")

    print("\n## Verdict: fastest exact method vs the best baseline")
    print("| alphabet | n | best method | T_method / T_best_baseline | beats baseline? |")
    print("|---|---|---|---|---|")
    for key in sorted(summ, key=lambda k: (k[1], k[0])):
        methods = {r: e for r, e in summ[key].items() if r.startswith("method")}
        base = min((summ[key][b]["median_ms"] for b in BASELINES if b in summ[key]), default=None)
        if not methods or not base:
            continue
        name, e = min(methods.items(), key=lambda kv: kv[1]["median_ms"])
        ratio = e["median_ms"] / base
        print(f"| {key[1]} | {key[0]} | {name} | {ratio:.3f} | {'YES' if ratio < 1 else 'no'} |")


if __name__ == "__main__":
    main(sys.argv[1])

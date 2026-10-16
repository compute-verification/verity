"""run.sh's summary: the gates, hot items with each arm's ratio to plain cuBLASLt, the edited GEMM's time by part, the
per-call interleaving and ncu's counters. python3 summary.py OUT_DIR"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.startswith("{")] if p.exists() else []


def ncu(p: Path) -> dict[str, str]:
    if not p.exists():
        return {}
    lines = p.read_text().splitlines()
    start = next((i for i, x in enumerate(lines) if x.startswith('"ID"')), None)
    if start is None:
        return {}
    return {r["Metric Name"]: r["Metric Value"] for r in csv.DictReader(io.StringIO("\n".join(lines[start:])))}


def main(o: Path) -> None:
    print(f"# edited GEMM at o_proj ({o})\n")
    print("## gates\n")
    for g in jl(o / "gate.jsonl"):
        print(f"- device {g['m']} x {g['n']} x {g['k']}: device_ok {g['device_ok']}, control_rejected {g['control_rejected']}, {g['device']}")
    for r in jl(o / "ref.jsonl"):
        print(f"- CPU ref {r['m']} x {r['n']} x {r['k']}: ok {r['ok']}, tiles {r['tiles']}")
    b = jl(o / "bench.jsonl")
    if b:
        a = b[0]["arms"]
        pl = a.get("plain", {}).get("ms")
        print("\n## hot items (median of the reps, ms)\n\n| arm | ms | / plain | MHz |\n|---|---|---|---|")
        for name, v in a.items():
            print(f"| {name} | {v['ms']:.4f} | {v['ms'] / pl:.3f} | {v['mhz']:.0f} |" if pl else f"| {name} | {v['ms']:.4f} | | |")
        g = lambda n: a.get(n, {}).get("ms")  # noqa: E731
        if pl and g("e_plain") and g("wedit"):
            print("\n## the edited GEMM by part (hot items, ms)\n")
            print(f"- mainloop + 2-atom tail against cuBLASLt (e_plain - plain): {g('e_plain') - pl:+.4f}")
            for v in [n for n in a if n.startswith("e_l")] + ["t_duo32"]:
                if g(v):
                    print(f"- receipts in {v} ({v} - e_plain): {g(v) - g('e_plain'):+.4f}; wedit + {v} = {g('wedit') + g(v):.4f}"
                          f" = {(g('wedit') + g(v)) / pl:.3f}x plain")
            for v in ("e_plain", "e_l2h"):
                if g(v) and g(v + "_nc"):
                    print(f"- the issuer's tile place once a tile in {v} ({v} - {v}_nc): {g(v) - g(v + '_nc'):+.4f}")
            print(f"- weights + U and P_B (wedit): {g('wedit'):.4f}")
            if g("qdither") and g("qplain"):
                print(f"- the dither and S in the quantizer (qdither - qplain): {g('qdither') - g('qplain'):+.4f}"
                      f" (qplain {g('qplain'):.4f}, qdither {g('qdither'):.4f})")
            if g("qwe") and g("qdither"):
                print(f"- both producers in one launch (qwe): {g('qwe'):.4f}, against qdither + wedit"
                      f" {g('qdither') + g('wedit'):.4f} ({g('qwe') - g('qdither') - g('wedit'):+.4f})")
    il = jl(o / "il.jsonl")
    if il:
        print("\n## per call, interleaved (41 reps, medians)\n\n| arm | base | ms | base ms | ratio | MHz (min-max) |\n|---|---|---|---|---|---|")
        for r in il:
            print(f"| {r['il']} | {r['base']} | {r['ms']:.4f} | {r['base_ms']:.4f} | {r['ratio']:.4f} | {r['mhz']:.0f} ({r['mhz_min']:.0f}-{r['mhz_max']:.0f}) |")
    for p in sorted(o.glob("ncu-*.csv")):
        d = ncu(p)
        if d:
            print(f"\n### {p.stem}\n\n" + "; ".join(f"{k} {v}" for k, v in d.items()))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

"""beside.sh's tables and VERDICT lines, from its output directory.

lowreg: per variant, register cap and warpgroups an SM, the median over rounds of compressions a second and an SM-clock,
against BAR (sx::compress<0, 1>'s full-occupancy rate on the H200, 4.355 G/s at 1830 MHz: 0.01803 an SM-clock), the
registers ptxas gave and the local memory (spills). The verdict: for one warpgroup an SM, the fewest registers at which a
variant reaches 90% of BAR an SM-clock, and the best rate at each cap.

    python3 beside_summary.py OUT_DIR    # markdown tables, then VERDICT lines; summary.json beside them
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BAR_GS, BAR_CLK = 4.355, 0.01803
CAPS = (64, 72, 80, 96, 128, 255)


def jsonl(path: Path) -> list[dict]:
    out, bad = [], []
    if not path.exists():
        return out
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            bad.append((n, str(e)))
    if bad:
        print(f"beside_summary: {path}: {len(bad)} line(s) left out, unparsed (first, line {bad[0][0]}: {bad[0][1]})",
              file=sys.stderr)
    return out


def lowreg(o: Path, summary: dict) -> list[str]:
    rows = [r for r in jsonl(o / "lowreg-rates.jsonl") if "var" in r]
    if not rows:
        return ["VERDICT lowreg: not run (no lowreg-rates.jsonl)"]
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["var"], r["cap"], r["wgs"])].append(r)
    med = {}
    for k, rs in cells.items():
        med[k] = {"gcomp_per_s": statistics.median(r["gcomp_per_s"] for r in rs),
                  "per_clk": statistics.median(r["comp_per_sm_clk"] for r in rs),
                  "mhz": statistics.median(r["mhz"] for r in rs), "regs": rs[0]["regs"], "local": rs[0]["local_bytes"],
                  "rounds": len(rs)}
    summary["lowreg"] = {f"{v}/{c}/{w}": m for (v, c, w), m in med.items()}
    variants = list(dict.fromkeys(k[0] for k in med))
    lines = []
    for w in sorted({k[2] for k in med}):
        lines += [f"\n### {w} warpgroup(s) an SM: G compressions/s (an SM-clock as % of {BAR_CLK}), regs, spill B", "",
                  "| variant | " + " | ".join("none" if c == 255 else str(c) for c in CAPS) + " |",
                  "|---" * (len(CAPS) + 1) + "|"]
        for v in variants:
            cols = []
            for c in CAPS:
                m = med.get((v, c, w))
                cols.append("" if m is None else
                            f"{m['gcomp_per_s']:.3f} ({100 * m['per_clk'] / BAR_CLK:.0f}%) {m['regs']}r"
                            + (f" {m['local']}B" if m["local"] else ""))
            lines.append(f"| {v} | " + " | ".join(cols) + " |")
    lines.append("")
    for c in CAPS:
        best = max(((v, med[(v, c, 1)]) for v in variants if (v, c, 1) in med), key=lambda x: x[1]["per_clk"], default=None)
        if best:
            v, m = best
            lines.append(f"VERDICT lowreg cap {'none' if c == 255 else c}, 1 warpgroup an SM: best {v} {m['gcomp_per_s']:.3f} "
                         f"G/s at {m['mhz']:.0f} MHz, {m['per_clk']:.5f} an SM-clock = {m['per_clk'] / BAR_CLK:.3f}x the bar "
                         f"{BAR_CLK} ({BAR_GS} G/s at 1830 MHz); {m['regs']} registers, {m['local']} B local")
    hit = [(c, v) for c in CAPS for v in variants if (v, c, 1) in med and med[(v, c, 1)]["per_clk"] >= 0.9 * BAR_CLK]
    if hit:
        c, v = min(hit)
        lines.append(f"VERDICT lowreg: the fewest registers at which one warpgroup an SM reaches 90% of the bar: cap "
                     f"{'none' if c == 255 else c} ({v}, {med[(v, c, 1)]['per_clk'] / BAR_CLK:.3f}x)")
    else:
        top = max(((v, c) for v in variants for c in CAPS if (v, c, 1) in med), key=lambda k: med[(k[0], k[1], 1)]["per_clk"])
        lines.append(f"VERDICT lowreg: no cap reaches 90% of the bar with one warpgroup an SM; the best is {top[0]} at cap "
                     f"{top[1]}, {med[(top[0], top[1], 1)]['per_clk'] / BAR_CLK:.3f}x")
    return lines


def main(o: Path) -> int:
    summary: dict = {}
    out = lowreg(o, summary)
    print("\n".join(out))
    (o / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))

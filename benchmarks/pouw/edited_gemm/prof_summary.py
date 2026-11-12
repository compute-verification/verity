"""Summarize profile.sh's run dir: the parts of the slice's edited GEMM at o_proj against plain cuBLASLt int8, and ncu's
counters per kernel. Usage: prof_summary.py OUT_DIR"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys


def ncu_rows(path: str) -> list[dict[str, str]]:
    """An ncu CSV's rows by column name, read from its header line on; none when it has no header."""
    with open(path, errors="replace") as fh:
        lines = fh.read().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID"')), None)
    if start is None:
        return []
    return list(csv.DictReader(lines[start:]))


def main(out: str) -> None:
    """Print the summary of profile.sh's run directory ``out``."""
    with open(os.path.join(out, "bench.jsonl")) as fh:
        bench = [json.loads(line) for line in fh if line.strip()]
    print("# The slice's edited GEMM at o_proj, by part\n")
    for b in bench:
        arms = {k: v["ms"] for k, v in b["arms"].items()}
        mhz = {k: v["mhz"] for k, v in b["arms"].items()}
        p = arms["plain"]
        print(f"shape {b['shape']} m {b['m']} n {b['n']} k {b['k']}, {b['gpu']}, plain cuBLASLt {p:.4f} ms (algo {b['lt_algo'][0]})\n")
        print("| arm | ms | / plain | MHz |\n|---|---|---|---|")
        for k, v in arms.items():
            print(f"| {k} | {v:.4f} | {v / p:.3f} | {mhz[k]:.0f} |")

        def g(k: str) -> float:
            return arms.get(k, float("nan"))

        parts = [
            ("custom mainloop, no receipts, against cuBLASLt (cplain - plain)", g("cplain") - p),
            ("duo receipts at L = 32 in the mainloop (cduo - cplain)", g("cduo") - g("cplain")),
            ("the peel as a tail stage (tK - cplain)", g("tK") - g("cplain")),
            ("A' pass (formA)", g("formA")),
            ("B side: B' = B + F, S's digits, U's hashes in its first blocks (bsp)", g("bsp")),
            ("U's expansion alone (genU)", g("genU")),
            ("F's expansion alone (genF)", g("genF")),
        ]
        tot = g("td_bfus")
        print(f"\n**td_bfus {tot:.4f} ms = {tot / p:.3f} x plain**; parts (each timed alone, hot):\n")
        print("| part | ms | / plain |\n|---|---|---|")
        for name, v in parts:
            print(f"| {name} | {v:.4f} | {v / p:.3f} |")
        s = (g("cduo") - p) + (g("tK") - g("cplain")) + g("formA") + g("bsp")
        print(f"| sum of mainloop + receipts + tail + formA + bsp (over plain) | {s:.4f} | {s / p:.3f} |")
        print(f"| td_bfus - plain | {tot - p:.4f} | {(tot - p) / p:.3f} |")
    il = os.path.join(out, "il.jsonl")
    if os.path.exists(il):
        print("\n## Per-call interleaving (the slice's: the arm, then plain, alternating which goes first)\n")
        print("| arm | arm ms | plain ms | ratio | MHz |\n|---|---|---|---|---|")
        with open(il) as fh:
            for line in fh:
                if line.strip():
                    j = json.loads(line)
                    print(f"| {j['il']} | {j['edited_ms']:.4f} | {j['plain_ms']:.4f} | {j['ratio']:.3f} | {j['mhz']:.0f} |")
    print("\n## ncu (one call, --clock-control none)\n")
    for f in sorted(glob.glob(os.path.join(out, "ncu-*.csv"))):
        arm = os.path.basename(f)[4:-4]
        rows = ncu_rows(f)
        extra = ncu_rows(f.replace("ncu-", "ncu2-")) if not arm.startswith("stall-") else []
        if not rows:
            with open(f, errors="replace") as fh:
                print(f"{arm}: no rows ({fh.read()[-300:]!r})\n")
            continue
        by = {}
        for r in rows + extra:
            by.setdefault((r["ID"], r["Kernel Name"][:60]), {})[r["Metric Name"]] = r["Metric Value"]
        print(f"### {arm}\n")
        for (kid, name), m in by.items():
            if arm.startswith("stall-"):
                top = sorted(((float(v.replace(",", "")), k.split("stalled_")[1].split("_per")[0]) for k, v in m.items()
                              if v not in ("", "n/a")), reverse=True)[:6]
                print(f"- {name}: " + ", ".join(f"{k} {v:.2f}" for v, k in top))
            else:
                print(f"- {name}: " + "; ".join(f"{k} {v}" for k, v in m.items()))
        print()


if __name__ == "__main__":
    main(sys.argv[1])

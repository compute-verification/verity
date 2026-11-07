"""Tables from sweep.py's breakeven_surface.json files: each case's bound at beta, floor and break-even beta at FP8
and FP4, its serve tax by its replay design and its total cost (the tax plus beta) at break-even.

    python benchmarks/spot_figures/breakeven/report.py SURFACE.json... [--scope "world week"] [--cases NAME...]
"""
import argparse
import json
import re
from pathlib import Path

#: the serve tax's parts outside the edge (the certified tick, the deterministic top-k), the measured pause at an edge
#: (rank 0's copy of its 37.26 GB KV pool and each rank's hashing of its own), and the hashing alone at the census's
#: hash rate, a rank hashing its pool in parallel
BASE_TAX = 0.016260162601626105 + 0.0025
PAUSE = 0.98
POOL_BYTES = 37.26e9
HASH_RATE = 3.98e11
HASH_PAUSE = POOL_BYTES / HASH_RATE
WINDOW = 600.0


def tax(case: str) -> tuple[float, str]:
    """The serve tax of a case by its replay design: the snapshot (copy and hash) at each edge; the live K/V hashed at
    each edge without the copy (`edgeL-...-live`); no K/V committed (`edge0-...-live`, a request's lifetime as its
    replay block)."""
    m = re.search(r"edge(\d+)", case)
    if (m and m.group(1) == "0") or case.endswith("+life"):
        return BASE_TAX, "no K/V committed"
    if "live10" in case:
        return BASE_TAX + HASH_PAUSE / 10, "K/V hashed live every 10 s"
    if m and "-live" in case:
        return BASE_TAX + HASH_PAUSE / float(m.group(1)), f"K/V hashed live every {m.group(1)} s"
    edge = float(m.group(1)) if m else WINDOW
    return BASE_TAX + PAUSE / edge, f"snapshot every {edge:g} s"


def load(paths):
    rows = []
    for p in paths:
        rows += json.loads(Path(p).read_text())["rows"]
    return rows


def fmt_beta(b):
    return "never" if b is None else f"{b:.3g}" if b >= 1 else f"{b:.2%}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("surface", nargs="+")
    ap.add_argument("--scope", default=None)
    ap.add_argument("--cases", nargs="*")
    args = ap.parse_args(argv)
    rows = load(args.surface)
    if args.scope:
        rows = [r for r in rows if r["scope"] == args.scope]
    if args.cases:
        order = {c: i for i, c in enumerate(args.cases)}
        rows = sorted((r for r in rows if r["case"] in order), key=lambda r: (order[r["case"]], r["scope"]))
    print("| case | scope | h | bound at β = 1% | floor | β at FP8 | β at FP4 | tax | total at FP8 | total at FP4 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        t, _ = tax(r["case"])
        b8, b4 = r["beta"]["FP8"], r["beta"]["FP4"]
        tot = [("never" if b is None else (f"{t + b:.2%}" if t + b < 1 else f"{t + b:.3g}")) for b in (b8, b4)]
        print(f"| {r['case']} | {r['scope']} | {r['h']:.3g} | {r['bound']['0.01'] / 8e12:,.4g} TB | "
              f"{r['floor'] / 8e9:,.4g} GB | {fmt_beta(b8)} | {fmt_beta(b4)} | {t:.2%} | {tot[0]} | {tot[1]} |")


if __name__ == "__main__":
    main()

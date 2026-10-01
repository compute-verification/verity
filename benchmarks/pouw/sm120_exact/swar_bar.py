#!/usr/bin/env python3
"""`preadd-floor/sm120` against SWAR, with the bias removal priced (the assessor's 12:06 AM PDT ruling): can any composition
admissible in v1's 128-deep promotion group get under honest cost?  CPU only.

A fitting LSB class of width w saves 0.4825·P per honest MAC (4.14 against the floor's 8 W1 per written form element; P the
route's A′ pre-add term) and pays 7.94/w for the removal (one write per output word per class).  With f(w) the measured
share of w-wide leaf row segments that fit a common 9-bit window (swar_window.py's summary, the largest over the families,
an upper bound for every form), the prover nests classes (a fitting w-wide class lies inside fitting w/2-wide ones), so its
saving per honest MAC is at most max over width sets of Σ (f(w) − f(next wider)) · (0.4825·P − 7.94/w)⁺.  A route with
products plus post-adds Q and margin m (Q + P ≥ 1 + m) gets under honest cost only if Q + P − saving(P) < 1 for some
P ≥ 1 + m − Q; the bar is the largest such Q.  Against it: the cheapest products of any composition whose k-factors
multiply to at most 8 (leaf inner dimension ≥ 16), at W1's 2.0 leaf, from the minimum rank ratio per k-factor over a
catalogue of best known ranks (rank is invariant under permuting a format's dimensions).

`--removal flat` prices the removal as the assessor's 12:23 AM PDT correction does: every strategy strips the bias before
each 128-deep group's FP32 promotion, one write per honest output per group, 7.94/128 per honest MAC whatever the class
width, so the saving is 0.4825·P·f − 7.94/128 with f the largest share at any width.  f is the larger of the contiguous
and non-stride readings (a prover may permute k inside an exact group).  Under it the bar grows with f, so the bars at
f = 1 hold for every input.

    python benchmarks/pouw/sm120_exact/swar_bar.py --swar SUMMARY.json --catalogue FMM.txt [--removal flat] [--out DIR]

`--catalogue` is the text of fmm.univ-lille.fr's table (rows `| ⟨X×Y×Z:N⟩ | N | XYZ | ...`).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_result  # noqa: E402

SAVE, LEAF, WRITE, GROUP = 1 - 4.14 / 8, 2.0, 7.94, 128
MARGINS = {"whole unit": 0.013, "measured regions": 0.092}


def saving(P: float, f: dict[int, float], removal: str = "per-class") -> float:
    if removal == "flat":
        return max(0.0, SAVE * P * max(f.values()) - WRITE / GROUP)
    best = 0.0
    for k in range(1, len(f) + 1):
        for S in itertools.combinations(sorted(f, reverse=True), k):
            prev, tot = 0.0, 0.0
            for w in S:
                tot += (f[w] - prev) * max(0.0, SAVE * P - WRITE / w)
                prev = f[w]
            best = max(best, tot)
    return best


def bar(m: float, f: dict[int, float], removal: str = "per-class") -> float:
    for Q in np.arange(1.0, -0.0005, -0.001):
        if min(Q + P - saving(P, f, removal) for P in np.linspace(max(0.0, 1 + m - Q), 3.0, 3001)) < 1:
            return round(float(Q), 3)
    return 0.0


def ranks(path: Path) -> dict[tuple[int, int, int], int]:
    out = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\| ⟨(\d+)×(\d+)×(\d+):(\d+)⟩ \| (\d+) \| (\d+) \|", line)
        if m:
            a, b, c, r = map(int, m.groups()[:4])
            key = tuple(sorted((a, b, c)))
            out[key] = min(r, out.get(key, r))
    return out


def factorizations(n: int, lo: int = 2) -> list[tuple[int, ...]]:
    out = [()]
    for c in range(lo, n + 1):
        out += [(c, *rest) for rest in factorizations(n // c, c)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swar", required=True)
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--max-k-split", type=int, default=8)
    ap.add_argument("--removal", choices=("per-class", "flat"), default="per-class")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    fams = json.loads(Path(args.swar).read_text())["families"]
    f = {int(L): max(r[part][L]["W9"]["1"] for r in fams.values() for part in ("L", "nonstride") if part in r)
         for L in ("16", "32", "64", "128")}
    R = ranks(Path(args.catalogue))
    per_c = {}
    for c in range(2, args.max_k_split + 1):
        best = min((r / (k[0] * k[1] * k[2]), k) for k, r in R.items() if c in k)
        per_c[c] = {"min_ratio": round(best[0], 6), "format": list(best[1]), "rank": R[best[1]]}
    comps = []
    for split in factorizations(args.max_k_split):
        if split and np.prod(split) <= args.max_k_split:
            ratio = float(np.prod([per_c[c]["min_ratio"] for c in split]))
            comps.append({"k_factors": list(split), "products": round(LEAF * ratio, 6)})
    comps.sort(key=lambda r: r["products"])
    bars = {name: bar(m, f, args.removal) for name, m in MARGINS.items()}
    bars_any = {name: bar(m, {L: 1.0 for L in f}, args.removal) for name, m in MARGINS.items()}
    cheapest = comps[0]
    ok = all(cheapest["products"] >= b for b in bars.values())
    ok_any = all(cheapest["products"] >= b for b in bars_any.values())
    summary = {"tool": "pouw_sm120_swar_bar", "removal": args.removal, "f_W9": f, "formats": len(R), "per_k_factor": per_c,
               "compositions": comps, "bars": bars, "bars_any_input": bars_any, "cheapest": cheapest,
               "swar_breaks_nothing": ok, "swar_breaks_nothing_any_input": ok_any}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("removal", "f_W9", "formats", "bars", "bars_any_input", "cheapest",
                                              "swar_breaks_nothing", "swar_breaks_nothing_any_input")}))
    run_result.write(path, True, "sm120-swar-bar",
                     f"removal {args.removal}: cheapest admissible composition {cheapest['k_factors']} costs "
                     f"{cheapest['products']} of honest work in products against SWAR's bars {bars} at the measured shares "
                     f"and {bars_any} at share 1: {'no route breaks' if ok else 'a route may break'} at the measured shares, "
                     f"{'none' if ok_any else 'a route may break'} on any input ({len(R)} catalogued formats)",
                     [("cheapest_products", cheapest["products"], "of honest"),
                      *((f"bar_{n.replace(' ', '_')}", b, "of honest") for n, b in bars.items()),
                      *((f"bar_any_input_{n.replace(' ', '_')}", b, "of honest") for n, b in bars_any.items())])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The extra tensor work a cheater's own MMAs find inside a timed window (CMP-12, W6b's in-window ε term), derived.

  python mma_window.py cutlass-cheat.json ... [--tau T ...] [--json OUT]

Input: `cutlass_receipt`'s JSON from job.sh's cheat stage, where each design kernel runs beside its `_mmX[aA]` variants
(CR_XMMA: the same kernel plus X mma.sync m16n8k32 E4M3 a warp after every 32-deep k-block, on the k-block's own
fragments, into A accumulator tiles of the cheater's own). A prefill k-block is 16 honest MMAs a warp, so the variant
completes X / 16 extra MACs per honest MAC; the script counts them from the kernel's geometry (tiles × MMA warps ×
k-blocks × X × 4,096) and divides by m·n·k.

Per shape, a variant's time ratio is the median over rounds of its time over the design's in the same round (every
round times every kernel at the shape, in a shuffled order). Per phase (p2048, p8192), the four linears are weighted
by the design's time, as one layer runs each once. At each τ, the largest extra-MAC share among variants whose ratio
is within τ + RESOLUTION (the shuffled A/A controls' resolution, windows_eps.RESOLUTION), and the share a cheater mixing
the two bracketing variants reaches at exactly τ (linear between measured points, from the null at ratio 1). Several
runs pool their variants' points: each ratio is against its own run's design.

ε_in's tensor part at τ reads the window as windows_eps.timed does, with the cheater's in-kernel MACs added: the most of
share + BEST_ETA·max(0, τ − s)/η_w over the variants that fit and the null (share 0, s 1), where η_w is the GEMM's η
(windows_eps.ETA_GEMM) times the design's rate kept against plain in this run. The null alone, BEST_ETA(τ − 1)/η_w, is
the reading with no tensor work inside the kernel (windows_eps's CUDA-only rows add κ·added/ρ to it). The table's ε is
the larger of that and the mixed share at τ.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from windows_eps import BEST_ETA, ETA_GEMM  # noqa: E402

RESOLUTION = 0.002
TAUS = (1.00, 1.01, 1.02, 1.05)
VARIANT = re.compile(r"_mm(\d+)(?:a(\d+))?(b)?$")
MMA_MACS = 16 * 8 * 32
KBLOCK = 32


def ratios(rows, design, variant, shape):
    d, v = rows[(design, shape)], rows[(variant, shape)]
    if "ms_reps" in d and "ms_reps" in v:
        return statistics.median(x / y for x, y in zip(v["ms_reps"], d["ms_reps"]))
    return v["wall_vs_plain"] / d["wall_vs_plain"]


def frontier(points, tau):
    """points: (ratio, share) with the null (1, 0). The largest share at ratio ≤ τ + RESOLUTION, and at exactly τ by
    mixing two points (the upper concave hull's value at τ)."""
    fit = max(s for r, s in points if r <= tau + RESOLUTION)
    best = fit
    for r1, s1 in points:
        for r2, s2 in points:
            if r1 <= tau < r2:
                best = max(best, s1 + (s2 - s1) * (tau - r1) / (r2 - r1))
    return fit, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cheat", nargs="+")
    ap.add_argument("--tau", type=float, action="append")
    ap.add_argument("--json")
    a = ap.parse_args()
    taus = a.tau or list(TAUS)
    out = {"source": a.cheat, "resolution": RESOLUTION, "rows": [], "tau": []}
    print("| run | design | variant | X | xacc | phase | shape | extra MAC share | time ratio vs design | design vs plain |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for run, path in enumerate(a.cheat):
        table(json.load(open(path)), run, out)
    print()
    tau_table(out, taus)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


def table(d, run, out):
    kern = {k["name"]: k for k in d["kernels"]}
    rows = {(r["kernel"], r["shape"]): r for r in d["rows"]}
    shapes = sorted({r["shape"] for r in d["rows"]})
    phases = sorted({s.split("_")[0] for s in shapes})
    variants = sorted({r["kernel"] for r in d["rows"] if VARIANT.search(r["kernel"])},
                      key=lambda k: (VARIANT.sub("", k), bool(VARIANT.search(k)[3]), int(VARIANT.search(k)[1]),
                                     int(VARIANT.search(k)[2] or 0)))
    for v in variants:
        base = VARIANT.sub("", v)
        k = kern[v]
        warps = k["threads"] // 32
        for ph in phases + []:
            ss = [s for s in shapes if s.startswith(ph + "_") and (base, s) in rows and (v, s) in rows]
            if not ss:
                continue
            per = {}
            for s in ss:
                r = rows[(v, s)]
                tiles = (r["m"] // k["tile"][0]) * (r["n"] // k["tile"][1])
                extra = tiles * warps * (r["k"] // KBLOCK) * k["xmma"] * MMA_MACS
                per[s] = (ratios(rows, base, v, s), extra / (r["m"] * r["n"] * r["k"]), rows[(base, s)]["ms"],
                          rows[(base, s)]["wall_vs_plain"], extra, r["m"] * r["n"] * r["k"])
            for s in ss:
                q, sh, _, dvp, ex, hon = per[s]
                print(f"| {run} | {base} | {v} | {k['xmma']} | {k['xacc']} | {ph} | {s} | {sh:.4f} | {q:.4f} | {dvp:.4f} |")
                out["rows"].append({"run": run, "design": base, "variant": v, "xmma": k["xmma"], "xacc": k["xacc"], "phase": ph,
                                    "shape": s, "extra_macs": ex, "honest_macs": hon, "extra_share": sh, "ratio": q,
                                    "design_vs_plain": dvp})
            tb = sum(per[s][2] for s in ss)
            q = sum(per[s][2] * per[s][0] for s in ss) / tb
            ex, hon = sum(per[s][4] for s in ss), sum(per[s][5] for s in ss)
            dvp = sum(per[s][2] for s in ss) / sum(rows[("p_plain", s)]["ms"] for s in ss)
            print(f"| {run} | {base} | {v} | {k['xmma']} | {k['xacc']} | {ph} | **phase** | {ex / hon:.4f} | **{q:.4f}** | {dvp:.4f} |")
            out["rows"].append({"run": run, "design": base, "variant": v, "xmma": k["xmma"], "xacc": k["xacc"], "phase": ph,
                                "shape": ph, "extra_macs": ex, "honest_macs": hon, "extra_share": ex / hon, "ratio": q,
                                "design_vs_plain": dvp})


def tau_table(out, taus):
    designs = sorted({r["design"] for r in out["rows"]})
    phases = sorted({r["phase"] for r in out["rows"]})
    print("| design | phase | η_w | " + " | ".join(f"τ = {t:.2f}: fits / mixed / ε_in tensor (null)" for t in taus) + " |")
    print("|---|---|---|" + "---|" * len(taus))
    for base in designs:
        for ph in phases:
            pts = [(1.0, 0.0)] + [(r["ratio"], r["extra_share"]) for r in out["rows"]
                                  if r["design"] == base and r["phase"] == ph and r["shape"] == ph]
            if len(pts) == 1:
                continue
            dvp = next(r["design_vs_plain"] for r in out["rows"] if r["design"] == base and r["shape"] == ph
                       and r["run"] == 0) if any(r["run"] == 0 and r["design"] == base for r in out["rows"]) else next(
                r["design_vs_plain"] for r in out["rows"] if r["design"] == base and r["shape"] == ph)
            eta_w = ETA_GEMM[ph] / dvp
            cells = []
            for t in taus:
                fit, mixed = frontier(pts, t)
                null = BEST_ETA * max(0.0, t - 1) / eta_w
                eps = max(sh + BEST_ETA * max(0.0, t - q) / eta_w for q, sh in pts if q <= t + RESOLUTION)
                cells.append(f"{fit:.4f} / {mixed:.4f} / {max(eps, mixed):.4f} ({null:.4f})")
                out["tau"].append({"design": base, "phase": ph, "tau": t, "eta_w": eta_w, "fits": fit, "mixed": mixed,
                                   "eps_tensor": max(eps, mixed), "eps_fits": eps, "eps_null": null})
            print(f"| {base} | {ph} | {eta_w:.3f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

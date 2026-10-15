#!/usr/bin/env python3
"""The τ an honest prover needs, from `cutlass_receipt --drift` (job.sh's drift stage; CMP-8 round 3), derived.

  python drift_tau.py cutlass-drift.json drift.csv.gz [--slices 24] [--json OUT]

A window closes at τ × the verifier's reference time. A cheater runs at the best clock the hardware gives, and the
timed-window bound (`windows_eps.timed`) reads τ against the design's time at that clock. So the τ an honest prover
needs, for a window of one GEMM launch to pass with probability q, is its launch time's q-quantile over its fastest:
τ_q = t_q / t_0.001 (the 0.1% quantile, a robust minimum). Reading it against the median instead (t_q / t_med) is the
τ a verifier that calibrates on the median would set; a cheater at the best clock then gains t_med / t_0.001 on top.

The CSV's blocks (16 launches back to back, NVML read after each) also give the slow drift: the run cut into --slices
equal spans of time, each span's median block time, and their spread, against the clock, temperature and power NVML
read (read only; nothing set them).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict

REASONS = {0x1: "gpu idle", 0x2: "applications clocks", 0x4: "sw power cap", 0x8: "hw slowdown", 0x10: "sync boost",
           0x20: "sw thermal", 0x40: "hw thermal", 0x80: "hw power brake", 0x100: "display clocks"}


def quantile(xs, q):
    """The q-quantile of xs, as `cutlass_receipt`'s `quant` takes it (the floor of q·(n − 1) in sorted order)."""
    s = sorted(xs)
    return s[math.floor(q * (len(s) - 1))]


def tau(t_q, t_fast):
    """The window over the fastest honest time that passes the honest launches at the quantile t_q."""
    return t_q / t_fast


def reasons(mask):
    return [n for b, n in REASONS.items() if mask & b] or ["none"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drift")
    ap.add_argument("csv")
    ap.add_argument("--slices", type=int, default=24)
    ap.add_argument("--json")
    a = ap.parse_args()
    d = json.load(open(a.drift))
    rows = list(csv.DictReader(gzip.open(a.csv, "rt") if a.csv.endswith(".gz") else open(a.csv)))
    span = max(float(r["t_s"]) for r in rows)
    out = {"seconds": d["drift"]["seconds"], "rounds": d["drift"]["rounds"], "block": d["drift"]["block"], "kernels": []}
    print(f"{d['drift']['rounds']} rounds over {d['drift']['seconds']:.0f} s, {d['drift']['block']} launches a block\n")
    print("| kernel | shape | launches | median ms | t_0.001 | t_0.999 | max | τ_99.9 (over t_0.001) | τ_99 | "
          "τ_99.9 over the median | 16-launch window τ_99.9 | slice medians' spread | SM MHz min / median / max |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in rows:
        by[(r["kernel"], r["shape"])].append(r)
    for k in d["rows"]:
        bl = by[(k["kernel"], k["shape"])]
        sl = defaultdict(list)
        for r in bl:
            sl[min(a.slices - 1, int(float(r["t_s"]) / span * a.slices))].append(float(r["ms_med"]))
        meds = [quantile(v, 0.5) for _, v in sorted(sl.items())]
        spread = max(meds) / min(meds)
        bm = [float(r["ms_med"]) for r in bl]
        e = {**k, "tau_999": tau(k["ms_q0999"], k["ms_q0001"]), "tau_99": tau(k["ms_q099"], k["ms_q0001"]),
             "tau_999_over_median": k["ms_q0999"] / k["ms_med"],
             "tau_999_block": tau(quantile(bm, 0.999), quantile(bm, 0.001)),
             "slice_medians": meds, "slice_spread": spread,
             "temp_c": k["temp_c"], "reasons": reasons(k["reasons_any"])}
        out["kernels"].append(e)
        print(f"| {k['kernel']} | {k['shape']} | {k['launches']} | {k['ms_med']:.4f} | {k['ms_q0001']:.4f} | "
              f"{k['ms_q0999']:.4f} | {k['ms_max']:.4f} | {e['tau_999']:.4f} | {e['tau_99']:.4f} | "
              f"{e['tau_999_over_median']:.4f} | {e['tau_999_block']:.4f} | {spread:.4f} | "
              f"{k['mhz_min']:.0f} / {k['mhz_med']:.0f} / {k['mhz_max']:.0f} |")
    temps = [t for e in out["kernels"] for t in e["temp_c"]]
    mask = 0
    for e in out["kernels"]:
        mask |= e["reasons_any"]
    out["temp_c"] = [min(temps), max(temps)]
    out["power_w_max"] = max(e["power_w_max"] for e in out["kernels"])
    out["reasons"] = reasons(mask)
    print(f"\ntemperature {min(temps)}–{max(temps)} C, power ≤ {out['power_w_max']:.0f} W, clock-event reasons seen: "
          f"{', '.join(out['reasons'])}")
    for ph in sorted({e["shape"].split("_")[0] for e in out["kernels"]}):
        for kn in sorted({e["kernel"] for e in out["kernels"]}):
            es = [e for e in out["kernels"] if e["kernel"] == kn and e["shape"].startswith(ph + "_")]
            if es:
                print(f"{kn} {ph}: τ_99.9 worst linear {max(e['tau_999'] for e in es):.4f}, "
                      f"time-weighted {sum(e['tau_999'] * e['ms_med'] for e in es) / sum(e['ms_med'] for e in es):.4f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

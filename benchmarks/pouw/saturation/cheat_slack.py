#!/usr/bin/env python3
"""The CUDA slack a cheater's fused kernel finds beside each served receipt design (CMP-8 round 2, section 2), derived.

  python cheat_slack.py cutlass-cheat.json [--drop X ...] [--json OUT]

--drop X skips the X-step variants: runs before the cheater's store test read z[0] have x1 kernels whose chain nvcc
removed (their SASS equals the design's), so their X = 1 rows measure nothing.

Input: `cutlass_receipt`'s JSON from job.sh's cheat stage, where each design kernel runs beside its `_xX` variants: the
same kernel plus X alu phi1 steps a thread after every 32-deep k-block on chains that never read the accumulators. Per
phase (the four linears weighted by the design's time, as one layer runs each once):

- added: the cheater's steps per SM-clock over φ's solo rate (21.2 alu steps an SM-clock, round 1), at the variant's time;
- slowdown: the variant's time over the design's, from each kernel's median against the plain kernel in the same rounds;
- free share: (added work at φ's solo rate − the time it added) / added work, both in units of the design's time;
- ν_f: the most free capacity any variant shows, added work − the time it added, as a share of the window's CUDA
  capacity at the design's time (the lemma's ν: what the cheater gains for free);
- ν_f at no rate loss: the most added share among variants within NOISE of the design's time.
"""
from __future__ import annotations

import argparse
import json
import re

PHI_SOLO = 21.2
NOISE = 0.005
KB = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cheat")
    ap.add_argument("--drop", type=int, action="append", default=[])
    ap.add_argument("--json")
    a = ap.parse_args()
    d = json.load(open(a.cheat))
    sms = d["sms"]
    threads = {k["name"]: k["threads"] for k in d["kernels"]}
    rows = {(r["kernel"], r["shape"]): r for r in d["rows"]}
    phases = sorted({r["shape"].split("_")[0] for r in d["rows"]})
    out = []
    print("| design | X | phase | added (share of φ solo) | slowdown | free share | free capacity | rate kept vs plain |")
    print("|---|---|---|---|---|---|---|---|")
    variants = sorted({r["kernel"] for r in d["rows"] if re.search(r"_x\d+$", r["kernel"])},
                      key=lambda k: (k.rsplit("_x", 1)[0], int(k.rsplit("_x", 1)[1])))
    for v in variants:
        base, x = v.rsplit("_x", 1)
        x = int(x)
        if x in a.drop:
            continue
        for ph in phases:
            shapes = [s for (k, s) in rows if k == base and s.startswith(ph + "_")]
            if not shapes:
                continue
            tb = sum(rows[(base, s)]["ms"] for s in shapes)
            tv = sum(rows[(v, s)]["ms"] for s in shapes)
            tp = sum(rows[("p_plain", s)]["ms"] for s in shapes)
            slow = sum(rows[(base, s)]["ms"] * rows[(v, s)]["wall_vs_plain"] / rows[(base, s)]["wall_vs_plain"]
                       for s in shapes) / tb
            steps = sum(rows[(v, s)]["m"] // 128 * (rows[(v, s)]["n"] // 128) * threads[v] * rows[(v, s)]["k"] // 128
                        * KB * x for s in shapes)
            clocks = sum(rows[(v, s)]["ms"] * 1e-3 * rows[(v, s)]["sm_mhz"] * 1e6 for s in shapes) * sms
            added = steps / clocks / PHI_SOLO
            added_base_units = added * slow
            free = (added_base_units - (slow - 1)) / added_base_units
            kept = tp / tv
            net = added_base_units - (slow - 1)
            out.append({"design": base, "extra": x, "phase": ph, "added": added, "slowdown": slow, "free": free,
                        "net": net, "rate_kept_vs_plain": kept})
            print(f"| {base} | {x} | {ph} | {100 * added:.1f}% | {slow:.4f}× | {100 * free:.0f}% | {100 * net:.1f}% | "
                  f"{100 * kept:.1f}% |")
    nu, net = {}, {}
    for o in out:
        k = (o["design"], o["phase"])
        net[k] = max(net.get(k, 0.0), o["net"])
        if o["slowdown"] <= 1 + NOISE:
            k = (o["design"], o["phase"])
            nu[k] = max(nu.get(k, 0.0), o["added"])
    print("\nν_f (most free capacity):", json.dumps({f"{k}@{p}": round(v, 4) for (k, p), v in sorted(net.items())}))
    print("ν_f at no rate loss (within", f"{100 * NOISE:.1f}%):",
          json.dumps({f"{k}@{p}": round(v, 4) for (k, p), v in sorted(nu.items())}))
    if a.json:
        json.dump({"rows": out, "nu_f": {f"{k}@{p}": v for (k, p), v in net.items()},
                   "nu_f_no_rate_loss": {f"{k}@{p}": v for (k, p), v in nu.items()}}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

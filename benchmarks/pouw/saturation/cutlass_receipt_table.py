#!/usr/bin/env python3
"""Tables from `cutlass_receipt`'s JSON (job.sh's cgemm stage), with `vllm_gemm_ref.py`'s beside it.

  python cutlass_receipt_table.py cutlass-receipt.json [--ref vllm-gemm-ref.json] [--json OUT]

Rate kept is the receipt-free kernel's time over the receipted kernel's, per SM clock (``cycles_vs_plain``, paired round by
round). A phase's rate kept weights its four linears by the receipt-free kernel's time, as one layer runs each once. φ share
is φ steps per SM clock over φ alone's rate (`phi_probe.cu`: 21.2 alu, 26.6 imad, `r20261015-052731-4476`). The slot
model's load is T_φ/T_mma = h_φ · rounds / L × (receipt-free MACs/SM/clk) / 128, with h_φ 6.0 (alu) and 4.8 (imad) slots.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

PHI_ALONE = {0: 21.2, 1: 26.6}
H_PHI = {0: 6.0, 1: 4.8}
VAR = {0: "alu", 1: "imad"}


def load(path):
    text = open(path).read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # A run cut short leaves the rows array open; keep the rows printed so far.
        rows = [json.loads(m) for m in re.findall(r"\{\"shape\"[^{}]*\}", text)]
        return {"rows": rows, "truncated": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt")
    ap.add_argument("--ref")
    ap.add_argument("--json")
    a = ap.parse_args()
    d = load(a.receipt)
    rows = [r for r in d.get("rows", []) if r]
    plain = {r["shape"]: r for r in rows if r["kernel"].endswith("_plain")}
    kernels = {k["name"]: k for k in d.get("kernels", [])}
    ref = {}
    if a.ref:
        ref = {r["shape"]: r for r in json.load(open(a.ref))["rows"]}

    print("| shape | m×n×k | plain TFLOPS | plain MACs/SM/clk | vLLM TFLOPS | plain / vLLM time |")
    print("|---|---|---|---|---|---|")
    for s, p in plain.items():
        v = ref.get(s)
        vt, ratio = (v["tflops"], f"{p['ms'] / v['ms']:.3f}") if v else ("—", "—")
        print(f"| {s} | {p['m']}×{p['n']}×{p['k']} | {p['tflops']} | {p['macs_per_sm_clk']} | {vt} | {ratio} |")

    out = []
    print("\n| kernel | φ | L | shape | slot load | rate kept | φ share |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["kernel"].endswith("_plain") or r["shape"] not in plain:
            continue
        p = plain[r["shape"]]
        load_ = H_PHI[r["var"]] * r["rounds"] / r["L"] * p["macs_per_sm_clk"] / 128
        kept = 1 / r["cycles_vs_plain"]
        share = r["phi1_per_sm_clk"] / PHI_ALONE[r["var"]]
        kn = kernels.get(r["kernel"], {})
        if kn.get("forced"):
            # CR_FORCED's steps, F a thread after every 32-deep k-block, on top of the receipt's φ: alu φ1 at φ's solo
            # rate, or FFMAs (fvar 2) at the FMA pipe's 128 an SM-clock, both as shares of the SM's CUDA slots
            steps = r["m"] // 128 * (r["n"] // 128) * kn["threads"] * r["k"] // 32 * kn["forced"]
            share += steps / (r["ms"] * 1e3 * r["sm_mhz"] * d["sms"]) / (128.0 if kn.get("fvar") == 2 else PHI_ALONE[0])
        out.append({**r, "slot_load": load_, "rate_kept": kept, "phi_share": share, "plain_ms": p["ms"],
                    "forced": kn.get("forced", 0), "fvar": kn.get("fvar", 0), "honest": kn.get("honest", 0),
                    "threads": kn.get("threads", 256)})
        print(f"| {r['kernel']} | {r['rounds']}× {VAR[r['var']]} | {r['L']} | {r['shape']} | {load_:.2f} | "
              f"{100 * kept:.1f}% | {100 * share:.0f}% |")

    phase = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for o in out:
        ph = o["shape"].split("_")[0]
        if ph == "cube8192":
            continue
        acc = phase[(o["kernel"], ph)]
        acc[0] += o["plain_ms"]
        acc[1] += o["plain_ms"] * o["cycles_vs_plain"]
        acc[2] += o["slot_load"] * o["plain_ms"]
        acc[3] += o["phi_share"] * o["plain_ms"]
    print("\n| kernel | phase | slot load | rate kept (4 linears) | φ share |")
    print("|---|---|---|---|---|")
    summary = []
    for (k, ph), (t0, t1, ld, sh) in sorted(phase.items(), key=lambda x: (x[0][1], x[0][0])):
        summary.append({"kernel": k, "phase": ph, "slot_load": ld / t0, "rate_kept": t0 / t1, "phi_share": sh / t0})
        print(f"| {k} | {ph} | {ld / t0:.2f} | {100 * t0 / t1:.1f}% | {100 * sh / t0:.0f}% |")
    if a.json:
        json.dump({"rows": out, "phases": summary, "truncated": d.get("truncated", False)}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

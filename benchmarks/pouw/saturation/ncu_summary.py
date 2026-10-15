#!/usr/bin/env python3
"""One decode step's kernels under ncu (`vllm_split.py --ncu-step`, ``ncu --page raw --csv``), summarized as the CUDA cores'
busy share per kernel category (CMP-8, Daniel's #7).

  python ncu_summary.py RAW.csv --out SUMMARY.json [--sms 188]

A kernel's CUDA-core budget is 128 lane slots per SM per elapsed cycle (one FP32 FADD a lane, the W1 price of 8 dense E4M3
MACs), over every SM, so ``slot_share`` = thread instructions executed / (128 · SMs · elapsed SM cycles): the share of the
model's word-gate slots the kernel used. Beside it, ncu's own utilizations (percent of peak over elapsed cycles) of the FMA,
ALU and XU pipes, of issue slots, and of DRAM bandwidth. Categories are `vllm_split.category`'s; shares are time-weighted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vllm_split import category  # noqa: E402

LANES = 128
PCTS = {"fma_pct": "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
        "alu_pct": "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed",
        "xu_pct": "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed",
        "fma_cycles_pct": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "alu_cycles_pct": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "issue_pct": "sm__inst_issued.avg.pct_of_peak_sustained_elapsed",
        "tensor_pct": "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed",
        "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed"}


def num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        lines = [ln for ln in f if not ln.startswith("==")]
    r = list(csv.DictReader(lines))
    return [x for x in r if x.get("ID", "").strip().isdigit()]


def summarize(path: Path, sms: int) -> dict:
    kernels = []
    for r in rows(path):
        dur = num(r.get("gpu__time_duration.sum", ""))
        cyc = num(r.get("sm__cycles_elapsed.avg", ""))
        lanes = num(r.get("smsp__thread_inst_executed.sum", ""))
        if not dur or not cyc:
            continue
        k = {"kernel": r["Kernel Name"], "category": category(r["Kernel Name"]), "us": dur / 1e3,
             "slot_share": lanes / (LANES * sms * cyc) if lanes is not None else None,
             "lane_ops": lanes, "sm_cycles": cyc * sms}
        for key, metric in PCTS.items():
            k[key] = num(r.get(metric, ""))
        kernels.append(k)
    agg: dict = defaultdict(lambda: defaultdict(float))
    for k in kernels:
        for group in (k["category"], "non_gemm" if k["category"] != "gemm" else "gemm_all", "step"):
            a = agg[group]
            a["us"] += k["us"]
            a["n"] += 1
            a["lane_ops"] += k["lane_ops"] or 0.0
            a["sm_cycles"] += k["sm_cycles"]
            for key in PCTS:
                if k[key] is not None:
                    a[f"{key}_x_us"] += k[key] * k["us"]
                    a[f"{key}_us"] += k["us"]
    out = {}
    for group, a in sorted(agg.items()):
        o = {"kernels": int(a["n"]), "us": round(a["us"], 2), "lane_ops": a["lane_ops"],
             "slot_share": a["lane_ops"] / (LANES * a["sm_cycles"]) if a["sm_cycles"] else None}
        for key in PCTS:
            if a[f"{key}_us"]:
                o[key] = round(a[f"{key}_x_us"] / a[f"{key}_us"], 3)
        out[group] = o
    top = sorted(kernels, key=lambda k: -k["us"])[:30]
    return {"sms": sms, "lanes_per_sm_clk": LANES, "kernels": len(kernels), "by_group": out,
            "top": [{**{x: (round(v, 4) if isinstance(v, float) else v) for x, v in k.items()}, "kernel": k["kernel"][:140]}
                    for k in top]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", type=Path)
    ap.add_argument("--sms", type=int, default=188)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    doc = summarize(args.raw, args.sms)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({g: {k: v for k, v in o.items() if k in ("us", "slot_share", "fma_pct", "alu_pct", "issue_pct")}
                      for g, o in doc["by_group"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

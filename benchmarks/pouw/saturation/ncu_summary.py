#!/usr/bin/env python3
"""One decode step's kernels under ncu (`vllm_split.py --ncu-step`, ``ncu --page raw --csv``), summarized as the CUDA cores'
busy share per kernel category (CMP-8, Daniel's #7).

  python ncu_summary.py RAW.csv --out SUMMARY.json [--sms 188]

A kernel's CUDA-core budget is 128 lane slots per SM per elapsed cycle (one FP32 FADD a lane, the W1 price of 8 dense E4M3
MACs), over every SM, so ``slot_share`` = thread instructions executed / (128 · SMs · elapsed SM cycles): the share of the
model's word-gate slots the kernel used. Beside it, ncu's own utilizations (percent of peak over elapsed cycles) of the FMA,
ALU and XU pipes, of issue slots, and of DRAM bandwidth. Categories are `vllm_split.category`'s; shares are time-weighted.

sm_120's integer ALU runs at half rate (64 lanes/SM/clk, `phi_probe.cu`), so an ALU instruction costs 2 slots of the same
128. ncu's ALU peak is 128 lanes/SM/clk, twice the real rate (φ alone, ALU-bound, reads 49%), so ``alu_pct`` is half the
pipe's real utilization. ``slot_share_alu2`` counts each ALU thread instruction twice, inferring their number from
``alu_pct`` against ncu's peak (full warps assumed, so it stays an upper bound).
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
NCU_ALU_PEAK_LANES = 128
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


SCALE = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9}
TIME_US = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3, "s": 1e6, "second": 1e6}


def rows(path: Path) -> tuple[list[dict], dict]:
    """The kernel rows of ``ncu --csv --page raw`` and its units row (ncu scales each column's unit to its values)."""
    with path.open(newline="") as f:
        lines = [ln for ln in f if not ln.startswith("==")]
    r = list(csv.DictReader(lines))
    units = next((x for x in r if not x.get("ID", "").strip()), {})
    return [x for x in r if x.get("ID", "").strip().isdigit()], units


def scaled(r: dict, units: dict, metric: str, base: str) -> float | None:
    v, u = num(r.get(metric, "")), (units.get(metric) or "").strip()
    if v is None:
        return None
    if base == "us":
        if u not in TIME_US:
            raise SystemExit(f"{metric}: unit {u!r} is not a time unit")
        return v * TIME_US[u]
    prefix = u[: -len(base)] if u.endswith(base) else None
    if prefix not in SCALE:
        raise SystemExit(f"{metric}: unit {u!r} is not a multiple of {base!r}")
    return v * SCALE[prefix]


def summarize(path: Path, sms: int) -> dict:
    kernels = []
    table, units = rows(path)
    for r in table:
        dur = scaled(r, units, "gpu__time_duration.sum", "us")
        cyc = scaled(r, units, "sm__cycles_elapsed.avg", "cycle")
        lanes = scaled(r, units, "smsp__thread_inst_executed.sum", "inst")
        if not dur or not cyc:
            continue
        k = {"kernel": r["Kernel Name"], "category": category(r["Kernel Name"]), "us": dur,
             "slot_share": lanes / (LANES * sms * cyc) if lanes is not None else None,
             "lane_ops": lanes, "sm_cycles": cyc * sms}
        for key, metric in PCTS.items():
            k[key] = num(r.get(metric, ""))
        k["alu_lane_ops"] = (k["alu_pct"] or 0.0) / 100 * NCU_ALU_PEAK_LANES * cyc * sms
        k["slot_share_alu2"] = ((lanes + k["alu_lane_ops"]) / (LANES * sms * cyc)) if lanes is not None else None
        kernels.append(k)
    agg: dict = defaultdict(lambda: defaultdict(float))
    for k in kernels:
        for group in (k["category"], "non_gemm" if k["category"] != "gemm" else "gemm_all", "step"):
            a = agg[group]
            a["us"] += k["us"]
            a["n"] += 1
            a["lane_ops"] += k["lane_ops"] or 0.0
            a["alu_lane_ops"] += k["alu_lane_ops"]
            a["sm_cycles"] += k["sm_cycles"]
            for key in PCTS:
                if k[key] is not None:
                    a[f"{key}_x_us"] += k[key] * k["us"]
                    a[f"{key}_us"] += k["us"]
    out = {}
    for group, a in sorted(agg.items()):
        o = {"kernels": int(a["n"]), "us": round(a["us"], 2), "lane_ops": a["lane_ops"],
             "slot_share": a["lane_ops"] / (LANES * a["sm_cycles"]) if a["sm_cycles"] else None,
             "slot_share_alu2": (a["lane_ops"] + a["alu_lane_ops"]) / (LANES * a["sm_cycles"]) if a["sm_cycles"] else None}
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
    print(json.dumps({g: {k: v for k, v in o.items()
                          if k in ("us", "slot_share", "slot_share_alu2", "fma_pct", "alu_pct", "issue_pct")}
                      for g, o in doc["by_group"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""h, the MAC-times a committed byte costs, from one hash_cost.sh run's out/ (hash_cost.json, smi.csv): stdlib only, no GPU.

    python3 benchmarks/served_commit/hash_cost_h.py RUN_OUT [OUT_JSON]

A MAC-time is one int8 MAC at the P1a ruler, 1 / (4,096 x SMs x f) seconds at the SM clock f nvidia-smi sampled over that point's window,
so h is the SM-cycles a byte holds the whole card, times 4,096:
  alone:  h = ruler x (hash time) / bytes                       the leaves alone (leaf), with the best tree form added (commit)
  beside: h = ruler x (the GEMM loop's extra time) / bytes      the leaves on a low-priority stream beside the served int8 GEMM
          h_noise: the same for the median |A' - A| of the loop alone before and after (A-B-A runs only)
Beside, the hash also needs room: at a point hashing r GB/s while the GEMM runs, a forward of t_G ms carries r x t_G bytes at that h.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

MACS_SM_CLK = 4096


def clocks(smi: Path) -> list[tuple[float, float]]:
    out = []
    with smi.open() as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[1].strip().split(" ")[0].isdigit():
                continue
            t = dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.UTC).timestamp()
            out.append((t, float(row[1].strip().split(" ")[0])))
    return out


def window_mhz(cl: list[tuple[float, float]], t0: float, t1: float) -> float | None:
    v = [mhz for t, mhz in cl if t0 <= t <= t1 and mhz > 600]
    return statistics.median(v) if v else None


def analyze(run: Path) -> dict:
    res = json.loads((run / "hash_cost.json").read_text())
    smi = next((p for p in (run / "smi.csv", run.parent / "smi.csv") if p.exists()), None)   # an ARMS run samples once, in its parent
    cl = clocks(smi) if smi else []
    # nvidia-smi's timestamps are the node's local time; the marks are UTC epoch seconds: align on the first sample's offset
    marks = res["clock_marks"]
    span = {}
    for (tag, t0), (_, t1) in zip(marks, marks[1:]):
        span[tag] = (t0, t1)
    sms = res["sms"]
    gemm = res["gemm"]
    all_mhz = [m for _, m in cl if m > 600]
    f_default = statistics.median(all_mhz) if all_mhz else 1830.0
    offset = 0.0
    if cl and marks:
        offset = min((abs((cl[0][0] - marks[0][1]) - h * 3600), h * 3600) for h in range(-12, 13))[1]

    def mhz(tag: str) -> float:
        if tag not in span:
            return f_default
        t0, t1 = span[tag]
        return window_mhz(cl, t0 + offset, t1 + offset) or f_default

    f_g = mhz("gemm_alone")
    # gemm_alone is a ~50 ms window, mostly between two 200 ms samples: then f_g is the median of every loaded sample
    g_src = "window" if "gemm_alone" in span and window_mhz(cl, *(t + offset for t in span["gemm_alone"])) else "median of loaded samples"
    out = {"run_out": str(run), "device": res["device"], "sms": sms, "T": res["T"], "clock_offset_s": offset, "gemm": {
        **gemm, "mhz": f_g, "mhz_source": g_src,
        "theta_at_clock": round(gemm["tmacs_per_s"] * 1e12 / (MACS_SM_CLK * sms * f_g * 1e6), 4)}, "rows": {}}
    for form, fd in res["forms"].items():
        for rb, e in fd.get("sizes", {}).items():
            if "leaf_ms" not in e:
                continue
            f_a = mhz(f"{form}:{rb}:alone")
            ruler = MACS_SM_CLK * sms * f_a * 1e6
            row = {"form": form, "row_bytes": int(rb), "mhz_alone": f_a, "leaf_gbps": e["leaf_gbps"],
                   "h_leaf": round(ruler / (e["leaf_gbps"] * 1e9), 1)}
            if "commit_gbps" in e:
                row.update(commit_gbps=e["commit_gbps"], h_commit=round(ruler / (e["commit_gbps"] * 1e9), 1),
                           h_tree=round(ruler * min(x for x in (e["tree"]["level_form_ms"], e["tree"]["levels10_form_ms"]) if x) * 1e-3
                                        / res["total_bytes"], 1), tree=e["tree"])
            pts = []
            for b in e.get("beside", []):
                f_b = mhz(f"{form}:{rb}:cap{b['cap']}:beside")
                ruler_b = MACS_SM_CLK * sms * f_b * 1e6
                pts.append({"cap": b["cap"], "warps_per_sm": round(b["cap"] * int(form[1:]) / 32 / sms, 2) if b["cap"] else None,
                            "mhz": f_b, "alone_gbps": b["alone_gbps"], "beside_gbps": b["hash_gbps"], "kappa": b["kappa_window"],
                            "gemm_slowdown": b["gemm_slowdown_total"], "delta_ms": b["gemm_delta_ms"], "overlapped": b["overlapped"],
                            "h_beside": round(ruler_b * b["gemm_delta_ms"] * 1e-3 / b["bytes"], 1),
                            "bytes_per_fwd": round(b["hash_gbps"] * 1e9 * b["loop_alone_ms"] / b["n_fwd"] * 1e-3)})
                if "aa_ms" in b:   # A-B-A runs: the median |A' - A| and B's delta range over reps, in h
                    pts[-1].update(h_noise=round(ruler_b * b["aa_ms"] * 1e-3 / b["bytes"], 1),
                                   h_range=[round(ruler_b * d * 1e-3 / b["bytes"], 1) for d in b["delta_ms_range"]])
            row["beside"] = pts
            out["rows"][f"{form}:{rb}"] = row
    out.update(phases=res.get("phases", "4"), form_kind=res.get("form", "split"), stride0=res.get("stride0", False),
               gemm_priority=res.get("gemm_priority", -100))
    return out


def main() -> None:
    text = json.dumps(analyze(Path(sys.argv[1])), indent=1)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

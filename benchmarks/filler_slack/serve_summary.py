"""serve.sh's arms in one table: `python3 serve_summary.py OUT > summary.json`.

Per arm: the filler alone before the engine starts and beside the idle engine (its share of the tensor peak, rate inside
its blocks, power, SM clock), and per decode point the step time against the stock arms' mean (compare_fill.py's
baseline: every `stock` arm, `warmup` left out), the filler's share of the tensor peak over the point (at the window's
NVML clock and at 2,430 MHz), its rate inside its blocks relative to its rate alone, and the board's power and clock.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def main() -> int:
    o = Path(sys.argv[1])
    arms = [json.loads(p.read_text()) for p in sorted(o.glob("arm-*.json"), key=lambda p: int(p.name.split("-")[1]))]
    stock: dict[tuple, list[float]] = {}
    for a in arms:
        if a["arm"] == "stock":
            for p in a["points"]:
                if "step_ms" in p:
                    stock.setdefault((p["batch"], p["context"]), []).append(p["step_ms"])
    cmp = json.loads((o / "compare.json").read_text()) if (o / "compare.json").exists() else {}
    out = {"gates_pass": cmp.get("gates_pass"), "gates": cmp.get("arms"), "arms": []}
    for a in arms:
        f = a.get("filler", {})
        alone = f.get("alone_before", {})
        row = {"arm": a["arm"], "complete": a.get("complete", False), "check": f.get("check"),
               "streams_ok": a.get("streams_seen") == [a.get("step_stream")]}
        for k in ("alone_before", "beside_idle_engine"):
            if k in f:
                row[k] = {x: f[k].get(x) for x in ("share_of_peak", "share_of_peak_at_max_clock", "mma_per_sm_clk_in_block",
                                                   "power_w", "power_w_max", "sm_mhz", "sm_mhz_min", "most_blocks_on_one_sm")}
        if "idle_engine" in a:
            row["idle_engine"] = a["idle_engine"]
        pts = []
        for p in a["points"]:
            base = stock.get((p["batch"], p["context"]))
            q = {"batch": p["batch"], "context": p["context"], "step_ms": p.get("step_ms"), "p90_ms": p.get("p90_ms"),
                 "tokens_per_s": p.get("tokens_per_s")}
            if base and p.get("step_ms"):
                q["stock_ms"] = st.fmean(base)
                q["slowdown"] = p["step_ms"] / q["stock_ms"]
            nv = p.get("nvml", {})
            q["power_w"], q["sm_mhz"], q["sm_mhz_min"] = nv.get("power_w"), nv.get("sm_mhz"), nv.get("sm_mhz_min")
            fl = p.get("filler")
            if fl:
                q["filler_share_of_peak"] = fl.get("share_of_peak")
                q["filler_share_at_max_clock"] = fl.get("share_of_peak_at_max_clock")
                q["filler_mma_per_sm_clk_in_block"] = fl.get("mma_per_sm_clk_in_block")
                if alone.get("mma_per_sm_clk_in_block") and fl.get("mma_per_sm_clk_in_block"):
                    q["filler_in_block_vs_alone"] = fl["mma_per_sm_clk_in_block"] / alone["mma_per_sm_clk_in_block"]
                if alone.get("share_of_peak") and fl.get("share_of_peak"):
                    q["filler_share_vs_alone"] = fl["share_of_peak"] / alone["share_of_peak"]
                q["most_blocks_on_one_sm"] = fl.get("most_blocks_on_one_sm")
            pts.append(q)
        row["points"] = pts
        out["arms"].append(row)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

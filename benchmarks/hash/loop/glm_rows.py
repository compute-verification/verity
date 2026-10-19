"""A GLM-5.3 TP8 row for the hash loop from one of circuits' served runs (WS-D: `out/<arm>/summary.json` and `runs.jsonl`, control
and the commit arm paired 3 times a batch), as the loop's result doc (research/result/v0.1), so plot.py draws it as workload glm-5.3.
Measures at the loop's batch (B32): tpot_overhead_pct, the median over the pairs of 100 (TPOT_arm / TPOT_control - 1), with
`range` its min and max over the pairs (3 pairs: a range, not a 95% interval); throughput_overhead_pct, 100 (1 - tok/s_arm /
tok/s_control) a pair from runs.jsonl's generated_tokens / wall_s, median and range the same way.

    python3 benchmarks/hash/loop/glm_rows.py RUN_DIR --arm-dir rows --arm sha512 > result.json
    research data put --kind bench-result/v1 --meta @result.json --file result.json --preserve
    python3 benchmarks/hash/loop/label.py RUN --row 0 --workload glm-5.3 --result-art art:... --hypothesis "..."
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--arm-dir", required=True, help="the arm's directory under out/ (rows, rows-layer-4k)")
    ap.add_argument("--arm", required=True, help="the loop's arm name (sha512, sha512-layer)")
    ap.add_argument("--batch", default="32")
    a = ap.parse_args()
    d = a.run_dir / "out" / a.arm_dir
    summ = json.loads((d / "summary.json").read_text())
    b = summ["batches"][a.batch]
    runs = [json.loads(line) for line in (d / "runs.jsonl").read_text().splitlines() if line.strip()]
    tps: dict[tuple[str, int], float] = {}
    for r in runs:
        if str(r["batch"]) == a.batch:
            tps[(r["arm"], r["pair"])] = r["generated_tokens"] / r["wall_s"]
    pairs = sorted({p for _, p in tps})
    tput = [100 * (1 - tps[(summ["arm"], p)] / tps[("control", p)]) for p in pairs]
    tr = b["tpot_ratio"]
    meas = [
        {"name": "tpot_overhead_pct", "arm": a.arm, "value": round(100 * (tr["median"] - 1), 3), "unit": "%", "n": tr["pairs"],
         "range": [round(100 * (tr["min"] - 1), 3), round(100 * (tr["max"] - 1), 3)]},
        {"name": "throughput_overhead_pct", "arm": a.arm, "value": round(statistics.median(tput), 3), "unit": "%", "n": len(tput),
         "range": [round(min(tput), 3), round(max(tput), 3)]},
        {"name": "tpot_ms", "arm": a.arm, "value": round(1e3 * b[summ["arm"]]["tpot_s"], 4), "unit": "ms"},
        {"name": "tpot_ms", "arm": "stock", "value": round(1e3 * b["control"]["tpot_s"], 4), "unit": "ms"},
    ]
    ok = bool(summ.get("gates_passed")) and bool(b.get("gates_passed")) and bool(b.get("control_tokens_equal"))
    run_id = a.run_dir.name
    doc = {
        "schema": "research/result/v0.1",
        "run_id": run_id,
        "workload_fingerprint": {"bench": "circuits WS-D served GLM-5.3 TP8 eager", "batch": int(a.batch), "arm": summ["arm"],
                                 "committer": summ.get("committer"), "row_bytes": summ.get("row_bytes"), "world": summ.get("world"),
                                 "pairs": tr["pairs"], "pin": {"state": "unpinned"}},
        "validation": {"status": "passed" if ok else "failed", "checker": "circuits WS-D gates (tokens equal to control)",
                       "detail": f"gates_passed {summ.get('gates_passed')}, B{a.batch} control_tokens_equal {b.get('control_tokens_equal')}",
                       "evidence": []},
        "measurements": meas,
        "measurement_files": [f"out/{a.arm_dir}/summary.json", f"out/{a.arm_dir}/runs.jsonl"],
    }
    json.dump(doc, sys.stdout, indent=1)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

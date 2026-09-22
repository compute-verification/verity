#!/usr/bin/env python
"""[L1 RECORD 2026-09-10] Cost log by class x stage from the record's costlog.jsonl (MEASURED clocks appended by every stage script) plus the
pod-hours (MODELED at RunPod list prices until the invoice).  Writes costlog.json and prints the markdown table for report.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-pod-hours", type=float, required=True)
    ap.add_argument("--cpu-pod-hours", type=float, required=True)
    ap.add_argument("--gpu-rate", type=float, default=0.69, help="$/h RunPod RTX 4090 secure cloud list price (MODELED)")
    ap.add_argument("--cpu-rate", type=float, default=0.10, help="$/h RunPod CPU pod list price (MODELED)")
    a = ap.parse_args(argv)
    rows = []
    bad = 0
    for line in (Path(a.record) / "costlog.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1  # one malformed early line (a `bc`-less shell wrote an empty seconds field); its stage was re-logged by the script
    agg: OrderedDict[tuple[str, str], dict] = OrderedDict()
    for r in rows:
        stage = r["stage"]
        m = re.match(r"^(prove|verify) chunk\d+ (.*)$", stage)
        if m:
            stage = f"{m.group(1)} per chunk {m.group(2)}"
        key = (r["class"], stage)
        e = agg.setdefault(key, {"class": r["class"], "stage": stage, "seconds": 0.0, "n": 0, "hardware": r.get("hardware", ""), "label": r.get("label", "MEASURED"), "batch": r.get("batch", ""), "notes": []})
        e["seconds"] += float(r["seconds"])
        e["n"] += 1
        if r.get("note") and len(e["notes"]) < 2:
            e["notes"].append(r["note"])
    out_rows = []
    for e in agg.values():
        e["seconds"] = round(e["seconds"], 1)
        if e["n"] > 1:
            e["batch"] = f"{e['n']} x ({e['batch']})"
            e["mean_seconds"] = round(e["seconds"] / e["n"], 1)
        out_rows.append(e)
    by_class: dict[str, float] = {}
    for e in out_rows:
        by_class[e["class"]] = round(by_class.get(e["class"], 0.0) + e["seconds"], 1)
    pods = {"record-gpu": {"hours": a.gpu_pod_hours, "rate_usd_per_h": a.gpu_rate, "usd": round(a.gpu_pod_hours * a.gpu_rate, 2), "hardware": "1x RTX 4090, 8.5 vCPU (cgroup quota; 80 visible), 754 GiB host RAM (shared)", "label": "MODELED (list price; hours MEASURED from pod create to terminate)"},
            "record-cpu": {"hours": a.cpu_pod_hours, "rate_usd_per_h": a.cpu_rate, "usd": round(a.cpu_pod_hours * a.cpu_rate, 2), "hardware": "8 vCPU CPU pod", "label": "MODELED (list price; hours MEASURED)"}}
    doc = {"schema": "veritor/rec-p1-costlog/v1", "rows": out_rows, "seconds_by_class": by_class, "pods": pods, "total_usd_modeled": round(sum(p["usd"] for p in pods.values()), 2), "malformed_lines_skipped": bad,
           "note": "seconds are wall clocks of the stage scripts (MEASURED); classes: serving (prover-side, the model runs), replay (prover-side re-execution + interior commitment), proving (SP1 cuda), verifier (everything the verifier runs: sampling, mapping, statements, host verify, checks, negatives)"}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    md = ["| class | stage | hardware | batch | seconds | label |", "|---|---|---|---|---:|---|"]
    for e in out_rows:
        extra = f" (mean {e['mean_seconds']} s)" if "mean_seconds" in e else ""
        md.append(f"| {e['class']} | {e['stage']} | {e['hardware']} | {e['batch']} | {e['seconds']}{extra} | {e['label']} |")
    md.append("")
    md.append("| class | seconds |\n|---|---:|")
    for k, v in by_class.items():
        md.append(f"| {k} | {v} |")
    md.append("")
    md.append(f"Pods (MODELED at list price): record-gpu {a.gpu_pod_hours} h x ${a.gpu_rate}/h = ${pods['record-gpu']['usd']}; record-cpu {a.cpu_pod_hours} h x ${a.cpu_rate}/h = ${pods['record-cpu']['usd']}; total ≈ ${doc['total_usd_modeled']}.")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())

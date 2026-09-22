"""[COMPACT-2] Turn a finished `compact_whole_b0.py` run into the VBENCH Table 3(c) stage file.

    python scripts/compact_stages_report.py <out>.stages.json <out>c.json docs/data/compact/whole_b0_stages.json \
        [--pod "jciert26rhfysr: cpu3m 32 vCPU / 256 GB"] [--commit SHA] [--baseline "text"]

Per-stage wall seconds and RSS (psutil rss before/after, ru_maxrss high-water) for adapter -> validate_plan -> Boundary -> Bound,
plus the input sizes (gates, nodes, runs, leaves), CPU model, thread count and cold/reused state.  The Boundary stage is folded
into validate_plan because validate_plan is what constructs every unit's Boundary; the separate `boundary` row is the statistics pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def lscpu_line() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    keep = ("Model name", "Socket(s)", "Thread(s) per core", "CPU max MHz")
    return "; ".join(line.split(":", 1)[1].strip() for line in out.splitlines() if line.split(":")[0].strip() in keep)


def stage(name: str, r: dict, **extra) -> dict:
    return {"stage": name, "wall_s": r.get("seconds"), "rss_before_mb": r["rss_before_mb"], "rss_after_mb": r.get("rss_after_mb"),
            "maxrss_so_far_mb": r.get("maxrss_mb"), "unfinished": "seconds" not in r, **extra}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages_json")
    ap.add_argument("run_json", help="<out>.json or the --recompare <out>c.json (needs plan_results, root_inputs, stages)")
    ap.add_argument("out")
    ap.add_argument("--pod", default="")
    ap.add_argument("--commit", default="")
    ap.add_argument("--baseline", default="", help="free text: the run without a given optimisation, if measured")
    ap.add_argument("--lscpu", default=lscpu_line())
    ap.add_argument("--append-plan", action="store_true",
                    help="add this run's plan under `plans[<spec>]` of an existing stage file (the root-plan rows are left untouched)")
    args = ap.parse_args()

    st = json.load(open(args.stages_json))
    d = json.load(open(args.run_json))
    rows = {r["stage"]: r for r in st["stages"]}
    spec = next(iter(d["plan_results"]))
    pr = d["plan_results"][spec]
    prior = d.get("prior_stages") or d["stages"]
    program, adapt = rows["program"], rows["adapt"]
    validate = rows.get(f"validate[{spec}]", {})
    boundary = rows.get(f"boundary[{spec}]")
    bound = rows.get(f"bound[{spec}]")
    bnd = pr.get("bound") or {}

    if args.append_plan:
        out_path = Path(args.out)
        doc = json.loads(out_path.read_text())
        entry = {
            "plan": {"name": pr.get("plan"), "constructor": pr.get("constructor"), "vu_level": pr.get("vu_level"), "cu_strategy": pr.get("cu_strategy"),
                     "vus": pr.get("vus"), "cus": pr.get("cus")},
            "run": args.pod, "commit": args.commit,
            "status": pr.get("status"), "codes": pr.get("codes"),
            "stages": [stage(r["stage"], r, **{k: v for k, v in r.items() if k not in ("stage", "seconds", "rss_before_mb", "rss_after_mb", "maxrss_mb",
                                                                                           "started_s", "inputs", "compaction", "codes", "reasons")})
                       for r in st["stages"] if r["stage"] not in ("symbolic", "compare")],
            "boundary": pr.get("boundary"),
            "bound": None if not bnd else {"U_bits": bnd.get("U_bits"), "capped": bnd.get("capped"), "output_bits": bnd.get("output_bits"),
                                           "uncapped_terms": (bnd.get("report") or {}).get("terms"), "rows": bnd.get("rows"),
                                           "other_policies": bnd.get("other_policies")},
            "totals": {"wall_s_program_to_bound": round(sum(r["seconds"] for r in st["stages"] if r["stage"] not in ("symbolic", "compare") and "seconds" in r), 2),
                       "peak_rss_mb": max(r.get("maxrss_mb") or 0 for r in st["stages"])},
            "final_status": d.get("status"),
            "note": args.baseline,
        }
        doc.setdefault("plans", {})[spec] = entry
        out_path.write_text(json.dumps(doc, indent=1) + "\n")
        print(args.out, spec, json.dumps(entry["totals"]), entry["final_status"])
        return 0

    doc = {
        "schema": "compact/whole-b0-stages/v1",
        "lane": "COMPACT-2",
        "record": st.get("record"),
        "program": program["function"],
        "program_digest": program.get("digest"),
        "commit": args.commit,
        "run": args.pod,
        "machine": {"cpu": st["machine"]["cpu"], "lscpu": args.lscpu, "host_cpus_visible": st["machine"]["nproc"],
                    "platform": st["machine"]["platform"], "python": st["machine"]["python"], "threads": 1,
                    "state": st["machine"]["state"], "note": "pure-Python, single thread, no numpy in the hot path"},
        "policy": st["policy"],
        "inputs": {
            "gates": program["gates"], "input_gates": program["input_gates"], "root_nodes": program["root_nodes"],
            "definitions": adapt["definitions"], "nodes": adapt["nodes"], "ref_runs": adapt["ref_runs"], "ref_leaves": adapt["ref_leaves"],
            "unrolled_scan_bodies": adapt["unrolled"], "compaction": next((r.get("compaction") for r in prior if r["stage"] == "adapt"), None),
            "plan": {"name": pr["plan"], "constructor": pr["constructor"], "vu_level": pr["vu_level"], "cu_strategy": pr["cu_strategy"],
                     "vus": pr["vus"], "cus": pr["cus"]},
        },
        "stages": [
            stage("program (checkpoint descriptor -> verity.ir Program, incl. descriptor digest)", program, gates=program["gates"],
                  digest_s=program.get("digest_seconds")),
            stage("adapter (verity.ir -> plans.ir, compact=True)", adapt, definitions=adapt["definitions"], nodes=adapt["nodes"],
                  ref_runs=adapt["ref_runs"], ref_leaves=adapt["ref_leaves"]),
            stage("circuit + plan", rows[f"plan[{spec}]"], vus=pr["vus"], cus=pr["cus"], circuit_s=rows["circuit"]["seconds"]),
            stage("validate_plan (includes hierarchical_boundary for every VU/CU: Boundary objects are built inside validate_plan)", validate,
                  status=validate["status"], hops=validate["hops"], escape_steps=validate["escape_steps"], vu_edges=validate["vu_edges"],
                  cu_edges=validate["cu_edges"], boundary_gates_vu=pr["boundary"]["vu"]["boundary_gates"],
                  boundary_runs_vu=pr["boundary"]["vu"]["boundary_runs"], boundary_gates_cu=pr["boundary"]["cu"]["boundary_gates"],
                  boundary_runs_cu=pr["boundary"]["cu"]["boundary_runs"], largest_unit_boundary_gates=pr["boundary"]["vu"]["largest_unit_boundary_gates"]),
            stage("boundary statistics pass (walk every Boundary's GateMap runs)", boundary, units=pr["boundary"]["vu"]["units"] + pr["boundary"]["cu"]["units"]),
            stage("Bound (hierarchical_table: run-aware Intrinsic/Reach memo + fold at (q, s, delta))", bound, rows=bound["rows"],
                  vu_kinds=bound["vu_kinds"], cu_kinds=bound["cu_kinds"], reach_steps=bound["reach_steps"], intrinsic_entries=bound["intrinsic_entries"],
                  U_bits=bound["U_bits"], capped=bound["capped"], output_bits=bound["output_bits"],
                  uncapped_knapsack_bits=bnd["report"]["terms"]["knapsack_bits"]),
        ],
        "totals": {"wall_s_program_to_bound": round(sum(r["seconds"] for r in st["stages"] if r["stage"] not in ("symbolic", "compare")), 2),
                   "peak_rss_mb": max(r["maxrss_mb"] for r in st["stages"])},
        "baseline": args.baseline,
        "note": "Per-stage wall seconds and RSS for the whole-B0 program pinned by the record's certificate; the Boundary stage is folded into "
                "validate_plan because validate_plan constructs every unit's Boundary (the separate row is only the statistics pass).",
    }
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(args.out, json.dumps(doc["totals"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

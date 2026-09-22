#!/usr/bin/env python
"""[L1 RECORD 2026-09-10] The record's population directory for P1/cu=whole of the B0 program, from the validated artifact.

Writes, into ``--out``:

* ``population.compact.json`` -- ``veritor/population/v2`` rows of the plan (``e2e.population_v2 build --no-validate``: the plan's
  UnitRuns, no validation here); the partition digest is RE-DERIVED from the rows and must equal the validated one
  (``data/rec/p1-b0-lp32s15-validated.json#plan.partition_digest``), else REFUSED;
* ``compact_validation.json`` -- ``veritor/compact-validation/v1`` carrying COMPACT-P1's verdict VERBATIM (status, reasons, coverage) as the
  validation scope; nothing is re-validated or relabelled here;
* ``maps.json`` / ``index.json`` -- the headers ``e2e.protocol_instance make`` reads (program gates, input gates, plan identity).

    PYTHONPATH=. .venv/bin/python scripts/rec_p1_population.py --validated data/rec/p1-b0-lp32s15-validated.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--validated", default=str(ROOT / "data" / "rec" / "p1-b0-lp32s15-validated.json"))
    ap.add_argument("--case", default="B0")
    ap.add_argument("--lp", type=int, default=32)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--plan", default="P1/cu=whole")
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    val = json.loads(Path(a.validated).read_text())
    plan, prog, ver = val["plan"], val["program"], val["validation"]
    if plan["name"] != a.plan:
        print(f"[rec_p1_population] REFUSED: validated artifact is for plan {plan['name']!r}, not {a.plan!r}")
        return 1
    expect = plan["partition_digest"]
    # 1. rows (population/v2), digest re-derived and compared with the validated partition digest
    t = time.perf_counter()
    cmd = [a.python, "-m", "e2e.population_v2", "build", "--case", a.case, "--lp", str(a.lp), "--steps", str(a.steps), "--plan", a.plan, "--out", str(out),
           "--no-validate", "--expect-partition-digest", expect]
    print("[rec_p1_population] $", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, env={**dict(__import__("os").environ), "PYTHONPATH": str(POC)}).returncode
    if rc != 0:
        print(f"[rec_p1_population] REFUSED: population_v2 build exited {rc}")
        return rc
    secs = round(time.perf_counter() - t, 1)
    pop = json.loads((out / "population.compact.json").read_text())
    for k, v in (("partition_digest", expect), ("program_digest", prog["digest"]), ("vu_count", plan["vus"]), ("ru_count", plan["cus"]),
                 ("plan_digest", val["plan_digest"]["digest"])):
        if pop.get(k) != v:
            print(f"[rec_p1_population] REFUSED: population {k} {pop.get(k)!r} != validated artifact {v!r}")
            return 2
    # 2. the validation artifact: COMPACT-P1's verdict verbatim
    scope = (f"{ver['vus_validated']} of {plan['vus']} VUs validated ({100.0 * ver['vus_validated'] / plan['vus']:.1f}%); status {ver['status']!r}; "
             f"reasons {ver.get('reasons')}; gaps {ver.get('gaps')}; mode: {ver.get('mode')}")
    art = {"schema": "veritor/compact-validation/v1", "program_digest": prog["digest"], "program": prog["function"],
           "gates": prog["gates"], "input_gates": prog["input_gates"],
           "plan_name": plan["name"], "plan_digest": val["plan_digest"]["digest"], "plan_digest_kind": val["plan_digest"]["kind"],
           "partition_digest": expect, "partition_digest_kind": plan["partition_digest_kind"],
           "vu_level": plan["vu_level"], "cu_strategy": plan["cu_strategy"],
           "status": ver["status"], "codes": [r[0] for r in ver.get("reasons", [])], "reasons": ver.get("reasons"),
           "vus": plan["vus"], "cus": plan["cus"], "vus_validated": ver["vus_validated"], "gaps": ver.get("gaps"),
           "vu_edges": ver.get("vu_edges"), "cu_edges": ver.get("cu_edges"), "seconds": None, "maxrss_mb": ver.get("worker_maxrss_mb"),
           "scope": scope, "validator": "validate_plan(retain=False), sharded (COMPACT-P1)",
           "verdict_source": f"{a.validated} (schema {val['schema']}, lane {val['lane']}, {val['utc']}): COMPACT-P1's sharded streamed validation of "
                             f"whole B0 -- cited VERBATIM; NOT recomputed here; population rows built here re-derive the same partition digest",
           "utc": utc(), "label": "MEASURED (COMPACT-P1 pods) -- cited; validation scope as stated in `scope`"}
    (out / "compact_validation.json").write_text(json.dumps(art, indent=1, sort_keys=True) + "\n")
    # 3. headers protocol_instance reads
    header = {"schema": "veritor/population-maps/v1", "program": {"name": prog["function"], "digest": prog["digest"], "gates": prog["gates"]},
              "plan": {"name": plan["name"], "descriptor_digest": val["plan_digest"]["digest"], "digest_kind": val["plan_digest"]["kind"],
                       "partition_digest": expect, "partition_digest_kind": plan["partition_digest_kind"], "vus": plan["vus"], "cus": plan["cus"], "status": ver["status"]},
              "counts": {"program_values": None, "vus": plan["vus"], "rus": plan["cus"], "label": "MEASURED (validated plan counts, cited)"},
              "tables": "not materialized (population/v2 rows are the population)", "population_v2": {"content_sha256": pop["content_sha256"], "rows": len(pop["rows"]),
                                                                                                          "kinds": len(pop["kinds"]["entries"]), "build_seconds": secs}}
    (out / "maps.json").write_text(json.dumps(header, indent=1, sort_keys=True) + "\n")
    (out / "index.json").write_text(json.dumps({"case": a.case, "LP": a.lp, "STEPS": a.steps, "program": {"name": prog["function"], "digest": prog["digest"],
                                                "gates": prog["gates"], "input_gates": prog["input_gates"]}, "plan": header["plan"]}, indent=1, sort_keys=True) + "\n")
    rss = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)
    print(f"[rec_p1_population] OK -> {out}: population/v2 {len(pop['rows'])} rows, {pop['vu_count']} VUs / {pop['ru_count']} RU per instance, partition {expect[:16]} "
          f"(re-derived == validated), content {pop['content_sha256'][:16]}; validation scope: {scope[:120]}...; {secs} s, child maxrss {rss} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

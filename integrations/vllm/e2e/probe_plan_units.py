"""Measure RA's validator (``veritor.plans.validate``) over candidate B1 plan units under an address-space cap.

    python -m e2e.probe_plan_units --case B1 --cap-gb 28 --out probe.json [--units layer gemm:1536x2048 ...]

One subprocess per unit and partition (P1, then P2); each records validate seconds, peak RSS and the outcome
(``valid`` / ``invalid`` / ``MemoryError`` under the cap / killed).  A measurement tool for the plan stage's unit
choice on a memory-bound pod (int3: the 31 GB cgroup of the RTX 4090 pod), not a stage of the pipeline.
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


def _child(case: str, unit: str, tag: str, cap_gb: float, out: Path) -> None:
    cap = int(cap_gb * (1 << 30))
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    doc: dict = {"case": case, "unit": unit, "partition": tag, "cap_gb": cap_gb, "status": "starting"}

    def dump() -> None:
        doc["maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
        out.write_text(json.dumps(doc, indent=1, sort_keys=True))

    dump()
    try:
        from e2e.run_composed import static_configuration, unit_plans
        from veritor.plans.validate import validate_plan

        t0 = time.perf_counter()
        Q, c, partitions = unit_plans(static_configuration(case), unit)
        doc["build_seconds"] = round(time.perf_counter() - t0, 2)
        doc["program"] = Q.name
        doc["status"] = "built"
        dump()
        plan = dict(partitions)[tag]
        doc["vus"] = len(plan.verification_units)
        t0 = time.perf_counter()
        v = validate_plan(c, plan)
        doc["validate_seconds"] = round(time.perf_counter() - t0, 2)
        doc["status"] = v.status
        doc["codes"] = [str(x) for x in v.codes]
        doc["vu_edges"], doc["cu_edges"] = v.vu_edges, v.cu_edges
        dump()
        sys.exit(0 if v.status == "valid" else 1)
    except MemoryError:
        doc["status"] = "MemoryError under the address-space cap"
        dump()
        sys.exit(3)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        doc["status"] = f"error: {type(e).__name__}: {e}"[:400]
        dump()
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="B1")
    ap.add_argument("--cap-gb", type=float, default=28.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", nargs="*", default=["layer", "gemm:1536x2048", "gemm:1536x1536", "gemm:8960x1536", "gemm:1536x17920"])
    ap.add_argument("--partitions", nargs="*", default=["P1", "P2"])
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--_child", nargs=3, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    out = Path(a.out)
    if a._child:
        unit, tag, path = a._child
        _child(a.case, unit, tag, a.cap_gb, Path(path))
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for unit in a.units:
        for tag in a.partitions:
            tmp = out.with_suffix(f".{unit.replace(':', '-')}.{tag}.json")
            t0 = time.perf_counter()
            try:
                rc = subprocess.call([sys.executable, "-m", "e2e.probe_plan_units", "--case", a.case, "--cap-gb", str(a.cap_gb), "--out", str(out),
                                      "--_child", unit, tag, str(tmp)], timeout=a.timeout, cwd=str(POC))
                timed_out = False
            except subprocess.TimeoutExpired:
                rc, timed_out = None, True
            doc = json.loads(tmp.read_text()) if tmp.exists() else {"unit": unit, "partition": tag, "status": "no record"}
            doc["exit_code"], doc["wall_seconds"] = rc, round(time.perf_counter() - t0, 1)
            if timed_out:
                doc["status"] = f"timeout after {a.timeout}s at phase {doc.get('status')}"
            elif rc not in (0, 1, 2, 3):
                doc["status"] = f"killed (exit {rc}) at phase {doc.get('status')}"
            results.append(doc)
            print(json.dumps(doc, sort_keys=True), flush=True)
            out.write_text(json.dumps({"schema": "veritor-plans/unit-probe/v1", "case": a.case, "cap_gb": a.cap_gb, "results": results}, indent=1, sort_keys=True))
            if doc["status"] not in ("valid", "invalid") and tag == "P1":
                break  # P2 refines P1's units; if P1 does not fit, P2 does not either


if __name__ == "__main__":
    main()

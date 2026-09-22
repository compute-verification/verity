"""Attempt RA's validator over the *whole* ``Serve`` program (int3-brief §3: "if memory allows, else the finding
with the number").  Run as a subprocess under an address-space cap so the attempt is a measurement, not a risk to
the run that launches it (``run_composed`` stage ``plan``, ``--full-program try``)::

    python -m e2e.validate_full --case B0 --workload e2e/workload-B0-small.json --out plan/full_program.json --cap-gb 20

The plan validated is the root-level analogue of ``layer_plans``' P1: one VU per root node of ``Serve`` (every
member / iteration of the node), one CU over the ``Input`` batches and one CU over every call -- the coarsest
partition, i.e. the cheapest question the validator can be asked about the full program.  Whatever happens (valid,
invalid, MemoryError under the cap, killed) is written to ``--out`` by this process or, on a crash, by the parent
from the exit status; every number is labelled with the phase it belongs to.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def maxrss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(ru / 1024, 1) if platform.system() == "Linux" else round(ru / 1024 / 1024, 1)  # KiB on Linux, bytes on macOS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", default="B0")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-gb", type=float, default=20.0, help="RLIMIT_AS for this process (0 = no cap)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"schema": "verity-vllm/e2e-composed/full-program-validation/v1", "case": args.case, "cap_gb": args.cap_gb,
                           "plan": "root-level P1: one VU per root node of Serve (all members/iterations), CU(inputs) + CU(calls)",
                           "validated": False, "phases": {}, "platform": platform.platform(), "python": sys.version.split()[0]}

    def dump(status: str, **extra: Any) -> None:
        doc.update(extra)
        doc["status"] = status
        doc["maxrss_mb"] = maxrss_mb()
        out.write_text(json.dumps(doc, indent=1, default=str) + "\n")

    if args.cap_gb > 0 and platform.system() == "Linux":
        cap = int(args.cap_gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    dump("started")
    try:
        from verity.ir import Program, bind
        from verity_vllm.registry import b1

        from veritor.plans.boundary import Part, Parts
        from veritor.plans.plan import VerificationPlan
        from veritor.plans.validate import validate_plan
        from veritor.plans.verity_ir_adapter import program_from_verity_ir

        workload = json.loads(Path(args.workload).read_text())
        lp = max(int(x) for x in workload["prompt_lengths"])
        steps = max(int(x) for x in workload["max_tokens"]) - 1
        config = dict(getattr(b1, f"{args.case}_CONFIG"))
        t = time.perf_counter()
        program = Program(bind(b1.Serve, C=config, LP=lp, STEPS=steps))
        doc["phases"]["program"] = {"seconds": round(time.perf_counter() - t, 2), "gates": program.gates, "input_gates": program.input_gates, "maxrss_mb": maxrss_mb()}
        dump("program built")
        t = time.perf_counter()
        Q = program_from_verity_ir(program)
        doc["phases"]["bridge"] = {"seconds": round(time.perf_counter() - t, 2), "maxrss_mb": maxrss_mb()}
        dump("bridged")
        t = time.perf_counter()
        c = Q.circuit()
        root = Q.root
        doc["phases"]["circuit"] = {"seconds": round(time.perf_counter() - t, 2), "maxrss_mb": maxrss_mb(), "root_nodes": len(root.nodes),
                                    "root_forms": sorted({n.form for n in root.nodes}), "size": getattr(c, "size", None)}
        dump("circuit built")
        inputs = [k for k, n in enumerate(root.nodes) if n.form == "batch" and n.fn.name == "Input"]
        calls = [k for k, n in enumerate(root.nodes) if k not in inputs]

        def whole(k: int) -> Parts:
            return Parts(c, [Part((), k, 0, root.nodes[k].n)])

        t = time.perf_counter()
        vus = [whole(k) for k in inputs] + [whole(k) for k in calls]
        cus = [Parts(c, [Part((), k, 0, root.nodes[k].n) for k in inputs]), Parts(c, [Part((), k, 0, root.nodes[k].n) for k in calls])]
        plan = VerificationPlan(f"root P1 over {Q.name}", vus, cus)
        doc["phases"]["plan"] = {"seconds": round(time.perf_counter() - t, 2), "vus": len(vus), "cus": len(cus), "maxrss_mb": maxrss_mb()}
        dump("plan built")
        t = time.perf_counter()
        v = validate_plan(c, plan)
        doc["phases"]["validate"] = {"seconds": round(time.perf_counter() - t, 2), "maxrss_mb": maxrss_mb()}
        dump("valid" if v.status == "valid" else v.status, validated=True, verdict={"status": v.status, "codes": [str(x) for x in v.codes], "reasons": list(v.reasons)[:5],
                                                                                    "vu_edges": v.vu_edges, "cu_edges": v.cu_edges, "visits": v.visits})
        return 0 if v.status == "valid" else 1
    except MemoryError:
        dump("unsupported: MemoryError under the address-space cap", error="MemoryError", traceback=traceback.format_exc()[-2000:])
        return 3
    except Exception as e:  # noqa: BLE001
        dump(f"error: {type(e).__name__}", error=f"{type(e).__name__}: {e}"[:400], traceback=traceback.format_exc()[-2000:])
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Whole-``Serve`` plan-bridge OOM diagnostic (INT-PAD-2; integration-pad.md §6 req. 3).

The measured failure: ``Serve{C=B0,LP=32,STEPS=15}`` (518,837,608 gates) builds at 5.8 GB RSS and the
bridge to the plan IR (``veritor.plans.verity_ir_adapter.program_from_verity_ir``) raises MemoryError at
RSS 17.9 GB under a 20 GB cap (e2e-composed.md §11.5).  This box's guardian SIGKILLs any process over
4 GB RSS, so the whole program cannot be re-run here; instead this module measures the bridge on a
ladder of scaled-down instances (``NL`` = layer count, same shape otherwise) in one capped subprocess
per point, fits bridge-memory against gate count, and extrapolates to the full program:

* if the fit is affine in gates and the extrapolated need fits a feasible box (~115 GB on the capture
  pod, 256 GB commodity), the failure is a *memory wall* -- a one-shot larger-memory diagnostic decides;
* if superlinear, or the extrapolated need exceeds feasible RAM, the failure is *structural* and goes
  back to PLAN/PAD-SPEC with these numbers.

    python -m e2e.bridge_diag --out /tmp/bridge-diag [--nl 1 2 3] [--cap-gb 3.3] [--full]

``--full`` skips the ladder and runs the whole program (build + bridge + plan + validate), for the
larger-memory box; every phase is dumped as it completes so an OOM still leaves the numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

FULL_GATES = 518_837_608  # Serve{C=B0,LP=32,STEPS=15}, e2e-composed.md §11.5
FULL_BUILD_RSS_MB = 5_800.0  # measured on the pod (same doc); the program object stays live during the bridge


def maxrss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(ru / 1024, 1) if platform.system() == "Linux" else round(ru / 1024 / 1024, 1)


def _point(nl: int, lp: int, steps: int, trace: bool, v: int | None = None) -> dict[str, Any]:
    """One ladder point, run inside the capped subprocess: build Serve{C=B0[NL=nl]}, bridge, count."""

    from verity.ir import Program, bind
    from verity_vllm.registry import b1

    from veritor.plans.verity_ir_adapter import adapt

    config = dict(b1.B0_CONFIG)
    config["NL"] = nl
    if v is not None:
        config["V"] = v  # the embedding/lm-head tables dominate the gate count; a reduced vocabulary
        # gives ladder points that fit under the box guardian while the x-axis stays `gates`
    doc: dict[str, Any] = {"nl": nl, "lp": lp, "steps": steps, "v": config["V"]}
    t = time.perf_counter()
    program = Program(bind(b1.Serve, C=config, LP=lp, STEPS=steps))
    doc["build_s"] = round(time.perf_counter() - t, 2)
    doc["gates"] = program.gates
    doc["build_maxrss_mb"] = maxrss_mb()
    if trace:
        tracemalloc.start(1)
    t = time.perf_counter()
    adapted = adapt(program)
    doc["bridge_s"] = round(time.perf_counter() - t, 2)
    doc["bridge_maxrss_mb"] = maxrss_mb()
    root = adapted.program.root
    doc["root_nodes"] = len(root.nodes)
    doc["unrolled_batches"] = len(adapted.unrolled)
    doc["unrolled_members"] = sum(m for _, _, m in adapted.unrolled)
    doc["fn_definitions"] = len({id(n.fn) for n in root.nodes})
    if trace:
        current, peak = tracemalloc.get_traced_memory()
        snap = tracemalloc.take_snapshot()
        top = snap.statistics("filename")[:8]
        doc["tracemalloc"] = {
            "current_mb": round(current / 1e6, 1),
            "peak_mb": round(peak / 1e6, 1),
            "top_files": [{"file": s.traceback[0].filename.split("/")[-1], "mb": round(s.size / 1e6, 1), "count": s.count} for s in top],
        }
        tracemalloc.stop()
    return doc


def _full(cap_gb: float, out: Path) -> dict[str, Any]:
    """The whole program on a larger-memory box: build -> bridge -> root P1 plan -> validate, dumped per phase."""

    from verity.ir import Program, bind
    from verity_vllm.registry import b1

    from veritor.plans.boundary import Part, Parts
    from veritor.plans.plan import VerificationPlan
    from veritor.plans.validate import validate_plan
    from veritor.plans.verity_ir_adapter import program_from_verity_ir

    doc: dict[str, Any] = {"mode": "full", "cap_gb": cap_gb, "case": "B0", "lp": 32, "steps": 15}

    def dump(status: str) -> None:
        doc["status"] = status
        doc["maxrss_mb"] = maxrss_mb()
        out.write_text(json.dumps(doc, indent=1, default=str) + "\n")

    # in-run RSS curve, so a death under the cap still leaves the growth slope
    import threading

    samples: list[tuple[float, float]] = []
    doc["rss_samples"] = samples
    t0 = time.perf_counter()

    def _sample() -> None:
        while True:
            time.sleep(10)
            samples.append((round(time.perf_counter() - t0, 1), maxrss_mb()))
            try:
                out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
            except Exception:
                pass

    threading.Thread(target=_sample, daemon=True).start()

    workload = json.loads((HERE / "workload-B0-small.json").read_text())
    lp = max(int(x) for x in workload["prompt_lengths"])
    steps = max(int(x) for x in workload["max_tokens"]) - 1
    doc["lp"], doc["steps"] = lp, steps
    t = time.perf_counter()
    program = Program(bind(b1.Serve, C=dict(b1.B0_CONFIG), LP=lp, STEPS=steps))
    doc["build"] = {"seconds": round(time.perf_counter() - t, 2), "gates": program.gates, "maxrss_mb": maxrss_mb()}
    dump("built")
    t = time.perf_counter()
    Q = program_from_verity_ir(program)
    doc["bridge"] = {"seconds": round(time.perf_counter() - t, 2), "maxrss_mb": maxrss_mb()}
    dump("bridged")
    t = time.perf_counter()
    c = Q.circuit()
    root = Q.root
    doc["circuit"] = {"seconds": round(time.perf_counter() - t, 2), "maxrss_mb": maxrss_mb(), "root_nodes": len(root.nodes)}
    dump("circuit")
    inputs = [k for k, n in enumerate(root.nodes) if n.form == "batch" and n.fn.name == "Input"]
    calls = [k for k in range(len(root.nodes)) if k not in inputs]
    vus = [Parts(c, [Part((), k, 0, root.nodes[k].n)]) for k in inputs + calls]
    cus = [Parts(c, [Part((), k, 0, root.nodes[k].n) for k in inputs]), Parts(c, [Part((), k, 0, root.nodes[k].n) for k in calls])]
    plan = VerificationPlan(f"root P1 over {Q.name}", vus, cus)
    doc["plan"] = {"vus": len(vus), "cus": len(cus), "maxrss_mb": maxrss_mb()}
    dump("plan built")
    t = time.perf_counter()
    v = validate_plan(c, plan)
    doc["validate"] = {"seconds": round(time.perf_counter() - t, 2), "maxrss_mb": maxrss_mb(),
                       "status": v.status, "codes": [str(x) for x in v.codes], "vu_edges": v.vu_edges, "cu_edges": v.cu_edges}
    dump("validated: " + v.status)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--nl", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--lp", type=int, default=32)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--v", type=int, default=None, help="vocabulary override for the ladder (None = the case's V)")
    ap.add_argument("--cap-gb", type=float, default=3.3)
    ap.add_argument("--full", action="store_true", help="whole program (build+bridge+plan+validate); for the larger-memory box")
    ap.add_argument("--point", type=int, default=None, help=argparse.SUPPRESS)  # internal: run one point in this process
    ap.add_argument("--trace", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.cap_gb > 0 and platform.system() == "Linux" and (args.point is not None or args.full):
        cap = int(args.cap_gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    if args.full:
        doc = _full(args.cap_gb, out / "full_program.json")
        print(json.dumps({k: doc[k] for k in ("status", "maxrss_mb") if k in doc}))
        return 0

    if args.point is not None:
        try:
            doc = _point(args.point, args.lp, args.steps, args.trace, args.v)
        except MemoryError:
            doc = {"nl": args.point, "status": "MemoryError", "maxrss_mb": maxrss_mb()}
        (out / f"point-nl{args.point}.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps(doc)[:400])
        return 0

    # parent: one subprocess per point (the guardian kills >4 GB RSS; RLIMIT_AS in the child raises
    # MemoryError first), then the fit + extrapolation
    points: list[dict[str, Any]] = []
    for nl in args.nl:
        cmd = [sys.executable, "-m", "e2e.bridge_diag", "--out", str(out), "--point", str(nl),
               "--lp", str(args.lp), "--steps", str(args.steps), "--cap-gb", str(args.cap_gb)]
        if args.v is not None:
            cmd += ["--v", str(args.v)]
        if nl == max(args.nl):
            cmd.append("--trace")  # allocation-site attribution on the biggest point that fits
        r = subprocess.run(cmd, cwd=str(POC), capture_output=True, text=True)
        f = out / f"point-nl{nl}.json"
        if f.exists():
            points.append(json.loads(f.read_text()))
            print(f"nl={nl}: {points[-1].get('status', 'ok')} gates={points[-1].get('gates')} "
                  f"build={points[-1].get('build_maxrss_mb')}MB bridge_peak={points[-1].get('bridge_maxrss_mb')}MB")
        else:
            points.append({"nl": nl, "status": f"no output (exit {r.returncode})", "stderr": r.stderr[-500:]})
            print(f"nl={nl}: FAILED exit {r.returncode}")

    ok = [p for p in points if "gates" in p and "bridge_maxrss_mb" in p]
    doc: dict[str, Any] = {"schema": "veritor/bridge-diag/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "failure_reproduced": "e2e-composed.md §11.5: bridge MemoryError at 17.9 GB / 20 GB cap on 518,837,608 gates",
                           "ladder": points}
    if len(ok) >= 2:
        # affine fit of bridge increment (peak - build) against gates, least squares
        xs = [p["gates"] for p in ok]
        ys = [p["bridge_maxrss_mb"] - p["build_maxrss_mb"] for p in ok]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - b * mx
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        pred_full_mb = a + b * FULL_GATES
        doc["fit"] = {"model": "bridge_increment_mb = a + b*gates", "a_mb": round(a, 1), "b_mb_per_gate": b,
                      "bytes_per_gate": round(b * 1e6 / (1 << 20) * 1.048576, 3), "residuals_mb": [round(r, 1) for r in resid]}
        doc["extrapolation"] = {
            "full_gates": FULL_GATES,
            "predicted_bridge_increment_mb": round(pred_full_mb, 0),
            "predicted_total_mb": round(pred_full_mb + FULL_BUILD_RSS_MB, 0),
            "note": "total = predicted bridge increment + the measured 5.8 GB full-program build RSS; "
                    "validate_plan memory NOT included (the recorded failure is in the bridge, before validate)",
            "consistent_with_recorded_oom": bool(pred_full_mb + FULL_BUILD_RSS_MB > 17_900.0),
        }
    (out / "bridge_diag.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("bridge_diag: written", out / "bridge_diag.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

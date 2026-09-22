#!/usr/bin/env python3
"""[BRIDGE lane] Whole-Serve bridge memory diagnostic (GUIDANCE §36: one-shot high-memory diagnostic).

Reproduces exactly the phases of ``e2e/validate_full.py`` (program build -> bridge
``program_from_verity_ir`` -> circuit -> coarsest root plan -> ``validate_plan``) with a
background RSS sampler that appends to ``<out>.rss.tsv`` (flushed per line, so the timeline
survives a SIGKILL from a memory guardian or the OOM killer).  Nothing in ``e2e/`` is edited;
this is a measurement wrapper, not a fix.

    python scripts/bridge_diag.py --case B0 --lp 32 --steps 15 --out /tmp/bridge/full.json

``--steps``/``--lp`` default to the values ``validate_full`` derives from
``e2e/workload-B0-small.json`` (LP=32, STEPS=15 = max_tokens-1).  Smaller values scale the
program down for growth-profile runs.  ``--tracemalloc`` snapshots the top allocation sites
after the bridge (small scales only; it multiplies wall time).  ``--cap-gb`` sets RLIMIT_AS
on Linux (no-op on macOS, where the box guardian's 4 GB RSS SIGKILL is the effective cap).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import threading
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


def rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1 << 20), 1)
    except Exception:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(ru / 1024, 1) if platform.system() == "Linux" else round(ru / (1 << 20), 1)


def maxrss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(ru / 1024, 1) if platform.system() == "Linux" else round(ru / (1 << 20), 1)


class Sampler(threading.Thread):
    """Appends ``elapsed_s\trss_mb\tphase`` lines; the file survives a SIGKILL."""

    def __init__(self, path: Path, period_s: float) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.period_s = period_s
        self.phase = "start"
        self.t0 = time.monotonic()
        self.stop = threading.Event()
        self.peak = 0.0

    def run(self) -> None:
        with self.path.open("a", buffering=1) as fh:
            fh.write(f"# pid={os.getpid()} platform={platform.platform()}\n")
            while not self.stop.wait(self.period_s):
                r = rss_mb()
                self.peak = max(self.peak, r)
                fh.write(f"{time.monotonic() - self.t0:.2f}\t{r}\t{self.phase}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", default="B0")
    ap.add_argument("--lp", type=int, default=32)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--nl", type=int, default=0, help="override the config's layer count (growth-profile runs; 0 = native)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=INT",
                    help="override any integer config static (growth-profile runs), e.g. --set V=2048")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-gb", type=float, default=0.0, help="RLIMIT_AS (Linux only; 0 = none)")
    ap.add_argument("--sample-ms", type=int, default=500)
    ap.add_argument("--tracemalloc", action="store_true", help="attribute bridge allocations (small scales)")
    ap.add_argument("--expanded", action="store_true",
                    help="[COMPACT] use the adapter's expanded reference path (compact=False; the pre-§39 behaviour)")
    ap.add_argument("--bound", action="store_true", help="[COMPACT] after validate: hierarchical_table + bound_of")
    ap.add_argument("--samples", type=int, default=0,
                    help="[COMPACT] sampled-instance lookups after the circuit is built: N random gates (gate(i)) and "
                         "N random root call instances (declared args/returns), timed, with visits/hops")
    ap.add_argument("--skip-validate", action="store_true", help="[COMPACT] stop after the plan is built")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(out.with_suffix(out.suffix + ".rss.tsv"), args.sample_ms / 1000)
    sampler.start()

    doc: dict[str, Any] = {
        "schema": "verity-vllm/bridge-diag/v1",
        "case": args.case,
        "lp": args.lp,
        "steps": args.steps,
        "cap_gb": args.cap_gb,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "phases": {},
    }

    def dump(status: str, **extra: Any) -> None:
        doc.update(extra)
        doc["status"] = status
        doc["maxrss_mb"] = maxrss_mb()
        doc["sampler_peak_mb"] = sampler.peak
        out.write_text(json.dumps(doc, indent=1, default=str) + "\n")

    if args.cap_gb > 0 and platform.system() == "Linux":
        cap = int(args.cap_gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    dump("started")

    def phase(name: str) -> None:
        sampler.phase = name

    try:
        from verity.ir import Program, bind
        from verity_vllm.registry import b1

        from veritor.plans.boundary import Part, Parts
        from veritor.plans.plan import VerificationPlan
        from veritor.plans.validate import validate_plan
        from veritor.plans.verity_ir_adapter import program_from_verity_ir

        config = dict(getattr(b1, f"{args.case}_CONFIG"))
        if args.nl:
            config["NL"] = args.nl
            doc["nl_override"] = args.nl
        for kv in args.set:
            k, _, v = kv.partition("=")
            config[k] = int(v)
            doc.setdefault("config_overrides", {})[k] = int(v)

        phase("program")
        t = time.perf_counter()
        program = Program(bind(b1.Serve, C=config, LP=args.lp, STEPS=args.steps))
        doc["phases"]["program"] = {"seconds": round(time.perf_counter() - t, 2), "gates": program.gates,
                                    "input_gates": program.input_gates, "rss_mb": rss_mb(), "maxrss_mb": maxrss_mb()}
        dump("program built")

        if args.tracemalloc:
            import tracemalloc

            tracemalloc.start(10)

        phase("bridge")
        t = time.perf_counter()
        from veritor.plans.verity_ir_adapter import adapt

        adapted = adapt(program, compact=not args.expanded)
        Q = adapted.program
        doc["phases"]["bridge"] = {"seconds": round(time.perf_counter() - t, 2), "rss_mb": rss_mb(), "maxrss_mb": maxrss_mb(),
                                   "compact": not args.expanded, "compaction": adapted.compaction.to_json(),
                                   "unrolled": len(adapted.unrolled)}
        del adapted
        dump("bridged")

        if args.tracemalloc:
            snap = tracemalloc.take_snapshot()
            top = snap.statistics("lineno")[:25]
            doc["bridge_tracemalloc_top"] = [f"{s.traceback[0].filename}:{s.traceback[0].lineno} "
                                             f"{s.size / (1 << 20):.1f} MB ({s.count} blocks)" for s in top]
            tracemalloc.stop()
            dump("bridged (tracemalloc)")

        phase("circuit")
        t = time.perf_counter()
        c = Q.circuit()
        root = Q.root
        doc["phases"]["circuit"] = {"seconds": round(time.perf_counter() - t, 2), "rss_mb": rss_mb(), "maxrss_mb": maxrss_mb(),
                                    "root_nodes": len(root.nodes), "size": getattr(c, "size", None)}
        dump("circuit built")

        inputs = [k for k, n in enumerate(root.nodes) if n.form == "batch" and n.fn.name == "Input"]
        calls = [k for k, n in enumerate(root.nodes) if k not in inputs]

        if args.samples:
            import random

            phase("samples")
            rng = random.Random(0)
            t = time.perf_counter()
            v0, h0 = c.visits, c.hops
            gates = rng.sample(range(c.size), min(args.samples, c.size))
            ops = [(g, c.gate(g).prim.name, len(c.gate(g).operands)) for g in gates]
            t_gates = time.perf_counter() - t
            t = time.perf_counter()
            picks = rng.sample(calls, min(args.samples, len(calls)))
            iface = []
            for k in picks:
                node = root.nodes[k]
                m = rng.randrange(node.n)
                call = c.call([(k, m)])
                iface.append((k, m, call.fn.name, len(call.args), len(call.returns)))
            t_calls = time.perf_counter() - t
            doc["phases"]["samples"] = {"gates": len(ops), "gate_seconds": round(t_gates, 3), "root_calls": len(iface),
                                        "call_interface_seconds": round(t_calls, 3),
                                        "interface_leaves": sum(a + r for *_, a, r in iface),
                                        "visits": c.visits - v0, "hops": c.hops - h0, "rss_mb": rss_mb(), "maxrss_mb": maxrss_mb(),
                                        "example_gates": ops[:3], "example_calls": iface[:3]}
            dump("sampled")

        def whole(k: int) -> Parts:
            return Parts(c, [Part((), k, 0, root.nodes[k].n)])

        phase("plan")
        t = time.perf_counter()
        vus = [whole(k) for k in inputs] + [whole(k) for k in calls]
        cus = [Parts(c, [Part((), k, 0, root.nodes[k].n) for k in inputs]),
               Parts(c, [Part((), k, 0, root.nodes[k].n) for k in calls])]
        plan = VerificationPlan(f"root P1 over {Q.name}", vus, cus)
        doc["phases"]["plan"] = {"seconds": round(time.perf_counter() - t, 2), "vus": len(vus), "cus": len(cus),
                                 "rss_mb": rss_mb(), "maxrss_mb": maxrss_mb()}
        dump("plan built")
        if args.skip_validate:
            dump("plan built (validate skipped)")
            sampler.stop.set()
            return 0

        phase("validate")
        t = time.perf_counter()
        v = validate_plan(c, plan)
        doc["phases"]["validate"] = {"seconds": round(time.perf_counter() - t, 2), "rss_mb": rss_mb(), "maxrss_mb": maxrss_mb(),
                                     "visits": v.visits, "hops": v.hops, "escape_steps": v.escape_steps,
                                     "boundary_gates": sum(len(b.inputs) + len(b.outputs) for b in (*v.vu_boundaries, *v.cu_boundaries))}
        dump("valid" if v.status == "valid" else v.status, validated=True,
             verdict={"status": v.status, "codes": [str(x) for x in v.codes], "reasons": list(v.reasons)[:5],
                      "vu_edges": v.vu_edges, "cu_edges": v.cu_edges, "visits": v.visits})
        if args.bound and v.status == "valid":
            from fractions import Fraction

            from veritor.core.policy import VerificationPolicy
            from veritor.plans.bound import bound_of, hierarchical_table

            phase("bound")
            t = time.perf_counter()
            table = hierarchical_table(c, plan, v)
            b = bound_of(table.table, VerificationPolicy(q=Fraction(1, 2), s=Fraction(1, 2)), Fraction(1, 64))
            doc["phases"]["bound"] = {"seconds": round(time.perf_counter() - t, 2), "rss_mb": rss_mb(), "maxrss_mb": maxrss_mb(),
                                      "visits": table.visits, "hops": table.hops, "steps": table.steps,
                                      "vu_kinds": table.vu_kinds, "cu_kinds": table.cu_kinds, "contexts": table.extra,
                                      "bits": b.bits, "digest": table.table.digest}
            dump("bounded")
        sampler.stop.set()
        return 0 if v.status == "valid" else 1
    except MemoryError:
        dump("unsupported: MemoryError", error="MemoryError", traceback=traceback.format_exc()[-2000:])
        sampler.stop.set()
        return 3
    except Exception as e:  # noqa: BLE001
        dump(f"error: {type(e).__name__}", error=f"{type(e).__name__}: {e}"[:400], traceback=traceback.format_exc()[-2000:])
        sampler.stop.set()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

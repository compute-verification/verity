#!/usr/bin/env python3
"""[SUMMARY-C] Compact (run-aware) vs dense canonical plans through plan -> validate -> kind table -> Summary,
with wall time and RSS per stage, at laptop shapes (TINY, small B0 slices); the whole-B0 commands are the
phase-2 list in ``docs/vllm-poc/reports/SUMMARY-C-2026-09-09.md``.

    python scripts/summary_probe.py --case TINY --plans P1/cu=operator,P3/cu=whole --out /tmp/probe/tiny
    python scripts/summary_probe.py --case B0 --nl 1 --lp 4 --steps 2 --plans P3/cu=whole --modes compact --out ...

Per plan and mode (``dense`` = one ``Parts`` per unit, ``compact`` = ``UnitRuns``): ``plan`` (construction),
``validate``, ``table`` (``hierarchical_table``), ``summarize`` (the ``Summary`` from the verdict + table, cached to
``--cache``), ``evaluate`` (``bound_summary`` at ``--policies`` points from the summary alone).  When both modes run,
the kind-table digests, the summaries' populations and the bounds are compared (must be equal).  Every number is
MEASURED on the machine named in the record.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from compact_whole_b0 import (  # noqa: E402
    build_config,
    build_program,
    ir_sizes,
    maxrss_mb,
    rss_mb,
    unrolled_runs,
)

SCHEMA = "veritor/summary-probe/v1"
TINY_CONFIG = {
    "model": "tiny",
    "revision": None,
    "H": 32,
    "NL": 2,
    "NH": 2,
    "KVH": 1,
    "D": 16,
    "I": 32,
    "V": 64,
    "QKV": 64,
    "GU": 64,
    "EPS": 1e-5,
    "ROPE_THETA": 1e4,
    "MAX_POS": 16,
    "BIAS": True,
    "tie_word_embeddings": True,
    "profile": {},
}


def stage(rows: list[dict[str, Any]], name: str, fn, **inputs: Any) -> Any:
    gc.collect()
    before = rss_mb()
    t = time.perf_counter()
    result = fn()
    row = {
        "stage": name,
        "inputs": inputs,
        "seconds": round(time.perf_counter() - t, 3),
        "rss_before_mb": before,
        "rss_after_mb": rss_mb(),
        "maxrss_mb": maxrss_mb(),
    }
    rows.append(row)
    print(
        f"  {name:32s} {row['seconds']:9.3f}s  rss {before:7.1f} -> {row['rss_after_mb']:7.1f} MB (max {row['maxrss_mb']:.0f})  {inputs}",
        flush=True,
    )
    return result


def make_plan(spec: str, circuit, classifier, compact: bool):
    from veritor.plans.canonical import PLAN_NAMES, canonical_plan

    head, _, cu = spec.partition("/cu=")
    cp = canonical_plan(circuit, PLAN_NAMES[head], cu or "whole", classifier, compact=compact)
    return cp.plan, {
        "vu_level": cp.vu_level,
        "cu_strategy": cp.cu_strategy,
        "constructor": "veritor.plans.canonical",
    }


def run_mode(
    spec: str,
    mode: str,
    circuit,
    classifier,
    program_digest: tuple[str, str],
    policies,
    cache: Path | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from veritor.plans.bound import hierarchical_table
    from veritor.plans.runs import UnitRuns
    from veritor.plans.summary import bound_summary, cached_summary, plan_digest, summarize
    from veritor.plans.validate import validate_plan

    tag = f"{spec}[{mode}]"
    plan, meta = stage(
        rows, f"plan {tag}", lambda: make_plan(spec, circuit, classifier, mode == "compact")
    )
    out: dict[str, Any] = {
        "plan": plan.name,
        "mode": mode,
        "vus": len(plan.verification_units),
        "cus": len(plan.commitment_units),
        **meta,
    }
    if isinstance(plan.verification_units, UnitRuns):
        out["vu_runs"] = len(plan.verification_units.runs)
    if isinstance(plan.commitment_units, UnitRuns):
        out["cu_runs"] = len(plan.commitment_units.runs)
    out["plan_digest"] = dict(zip(("kind", "digest"), map(str, plan_digest(plan)), strict=True))
    verdict = stage(
        rows,
        f"validate {tag}",
        lambda: validate_plan(circuit, plan),
        vus=out["vus"],
        cus=out["cus"],
    )
    out.update(
        status=verdict.status,
        codes=list(verdict.codes),
        vu_edges=verdict.vu_edges,
        cu_edges=verdict.cu_edges,
        escape_steps=verdict.escape_steps,
        reasons=[list(r) for r in verdict.reasons[:3]],
    )
    if verdict.status != "valid":
        return out
    result = stage(
        rows, f"table {tag}", lambda: hierarchical_table(circuit, plan, verdict), vus=out["vus"]
    )
    out["table_digest"] = result.table.digest
    out["rows"] = len(result.table.rows)

    def do_summary():
        if cache is None:
            return summarize(
                circuit, plan, verdict, program_digest=program_digest, plan_meta=meta, result=result
            ), None
        return cached_summary(
            cache, circuit, plan, program_digest=program_digest, plan_meta=meta, verdict=verdict, result=result
        )

    summary, hit = stage(rows, f"summarize {tag}", do_summary, cached=cache is not None)
    out["summary"] = {
        "digest": summary.digest,
        "cache_hit": hit,
        "N_V": summary.N_V.to_json(),
        "N_R": summary.N_R.to_json(),
        "v_max": summary.v_max,
        "W_out": summary.W_out,
        "vu_kinds": len(summary.vu_kinds),
        "ru_kinds": len(summary.ru_kinds),
        "M_t": {str(t): summary.M_t(t).value for t in summary.thresholds},
        "boundary": summary.boundary.to_json(),
        "json_bytes": len(json.dumps(summary.to_json())),
    }

    def do_eval():
        table = summary.table()
        return [
            {
                "q": str(q),
                "s": str(s),
                "delta": str(d),
                "U_bits": float(b.bits),
                "capped": bool(b.capped),
            }
            for (q, s, d) in policies
            for b in (bound_summary(summary, _policy(q, s), d, table=table),)
        ]

    out["bounds"] = stage(rows, f"evaluate {tag}", do_eval, points=len(policies))
    return out


def _policy(q: Fraction, s: Fraction):
    from veritor.core.policy import VerificationPolicy

    return VerificationPolicy(q=q, s=s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--case", default="TINY", help="TINY, or a verity_vllm.registry.b1 *_CONFIG name (B0, B1)"
    )
    ap.add_argument("--lp", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--nl", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=INT")
    ap.add_argument("--plans", default="P1/cu=operator,P3/cu=whole")
    ap.add_argument("--modes", default="dense,compact")
    ap.add_argument("--policies", default="1/2:1/2:1/64,1:1/4096:1/1000000,1/2:1/4096:1/1000000")
    ap.add_argument("--cache", default="", help="summary cache directory (empty: no cache)")
    ap.add_argument("--out", required=True, help="output prefix (<out>.json)")
    ap.add_argument("--recursion-limit", type=int, default=100_000)
    args = ap.parse_args(argv)
    sys.setrecursionlimit(args.recursion_limit)

    from veritor.plans.canonical import Classifier
    from veritor.plans.verity_ir_adapter import adapt

    rows: list[dict[str, Any]] = []
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "label": "MEASURED",
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "args": vars(args),
        "stages": rows,
    }
    if args.case == "TINY":
        config = dict(TINY_CONFIG)
        for kv in args.set:
            k, _, v = kv.partition("=")
            config[k] = int(v)
        if args.nl:
            config["NL"] = args.nl
    else:
        config, _ = build_config(args.case, args.nl, args.set)
    doc["config"] = {k: v for k, v in config.items() if k != "profile"}
    program_row: dict[str, Any] = {}
    program = stage(
        rows,
        "program",
        lambda: build_program(config, args.lp, args.steps, True, program_row),
        case=args.case,
        LP=args.lp,
        STEPS=args.steps,
    )
    doc["program"] = program_row
    adapted = stage(rows, "adapt", lambda: adapt(program, compact=True), gates=program.gates)
    doc["ir"] = ir_sizes(adapted.program)
    circuit = stage(
        rows,
        "circuit",
        lambda: adapted.program.circuit(),
        root_nodes=len(adapted.program.root.nodes),
    )
    doc["circuit_size"] = circuit.size
    classifier = Classifier(unrolled_runs(adapted))
    program_digest = ("verity-ir/program_digest", program_row["digest"])
    policies = [tuple(Fraction(x) for x in p.split(":")) for p in args.policies.split(",")]
    cache = Path(args.cache) if args.cache else None

    results: dict[str, Any] = {}
    doc["plans"] = results
    for spec in args.plans.split(","):
        per: dict[str, Any] = {}
        results[spec] = per
        for mode in args.modes.split(","):
            print(f"[{spec}] mode={mode}", flush=True)
            per[mode] = run_mode(
                spec, mode, circuit, classifier, program_digest, policies, cache, rows
            )
            gc.collect()
        if (
            "dense" in per
            and "compact" in per
            and per["dense"].get("status") == "valid" == per["compact"].get("status")
        ):
            d, c = per["dense"], per["compact"]
            per["agree"] = {
                "table_digest": d["table_digest"] == c["table_digest"],
                "populations": {
                    k: d["summary"][k] == c["summary"][k]
                    for k in ("N_V", "N_R", "v_max", "W_out", "M_t", "boundary")
                },
                "bounds": d["bounds"] == c["bounds"],
                "plan_digest_differs_by_kind": d["plan_digest"]["kind"] != c["plan_digest"]["kind"],
            }
            print(f"[{spec}] agree: {per['agree']}", flush=True)
    doc["maxrss_mb"] = maxrss_mb()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"wrote {out.with_suffix('.json')}  maxrss {doc['maxrss_mb']} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

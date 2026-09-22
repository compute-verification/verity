#!/usr/bin/env python3
"""[COMPACT-P1] Compact construction + STREAMING validation of a canonical plan (default ``P1/cu=whole``) of
``Serve_v1{C,LP,STEPS}``: the partition digest, unit counts per kind (vs the Table-3a census when given), wall
time and peak RSS per stage -- one JSON record.  Nothing per unit is kept: ``validate_plan(..., retain=False)``
folds every boundary into totals and counts quotient edges exactly (``veritor.plans.validate``); the plan itself
is ``UnitRuns`` (``canonical_plan(..., compact=True)``).

    python scripts/compact_p1_validate.py --case TINY --out /tmp/p1-tiny.json
    python scripts/compact_p1_validate.py --case B0 --nl 1 --set V=1024 --lp 2 --steps 1 --out /tmp/p1-slice.json
    python scripts/compact_p1_validate.py --case B0 --lp 32 --steps 15 --workers 16 --census docs/data/table3a/B0-serve-lp32-s15.json --out ...

``--workers N`` shards the verification side into N index ranges validated in forked worker processes (the
circuit and plan are shared copy-on-write; each worker returns its ``BoundaryStats`` + edge count + reasons);
the partition, refinement and commitment side are checked in the parent.  The merged verdict is exactly the
single-process one (tests: ``tests/veritor/plans/test_streaming_validate.py``).  Every number is MEASURED on
the machine named in the record unless labelled otherwise.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from compact_whole_b0 import build_config, build_program, cpu_model, maxrss_mb, rss_mb, unrolled_runs  # noqa: E402

SCHEMA = "veritor/compact-p1-validation/v1"
TINY_CONFIG = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64, "QKV": 64,
               "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True, "tie_word_embeddings": True, "profile": {}}


def stage(rows: list[dict[str, Any]], name: str, fn, **inputs: Any) -> Any:
    gc.collect()
    before = rss_mb()
    t = time.perf_counter()
    result = fn()
    row = {"stage": name, "inputs": inputs, "seconds": round(time.perf_counter() - t, 3), "rss_before_mb": before,
           "rss_after_mb": rss_mb(), "maxrss_mb": maxrss_mb()}
    rows.append(row)
    print(f"  {name:28s} {row['seconds']:10.3f}s  rss {before:8.1f} -> {row['rss_after_mb']:8.1f} MB (max {row['maxrss_mb']:.0f})  {inputs}", flush=True)
    return result


def kind_counts(plan, circuit) -> dict[str, int]:
    """Units per ``(callee name, node form)`` of the stopped nodes, ``O(runs)`` -- the census's ``(fn, form)`` key.
    An unrolled run counts under the form the adapter unrolled (``batch``); a scan member counts as ``scan``."""

    from veritor.plans.runs import UnitRuns

    vus = plan.verification_units
    counts: Counter[str] = Counter()
    if isinstance(vus, UnitRuns):
        for run in vus.runs:
            scope = circuit.scope_at(run.path)
            node = scope.fn.nodes[run.node_idx]
            form = "batch" if run.unrolled else node.form
            counts[f"{node.fn.name}|{form}"] += run.count
    else:
        for unit in vus:
            part = unit.parts[0]
            scope = circuit.scope_at(part.path)
            node = scope.fn.nodes[part.node_idx]
            counts[f"{node.fn.name}|{node.form}"] += 1
    return dict(sorted(counts.items()))


_VERSION = re.compile(r"_v\d+")
_DOT_PARAM = re.compile(r",?DOT=[^,}]*")


def kind_key(name_form: str) -> str:
    """The census and the plan IR name one definition differently (``GemmCoordinate_v1{K=576}`` in the plan IR vs the
    registry id ``GemmCoordinate_v2{K=576,DOT=AmpereBF16TcDot16_v1}`` in the census; ``Const32[0x0]_v1`` vs
    ``Const32[0x0]``): compare on the version-free name with the ``DOT`` parameter dropped, plus the node form."""

    return _DOT_PARAM.sub("", _VERSION.sub("", name_form)).replace("{}", "")


def census_compare(counts: dict[str, int], census_path: Path) -> dict[str, Any]:
    census = json.loads(census_path.read_text())
    rows: dict[str, int] = {}
    for r in census["rows"]:
        for rel in r["relations"]:
            k = kind_key(f"{rel['fn']}|{rel['form']}")
            rows[k] = rows.get(k, 0) + int(rel["count"])
    plan_non_input: dict[str, int] = {}
    for k, v in counts.items():
        if not k.startswith("Input"):
            kk = kind_key(k)
            plan_non_input[kk] = plan_non_input.get(kk, 0) + v
    mism = {k: {"plan": plan_non_input.get(k), "census": rows.get(k)} for k in sorted(set(rows) | set(plan_non_input))
            if plan_non_input.get(k) != rows.get(k)}
    return {"census": str(census_path), "census_N_V": census.get("N_V"), "plan_non_input_units": sum(plan_non_input.values()),
            "plan_input_units": sum(v for k, v in counts.items() if k.startswith("Input")),
            "kinds_census": len(rows), "kinds_plan_non_input": len(plan_non_input), "key": "kind_key(fn|form): version suffix and DOT parameter dropped",
            "per_kind": {k: {"plan": plan_non_input.get(k), "census": rows.get(k)} for k in sorted(set(rows) | set(plan_non_input))},
            "mismatches": mism, "agree": not mism and sum(plan_non_input.values()) == census.get("N_V")}


_SHARED: dict[str, Any] = {}


def _worker(args: tuple[int, int, int]) -> dict[str, Any]:
    from veritor.plans.validate import validate_plan

    w, lo, hi = args
    circuit, plan = _SHARED["circuit"], _SHARED["plan"]
    t = time.perf_counter()
    last = [t]

    def progress(done: int, total: int) -> None:
        now = time.perf_counter()
        if now - last[0] >= 60:
            last[0] = now
            print(f"    worker {w}: {done}/{total} VUs  {now - t:.0f}s  rss {rss_mb():.0f} MB", flush=True)

    kwargs = dict(_SHARED.get("validate_kwargs", {}))
    widths = None
    if _SHARED.get("widths"):  # [A1, contract R11] per-kind interface widths observed on the validator's own boundaries
        from e2e.population_v3 import WidthCensus

        widths = WidthCensus(plan, circuit)
        kwargs["on_boundary"] = widths.on_boundary
    v = validate_plan(circuit, plan, retain=False, vu_range=(lo, hi), progress=progress, **kwargs)
    part = {"worker": w, "range": [lo, hi], "status": v.status, "reasons": [list(r) for r in v.reasons[:5]], "vu_edges": v.vu_edges,
            "escape_steps": v.escape_steps, "cu_escape_steps": v.cu_escape_steps, "vu_stats": v.vu_stats.to_json(), "cu_stats": v.cu_stats.to_json(),
            "cu_edges": v.cu_edges, "seconds": round(time.perf_counter() - t, 3), "maxrss_mb": maxrss_mb(), "pid": os.getpid()}
    if widths is not None:
        part["widths"] = widths.to_json()
    return part


INPUT_VU_WEIGHT, RELATION_VU_WEIGHT = 1, 45
"""Relative streaming cost of an ``Input`` coordinate VU (no inputs; the boundary is one output) and a relation VU
(a full escape analysis), MEASURED on the B0 pod run: ~78 k VUs/s vs ~1.7 k VUs/s per worker."""


def weighted_cuts(plan, circuit, chunks: int) -> list[int]:
    """``chunks + 1`` VU indices cutting the verification side into ranges of about equal *work* (``O(runs)``): the
    135 M weight-``Input`` VUs of B0 are trivial and sit in front of the 31 M relation VUs, so equal-count shards
    leave most workers idle while one walks the relations."""

    from veritor.plans.runs import UnitRuns

    vus = plan.verification_units
    n = len(vus)
    if chunks <= 1 or not isinstance(vus, UnitRuns):
        return [n * i // chunks for i in range(chunks + 1)]
    weighted: list[tuple[int, int, int]] = []  # (first index, count, weight per VU)
    total = start = 0
    for run in vus.runs:
        node = circuit.scope_at(run.path).fn.nodes[run.node_idx]
        w = INPUT_VU_WEIGHT if node.fn.name.startswith("Input") else RELATION_VU_WEIGHT
        weighted.append((start, run.count, w))
        total += run.count * w
        start += run.count
    cuts = [0]
    acc = 0
    for s, c, w in weighted:
        while len(cuts) < chunks and acc + c * w >= len(cuts) * total / chunks:
            need = (len(cuts) * total / chunks - acc) / w
            cuts.append(max(cuts[-1], s + min(c, max(0, int(need)))))
        acc += c * w
    while len(cuts) < chunks:
        cuts.append(n)
    cuts.append(n)
    return cuts


def sharded_validate(circuit, plan, workers: int, rows: list[dict[str, Any]], chunks: int | None = None, reverse: bool = False,
                     parts_path: Path | None = None, ranges: list[list[int]] | None = None) -> dict[str, Any]:
    """Every shard validates the partition, refinement and commitment side itself (``O(runs)``; the CU boundary is
    recomputed per shard -- identical by construction) and streams its VU range; the parent merges.  ``chunks``
    (default ``workers``) work-balanced ranges (:func:`weighted_cuts`) are handed to the ``workers`` forked
    processes dynamically, so a slow range does not idle the others.  ``reverse`` hands them out last-first (a
    second machine working towards the first); each finished part is appended to ``parts_path`` (JSON lines) so
    :mod:`compact_p1_merge` can combine the parts of several machines."""

    import gc

    n = len(plan.verification_units)
    if ranges is not None:  # explicit ranges, in the order to hand them out (a machine taking a chosen share of the work)
        tasks = [(i, int(lo), int(hi)) for i, (lo, hi) in enumerate(ranges)]
        bounds = None
        chunks = len(tasks)
        print(f"  shards ({chunks} explicit ranges over {workers} workers): {ranges}", flush=True)
    else:
        chunks = chunks or workers
        bounds = weighted_cuts(plan, circuit, chunks)
        print(f"  shards ({chunks} work-balanced ranges over {workers} workers{', reversed' if reverse else ''}): {bounds}", flush=True)
        order = list(range(chunks))[::-1] if reverse else list(range(chunks))
        tasks = [(w, bounds[w], bounds[w + 1]) for w in order]
    _SHARED["circuit"], _SHARED["plan"] = circuit, plan
    ctx = mp.get_context("fork")
    t = time.perf_counter()
    gc.collect()
    gc.freeze()  # the forked workers never garbage-collect (and so never copy-on-write) the parent's plan and circuit
    parts: list[dict[str, Any]] = []
    with ctx.Pool(workers) as pool:
        for part in pool.imap_unordered(_worker, tasks, chunksize=1):
            parts.append(part)
            print(f"    part {part['worker']} [{part['range'][0]}, {part['range'][1]}) {part['status']} in {part['seconds']:.0f}s  "
                  f"({len(parts)}/{chunks} done, {time.perf_counter() - t:.0f}s)", flush=True)
            if parts_path is not None:
                with parts_path.open("a") as f:
                    f.write(json.dumps({"bounds": bounds, "part": part}, default=str) + "\n")
    gc.unfreeze()
    return merge_parts(parts, bounds, workers, chunks, n, time.perf_counter() - t, rows)


def coverage_gaps(tiles: list[tuple[int, int]], n: int) -> list[list[int]]:
    """``[lo, hi]`` ranges of ``[0, n)`` that the sorted ``tiles`` leave uncovered, and (negative-length) overlaps."""

    gaps: list[list[int]] = []
    pos = 0
    for lo, hi in sorted(tiles):
        if lo != pos:
            gaps.append([pos, lo])  # lo < pos: an overlap
        pos = max(pos, hi)
    if pos != n:
        gaps.append([pos, n])
    return gaps


def merge_parts(parts: list[dict[str, Any]], bounds: list[int] | None, workers: int, chunks: int, n: int, seconds: float,
                rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The verdict of the whole verification side from its per-range parts: statuses and reasons collected, VU totals
    summed, the commitment side (identical in every part) taken once; ``rejected`` unless the ranges tile ``[0, n)``."""

    from veritor.plans.validate import BoundaryStats

    parts = sorted(parts, key=lambda p: p["range"][0])
    merged_vu, merged_cu = BoundaryStats(), BoundaryStats()
    for p in parts:
        for k, v in p["vu_stats"].items():
            setattr(merged_vu, k, getattr(merged_vu, k) + v)
    cu = parts[0]["cu_stats"]
    for k, v in cu.items():
        setattr(merged_cu, k, v)
    statuses = {p["status"] for p in parts}
    reasons = [r for p in parts for r in p["reasons"]]
    out = {"mode": f"sharded x{workers} (fork), {chunks} work-balanced ranges", "bounds": bounds,
           "status": "valid" if statuses == {"valid"} else ("unsupported" if "unsupported" in statuses else "rejected"),
           "reasons": reasons[:10], "vu_edges": sum(p["vu_edges"] for p in parts), "cu_edges": parts[0]["cu_edges"],
           "escape_steps": sum(p["escape_steps"] - p["cu_escape_steps"] for p in parts) + parts[0]["cu_escape_steps"],
           "cu_escape_steps": parts[0]["cu_escape_steps"],
           "vu_stats": merged_vu.to_json(), "cu_stats": merged_cu.to_json(), "seconds": round(seconds, 3),
           "worker_seconds_sum": round(sum(p["seconds"] for p in parts), 3),
           "workers": parts, "worker_maxrss_mb": max(p["maxrss_mb"] for p in parts), "worker_maxrss_sum_mb": sum(p["maxrss_mb"] for p in parts)}
    tiles = [tuple(p["range"]) for p in parts]
    gaps = coverage_gaps(tiles, n)
    if gaps or merged_vu.units != n:
        out["status"] = "rejected"
        out["reasons"].append(["uncovered", f"parts do not tile [0, {n}) once: gaps/overlaps {gaps[:8]}; {merged_vu.units} of {n} verification units validated"])
        out["gaps"] = gaps
    if any(p["cu_stats"] != cu or p["cu_edges"] != out["cu_edges"] or p["cu_escape_steps"] != out["cu_escape_steps"] for p in parts):
        out["status"] = "rejected"
        out["reasons"].append(["cu-mismatch", "the parts disagree on the commitment side"])
    if all("widths" in p for p in parts):  # [A1, R11] the exhaustive per-kind widths, merged exactly
        from e2e.population_v3 import WIDTHS_SCHEMA, merge_widths

        kinds = merge_widths([p["widths"] for p in parts])
        wu = sum(p["widths"]["units"] for p in parts)
        out["widths"] = {"schema": WIDTHS_SCHEMA, "units": wu, "kinds": kinds, "complete": wu == n and not gaps,
                         "label": "MEASURED (exhaustive: every verification unit's boundary as computed by validate_plan; observer on_boundary)"}
    if rows is not None:
        rows.append({"stage": "validate[sharded]", "inputs": {"workers": workers, "vus": n}, "seconds": out["seconds"], "rss_before_mb": None,
                     "rss_after_mb": rss_mb(), "maxrss_mb": maxrss_mb(), "worker_maxrss_mb": out["worker_maxrss_mb"]})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", default="TINY", help="TINY, or a verity_vllm.registry.b1 *_CONFIG name (B0, B1)")
    ap.add_argument("--lp", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--nl", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=INT")
    ap.add_argument("--plan", default="P1/cu=whole")
    ap.add_argument("--inputs", default="units", choices=("units", "prescribed"),
                    help="A1 (contract R2): 'prescribed' leaves the root Input batches out of every VU and CU (the population is the computed "
                         "gates; the intervals are validated as prescribed_outside on both sides); 'units' is the historical construction")
    ap.add_argument("--widths", action="store_true",
                    help="A1 (contract R11): record every kind's actual interface widths (histogram of |S.out| / out bits per kind) from the "
                         "boundaries the validator computes -- validation['widths'] (veritor/a1/kind-widths-measured/v1 table)")
    ap.add_argument("--workers", type=int, default=1, help="> 1: shard the verification side over forked workers")
    ap.add_argument("--chunks", type=int, default=0, help="work-balanced VU ranges to hand to the workers (default: --workers)")
    ap.add_argument("--reverse", action="store_true", help="hand the ranges out last-first (a second machine meeting the first)")
    ap.add_argument("--skip-plan-digest", action="store_true", help="skip the O(runs)-memory plan-rule digest (small machines)")
    ap.add_argument("--ranges", default="", help="JSON file: explicit [lo, hi] VU ranges to validate, in hand-out order (with --workers)")
    ap.add_argument("--census", default="", help="Table-3a census JSON to compare per-kind unit counts against")
    ap.add_argument("--no-validate", action="store_true", help="plan, digest and counts only (no validation)")
    ap.add_argument("--recursion-limit", type=int, default=100_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sys.setrecursionlimit(args.recursion_limit)

    from veritor.plans.canonical import PLAN_NAMES, Classifier, canonical_plan
    from veritor.plans.summary import PARTITION_DIGEST, partition_digest, plan_digest
    from veritor.plans.verity_ir_adapter import adapt

    rows: list[dict[str, Any]] = []
    doc: dict[str, Any] = {"schema": SCHEMA, "label": "MEASURED", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "host": platform.node(), "platform": platform.platform(), "cpu": cpu_model(), "cpus": os.cpu_count(),
                           "python": sys.version.split()[0], "args": vars(args), "stages": rows}
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
    program = stage(rows, "program", lambda: build_program(config, args.lp, args.steps, True, program_row), case=args.case, LP=args.lp, STEPS=args.steps)
    doc["program"] = program_row
    adapted = stage(rows, "adapt", lambda: adapt(program, compact=True), gates=program.gates)
    circuit = stage(rows, "circuit", lambda: adapted.program.circuit(), root_nodes=len(adapted.program.root.nodes))
    classifier = Classifier(unrolled_runs(adapted))
    head, _, cu = args.plan.partition("/cu=")
    from veritor.plans.canonical import PRESCRIBED_SUFFIX

    plan_label = args.plan + (PRESCRIBED_SUFFIX if args.inputs == "prescribed" else "")
    cp = stage(rows, "plan[compact]", lambda: canonical_plan(circuit, PLAN_NAMES[head], cu or "whole", classifier, name=plan_label, compact=True,
                                                            inputs=args.inputs), inputs=args.inputs)
    plan = cp.plan
    validate_kwargs: dict[str, Any] = {}
    if args.inputs == "prescribed":
        validate_kwargs = {"prescribed_outside": cp.prescribed_outside, "prescribed_outside_sides": ("verification", "commitment")}
    _SHARED["validate_kwargs"] = validate_kwargs
    _SHARED["widths"] = bool(args.widths)
    from veritor.plans.runs import UnitRuns

    pinfo: dict[str, Any] = {"name": plan.name, "vu_level": cp.vu_level, "cu_strategy": cp.cu_strategy, "constructor": "veritor.plans.canonical(compact=True)",
                             "inputs": cp.inputs, "prescribed_outside": [list(iv) for iv in cp.prescribed_outside],
                             "prescribed_outside_gates": cp.prescribed_outside_gates,
                             "vus": len(plan.verification_units), "cus": len(plan.commitment_units),
                             "vu_runs": len(plan.verification_units.runs) if isinstance(plan.verification_units, UnitRuns) else None,
                             "cu_runs": len(plan.commitment_units.runs) if isinstance(plan.commitment_units, UnitRuns) else None}
    # plan-rule/v1 digest: canonical JSON of the whole plan document, ``O(runs)`` transient memory (9 GB on whole B0)
    if not args.skip_plan_digest:
        pinfo["plan_digest"] = dict(zip(("kind", "digest"), map(str, stage(rows, "plan_digest", lambda: plan_digest(plan), runs=pinfo["vu_runs"])), strict=True))
    pinfo["partition_digest"] = stage(rows, "partition_digest", lambda: partition_digest(plan), vus=pinfo["vus"])
    pinfo["partition_digest_kind"] = PARTITION_DIGEST
    counts = stage(rows, "kind_counts", lambda: kind_counts(plan, circuit), runs=pinfo["vu_runs"])
    pinfo["kind_counts"] = counts
    pinfo["input_units"] = sum(v for k, v in counts.items() if k.startswith("Input"))
    pinfo["non_input_units"] = pinfo["vus"] - pinfo["input_units"]
    if args.census:
        pinfo["census"] = census_compare(counts, Path(args.census))
        print(f"  census agree: {pinfo['census']['agree']}  plan non-input {pinfo['census']['plan_non_input_units']} vs census N_V {pinfo['census']['census_N_V']}; "
              f"mismatches {len(pinfo['census']['mismatches'])}", flush=True)
    doc["plan"] = pinfo
    print(f"  partition_digest {pinfo['partition_digest']}  ({pinfo['vus']} VUs in {pinfo['vu_runs']} runs; {pinfo['cus']} CUs)", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # the plan-side record before validation (a snapshot compact_p1_merge combines with the parts of several machines)
    out.with_suffix(".plan.json").write_text(json.dumps({**doc, "maxrss_mb": maxrss_mb()}, indent=1, default=str) + "\n")

    if not args.no_validate:
        from veritor.plans.validate import validate_plan

        if args.ranges and args.workers == 1:  # the given ranges one after another in this process (a laptop taking one range)
            _SHARED["circuit"], _SHARED["plan"] = circuit, plan
            t0 = time.perf_counter()
            parts = []
            for i, (lo, hi) in enumerate(json.loads(Path(args.ranges).read_text())):
                parts.append(_worker((i, lo, hi)))
                print(f"    part {i} [{lo}, {hi}) {parts[-1]['status']} in {parts[-1]['seconds']:.0f}s", flush=True)
                with out.with_suffix(".parts.jsonl").open("a") as f:
                    f.write(json.dumps({"bounds": None, "part": parts[-1]}, default=str) + "\n")
            val = merge_parts(parts, None, 1, len(parts), pinfo["vus"], time.perf_counter() - t0, rows)
        elif args.workers > 1:
            val = sharded_validate(circuit, plan, args.workers, rows, chunks=args.chunks or None, reverse=args.reverse,
                                   parts_path=out.with_suffix(".parts.jsonl"),
                                   ranges=json.loads(Path(args.ranges).read_text()) if args.ranges else None)
        else:
            last = [time.perf_counter()]

            def progress(done: int, total: int) -> None:
                now = time.perf_counter()
                if now - last[0] >= 60:
                    last[0] = now
                    print(f"    {done}/{total} VUs  rss {rss_mb():.0f} MB", flush=True)

            widths = None
            if args.widths:
                from e2e.population_v3 import WIDTHS_SCHEMA, WidthCensus, merge_widths

                widths = WidthCensus(plan, circuit)
                validate_kwargs["on_boundary"] = widths.on_boundary
            v = stage(rows, "validate[stream]", lambda: validate_plan(circuit, plan, retain=False, progress=progress, **validate_kwargs),
                      vus=pinfo["vus"], cus=pinfo["cus"])
            val = {"mode": "single process, retain=False", **v.to_json()}
            if widths is not None:
                val["widths"] = {"schema": WIDTHS_SCHEMA, "units": widths.units, "kinds": merge_widths([widths.to_json()]), "complete": widths.units == pinfo["vus"],
                                 "label": "MEASURED (exhaustive: every verification unit's boundary as computed by validate_plan; observer on_boundary)"}
        val["validated"] = val["status"] == "valid"
        doc["validation"] = val
        print(f"  validation: {val['status']}  vu_edges={val['vu_edges']} cu_edges={val['cu_edges']}  vu_stats={val['vu_stats']}  cu_stats={val['cu_stats']}", flush=True)
    doc["maxrss_mb"] = maxrss_mb()
    doc["seconds_total"] = round(sum(r["seconds"] for r in rows), 3)
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"wrote {out}  maxrss {doc['maxrss_mb']} MB  total {doc['seconds_total']} s", flush=True)
    return 0 if args.no_validate or doc.get("validation", {}).get("validated") else 1


if __name__ == "__main__":
    raise SystemExit(main())

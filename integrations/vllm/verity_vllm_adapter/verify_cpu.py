"""CPU-only structural verifier (WB2, 2026-09-07) -- what can be checked from an archive without a GPU.

Runs the checks of ``verity_vllm_adapter.verify`` that need neither the runtime nor the checkpoint and
writes a Table 1 / Table 3 row whose verdict is **at most PARTIAL** (no opening is authenticated against a
relation, no relation is exercised, no challenge is drawn):

1. capture format + description structure (``FAIL structural: description validation: …`` on WA's M8b);
2. every committed tree recomputed from the archive -- replay-unit trees and boundary trees from their
   stored bytes / boundary.json; the weights tree from the index's recorded leaves (the checkpoint is not
   read: *not* the model binding);
3. first-stage completeness (every boundary and every step committed);
4. output binding (C3c): boundary roots recomputed from ``production.jsonl``, with the chosen-logprob status
   (``bound`` / ``unbound``) that the committed leaf set determines; a falsified logprob in a bound bundle is
   ``FAIL structural: output binding: …`` (WA's M2 on the derived bundle);
5. the CPU-only binding negatives (N17 dropped request, N19 falsified logprob; N15/N16 need the checkpoint
   and are reported ``inapplicable`` here).

The row's ``verification_evidence`` is *derived bundle (CPU-only structural checks; no challenge)* when the
bundle carries ``DERIVED.json`` (``verity_vllm_adapter.derive_logprobs``), else *CPU-only structural checks
(no challenge)*.  Never a substitute for a verified row; exists so the bound capture format and its negatives
can be exercised from the archive alone before a GPU lane re-captures.

Usage::

    python -m verity_vllm_adapter.verify_cpu <commit dir> --production <dir> --out <dir> [--case B0] [--worker WB2]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from verity_vllm import commit as vcommit
from verity_vllm import firststage, rows
from verity_vllm.capture import CaptureError, CaptureReader, WEIGHTS_TREE_ID
from verity_vllm.commit import TreeCache
from verity_vllm_adapter import binding as vbinding
from verity_vllm_adapter import profile as prof
from verity_vllm_adapter import verify_bindings as VB
from verity_vllm_adapter.derive_logprobs import DERIVED_FILE
from verity_vllm_adapter.verify import load_description

WORKER = os.environ.get("VERITY_WORKER", "WB2")
CPU_NEGATIVES = ("N17", "N19")


def recompute_trees(reader: CaptureReader, commitments: dict[str, Any]) -> dict[str, Any]:
    """Rebuild every committed tree from the archive and compare roots / leaf counts / ids digests."""
    cache = TreeCache(reader, recompute_leaves=True)
    out: dict[str, Any] = {"trees": 0, "matched": 0, "mismatched": [], "weights": None}
    for tid, rec in commitments["trees"].items():
        out["trees"] += 1
        try:
            tree = cache.tree(tid)
        except (vcommit.CommitError, CaptureError) as e:
            out["mismatched"].append(f"{tid}: {e}")
            continue
        same = tree.root.hex() == rec["root"] and tree.leaf_count == rec["leaf_count"] and tree.ids_digest.hex() == rec["ids_digest"]
        if same:
            out["matched"] += 1
        else:
            out["mismatched"].append(f"{tid}: root {tree.root.hex()[:12]}… vs committed {rec['root'][:12]}…, leaves {tree.leaf_count} vs {rec['leaf_count']}")
        if tid == WEIGHTS_TREE_ID:
            stored = (reader.run_dir / "weights" / "values.bin").exists()
            out["weights"] = ("leaves rehashed from the stored values.bin" if stored
                              else "index's recorded leaf hashes (values referenced by checkpoint sha; the checkpoint is not read here)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--production", required=True, help="dir holding production.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--case", default=None)
    ap.add_argument("--worker", default=WORKER)
    ap.add_argument("--adapter-revision", default="unknown")
    ap.add_argument("--negatives", default=",".join(CPU_NEGATIVES))
    ap.add_argument("--negative-limit", type=int, default=4)
    ap.add_argument("--semantic-profile", default=str(vbinding.DEFAULT_SEMANTIC_PROFILE))
    ap.add_argument("--experiment-id", default=None, help="row id (default <case>-<profile_id>-<utc>; a derived bundle should name its source)")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "verify-cpu"
    out.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    try:
        reader = CaptureReader(run_dir)
    except CaptureError as e:
        raise VB.StructuralFailure("capture format", str(e)) from e
    description = load_description(reader, known_relations=())  # structure; relation ids need the runtime registry
    try:
        commitments = vcommit.load_commitments(reader)
    except vcommit.CommitError as e:
        raise VB.StructuralFailure("commitments", str(e)) from e
    manifest = reader.manifest
    profile_id = reader.profile_id
    case = args.case or manifest.get("case", "B?")
    production_path = Path(args.production) / "production.jsonl"
    derived_doc = None
    for cand in (run_dir.parent / DERIVED_FILE, run_dir / DERIVED_FILE):
        if cand.exists():
            derived_doc = json.loads(cand.read_text())
            break
    log(f"bundle {run_dir} profile {profile_id} case {case}; {'DERIVED bundle (' + derived_doc['tool'] + ')' if derived_doc else 'captured bundle'}")
    log(f"description: {len(description.nodes)} nodes validate structurally (relation ids not checked: no runtime registry on CPU)")

    structural_problems: list[str] = []
    trees = recompute_trees(reader, commitments)
    log(f"trees: {trees['matched']}/{trees['trees']} committed roots recomputed from the archive; weights: {trees['weights']}")
    structural_problems += [f"tree: {m}" for m in trees["mismatched"][:3]]

    fs = firststage.first_stage_problems(reader, commitments)
    log(f"first stage: {fs.requests} requests, {fs.steps} steps, {'ok' if fs.ok else 'PROBLEMS ' + '; '.join(fs.problems[:3])}")
    structural_problems += [f"first-stage: {p}" for p in fs.problems[:3]]

    ob = vbinding.output_binding(reader, commitments, production_path)
    log(f"binding output      {'ok ' if ob.ok else 'FAILED'} {ob.seconds:6.1f}s  {ob.detail[:300]}")
    if not ob.ok:
        structural_problems.append(f"output binding: {'; '.join(ob.data.get('problems', [])[:2]) or ob.detail}")
    logprobs = ob.data.get("logprobs", "unbound")

    ids = [x for x in args.negatives.split(",") if x]
    negatives = vbinding.binding_negatives(reader, commitments, cp=None, production_path=production_path, out_dir=out / "negatives",
                                           semantic_profile_path=args.semantic_profile, ids=ids)
    inapplicable = {d["negative_id"]: d["inapplicable"] for d in negatives if d.get("inapplicable")}
    not_run = [d["negative_id"] for d in negatives if d["attempted"] == 0 and not d.get("inapplicable")]
    for d in negatives:
        log(f"negative {d['negative_id']}: attempted {d['attempted']} rejected {d['rejected']}"
            + (f"  inapplicable: {d['inapplicable']}" if d.get("inapplicable") else ""))
    structural = {"result": "ok" if not structural_problems else "failed",
                  "failing_invariant": structural_problems[0] if structural_problems else None}

    utc = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    exp_id = args.experiment_id or rows.experiment_id(case, profile_id, utc)
    common = rows.Common(exp_id, case, profile_id, args.worker, args.adapter_revision)
    t3 = rows.table3_rows(common, [{k: d[k] for k in ("negative_id", "attempted", "rejected", "unexpectedly_accepted", "reproducer")}
                                   for d in negatives if not d.get("inapplicable")])
    negative = rows.negative_summary(t3)
    request_ids = reader.request_ids()
    boundaries = [reader.boundary(r) for r in request_ids]
    zero = {"exact": 0, "attempted": 0}
    challenge_field = {
        "mode": "none", "source": "verify_cpu (no challenge drawn)",
        "bindings": {"output": ob.ok, "logprobs": logprobs},
        **({"negatives_inapplicable": inapplicable} if inapplicable else {}),
    }
    evidence = {"label": rows.EVIDENCE_DERIVED if derived_doc else rows.EVIDENCE_CPU_ONLY, "verifier": args.worker, "report": None,
                "note": ("derived by " + derived_doc["tool"] + " from " + str(derived_doc["derived_from"]["label"]) +
                         "; a fresh GPU capture in the new format is required for a re-labelled row") if derived_doc
                else "no opening authenticated against a relation, no relation exercised, no challenge drawn"}
    # nothing is attempted on CPU: with the structural checks ok and every applicable negative rejected the
    # row is NOT_RUN (a blocker kind, not a verdict); otherwise the FAIL rule applies unblocked
    clean = not structural_problems and negative["rejected"] == negative["attempted"] and not not_run
    blocker = ({"kind": "NOT_RUN", "reason": f"CPU-only structural checks only (no GPU): structural ok, negatives {negative['rejected']}/{negative['attempted']}, "
                                             f"chosen logprobs {logprobs}; openings, relations, coverage and challenge not run"} if clean else None)
    t1 = rows.table1_row(
        common,
        blocker=blocker,
        workload={"requests": len(request_ids), "generated_positions": sum(len(b.output_token_ids) for b in boundaries),
                  "input_len_range": [min((len(b.prompt_token_ids) for b in boundaries), default=0), max((len(b.prompt_token_ids) for b in boundaries), default=0)],
                  "peak_concurrency_achieved": 0},
        production_changes="not assessed (CPU-only structural verification; see the source row)",
        replay_restrictions=[],
        token_selection=dict(zero),
        boundary_noninterference={"identical": False, "compared": 0, "mismatched_scalars": 0, "reference_method": "not run (CPU-only)"},
        required_coverage={"bound": 0, "required": 0},
        authentication={"valid": 0, "attempted": 0},
        relation_checks={"same_runtime": dict(zero), "independent": dict(zero)},
        structural=structural,
        negative=negative,
        challenge=challenge_field,
        negatives_not_run=not_run,
        qualifications=[f"CPU-only: {trees['matched']}/{trees['trees']} committed trees recomputed from the archive (weights: {trees['weights']}); "
                        "no opening authenticated against a relation, no relation exercised, no coverage sample, no challenge",
                        f"chosen logprobs {logprobs}"] + (["DERIVED bundle: boundary trees rewritten from the source archive's production.jsonl; not a capture"] if derived_doc else []),
        bindings={"output": ob.data},
        verification_evidence=evidence,
    )
    rows.write_jsonl(out / "table1.jsonl", [t1])
    rows.write_jsonl(out / "table3.jsonl", t3)
    doc = {"experiment_id": exp_id, "case": case, "profile_id": profile_id, "worker": args.worker, "utc": utc, "cpu_only": True,
           "derived": derived_doc, "trees": trees, "first_stage": fs.to_json(), "bindings": {"output": {"ok": ob.ok, "detail": ob.detail, **ob.data}},
           "negatives": negatives, "negatives_inapplicable": inapplicable, "structural": structural, "structural_problems": structural_problems,
           "verification_evidence": evidence, "table1": t1, "seconds": round(time.perf_counter() - t_start, 3)}
    (out / "verify.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    for name in ("manifest.json", "description.json", "commitments.json"):
        if (run_dir / name).exists():
            (out / name).write_text((run_dir / name).read_text())
    if derived_doc is not None:
        (out / DERIVED_FILE).write_text(json.dumps(derived_doc, indent=1, sort_keys=True) + "\n")
    log(f"RESULT {t1['result']} first_blocker={t1['first_blocker']} structural={structural} negatives={negative} logprobs={logprobs} "
        f"evidence={evidence['label']!r} ({time.perf_counter() - t_start:.1f}s)")
    (out / "log.txt").write_text("\n".join(log_lines) + "\n")
    return 0


def run(argv: list[str] | None = None) -> int:
    """Like ``verify.run``: whatever stops the pipeline is a ``FAIL structural: …`` row, never a traceback alone."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--case", default=None)
    ap.add_argument("--worker", default=WORKER)
    ap.add_argument("--adapter-revision", default="unknown")
    a, _ = ap.parse_known_args(argv)
    try:
        return main(argv)
    except SystemExit:
        raise
    except VB.StructuralFailure as e:
        reason, result, tb = f"{e.check}: {e.detail}", "failed", traceback.format_exc()
    except Exception as e:  # noqa: BLE001
        reason, result, tb = VB.structural_reason(e), "failed", traceback.format_exc()
    print(tb, file=sys.stderr, flush=True)
    run_dir = Path(a.run_dir)
    out = Path(a.out) if a.out else run_dir / "verify-cpu"
    profile_id, case = "unknown", a.case or "B?"
    try:
        m = json.loads((run_dir / "manifest.json").read_text())
        profile_id, case = prof.profile_id_of(m), a.case or m.get("case", case)
    except Exception:  # noqa: BLE001
        pass
    VB.refusal_row(out, case=case, profile_id=profile_id, worker=a.worker, adapter_revision=a.adapter_revision,
                   reason=reason[:600], exception=tb, result=result)
    return 1


if __name__ == "__main__":
    raise SystemExit(run())

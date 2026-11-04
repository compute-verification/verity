"""The evaluation job and witness checker (fixed): the only process that reads the held-out sets, run by candidate.sh from
the pinned measurement commit after the training job (sandbox.py) ended, outside its sandbox.

    python evaluate.py --submission DIR --timing FILE --run RUN.json --tree TREE.json --island ISLAND.json \\
        --out witness.json --scored OUT_RUN_DIR
    python evaluate.py --start NAME --out start.json     # a start's own score and evaluation cost
    CW_FINAL=1 python evaluate.py ... --split final      # final scoring (the scorer's, after the search)

For a submission it checks, in order, and records every result in the witness record:

1. θ: the start's weights match its registered digest, which is `prepare.START_DIGESTS`'s, and the submission and the
   harness's record name that digest;
2. E: decode(θ, E) with the fixed decoder rebuilds a well-formed model, in θ's architecture (θ's tensors, shapes and
   dtypes) or, for a model format, its own (`prepare.model_config`: any lf shape with θ's vocabulary and context); that
   model is the candidate (the training job keeps no model of its own). Nothing caps its size or cost: its parameters
   and evaluation cost are recorded (`prepare.eval_cost`), and each scored checkpoint's forward gates per token
   (`te_core.cw.score_run`);
3. changed: the declared set covers every tensor that differs from θ or isn't θ's; φ and the lowest updated layer are
   computed from it, at the model's own positions;
4. the harness's record (`te_core.run --script`, RUN.json) is this E's (`harness_submission`) and found nothing invalid;
   FLOPs, 2x the MACs it counted, within its tier's budget (`prepare.budget`: TRAIN_FLOP_BUDGET, or the long tier's
   LONG_C0 x the start's C0, which a board assignment names, `tier_assigned`) x HARNESS_FLOP_SLACK; its steps' wall
   seconds within the tier's; the declared trainable set covers `changed` (the analytic `prepare.train_flops` is recorded
   beside it);
5. the training job's whole process tree (TREE.json, sandbox.py): every process's CPU seconds but the harness's own
   within TREE_CPU_SLACK, and no process but the harness's computed on a GPU, or on a GPU it didn't lease
   (`tree_counted`); the seconds from the job's start to the submitted E's write are recorded;
6. the source this runs from is its commit's (`source_clean`), recorded as `measurement_commit` with the island
   (ISLAND.json: the candidate's `islands/<name>/`, its commit and digest);

(nothing caps E's size, its format or the updated layers: 8·|E|, the format and φ are recorded, and te_core.formalism
scores what the run did), then scores every checkpoint's E (`te_core.cw.score_run`): each E file checked against the digest the harness recorded,
decoded, hashed, and scored on a fresh slice of the search pool drawn after its weights were hashed, paired with θ on the
same slice, its promotion value sealed under that digest (`scored`). The submitted E's is the score; the FLOPs-to-target
curve is the checkpoints' (the first E in each 5% of the run). A candidate is valid when every check passes. The scored
record goes to OUT_RUN_DIR/run.json, which the te_core ledger ingests. The promotion and final sets are sealed:
`--split final` runs only in final mode (`CW_FINAL=1`, which also loads the sealed start), where every score is the
final set's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

import prepare
from te_core.cw import score_run

HERE = Path(__file__).resolve().parent
#: CPU seconds the training job's processes may use beyond the harness's own (its interpreter's exit after the record is
#: written), or 1% of the tree's, if more: a candidate starts no process, so more is work the counter didn't see
TREE_CPU_SLACK = 10.0


def eval_cost(split: str, rec: dict) -> dict:
    toks, man = prepare.held_out(split, rec["tokenizer"])
    return prepare.eval_cost(rec, len(toks), man["bytes"]) | {"set": split}


def source_clean() -> bool | str:
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "."], cwd=HERE,
                       capture_output=True, text=True)
    if r.returncode:
        return f"unknown: git status failed ({r.stderr.strip()[-200:]})"
    return r.stdout.strip() == "" or f"edited: {r.stdout.split()[:6]}"


def tree_of(tree: dict | None, r: dict) -> tuple[dict, bool]:
    if not tree or "cpu_s" not in tree:
        return {"error": "no record of the training job's process tree"}, False
    own = r["totals"].get("self_cpu_s")
    other = tree["cpu_s"] - (own or 0.0)
    leased = [u for u in os.environ.get("GPU_LEASE_UUID", "").split(",") if u]
    foreign = [a for a in tree.get("gpu_apps", [])
               if a["pid"] != tree["main_pid"] or (leased and a["gpu_uuid"] not in leased)]
    sub = r.get("submitted", {})
    out = {"cpu_s": tree["cpu_s"], "harness_cpu_s": own, "other_cpu_s": other,
           "other_processes": tree.get("other_processes"), "wall_s": tree.get("wall_s"),
           "e_written_s": sub["at"] - tree["started"] if sub.get("at") and tree.get("started") else None,
           "gpu_samples": tree.get("gpu_samples"), "gpu_apps": tree.get("gpu_apps", []), "foreign_gpu_apps": foreign,
           "leased": leased, "slack_cpu_s": max(TREE_CPU_SLACK, 0.01 * tree["cpu_s"])}
    return out, own is not None and other <= out["slack_cpu_s"] and not foreign


def check(sub: Path, timing: Path, run: Path, tree: Path, island: Path, scored: Path) -> dict:
    meta = json.loads((sub / "meta.json").read_text())
    rec = prepare.start_record(meta["start"])
    theta = prepare.load_theta(meta["start"])
    isl = json.loads(island.read_text()) if island.is_file() else None
    b = prepare.budget(meta["start"]) | {"flop_slack": prepare.HARNESS_FLOP_SLACK,
                                         "assigned_by": os.environ.get("CW_TIER_BY") or None}
    out = {"schema": "covert-witness/witness/v5", "version": prepare.VERSION, "data": prepare.DATA,
           "measurement_commit": os.environ.get("RESEARCH_SOURCE_SHA", ""), "island": isl,
           "start": meta["start"], "theta_digest": rec["digest"], "budget": b, "checks": {}, "notes": meta.get("notes")}
    c = out["checks"]
    c["tier_assigned"] = b["tier"] == "search" or bool(b["assigned_by"])
    c["theta"] = meta["theta_digest"] == rec["digest"]
    c["source_clean"] = source_clean()
    c["island_recorded"] = bool(isl and isl.get("commit") and isl.get("digest"))
    blob = (sub / "E.bin").read_bytes()
    out["E"] = {"bytes": len(blob), "bits": 8 * len(blob), "format": prepare.format_of(blob),
                "sha256": hashlib.sha256(blob).hexdigest()}
    try:
        decoded = prepare.decode(theta, blob)
        out["E"]["rebuild_macs"] = prepare.rebuild_macs(theta, blob)
    except (ValueError, KeyError, RuntimeError) as e:
        c["decode"] = f"failed: {e}"
        return out
    try:
        config = prepare.model_config(rec, blob)
    except ValueError as e:
        c["decode"] = f"failed: {e}"
        return out
    own = config != rec["config"]
    if own:
        shapes = prepare.architecture(config)
        c["architecture"] = set(decoded) == set(shapes) and all(tuple(decoded[k].shape) == shapes[k] for k in shapes)
    else:
        c["architecture"] = set(decoded) == set(theta) and all(
            decoded[k].shape == theta[k].shape and decoded[k].dtype == theta[k].dtype for k in theta)
    mrec = prepare.model_record(rec, config, decoded)
    out["model"] = {"architecture": "its own" if own else "θ's", "config": config if own else None,
                    "parameters": int(sum(v.numel() for v in decoded.values())), "start_parameters": rec["parameters"]}
    changed = {k for k in decoded if k not in theta or not prepare.same_bits(decoded[k], theta[k])}
    declared = set(meta["changed"])
    c["changed_covered"] = changed <= declared
    out["changed"] = {"declared": sorted(declared), "actual": sorted(changed),
                      "declared_parameters": int(sum(decoded[k].numel() for k in declared if k in decoded)),
                      "actual_parameters": int(sum(decoded[k].numel() for k in changed))}
    out["structure"] = prepare.deep_share(mrec, declared)
    out["structure_actual"] = prepare.deep_share(mrec, changed)
    del decoded
    trainable = meta["trainable"]
    steps = prepare.read_steps(sub / "steps.jsonl")
    try:
        meter = prepare.train_flops(mrec, trainable, steps)
    except (ValueError, KeyError) as e:
        c["meter"] = f"failed: {e}"
        return out
    out["train"] = meter
    trained = set(trainable.get("full", [])) | set(trainable.get("adapters", {}))
    c["trainable_covers_changed"] = changed <= trained
    r = json.loads(run.read_text())
    tot = r["totals"]
    flops = 2 * tot["macs"]
    out["harness"] = {"te_core": r.get("te_core"), "macs": tot["macs"], "flops": flops, "steps": len(r["steps"]),
                      "over_train_flops": flops / meter["flops"] if meter["flops"] else None,
                      "seconds": {k: tot.get(k) for k in ("wall_s", "gpu_s", "cpu_s", "process_wall_s", "self_cpu_s")},
                      "forward_macs_per_token": r.get("forward_macs_per_token"), "c": r.get("c"),
                      "flags": r.get("flags", []), "invalid": r.get("invalid", []), "stopped": r.get("stopped"),
                      "submitted": r.get("submitted", {}), "families": [
                          {k: f.get(k) for k in ("family", "passed", "failed", "covers")} for f in r.get("families", [])]}
    c["harness_theta"] = r.get("start", {}).get("digest") == rec["digest"]
    c["harness_submission"] = r.get("submitted", {}).get("e_sha256") == out["E"]["sha256"]
    c["harness_valid"] = not r.get("invalid")
    c["flop_budget"] = flops <= prepare.HARNESS_FLOP_SLACK * b["train_flops"]
    out["gpu_seconds"] = tot["wall_s"]
    c["gpu_seconds"] = tot["wall_s"] <= b["gpu_seconds"]
    out["implied_flop_per_second"] = flops / tot["wall_s"] if tot["wall_s"] else None
    out["tree"], c["tree_counted"] = tree_of(json.loads(tree.read_text()) if tree.is_file() else None, r)
    t = json.loads(timing.read_text())
    out["train_rc"], out["process_seconds"] = t.get("rc"), t.get("process_seconds")
    c["train_exit"] = t.get("rc") == 0
    split = prepare.FINAL if prepare.FINAL_MODE else prepare.SEARCH_POOL
    out["eval_cost"] = eval_cost(split, mrec) | {
        "rule": "the submitted model's own (θ's when it keeps θ's architecture); nothing caps it: each scored checkpoint's "
                "forward gates per token are counted (score_run) and scoring reads them"}
    r = score_run(prepare, r, run.parent, theta, rec, prepare.DEVICE)
    c["scored"] = not r["scored"]["problems"]
    submitted = next((cp for cp in r["checkpoints"] if cp.get("submitted")), None)
    out["score"] = submitted.get("score") if submitted else None
    if out["score"] is not None and "slice" in out["score"]:
        out["score"]["slice"]["digest"] = "te_core.run.theta_sha256"
    out["start_bpb"] = r["start"]["val_bpb"]
    out["valid"] = all(v is True for v in c.values())
    out["curve"] = [{"step": cp["step"], "train_flops": 2 * cp["macs"], "e_bytes": cp.get("e_bytes"),
                     "bpb": cp.get("val_bpb"), "rebuild_macs": cp.get("rebuild_macs"),
                     "source": cp.get("source")} for cp in r.get("checkpoints", [])]
    failed = {k: v for k, v in c.items() if v is not True}
    r["evaluation"] = {"by": "covert_witness/evaluate.py", "measurement_commit": out["measurement_commit"],
                       "version": prepare.VERSION, "data": prepare.DATA, "budget": out["budget"],
                       "island": {k: v for k, v in (isl or {}).items() if k != "files"}, "tree": out["tree"],
                       "witness_valid": out["valid"], "witness_checks": failed}
    r["invalid"] = [*r.get("invalid", []), *(f"witness {k}: {v}" for k, v in failed.items())]
    scored.mkdir(parents=True, exist_ok=True)
    (scored / "run.json").write_text(json.dumps(r) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--submission", type=Path)
    ap.add_argument("--start")
    ap.add_argument("--timing", type=Path)
    ap.add_argument("--run", type=Path, help="the harness's run.json (te_core.run --script), beside its E/ files")
    ap.add_argument("--tree", type=Path, help="the training job's process tree (sandbox.py)")
    ap.add_argument("--island", type=Path, help="the island's record (candidate.sh)")
    ap.add_argument("--scored", type=Path, help="where the scored run.json goes")
    ap.add_argument("--split", default=None, choices=("search", "final"), help="final: only with CW_FINAL=1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.split is not None and (a.split == "final") != prepare.FINAL_MODE:
        ap.error("the final set is sealed: final scoring is `CW_FINAL=1 python evaluate.py ... --split final`, "
                 "and in final mode every score is the final set's")
    torch.backends.cuda.matmul.allow_tf32 = True
    if a.start:
        rec = prepare.start_record(a.start)
        theta = prepare.load_theta(a.start)
        scorer = prepare.Scorer(rec, theta)
        out = {"schema": "covert-witness/start-score/v2", "start": a.start, "digest": rec["digest"],
               "parameters": rec["parameters"], "score": {"split": scorer.split, "bpb": scorer.start_bpb},
               "eval_cost": eval_cost(scorer.split, rec),
               "structure_full_update": prepare.deep_share(rec, set(theta)), "provenance": rec["provenance"]}
    else:
        if not (a.submission and a.timing and a.run and a.tree and a.island and a.scored):
            ap.error("--submission, --timing, --run, --tree, --island and --scored, or --start")
        out = check(a.submission, a.timing, a.run, a.tree, a.island, a.scored)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    s = out.get("score") or {}
    print(json.dumps({"valid": out.get("valid"), "bpb": s.get("bpb"), "E_bytes": out.get("E", {}).get("bytes"),
                      "phi": out.get("structure", {}).get("phi"), "train_flops": out.get("harness", {}).get("flops"),
                      "checks": out.get("checks")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

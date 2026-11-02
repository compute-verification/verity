"""The evaluator and witness checker (fixed): with the harness's evaluator, the only process that reads the held-out sets.

    python evaluate.py --submission DIR --class CLASS --timing FILE --run RUN.json --out witness.json
    python evaluate.py --start NAME --out start.json     # a start's own score and evaluation cost
    CW_FINAL=1 python evaluate.py ... --split final      # final scoring (the scorer's, after the search)

For a submission it checks, in order, and records every result in the witness record:

1. θ: the start's weights match its registered digest, and the submission and the harness's record name that digest;
2. E: decode(θ, E) rebuilds a model with θ's tensors, shapes and dtypes, bit-equal to the submitted model.safetensors, so
   its size and its evaluation FLOPs and weight bytes per byte equal the start's (`prepare.eval_cost`);
3. changed: the declared set covers every tensor that differs from θ; φ and the lowest updated layer are computed from it;
4. the harness's record (`te_core.run --script`, RUN.json) is this E's (`harness_submission`) and found nothing invalid;
   FLOPs, 2x the MACs it counted, within TRAIN_FLOP_BUDGET x HARNESS_FLOP_SLACK; its steps' wall seconds within
   GPU_SECONDS_CAP; the declared trainable set covers `changed` (the analytic `prepare.train_flops` is recorded beside it);
5. the class's caps on 8·|E|, E's format and the updated layers;

then scores decode(θ, E) (`prepare.Scorer.search`): hashed first, then on a fresh slice of the search pool, paired with
θ on the same slice. The FLOPs-to-target curve is the harness's checkpoints (the first E in each 5% of the run, scored
the same way as it ran, each on its own slice). A candidate is valid when every check passes. The promotion and final
sets are sealed: the harness seals each scored model's promotion value, and `--split final` runs only in final mode
(`CW_FINAL=1`, which also loads the sealed start), where every score is the final set's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

import prepare


def score(scorer: prepare.Scorer, model: dict[str, torch.Tensor]) -> dict:
    t0 = time.perf_counter()
    out = scorer.search(model, prepare.digest(model))
    if "slice" in out:
        out["slice"]["digest"] = "prepare.digest"
    return out | {"seconds": time.perf_counter() - t0}


def eval_cost(scorer: prepare.Scorer, rec: dict) -> dict:
    toks, man = prepare.held_out(scorer.split, rec["tokenizer"])
    return prepare.eval_cost(rec, len(toks), man["bytes"]) | {"set": scorer.split}


def check(sub: Path, klass: str, timing: Path, run: Path) -> dict:
    caps = prepare.CLASSES[klass]
    meta = json.loads((sub / "meta.json").read_text())
    rec = prepare.start_record(meta["start"])
    theta = prepare.load_theta(meta["start"])
    out = {"schema": "covert-witness/witness/v3", "version": prepare.VERSION, "class": klass, "caps": caps, "start": meta["start"],
           "theta_digest": rec["digest"], "budget": {"train_flops": prepare.TRAIN_FLOP_BUDGET,
                                                      "flop_slack": prepare.HARNESS_FLOP_SLACK,
                                                      "gpu_seconds": prepare.GPU_SECONDS_CAP}, "checks": {}, "notes": meta.get("notes")}
    c = out["checks"]
    c["theta"] = meta["theta_digest"] == rec["digest"]
    if meta.get("class") != klass:
        c["class_named"] = False
    blob = (sub / "E.bin").read_bytes()
    out["E"] = {"bytes": len(blob), "bits": 8 * len(blob), "format": prepare.format_of(blob)}
    try:
        decoded = prepare.decode(theta, blob)
    except (ValueError, KeyError, RuntimeError) as e:
        c["decode"] = f"failed: {e}"
        return out
    from safetensors.torch import load_file

    claimed = load_file(str(sub / "model.safetensors"))
    c["architecture"] = set(claimed) == set(theta) and all(
        claimed[k].shape == theta[k].shape and claimed[k].dtype == theta[k].dtype for k in theta)
    c["bit_equal"] = c["architecture"] and all(prepare.same_bits(decoded[k], claimed[k]) for k in theta)
    changed = {k for k in theta if not prepare.same_bits(decoded[k], theta[k])}
    declared = set(meta["changed"])
    c["changed_covered"] = changed <= declared
    out["changed"] = {"declared": sorted(declared), "actual": sorted(changed),
                      "declared_parameters": int(sum(theta[k].numel() for k in declared if k in theta)),
                      "actual_parameters": int(sum(theta[k].numel() for k in changed))}
    out["structure"] = prepare.deep_share(rec, declared)
    out["structure_actual"] = prepare.deep_share(rec, changed)
    trainable = meta["trainable"]
    steps = prepare.read_steps(sub / "steps.jsonl")
    try:
        meter = prepare.train_flops(rec, trainable, steps)
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
                      "seconds": {k: tot[k] for k in ("wall_s", "gpu_s", "cpu_s", "process_wall_s")},
                      "forward_macs_per_token": r.get("forward_macs_per_token"), "c": r.get("c"),
                      "flags": r.get("flags", []), "invalid": r.get("invalid", []), "stopped": r.get("stopped"),
                      "submitted": r.get("submitted", {}), "families": [
                          {k: f.get(k) for k in ("family", "passed", "failed", "covers")} for f in r.get("families", [])]}
    c["harness_theta"] = r.get("start", {}).get("digest") == rec["digest"]
    c["harness_submission"] = r.get("submitted", {}).get("e_sha256") == hashlib.sha256(blob).hexdigest()
    c["harness_valid"] = not r.get("invalid")
    c["flop_budget"] = flops <= prepare.HARNESS_FLOP_SLACK * prepare.TRAIN_FLOP_BUDGET
    out["gpu_seconds"] = tot["wall_s"]
    c["gpu_seconds"] = tot["wall_s"] <= prepare.GPU_SECONDS_CAP
    out["implied_flop_per_second"] = flops / tot["wall_s"] if tot["wall_s"] else None
    t = json.loads(timing.read_text())
    out["train_rc"], out["process_seconds"] = t.get("rc"), t.get("process_seconds")
    c["train_exit"] = t.get("rc") == 0
    if caps["max_e_bytes"] is not None:
        c["e_size"] = len(blob) <= caps["max_e_bytes"]
    if caps["formats"] is not None:
        c["e_format"] = out["E"]["format"] in caps["formats"]
    if caps["top_blocks"] is not None:
        L = rec["config"]["n_layer"]
        lowest = out["structure"]["lowest_updated_layer"]
        c["shallow"] = lowest is None or lowest >= L + 1 - caps["top_blocks"]
    scorer = prepare.Scorer(rec, theta)
    out["eval_cost"] = eval_cost(scorer, rec) | {
        "rule": "the model has θ's tensors, shapes and dtypes (checks.architecture), so its parameters, evaluation FLOPs "
                "and weight bytes per byte are the start's"}
    out["valid"] = all(v is True for v in c.values())
    out["score"] = score(scorer, decoded)
    out["curve"] = [{"step": cp["step"], "train_flops": 2 * cp["macs"], "e_bytes": cp.get("e_bytes"), "bpb": cp.get("val_bpb"),
                     "source": cp.get("source")} for cp in r.get("checkpoints", [])]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--submission", type=Path)
    ap.add_argument("--start")
    ap.add_argument("--class", dest="klass", choices=sorted(prepare.CLASSES))
    ap.add_argument("--timing", type=Path)
    ap.add_argument("--run", type=Path, help="the harness's run.json (te_core.run --script)")
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
               "eval_cost": eval_cost(scorer, rec),
               "structure_full_update": prepare.deep_share(rec, set(theta)), "provenance": rec["provenance"]}
    else:
        if not (a.submission and a.klass and a.timing and a.run):
            ap.error("--submission, --class, --timing and --run, or --start")
        out = check(a.submission, a.klass, a.timing, a.run)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    s = out.get("score", {})
    print(json.dumps({"valid": out.get("valid"), "bpb": s.get("bpb"), "E_bytes": out.get("E", {}).get("bytes"),
                      "phi": out.get("structure", {}).get("phi"), "train_flops": out.get("harness", {}).get("flops"),
                      "checks": out.get("checks")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

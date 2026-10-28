"""[D88] The run directory as an *Experiment*: one index that keeps the campaign's four records apart.

    Experiment
        specification    what we intended to run -- known BEFORE execution        spec.json (+ capture/spec_tokens.json)
        execution        what actually happened                                   capture/, control/, census/, nonint/, run.json#realized
        representation   the Verity Program and how it was pinned to the run     program.json, instances.jsonl, accesses.jsonl.gz, wiring,
                                                                                  provenance, resolution, prescribed inputs, transcript values
        checks           what we tested and what it said                         gates.json, replay_*.json, compare.json, …, stages.json (cost)

Nothing here is a new Verity object and no large file is duplicated: `experiment.json` references the run's files by relative path with
their byte size, a sha256 for the small ones, and a presence flag.  Two disciplines are enforced mechanically:

* `spec.json` is written once, before the GPU phase, from the workload file + the driver's arguments + the checkpoint manifest + the HF
  config, and is never rewritten by a later phase (a CPU re-run with different arguments records the mismatch in `experiment.json`
  instead of touching the spec).  The prompt token ids are the workload loader's, written by the capture process before its first
  engine step (`capture/spec_tokens.json`, `known_before_execution: true`).  Nothing observed from the run goes into either file; the
  realized shape lives in `run.json["realized"]` and `capture/tokens.json` (execution).
* Transcript values are reported with their availability: the snapshot index lists every captured tensor and every tensor skipped for
  size (`skipped: "spanned bytes … > snapshot_max_bytes …"`); `replay_a.json["vu_population"]` counts the VUs whose values are
  available / oversize / zero-mass.  A VU whose values are unavailable is never counted as evidenced -- that is the replay's rule,
  restated here as data.

The future config-driven Program predictor reads `specification` (+ `representation.program`) as one training example; `execution`
is what validates it.  `checks.classification` (PASS / ADAPTER / MECHANISM / SCALE) is the *campaign's* judgement and is filled from the
coverage table, not by the run.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from verity_vllm.config import RowEngine
from verity_vllm.pipeline import research_tools

SPEC_SCHEMA = "verity-gen/spec/v1"
EXPERIMENT_SCHEMA = "verity-gen/experiment/v1"
SMALL_FILE_SHA_LIMIT = 64 << 20  # sha256 only for files up to 64 MiB (program.json, reports); bigger ones carry size only


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest_doc(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def _ref(run: Path, rel: str, *, sha: bool = True) -> dict:
    """A file reference: relative path, presence, bytes, sha256 (small files), mtime.  Directories: presence + total bytes."""
    p = run / rel
    if not p.exists():
        return {"path": rel, "present": False}
    if p.is_dir():
        n = 0
        total = 0
        for f in p.rglob("*"):
            if f.is_file():
                n += 1
                total += f.stat().st_size
        return {"path": rel, "present": True, "dir": True, "files": n, "bytes": total}
    st = p.stat()
    out = {"path": rel, "present": True, "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))}
    if sha and st.st_size <= SMALL_FILE_SHA_LIMIT:
        out["sha256"] = _sha256(p)
    return out


def _load(run: Path, rel: str):
    p = run / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):  # an unreadable report is reported as absent
        return None


def code_version(repo: Path) -> dict:
    """The tree the run was produced by: the pod's EXPORT.json (ship.sh: commit + dirty-patch digest) when present, else `git rev-parse`."""
    exp = repo / "EXPORT.json"
    if exp.exists():
        try:
            d = json.loads(exp.read_text())
        except (OSError, ValueError):
            d = None
        if isinstance(d, dict):
            return {"source": "EXPORT.json", "commit": d.get("commit"), "dirty_patch_sha256": d.get("patch_sha256") or d.get("dirty_patch"),
                    "untracked": d.get("untracked"), "fallback": d.get("fallback")}
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10).stdout.strip()
        code = [f":/{d}" for d in research_tools.CODE_DIRS]
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "verity_vllm", *code], cwd=repo,
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return {"source": "git", "commit": sha or None, "dirty": bool(dirty)}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"source": "unknown"}


# ----------------------------------------------------------------------------------------------------------------------------------
# specification
# ----------------------------------------------------------------------------------------------------------------------------------

def build_spec(*, config_id: str, case: str, profile: str | None, workload_path: str | None, workload_doc: dict | None,
               checkpoint: dict | None, checkpoint_dir: str | None, hf_config: dict | None, engine_args: dict, max_num_seqs: int,
               sampling: dict, tp: int = 1, dtype: str | None = None, snapshot_steps: list[int] | None, snapshot_max_bytes: int | None,
               initial_state: dict | None = None, harness: dict | None = None) -> dict:
    """Everything known before execution, in the three blocks (configuration / workload / initial state).  The concrete requests are
    copied out of the workload file (text prompts, per-request max_tokens, arrival steps, request ids); their token ids come from the
    capture process's `spec_tokens.json` (same information, produced by the tokenizer before the first engine step)."""
    wl = workload_doc or {}
    reqs = []
    for r in wl.get("requests") or []:
        rec = {k: r.get(k) for k in ("request_id", "prompt", "prompt_token_ids", "max_tokens", "arrive_step", "seed", "n", "variant",
                                      "prompt_len_target", "n_generated_target") if k in r}
        if "prompt_token_ids" in rec:
            rec["prompt_len"] = len(rec["prompt_token_ids"])
        reqs.append(rec)
    spec = {
        "schema": SPEC_SCHEMA, "known_before_execution": True, "config_id": config_id,
        "configuration": {
            "case": case,
            "checkpoint": {k: (checkpoint or {}).get(k) for k in ("repo", "revision", "role", "files", "sha256", "digest") if (checkpoint or {}).get(k) is not None},
            "checkpoint_dir": checkpoint_dir,
            "hf_config": hf_config,
            "profile": profile,                       # vLLM build / arch / execution mode profile id (e.g. vllm_d9105ea80_sm89_eager)
            "engine_args": dict(engine_args),         # the LLM(**kw) overrides the harness will pass (workload-required + CLI)
            "max_num_seqs": max_num_seqs,
            # the served dtype: the driver's explicit value, else the engine-argument override the harness passes to LLM(**kw)
            # (a workload's `engine.args.dtype`, e.g. float16 -- the fp16 column), else the checkpoint's torch_dtype
            "dtype": dtype or (engine_args or {}).get("dtype") or (hf_config or {}).get("torch_dtype"),
            "tensor_parallel": tp,
            "snapshot_steps_requested": snapshot_steps, "snapshot_max_bytes": snapshot_max_bytes,
            "harness": harness or {},
        },
        "workload": {
            "path": workload_path,
            "file_sha256": _sha256(Path(workload_path)) if workload_path and Path(workload_path).exists() else None,
            "serving_profile": wl.get("serving_profile"), "context_class": wl.get("context_class"),
            "length_seed": wl.get("length_seed"), "prompt_seed": wl.get("prompt_seed"), "replicate": wl.get("replicate"),
            "expected_engine_steps": wl.get("expected_engine_steps"), "engine_args_required": dict(RowEngine.of(wl).args),
            "sampling": {**(wl.get("sampling") or {}), **{k: v for k, v in (sampling or {}).items() if v is not None}},
            "variants": wl.get("variants"),           # named sampling parameter sets the `sampling.variant` / per-request `variant` select
            "n_requests": len(reqs), "requests": reqs,
            "tokens": "capture/spec_tokens.json",     # the same prompt token ids as the capture process saw them before its first engine step
        },
        "initial_state": initial_state or {"kind": "fresh", "note": "fresh engine, empty KV cache; any prefix reuse happens inside this run's own requests"},
    }
    spec["digest"] = _digest_doc({k: v for k, v in spec.items() if k != "digest"})
    return spec


def write_spec(run: Path, spec: dict) -> dict:
    """Write spec.json once.  If it exists, leave it alone and return a mismatch note when this call would have written something
    different (a CPU re-run with other arguments does not rewrite the specification)."""
    p = run / "spec.json"
    if p.exists():
        try:
            old = json.loads(p.read_text())
        except (OSError, ValueError):
            old = {}
        same = old.get("digest") == spec.get("digest")
        return {"written": False, "existing_digest": old.get("digest"), "would_be_digest": spec.get("digest"), "same": same}
    run.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(spec, indent=1, default=str) + "\n")
    return {"written": True, "digest": spec["digest"]}


# ----------------------------------------------------------------------------------------------------------------------------------
# the index
# ----------------------------------------------------------------------------------------------------------------------------------

def _transcript_values(run: Path) -> dict:
    """Availability of transcript values: the snapshot index (captured vs skipped-for-size) and the replay's VU population counts."""
    snaps = _load(run, "capture/snapshots.json")
    out: dict = {"snapshot_index": _ref(run, "capture/snapshots.json", sha=False), "values_dir": _ref(run, "capture/values")}
    if isinstance(snaps, dict):
        entries = snaps.get("snapshots") or snaps.get("entries") or []
    elif isinstance(snaps, list):
        entries = snaps
    else:
        entries = []
    if entries:
        skipped = [e for e in entries if isinstance(e, dict) and e.get("skipped")]
        steps = sorted({int(e.get("step")) for e in entries if isinstance(e, dict) and e.get("step") is not None})
        out.update({"tensors_captured": len(entries) - len(skipped), "tensors_skipped_oversize": len(skipped), "snapshot_steps": steps,
                    "skipped_examples": [{"path": e.get("path"), "shape": e.get("shape"), "reason": e.get("skipped")} for e in skipped[:5]]})
    ra = _load(run, "replay_a.json")
    if isinstance(ra, dict) and ra.get("vu_population"):
        out["vu_population"] = ra["vu_population"]      # available / oversize / zero-mass / partially tiled specs (D85)
    out["rule"] = "a VU whose required values are neither captured nor recomputable is not evidenced (replay.tier_a samples only from strata with values)"
    return out


def _phase_realized(run: Path) -> dict[str, dict | None]:
    """Per phase: {engine_steps, requests: {rid: {first_step, finish_step, n_generated}}} from that phase's own record --
    capture/control/nonint variants from tokens.json#timing.requests (D87), the census from census.json#realized."""
    out: dict[str, dict | None] = {}
    def from_tokens(rel: str) -> dict | None:
        d = _load(run, rel)
        if not d:
            return None
        t = d.get("timing") or {}
        reqs = t.get("requests") or {}
        return {"engine_steps": t.get("engine_steps", d.get("engine_steps")),
                "requests": {rid: {k: r.get(k) for k in ("first_step", "finish_step", "n_generated", "arrive_step_realized")} for rid, r in reqs.items()}}
    out["capture"] = from_tokens("capture/tokens.json")
    out["control"] = from_tokens("control/tokens.json")
    for v in ("a_observer", "b_hooks", "c_bare"):
        out[f"nonint/{v}"] = from_tokens(f"nonint/{v}/tokens.json")
    c = _load(run, "census/census.json")
    if c:
        r = c.get("realized")
        out["census"] = ({"engine_steps": r.get("engine_steps"),
                          "requests": {rid: {k: q.get(k) for k in ("first_step", "finish_step", "n_generated", "arrive_step_realized")} for rid, q in (r.get("requests") or {}).items()}}
                         if r else {"engine_steps": len(c.get("steps") or []), "requests": None, "note": "census without a realized block (pre-a9a13179f)"})
    else:
        out["census"] = None
    return out


def realized_consistency(run: Path) -> dict:
    """Spec §5: record AND check the realized facts of every phase against the capture -- the same spec digest does not prove the
    census or a non-interference variant honoured the arrivals (the census did not, before a9a13179f).  Informational: gates G1 (census
    alignment, non-interference step counts) remain the verdict; this names the phase and the request that differ."""
    ph = _phase_realized(run)
    ref = ph.get("capture")
    problems: list[str] = []
    compared: list[str] = []
    missing = [k for k, v in ph.items() if v is None]
    if not ref:
        return {"status": "no capture record", "phases_missing": missing}
    for name, d in ph.items():
        if name == "capture" or d is None:
            continue
        compared.append(name)
        if d.get("engine_steps") is not None and ref.get("engine_steps") is not None and int(d["engine_steps"]) != int(ref["engine_steps"]):
            problems.append(f"{name}: engine_steps {d['engine_steps']} != capture {ref['engine_steps']}")
        if d.get("requests") and ref.get("requests"):
            for rid, rr in ref["requests"].items():
                q = d["requests"].get(rid)
                if q is None:
                    problems.append(f"{name}: request {rid[:8]} absent")
                    continue
                for k in ("first_step", "finish_step", "n_generated"):
                    if rr.get(k) is not None and q.get(k) is not None and rr[k] != q[k]:
                        problems.append(f"{name}: request {rid[:8]} {k} {q[k]} != capture {rr[k]}")
    return {"status": "consistent" if not problems else "inconsistent", "phases_compared": compared, "phases_missing": missing,
            "capture_engine_steps": ref.get("engine_steps"), "problems": problems[:40], "n_problems": len(problems)}


def vu_partition_index(qu: dict | None, gates: dict | None) -> dict:
    """Checks-section entry for the query-use document: its identities and verdict fields lifted verbatim (frozen shape
    `{"schema", "binding_id", "use": {"kind", "w_max"}, "verdict": {partition{status,codes}, width{status,w_max,max_seen,violators,
    unsupported}, count, status}}`), plus G8's status from gates.json.  `present: false` when the run has no document."""
    g8 = ((gates or {}).get("gates") or {}).get("G8") if isinstance(gates, dict) else None
    gate = {"status": g8.get("status"), "reason": g8.get("reason")} if isinstance(g8, dict) else None
    if not isinstance(qu, dict):
        return {"present": False, "gate": gate}
    v = qu.get("verdict") if isinstance(qu.get("verdict"), dict) else {}
    part = v.get("partition") if isinstance(v.get("partition"), dict) else {}
    width = v.get("width") if isinstance(v.get("width"), dict) else {}
    binding = (g8.get("detail") or {}).get("binding") if isinstance(g8, dict) else None
    return {"present": True, "schema": qu.get("schema"), "binding_id": qu.get("binding_id"), "use": qu.get("use"),
            "partition": {"status": part.get("status"), "codes": list(part.get("codes") or [])[:32]},
            "width": {"status": width.get("status"), "w_max": width.get("w_max"), "max_seen": width.get("max_seen"),
                      "n_violators": len(width.get("violators") or []), "n_unsupported": len(width.get("unsupported") or [])},
            "count": v.get("count"), "status": v.get("status"), "verdict_status": v.get("status"),
            "binding": binding if isinstance(binding, dict) else {"checked": False}, "gate": gate}


def build_experiment(run: Path, *, repo: Path | None = None, spec_note: dict | None = None) -> dict:
    """The index over an existing run directory.  Cheap: file stats, sha256 of the small files, a few fields lifted from the reports."""
    man = _load(run, "run.json") or {}
    spec = _load(run, "spec.json") or {}
    gates = _load(run, "gates.json") or {}
    ra = _load(run, "replay_a.json") or {}
    stages = _load(run, "stages.json") or []
    fold = _load(run, "fold_summary.json") or {}
    qu = _load(run, "query_use.json")
    prog_ref = _ref(run, "program.json")
    if not prog_ref.get("present"):
        prog_ref = _ref(run, "program.json.gz")
    inst_ref = _ref(run, "instances.jsonl", sha=False)
    if not inst_ref.get("present"):
        inst_ref = _ref(run, "instances.jsonl.gz", sha=False)
    realized = man.get("realized")
    doc = {
        "schema": EXPERIMENT_SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config_id": man.get("config_id"),
        "run_dir": str(run),
        "specification": {
            "spec": _ref(run, "spec.json"), "spec_digest": spec.get("digest"), "spec_note": spec_note,
            "spec_tokens": _ref(run, "capture/spec_tokens.json"),
            "workload_file": {"path": man.get("workload_path"), "sha256": (spec.get("workload") or {}).get("file_sha256")},
            "checkpoint": (spec.get("configuration") or {}).get("checkpoint") or {"checkpoint_dir": man.get("checkpoint_dir")},
            "serving_profile": (spec.get("workload") or {}).get("serving_profile") or (man.get("workload") or {}).get("serving_profile"),
            "context_class": (spec.get("workload") or {}).get("context_class") or (man.get("workload") or {}).get("context_class"),
            "initial_state": spec.get("initial_state"),
        },
        "execution": {
            "observation_log": _ref(run, "capture/log.jsonl.gz", sha=False),
            "tokens": _ref(run, "capture/tokens.json"), "versions": _ref(run, "capture/versions.json"), "stats": _ref(run, "capture/stats.json"),
            "control_tokens": _ref(run, "control/tokens.json"),
            "census": _ref(run, "census/census.json"), "census_traces": _ref(run, "census/traces"),
            "noninterference": _ref(run, "nonint/noninterference.json"),
            "realized": realized,                      # per-request prompt/generated lengths, arrival/first/finish steps, engine steps (run.json)
            "realized_consistency": realized_consistency(run),   # every phase's own realized arrivals vs the capture's (spec §5)
            "observed_vs_interpreted": "capture/log.jsonl.gz, tokens.json, census/ are runtime facts; resolution.json onward is interpretation",
        },
        "representation": {
            "program": prog_ref, "program_digest": fold.get("digest") or fold.get("program_digest"),
            "definitions": fold.get("n_definitions") or fold.get("definitions"),
            "instances_correspondence": inst_ref,      # per Compute instance: step, root event id, pattern, spec id, rows, operand/output views + refs
            "accesses": _ref(run, "accesses.jsonl.gz", sha=False),
            "wiring": _ref(run, "wiring.json", sha=False), "provenance": _ref(run, "provenance.json", sha=False),
            "derivation": {"resolution": _ref(run, "resolution.json", sha=False), "fold_summary": _ref(run, "fold_summary.json"),
                           "fold_summary_2": _ref(run, "fold_summary_2.json"), "root_policy": _ref(run, "root_policy.json"),
                           "note": "runtime event -> pattern/resolution (resolution.json) -> Definition + statics -> Instances (instances.jsonl)"},
            "prescribed_inputs": {
                "weights": {"checkpoint_dir": man.get("checkpoint_dir"), "verified_against_manifest": man.get("verify_weights"),
                            "manifest": man.get("checkpoints")},
                "prompt_tokens": "capture/spec_tokens.json",
                "initial_state": (spec.get("initial_state") or {}).get("kind"),
                "note": "known-before-execution does not put a value into the Program statics; weights and prompts are Input gates. "
                        "Generated tokens and MoE routes are computed circuit values, not free inputs.",
            },
            "transcript_values": _transcript_values(run),
            "versions": {"code": code_version(repo) if repo else None, "profile": man.get("profile"),
                         "capture_versions": _ref(run, "capture/versions.json"), "cos_sin": _ref(run, "capture/cos_sin_cache.npy", sha=False)},
        },
        "checks": {
            "settings": {"sampling": man.get("sampling"), "snapshot_steps": man.get("snapshot_steps"), "tier_b_steps": man.get("tier_b_steps"),
                         "parallel_replay": man.get("parallel_replay"), "skip": man.get("skip"),
                         "gates_constants": (gates.get("constants") if isinstance(gates, dict) else None),
                         "tier_a": {k: ra.get(k) for k in ("k", "seed", "strata", "n_strata", "n_evaluated") if isinstance(ra, dict) and k in ra}},
            "results": {"verdict": gates.get("verdict"), "gates": _ref(run, "gates.json"), "card": _ref(run, "card.json"),
                        "replay_a": _ref(run, "replay_a.json", sha=False), "replay_chain": _ref(run, "replay_chain.json", sha=False),
                        "compare": _ref(run, "compare.json", sha=False), "census_reconcile": _ref(run, "census_reconcile.json", sha=False),
                        "golden": _ref(run, "golden.json"), "quarantine": _ref(run, "quarantine.json"), "holdouts": _ref(run, "holdouts.json"),
                        "query_use": _ref(run, "query_use.json"), "query": _ref(run, "query.json")},
            # G8: the VU-partition use document (`verity-ir/query-use/v1`) and what the gate said about it
            "vu_partition": vu_partition_index(qu, gates),
            "costs": {"stages": [{"stage": r.get("stage"), "status": r.get("status"), "seconds": r.get("seconds")} for r in stages if isinstance(r, dict)],
                      "total_seconds": round(sum(float(r.get("seconds") or 0) for r in stages if isinstance(r, dict)), 1)},
            "classification": None, "classified_by": None,   # PASS / ADAPTER / MECHANISM / SCALE: the campaign table's judgement, filled by the lead
        },
    }
    return doc


def write_experiment(run: Path, *, repo: Path | None = None, spec_note: dict | None = None) -> dict:
    doc = build_experiment(run, repo=repo, spec_note=spec_note)
    (run / "experiment.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return doc

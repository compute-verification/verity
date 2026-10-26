"""The typed outputs of one sweep stage for the `research` store (`outputs.json`, schema research/outputs/v0.1).

    verity-vllm research-outputs write <evidence-dir> <row-id> <stage> [--run-dir DIR] [--input build=art:...] [--input match=art:...]
    verity-vllm research-outputs digest <evidence-dir> <row-id>      # the Program digest of record (build_summary.json)

`run_row_v2.sh` calls this on EXIT of a `stage` run (the runner set RESEARCH_RUN_DIR): the stage's small account files are copied
from the evidence dir into `$RESEARCH_RUN_DIR/outputs/<name>/` and declared in `$RESEARCH_RUN_DIR/outputs.json`, which
`research.store.attempt.publish` turns into one artifact per entry:

    plan    -> vllm-commit-plan/v1    plan/commit_plan.json (the Commit's plan: its key and the producer facts), plan.log,
                                      weights_of_record.log, manifest.log; meta {row, stage, program_digest, plan_key, stage_status}; refs.build
    build   -> vllm-build/v1          build_summary.json, build_workload/{workload_program.json, descriptor.json.gz} (the workload
                                      Program), build_step/ and build_request*/ descriptor files (never
                                      instances.json.gz), manifest.json (the required-value manifest, when built), target_family / admission /
                                      source_identity / tree / release records; meta {row, stage, program_digest, workload_digest,
                                      correspondence_digest?, required_manifest, stage_status}
    match   -> vllm-match-record/v1   match/instances.jsonl, match/program.json, match/run.json, match/versions.json, match/control/tokens.json,
                                      match/capture/tokens.json, global_match.json, match_summary.json, match_compare.json, match_decomp.json,
                                      match_shape.json, match/fold_summary.json -- NOT the capture planes (match/capture/log.jsonl.gz);
                                      meta {row, stage, program_digest, stage_status, global_match_ok}; refs.build = the consumed Build
    commit  -> vllm-verdict/v1        verdict.json (the verdict of record, vllm-verdict/v1), commit/ (commit_delta's verdict.json,
                                      sampled_replay_p*.json, versions.json, manifest_verify.json, ...; small files), weights_of_record.log,
                                      manifest.log, manifest_verify.log; meta {row, stage, program_digest, verdict, outcome, stage_status};
                                      refs.build, refs.match  (commit_summary.json and the finalize board are retired);
                                      a config run (`config` verdict line) decides in config_record.json instead, at TP1 and TP
    replay  -> vllm-verdict/v1        config_record.json beside the rewritten Commit record (a TP row that is not a config run decides in
                                      commit/summary.json, as tp-commit-replay folded it); and one vllm-replay-slim/v1 per slim keep
                                      (`replay_slim_p<pair>`: commit/replay_slim_p<pair>/ whole, planes included; meta = its slim.json's
                                      pair, run_root, program_digest, picked, bytes, dump_sha256, commit_pass).  Never a replay bundle.
    (all)   -> execution-evidence/v1  timeline.jsonl, stages.txt, admission.json, timeline_report.json; meta {row, stage}
    (all)   -> run-files/v1 `logs`    row.log and every top-level *.log of the evidence dir

A tensor-parallel row (tp_stage.sh; build_summary.json carries `tp.world` and `ranks`) keeps its Build per rank under build/rank<r>/:
the build entry takes each rank's build_step/, build_request*/ and build_workload/{workload_program.json, descriptor.json.gz} (every
Program a rank serves, so `pipeline.circuit_db.programs` decodes the row from this artifact alone), the match entry tp.match's
match/tp2_match.json (+ the fold Match's), the verdict entry tp.commit's commit/ (its decision document is commit/summary.json: a TP
Commit writes no verdict.json), and `program_digest` is the TP manifest's key digest -- the `tp-ranks` fold of the rank workload
digests (the manifest's rank merge), with `rank_workload_digests` beside it.  A Commit stage that reported PASS without its decision
document is refused by name (no outputs.json), never published with empty fields.

Files larger than MAX_FILE are left out and named in meta.skipped (the raw planes and folds stay on the volume; the kind says so).
Stdlib only; never raises out of `main` (an outputs.json that could not be written is reported, the stage's own exit status stands).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.config import option
from verity_vllm.query.manifest.format import tp_ranks_fold_digest

SCHEMA = "research/outputs/v0.1"
MAX_FILE = 512 << 20                      # per file; the required-value manifest of a long row is ~100+ MB
NEVER = ("instances.json.gz", "log.jsonl.gz")
PLANE_SUFFIXES = (".npy", ".npz", ".pt", ".bin", ".safetensors")

KINDS = {"build": "vllm-build/v1", "plan": "vllm-commit-plan/v1", "match": "vllm-match-record/v1", "commit": "vllm-verdict/v1", "replay": "vllm-verdict/v1"}
SLIM_KIND = "vllm-replay-slim/v1"
#: never in a stage's account files: the replay bundles (never published) and the slim keeps (published whole, as their own entries)
KEPT_APART = ("replay_bundle_p", "replay_slim_p")
SLIM_META = ("pair", "run_root", "program_digest", "picked", "bytes", "dump_sha256", "commit_pass")

BUILD_FILES = ["build_summary.json", "build_workload/workload_program.json", "build_workload/descriptor.json.gz", "manifest.json",
               "target_family.build.json", "target_family.json",
               "admission.json", "source_identity.json", "tree.json", "release.json", "build_request/digest.txt"]
MATCH_FILES = ["match/instances.jsonl", "match/program.json", "match/run.json", "match/versions.json", "match/control/tokens.json",
               "match/capture/tokens.json", "global_match.json", "match_summary.json", "match_compare.json", "match_decomp.json",
               "match_shape.json", "match/fold_summary.json", "match_stoch.json",
               "match/tp2_match.json", "match/fold_match/tp2_fold_match.json", "match/capture/topology.json", "match/control/topology.json",
               "match/capture/versions.json", "match/control/versions.json"]
TP_BUILD_GLOBS = ["build/rank*/build_step", "build/rank*/build_request*", "build/rank*/build_workload/workload_program.json",
                  "build/rank*/build_workload/descriptor.json.gz"]
TP_LOG_GLOBS = ["build/rank*/*.log", "match/*.log"]
PLAN_FILES = ["plan/commit_plan.json", "plan.log", "weights_of_record.log", "manifest.log"]
COMMIT_FILES = ["verdict.json", "config_record.json", "weights_of_record.log", "manifest.log", "manifest_verify.log"]
EVIDENCE_FILES = ["timeline.jsonl", "stages.txt", "admission.json", "timeline_report.json"]


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verdicts(stages_txt: Path) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    if not stages_txt.is_file():
        return out
    for line in stages_txt.read_text(errors="replace").splitlines():
        parts = line.split(" ", 2)
        if len(parts) >= 2:
            out[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return out


def row_batch(row: str) -> int:
    m = re.search(r"__b(\d+)__", row)
    return int(m.group(1)) if m else 1


def tp_world(build_summary: Any) -> int:
    """The tensor-parallel world of a tp_stage.sh Build (`tp.world` of build_summary.json), 1 for a TP1 Build."""
    tp = build_summary.get("tp") if isinstance(build_summary, dict) else None
    return int(tp.get("world") or 1) if isinstance(tp, dict) else 1


def program_digest(d: Path, row: str = "") -> dict[str, Any]:
    """The Build's digests of record, from build_summary.json (never from the manifest): `program_digest` = the request Program's at
    B=1, GP-01's workload Program digest at B>=2 (the harness's `program_digest_of_record`); the step and workload digests beside."""
    s = _read_json(d / "build_summary.json") or {}
    if not isinstance(s, dict):
        s = {}
    world = tp_world(s)
    if world > 1:
        ranks = s.get("ranks") if isinstance(s.get("ranks"), dict) else {}
        wds = [((ranks.get(f"rank{r}") or {}).get("workload") or {}).get("workload_digest") for r in range(world)]
        step = (ranks.get("rank0") or {}).get("build_step") or {}
        # the TP manifest's header digest (`tp-ranks-fold`, required.merge_ranks): the fold of the rank digests its identities are keyed by
        of_record = tp_ranks_fold_digest(wds) if all(wds) else None
        return {"program_digest": of_record, "request_digest": None, "step_digest": step.get("program_digest"), "workload_digest": None,
                "rank_workload_digests": wds, "tp": world}
    req = s.get("request") if isinstance(s.get("request"), dict) else {}
    step = s.get("step") if isinstance(s.get("step"), dict) else {}
    wl = s.get("workload") if isinstance(s.get("workload"), dict) else {}
    request_digest, step_digest, workload_digest = req.get("program_digest"), step.get("program_digest"), wl.get("workload_digest")
    of_record = (workload_digest or request_digest) if row_batch(row) > 1 else (request_digest or s.get("program_digest"))
    out = {"program_digest": of_record, "request_digest": request_digest, "step_digest": step_digest, "workload_digest": workload_digest}
    corr = s.get("correspondence") if isinstance(s.get("correspondence"), dict) else {}
    cd = s.get("correspondence_digest") or corr.get("digest") or corr.get("correspondence_digest")
    if cd:
        out["correspondence_digest"] = cd
    return out


def _eligible(p: Path) -> bool:
    if not p.is_file() or p.is_symlink():
        return False
    if p.name in NEVER or p.suffix in PLANE_SUFFIXES:
        return False
    return p.stat().st_size <= MAX_FILE


def _collect(d: Path, rels: list[str], globs: list[str] = ()) -> tuple[list[str], list[str]]:
    """(present, skipped) relative paths: the listed files plus every eligible file under the listed glob dirs."""
    present: list[str] = []
    skipped: list[str] = []
    for rel in rels:
        p = d / rel
        if p.is_file():
            (present if _eligible(p) else skipped).append(rel)
    for pat in globs:
        for p in sorted(d.glob(pat)):
            if p.is_dir():
                for f in sorted(x for x in p.rglob("*") if x.is_file() and not any(q.startswith(KEPT_APART) for q in x.relative_to(p).parts)):
                    rel = f.relative_to(d).as_posix()
                    (present if _eligible(f) else skipped).append(rel)
            elif p.is_file():
                rel = p.relative_to(d).as_posix()
                (present if _eligible(p) else skipped).append(rel)
    seen: list[str] = []
    for r in present:
        if r not in seen:
            seen.append(r)
    return seen, sorted(set(skipped))


def _stage_into(d: Path, rels: list[str], into: Path) -> int:
    n = 0
    for rel in rels:
        src, dst = d / rel, into / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            n += 1
    return n


def _entry(name: str, kind: str, meta: dict[str, Any], run_dir: Path, d: Path, present: list[str], skipped: list[str],
           refs: dict[str, str] | None = None, optional: bool = False) -> dict[str, Any] | None:
    """One outputs.json entry.  No eligible file: None when `optional` (the entry is dropped), else an entry naming an absent tree so the
    publish fails by name (a stage that reported PASS must have left its account files)."""
    if not present:
        return None if optional else {"name": name, "kind": kind, "meta": {**meta, "skipped": skipped}, "tree": f"outputs/{name}", "refs": refs or {}}
    into = run_dir / "outputs" / name
    n = _stage_into(d, present, into)
    meta = {**meta, "files": len(present), "skipped": skipped, "evidence_dir": str(d)}
    return {"name": name, "kind": kind, "meta": meta, "tree": f"outputs/{name}", "refs": refs or {}, "staged": n}


def slim_outputs(d: Path, row: str, stage: str, run_dir: Path) -> list[dict[str, Any]]:
    """One `vllm-replay-slim/v1` entry per slim keep a decided replay left (`commit/replay_slim_p<pair>/`, `cpu_replay.keep`): every
    file, the .bin planes and their sizes included, linked into the run dir (copied where a link can't be made)."""
    out = []
    for k in sorted((d / "commit").glob("replay_slim_p*")):
        doc = _read_json(k / "slim.json")
        if not k.is_dir() or k.name.endswith(".partial") or not isinstance(doc, dict):
            continue
        into = run_dir / "outputs" / k.name
        for f in sorted(x for x in k.rglob("*") if x.is_file() and not x.is_symlink()):
            dst = into / f.relative_to(k)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy2(f, dst)
        out.append({"name": k.name, "kind": SLIM_KIND, "tree": f"outputs/{k.name}", "refs": {},
                    "meta": {"row": row, "stage": stage, **{x: doc.get(x) for x in SLIM_META}, "evidence_dir": str(d)}})
    return out


def build_outputs(d: Path, row: str, stage: str, run_dir: Path, inputs: dict[str, str]) -> dict[str, Any]:
    v = verdicts(d / "stages.txt")
    status = v.get(stage, (None, ""))[0]
    digests = program_digest(d, row)
    base_meta = {"row": row, "stage": stage, **digests, "stage_status": status}
    refs = {k: a for k, a in inputs.items() if k in ("build", "match") and isinstance(a, str) and a.startswith("art:")}
    entries: list[dict[str, Any]] = []
    passed = status == "PASS"                 # a failed stage still publishes its evidence + logs; its account artifact is then optional
    tp = digests.get("tp", 1) > 1
    if stage == "build":
        present, skipped = _collect(d, BUILD_FILES, ["build_request*", "build_step", *(TP_BUILD_GLOBS if tp else [])])
        present = [r for r in present if not r.endswith("instances.json.gz")]
        meta = {**base_meta, "required_manifest": (d / "manifest.json").is_file(), "precheck": v.get("precheck", (None, ""))[0]}
        e = _entry("build", KINDS[stage], meta, run_dir, d, present, skipped, optional=not passed)
    elif stage == "plan":
        present, skipped = _collect(d, PLAN_FILES)
        plan = _read_json(d / "plan" / "commit_plan.json")
        meta = {**base_meta, "plan_key": plan.get("key") if isinstance(plan, dict) else None}
        e = _entry("plan", KINDS[stage], meta, run_dir, d, present, skipped, refs={k: refs[k] for k in ("build",) if k in refs}, optional=not passed)
    elif stage == "match":
        present, skipped = _collect(d, MATCH_FILES)
        gm = _read_json(d / "global_match.json") or {}
        meta = {**base_meta, "global_match_ok": gm.get("verdict") == "PASS" if isinstance(gm, dict) and "verdict" in gm else None,
                "within_step": gm.get("within_step") if isinstance(gm, dict) else None}
        if tp:
            tm = _read_json(d / "match" / "tp2_match.json") or {}
            meta["tp_match_ok"] = tm.get("pass") if isinstance(tm, dict) else None
        e = _entry("match", KINDS[stage], meta, run_dir, d, present, skipped, refs={k: refs[k] for k in ("build",) if k in refs}, optional=not passed)
    elif stage == "commit":
        present, skipped = _collect(d, COMMIT_FILES, ["commit"])
        # the Commit's decision document: verdict.json (vllm-verdict/v1) at TP1; tp.commit's commit/summary.json on a TP row (no
        # vllm-verdict/v1 grades a tp.commit record: commit.verdict reads commit_delta's documents).  A PASS stage without it is refused
        # by name; a stage that did not pass records the absence (`decision_record_missing`), never an empty document.  A config run
        # (a `config` verdict line: no Match, no verdict of record) decides in config_record.json, at TP1 and TP alike
        config = "config" in v
        rel = "config_record.json" if config else ("commit/summary.json" if tp else "verdict.json")
        doc = _read_json(d / rel)
        if not isinstance(doc, dict):
            if passed:
                raise ValueError(f"commit stage PASS but its decision document {rel} is absent or unreadable under {d}")
            doc = None
        vd = _read_json(d / "commit" / "verdict.json") or {}          # commit_delta's own decision document (TP1)
        meta = {**base_meta, "verdict": (vd.get("verdict") if isinstance(vd, dict) else None) or status, "decision_record": rel}
        if doc is None:
            meta["decision_record_missing"] = rel
        elif config:
            rp = doc.get("replay") or {}
            meta.update(outcome=doc.get("outcome"), why=doc.get("why"), replay_equal=rp.get("equal"), replay_k=rp.get("k"))
        elif tp:
            meta.update(tp_pass=doc.get("pass"), tokens_equal=doc.get("tokens_equal"), pairs=doc.get("pairs"))
        else:
            runs_ = doc.get("runs") or {}
            meta.update(outcome=doc.get("outcome"),
                        n_runs=(int(runs_.get("instrumented") or 0) + int(runs_.get("control") or 0)) if runs_ else None,
                        runs_failed=runs_.get("failed") if runs_ else None)
        e = _entry("verdict", KINDS[stage], meta, run_dir, d, present, skipped, refs=refs, optional=not passed)
    elif stage == "replay":
        # a deferred C2, decided on CPU, beside the rewritten Commit record: a config run's decision document is config_record.json (TP1
        # and TP); any other TP row's is commit/summary.json as tp-commit-replay folded the ranks' replays into it
        rel = "commit/summary.json" if tp and "config" not in v else "config_record.json"
        present, skipped = _collect(d, ["config_record.json", *COMMIT_FILES], ["commit"])
        doc = _read_json(d / rel)
        if not isinstance(doc, dict) and passed:
            raise ValueError(f"replay stage PASS but its decision document {rel} is absent or unreadable under {d}")
        meta = {**base_meta, "verdict": status, "decision_record": rel}
        if not isinstance(doc, dict):
            meta["decision_record_missing"] = rel
        elif rel == "config_record.json":
            meta["outcome"] = doc.get("outcome")
        else:
            meta.update(tp_pass=doc.get("pass"), tokens_equal=doc.get("tokens_equal"), pairs=doc.get("pairs"))
        e = _entry("verdict", KINDS[stage], meta, run_dir, d, present, skipped, refs=refs, optional=not passed)
    else:
        raise ValueError(f"unknown stage {stage!r}")
    if e is not None:
        entries.append(e)
    if stage == "replay":
        entries.extend(slim_outputs(d, row, stage, run_dir))
    present, skipped = _collect(d, EVIDENCE_FILES)
    e = _entry("evidence", "execution-evidence/v1", {"row": row, "stage": stage, "stage_status": status}, run_dir, d, present, skipped, optional=True)
    if e is not None:
        entries.append(e)
    present, skipped = _collect(d, ["row.log"], ["*.log", *(TP_LOG_GLOBS if tp else [])])
    e = _entry("logs", "run-files/v1", {"row": row, "stage": stage}, run_dir, d, present, skipped, optional=True)
    if e is not None:
        entries.append(e)
    for e in entries:
        e.pop("staged", None)
        if not e["refs"]:
            e.pop("refs")
    return {"schema": SCHEMA, "outputs": entries, "row": row, "stage": stage, "evidence_dir": str(d), "inputs": refs}


def inputs_from_job(run_dir: Path) -> dict[str, str]:
    job = _read_json(run_dir / "job.json") or {}
    ins = ((job.get("store") or {}).get("inputs") or {}) if isinstance(job, dict) else {}
    return {k: v for k, v in ins.items() if isinstance(v, str) and v.startswith("art:")}


@dataclass(frozen=True, kw_only=True)
class WriteOptions:
    """Declare the stage's outputs in RUN_DIR/outputs.json."""

    evidence_dir: str = option("evidence_dir")
    row: str = option("row")
    stage: str = option("stage", choices=sorted(KINDS))
    run_dir: str | None = option("--run-dir", env="RESEARCH_RUN_DIR", default=None)
    input: list[str] = option("--input", action="append", default=[], metavar="NAME=art:...",
                              help="an input artifact (default: job.json store.inputs)")
    cmd: str = field(default="write", init=False)


@dataclass(frozen=True, kw_only=True)
class DigestOptions:
    """Print the Program digest of record (build_summary.json), or an empty line."""

    evidence_dir: str = option("evidence_dir")
    row: str = option("row")
    cmd: str = field(default="digest", init=False)


COMMANDS = {"write": WriteOptions, "digest": DigestOptions}


def main(a: WriteOptions | DigestOptions) -> int:
    if a.cmd == "digest":
        print(program_digest(Path(a.evidence_dir), a.row).get("program_digest") or "")
        return 0
    if not a.run_dir:
        print("research_outputs: RESEARCH_RUN_DIR not set; nothing written", file=sys.stderr)
        return 0
    run_dir = Path(a.run_dir)
    try:
        inputs = inputs_from_job(run_dir)
        for spec in a.input:
            k, _, v = spec.partition("=")
            if k and re.fullmatch(r"art:[0-9a-f]{64}", v or ""):
                inputs[k] = v
        doc = build_outputs(Path(a.evidence_dir), a.row, a.stage, run_dir, inputs)
        tmp = run_dir / "outputs.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
        os.replace(tmp, run_dir / "outputs.json")
        names = ", ".join(f"{e['name']}={e['kind']}({e['meta'].get('files', 0)} files)" for e in doc["outputs"])
        print(f"research_outputs: {run_dir / 'outputs.json'}: {names}")
    except Exception as e:  # noqa: BLE001 -- never mask the stage's own exit status
        print(f"research_outputs: FAILED to write outputs.json: {type(e).__name__}: {e}", file=sys.stderr)
    return 0

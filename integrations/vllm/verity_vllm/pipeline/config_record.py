"""The config record (`verity-vllm/config-record/v1`): what a row's run keeps, and the circuit database's record of the row.

A config run is one job over one vLLM configuration: the Build (the Programs and the `Q_word` manifest, with the strict word check), ONE
instrumented Commit (no pairs; a control arm only as the slowdown's baseline) and a k-unit random replay.  Its record is small (kB) and
keyed by the config (the row id): the Build's Program digests and manifest, the word check, the Commit's run root, openings and verdict,
the replay's result with the units that differed, and the timing per arm.  Beside each request Program's digest is its census/v1 digest
(`census_digests`), the census the Build cached: a function of the Program alone, so it adds no trust, and a reader binds the census
and the memory map's structure to the Program without recomputing them.  Everything is read from the files the stages write in the row
dir; nothing here runs a stage.  A TP row's record has the same parts, each with `per_rank` beside the total: the ranks' Program and
workload digests, run roots, openings and replays, the global manifest, the combined run root and the replay summed over the ranks.

A row of record (Build, Match, Commit) keeps the same record, with the Match (`match`) and the Commit read from that Commit's own
summary (`summary_part`, as `verity_vllm.records.publish` passes it), since a later run rewrites the row dir's; a single-GPU Commit's
summary decides nothing, so its Commit is its decision document, the row dir's `verdict.json` (`single_gpu_commit_part`).  Every
record states the row's config (`row`: its file without the requests, the file's sha256 and the κ profile) and, once published,
`evidence`: the art: ids of the Programs per rank, the manifest, the Match and Commit records and the conformance captures
(`verity_vllm.records`).

`stages` is each stage's last verdict in the row dir's `stages.txt`.  The Build's own (`build.stage`) is its status and the in-stage
manifest's exit code; a Build whose Programs passed while that manifest alone failed is superseded by the manifest of record
(`evidence.manifest`, `build.stage.superseded_by`), and passes only so.

`served` is the switches that selected the Commit's kernels, as each serving process read them back, against what the row's config
declares (`engine.served`, from the Commit's summary.json).

Outcome PASS: the Build passed, the word check passed (a config run) or the Match passed (a row of record), the Commit verdict passed
(`check.verdict.commit_verdict`, which includes the replay), the replay evaluated k units, all equal, and the served switches passed (a
Commit that read none back is not run).  Anything else is FAIL with the first reason; a failure is a finding of the sweep, not a stop.
Served switches that failed come before the Commit: the Commit fails closed on them before it runs.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from verity_vllm.config import RECORDS, WORKLOADS, RowConfig, RowId, UsageError, done_source_refusal, workload_file
from verity_vllm.engine import served as SV
from verity_vllm.pipeline.research_outputs import verdicts

SCHEMA = "verity-vllm/config-record/v1"
NAME = "config_record.json"
#: The evidence store's kind of a published record (`verity_vllm.records`).
KIND = "vllm-config-record/v1"
MANIFEST_RC = re.compile(r"\bmanifest_rc=(-?\d+)\b")
INDEX = RECORDS
INDEX_SCHEMA = "verity-vllm/records-index/v1"
#: The tiers of a row's rebuild (`pipeline.neutrality.TIER_FLAGS`).
TIERS = ("fast", "heavy")
INDEX_ABOUT = ("Per row of record: `record`, its config record's art: id (`verity-vllm records publish`; null for a row pinned "
               "before it has one), and its circuit's pins: every rank's Program digests (step, each request shape, GP-01's workload) "
               "as the Build `pinned_from` names made them (the record, when they are its Programs), the manifest digest and the tier "
               "of its rebuild.  `verity-vllm neutrality` rebuilds a row and compares; the PR that changes a circuit on purpose "
               "re-pins it (`verity-vllm neutrality ROW --update`) and re-publishes its record.  A switched row's `aot` names the "
               "comparison run whose predicted Programs equal its traced Build's on every rank (`verity-vllm predict compare`) and "
               "that traced Build (`trace`): its record path's Build is the AOT Build (`verity_vllm.program.predict.aot`), its pins the "
               "predicted Programs, and its rebuild the trace.")


def load_index(path: Path) -> dict[str, Any]:
    """The circuit database's index (`manifests/records.json`, `INDEX_SCHEMA`, `INDEX_ABOUT`): `rows`, per row of record its
    `record` and its circuit's pins (`programs` per rank as `epoch.build_programs` reads them, `manifest_digest`, `tier`,
    `pinned_from`), which `verity_vllm.records` publishes and `verity-vllm neutrality` compares.  No file is an empty index."""
    if not path.is_file():
        return {"schema": INDEX_SCHEMA, "about": INDEX_ABOUT, "rows": {}}
    doc = json.loads(path.read_text())
    if doc.get("schema") != INDEX_SCHEMA:
        raise UsageError(f"{path}: schema {doc.get('schema')!r}, not {INDEX_SCHEMA}")
    return doc


def write_index(doc: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps({**doc, "about": INDEX_ABOUT, "rows": dict(sorted(doc["rows"].items()))}, indent=1) + "\n")


def aot_of(row: str, path: Path = INDEX) -> dict[str, Any] | None:
    """A switched row's `aot` (`comparison`: the run that found its predicted Programs equal to its traced Build's on every rank;
    `trace`: that Build, a `vllm-build/v1`), None for a row on the trace."""
    return (load_index(path)["rows"].get(row) or {}).get("aot")


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _jsonl(p: Path) -> list[dict]:
    try:
        return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    except (OSError, ValueError):
        return []


def config_of(row: str) -> dict[str, Any]:
    """The configuration a row id names (`config.RowId`), under the sweep plan's names."""
    try:
        r = RowId.parse(row)
    except ValueError:
        return {"row": row}
    return {"row": row, "model": r.model, "dtype": r.dtype, "gpu": r.target, "tp": r.world, "batch": r.batch, "input": r.max_input,
            "output": r.max_output, "arrivals": r.arrivals, "sampling": r.sampling, "engine": r.execution}


def row_part(row: str, profile: str | None = None) -> dict[str, Any] | None:
    """The row's config (`config.RowConfig`, its file without the requests), the file's sha256 and the κ profile its Match folds with
    (`profile`, else the role's: `observe.fold.profiles.generic.module_name`).  None for a row without a file."""
    p = workload_file(WORKLOADS / f"{row}.json")
    if not p.is_file():
        return None
    cfg = RowConfig.load(p)
    if profile is None and cfg.case:
        from verity_vllm.observe.fold.profiles.generic import module_name
        profile = module_name(cfg.case, int(config_of(row).get("tp") or 1))
    return {"file": f"workloads/{row}.json", "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "profile": profile,
            "config": {k: v for k, v in cfg.to_json().items() if k != "requests"}}


def index_refusal(row: str) -> str | None:
    """Why `INDEX` may not list `row`: a row it lists is a row of record, so its file must be done only at its public cap
    (`config.done_source_refusal`, with `of_record`); None when it is.  `records publish`, `pull` and `switch` refuse with it before
    they touch the index, and `row chain` before any stage when it would publish."""
    p = workload_file(WORKLOADS / f"{row}.json")
    return done_source_refusal(json.loads(p.read_text()) if p.is_file() else {"sweep": {"row_id": row}}, of_record=True)


def match_part(d: Path, world: int) -> dict[str, Any] | None:
    """The Match's verdict: a TP row's fold Match (`match/fold_match/tp2_fold_match.json`: per-rank fold, compare and GM-01, the
    cross-rank check, the profile it folded with), a single-GPU row's global Match (`global_match.json`, GM-01's ``verdict``).  None when
    no Match ran."""
    if world > 1:
        fm = _json(d / "match" / "fold_match" / "tp2_fold_match.json")
        if not fm:
            return None
        return {"ok": fm.get("pass") is True, "schema": fm.get("schema"), "profile": fm.get("profile"), "world": fm.get("world"),
                "legs": fm.get("legs"), "problems": (fm.get("problems") or [])[:16]}
    gm = _json(d / "global_match.json")
    return {"ok": gm.get("verdict") == "PASS", "schema": gm.get("schema")} if isinstance(gm, dict) else None


SUMMARY_GATES = ("weights_pin", "boundary_linkage", "cross_rank_collectives", "fold_match_binding", "query_population", "oracle_compare",
                 "value_check")


def _gate(v: Any) -> bool | None:
    if not isinstance(v, dict):
        return v if isinstance(v, bool) else None
    return v["result"] if isinstance(v.get("result"), bool) else (v["verdict"] == "PASS" if "verdict" in v else None)


def summary_part(sm: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(commit, replay) of a Commit of record from its summary (`commit/summary.json`, every pair): its pass, each pair's verdict and
    run root (per rank too), the openings, each gate's result, the walls and the required manifest it committed against."""
    rm = sm.get("required_manifest") or {}
    roots = sm.get("tp_run_roots") or sm.get("run_roots") or []
    gates = {k: _gate(sm.get(k)) for k in SUMMARY_GATES if k in sm}
    if isinstance(sm.get("match_oracle"), dict):
        gates["match_oracle"] = all(_gate(v) is True for v in sm["match_oracle"].values()) and bool(sm["match_oracle"])
    per_pair = [p for p in sm.get("per_rank_roots") or [] if isinstance(p, dict)]
    sr = sm.get("sampled_replay") or {}
    commit = {"ok": sm.get("pass") is True, "pairs": sm.get("pairs"), "pass_per_pair": sm.get("commit_pass"), "run_root": roots[0] if roots
              and len(set(roots)) == 1 else None, "run_roots": roots, "openings": sm.get("openings"), "tokens_equal": sm.get("tokens_equal"),
              "gates": gates, "manifest": {k: rm.get(k) for k in ("manifest_digest", "identities", "program_digest_per_rank")},
              "wall": {k: sm.get(k) for k in ("control_wall", "treatment_wall", "delta_wall_pct")}, "schema": sm.get("schema"),
              "per_rank": {r: {"run_roots": [p.get(r) for p in per_pair]} for r in sorted({r for p in per_pair for r in p}, key=int)}
              if per_pair else None}
    smp = sr.get("sample") or {}
    replay = {"ok": sr.get("result") is True and int(smp.get("mismatch_n") or 0) == 0, "result": sr.get("result"),
              **{k: smp.get(k) for k in ("picked", "evaluated", "equal", "mismatch_n", "not_evaluated_n")},
              "population": (sr.get("population") or {}).get("by_family"),
              "per_rank": {r: {k: v.get(k) for k in ("result", "grade")} for r, v in sorted((sr.get("per_rank") or {}).items(), key=lambda kv: int(kv[0]))
                           if isinstance(v, dict)} or None}
    return commit, replay


def stages_part(d: Path) -> dict[str, dict[str, str]]:
    """Each stage's last verdict in the row dir's `stages.txt`: stage -> {status, line}."""
    return {s: {"status": st, "line": rest[:240]} for s, (st, rest) in verdicts(d / "stages.txt").items()}


def build_stage(stage: dict[str, str] | None, programs_ok: bool, manifest: str | None) -> dict[str, Any] | None:
    """The Build stage's verdict: its status, the in-stage manifest's exit code (`manifest_rc`, a TP Build's; None when the Programs
    failed before it ran) and, when the Programs passed and that manifest alone failed, `superseded_by`: the manifest of record
    (`manifest`).  None when the row dir has no Build verdict."""
    if stage is None:
        return None
    m = MANIFEST_RC.search(stage["line"])
    out: dict[str, Any] = {"status": stage["status"], "manifest_rc": int(m.group(1)) if m else None}
    if out["status"] != "PASS" and programs_ok and out["manifest_rc"] not in (None, 0) and manifest:
        out["superseded_by"] = manifest
    return out


def census_digests(parts: dict[str, Any]) -> dict[str, str]:
    """Each Build part's census/v1 digest beside its Program digest (`row_records.census_digest`): the request Programs'."""
    return {k: v["census_digest"] for k, v in sorted(parts.items()) if isinstance(v, dict) and v.get("census_digest")}


def build_part(d: Path) -> dict[str, Any]:
    bs = _json(d / "build_summary.json") or {}
    programs = {k: (v or {}).get("program_digest") for k, v in sorted(bs.items()) if k not in ("workload",) and isinstance(v, dict)}
    man = _json(d / "manifest_header.json") or _json(d / "manifest.json") or {}
    q = man.get("query") or {}
    return {"ok": bool(bs) and all((v or {}).get("ok") for v in bs.values() if isinstance(v, dict)),
            "programs": programs, "census_digests": census_digests(bs), "workload_digest": (bs.get("workload") or {}).get("workload_digest"),
            "manifest_digest": man.get("manifest_digest"), "query_id": q.get("query_id")}


def word_check_part(d: Path) -> dict[str, Any]:
    w = _json(d / "strict_word.json") or {}
    return {k: w.get(k) for k in ("ok", "check", "word_lines", "components", "calls", "units", "gates", "problems")} if w else {"ok": None}


def commit_part(d: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """(commit, replay) of the instrumented run in `commit/runs.jsonl` (the config run has one)."""
    runs = [r for r in _jsonl(d / "commit" / "runs.jsonl") if r.get("arm") != "control"]
    if not runs:
        return {"ok": None, "runs": 0}, {"ok": None}
    r = runs[-1]
    c = r.get("commit") or {}
    ops = c.get("openings") or {}
    commit = {"ok": bool(c.get("commit_pass")), "reason": c.get("commit_fail_reason"), "runs": len(runs), "run_root": c.get("run_root"),
              "openings": {"n": ops.get("n"), "verified": ops.get("verified")},
              "tensors": {"expected": c.get("expected_tensors"), "committed": c.get("committed_tensors")},
              "coverage_ok": (c.get("coverage") or {}).get("ok"), "wall_s": r.get("wall_s"), "engine_steps": r.get("engine_steps"),
              "host_mem_kb": r.get("host_mem_kb")}
    if c.get("weights_attest"):
        commit["weights_attest"] = c["weights_attest"]
    return commit, replay_of(c.get("sampled_replay") or {})


def single_gpu_commit_part(d: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """(commit, replay) of a single-GPU Commit of record: its decision document (`verdict.json`, vllm-verdict/v1, over every pair and
    the replay) decides it and names the manifest it committed against; its last instrumented run (`commit_part`) the rest."""
    commit, replay = commit_part(d)
    v = _json(d / "verdict.json") or {}
    commit.update(ok=v.get("outcome") == "PASS", outcome=v.get("outcome"), manifest={"manifest_digest": v.get("manifest_digest")})
    return commit, replay


def replay_of(sr: dict[str, Any]) -> dict[str, Any]:
    """One sampled replay's result (`check.replay.driver.sampled_replay`), with the units that differed."""
    smp = sr.get("sample") or {}
    mism = sr.get("mismatch") or []
    return {"ok": sr.get("result") is True and int(smp.get("mismatch_n") or 0) == 0, "result": sr.get("result"), "grade": sr.get("grade"),
            "strata_by": smp.get("strata_by"), "k": smp.get("k"), "picked": smp.get("picked"), "evaluated": smp.get("evaluated"),
            "equal": smp.get("equal"), "mismatch_n": smp.get("mismatch_n"), "not_evaluated_n": smp.get("not_evaluated_n"),
            "seed": smp.get("seed"), "seed_source": smp.get("seed_source"),
            "differed": [{k: m.get(k) for k in ("i", "family", "request_id", "step", "op_path") if k in m} for m in mism[:64]],
            "by_family": (sr.get("mismatch_by") or {}).get("by_family"), "why": sr.get("why")}


def tp_build_part(d: Path, world: int) -> dict[str, Any]:
    """A TP row's Build (`row_records.tp_build_summary`): per rank its Program digests and GP-01 workload digest, and the global manifest."""
    ranks = (_json(d / "build_summary.json") or {}).get("ranks") or {}
    per = {}
    for r in range(world):
        rk = ranks.get(f"rank{r}") or {}
        per[str(r)] = {"ok": bool(rk) and all((v or {}).get("ok") for v in rk.values() if isinstance(v, dict)),
                       "programs": {k: (v or {}).get("program_digest") for k, v in sorted(rk.items()) if k != "workload" and isinstance(v, dict)},
                       "census_digests": census_digests(rk), "workload_digest": (rk.get("workload") or {}).get("workload_digest")}
    man = _json(d / "manifest_header.json") or _json(d / "manifest.json") or {}
    return {"ok": all(p["ok"] for p in per.values()), "world": world, "per_rank": per, "manifest_digest": man.get("manifest_digest"),
            "query_id": (man.get("query") or {}).get("query_id"), "program_digest_per_rank": man.get("program_digest_per_rank")}


def _sum(parts: list[dict[str, Any]], key: str) -> int | None:
    vals = [p.get(key) for p in parts]
    return None if any(v is None for v in vals) else sum(int(v) for v in vals)


def tp_commit_part(d: Path, world: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """(commit, replay) of a TP row's instrumented run in `commit/runs.jsonl` (`tp-commit --only-arm instrumented`): per rank its run root,
    openings, coverage and replay, and in total the combined run root and the replay summed over the ranks."""
    runs = [r for r in _jsonl(d / "commit" / "runs.jsonl") if r.get("arm") != "control"]
    if not runs:
        return {"ok": None, "runs": 0}, {"ok": None}
    r = runs[-1]
    c = r.get("commit") or {}
    ranks = sorted((c.get("per_rank") or {}).items(), key=lambda kv: int(kv[0]))
    per_c, per_r = {}, {}
    for rk, b in ranks:
        ops, ex = b.get("openings") or {}, b.get("extra") or {}
        per_c[str(rk)] = {"ok": b.get("commit_pass") is True, "reason": None if b.get("commit_pass") else b.get("commit_fail_reason"),
                          "run_root": b.get("run_root"), "openings": {"n": ops.get("n"), "verified": ops.get("verified")},
                          "tensors": {"expected": ex.get("expected_tensors"), "committed": ex.get("committed_tensors")},
                          "coverage_ok": (((b.get("binding") or {}).get("coverage")) or {}).get("ok")}
        per_r[str(rk)] = replay_of(b.get("sampled_replay") or {})
    ok = c.get("commit_pass") is True
    ops = c.get("openings") or {}
    commit = {"ok": ok, "reason": None if ok else c.get("commit_fail_reason"), "runs": len(runs), "run_root": c.get("tp_run_root"),
              "openings": {"n": ops.get("n"), "verified": ops.get("verified")}, "tensors": {"committed": c.get("tensors")},
              "coverage_ok": all(p["coverage_ok"] is True for p in per_c.values()) if per_c else None, "wall_s": r.get("wall_s"),
              "engine_steps": r.get("engine_steps"), "per_rank": per_c}
    rs = list(per_r.values())
    complete = len(rs) == world
    replay = {"ok": complete and all(p["ok"] for p in rs), "result": complete and all(p["result"] is True for p in rs),
              "grade": "/".join(str(p["grade"]) for p in rs), "strata_by": rs[0]["strata_by"] if rs else None,
              **{k: _sum(rs, k) for k in ("k", "picked", "evaluated", "equal", "mismatch_n", "not_evaluated_n")},
              "differed": [dict(m, rank=int(rk)) for rk, p in per_r.items() for m in p["differed"]][:64],
              "by_family": {f: sum(int((p["by_family"] or {}).get(f) or 0) for p in rs) for f in sorted({f for p in rs for f in (p["by_family"] or {})})},
              "why": "; ".join(f"rank {rk}: {p['why']}" for rk, p in per_r.items() if p.get("why")) or None, "per_rank": per_r}
    return commit, replay


ENGINE_KEYS = ("VLLM_BATCH_INVARIANT", "CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE", "vllm_batch_invariant_mode",
               "torch_allow_bf16_reduced_precision_reduction", "torch_preferred_blas_library", "compute_capability")


def engine_part(d: Path) -> dict[str, Any] | None:
    """The batch-invariant configuration the instrumented run's replay measured in the engine's process (`engine_of_record`): the
    cuBLAS(Lt) workspace pin its kernel selection is conditioned on, among the rest.  None when the Commit recorded no replay."""
    runs = [r for r in _jsonl(d / "commit" / "runs.jsonl") if r.get("arm") != "control"]
    eor = (((runs[-1].get("commit") or {}).get("sampled_replay") or {}).get("engine_of_record") if runs else None) or None
    return {k: eor.get(k) for k in ENGINE_KEYS} if eor else None


def timing_part(d: Path) -> dict[str, Any]:
    """Per arm, the run's wall and its engine steps' prefill (the first step) and mean decode time; with the control arm (the config
    run's `--config-baseline 1`), the extraction slowdown per phase: instrumented / control."""
    arms = {r.get("arm"): r for r in _jsonl(d / "commit" / "runs.jsonl")}
    t = {a: {"wall_s": r.get("wall_s"), "prefill_s": r.get("step_wall_prefill_s"), "decode_mean_s": r.get("step_wall_decode_mean_s"),
             "engine_steps": r.get("engine_steps")} for a, r in sorted(arms.items()) if a in ("control", "instrumented")}
    i, c = t.get("instrumented") or {}, t.get("control") or {}
    t["slowdown"] = {k: (i[f] / c[f] if i.get(f) and c.get(f) else None)
                     for k, f in (("prefill", "prefill_s"), ("decode", "decode_mean_s"), ("wall", "wall_s"))} if c else None
    return t


def record(d: str | Path, row: str, *, tree: str | None = None, device: dict[str, Any] | None = None,
           summary: dict[str, Any] | None = None, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """The config record of the run in row dir `d` (a TP row's per rank and in total); with `summary`, a Commit of record's
    (`summary_part`: the row dir's own `commit/` is then not read), and `evidence`, the art: ids the record cites."""
    d = Path(d)
    world = int(config_of(row).get("tp") or 1)
    build, word, match = (tp_build_part(d, world) if world > 1 else build_part(d)), word_check_part(d), match_part(d, world)
    stages, programs_ok = stages_part(d), build["ok"]
    stage = build["stage"] = build_stage(stages.get("build"), programs_ok, (evidence or {}).get("manifest"))
    build["ok"] = programs_ok and (stage is None or stage["status"] == "PASS" or "superseded_by" in stage)
    if summary is not None:
        (commit, replay), engine, timing = summary_part(summary) if world > 1 else single_gpu_commit_part(d), None, {}
    else:
        commit, replay = tp_commit_part(d, world) if world > 1 else commit_part(d)
        engine, timing = engine_part(d), timing_part(d)
    served = SV.of_summary(summary if summary is not None else _json(d / "commit" / "summary.json"))
    gate = ("match", match["ok"]) if match is not None else ("word check", word.get("ok"))
    switches = ("served switches", (served or {}).get("ok"))
    checks = (("build", build["ok"]), gate, *((switches,) if switches[1] is False else ()), ("commit", commit.get("ok")),
              ("replay", replay.get("ok")), switches)
    why = next((f"{name} {'FAIL' if ok is False else 'not run'}" for name, ok in checks if ok is not True), None)
    if why == "served switches FAIL":
        why = f"served switches FAIL: {'; '.join(served.get('problems') or [])}"
    if why == "build FAIL" and programs_ok and stage and stage["manifest_rc"] not in (None, 0):
        why = f"build FAIL: the in-stage manifest exited {stage['manifest_rc']} and no manifest of record supersedes it (evidence.manifest)"
    if why == "commit FAIL" and commit.get("reason"):
        why = f"commit FAIL: {commit['reason']}"
    return {"schema": SCHEMA, "config": config_of(row), "row": row_part(row, (match or {}).get("profile")), "tree": tree, "device": device,
            "outcome": "PASS" if why is None else "FAIL", "why": why, "build": build, "word_check": word, "match": match, "commit": commit,
            "replay": replay, "engine": engine, "served": served, "timing": timing, "stages": stages, "evidence": evidence or {}}


def write(d: str | Path, row: str, **kw: Any) -> dict[str, Any]:
    rec = record(d, row, **kw)
    (Path(d) / NAME).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    return rec

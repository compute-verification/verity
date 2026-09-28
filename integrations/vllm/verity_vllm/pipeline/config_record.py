"""The config record (`verity-vllm/config-record/v1`): what a config run keeps (docs/vllm-config-sweep-plan.md §2 step 4, §5).

A config run is one job over one vLLM configuration: the Build (the Programs and the `Q_word` manifest, with the strict word check), ONE
instrumented Commit (no control arm, no pairs) and a k-unit random replay.  Its record is small (kB) and keyed by the config (the row id):
the Build's Program digests and manifest, the word check, the Commit's run root, openings and verdict, and the replay's result with the
units that differed.  Everything is read from the files the stages write in the row dir; nothing here runs a stage.

Outcome PASS: the Build passed, the word check passed, the instrumented run's Commit verdict passed (`check.verdict.commit_verdict`, which
includes the replay) and the replay evaluated k units, all equal.  Anything else is FAIL with the first reason; a failure is a finding of
the sweep, not a stop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verity_vllm.config import RowId

SCHEMA = "verity-vllm/config-record/v1"
NAME = "config_record.json"


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


def build_part(d: Path) -> dict[str, Any]:
    bs = _json(d / "build_summary.json") or {}
    programs = {k: (v or {}).get("program_digest") for k, v in sorted(bs.items()) if k not in ("workload",) and isinstance(v, dict)}
    man = _json(d / "manifest_header.json") or _json(d / "manifest.json") or {}
    q = man.get("query") or {}
    return {"ok": bool(bs) and all((v or {}).get("ok") for v in bs.values() if isinstance(v, dict)),
            "programs": programs, "workload_digest": (bs.get("workload") or {}).get("workload_digest"),
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
    sr = c.get("sampled_replay") or {}
    smp = sr.get("sample") or {}
    mism = sr.get("mismatch") or []
    replay = {"ok": sr.get("result") is True and int(smp.get("mismatch_n") or 0) == 0, "result": sr.get("result"), "grade": sr.get("grade"),
              "strata_by": smp.get("strata_by"), "k": smp.get("k"), "picked": smp.get("picked"), "evaluated": smp.get("evaluated"),
              "equal": smp.get("equal"), "mismatch_n": smp.get("mismatch_n"), "not_evaluated_n": smp.get("not_evaluated_n"),
              "seed": smp.get("seed"), "seed_source": smp.get("seed_source"),
              "differed": [{k: m.get(k) for k in ("i", "family", "request_id", "step", "op_path") if k in m} for m in mism[:64]],
              "by_family": (sr.get("mismatch_by") or {}).get("by_family"), "why": sr.get("why")}
    return commit, replay


def record(d: str | Path, row: str, *, tree: str | None = None, device: dict[str, Any] | None = None,
           stages: dict[str, Any] | None = None) -> dict[str, Any]:
    """The config record of the run in row dir `d`."""
    d = Path(d)
    build, word = build_part(d), word_check_part(d)
    commit, replay = commit_part(d)
    checks = (("build", build["ok"]), ("word check", word.get("ok")), ("commit", commit.get("ok")), ("replay", replay.get("ok")))
    why = next((f"{name} {'FAIL' if ok is False else 'not run'}" for name, ok in checks if ok is not True), None)
    if why == "commit FAIL" and commit.get("reason"):
        why = f"commit FAIL: {commit['reason']}"
    return {"schema": SCHEMA, "config": config_of(row), "tree": tree, "device": device, "outcome": "PASS" if why is None else "FAIL",
            "why": why, "build": build, "word_check": word, "commit": commit, "replay": replay, "stages": stages or {}}


def write(d: str | Path, row: str, **kw: Any) -> dict[str, Any]:
    rec = record(d, row, **kw)
    (Path(d) / NAME).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    return rec

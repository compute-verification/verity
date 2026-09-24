"""Run ids and the local run registry.

A run is one recorded execution attempt.  Its directory is the telemetry attempt dir (job.json, status.json, events.jsonl,
resources.jsonl, failure.json, stdout.log, stderr.log, and the workload's result.json) and nothing else is needed to describe
it.  The registry is the directory that holds run directories: an index, not a second source of truth.  Reading a run needs no
service and no credentials.  Everything derived (result validity, cost estimate) is computed from the files at read time.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import record, result

DEFAULT_ROOT = Path.home() / ".research" / "runs"
TERMINAL = ("done", "failed", "cancelled")
ACTIVE = ("submitted", "admitted", "running", "preserving")


def runs_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("RESEARCH_RUNS", DEFAULT_ROOT)).expanduser()


def new_run_id(now: float | None = None) -> str:
    """Sortable, human-typable: r20260921-171502-3f9a.  Identifies this attempt; two identical repetitions get two ids."""
    t = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc)
    return f"r{t:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def summarize(run_dir: Path) -> dict[str, Any]:
    """One flat record per run for listings and inspect.  Missing files are reported, not raised: a run that died before
    writing status.json is still a run.  Sections a v0.1 record lacks are None (unknown)."""
    job = record.read_job(run_dir)
    raw_job = job or (_read(run_dir / "job.json") or {})
    status = _read(run_dir / "status.json") or {}
    failure = _read(run_dir / "failure.json") or {}
    own = raw_job.get("ownership") or {}
    comp = raw_job.get("computation") or {}
    exe = raw_job.get("execution") or {}
    run_sec = (job or {}).get("run") or {}
    transitions = status.get("transitions") or []
    started = next((t.get("t_utc") for t in transitions if t.get("state") == "running"), None)
    ended = transitions[-1].get("t_utc") if transitions and transitions[-1].get("state") in TERMINAL else None
    run_id = run_sec.get("id") or own.get("job") or run_dir.name
    res = result.check(run_dir, run_id)
    manifest = _read(run_dir / "preserved.json")
    remote_rec = _read(run_dir / "remote.json")
    if not status and (run_dir / "refused.json").exists():
        state = "refused"                        # the machine refused it (launch-request, or the runner before the workload)
    elif remote_rec and not status:
        phase = remote_rec.get("phase")
        state = {"launched": "submitted", "refused": "refused", "unknown": "unknown"}.get(phase, "submitted")
    else:
        state = status.get("state", "unknown" if not raw_job else "submitted")
    if remote_rec and not raw_job:
        # not fetched yet: what the control side knew at launch
        own = {"project": remote_rec.get("project")}
        stage = next((remote_rec["launch_args"][i + 1] for i in range(len(remote_rec.get("launch_args") or []) - 1)
                      if remote_rec["launch_args"][i] == "--stage"), None)
        comp = {"stage": stage, "tree_sha": remote_rec.get("source_sha")}
        exe = {"command": remote_rec.get("command")}
    return {
        "id": run_dir.name,
        "run_id": run_id,
        "dir": str(run_dir),
        "schema": raw_job.get("schema"),
        "readable": job is not None or not raw_job,
        "project": own.get("project"),
        "job": own.get("job"),
        "lane": own.get("lane"),
        "stage": comp.get("stage"),
        "tree_sha": comp.get("tree_sha"),
        "workload_digest": comp.get("workload_digest"),
        "state": state,
        "rc": status.get("rc"),
        "signal": status.get("signal"),
        "wall_s": status.get("wall_s"),
        "class": failure.get("class"),
        "certainty": failure.get("certainty"),
        "why": failure.get("why"),
        "started_utc": started,
        "ended_utc": ended,
        "command": exe.get("command"),
        "remote": run_sec.get("remote") or ({"machine": remote_rec.get("machine"), "pod_id": remote_rec.get("pod_id"), "root": remote_rec.get("root"),
                                              "tool": remote_rec.get("tool"), "source_sha": remote_rec.get("source_sha")} if remote_rec else None),
        "remote_rec": remote_rec,
        "tool": (job or {}).get("tool"),
        "fingerprint_expected": ((job or {}).get("workload_fingerprint") or {}).get("expected") if job else None,
        "fingerprint_observed": res.fingerprint,
        "result": res.summary(),
        "result_present": res.present,
        "result_valid": res.valid,
        "result_errors": res.errors,
        "validation": res.validation_status,
        "measurements": (res.doc or {}).get("measurements") if res.valid else None,
        "cost": record.cost_estimate(job, status),
        "preserved": manifest,
        "has_job": bool(raw_job),
        "has_status": bool(status),
        "has_failure": bool(failure),
    }


def list_runs(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and ((p / "job.json").exists() or (p / "remote.json").exists())]
    return [summarize(p) for p in sorted(dirs, key=lambda p: p.name)]


def resolve(root: Path, ref: str) -> Path:
    """Accept a full id, a unique prefix, or a path."""
    p = Path(ref).expanduser()
    if p.is_dir():
        return p
    if (root / ref).is_dir():
        return root / ref
    matches = [d for d in root.iterdir() if d.is_dir() and d.name.startswith(ref)] if root.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no run matching {ref!r} under {root}")
    raise LookupError(f"{ref!r} is ambiguous: {', '.join(sorted(m.name for m in matches))}")

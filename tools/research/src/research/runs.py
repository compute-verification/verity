"""Run ids and the local run registry.

A run is one recorded execution attempt.  Its directory is the telemetry attempt dir (job.json, status.json, events.jsonl,
resources.jsonl, failure.json, stdout.log, stderr.log) and nothing else is needed to describe it; the registry is just the
directory that holds them, so reading an old run needs no service and no credentials.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path.home() / ".research" / "runs"


def runs_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("RESEARCH_RUNS", DEFAULT_ROOT)).expanduser()


def new_run_id(now: float | None = None) -> str:
    """Sortable, human-typable: r20260921-171502-3f9a."""
    t = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc)
    return f"r{t:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def summarize(run_dir: Path) -> dict[str, Any]:
    """One flat record per run for listings.  Missing files are reported, not raised: a run that died before
    writing status.json is still a run."""
    job = _read(run_dir / "job.json") or {}
    status = _read(run_dir / "status.json") or {}
    failure = _read(run_dir / "failure.json") or {}
    own = job.get("ownership", {})
    comp = job.get("computation", {})
    exe = job.get("execution", {})
    transitions = status.get("transitions", [])
    started = next((t.get("t_utc") for t in transitions if t.get("state") == "running"), None)
    ended = transitions[-1].get("t_utc") if transitions and transitions[-1].get("state") in ("done", "failed", "cancelled") else None
    return {
        "id": run_dir.name,
        "dir": str(run_dir),
        "project": own.get("project"),
        "job": own.get("job"),
        "lane": own.get("lane"),
        "stage": comp.get("stage"),
        "tree_sha": comp.get("tree_sha"),
        "workload_digest": comp.get("workload_digest"),
        "state": status.get("state", "unknown" if not job else "submitted"),
        "rc": status.get("rc"),
        "signal": status.get("signal"),
        "class": failure.get("class"),
        "certainty": failure.get("certainty"),
        "why": failure.get("why"),
        "started_utc": started,
        "ended_utc": ended,
        "command": exe.get("command"),
        "has_job": bool(job),
        "has_status": bool(status),
        "has_failure": bool(failure),
    }


def list_runs(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / "job.json").exists()]
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

"""The run record: job.json v0.2 = the harness's v0.1 plus identity the harness does not know.

    tool                  which `research` (version, git sha, dirty) and which telemetry schema/collector/classifier wrote the record
    run                   run id, registry root, where it was launched from, remote target if any
    workload_fingerprint  {"expected": ...} supplied before launch; the OBSERVED fingerprint is the workload's, in result.json
    cost                  provider, pod id, rate + when it was observed, sharing; the estimate is derived at read time (cost_estimate)
    store                 optional: tool / derivation / reuse_key / key params / closure / conditions computed before launch (research.store)

Records written by the harness alone are v0.1 and stay readable: `read_job` returns the same shape with the v0.2 sections set to None,
which means unknown, never a default.  The harness is never modified for this; `research run` pre-writes job.json and the harness keeps it.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

from . import __version__
from .telemetry import CLASSIFIER_VERSION, COLLECTOR_VERSION, JOB_SCHEMA as JOB_SCHEMA_V1, SCHEMA as TELEMETRY_SCHEMA
from .telemetry.events import utc_now_ms

JOB_SCHEMA = "telemetry/job/v0.2"
READABLE_JOB_SCHEMAS = (JOB_SCHEMA_V1, JOB_SCHEMA)
V2_SECTIONS = ("tool", "run", "workload_fingerprint", "cost", "store")   # store: optional, None when the run had no store step


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def tool_identity() -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    sha = _git(["rev-parse", "HEAD"], here)
    dirty = None
    if sha:
        porcelain = _git(["status", "--porcelain", "--", str(here)], here)
        dirty = None if porcelain is None else bool(porcelain)
    snapshot = os.environ.get("RESEARCH_TOOL_SNAPSHOT")
    return {"research_version": __version__, "research_git_sha": sha, "research_git_sha_reason": None if sha else ("shipped snapshot" if snapshot else "not a git checkout"),
            "research_dirty": dirty, "research_snapshot": snapshot, "telemetry_schema": TELEMETRY_SCHEMA, "job_schema_base": JOB_SCHEMA_V1,
            "collector_version": COLLECTOR_VERSION, "classifier_version": CLASSIFIER_VERSION}


def cost_section(*, provider: str | None = None, pod_id: str | None = None, machine: str | None = None, rate_usd_per_hour: float | None = None,
                 rate_observed_utc: str | None = None, shared: bool | None = None, reason: str | None = None) -> dict[str, Any]:
    if rate_usd_per_hour is None and reason is None:
        reason = "no billed provider recorded" if provider is None else "rate not observed at launch"
    return {"provider": provider, "pod_id": pod_id, "machine": machine, "currency": "USD", "rate_usd_per_hour": rate_usd_per_hour,
            "rate_observed_utc": rate_observed_utc, "shared": shared, "reason": reason}


def extend_job(job: dict[str, Any], *, run_id: str, runs_root: Path, expected_fingerprint: dict[str, Any] | None,
               cost: dict[str, Any] | None = None, remote: dict[str, Any] | None = None, store: dict[str, Any] | None = None,
               cwd: dict[str, Any] | None = None) -> dict[str, Any]:
    """v0.1 job (from telemetry.job.build_job) -> v0.2.  Adds sections; changes nothing the harness wrote.

    `store` (optional): what `research run` computed BEFORE launch for the evidence store -- tool identity, derivation id,
    reuse_key, key params (content-resolved), inputs, closure manifest, conditions, scratch dirs (research.store README §4/§6).
    `cwd` (optional): {"path": the workload's cwd, "basis": "run_dir" | "source" | "clone" | "explicit"} -> run.cwd / run.cwd_basis."""
    if job.get("schema") != JOB_SCHEMA_V1:
        raise ValueError(f"extend_job expects a {JOB_SCHEMA_V1} record, got {job.get('schema')!r}")
    out = dict(job)
    out["schema"] = JOB_SCHEMA
    out["tool"] = tool_identity()
    out["run"] = {"id": run_id, "runs_root": str(runs_root), "created_utc": utc_now_ms(), "launched_from": socket.gethostname(), "remote": remote,
                  "cwd": (cwd or {}).get("path"), "cwd_basis": (cwd or {}).get("basis")}
    out["workload_fingerprint"] = {"expected": expected_fingerprint,
                                   "observed_in": "result.json (workload_fingerprint); expected and observed are never merged"}
    out["cost"] = cost if cost is not None else cost_section()
    out["store"] = store
    return out


def read_job(run_dir: str | Path) -> dict[str, Any] | None:
    """Normalized read.  Unknown schema -> None.  v0.1 -> v0.2 sections present and None (unknown)."""
    p = Path(run_dir) / "job.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") not in READABLE_JOB_SCHEMAS:
        return None
    for k in V2_SECTIONS:
        doc.setdefault(k, None)
    return doc


def cost_estimate(job: dict[str, Any] | None, status: dict[str, Any] | None) -> dict[str, Any]:
    """rate x run wall time.  Run occupancy only: setup, idle, retention, and teardown sit outside the command's interval.
    Missing pricing is unknown, not zero.  A shared pod is reported, not divided."""
    cost = (job or {}).get("cost") or {}
    rate = cost.get("rate_usd_per_hour")
    wall = (status or {}).get("wall_s")
    if rate is None:
        return {"usd": None, "reason": cost.get("reason") or "rate unknown", "basis": None}
    if wall is None:
        return {"usd": None, "reason": "run has no wall_s yet (still running, or died before the harness closed it)", "basis": None}
    return {"usd": round(rate * wall / 3600.0, 4), "reason": None, "wall_s": wall, "rate_usd_per_hour": rate,
            "shared": cost.get("shared"), "basis": "rate x run wall time; run occupancy only, not the pod's bill"}

"""result.json: the small, language-neutral interface by which a workload reports what it did.

The runner gives the workload RESEARCH_RUN_ID and RESEARCH_RUN_DIR.  The workload writes $RESEARCH_RUN_DIR/result.json (temp file
beside it, then os.replace) once, at the end.  Rust, Python, or shell can do this without importing research.  Progress goes to logs
and events, not to repeated rewrites of the file labeled as the final result.

    {
      "schema": "research/result/v0.1",
      "run_id": "<must equal the run's id>",
      "workload_fingerprint": {...},                  # application-defined identity of what ACTUALLY ran (binary hashes, keys, inputs, mode)
      "validation": {"status": "passed"|"failed"|"not_run", "checker": "<name>", "detail": "<text>", "evidence": ["rel/path", ...]},
      "measurements": [{"name": "prove_wall", "value": 12.3, "unit": "s"}, ...],   # explicit units; details stay in the workload's own files
      "measurement_files": ["results.jsonl", ...],    # relative paths to the detailed records
      "artifacts": ["proofs/", ...]                   # relative paths worth preserving
    }

Four facts are kept separate and reported separately: the process exited (status.json), the result is well-formed (this check), the
workload's own checker reported passed (validation.status), the evidence was preserved (the preservation manifest).  `research` says
"validation reported passed"; it does not verify proofs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESULT_SCHEMA = "research/result/v0.1"
MAX_BYTES = 1 << 20
VALIDATION_STATUSES = ("passed", "failed", "not_run")
ENV_RUN_ID = "RESEARCH_RUN_ID"
ENV_RUN_DIR = "RESEARCH_RUN_DIR"
# [R19 tele, F-r17-tp-06] the git sha of the tree the workload runs from (`research run --source`: the shipped `git archive` has no .git,
# so a source-identity precheck inside the workload cannot derive it); set by launch_request from request.json remote.source_sha and by
# the local runner from --tree-sha.  Absent when the run has no declared tree.
ENV_SOURCE_SHA = "RESEARCH_SOURCE_SHA"


@dataclass
class ResultCheck:
    present: bool
    valid: bool
    errors: list[str] = field(default_factory=list)
    doc: dict[str, Any] | None = None
    validation_status: str | None = None      # passed | failed | not_run | None (absent)
    fingerprint: dict[str, Any] | None = None
    artifacts_missing: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.present:
            return "absent"
        if not self.valid:
            return "invalid: " + "; ".join(self.errors)
        s = f"valid ({RESULT_SCHEMA})"
        if self.artifacts_missing:
            s += f"; {len(self.artifacts_missing)} listed artifact(s) missing"
        return s


def _rel_inside(run_dir: Path, rel: Any) -> tuple[Path | None, str | None]:
    if not isinstance(rel, str) or not rel:
        return None, f"path must be a non-empty string, got {rel!r}"
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        return None, f"path must be relative to the run directory and stay inside it: {rel!r}"
    return run_dir / p, None


def check(run_dir: str | Path, expected_run_id: str | None) -> ResultCheck:
    """Load and validate <run_dir>/result.json.  Absent is a distinct outcome from invalid."""
    d = Path(run_dir)
    p = d / "result.json"
    if not p.exists():
        return ResultCheck(present=False, valid=False)
    try:
        size = p.stat().st_size
    except OSError as e:
        return ResultCheck(present=True, valid=False, errors=[f"stat failed: {e.strerror}"])
    if size > MAX_BYTES:
        return ResultCheck(present=True, valid=False, errors=[f"result.json is {size} bytes (> {MAX_BYTES}); keep large data in referenced files"])
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return ResultCheck(present=True, valid=False, errors=[f"not valid JSON: {e}"])
    return check_doc(d, doc, expected_run_id)


def check_doc(run_dir: Path, doc: Any, expected_run_id: str | None) -> ResultCheck:
    """Validate an in-memory document; relative paths are resolved against run_dir."""
    d = Path(run_dir)
    if not isinstance(doc, dict):
        return ResultCheck(present=True, valid=False, errors=["top level must be an object"])
    errors: list[str] = []
    if doc.get("schema") != RESULT_SCHEMA:
        errors.append(f"schema must be {RESULT_SCHEMA!r}, got {doc.get('schema')!r}")
    rid = doc.get("run_id")
    if not isinstance(rid, str) or not rid:
        errors.append("run_id must be a non-empty string")
    elif expected_run_id is not None and rid != expected_run_id:
        errors.append(f"run_id {rid!r} does not match this run {expected_run_id!r} (result attached to the wrong run)")
    fp = doc.get("workload_fingerprint")
    if fp is not None and not isinstance(fp, dict):
        errors.append("workload_fingerprint must be an object when present")
    val = doc.get("validation")
    vstatus = None
    if val is not None:
        if not isinstance(val, dict):
            errors.append("validation must be an object when present")
        else:
            vstatus = val.get("status")
            if vstatus not in VALIDATION_STATUSES:
                errors.append(f"validation.status must be one of {VALIDATION_STATUSES}, got {vstatus!r}")
            for ev in val.get("evidence") or []:
                _, err = _rel_inside(d, ev)
                if err:
                    errors.append(f"validation.evidence: {err}")
    meas = doc.get("measurements")
    if meas is not None:
        if not isinstance(meas, list):
            errors.append("measurements must be a list")
        else:
            for i, m in enumerate(meas):
                if not isinstance(m, dict) or not isinstance(m.get("name"), str) or "value" not in m or not isinstance(m.get("unit"), str):
                    errors.append(f"measurements[{i}] needs name (str), value, unit (str)")
    missing: list[str] = []
    for key in ("measurement_files", "artifacts"):
        items = doc.get(key)
        if items is None:
            continue
        if not isinstance(items, list):
            errors.append(f"{key} must be a list of relative paths")
            continue
        for rel in items:
            path, err = _rel_inside(d, rel)
            if err:
                errors.append(f"{key}: {err}")
            elif not path.exists():
                missing.append(rel)
    return ResultCheck(present=True, valid=not errors, errors=errors, doc=doc, validation_status=vstatus if isinstance(vstatus, str) else None,
                       fingerprint=fp if isinstance(fp, dict) else None, artifacts_missing=missing)


def write(run_dir: str | Path, doc: dict[str, Any]) -> Path:
    """For Python workloads that want a helper: validate, then temp file beside the destination + os.replace (atomic on POSIX,
    same filesystem).  Atomic visibility is not durability; preservation is a separate step."""
    d = Path(run_dir)
    chk = check_doc(d, doc, doc.get("run_id") if isinstance(doc, dict) else None)
    if not chk.valid:
        raise ValueError("refusing to publish an invalid result: " + "; ".join(chk.errors))
    p = d / "result.json"
    tmp = d / f"result.json.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def template(run_id: str) -> dict[str, Any]:
    return {"schema": RESULT_SCHEMA, "run_id": run_id, "workload_fingerprint": {}, "validation": {"status": "not_run", "checker": None, "detail": None, "evidence": []},
            "measurements": [], "measurement_files": [], "artifacts": []}


def env_for(run_id: str, run_dir: str | Path, source_sha: str | None = None) -> dict[str, str]:
    env = {ENV_RUN_ID: run_id, ENV_RUN_DIR: str(run_dir)}
    if source_sha:
        env[ENV_SOURCE_SHA] = source_sha
    return env

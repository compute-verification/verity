"""The attempt manifest (store README §4) and `publish` (§10): what `research run` does at completion or failure, and what
`research data attempt publish RUN_DIR` does after the fact for runs that predate the store.

    build(run_dir, ...)    -> the manifest dict, from job.json (v0.1 or v0.2, with or without its `store` section), status.json,
                              failure.json, result.json, launch.json, remote.json; unknown facts are None, never guessed
    publish(run_dir, ...)  -> {"published": True, "id", "derivation", "reuse_key", "outputs", "telemetry", "n_outputs"}
                              or {"published": False, "error": ...} with store_publish_error.json written beside the run's records.
                              Never raises into the runner.  Idempotent: an attempt already in the store is a no-op.

Steps: result.json (valid) -> artifact of kind `result.kind` (default bench-result/v1), meta = the document minus its path lists,
refs.run_files -> the ONE `run-files/v1` tree made of every path in `artifacts` + `measurement_files`; events.jsonl /
resources.jsonl -> `telemetry/v1`; then `put_attempt` (write-once).  The local store is opened lazily (`research.store.local`,
lane store-core); `store=` accepts any object with put_artifact / put_attempt / get_attempt.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

from .. import env as E
from .. import record, result
from ..telemetry.events import utc_now_ms
from ..telemetry.job import write_json_atomic
from . import derivation as D
from .tool import Tool

ATTEMPT_SCHEMA = "research/attempt/v0.1"
PUBLISH_ERROR_SCHEMA = "research/store-publish-error/v0.1"
TERMINAL = ("done", "failed", "cancelled")
DEFAULT_RESULT_KIND = "bench-result/v1"


def store_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("RESEARCH_STORE") or Path.home() / ".research" / "store").expanduser()


def open_store(store: Any = None, root: str | os.PathLike[str] | None = None) -> Any:
    if store is not None:
        return store
    from .local import LocalStore  # lane store-core

    return LocalStore(store_root(root))


def _read(p: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _tool_identity(tool: Any, store_sec: dict[str, Any], argv: list[str]) -> dict[str, Any]:
    if isinstance(tool, Tool):
        return tool.identity()
    if isinstance(tool, dict):
        return {"name": tool.get("name"), "version": tool.get("version")}
    if isinstance(tool, str):
        return {"name": tool, "version": None}
    if isinstance(store_sec.get("tool"), dict):
        t = store_sec["tool"]
        return {"name": t.get("name"), "version": t.get("version")}
    return {"name": Path(argv[0]).name if argv else None, "version": None}


def build(run_dir: Path | str, *, tool: Tool | dict[str, Any] | str | None = None, conditions: dict[str, Any] | None = None,
          argv: list[str] | None = None, params: dict[str, Any] | None = None, inputs: dict[str, str] | None = None,
          outputs: dict[str, str | None] | None = None, key_params_resolved: dict[str, Any] | None = None,
          telemetry: dict[str, str | None] | None = None, closure: dict[str, str] | None = None,
          derivation: str | None = None, reuse_key: str | None | Any = ..., campaign: str | None = None) -> dict[str, Any]:
    """The §4 manifest.  Explicit arguments win; then job.json's `store` section (written by `research run --tool`); then
    what the records imply; then None."""
    d = Path(run_dir)
    job = record.read_job(d) or _read(d / "job.json") or {}
    status = _read(d / "status.json") or {}
    failure = _read(d / "failure.json") or {}
    launch = _read(d / "launch.json") or {}
    remote_rec = _read(d / "remote.json") or {}
    store_sec = job.get("store") or {}
    own = job.get("ownership") or {}
    run_sec = job.get("run") or {}
    exe = job.get("execution") or {}
    run_id = run_sec.get("id") or own.get("job") or d.name
    argv = list(argv if argv is not None else (exe.get("command") or status.get("command") or remote_rec.get("command") or []))
    ident = _tool_identity(tool, store_sec, argv)
    key_params = list(tool.key_params) if isinstance(tool, Tool) else list(store_sec.get("key_params") or [])
    key_conditions = list(tool.key_conditions) if isinstance(tool, Tool) else list(store_sec.get("key_conditions") or [])
    params = params if params is not None else (store_sec.get("params") if store_sec.get("params") is not None else {"argv": argv})
    key_params_resolved = key_params_resolved if key_params_resolved is not None else (store_sec.get("key_params_resolved") or {})
    inputs = inputs if inputs is not None else (store_sec.get("inputs") or {})
    conds = dict(conditions) if conditions is not None else (dict(store_sec["conditions"]) if isinstance(store_sec.get("conditions"), dict)
                                                          else E.conditions_from_job(job))
    for k in E.KEYS:
        conds.setdefault(k, None)
    if campaign is not None:
        conds["exec.campaign"] = campaign
    closure = closure if closure is not None else (store_sec.get("closure") or {})
    drv = derivation or store_sec.get("derivation") or D.derivation_id(ident, inputs, key_params_resolved, conds, key_conditions)
    rk = reuse_key if reuse_key is not ... else (store_sec["reuse_key"] if "reuse_key" in store_sec else D.reuse_key(drv, conds))
    res = result.check(d, run_id)
    transitions = status.get("transitions") or []
    start = status.get("started_utc") or next((t.get("t_utc") for t in transitions if t.get("state") == "running"), None)
    end = status.get("ended_utc") or (transitions[-1].get("t_utc") if transitions and transitions[-1].get("state") in TERMINAL else None)
    exclusive = conds.get("exec.exclusive")
    if exclusive is None and launch.get("argv"):
        exclusive = "--exclusive-lock" in launch["argv"]
    cost_sec = job.get("cost") or {}
    est = record.cost_estimate(job, status)
    executed_dir = f"{run_sec['runs_root'].rstrip('/')}/{run_id}" if run_sec.get("runs_root") else str(d)
    return {
        "schema": ATTEMPT_SCHEMA,
        "id": run_id,
        "tool": ident,
        "derivation": drv,
        "reuse_key": rk,
        "argv": argv,
        "params": params,
        "key_params": key_params,
        "inputs": inputs,
        "conditions": conds,
        "key_conditions": key_conditions,
        "source": E.source_identity(conds),
        "closure": closure,
        "execution": {"start_utc": start, "end_utc": end, "wall_s": status.get("wall_s"), "exclusive": exclusive,
                      "load_start": conds.get("exec.load_start"), "load_end": conds.get("exec.load_end"),
                      "host": conds.get("hw.hostname") or (job.get("environment") or {}).get("hostname") or launch.get("hostname"),
                      "project": own.get("project") or conds.get("exec.project"), "campaign": conds.get("exec.campaign"),
                      "run_dir": executed_dir, "stage": (job.get("computation") or {}).get("stage"),
                      "remote": {k: (run_sec.get("remote") or {}).get(k) for k in ("machine", "pod_id", "root", "source_sha")} if run_sec.get("remote") else None},
        "cost": {"provider": cost_sec.get("provider"), "pod_id": cost_sec.get("pod_id"), "rate_usd_per_hour": cost_sec.get("rate_usd_per_hour"),
                 "usd": est.get("usd"), "shared": cost_sec.get("shared")},
        "status": {"state": status.get("state"), "rc": status.get("rc"), "class": failure.get("class"), "failure": failure.get("why"),
                   "validation": res.validation_status, "result": res.summary(), "timed_out": status.get("timed_out")},
        "outputs": dict(outputs) if outputs is not None else {"result": None, "run_files": None},
        "telemetry": dict(telemetry) if telemetry is not None else {"events": None, "resources": None},
    }


# ---- publish --------------------------------------------------------------------------------------------------------------------------------

def _stage_run_files(run_dir: Path, rels: list[str], into: Path) -> tuple[int, list[str]]:
    """Copy each listed path (file or directory, recursively) under `into` preserving the relative path.  Returns (files, missing)."""
    n, missing = 0, []
    for rel in rels:
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or not rel:
            missing.append(rel)
            continue
        src = run_dir / p
        if src.is_file():
            dst = into / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
                n += 1
        elif src.is_dir():
            for f in sorted(x for x in src.rglob("*") if x.is_file()):
                dst = into / f.relative_to(run_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(f, dst)
                    n += 1
        else:
            missing.append(rel)
    return n, missing


def _dedupe(items: list[Any]) -> list[str]:
    seen: list[str] = []
    for x in items:
        if isinstance(x, str) and x not in seen:
            seen.append(x)
    return seen


def _resolve_tool(tool: Any, store_sec: dict[str, Any]) -> Any:
    from . import tools_registry

    if isinstance(tool, str):
        return tools_registry.load(tool)
    if tool is None and isinstance(store_sec.get("tool"), dict) and store_sec["tool"].get("spec"):
        try:
            return tools_registry.load(store_sec["tool"]["spec"])
        except Exception:  # noqa: BLE001  the declaration may not be importable here; the recorded identity still is
            return None
    return tool


def _publish(run_dir: Path, *, tool: Any, store: Any, remote_push: bool | None, conditions: dict[str, Any] | None, force: bool,
             build_kwargs: dict[str, Any]) -> dict[str, Any]:
    st = open_store(store)
    job = record.read_job(run_dir) or _read(run_dir / "job.json") or {}
    if not job:
        raise FileNotFoundError(f"{run_dir}: no readable job.json (not fetched yet, or refused before it ran)")
    status = _read(run_dir / "status.json") or {}
    run_id = (job.get("run") or {}).get("id") or (job.get("ownership") or {}).get("job") or run_dir.name
    existing = st.get_attempt(run_id)
    if existing:
        outs = {**(existing.get("outputs") or {}), **(existing.get("telemetry") or {})}
        return {"published": True, "already": True, "id": run_id, "derivation": existing.get("derivation"), "reuse_key": existing.get("reuse_key"),
                "outputs": existing.get("outputs"), "telemetry": existing.get("telemetry"), "n_outputs": sum(1 for v in outs.values() if v)}
    state = status.get("state")
    if state not in TERMINAL and not force:
        raise RuntimeError(f"run {run_id} is {state or 'without status.json'}: the store only sees completed or failed attempts (--force to override)")
    tool = _resolve_tool(tool, job.get("store") or {})
    outputs: dict[str, str | None] = {"result": None, "run_files": None}
    res = result.check(run_dir, run_id)
    if res.valid and res.doc is not None:
        doc = res.doc
        rels = _dedupe(list(doc.get("artifacts") or []) + list(doc.get("measurement_files") or []))
        if rels:
            with tempfile.TemporaryDirectory(prefix="research-run-files-") as tmp:
                n, missing = _stage_run_files(run_dir, rels, Path(tmp))
                if n:
                    outputs["run_files"] = st.put_artifact("run-files/v1", {"run_id": run_id, "files": n, "listed": rels, "missing": missing}, tree=Path(tmp))
        meta = {k: v for k, v in doc.items() if k not in ("artifacts", "measurement_files")}
        kind = doc.get("kind") if isinstance(doc.get("kind"), str) else DEFAULT_RESULT_KIND
        refs = {"run_files": outputs["run_files"]} if outputs["run_files"] else None
        outputs["result"] = st.put_artifact(kind, meta, refs)
    telemetry: dict[str, str | None] = {"events": None, "resources": None}
    for name in ("events", "resources"):
        p = run_dir / f"{name}.jsonl"
        if p.is_file() and p.stat().st_size > 0:
            telemetry[name] = st.put_artifact("telemetry/v1", {"run_id": run_id, "stream": name, "bytes": p.stat().st_size}, file=p)
    doc = build(run_dir, tool=tool, conditions=conditions, outputs=outputs, telemetry=telemetry, **build_kwargs)
    st.put_attempt(doc)
    out = {"published": True, "id": run_id, "derivation": doc["derivation"], "reuse_key": doc["reuse_key"], "outputs": outputs, "telemetry": telemetry,
           "n_outputs": sum(1 for v in {**outputs, **telemetry}.values() if v)}
    if remote_push is not False and hasattr(st, "push_attempt"):        # §10 step 4 belongs to the storage half; hook when it exists
        try:
            out["preserved"] = st.push_attempt(run_id)
        except Exception as e:  # noqa: BLE001
            out["preserve_error"] = f"{type(e).__name__}: {e}"
    return out


def publish(run_dir: Path | str, *, tool: Any = None, store: Any = None, remote_push: bool | None = None, conditions: dict[str, Any] | None = None,
            force: bool = False, **build_kwargs: Any) -> dict[str, Any]:
    """README §10 steps 1-3.  Never raises: a failure is recorded in <run_dir>/store_publish_error.json and returned."""
    d = Path(run_dir)
    try:
        return _publish(d, tool=tool, store=store, remote_push=remote_push, conditions=conditions, force=force, build_kwargs=build_kwargs)
    except Exception as e:  # noqa: BLE001  the runner's exit status is the workload's, never the store's
        err = {"schema": PUBLISH_ERROR_SCHEMA, "at_utc": utc_now_ms(), "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
        try:
            write_json_atomic(d / "store_publish_error.json", err)
        except OSError:
            pass
        return {"published": False, "error": err["error"]}

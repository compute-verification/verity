"""The attempt manifest (store README §4) and `publish` (§10): what `research run` does at completion or failure, and what
`research data attempt publish RUN_DIR` does after the fact for runs that predate the store.

    build(run_dir, ...)    -> the manifest dict, from job.json (v0.1 or v0.2, with or without its `store` section), status.json,
                              failure.json, result.json, launch.json, remote.json; unknown facts are None, never guessed
    publish(run_dir, ...)  -> {"published": True, "id", "derivation", "reuse_key", "outputs", "telemetry", "n_outputs"}
                              or {"published": False, "error": ...} with store_publish_error.json written beside the run's records.
                              Never raises into the runner.  Idempotent: an attempt already in the store is a no-op.

Steps: result.json (valid) -> artifact of kind `result.kind` (default bench-result/v1), meta = the document minus its path lists
(over the meta cap: `result_meta`'s derived meta, with the result.json as payload), refs.run_files -> the ONE `run-files/v1` tree made of every path in `artifacts` + `measurement_files`; `outputs.json`
(schema research/outputs/v0.1, see `typed_outputs`) -> one named, typed artifact per entry; events.jsonl /
resources.jsonl -> `telemetry/v1`; then `put_attempt` (write-once).  The local store is opened lazily (`research.store.local`,
lane store-core); `store=` accepts any object with put_artifact / put_attempt / get_attempt.
"""
from __future__ import annotations

import hashlib
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
OUTPUTS_SCHEMA = "research/outputs/v0.1"
TERMINAL = ("done", "failed", "cancelled")
DEFAULT_RESULT_KIND = "bench-result/v1"
META_TRANSFORM = "circuit-check-cache-counts"


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
    if conds.get("exec.machine") is None and (run_sec.get("remote") or {}).get("machine"):     # a `--on <machine>` run: the machines.toml name
        conds["exec.machine"] = run_sec["remote"]["machine"]
    if campaign is not None:
        conds["exec.campaign"] = campaign
    closure = closure if closure is not None else (store_sec.get("closure") or {})
    declared = tool is not None or bool((store_sec.get("tool") or {}).get("spec"))
    recorded_drv = store_sec.get("derivation")
    if not declared and "argv" not in key_params_resolved:
        # an undeclared request is its whole redacted command line (research run's rule); a record from an older runner, or one
        # with no store section, would otherwise share one derivation with every other run of the same executable
        from ..telemetry.job import redact_argv

        key_params_resolved = {**key_params_resolved, "argv": redact_argv(argv)[0]}
        key_params, recorded_drv = ["argv"], None
    drv = derivation or recorded_drv or D.derivation_id(ident, inputs, key_params_resolved, conds, key_conditions)
    if reuse_key is not ...:
        rk = reuse_key
    elif not declared:
        rk = None                                                # no declared Tool, no reuse claim
    else:
        rk = store_sec["reuse_key"] if "reuse_key" in store_sec else D.reuse_key(drv, conds)
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
                      "machine": conds.get("exec.machine"),          # the machines.toml name (indexed as exec.machine); host is the container's
                      "project": own.get("project") or conds.get("exec.project"), "campaign": conds.get("exec.campaign"),
                      "run_dir": executed_dir, "stage": (job.get("computation") or {}).get("stage"),
                      "cwd": run_sec.get("cwd"), "cwd_basis": run_sec.get("cwd_basis"), "cwd_clone": run_sec.get("cwd_clone"),
                      "mem_gb_reserved": ((run_sec.get("remote") or {}).get("mem") or {}).get("gb"),
                      # source_sha is the verified source identity the machine executed from; source_archive_sha256 the digest of the archive
                      # that landed there (remote.py ship_source / READY.json); both None for a run that shipped no tree
                      "remote": {k: (run_sec.get("remote") or {}).get(k) for k in ("machine", "pod_id", "root", "source_sha", "source_dir",
                                                                                   "source_archive_sha256", "source_verified")} if run_sec.get("remote") else None},
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


def _cache_counts(meta: dict[str, Any]) -> dict[str, Any]:
    """`check.py`'s circuit-check step with its cache's per-target lists as their lengths (what #1712's `_step` records)."""
    steps = [{**s, "cache": {k: len(v) if isinstance(v, list) else v for k, v in s["cache"].items()}}
             if isinstance(s, dict) and s.get("step") == "circuit-check" and isinstance(s.get("cache"), dict) else s
             for s in meta.get("steps") or []]
    return {**meta, "steps": steps} if "steps" in meta else meta


def result_meta(run_dir: Path, doc: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    """The result artifact's meta and payload: the document minus its path lists, no payload.  A meta over the store's cap
    (model.META_MAX_BYTES) goes through META_TRANSFORM, says so in `meta_derived` (the transform, and the sha256 and size of the
    result.json it came from), and the result.json is the payload byte for byte, so the meta can be recomputed from it."""
    from . import canon
    from .model import META_MAX_BYTES

    meta = {k: v for k, v in doc.items() if k not in ("artifacts", "measurement_files")}
    if len(canon.dumps(meta)) <= META_MAX_BYTES:
        return meta, None
    p = run_dir / "result.json"
    raw = p.read_bytes()
    derived = {"transform": META_TRANSFORM, "from": "result.json", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return {**_cache_counts(meta), "meta_derived": derived}, p


def _rel_under(run_dir: Path, rel: Any) -> Path:
    """A payload path from outputs.json: relative to the run dir, never absolute, never escaping it."""
    if not isinstance(rel, str) or not rel:
        raise ValueError(f"outputs.json: payload path must be a non-empty string, got {rel!r}")
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"outputs.json: payload path {rel!r} must be relative to the run dir and not escape it")
    return run_dir / p


def typed_outputs(run_dir: Path, st: Any, run_id: str) -> dict[str, str]:
    """Publish the typed outputs a workload declared in `<run_dir>/outputs.json` (schema `research/outputs/v0.1`):

        {"schema": "research/outputs/v0.1",
         "outputs": [{"name": "build", "kind": "vllm-build/v1", "meta": {...}, "refs": {"build": "art:...", "evidence": "@evidence"},
                      "tree": "outputs/build" | "file": "outputs/x.json" | "files": ["a", "dir/"], "optional": false}, ...]}

    One artifact per entry, in order; `refs` values are `art:` ids or `@<name>` of an earlier entry.  `files` stages the listed
    paths (relative to the run dir) into one tree.  A missing payload is an error unless `optional` is set (then the entry is
    skipped).  Returns {name: art id}.  The manifest's `outputs` map gets these under their names (beside `result` / `run_files`)."""
    from .kinds import is_kind

    doc = _read(run_dir / "outputs.json")
    if doc is None:
        return {}
    if doc.get("schema") != OUTPUTS_SCHEMA:
        raise ValueError(f"outputs.json: schema {doc.get('schema')!r} != {OUTPUTS_SCHEMA!r}")
    entries = doc.get("outputs")
    if not isinstance(entries, list):
        raise ValueError("outputs.json: `outputs` must be a list")
    out: dict[str, str] = {}
    for i, ent in enumerate(entries):
        if not isinstance(ent, dict):
            raise ValueError(f"outputs.json: entry {i} is not an object")
        name, kind = ent.get("name"), ent.get("kind")
        if not isinstance(name, str) or not name or name == "argv":
            raise ValueError(f"outputs.json: entry {i} needs a name")
        if not is_kind(kind):
            raise ValueError(f"outputs.json: {name}: kind {kind!r} is not `<name>/v<int>`")
        if name in out:
            raise ValueError(f"outputs.json: duplicate output name {name!r}")
        meta = dict(ent.get("meta") or {})
        meta.setdefault("run_id", run_id)
        refs: dict[str, str] = {}
        for k, v in (ent.get("refs") or {}).items():
            if isinstance(v, str) and v.startswith("@"):
                if v[1:] not in out:
                    raise ValueError(f"outputs.json: {name}: ref {k}={v} names no earlier output")
                refs[k] = out[v[1:]]
            elif isinstance(v, str) and v.startswith("art:"):
                refs[k] = v
            elif v is None:
                continue
            else:
                raise ValueError(f"outputs.json: {name}: ref {k}={v!r} is neither an art: id nor @<earlier output>")
        optional = bool(ent.get("optional"))
        if "file" in ent:
            p = _rel_under(run_dir, ent["file"])
            if not p.is_file():
                if optional:
                    continue
                raise FileNotFoundError(f"outputs.json: {name}: file {ent['file']!r} is absent")
            out[name] = st.put_artifact(kind, meta, refs or None, file=p)
        elif "tree" in ent:
            p = _rel_under(run_dir, ent["tree"])
            if not p.is_dir() or not any(x.is_file() for x in p.rglob("*")):
                if optional:
                    continue
                raise FileNotFoundError(f"outputs.json: {name}: tree {ent['tree']!r} is absent or empty")
            out[name] = st.put_artifact(kind, meta, refs or None, tree=p)
        elif "files" in ent:
            rels = _dedupe(list(ent.get("files") or []))
            with tempfile.TemporaryDirectory(prefix=f"research-out-{name}-") as tmp:
                n, missing = _stage_run_files(run_dir, rels, Path(tmp))
                if n == 0:
                    if optional:
                        continue
                    raise FileNotFoundError(f"outputs.json: {name}: none of {rels} exists under the run dir")
                meta.setdefault("files", n)
                if missing:
                    meta.setdefault("missing", missing)
                out[name] = st.put_artifact(kind, meta, refs or None, tree=Path(tmp))
        else:
            out[name] = st.put_artifact(kind, meta, refs or None)
    return out


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


def _existing_attempt(st: Any, run_id: str) -> tuple[dict[str, Any] | None, str | None]:
    """`get_attempt` consults the remote when the local copy is absent; a remote that is configured but unusable here (credential
    env vars unset on this machine) must not stop a LOCAL publish.  Returns (doc or None, remote error or None)."""
    from .remote import ConfigError

    try:
        return st.get_attempt(run_id), None
    except ConfigError as e:
        p = st.attempt_path(run_id) if hasattr(st, "attempt_path") else None
        if p is not None and Path(p).exists():
            return json.loads(Path(p).read_bytes()), f"{type(e).__name__}: {e}"
        return None, f"{type(e).__name__}: {e}"


def _publish(run_dir: Path, *, tool: Any, store: Any, remote_push: bool | None, conditions: dict[str, Any] | None, force: bool,
             build_kwargs: dict[str, Any], run_record: bool = False) -> dict[str, Any]:
    st = open_store(store)
    job = record.read_job(run_dir) or _read(run_dir / "job.json") or {}
    if not job:
        raise FileNotFoundError(f"{run_dir}: no readable job.json (not fetched yet, or refused before it ran)")
    status = _read(run_dir / "status.json") or {}
    run_id = (job.get("run") or {}).get("id") or (job.get("ownership") or {}).get("job") or run_dir.name
    existing, remote_error = _existing_attempt(st, run_id)
    if remote_error:
        remote_push = False                                   # the local publish proceeds; nothing is pushed or called PRESERVED
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
        meta, payload = result_meta(run_dir, doc)
        kind = doc.get("kind") if isinstance(doc.get("kind"), str) else DEFAULT_RESULT_KIND
        refs = {"run_files": outputs["run_files"]} if outputs["run_files"] else None
        outputs["result"] = st.put_artifact(kind, meta, refs, file=payload)
    outputs.update(typed_outputs(run_dir, st, run_id))               # outputs.json: named, typed artifacts the workload declared
    if run_record:                                                   # the whole run dir, logs included (store.custody)
        from .custody import put_run_record

        outputs["run_record"] = put_run_record(st, run_dir, run_id)
    telemetry: dict[str, str | None] = {"events": None, "resources": None}
    for name in ("events", "resources"):
        p = run_dir / f"{name}.jsonl"
        if p.is_file() and p.stat().st_size > 0:
            telemetry[name] = st.put_artifact("telemetry/v1", {"run_id": run_id, "stream": name, "bytes": p.stat().st_size}, file=p)
    doc = build(run_dir, tool=tool, conditions=conditions, outputs=outputs, telemetry=telemetry, **build_kwargs)
    st.put_attempt(doc)
    out = {"published": True, "id": run_id, "derivation": doc["derivation"], "reuse_key": doc["reuse_key"], "outputs": outputs, "telemetry": telemetry,
           "n_outputs": sum(1 for v in {**outputs, **telemetry}.values() if v)}
    if remote_push is not False and hasattr(st, "push_attempt"):        # §10 step 4: preserve on the remote when one is usable here
        if remote_error or getattr(st, "remote", None) is None:
            out["preserved"] = False
            out["preserve_skipped"] = remote_error or getattr(st, "remote_error", None) or "no remote configured"
        else:
            try:
                arts = [a for a in {**outputs, **telemetry}.values() if a]
                reports = [st.push(a, preserve=True) for a in arts]
                st.push_attempt(run_id)
                out["preserved"] = all(r.get("preserved") for r in reports)
                out["preserved_artifacts"] = {r["id"]: bool(r.get("preserved")) for r in reports}
                if not out["preserved"]:                                # an upload that failed is not "no remote": say what failed
                    bad = next(r for r in reports if not r.get("preserved"))
                    out["preserve_error"] = f"{bad['id'][:16]}: {(bad.get('errors') or ['not preserved'])[0]}"
                if hasattr(st, "push_labels"):                          # the third leg: assertions already made about this attempt, its
                    from . import labels_sync as LS                     # derivation or its outputs (a re-publish after `label`) travel too

                    lrep = st.push_labels(LS.attempt_targets(doc))
                    out["labels"] = {"local": lrep["local"], "pushed": len(lrep["pushed"]), "already": lrep["already"], "failed": lrep["failed"]}
                    if lrep["failed"]:
                        out["preserved"] = False
                        out["preserve_error"] = f"labels: {len(lrep['failed'])} assertion(s) not pushed ({lrep['failed'][0]})"
            except Exception as e:  # noqa: BLE001
                out["preserved"] = False
                out["preserve_error"] = f"{type(e).__name__}: {e}"
    return out


def publish(run_dir: Path | str, *, tool: Any = None, store: Any = None, remote_push: bool | None = None, conditions: dict[str, Any] | None = None,
            force: bool = False, run_record: bool = False, **build_kwargs: Any) -> dict[str, Any]:
    """README §10 steps 1-3.  Never raises: a failure is recorded in <run_dir>/store_publish_error.json and returned.  `run_record`: the
    attempt also names the whole run dir as `outputs.run_record` (`run-record/v1`, store.custody)."""
    d = Path(run_dir)
    try:
        return _publish(d, tool=tool, store=store, remote_push=remote_push, conditions=conditions, force=force, build_kwargs=build_kwargs,
                        run_record=run_record)
    except Exception as e:  # noqa: BLE001  the runner's exit status is the workload's, never the store's
        err = {"schema": PUBLISH_ERROR_SCHEMA, "at_utc": utc_now_ms(), "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
        try:
            write_json_atomic(d / "store_publish_error.json", err)
        except OSError:
            pass
        return {"published": False, "error": err["error"]}

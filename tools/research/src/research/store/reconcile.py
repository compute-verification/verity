"""Attempt documents that differ between this store and the remote: find them and replace the local copy with the remote's, with an audit trail."""
from __future__ import annotations

import datetime as dt
import getpass
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import canon, ids
from .index import flatten
from .local import LocalStore, attempt_key, write_once
from .remote import md5_file
from .remote_s3 import is_md5_etag

LOG = "reconcile.log"
SUPERSEDED = ".superseded"
REASON = ("the remote's copy is the first write of a write-once key; the local copy was a second publish of the same run, made "
          "where the remote's copy was not visible")


def _named(doc: dict) -> set[str]:
    return {v for sec in ("outputs", "telemetry") for v in (doc.get(sec) or {}).values() if ids.is_art(v)}


def _last_modified(r: Any, key: str) -> str | None:
    try:
        with r._request("HEAD", key) as resp:
            return resp.headers.get("Last-Modified")
    except Exception:  # noqa: BLE001
        return None


def diverged(st: LocalStore, run_ids: list[str] | None = None, *, jobs: int = 32) -> list[dict]:
    r = st._require_remote()
    todo = run_ids if run_ids else st.attempt_ids()
    tmpdir = st.root / "scratch"
    tmpdir.mkdir(parents=True, exist_ok=True)

    def one(rid: str) -> dict | None:
        p = st.attempt_path(rid)
        if not p.exists():
            return None
        hd = r.head(attempt_key(rid))
        if hd is None:
            return None
        local = p.read_bytes()
        if hd["bytes"] == len(local) and is_md5_etag(hd.get("etag")) and hd["etag"] == md5_file(p):
            return None
        tmp = tmpdir / f".reconcile-{rid}-{os.getpid()}"
        try:
            r.get(attempt_key(rid), tmp)
            remote = tmp.read_bytes()
        finally:
            tmp.unlink(missing_ok=True)
        if remote == local:
            return None
        ld, rd = json.loads(local), json.loads(remote)
        if rd.get("id") != rid:
            return {"run": rid, "error": f"the remote's {attempt_key(rid)} names attempt {rd.get('id')!r}"}
        lf, rf = dict(flatten(ld)), dict(flatten(rd))
        return {"run": rid, "local_sha256": canon.sha256_bytes(local), "remote_sha256": canon.sha256_bytes(remote),
                "remote_last_modified": _last_modified(r, attempt_key(rid)),
                "differs": sorted(k for k in set(lf) & set(rf) if lf[k] != rf[k]),
                "local_only": sorted(set(lf) - set(rf)), "remote_only": sorted(set(rf) - set(lf)),
                "local_only_artifacts": sorted(_named(ld) - _named(rd)), "_remote": remote}

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        return [x for x in ex.map(one, todo) if x]


def apply(st: LocalStore, found: list[dict]) -> list[dict]:
    out = []
    sup = st.root / "attempts" / SUPERSEDED
    sup.mkdir(parents=True, exist_ok=True)
    log = st.root / LOG
    for f in found:
        if f.get("error"):
            continue
        rid, p = f["run"], st.attempt_path(f["run"])
        aside = sup / f"{rid}.{f['local_sha256'][:12]}.json"
        if not aside.exists():
            os.replace(p, aside)
        else:
            p.unlink()
        write_once(p, f["_remote"])
        st.index.index_attempt(json.loads(f["_remote"]))
        entry = {"utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "host": socket.gethostname(),
                 "user": getpass.getuser(), "action": "superseded-by-remote", "reason": REASON,
                 **{k: v for k, v in f.items() if not k.startswith("_")}, "superseded_path": str(aside.relative_to(st.root))}
        with log.open("a") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        out.append(entry)
    return out


__all__ = ["apply", "diverged"]

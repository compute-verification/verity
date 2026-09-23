"""The durability gate (F7): is every output of an attempt — or an artifact — verifiably on the remote?

    check(store, targets, mode="head") -> report; report["preserved"] is the gate

A target is a run id (every artifact its manifest names under `outputs` and `telemetry`, plus the attempt manifest itself) or
an `art:` id.  An artifact is PRESERVED when the remote holds its manifest (read back, sha256 == id) and every blob: remote size
== manifest size and, per `mode`, `head` = single-part ETag == local MD5 when the blob is local and the remote gives an MD5 ETag,
otherwise a full read-back sha256; `readback` = always read back; `recorded` = trust the local catalog's verified remote replica
(offline; what `push --verify`/`verify` recorded).  Nothing is uploaded here: `research data push` is the repair.

Without a usable remote (no store.toml, or credentials not in this environment) nothing is preserved, and the report says why.
"""
from __future__ import annotations

from typing import Any

from . import ids
from .local import (
    VERIFY_HEAD,
    VERIFY_READBACK,
    LocalStore,
    attempt_key,
    manifest_key,
    object_key,
    utc_now,
)
from .remote import md5_file
from .remote_s3 import SINGLE_PUT_MAX, is_md5_etag

MODES = ("head", "readback", "recorded")


def attempt_artifacts(doc: dict[str, Any]) -> dict[str, str]:
    """{`outputs.<name>` | `telemetry.<name>`: art id} for every artifact an attempt manifest names."""
    out: dict[str, str] = {}
    for sec in ("outputs", "telemetry"):
        for name, v in (doc.get(sec) or {}).items():
            if isinstance(v, str) and v.startswith("art:"):
                out[f"{sec}.{name}"] = v
    return out


def _check_artifact(st: LocalStore, art_id: str, mode: str) -> dict[str, Any]:
    rep: dict[str, Any] = {"id": art_id, "kind": None, "preserved": False, "checked": 0, "errors": [], "verified": None}
    try:
        m = st.get_manifest(art_id)
    except Exception as e:  # noqa: BLE001  an unreadable / absent manifest is "not preserved", with the reason
        rep["errors"].append(f"manifest: {type(e).__name__}: {e}")
        return rep
    rep["kind"] = m.kind
    hex_ = ids.hex_of(art_id)
    if mode == "recorded":
        rec = st.index.replicas(art_id).get("remote") or {}
        if rec.get("verified_utc"):
            rep.update(preserved=True, verified=rec.get("verified"), verified_utc=rec["verified_utc"], checked=len(m.blobs()) + 1)
        else:
            rep["errors"].append("no verified remote replica recorded in the local catalog")
        return rep
    r = st.remote
    if r is None:
        rep["errors"].append(f"remote unusable: {st.remote_error}" if getattr(st, "remote_error", None) else "no remote configured")
        return rep
    methods: set[str] = set()
    # the manifest: always read back (small; its sha256 IS the id)
    mkey = manifest_key(hex_)
    head = r.head(mkey)
    if head is None:
        rep["errors"].append(f"{mkey}: absent")
    elif head["bytes"] != len(m.canonical()):
        rep["errors"].append(f"{mkey}: remote size {head['bytes']} != {len(m.canonical())}")
    else:
        err = st._readback(r, mkey, hex_)  # noqa: SLF001
        if err:
            rep["errors"].append(err)
        else:
            rep["checked"] += 1
    files = m.payload.files if m.payload and m.payload.type == "tree" else []
    for h, size in m.blobs():
        key = object_key(h)
        head = r.head(key)
        if head is None:
            rep["errors"].append(f"{key}: absent")
            continue
        if head["bytes"] != size:
            rep["errors"].append(f"{key}: remote size {head['bytes']} != {size}")
            continue
        if size > SINGLE_PUT_MAX:
            rep["errors"].append(f"{key}: > 5 GiB; per-part check is not implemented")
            continue
        local = st.object_path(h)
        if mode == "head" and local.is_file() and is_md5_etag(head.get("etag")):
            if head["etag"] == md5_file(local):
                rep["checked"] += 1
                methods.add(VERIFY_HEAD)
            else:
                rep["errors"].append(f"{key}: remote ETag {head['etag']} != local MD5")
            continue
        err = st._readback(r, key, h)  # noqa: SLF001
        if err:
            rep["errors"].append(err)
        else:
            rep["checked"] += 1
            methods.add(VERIFY_READBACK)
    if not rep["errors"]:
        rep["preserved"] = True
        rep["verified"] = ",".join(sorted(methods)) or VERIFY_READBACK
        rep["verified_utc"] = utc_now()
        st.index.set_replica(art_id, "remote", rep["verified_utc"], r.describe(), verified=rep["verified"])
        st.index.set_presence(art_id, remote=True)
    return rep


def check(st: LocalStore, targets: list[str], mode: str = "head") -> dict[str, Any]:
    """The gate over several targets.  `preserved` is True iff every target is: for a run id, its attempt manifest is on the remote
    and every artifact it names is preserved (an attempt naming no artifact is preserved once its manifest is); for an art id, the
    artifact is.  `not_preserved` lists what is not, with reasons."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    r = st.remote
    rep: dict[str, Any] = {"mode": mode, "remote": r.describe() if r else None, "targets": [], "artifacts": {}, "not_preserved": [], "preserved": False}
    if r is None and mode != "recorded":
        rep["reason"] = f"remote unusable: {st.remote_error}" if getattr(st, "remote_error", None) else "no remote configured"
    seen: dict[str, dict[str, Any]] = {}

    def art(a: str) -> dict[str, Any]:
        if a not in seen:
            seen[a] = _check_artifact(st, a, mode)
        return seen[a]

    for t in targets:
        if t.startswith("art:"):
            a = art(t)
            tr: dict[str, Any] = {"target": t, "kind": "artifact", "preserved": a["preserved"], "artifacts": {t: a}}
            if not a["preserved"]:
                rep["not_preserved"].append({"id": t, "kind": a["kind"], "errors": a["errors"]})
        elif ids.is_run_id(t):
            tr = {"target": t, "kind": "attempt", "preserved": False, "attempt_manifest": None, "artifacts": {}, "missing_names": []}
            doc = None
            try:
                doc = st.get_attempt(t)
            except Exception as e:  # noqa: BLE001
                tr["error"] = f"{type(e).__name__}: {e}"
            if doc is None:
                tr.setdefault("error", f"attempt {t} is not in the store")
                rep["not_preserved"].append({"id": t, "kind": "attempt", "errors": [tr["error"]]})
            else:
                if mode == "recorded":
                    tr["attempt_manifest"] = None                       # unknown offline; not counted against the gate
                    man_ok = True
                elif r is None:
                    man_ok = False
                    tr["attempt_manifest"] = False
                else:
                    man_ok = r.head(attempt_key(t)) is not None
                    tr["attempt_manifest"] = man_ok
                if not man_ok:
                    rep["not_preserved"].append({"id": t, "kind": "attempt", "errors": [f"{attempt_key(t)}: absent on the remote" if r else rep.get("reason", "no remote")]})
                named = attempt_artifacts(doc)
                for name, a in named.items():
                    ar = art(a)
                    tr["artifacts"][a] = {**ar, "name": name}
                    if not ar["preserved"]:
                        tr["missing_names"].append(name)
                        rep["not_preserved"].append({"id": a, "kind": ar["kind"], "name": f"{t}:{name}", "errors": ar["errors"]})
                tr["preserved"] = man_ok and all(x["preserved"] for x in tr["artifacts"].values())
                tr["n_artifacts"] = len(named)
        else:
            tr = {"target": t, "kind": "?", "preserved": False, "error": f"not an artifact or run id: {t!r}"}
            rep["not_preserved"].append({"id": t, "kind": "?", "errors": [tr["error"]]})
        rep["targets"].append(tr)
    rep["artifacts"] = seen
    rep["preserved"] = bool(targets) and all(t["preserved"] for t in rep["targets"])
    return rep


def format_report(rep: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for t in rep["targets"]:
        if t["kind"] == "attempt":
            if t.get("error"):
                lines.append(f"{t['target']}  attempt  NOT preserved: {t['error']}")
                continue
            man = t.get("attempt_manifest")
            lines.append(f"{t['target']}  attempt manifest  {'on remote' if man else ('not checked (recorded mode)' if man is None else 'ABSENT on remote')}")
            for a, ar in t["artifacts"].items():
                lines.append(f"  {ar['name']:<22} {a}  {str(ar['kind']):<22} "
                             + (f"PRESERVED ({ar.get('verified')})" if ar["preserved"] else "NOT preserved: " + "; ".join(ar["errors"] or ["?"])))
            if not t["artifacts"]:
                lines.append("  (no output artifacts named by this attempt)")
        elif t["kind"] == "artifact":
            ar = t["artifacts"][t["target"]]
            lines.append(f"{t['target']}  {str(ar['kind']):<22} " + (f"PRESERVED ({ar.get('verified')})" if ar["preserved"] else "NOT preserved: " + "; ".join(ar["errors"] or ["?"])))
        else:
            lines.append(f"{t['target']}  NOT preserved: {t.get('error')}")
    n_bad = len(rep["not_preserved"])
    if rep.get("reason") and not rep["preserved"]:
        lines.append(f"remote: {rep['reason']}")
    lines.append("PRESERVED" + (f" on {rep['remote']}" if rep.get("remote") else "") if rep["preserved"] else f"NOT PRESERVED: {n_bad} item(s) not verifiably on the remote")
    return lines


__all__ = ["MODES", "attempt_artifacts", "check", "format_report"]

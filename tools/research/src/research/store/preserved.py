"""The durability gate (F7): is every output of an attempt — or an artifact — AND every assertion about them verifiably on the remote?

    check(store, targets, mode="head") -> report; report["preserved"] is the gate

A target is a run id (every artifact its manifest names under `outputs` and `telemetry`, plus the attempt manifest itself) or
an `art:` id.  Three legs, each reported by name (`legs`: `manifest`, `artifacts`, `labels`):
  manifest   the attempt manifest is on the remote (HEAD);
  artifacts  an artifact is PRESERVED when the remote holds its manifest (read back, sha256 == id) and every blob is present under
             its content key with the manifest's size (one HEAD each, `JOBS` at a time) and, per `mode`:
             `head` (default; no blob body is downloaded) --
               multipart ETag (`<hex>-<n>`): the put's parts record `<key>.parts.json` names this sha256 and size, its parts are
                 the plan for that size and part size, and md5(part MD5s)-n == the ETag the service computed from the parts it
                 stored (`etag-multipart`);
               single-part ETag: == the local blob's MD5 when the blob is local (`etag-md5`), else present with the right size
                 (`head-size`) -- sound because objects are only created by `put`s that sign the payload sha256, which for a
                 single-part blob IS its key: the service refused any body that did not hash to it, and keys are never overwritten;
             `readback` = every blob downloaded and sha256'd (the explicit, expensive mode: hours for a 55 GB tree);
             `recorded` = trust the local catalog's verified remote replica (offline; what `push --verify`/`verify` recorded);
  labels     every local assertion about the attempt, its derivation and its named artifacts (`labels_sync.attempt_targets`) is
             listed on the remote under its content key and HEADs back with its canonical size (`labels_sync.verify_targets`) --
             a preserved Attempt is not preserved until its assertions are durable too (r20: labels lived only on the pods).
             Not checkable offline: `recorded` mode reports it as not checked and does not count it, like the attempt manifest.
Nothing is uploaded here: `research data push` is the repair (it pushes the labels leg too).

Without a usable remote (no store.toml, or credentials not in this environment) nothing is preserved, and the report says why.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import ids
from . import labels_sync as LS
from .local import (
    VERIFY_HEAD,
    VERIFY_MULTIPART,
    VERIFY_READBACK,
    VERIFY_SIZE,
    LocalStore,
    attempt_key,
    manifest_key,
    object_key,
    utc_now,
)
from .remote import md5_file
from .remote_s3 import PARTS_SIDECAR_SCHEMA, SINGLE_PUT_MAX, is_md5_etag, multipart_etag, parts_of, plan_parts, sidecar_key

MODES = ("head", "readback", "recorded")
JOBS = 32                   # blobs checked at once (HEADs; read-backs in `readback` mode) unless the remote says (`store.toml [remote] jobs`)


def attempt_artifacts(doc: dict[str, Any]) -> dict[str, str]:
    """{`outputs.<name>` | `telemetry.<name>`: art id} for every artifact an attempt manifest names."""
    out: dict[str, str] = {}
    for sec in ("outputs", "telemetry"):
        for name, v in (doc.get(sec) or {}).items():
            if isinstance(v, str) and v.startswith("art:"):
                out[f"{sec}.{name}"] = v
    return out


def _sidecar_error(st: LocalStore, r, key: str, sha: str, size: int, etag: str) -> str | None:
    """None when the put's parts record (`<key>.parts.json`, remote_s3.parts_sidecar) accounts for the multipart object HEAD reported:
    it names this blob's sha256 and size, its parts are the plan for that size and part size, and the multipart ETag of their MD5s is
    the ETag the service computed from the parts it stored.  Else the reason."""
    skey = sidecar_key(key)
    tmp = st.root / f".sidecar-{os.getpid()}-{threading.get_ident()}"
    try:
        r.get(skey, tmp)
        side = json.loads(tmp.read_bytes())
    except KeyError:
        return f"{key}: multipart ETag {etag} but no parts record {skey} (`--mode readback` verifies by content)"
    except (OSError, ValueError) as e:
        return f"{skey}: unreadable: {type(e).__name__}: {e}"
    finally:
        tmp.unlink(missing_ok=True)
    try:
        parts, pb = side["parts"], int(side["part_bytes"])
        plan = [n for _, _, n in plan_parts(size, pb)]
        why = ("schema" if side.get("schema") != PARTS_SIDECAR_SCHEMA else "sha256" if side.get("sha256") != sha else "bytes" if side.get("bytes") != size
               else "part plan" if [p["bytes"] for p in parts] != plan or [p["n"] for p in parts] != list(range(1, len(plan) + 1))
               else "ETag" if multipart_etag([p["md5"] for p in parts]) != etag or side.get("etag") != etag else None)
    except (KeyError, TypeError, ValueError) as e:
        why = f"malformed ({type(e).__name__}: {e})"
    return None if why is None else f"{key}: parts record {skey} does not account for the object ({why}; remote ETag {etag})"


def _check_blob(st: LocalStore, r, h: str, size: int, mode: str) -> tuple[str | None, str | None]:
    """(error, method) for one blob (module docstring, `artifacts`).  Thread-safe."""
    key = object_key(h)
    try:
        head = r.head(key)
        if head is None:
            return f"{key}: absent", None
        if head["bytes"] != size:
            return f"{key}: remote size {head['bytes']} != {size}", None
        if mode == "readback":
            return st._readback(r, key, h), VERIFY_READBACK  # noqa: SLF001
        etag = head.get("etag")
        if parts_of(etag) is not None:
            return _sidecar_error(st, r, key, h, size, etag), VERIFY_MULTIPART
        if size > SINGLE_PUT_MAX:
            return f"{key}: {size} bytes but ETag {etag!r} is not a multipart ETag (no single PUT carries > 5 GiB)", None
        local = st.object_path(h)
        if is_md5_etag(etag) and local.is_file():
            return (None if etag == md5_file(local) else f"{key}: remote ETag {etag} != local MD5"), VERIFY_HEAD
        return None, VERIFY_SIZE
    except Exception as e:  # noqa: BLE001  one object's failure is reported, the rest are still checked
        return f"{key}: {type(e).__name__}: {e}", None


def _check_artifact(st: LocalStore, art_id: str, mode: str, jobs: int | None = None) -> dict[str, Any]:
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
    blobs = m.blobs()
    with ThreadPoolExecutor(max_workers=max(1, min(jobs or getattr(r, "jobs", None) or JOBS, len(blobs) or 1))) as ex:
        results = list(ex.map(lambda hs: _check_blob(st, r, hs[0], hs[1], mode), blobs))
    for err, method in results:
        if err:
            rep["errors"].append(err)
        else:
            rep["checked"] += 1
            methods.add(method)
    if not rep["errors"]:
        rep["preserved"] = True
        rep["verified"] = ",".join(sorted(methods)) or VERIFY_READBACK
        rep["verified_utc"] = utc_now()
        st.index.set_replica(art_id, "remote", rep["verified_utc"], r.describe(), verified=rep["verified"])
        st.index.set_presence(art_id, remote=True)
    return rep


def check(st: LocalStore, targets: list[str], mode: str = "head", jobs: int | None = None) -> dict[str, Any]:
    """The gate over several targets.  `preserved` is True iff every target is: for a run id, its attempt manifest is on the remote
    and every artifact it names is preserved (an attempt naming no artifact is preserved once its manifest is); for an art id, the
    artifact is.  `not_preserved` lists what is not, with reasons.  `jobs`: blobs checked at once (default the remote's, else `JOBS`)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    r = st.remote
    rep: dict[str, Any] = {"mode": mode, "remote": r.describe() if r else None, "targets": [], "artifacts": {}, "not_preserved": [], "preserved": False}
    if r is None and mode != "recorded":
        rep["reason"] = f"remote unusable: {st.remote_error}" if getattr(st, "remote_error", None) else "no remote configured"
    seen: dict[str, dict[str, Any]] = {}

    def art(a: str) -> dict[str, Any]:
        if a not in seen:
            seen[a] = _check_artifact(st, a, mode, jobs)
        return seen[a]

    def labels_leg(t: str, label_targets: list[str]) -> tuple[bool | None, dict[str, Any] | None]:
        """(ok | None when not checkable, report).  Missing or wrong-sized assertions land in not_preserved as kind `labels`."""
        if mode == "recorded" or r is None:
            return None, None
        try:
            lrep = LS.verify_targets(st, label_targets)
        except Exception as e:  # noqa: BLE001
            rep["not_preserved"].append({"id": t, "kind": "labels", "errors": [f"labels: {type(e).__name__}: {e}"]})
            return False, {"error": f"{type(e).__name__}: {e}"}
        errors = [f"{k}: absent on the remote" for tt in lrep["targets"].values() for k in tt["missing"]]
        errors += [e for tt in lrep["targets"].values() for e in tt["errors"]]
        if errors:
            rep["not_preserved"].append({"id": t, "kind": "labels", "errors": errors})
        return lrep["durable"], {"local": sum(x["local"] for x in lrep["targets"].values()), "remote": sum(x["remote"] for x in lrep["targets"].values()),
                                 "missing": sum(len(x["missing"]) for x in lrep["targets"].values()), "targets": lrep["targets"], "errors": errors}

    for t in targets:
        if t.startswith("art:"):
            a = art(t)
            tr: dict[str, Any] = {"target": t, "kind": "artifact", "preserved": a["preserved"], "artifacts": {t: a}}
            if not a["preserved"]:
                rep["not_preserved"].append({"id": t, "kind": a["kind"], "errors": a["errors"]})
            lab_ok, tr["labels"] = labels_leg(t, [t])
            tr["legs"] = {"artifacts": a["preserved"], "labels": lab_ok}
            tr["preserved"] = a["preserved"] and lab_ok is not False
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
                arts_ok = all(x["preserved"] for x in tr["artifacts"].values())
                lab_ok, tr["labels"] = labels_leg(t, LS.attempt_targets(doc))
                tr["legs"] = {"manifest": man_ok, "artifacts": arts_ok, "labels": lab_ok}
                tr["preserved"] = man_ok and arts_ok and lab_ok is not False
                tr["n_artifacts"] = len(named)
        else:
            tr = {"target": t, "kind": "?", "preserved": False, "error": f"not an artifact or run id: {t!r}"}
            rep["not_preserved"].append({"id": t, "kind": "?", "errors": [tr["error"]]})
        rep["targets"].append(tr)
    rep["artifacts"] = seen
    rep["preserved"] = bool(targets) and all(t["preserved"] for t in rep["targets"])
    return rep


def failures(t: dict[str, Any], rep: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Why one target of `check` is not preserved, one entry per failing check: {"leg": attempt|manifest|artifacts|labels, "what":
    the attempt / object key / `<name> <art id>` / label target it concerns, "why"}.  Empty when the target is preserved."""
    out: list[dict[str, Any]] = []
    tid = t["target"]
    if t.get("error"):
        return [{"leg": "attempt" if t["kind"] == "attempt" else "target", "what": tid, "why": t["error"]}]
    legs = t.get("legs") or {}
    if legs.get("manifest") is False:
        out.append({"leg": "manifest", "what": attempt_key(tid),
                    "why": "absent on the remote" if (rep or {}).get("remote") else ((rep or {}).get("reason") or "no remote")})
    for a, ar in (t.get("artifacts") or {}).items():
        if not ar["preserved"]:
            errs = ar.get("errors") or ["?"]
            out.append({"leg": "artifacts", "what": f"{ar['name']} {a}" if ar.get("name") else a,
                        "why": "; ".join(errs[:2]) + (f" (+{len(errs) - 2} more)" if len(errs) > 2 else "")})
    if legs.get("labels") is False:
        lab = t.get("labels") or {}
        if lab.get("error"):
            out.append({"leg": "labels", "what": tid, "why": lab["error"]})
        for lt, x in (lab.get("targets") or {}).items():
            if x["missing"]:
                out.append({"leg": "labels", "what": lt, "why": f"{len(x['missing'])} of {x['local']} local assertion(s) absent on the remote "
                                                               f"(first: {x['missing'][0]})"})
            for e in x["errors"][:2]:
                out.append({"leg": "labels", "what": lt, "why": e})
    return out


def _labels_line(t: dict[str, Any]) -> str:
    legs = t.get("legs") or {}
    lab = t.get("labels")
    if legs.get("labels") is None:
        return f"  {'labels':<22} not checked" + (" (recorded mode)" if lab is None and legs else "")
    if lab and lab.get("error"):
        return f"  {'labels':<22} NOT durable: {lab['error']}"
    n_local, n_remote, n_missing = lab["local"], lab["remote"], lab["missing"]
    extra = n_remote - (n_local - n_missing)
    tail = f"{n_local} local, {n_remote} on remote" + (f" ({extra} asserted elsewhere)" if extra > 0 else "")
    if legs["labels"]:
        return f"  {'labels':<22} DURABLE  {tail}"
    return f"  {'labels':<22} NOT durable: {n_missing} local assertion(s) absent on the remote  {tail}" + ("; " + "; ".join(lab["errors"][:3]) if lab.get("errors") and n_missing == 0 else "")


def _failing_legs(rep: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for t in rep["targets"]:
        for leg, ok in (t.get("legs") or {}).items():
            if ok is False and leg not in out:
                out.append(leg)
        if t.get("error") and "attempt" not in out and t["kind"] == "attempt":
            out.append("attempt")
    return out


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
            lines.append(_labels_line(t))
        elif t["kind"] == "artifact":
            ar = t["artifacts"][t["target"]]
            lines.append(f"{t['target']}  {str(ar['kind']):<22} " + (f"PRESERVED ({ar.get('verified')})" if ar["preserved"] else "NOT preserved: " + "; ".join(ar["errors"] or ["?"])))
            lines.append(_labels_line(t))
        else:
            lines.append(f"{t['target']}  NOT preserved: {t.get('error')}")
    n_bad = len(rep["not_preserved"])
    if rep.get("reason") and not rep["preserved"]:
        lines.append(f"remote: {rep['reason']}")
    legs = _failing_legs(rep)
    lines.append("PRESERVED" + (f" on {rep['remote']}" if rep.get("remote") else "") if rep["preserved"]
                 else f"NOT PRESERVED: {n_bad} item(s) not verifiably on the remote" + (f"  (legs failing: {', '.join(legs)})" if legs else ""))
    return lines


__all__ = ["MODES", "attempt_artifacts", "check", "failures", "format_report"]

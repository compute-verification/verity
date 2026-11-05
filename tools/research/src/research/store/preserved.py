"""The durability gate (F7): is every output of an attempt — or an artifact — AND every assertion about them verifiably on the remote?

    check(store, targets, mode="head") -> report; report["preserved"] is the gate

A target is a run id (every artifact its manifest names under `outputs` and `telemetry`, plus the attempt manifest itself) or
an `art:` id.  Three legs, each reported by name (`legs`: `manifest`, `artifacts`, `labels`):
  manifest   the attempt manifest is on the remote (HEAD);
  artifacts  an artifact is PRESERVED when the remote holds its manifest (read back, sha256 == id) and every blob: remote size ==
             manifest size and, per `mode`, `head` = when the blob is local, the remote ETag == its MD5 (single-part) or the
             `md5(md5s)-n` its parts yield (multipart, `etag-multipart`), otherwise a full read-back sha256 (any size);
             `readback` = always read back; `recorded` = trust the local catalog's
             verified remote replica (what `push --verify`/`verify` recorded), and where there is none (an upload made from a
             pod or another machine) run the `head` check against the remote, which records the replica when it passes -- so the
             verdict does not depend on which machine uploaded; offline only for artifacts already recorded;
  labels     every local assertion about the attempt, its derivation and its named artifacts (`labels_sync.attempt_targets`) is
             listed on the remote under its content key and HEADs back with its canonical size (`labels_sync.verify_targets`) --
             a preserved Attempt is not preserved until its assertions are durable too (r20: labels lived only on the pods).
             Not checkable offline: `recorded` mode reports it as not checked and does not count it, like the attempt manifest.
Nothing is uploaded here: `research data push` is the repair (it pushes the labels leg too).

Without a usable remote (no store.toml, or credentials not in this environment) nothing is preserved, and the report says why.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from . import ids
from . import labels_sync as LS
from .local import (
    VERIFY_HEAD,
    VERIFY_MULTIPART,
    VERIFY_READBACK,
    LocalStore,
    attempt_key,
    manifest_key,
    object_key,
    utc_now,
)
from .remote import md5_file
from .remote_s3 import PART_SIZE, choose_part_size_for, hash_parts, is_md5_etag, multipart_etag, parts_of, plan_parts

MODES = ("head", "readback", "recorded")
#: in a head-mode verdict: the attempt key is write-once on the remote, so no push makes this run preserved (store.reconcile)
DIFFERS = "the remote holds a DIFFERENT attempt document"


def attempt_artifacts(doc: dict[str, Any]) -> dict[str, str]:
    """{`outputs.<name>` | `telemetry.<name>`: art id} for every artifact an attempt manifest names."""
    out: dict[str, str] = {}
    for sec in ("outputs", "telemetry"):
        for name, v in (doc.get(sec) or {}).items():
            if isinstance(v, str) and v.startswith("art:"):
                out[f"{sec}.{name}"] = v
    return out


def _local_multipart_etag_matches(path, size: int, etag: str, part_size: int) -> bool:
    """Does the local blob yield the remote's multipart ETag `md5(md5s)-n`?  The part size is not on the object: try the one an
    upload from here picks, then an even split into `n` parts (an upload made elsewhere, as `put_multipart` does)."""
    n = parts_of(etag)
    for ps in dict.fromkeys((choose_part_size_for(size, part_size), -(-size // n))):
        if len(plan_parts(size, ps)) == n and multipart_etag([p["md5"] for p in hash_parts(path, ps)]) == etag:
            return True
    return False


def _check_artifact(st: LocalStore, art_id: str, mode: str, head_fallback: bool = True) -> dict[str, Any]:
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
            return rep
        if st.remote is None or not head_fallback:
            rep["errors"].append("no verified remote replica recorded in the local catalog"
                                 + (f", and the remote is unusable here: {st.remote_error}" if getattr(st, "remote_error", None) else ""))
            return rep
        mode = "head"
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
    for h, size in m.blobs():
        key = object_key(h)
        head = r.head(key)
        if head is None:
            rep["errors"].append(f"{key}: absent")
            continue
        if head["bytes"] != size:
            rep["errors"].append(f"{key}: remote size {head['bytes']} != {size}")
            continue
        local = st.object_path(h)
        if mode == "head" and local.is_file() and is_md5_etag(head.get("etag")):
            if head["etag"] == md5_file(local):
                rep["checked"] += 1
                methods.add(VERIFY_HEAD)
            else:
                rep["errors"].append(f"{key}: remote ETag {head['etag']} != local MD5")
            continue
        if mode == "head" and local.is_file() and parts_of(head.get("etag")):
            if _local_multipart_etag_matches(local, size, head["etag"], getattr(r, "part_size", PART_SIZE)):
                rep["checked"] += 1
                methods.add(VERIFY_MULTIPART)
            else:
                rep["errors"].append(f"{key}: remote multipart ETag {head['etag']} is not what the local blob yields")
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


def check(st: LocalStore, targets: list[str], mode: str = "head", head_fallback: bool = True, *, jobs: int = 1,
          on_target: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """The gate over several targets (`head_fallback=False`: `recorded` stays offline, a caller that bounds what it reads back).  `preserved` is True iff every target is: for a run id, its attempt manifest is on the remote
    and every artifact it names is preserved (an attempt naming no artifact is preserved once its manifest is); for an art id, the
    artifact is.  `not_preserved` lists what is not, with reasons.  `jobs` targets are checked at once (an artifact two targets name
    is checked once); `on_target(target report)` is called, on this thread, as each one finishes.  The report is in `targets` order
    whatever the completion order, and an exception while checking one target is that target's `error`, not the others'."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    r = st.remote
    rep: dict[str, Any] = {"mode": mode, "remote": r.describe() if r else None, "targets": [], "artifacts": {}, "not_preserved": [], "preserved": False}
    if r is None and mode != "recorded":
        rep["reason"] = f"remote unusable: {st.remote_error}" if getattr(st, "remote_error", None) else "no remote configured"
    seen: dict[str, dict[str, Any]] = {}
    art_locks: dict[str, threading.Lock] = {}
    locks_lock = threading.Lock()

    def art(a: str) -> dict[str, Any]:
        with locks_lock:
            lock = art_locks.setdefault(a, threading.Lock())
        with lock:
            if a not in seen:
                seen[a] = _check_artifact(st, a, mode, head_fallback)
            return seen[a]

    def labels_leg(t: str, label_targets: list[str], bad: list[dict[str, Any]]) -> tuple[bool | None, dict[str, Any] | None]:
        """(ok | None when not checkable, report).  Missing or wrong-sized assertions land in `bad` as kind `labels`."""
        if mode == "recorded" or r is None:
            return None, None
        try:
            lrep = LS.verify_targets(st, label_targets)
        except Exception as e:  # noqa: BLE001
            bad.append({"id": t, "kind": "labels", "errors": [f"labels: {type(e).__name__}: {e}"]})
            return False, {"error": f"{type(e).__name__}: {e}"}
        errors = [f"{k}: absent on the remote" for tt in lrep["targets"].values() for k in tt["missing"]]
        errors += [e for tt in lrep["targets"].values() for e in tt["errors"]]
        if errors:
            bad.append({"id": t, "kind": "labels", "errors": errors})
        return lrep["durable"], {"local": sum(x["local"] for x in lrep["targets"].values()), "remote": sum(x["remote"] for x in lrep["targets"].values()),
                                 "missing": sum(len(x["missing"]) for x in lrep["targets"].values()), "targets": lrep["targets"], "errors": errors}

    def one(t: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        bad: list[dict[str, Any]] = []
        try:
            return target(t, bad), bad
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            return {"target": t, "kind": "?", "preserved": False, "error": err}, [*bad, {"id": t, "kind": "?", "errors": [err]}]

    def target(t: str, bad: list[dict[str, Any]]) -> dict[str, Any]:
        if ids.is_short_art(t):
            hits = st.resolve_art(t)
            if len(hits) != 1:
                err = f"{t} is {'ambiguous' if hits else 'unknown'} (local catalog{' and remote' if r is not None else ''})"
                bad.append({"id": t, "kind": "?", "errors": [err]})
                return {"target": t, "kind": "?", "preserved": False, "error": err}
            t = hits[0]
        if t.startswith("art:"):
            a = art(t)
            tr: dict[str, Any] = {"target": t, "kind": "artifact", "preserved": a["preserved"], "artifacts": {t: a}}
            if not a["preserved"]:
                bad.append({"id": t, "kind": a["kind"], "errors": a["errors"]})
            lab_ok, tr["labels"] = labels_leg(t, [t], bad)
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
                bad.append({"id": t, "kind": "attempt", "errors": [tr["error"]]})
            else:
                if mode == "recorded":
                    tr["attempt_manifest"] = None                       # unknown offline; not counted against the gate
                    man_ok = True
                elif r is None:
                    man_ok = False
                    tr["attempt_manifest"] = False
                else:
                    hd = r.head(attempt_key(t))
                    local = st.attempt_path(t) if hasattr(st, "attempt_path") else None
                    differs = (hd is not None and local is not None and local.exists() and
                               (hd["bytes"] != local.stat().st_size or (is_md5_etag(hd.get("etag")) and hd["etag"] != md5_file(local))))
                    man_ok = hd is not None and not differs
                    tr["attempt_manifest"] = man_ok
                    if differs:
                        tr["attempt_manifest_differs"] = True
                if not man_ok:
                    why = (f"{attempt_key(t)}: {DIFFERS} for this run id (`research data "
                           f"reconcile-attempts {t}`)" if tr.get("attempt_manifest_differs") else
                           f"{attempt_key(t)}: absent on the remote" if r else rep.get("reason", "no remote"))
                    bad.append({"id": t, "kind": "attempt", "errors": [why]})
                named = attempt_artifacts(doc)
                for name, a in named.items():
                    ar = art(a)
                    tr["artifacts"][a] = {**ar, "name": name}
                    if not ar["preserved"]:
                        tr["missing_names"].append(name)
                        bad.append({"id": a, "kind": ar["kind"], "name": f"{t}:{name}", "errors": ar["errors"]})
                arts_ok = all(x["preserved"] for x in tr["artifacts"].values())
                lab_ok, tr["labels"] = labels_leg(t, LS.attempt_targets(doc), bad)
                tr["legs"] = {"manifest": man_ok, "artifacts": arts_ok, "labels": lab_ok}
                tr["preserved"] = man_ok and arts_ok and lab_ok is not False
                tr["n_artifacts"] = len(named)
        else:
            tr = {"target": t, "kind": "?", "preserved": False, "error": f"not an artifact or run id: {t!r}"}
            bad.append({"id": t, "kind": "?", "errors": [tr["error"]]})
        return tr

    done: list[tuple[dict[str, Any], list[dict[str, Any]]] | None] = [None] * len(targets)
    if jobs <= 1 or len(targets) <= 1:
        for i, t in enumerate(targets):
            done[i] = one(t)
            if on_target:
                on_target(done[i][0])
    else:
        _ = st.index                            # opened here: two workers opening it first would each open (and migrate) a catalog
        ex = ThreadPoolExecutor(max_workers=min(jobs, len(targets)))
        try:
            futs = {ex.submit(one, t): i for i, t in enumerate(targets)}
            for f in as_completed(futs):
                done[futs[f]] = f.result()
                if on_target:
                    on_target(done[futs[f]][0])
        finally:
            ex.shutdown(cancel_futures=True)
    for tr, bad in done:
        rep["targets"].append(tr)
        rep["not_preserved"] += bad
    rep["artifacts"] = {a: seen[a] for tr in rep["targets"] for a in tr.get("artifacts", {})}
    rep["preserved"] = bool(targets) and all(t["preserved"] for t in rep["targets"])
    return rep


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


def format_target(t: dict[str, Any]) -> list[str]:
    """One target's lines of the report (`check`'s `on_target` prints them as it finishes)."""
    if t["kind"] == "attempt":
        if t.get("error"):
            return [f"{t['target']}  attempt  NOT preserved: {t['error']}"]
        man = t.get("attempt_manifest")
        lines = [f"{t['target']}  attempt manifest  " + ("on remote" if man else "not checked (recorded mode)" if man is None else
                 "DIFFERS from the remote's copy (research data reconcile-attempts)" if t.get("attempt_manifest_differs") else "ABSENT on remote")]
        for a, ar in t["artifacts"].items():
            lines.append(f"  {ar['name']:<22} {a}  {str(ar['kind']):<22} "
                         + (f"PRESERVED ({ar.get('verified')})" if ar["preserved"] else "NOT preserved: " + "; ".join(ar["errors"] or ["?"])))
        if not t["artifacts"]:
            lines.append("  (no output artifacts named by this attempt)")
        return [*lines, _labels_line(t)]
    if t["kind"] == "artifact":
        ar = t["artifacts"][t["target"]]
        return [f"{t['target']}  {str(ar['kind']):<22} " + (f"PRESERVED ({ar.get('verified')})" if ar["preserved"] else "NOT preserved: " + "; ".join(ar["errors"] or ["?"])),
                _labels_line(t)]
    return [f"{t['target']}  NOT preserved: {t.get('error')}"]


def format_summary(rep: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    n_bad = len(rep["not_preserved"])
    if rep.get("reason") and not rep["preserved"]:
        lines.append(f"remote: {rep['reason']}")
    legs = _failing_legs(rep)
    lines.append("PRESERVED" + (f" on {rep['remote']}" if rep.get("remote") else "") if rep["preserved"]
                 else f"NOT PRESERVED: {n_bad} item(s) not verifiably on the remote" + (f"  (legs failing: {', '.join(legs)})" if legs else ""))
    return lines


def format_report(rep: dict[str, Any]) -> list[str]:
    return [line for t in rep["targets"] for line in format_target(t)] + format_summary(rep)


__all__ = ["MODES", "attempt_artifacts", "check", "format_report", "format_summary", "format_target"]

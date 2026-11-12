"""Withdraw artifacts from the shared store (`research data withdraw`, README §7.4): delete the blobs of artifacts that hold values
which must never be shared, whatever cites them, and leave a `withdrawn` label on each as its tombstone.

`gc` never deletes a cited artifact's blobs, and private values that were published by mistake are often cited (the
sealed bundle and the weights key of a served run, in three cited arts).  The manifest, the attempts and the labels stay, so a citation still
resolves and `fetch` reports the payload absent; the label says why.  A blob that an artifact outside the withdrawn set also holds
is not deleted: deleting it would break that artifact, and holding the same bytes it holds the same values, so the plan names it
and the caller withdraws it too or decides otherwise.  Every manifest is read (through gc's cache) to find those holders.

With `paths` only the files under those prefixes of the withdrawn arts' trees go (a run's `out/sampled-proofs/served/` in its
attempt record, say); a blob one of their other files also holds is kept and named like any other holder's.  The label's value
is then `"<reason> (only <prefix>, ...)"` (`label_value`); `withdrawals` reads it back, so `research data preserved` reports
those files withdrawn rather than lost, and `research data put --tree` refuses to put them back (`refused`).
"""
from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import vocab
from .gc import JOBS, MANIFEST_KEY, GcError, Reader, _parse_manifest

_ONLY = re.compile(r"(?P<reason>.*) \(only (?P<paths>[^()]+)\)", re.S)


@dataclass
class Plan:
    """`blobs`: blob hex -> bytes, held only by the withdrawn arts (deleted on apply); `shared`: blob hex -> the other arts holding it
    (kept); `missing`: requested ids with no manifest."""

    arts: list[str]
    blobs: dict[str, int] = field(default_factory=dict)
    shared: dict[str, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"arts": self.arts, "missing": self.missing, "blobs": len(self.blobs), "bytes": sum(self.blobs.values()),
                "shared": {b: holders for b, holders in sorted(self.shared.items())}, **({"paths": self.paths} if self.paths else {})}


def _files(remote: Any, h: str, tmp: Path) -> list[tuple[str, str, int]]:
    """(path, blob hex, bytes) of each file in art `h`'s payload."""
    local = tmp / f"{h}.json"
    remote.get(f"manifests/{h}.json", local)
    payload = json.loads(local.read_text()).get("payload") or {}
    if payload.get("type") == "file":
        return [(str(payload.get("path") or ""), payload["sha256"], int(payload["bytes"]))]
    return [(str(f["path"]), f["sha256"], int(f["bytes"])) for f in payload.get("files") or []]


def _under(path: str, prefixes: list[str]) -> bool:
    return any(path == p.rstrip("/") or path.startswith(p.rstrip("/") + "/") for p in prefixes)


def label_value(reason: str, paths: list[str] | None) -> str:
    """The `withdrawn` label's value: the reason, and with `--path` the prefixes it covers."""
    return f"{reason} (only {', '.join(paths)})" if paths else reason


def withdrawals(st: Any, art_id: str) -> list[dict[str, Any]]:
    """Each `withdrawn` label on `art_id` in this store's labels, oldest first: {reason, paths (None: the whole art), by, ref, ts}.
    Pull the remote's first (`labels_sync.read_through`): a withdrawal is made on another machine."""
    out = []
    for lab in st.labels(art_id, key=vocab.WITHDRAWN):
        v = lab.value if isinstance(lab.value, str) else json.dumps(lab.value)
        m = _ONLY.fullmatch(v)
        out.append({"reason": m["reason"] if m else v, "paths": m["paths"].split(", ") if m else None, "by": lab.by, "ref": lab.ref,
                    "ts": lab.ts})
    return out


def covering(ws: list[dict[str, Any]], path: str) -> dict[str, Any] | None:
    """The newest of `withdrawals` that covers the payload file `path`, or None."""
    return next((w for w in reversed(ws) if w["paths"] is None or _under(path, w["paths"])), None)


def file_paths(manifest: Any) -> dict[str, list[str]]:
    """{blob hex: the payload paths holding it} (a file payload's path is its name)."""
    p = manifest.payload
    if p is None:
        return {}
    if p.type == "file":
        return {p.sha256: [p.name]}
    out: dict[str, list[str]] = {}
    for f in p.files or []:
        out.setdefault(f.sha256, []).append(f.path)
    return out


def refused(st: Any, run_id: str, paths: list[str], warn: Callable[[str], None]) -> list[tuple[str, str, dict[str, Any]]]:
    """(path, art, withdrawal) for each of `paths` that a `withdrawn` label of an artifact of run `run_id` covers: the arts its
    attempt names and the catalog's arts with `meta.run_id` = `run_id`, their labels pulled from the remote first.  A label
    without `(only ...)` covers the paths that art's payload holds."""
    from . import labels_sync as LS  # noqa: PLC0415
    from .preserved import attempt_artifacts  # noqa: PLC0415

    arts = [r["id"] for r in st.index.select_artifacts(where={"run_id": run_id})]
    try:
        doc = st.get_attempt(run_id)
    except Exception as e:  # noqa: BLE001  an unreachable remote: the catalog's arts still answer
        warn(f"attempt {run_id}: {type(e).__name__}: {e}; checking only this catalog's artifacts of the run")
        doc = None
    arts = list(dict.fromkeys(arts + list(attempt_artifacts(doc).values() if doc else [])))
    if not arts:
        return []
    LS.read_through(st, arts, warn)
    out = []
    for a in arts:
        if not (ws := withdrawals(st, a)):
            continue
        held: set[str] = set()
        if any(w["paths"] is None for w in ws):
            try:
                held = {q for qs in file_paths(st.get_manifest(a)).values() for q in qs}
            except Exception as e:  # noqa: BLE001
                warn(f"{a} is withdrawn whole, but its manifest is unreadable here ({type(e).__name__}: {e}); its paths are not checked")
        for path in paths:
            if w := next((w for w in reversed(ws) if (path in held if w["paths"] is None else _under(path, w["paths"]))), None):
                out.append((path, a, w))
    return out


def _hex(art: str) -> str:
    h = art.removeprefix("art:")
    if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError(f"{art}: give the full art: id (64 hex digits)")
    return h


def plan(remote: Any, cache: Path, arts: list[str], *, jobs: int = JOBS, work: Path | None = None,
         log: Callable[[str], None] | None = None, paths: list[str] | None = None) -> Plan:
    say = log or (lambda s: None)
    want = sorted({_hex(a) for a in arts})
    listed = sorted(m.group(1) for k in remote.list("manifests/") if (m := MANIFEST_KEY.fullmatch(k)))
    p = Plan(arts=[f"art:{h}" for h in want], missing=[f"art:{h}" for h in want if h not in set(listed)], paths=list(paths or []))
    if p.missing:
        return p
    rd = Reader(remote, cache, jobs)
    with tempfile.TemporaryDirectory(prefix="store-withdraw-", dir=work) as tmp:
        db = sqlite3.connect(Path(tmp) / "blobs.sqlite")
        db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE TABLE b(art TEXT, digest TEXT, size INTEGER);")
        keys = [f"manifests/{h}.json" for h in listed]
        for i, (k, parsed) in enumerate(rd.each(keys, _parse_manifest)):
            h = k[len("manifests/"):-len(".json")]
            db.executemany("INSERT INTO b VALUES (?,?,?)", ((h, d.hex(), s) for d, s in parsed[5]))
            if i % 20000 == 0 and i:
                say(f"{i} manifests read")
        db.execute("CREATE INDEX bd ON b(digest)")
        db.execute("CREATE TABLE w(art TEXT PRIMARY KEY)")
        db.executemany("INSERT INTO w VALUES (?)", ((h,) for h in want))
        inside: set[str] | None = None
        outside: dict[str, set[str]] = {}
        if p.paths:
            inside = set()
            for h in want:
                for path, digest, _ in _files(remote, h, Path(tmp)):
                    (inside.add(digest) if _under(path, p.paths) else outside.setdefault(digest, set()).add(f"art:{h}"))
        for digest, size in db.execute("SELECT DISTINCT b.digest, b.size FROM b JOIN w ON w.art = b.art").fetchall():
            if inside is not None and digest not in inside:
                continue
            others = [f"art:{a}" for (a,) in db.execute("SELECT DISTINCT art FROM b WHERE digest = ? AND art NOT IN (SELECT art FROM w)",
                                                        (digest,))] + sorted(f"{a} (outside --path)" for a in outside.get(digest, ()))
            if others:
                p.shared[digest] = sorted(others)
            else:
                p.blobs[digest] = int(size)
        db.close()
    say(f"{len(p.arts)} artifact(s): {len(p.blobs)} blob(s) to delete, {sum(p.blobs.values()) / 2**30:.2f} GiB; "
        f"{len(p.shared)} held by other artifacts too, kept")
    return p


def apply(remote: Any, p: Plan, *, log: Callable[[str], None] | None = None) -> dict[str, int]:
    """Delete the plan's blobs (and their part lists); returns {deleted, absent}."""
    if p.missing:
        raise GcError(f"no manifest for {', '.join(p.missing)}; nothing deleted")
    say = log or (lambda s: None)
    deleted = absent = 0
    for b in sorted(p.blobs):
        if remote.delete(f"objects/sha256/{b}"):
            deleted += 1
        else:
            absent += 1
        remote.delete(f"objects/sha256/{b}.parts.json")
    say(f"deleted {deleted} blob(s); {absent} already absent")
    return {"deleted": deleted, "absent": absent}

"""Withdraw artifacts from the shared store (`research data withdraw`, README §7.4): delete the blobs of artifacts that hold values
which must never be shared, whatever cites them, and leave a `withdrawn` label on each as its tombstone.

`gc` never deletes a cited artifact's blobs, and private values that were published by mistake are often cited (the
sealed bundle and the weights key of a served run, in three cited arts).  The manifest, the attempts and the labels stay, so a citation still
resolves and `fetch` reports the payload absent; the label says why.  A blob that an artifact outside the withdrawn set also holds
is not deleted: deleting it would break that artifact, and holding the same bytes it holds the same values, so the plan names it
and the caller withdraws it too or decides otherwise.  Every manifest is read (through gc's cache) to find those holders.
"""
from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .gc import JOBS, MANIFEST_KEY, GcError, Reader, _parse_manifest


@dataclass
class Plan:
    """`blobs`: blob hex -> bytes, held only by the withdrawn arts (deleted on apply); `shared`: blob hex -> the other arts holding it
    (kept); `missing`: requested ids with no manifest."""

    arts: list[str]
    blobs: dict[str, int] = field(default_factory=dict)
    shared: dict[str, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"arts": self.arts, "missing": self.missing, "blobs": len(self.blobs), "bytes": sum(self.blobs.values()),
                "shared": {b: holders for b, holders in sorted(self.shared.items())}}


def _hex(art: str) -> str:
    h = art.removeprefix("art:")
    if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError(f"{art}: give the full art: id (64 hex digits)")
    return h


def plan(remote: Any, cache: Path, arts: list[str], *, jobs: int = JOBS, work: Path | None = None,
         log: Callable[[str], None] | None = None) -> Plan:
    say = log or (lambda s: None)
    want = sorted({_hex(a) for a in arts})
    listed = sorted(m.group(1) for k in remote.list("manifests/") if (m := MANIFEST_KEY.fullmatch(k)))
    p = Plan(arts=[f"art:{h}" for h in want], missing=[f"art:{h}" for h in want if h not in set(listed)])
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
        for digest, size in db.execute("SELECT DISTINCT b.digest, b.size FROM b JOIN w ON w.art = b.art").fetchall():
            others = [f"art:{a}" for (a,) in db.execute("SELECT DISTINCT art FROM b WHERE digest = ? AND art NOT IN (SELECT art FROM w)",
                                                        (digest,))]
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

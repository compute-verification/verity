"""Retention of a node's data (`research/retention/v0`): nothing under a node's /workspace is deleted without a record that allows it.

    research keep PATH --owner @LANE --expires DATE|never --priority low|normal|high|keep [--regenerate CMD] [--art ART | --preserve]
                       [--why TEXT] [--by URL] [--root R] [--store S]
    research keep show PATH [--root R]
    research retention report [--root R] [--depth N] [--json]
    research retention gc [--root R] [--policy default] [--high-pct 80] [--low-pct 75] [--force] [--jobs-dir D]... [--store S] [--json]
                          [--apply]
    research retention rm PATH... --approved-by @OWNER --ref REF [--by URL] [--root R] [--jobs-dir D]... [--store S] [--json] [--apply]

A record is a sidecar: `<dir>/.retention.json` for a directory, `<file>.retention.json` beside a file, so it travels with its data
and `cat` reads it. A directory its owner made read-only has its record beside it, `<dir>.retention.json`; one with both is invalid. It covers everything beneath its path: the walk never descends into a recorded path. Its fields are FIELDS:
`path` (relative to the root), `node`, `owner` (a lane handle), `by`, `created`, `expires` (UTC, or null for never), `priority`
(PRIORITIES), `preserved` (an `art:` id once the data is on the remote, or null), `regenerate` (how to rebuild it, or null), `why`.

A directory whose `.custody` marker (store.custody) says `preserved: true` and which has no sidecar has an implicit record
(`implicit`): owner its job.json's lane, else its campaign, else `runner`; `preserved` the marker's run record; `expires`
RUN_EXPIRY after the marker's `verified_utc`; priority `normal`. Nothing is written for it, and a sidecar always wins.

Policy `default` (POLICY) acts when the root's filesystem is over high-pct (or with --force) and deletes, lowest priority first and
then the oldest expiry, until it is under low-pct. A filesystem's use is the larger of its blocks' and its inodes' (`inode_usage`):
a disk out of inodes refuses every write as one out of blocks does (vy-dc-h200, 26 Oct: inodes at 100%, blocks at 32%). A path qualifies only with a valid record (explicit or implicit) recorded on
this node that is expired, `low` or `normal`, and preserved or regenerable; a preserved path also needs its artifact to hold every
file now under it at its size, and the artifact PRESERVED on the remote, checked just before the deletion (`store_check`); and no
file in the job-script directories may name it (`JobScripts`). An implicit record also needs its marker to still vouch for the run
dir (`custody.marker_problem`). A path holding a nested record is left whole. A dry run (the default) reads no remote.

`rm` deletes the paths it is given, with their owner's yes, for a cleanup the policy doesn't reach. A path qualifies only when a
record covers it (its own or an ancestor's), that record and every record beneath it (sidecars and implicit ones) are valid, name
the --approved-by handle as owner and are not `keep`, no job script names it, and nothing in it is read-only or a mount. A sealed
run dir (store.sealed) is the one read-only path either command deletes: whole, by `sudo -n vy-seal-runs delete` once every check
here passed, which redoes custody's as root; a path inside one is refused. A path no
record covers has nobody to say yes, so it is never deleted. A path that is, holds or lies under a `passes` directory (PASSES; not a Python package of that name, nor anything inside a
venv, `_holds_passes`) also
needs its covering record's `preserved` artifact to hold that record's whole path, PRESERVED on the remote (`store_check`, just
before --apply deletes): a yes alone never deletes retained passes. Each record is read again just before --apply deletes, and every
decision of an --apply goes to the node's deletion log, `<root>/LOG`, with who deleted, the approver and --ref. A dry run (the
default) deletes and logs nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import stat
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .disk import HOLD_PCT, RELEASE_PCT
from .store import custody as CU
from .store import ids
from .store import sealed as SL
from .telemetry.job import write_json_atomic

SCHEMA = "research/retention/v0"
SIDECAR = ".retention.json"
FIELDS = ("schema", "path", "node", "owner", "by", "created", "expires", "priority", "preserved", "regenerate", "why")
PRIORITIES = ("low", "normal", "high", "keep")
COLLECTABLE = ("low", "normal")
RUN_EXPIRY = timedelta(hours=48)
DEFAULT_ROOT = Path("/workspace")
DEFAULT_JOBS_DIRS = (Path("/workspace/pouw/fill/queue"), Path("/workspace/pouw/fill/running"))
POLICY = "default/v0"
POLICIES = ("default",)
KIND = "node-data/v1"
LOG = Path("research/retention/deletions.jsonl")
PASSES = "passes"
LOG_SCHEMA = "research/retention-log/v1"
RM_SCHEMA = "research/retention-rm/v0"
OWNER_RE = re.compile(r"^@[A-Za-z0-9][A-Za-z0-9._-]*$")
TS = "%Y-%m-%dT%H:%M:%SZ"
NAME_CHAR = r"[\w.-]"

disk_usage = shutil.disk_usage


def inode_usage(path: Path) -> tuple[int, int]:
    """(total, used) inodes of the filesystem holding `path`; (0, 0) where it doesn't count them."""
    st = os.statvfs(path)
    return st.f_files, st.f_files - st.f_ffree


class RecordError(ValueError):
    """A record that is not `research/retention/v0`, or a path that cannot have one; the message says how to fix it."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(s: Any) -> datetime:
    """An ISO-8601 UTC time (`2026-10-08T00:00:00Z`; fractional seconds and `+00:00` accepted); ValueError otherwise."""
    if not isinstance(s, str):
        raise ValueError(f"not a time: {s!r}")
    t = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    if t.tzinfo is None or t.utcoffset() != timedelta(0):
        raise ValueError(f"{s!r} is not in UTC (write it as YYYY-MM-DDTHH:MM:SSZ)")
    return t.astimezone(timezone.utc)


def fmt_utc(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime(TS)


def parse_expires(s: str, now: datetime) -> str | None:
    """`--expires`: `never` (None), `now`, a date (its midnight UTC), a UTC time, or an offset from now (`7d`, `36h`)."""
    if s == "never":
        return None
    if s == "now":
        return fmt_utc(now)
    if m := re.fullmatch(r"(\d+)([dh])", s):
        return fmt_utc(now + timedelta(**{"days" if m[2] == "d" else "hours": int(m[1])}))
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return fmt_utc(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc))
        return fmt_utc(parse_utc(s))
    except ValueError:
        raise RecordError(f"--expires {s!r}: give never, now, a date (2026-10-08), a UTC time (2026-10-08T12:00:00Z) or an offset (7d, 36h)") from None


def _utc_or_none(s: Any) -> datetime | None:
    try:
        return parse_utc(s)
    except ValueError:
        return None


# ---- the record -------------------------------------------------------------------------------------------------------------------------

def problems(doc: Any) -> list[str]:
    """What makes `doc` not a valid explicit record, each with how to fix it; [] when it is one."""
    if not isinstance(doc, dict):
        return ["a record is a JSON object: rewrite it with `research keep PATH ...`"]
    out = []
    if doc.get("schema") != SCHEMA:
        out.append(f"schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    if missing := [k for k in FIELDS if k not in doc]:
        out.append(f"missing {', '.join(missing)} (every field is present, null where it has no value)")
    if extra := sorted(set(doc) - set(FIELDS)):
        out.append(f"unknown field(s) {', '.join(extra)} (the fields are {', '.join(FIELDS)})")
    p = doc.get("path")
    if not isinstance(p, str) or not p or p.startswith("/") or any(c in ("", ".", "..") for c in p.split("/")):
        out.append(f"path must be relative to the root, with no empty, '.' or '..' component (pouw/fill-out/job-7), got {p!r}")
    if "node" in doc and (not isinstance(doc["node"], str) or not doc["node"]):
        out.append(f"node must be the host the data is on (vy-nebius-2), got {doc['node']!r}")
    if "owner" in doc and not (isinstance(doc["owner"], str) and OWNER_RE.match(doc["owner"])):
        out.append(f"owner must be a lane handle such as @compute-accounting, got {doc['owner']!r}")
    for k in ("by", "why"):
        if k in doc and doc[k] is not None and not isinstance(doc[k], str):
            out.append(f"{k} must be a string or null, got {doc[k]!r}")
    if "created" in doc and _utc_or_none(doc["created"]) is None:
        out.append(f"created must be a UTC time (2026-10-01T18:00:00Z), got {doc['created']!r}")
    if doc.get("expires") is not None and _utc_or_none(doc["expires"]) is None:
        out.append(f"expires must be a UTC time (2026-10-08T00:00:00Z) or null for never, got {doc['expires']!r}")
    if "priority" in doc and doc["priority"] not in PRIORITIES:
        out.append(f"priority must be one of {', '.join(PRIORITIES)}, got {doc['priority']!r}")
    if doc.get("preserved") is not None and not ids.is_art(doc["preserved"]):
        out.append(f"preserved must be a full art: id (art:<64 hex>) or null, got {doc['preserved']!r}")
    rg = doc.get("regenerate")
    if rg is not None and not (isinstance(rg, str) and rg.strip()):
        out.append(f"regenerate must be the command that rebuilds the data, or null, got {rg!r}")
    return out


def check(doc: Any) -> dict:
    if errs := problems(doc):
        raise RecordError("; ".join(errs))
    return doc


def beside(path: Path) -> Path:
    p = Path(path)
    return p.with_name(p.name + SIDECAR)


def _places(path: Path) -> list[Path]:
    p = Path(path)
    return [p / SIDECAR, beside(p)] if p.is_dir() and not p.is_symlink() else [beside(p)]


def sidecar(path: Path) -> Path:
    """Where `path`'s record lives or would be written: an existing sidecar, else `<dir>/.retention.json` for a directory it can
    write into, else `<path>.retention.json` beside it."""
    places = _places(path)
    for sc in places:
        if os.path.lexists(sc):
            return sc
    return places[0] if len(places) == 1 or os.access(path, os.W_OK | os.X_OK) else places[1]


def load(path: Path) -> dict | None:
    """`path`'s explicit record, or None when it has no sidecar; RecordError when the sidecar is unreadable or invalid, or when a
    directory has one inside and one beside it."""
    docs = []
    for sc in _places(path):
        try:
            docs.append((sc, json.loads(sc.read_text())))
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as e:
            if isinstance(e, PermissionError) and not os.access(sc.parent, os.R_OK | os.X_OK):
                raise RecordError(f"{sc.parent} can't be read, so whether it has a record is unknown") from None
            raise RecordError(f"{sc}: unreadable ({e}): rewrite it with `research keep`") from None
    if not docs:
        return None
    if len(docs) > 1:
        raise RecordError(f"{docs[0][0]} and {docs[1][0]} both record {path}: remove one")
    sc, doc = docs[0]
    try:
        return check(doc)
    except RecordError as e:
        raise RecordError(f"{sc}: {e}") from None


def write(path: Path, record: dict) -> Path:
    """Write `path`'s sidecar atomically (a temp file, then a rename) after validating it."""
    sc = sidecar(path)
    check(record)
    write_json_atomic(sc, {k: record[k] for k in FIELDS})
    return sc


def _read_json(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def implicit(run_dir: Path, rel: str) -> dict | None:
    """The record a directory with a preserved `.custody` marker has without a sidecar (module docstring); None for any other."""
    m = _read_json(Path(run_dir) / CU.MARKER)
    if not m or m.get("schema") != CU.SCHEMA or m.get("preserved") is not True:
        return None
    job = _read_json(Path(run_dir) / "job.json") or {}
    store = job.get("store") or {}
    owner = ((job.get("ownership") or {}).get("lane") or store.get("campaign") or (store.get("conditions") or {}).get("exec.campaign")
             or "runner")
    verified = _utc_or_none(m.get("verified_utc"))
    created = _utc_or_none((job.get("run") or {}).get("created_utc")) or verified
    rr = m.get("run_record")
    return {"schema": SCHEMA, "path": rel, "node": m.get("host"), "owner": owner, "by": None,
            "created": fmt_utc(created) if created else None, "expires": fmt_utc(verified + RUN_EXPIRY) if verified else None,
            "priority": "normal", "preserved": rr if ids.is_art(rr) else None, "regenerate": None,
            "why": f"run custody: the run record is on the remote, and the local copy expires {RUN_EXPIRY // timedelta(hours=1)} h after it"}


@dataclass
class Found:
    path: Path
    record: dict | None
    source: str                  # "sidecar" | "custody"
    error: str | None = None     # an invalid sidecar's problems


def record_at(path: Path, rel: str) -> Found | None:
    """The record of exactly `path` (its sidecar, else a directory's implicit one), or None."""
    p = Path(path)
    try:
        doc = load(p)
    except RecordError as e:
        return Found(p, None, "sidecar", str(e))
    if doc is not None:
        return Found(p, doc, "sidecar")
    if p.is_dir() and not p.is_symlink() and (doc := implicit(p, rel)) is not None:
        return Found(p, doc, "custody")
    return None


def relative(path: Path | str, root: Path | str) -> str:
    """`path` relative to `root` (posix); RecordError when it is not strictly under it."""
    p, r = Path(os.path.abspath(path)), Path(os.path.abspath(root))
    try:
        rel = p.relative_to(r)
    except ValueError:
        try:
            rel = p.resolve().relative_to(r.resolve())
        except ValueError:
            raise RecordError(f"{path} is not under the root {root}: pass --root") from None
    if rel == Path("."):
        raise RecordError(f"{path} is the root: a record covers a path under the root, never the root itself")
    return rel.as_posix()


def find(path: Path | str, root: Path | str = DEFAULT_ROOT) -> Found | None:
    """The record covering `path`: its own, else the nearest ancestor's below the root."""
    rel = relative(path, root)
    parts = rel.split("/")
    for n in range(len(parts), 0, -1):
        sub = "/".join(parts[:n])
        if found := record_at(Path(root) / sub, sub):
            return found
    return None


def target(path: Path | str, root: Path | str) -> str:
    """The relative path of data `research keep` may record; RecordError when there is none to record."""
    p = Path(path)
    if not os.path.lexists(p):
        raise RecordError(f"{p} does not exist")
    if p.name.endswith(SIDECAR):
        raise RecordError(f"{p} is a record, not data: give the path it records")
    return relative(p, root)


def keep(path: Path | str, *, owner: str, expires: str | None, priority: str, root: Path | str = DEFAULT_ROOT,
         regenerate: str | None = None, preserved: str | None = None, why: str | None = None, by: str | None = None,
         now: Callable[[], datetime] = utc_now, node: str | None = None) -> dict:
    """Write or update `path`'s sidecar and return the record. An update keeps `created`, and an optional field not given (None)
    keeps its value; an empty string clears it."""
    p = Path(path)
    rel = target(p, root)
    try:
        old = load(p) or {}
    except RecordError:
        old = {}
    given = {"regenerate": regenerate, "preserved": preserved, "why": why, "by": by}
    rec = {"schema": SCHEMA, "path": rel, "node": node or socket.gethostname(), "owner": owner,
           "created": old.get("created") or fmt_utc(now()), "expires": expires, "priority": priority}
    for k, v in given.items():
        rec[k] = (v or None) if v is not None else old.get(k)
    write(p, rec)
    return {k: rec[k] for k in FIELDS}


# ---- the walk -----------------------------------------------------------------------------------------------------------------------------

@dataclass
class Row:
    path: str
    bytes: int = 0
    files: int = 0
    record: dict | None = None
    source: str | None = None            # "sidecar" | "custody"; None: no record
    error: str | None = None
    nested: list[str] = field(default_factory=list)
    rest: bool = False                   # an unowned row for the unrecorded rest of a partly recorded area

    def state(self, now: datetime) -> str:
        if self.error:
            return "invalid"
        if self.record is None:
            return "unowned"
        return "expired" if expired(self.record, now) else "recorded"

    def doc(self, now: datetime) -> dict[str, Any]:
        r = self.record or {}
        return {"path": self.path, "state": self.state(now), "bytes": self.bytes, "files": self.files, "source": self.source,
                "owner": r.get("owner"), "expires": r.get("expires"), "priority": r.get("priority"), "preserved": r.get("preserved"),
                "regenerable": bool((r.get("regenerate") or "").strip()), "expired": bool(self.record) and expired(self.record, now),
                "rest": self.rest, "nested": self.nested, "error": self.error, "record": self.record}


def expired(record: dict, now: datetime) -> bool:
    e = _utc_or_none(record.get("expires"))
    return e is not None and e <= now


def measure(path: Path, errors: list[str]) -> tuple[int, int, list[Path]]:
    """(apparent bytes and count of every non-directory entry under `path`, the sidecars beneath its own), by os.scandir and
    lstat; symlinks are never followed."""
    try:
        st = os.lstat(path)
    except OSError as e:
        errors.append(f"{path}: {e.strerror or e}")
        return 0, 0, []
    if not stat.S_ISDIR(st.st_mode):
        return st.st_size, 1, []
    total, n, nested, stack = 0, 0, [], [str(path)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                        continue
                    try:
                        total += e.stat(follow_symlinks=False).st_size
                        n += 1
                    except OSError as x:
                        errors.append(f"{e.path}: {x.strerror or x}")
                    if e.name.endswith(SIDECAR) and not (e.name == SIDECAR and d == str(path)):
                        nested.append(Path(e.path))
        except OSError as x:
            errors.append(f"{d}: {x.strerror or x}")
    return total, n, sorted(nested)


@dataclass
class Scan:
    root: Path
    rows: list[Row]
    errors: list[str]


def scan(root: Path | str = DEFAULT_ROOT, depth: int = 2) -> Scan:
    """Every recorded path under `root` (never descending into one), and the areas in its top `depth` levels with no record
    beneath them (`unowned`). The unrecorded bytes of a partly recorded area are one `rest` row: the loose files of one in the
    top levels, everything unrecorded beneath one at `depth`."""
    root = Path(root)
    rows: list[Row] = []
    errors: list[str] = []

    def walk(d: Path, rel: str, level: int) -> tuple[int, int, bool]:
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as e:
            errors.append(f"{d}: {e.strerror or e}")
            return 0, 0, False
        names = {e.name for e in entries}
        unrec = unrec_n = loose = loose_n = 0
        has = False
        unowned: list[Row] = []
        for e in entries:
            r = f"{rel}/{e.name}" if rel else e.name
            is_dir = e.is_dir(follow_symlinks=False)
            if not is_dir and e.name.endswith(SIDECAR):
                if e.name == SIDECAR:
                    errors.append(f"{e.path} is ignored: a record covers a path under the root, never the root itself")
                elif e.name[: -len(SIDECAR)] in names:
                    continue
                else:
                    errors.append(f"{e.path} records {e.name[: -len(SIDECAR)]}, which is gone")
            found = record_at(Path(e.path), r) if is_dir or e.name + SIDECAR in names else None
            if found:
                b, n, nested = measure(Path(e.path), errors)
                rows.append(Row(r, b, n, found.record, found.source, found.error, [relative(x, root) for x in nested]))
                has = True
            elif is_dir:
                b, n, sub = walk(Path(e.path), r, level + 1)
                unrec, unrec_n, has = unrec + b, unrec_n + n, has or sub
                if not sub:
                    unowned.append(Row(r, b, n))
            else:
                try:
                    size = e.stat(follow_symlinks=False).st_size
                except OSError as x:
                    errors.append(f"{e.path}: {x.strerror or x}")
                    size = 0
                unrec, unrec_n, loose, loose_n = unrec + size, unrec_n + 1, loose + size, loose_n + 1
        if level < depth and (level == 0 or has):
            rows.extend(unowned)
            if loose_n:
                rows.append(Row(rel or ".", loose, loose_n, rest=True))
        elif level == depth and has and unrec_n:
            rows.append(Row(rel or ".", unrec, unrec_n, rest=True))
        return unrec, unrec_n, has

    walk(root, "", 0)
    return Scan(root, rows, errors)


ORDER = {"invalid": 0, "unowned": 0, "expired": 1, "recorded": 2}


def report(root: Path | str = DEFAULT_ROOT, depth: int = 2, *, now: Callable[[], datetime] = utc_now,
           usage: Callable[[Path], Any] = shutil.disk_usage, node: str | None = None) -> dict[str, Any]:
    """`research retention report --json`: unowned (and invalid) areas first, then expired paths, then the rest, each by size."""
    t = now()
    sc = scan(root, depth)
    docs = sorted((r.doc(t) for r in sc.rows), key=lambda d: (ORDER[d["state"]], -d["bytes"], d["path"]))
    totals: dict[str, dict[str, int]] = {s: {"paths": 0, "bytes": 0} for s in ("unowned", "invalid", "expired", "recorded", "collectable")}
    for d in docs:
        totals[d["state"]]["paths"] += 1
        totals[d["state"]]["bytes"] += d["bytes"]
        if d["state"] == "expired" and d["priority"] in COLLECTABLE and (d["preserved"] or d["regenerable"]):
            totals["collectable"]["paths"] += 1
            totals["collectable"]["bytes"] += d["bytes"]
    total, used, _ = usage(Path(root))
    return {"schema": "research/retention-report/v0", "root": str(root), "node": node or socket.gethostname(), "now": fmt_utc(t),
            "depth": depth, "filesystem": {"total": total, "used": used, "pct": round(100 * used / total, 2) if total else None},
            "rows": docs, "totals": totals, "errors": sc.errors}


# ---- gc -----------------------------------------------------------------------------------------------------------------------------------

class JobScripts:
    """The v0 in-use check: a path is in use while some file in the job-script directories names it (its path under the root,
    alone or as the tail of an absolute path, whole or as the prefix of a longer one). The directories are listed again on every
    call, so a job queued meanwhile counts; unchanged files are read once. Any read error makes every path in use."""

    def __init__(self, dirs: Sequence[Path | str], root: Path | str = DEFAULT_ROOT):
        self.dirs, self.root = [Path(d) for d in dirs], Path(root)
        self._cache: dict[str, tuple[tuple[int, int], str]] = {}

    def _texts(self) -> list[tuple[str, str]]:
        """(path, text) of every file, in the directories' order: a job that moves from queue/ to running/ meanwhile is read in one
        of them, so a file gone before it is read is skipped."""
        out = []
        for d in self.dirs:
            for base, dirnames, files in os.walk(d, onerror=_raise):
                dirnames.sort()
                for f in sorted(files):
                    p = os.path.join(base, f)
                    try:
                        st = os.stat(p)
                    except FileNotFoundError:
                        continue
                    key = (st.st_mtime_ns, st.st_size)
                    hit = self._cache.get(p)
                    if hit is None or hit[0] != key:
                        try:
                            with open(p, "rb") as fh:
                                hit = self._cache[p] = (key, fh.read().decode(errors="replace"))
                        except FileNotFoundError:
                            continue
                    out.append((p, hit[1]))
        return out

    def __call__(self, path: Path) -> str | None:
        rel = relative(path, self.root)
        pat = re.compile(rf"(?<!{NAME_CHAR}){re.escape(rel)}(?!{NAME_CHAR})")
        try:
            texts = self._texts()
        except OSError as e:
            return f"cannot read the job scripts ({e}), so any path may be in use"
        return next((f"{p} names it" for p, text in texts if pat.search(text)), None)


def _raise(e: OSError) -> None:
    raise e


def covers(man: Any, path: Path) -> str | None:
    """None when the artifact's payload holds every file now under `path` at its size (a directory's own sidecar and custody's
    markers aside); else what it lacks. A symlink is never held, and an unreadable directory raises."""
    pl = man.payload
    p = Path(path)
    if pl is None:
        return "the artifact has no payload"
    st = os.lstat(p)
    if pl.type == "file":
        if not stat.S_ISREG(st.st_mode):
            return f"{p} is not a regular file, and the artifact holds one file"
        return None if st.st_size == pl.bytes else f"{p} is {st.st_size} bytes, the artifact's file {pl.bytes}"
    if not stat.S_ISDIR(st.st_mode):
        return f"{p} is not a directory, and the artifact holds a tree"
    have = {f.path: f.bytes for f in pl.files or []}
    skip = CU.EXCLUDED | {SIDECAR}
    lacking = []
    for base, dirs, files in os.walk(p, onerror=_raise):
        dirs.sort()
        for name in sorted(files + [d for d in dirs if os.path.islink(os.path.join(base, d))]):
            q = os.path.join(base, name)
            rel = os.path.relpath(q, p).replace(os.sep, "/")
            if rel in skip:
                continue
            s = os.lstat(q)
            if not stat.S_ISREG(s.st_mode) or have.get(rel) != s.st_size:
                lacking.append(rel)
    if lacking:
        return f"{len(lacking)} file(s) under it are not in the artifact at their size, e.g. {lacking[0]}"
    return None


def store_check(store_root: str | None = None) -> Callable[[str, Path], str | None]:
    """The default remote check, run just before a deletion: the artifact holds the path's files (`covers`), and
    `preserved.check` in head mode finds it PRESERVED (remote sizes, and the single-part ETag against the local blob's MD5 where
    the blob is here, else a read-back; `LocalStore.verify` would read back every blob). None when both hold, else why not."""
    st = None

    def chk(art: str, path: Path) -> str | None:
        nonlocal st
        from .store import preserved as P
        from .store.local import LocalStore

        try:
            st = st or LocalStore(store_root)
            if why := covers(st.get_manifest(art), path):
                return why
            rep = P.check(st, [art], mode="head")
        except Exception as e:  # noqa: BLE001  any failure is a failed check: the path stays
            return f"{type(e).__name__}: {e}"
        if rep["preserved"]:
            return None
        errs = [e for x in rep["not_preserved"] for e in x.get("errors") or []]
        return (errs or [rep.get("reason") or "not preserved"])[0]

    return chk


@dataclass
class Action:
    row: Row
    action: str                  # "delete" | "skip"
    reason: str
    pct_after: float | None = None
    error: str | None = None

    def doc(self, now: datetime) -> dict[str, Any]:
        return {**self.row.doc(now), "action": self.action, "reason": self.reason, "pct_after": self.pct_after, "error": self.error}


def ineligible(row: Row, now: datetime, node: str) -> str | None:
    """Why the policy never deletes `row`'s path, from its record alone; None when it may."""
    rec = row.record
    if row.error:
        return f"invalid record: {row.error}"
    if rec is None:
        return "no record: a path without one is never deleted (`research keep PATH ...` gives it one)"
    if rec["priority"] not in COLLECTABLE:
        return f"priority {rec['priority']}: never collected"
    if rec.get("expires") is None:
        return "never expires"
    if not expired(rec, now):
        return f"not expired: expires {rec['expires']}"
    if not rec.get("preserved") and not (rec.get("regenerate") or "").strip():
        return "neither preserved nor regenerable: nothing would bring it back"
    if rec.get("node") and rec["node"] != node:
        return f"recorded on {rec['node']}, and this is {node}: `research keep` it here"
    if row.nested:
        return f"holds {len(row.nested)} nested record(s), e.g. {row.nested[0]}: a record covers everything beneath it, so remove those first"
    return None


def _why_deletable(rec: dict, verified: bool) -> str:
    how = f"preserved {rec['preserved']}" + (" (verified on the remote)" if verified else ", re-verified on the remote just before an --apply deletes it") \
        if rec.get("preserved") else "regenerable"
    return f"{rec['priority']}, expired {rec['expires']}, {how}"


def _read_only_dir(path: Path) -> str | None:
    for base, _dirs, _files in os.walk(path, onerror=_raise):
        if not os.access(base, os.W_OK | os.X_OK):
            return base
    return None


def _in_sealed(path: Path) -> str | None:
    if SL.sealed(Path(path).parent):
        return f"{path} is inside a sealed run dir: only the whole run dir is deleted"
    return None


def _delete(root: Path, path: Path) -> str | None:
    """Remove a recorded path with its sidecar; the error, or None. A tree holding a directory this process can't write into is
    left whole, since rmtree would stop partway; a sealed run dir (store.sealed) is deleted whole by vy-seal-runs, as root."""
    try:
        if root not in Path(os.path.abspath(path)).parents:
            return f"{path} is not under {root}"
        if os.path.ismount(path):
            return f"{path} is a mount point"
        if why := _in_sealed(path):
            return why
        if SL.sealed(path):
            if err := SL.delete(path):
                return f"sealed, and vy-seal-runs refused to delete it: {err}"
        elif os.path.isdir(path) and not os.path.islink(path):
            if ro := _read_only_dir(path):
                return f"{ro} is read-only: its owner makes it writable, or keeps it"
            shutil.rmtree(path)
        else:
            os.unlink(path)
        beside(path).unlink(missing_ok=True)
    except OSError as e:
        return f"{type(e).__name__}: {e}"
    return None


def gc(root: Path | str = DEFAULT_ROOT, *, policy: str = "default", high_pct: float = HOLD_PCT, low_pct: float = RELEASE_PCT,
       force: bool = False, apply: bool = False, jobs_dirs: Sequence[Path | str] | None = None, depth: int = 2,
       now: Callable[[], datetime] = utc_now, usage: Callable[[Path], Any] = shutil.disk_usage,
       remote_check: Callable[[str, Path], str | None] | None = None, in_use: Callable[[Path], str | None] | None = None,
       node: str | None = None, store_root: str | None = None,
       inodes: Callable[[Path], tuple[int, int]] | None = None) -> dict[str, Any]:
    """The policy (module docstring) over `root`: the plan, and with `apply` its deletions. `jobs_dirs` None means the
    DEFAULT_JOBS_DIRS that exist; `remote_check` None is `store_check(store_root)`; `in_use` None is `JobScripts(jobs_dirs)`;
    `inodes` (`inode_usage`) makes the filesystem's use the larger of its blocks' and its inodes', None its blocks' alone."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r} (have: {', '.join(POLICIES)})")
    if low_pct > high_pct:
        raise ValueError(f"--low-pct {low_pct:g} is over --high-pct {high_pct:g}")
    root = Path(os.path.abspath(root))
    t, node = now(), node or socket.gethostname()
    total, used, _ = usage(root)
    itotal, iused = inodes(root) if inodes else (0, 0)

    def fullness() -> float:
        return max(100 * used / total if total else 100.0, 100 * iused / itotal if itotal else 0.0)

    pct = fullness()
    plan: dict[str, Any] = {"schema": "research/retention-gc/v0", "policy": POLICY, "root": str(root), "node": node, "now": fmt_utc(t),
                            "apply": apply, "force": force, "high_pct": high_pct, "low_pct": low_pct, "pct_before": round(pct, 2),
                            "inodes_pct_before": round(100 * iused / itotal, 2) if itotal else None,
                            "acting": force or pct > high_pct, "actions": [], "freed_bytes": 0, "errors": []}
    if not plan["acting"]:
        plan["pct_after"] = plan["pct_before"]
        return plan
    if jobs_dirs is None:
        jobs_dirs = [d for d in DEFAULT_JOBS_DIRS if d.is_dir()]
    in_use = in_use or JobScripts(jobs_dirs, root)
    remote_check = remote_check or store_check(store_root)
    sc = scan(root, depth)
    plan["errors"] = sc.errors
    candidates, skips = [], []
    for row in sorted(sc.rows, key=lambda r: (-r.bytes, r.path)):
        why = ineligible(row, t, node)
        if why is None and row.source == "custody" and (problem := CU.marker_problem(root / row.path)):
            why = f"its custody marker no longer vouches for it: {problem}"
        if why is None and (u := in_use(root / row.path)):
            why = f"in use: {u}"
        if why:
            skips.append(Action(row, "skip", why))
        else:
            candidates.append(row)
    candidates.sort(key=lambda r: (PRIORITIES.index(r.record["priority"]), parse_utc(r.record["expires"]), r.path))
    acts = []
    for row in candidates:
        p = root / row.path
        if pct < low_pct:
            acts.append(Action(row, "skip", f"not needed: the filesystem is at {pct:.1f}%, under {low_pct:g}%"))
            continue
        fresh = record_at(p, row.path)
        why = None
        if fresh is None or fresh.record != row.record:
            why = "its record changed during gc: run gc again"
        elif u := in_use(p):
            why = f"in use: {u}"
        elif apply and row.record.get("preserved") and (r := remote_check(row.record["preserved"], p)):
            why = f"preserved {row.record['preserved']} is not verified on the remote: {r}"
        if why:
            acts.append(Action(row, "skip", why))
            continue
        if not apply:
            used -= row.bytes
            iused -= row.files
            pct = fullness()
            acts.append(Action(row, "delete", "would delete: " + _why_deletable(row.record, False), round(pct, 2)))
            plan["freed_bytes"] += row.bytes
            continue
        err = _delete(root, p)
        total, used, _ = usage(root)
        itotal, iused = inodes(root) if inodes else (0, 0)
        pct = fullness()
        acts.append(Action(row, "delete", "deleted: " + _why_deletable(row.record, bool(row.record.get("preserved"))), round(pct, 2), err))
        if not err:
            plan["freed_bytes"] += row.bytes
    plan["actions"] = [a.doc(t) for a in acts + skips]
    plan["pct_after"] = round(pct, 2)
    return plan


# ---- rm -----------------------------------------------------------------------------------------------------------------------------------

def _handle(owner: Any) -> str:
    """An owner as a lane handle: implicit records name the job's lane without the @."""
    return "@" + str(owner or "").lstrip("@")


def beneath(path: Path, root: Path) -> list[Found]:
    """Every record strictly beneath `path`: each sidecar's, and the implicit record of each preserved `.custody` marker without one.
    Symlinks are never followed; OSError when a directory can't be read."""
    p = Path(path)
    if not p.is_dir() or p.is_symlink():
        return []
    out: dict[Path, Found] = {}
    for base, dirs, files in os.walk(p, onerror=_raise):
        dirs.sort()
        here = Path(base)
        for name in sorted(files):
            if name == SIDECAR and here != p:
                data = here
            elif name.endswith(SIDECAR) and name != SIDECAR:
                data = here / name[: -len(SIDECAR)]
            elif name == CU.MARKER and here != p and not any(os.path.lexists(sc) for sc in _places(here)):
                data = here
            else:
                continue
            if data not in out and (found := record_at(data, relative(data, root))):
                out[data] = found
    return [out[k] for k in sorted(out)]


def _covering(path: Path, root: Path) -> tuple[Found | None, list[Found]]:
    """The record covering `path` (its own or the nearest ancestor's), or None, and every record beneath it."""
    return find(path, root), beneath(path, root)


def _docs(top: Found | None, below: list[Found]) -> list[tuple[Path, dict | None, str | None]]:
    return [(f.path, f.record, f.error) for f in ([top] if top else []) + below]


def _kept(f: Found, root: Path) -> str:
    r = f.record or {}
    why = f": {r['why']}" if r.get("why") else ""
    return f"{relative(f.path, root)} is kept by {_handle(r.get('owner'))} until {r.get('expires') or 'never'} (priority {r.get('priority')}{why})"


def _refusal(path: Path, root: Path, approved_by: str, top: Found | None, below: list[Found],
             in_use: Callable[[Path], str | None]) -> str | None:
    """Why `rm` may not delete `path` with `approved_by`'s yes, given the records `_covering` found; None when it may."""
    if top is None:
        return (f"no record covers it, so nobody can say yes: its owner records it first (`research keep {path} --owner @LANE "
                f"--expires now --priority low`), or deletes it")
    for f in (top, *below):
        if f.error:
            return f"invalid record at {f.path}: {f.error}"
        if _handle(f.record["owner"]) != approved_by:
            return f"{_kept(f, root)}, and --approved-by {approved_by} is not its owner: only {_handle(f.record['owner'])} can say yes"
        if f.record["priority"] == "keep":
            return f"{_kept(f, root)}: priority keep is never deleted, until its owner lowers it with `research keep`"
    if u := in_use(path):
        return f"in use: {u}"
    if why := _in_sealed(path):
        return why
    try:
        ro = _read_only_dir(path) if os.path.isdir(path) and not os.path.islink(path) and not SL.sealed(path) else None
        passes = _holds_passes(path, root)
    except OSError as e:
        return f"can't read all of it ({e}), so what it holds is unknown"
    if ro:
        return f"{ro} is read-only: its owner makes it writable first"
    if passes and not top.record.get("preserved"):
        return (f"it holds retained passes and {_kept(top, root)} names no preserved artifact: passes are deleted only once "
                f"preserved (`research keep {top.path} ... --preserve`), never on a yes alone")
    return None


CODE_DIRS = frozenset({".venv", "venv", "site-packages", "dist-packages", "node_modules", ".lake", "__pycache__"})
"""Directories that hold code, never a retained pass: not walked for PASSES (INF-264: torch's and vLLM's `passes` packages in every
venv refused proofs' twelve approved trees on cpu-1, 4.47M inodes)."""


def _pass_dir(d: Path) -> bool:
    """A PASSES directory that holds passes: not a Python package of that name (`torch/fx/passes`, `vllm/compilation/passes`)."""
    return d.name == PASSES and not (d / "__init__.py").exists()


def _holds_passes(path: Path, root: Path) -> bool:
    """Whether `path` is, holds or lies under a PASSES directory that holds passes (`_pass_dir`), outside CODE_DIRS."""
    parts = relative(path, root).split("/")
    if any(x == PASSES and not set(parts[:i]) & CODE_DIRS and _pass_dir(root.joinpath(*parts[:i + 1]))
           for i, x in enumerate(parts)):
        return True
    if not os.path.isdir(path) or os.path.islink(path):
        return False
    for base, dirs, _files in os.walk(path, onerror=_raise):
        dirs[:] = [d for d in dirs if d not in CODE_DIRS]
        if any(_pass_dir(Path(base) / d) for d in dirs if d == PASSES):
            return True
    return False


def rm(paths: Sequence[Path | str], *, approved_by: str, ref: str, root: Path | str = DEFAULT_ROOT, apply: bool = False,
       by: str | None = None, in_use: Callable[[Path], str | None] | None = None, now: Callable[[], datetime] = utc_now,
       node: str | None = None, log: Path | str | None = None, remote_check: Callable[[str, Path], str | None] | None = None,
       store_root: str | None = None) -> dict[str, Any]:
    """`research retention rm` (module docstring): the plan for `paths`, and with `apply` their deletions, every decision a line in the
    deletion log (`log`, default `root/LOG`). `in_use` None is `JobScripts` over the DEFAULT_JOBS_DIRS that exist; `remote_check` None
    is `store_check(store_root)`, read only by an --apply that deletes passes. RecordError when --approved-by is not a handle or --ref
    is empty, and with `apply` when `by` is missing or the log can't be written."""
    root = Path(os.path.abspath(root))
    if not OWNER_RE.match(approved_by or ""):
        raise RecordError(f"--approved-by {approved_by!r}: give the owner's lane handle, such as @compute-accounting")
    if not (ref or "").strip():
        raise RecordError("--ref: give where the owner said yes (a Slack permalink or thread ts)")
    t, node = now(), node or socket.gethostname()
    log_path = Path(log) if log else root / LOG
    if apply:
        if not by:
            raise RecordError("--by: say who deletes (your agent's URL; the default is $RESEARCH_AGENT_URL)")
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            open(log_path, "a").close()
        except OSError as e:
            raise RecordError(f"can't write the deletion log {log_path} ({e.strerror or e}), so nothing would record a deletion") from None
    in_use = in_use or JobScripts([d for d in DEFAULT_JOBS_DIRS if d.is_dir()], root)
    plan: dict[str, Any] = {"schema": RM_SCHEMA, "root": str(root), "node": node, "now": fmt_utc(t), "apply": apply,
                            "approved_by": approved_by, "ref": ref, "by": by, "actions": [], "freed_bytes": 0}

    def record(act: dict[str, Any]) -> None:
        plan["actions"].append(act)
        if apply:
            line = {"schema": LOG_SCHEMA, "t": fmt_utc(now()), "node": node, "by": by, "action": act["action"], "path": act["path"],
                    "rule": None, "owner": ",".join(act["owners"]) or None, "files": act["files"], "bytes": act["bytes"], "age_h": None,
                    "reason": f"research retention rm, approved by {approved_by} ({ref}): {act['reason']}"
                              + (f"; {act['error']}" if act["error"] else "")}
            with open(log_path, "a") as fh:
                fh.write(json.dumps(line) + "\n")

    for raw in paths:
        p = Path(os.path.abspath(raw))
        act: dict[str, Any] = {"path": str(p), "action": "refuse", "reason": "", "owners": [], "bytes": 0, "files": 0, "error": None}
        try:
            relative(p, root)
            if not os.path.lexists(p):
                raise RecordError(f"{p} does not exist")
            if p.name.endswith(SIDECAR):
                raise RecordError(f"{p} is a record, not data: give the path it records")
            if os.path.ismount(p):
                raise RecordError(f"{p} is a mount point")
            errors: list[str] = []
            act["bytes"], act["files"], _ = measure(p, errors)
            if errors:
                raise RecordError(f"can't read all of it ({errors[0]}), so what it holds is unknown")
            top, below = _covering(p, root)
        except (RecordError, OSError) as e:
            act["reason"] = str(e)
            record(act)
            continue
        act["owners"] = sorted({_handle(f.record["owner"]) for f in (top, *below) if f and f.record})
        if why := _refusal(p, root, approved_by, top, below, in_use):
            act["reason"] = why
            record(act)
            continue
        act["action"] = "delete"
        act["reason"] = f"{_kept(top, root)}, and its owner said yes" + (f" ({len(below)} record(s) beneath it, the same owner's)"
                                                                        if below else "")
        if apply:
            try:
                again = _covering(p, root)
            except (RecordError, OSError) as e:
                again, act["error"] = None, f"can't read its records again: {e}"
            if again is not None and _docs(*again) != _docs(top, below):
                act["error"] = "its records changed during rm: run it again"
            if not act["error"] and (u := in_use(p)):
                act["error"] = f"in use: {u}"
            if not act["error"] and _holds_passes(p, root):
                art = top.record["preserved"]
                remote_check = remote_check or store_check(store_root)
                if r := remote_check(art, top.path):
                    act["error"] = f"it holds retained passes, and preserved {art} is not verified on the remote: {r}"
            act["error"] = act["error"] or _delete(root, p)
            if act["error"]:
                act["action"] = "failed"
            else:
                plan["freed_bytes"] += act["bytes"]
        else:
            plan["freed_bytes"] += act["bytes"]
        record(act)
    return plan


# ---- the command line ---------------------------------------------------------------------------------------------------------------------

def _size(n: int) -> str:
    for unit, k in (("TiB", 40), ("GiB", 30), ("MiB", 20), ("KiB", 10)):
        if n >= 1 << k:
            return f"{n / (1 << k):.1f} {unit}"
    return f"{n} B"


def _short(art: str | None) -> str:
    return art[:16] if art else "-"


def _row_line(d: dict[str, Any]) -> str:
    path = f"{d['path']}/* (unrecorded rest)" if d["rest"] else d["path"]
    if d["state"] in ("unowned", "invalid"):
        tail = f"  {d['error']}" if d["error"] else ""
        return f"{d['state'].upper():<9} {_size(d['bytes']):>10}  {'-':<22} {'-':<20} {'-':<8} {'-':<16} {'-':<5}  {path}{tail}"
    return (f"{'EXPIRED' if d['expired'] else 'recorded':<9} {_size(d['bytes']):>10}  {str(d['owner'])[:22]:<22} "
            f"{str(d['expires'] or 'never'):<20} {d['priority']:<8} {_short(d['preserved']):<16} {'yes' if d['regenerable'] else 'no':<5}  "
            f"{path}" + (" (implicit: .custody)" if d["source"] == "custody" else "") + (f"  [{len(d['nested'])} nested]" if d["nested"] else ""))


def keep_main(argv: list[str]) -> int:
    if argv[:1] == ["show"]:
        return _show(argv[1:])
    ap = argparse.ArgumentParser(prog="research keep", description="write or update PATH's retention record (research/retention/v0); "
                                 "`research keep show PATH` prints the record covering PATH")
    ap.add_argument("path")
    ap.add_argument("--owner", required=True, metavar="@LANE", help="the lane that owns it (agents end, lanes persist)")
    ap.add_argument("--expires", required=True, metavar="DATE|never", help="a date, a UTC time, an offset (7d, 36h), or never")
    ap.add_argument("--priority", required=True, choices=PRIORITIES, help="gc takes low before normal, and never high or keep")
    ap.add_argument("--regenerate", default=None, metavar="CMD", help="how to rebuild it (an empty string clears it)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--art", default=None, metavar="ART", help="the art: id that preserves it (an empty string clears it)")
    g.add_argument("--preserve", action="store_true", help=f"put PATH in the evidence store as {KIND} (a copy into its objects/), push "
                                                          "and verify it, and record its art id")
    ap.add_argument("--why", default=None, metavar="TEXT")
    ap.add_argument("--by", default=None, metavar="URL", help="who writes the record (an agent's URL)")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="the root the record's path is relative to (default %(default)s)")
    ap.add_argument("--store", default=None, metavar="ROOT", help="with --preserve: the evidence store (default $RESEARCH_STORE or ~/.research/store)")
    ns = ap.parse_args(argv)
    try:
        if not OWNER_RE.match(ns.owner):
            raise RecordError(f"--owner {ns.owner!r}: give a lane handle such as @compute-accounting")
        if ns.art and not ids.is_art(ns.art):
            raise RecordError(f"--art {ns.art!r}: give a full art: id (art:<64 hex>)")
        t = utc_now()
        expires = parse_expires(ns.expires, t)
        rel = target(ns.path, ns.root)
        cover = find(Path(ns.path).parent, ns.root) if "/" in rel else None
        art = ns.art
        if ns.preserve and (art := _preserve(Path(ns.path), rel, ns.owner, ns.store)) is None:
            return 1
        rec = keep(ns.path, owner=ns.owner, expires=expires, priority=ns.priority, root=ns.root, regenerate=ns.regenerate,
                   preserved=art, why=ns.why, by=ns.by, now=lambda: t)
    except RecordError as e:
        print(f"research keep: {e}", file=sys.stderr)
        return 2
    if cover:
        print(f"research keep: warning: {cover.path}'s record covers this path, so the walk never sees this one, and gc leaves "
              f"{cover.path} whole while it exists", file=sys.stderr)
    print(f"research keep: {sidecar(Path(ns.path))}: owner {rec['owner']}, expires {rec['expires'] or 'never'}, priority {rec['priority']}"
          + (f", preserved {rec['preserved']}" if rec["preserved"] else "") + (", regenerable" if rec["regenerate"] else ""))
    return 0


def _preserve(path: Path, rel: str, owner: str, store_root: str | None) -> str | None:
    """`keep --preserve`: put the path in the evidence store and push it, verified; its art id, or None after a message."""
    from .store.local import LocalStore, StoreError

    st = LocalStore(store_root)
    try:
        payload = {"tree": path} if path.is_dir() and not path.is_symlink() else {"file": path}
        art = st.put_artifact(KIND, {"path": rel, "node": socket.gethostname(), "owner": owner}, **payload)
        rep = st.push(art, preserve=True)
    except (StoreError, OSError) as e:
        print(f"research keep --preserve: {e} (nothing recorded)", file=sys.stderr)
        return None
    if not rep.get("preserved"):
        print(f"research keep --preserve: {art} NOT preserved: {(rep.get('errors') or ['?'])[0]} (nothing recorded; `research data push "
              f"{art}` retries, then `research keep ... --art {art}`)", file=sys.stderr)
        return None
    return art


def _show(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research keep show", description="print the record covering PATH (its own, else an ancestor's), "
                                 "explicit or implicit, as JSON")
    ap.add_argument("path")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ns = ap.parse_args(argv)
    try:
        found = find(ns.path, ns.root)
    except RecordError as e:
        print(f"research keep show: {e}", file=sys.stderr)
        return 2
    if found is None:
        print(f"research keep show: {ns.path} has no record, so it is unowned (`research keep {ns.path} --owner @LANE --expires DATE "
              f"--priority P` gives it one)", file=sys.stderr)
        return 1
    if found.error:
        print(f"research keep show: {found.error}", file=sys.stderr)
        return 1
    if Path(os.path.abspath(found.path)) != Path(os.path.abspath(ns.path)):
        print(f"research keep show: {ns.path} is covered by the record of {found.path}", file=sys.stderr)
    doc = dict(found.record)
    if found.source == "custody":
        doc["implicit"] = str(found.path / CU.MARKER)
    print(json.dumps(doc, indent=2))
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    sub, rest = argv[0], argv[1:]
    if sub == "report":
        ap = argparse.ArgumentParser(prog="research retention report")
        ap.add_argument("--root", default=str(DEFAULT_ROOT))
        ap.add_argument("--depth", type=int, default=2, metavar="N", help="list unowned areas in the top N levels (default 2)")
        ap.add_argument("--json", action="store_true")
        ns = ap.parse_args(rest)
        if not Path(ns.root).is_dir():
            print(f"research retention report: --root {ns.root} is not a directory", file=sys.stderr)
            return 2
        rep = report(ns.root, ns.depth, now=utc_now, usage=disk_usage)
        if ns.json:
            print(json.dumps(rep, indent=2))
            return 0
        fs = rep["filesystem"]
        print(f"research retention report: {rep['root']} on {rep['node']} at {rep['now']}; filesystem {_size(fs['used'])} of "
              f"{_size(fs['total'])} used ({fs['pct']}%)")
        print(f"{'state':<9} {'size':>10}  {'owner':<22} {'expires':<20} {'priority':<8} {'preserved':<16} {'regen':<5}  path")
        for d in rep["rows"]:
            print(_row_line(d))
        tt = rep["totals"]
        print("totals: " + ", ".join(f"{k} {_size(v['bytes'])} in {v['paths']}" for k, v in tt.items() if k != "collectable" and v["paths"])
              + f"; collectable when the disk is full (expired, low or normal, preserved or regenerable): {_size(tt['collectable']['bytes'])} "
                f"in {tt['collectable']['paths']}")
        for e in rep["errors"]:
            print(f"error: {e}", file=sys.stderr)
        return 0
    if sub == "gc":
        ap = argparse.ArgumentParser(prog="research retention gc", description="policy default: a dry run unless --apply")
        ap.add_argument("--root", default=str(DEFAULT_ROOT))
        ap.add_argument("--policy", default="default", choices=POLICIES)
        ap.add_argument("--high-pct", type=float, default=HOLD_PCT, help="act only over this %% used (default %(default)s)")
        ap.add_argument("--low-pct", type=float, default=RELEASE_PCT, help="delete until under this %% (default %(default)s; 0 takes "
                                                                            "every qualifying path)")
        ap.add_argument("--force", action="store_true", help="act under --high-pct too")
        ap.add_argument("--jobs-dir", action="append", default=[], metavar="D", help="more job-script directories (beside "
                        + " and ".join(map(str, DEFAULT_JOBS_DIRS)) + " when they exist): a path a file there names is in use")
        ap.add_argument("--store", default=None, metavar="ROOT", help="the evidence store whose remote re-verifies preserved paths")
        ap.add_argument("--json", action="store_true")
        ap.add_argument("--apply", action="store_true", help="delete; without it nothing is deleted and no remote is read")
        ns = ap.parse_args(rest)
        if missing := [d for d in [ns.root, *ns.jobs_dir] if not Path(d).is_dir()]:
            print(f"research retention gc: {missing[0]} is not a directory", file=sys.stderr)
            return 2
        jobs = [d for d in DEFAULT_JOBS_DIRS if d.is_dir()] + [Path(d) for d in ns.jobs_dir]
        try:
            plan = gc(ns.root, policy=ns.policy, high_pct=ns.high_pct, low_pct=ns.low_pct, force=ns.force, apply=ns.apply,
                      now=utc_now, usage=disk_usage, remote_check=store_check(ns.store), in_use=JobScripts(jobs, ns.root),
                      inodes=inode_usage)
        except ValueError as e:
            print(f"research retention gc: {e}", file=sys.stderr)
            return 2
        failed = [a for a in plan["actions"] if a["error"]]
        if ns.json:
            print(json.dumps(plan, indent=2))
            return 1 if failed else 0
        mode = "applying" if ns.apply else "dry run (--apply deletes)"
        head = f"research retention gc: policy {plan['policy']}, {mode}: {plan['root']} at {plan['pct_before']}%"
        if not plan["acting"]:
            print(f"{head}, not over {ns.high_pct:g}%: nothing to do (--force acts anyway)")
            return 0
        print(f"{head} (acts over {ns.high_pct:g}%{', forced' if ns.force else ''}; deletes until under {ns.low_pct:g}%)")
        for a in plan["actions"]:
            verb = "DELETE" if a["action"] == "delete" and not a["error"] else "FAILED" if a["error"] else "skip"
            tail = f"  -> {a['pct_after']}%" if a["pct_after"] is not None else ""
            path = f"{a['path']}/* (unrecorded rest)" if a["rest"] else a["path"]
            print(f"{verb:<7} {_size(a['bytes']):>10}  {path}  {a['reason']}{tail}" + (f"  ERROR {a['error']}" if a["error"] else ""))
        n = sum(1 for a in plan["actions"] if a["action"] == "delete" and not a["error"])
        print(f"{'deleted' if ns.apply else 'would delete'} {n} path(s), {_size(plan['freed_bytes'])}: {plan['pct_before']}% -> "
              f"{plan['pct_after']}%" + ("" if ns.apply else "; nothing was deleted") + (f"; {len(failed)} deletion(s) failed" if failed else ""))
        for e in plan["errors"]:
            print(f"error: {e}", file=sys.stderr)
        return 1 if failed else 0
    if sub == "rm":
        ap = argparse.ArgumentParser(prog="research retention rm", description="delete PATHs with their owner's yes, checked against "
                                     "every record that covers them or lies beneath them; a dry run unless --apply")
        ap.add_argument("paths", nargs="+", metavar="PATH")
        ap.add_argument("--approved-by", required=True, metavar="@OWNER", help="the owner who said yes; every record covering a PATH or "
                        "beneath it must name this handle")
        ap.add_argument("--ref", required=True, metavar="REF", help="where the owner said yes (a Slack permalink or thread ts)")
        ap.add_argument("--by", default=os.environ.get("RESEARCH_AGENT_URL"), metavar="URL",
                        help="who deletes (default $RESEARCH_AGENT_URL); --apply needs it")
        ap.add_argument("--root", default=str(DEFAULT_ROOT))
        ap.add_argument("--jobs-dir", action="append", default=[], metavar="D", help="more job-script directories (beside "
                        + " and ".join(map(str, DEFAULT_JOBS_DIRS)) + " when they exist): a path a file there names is in use")
        ap.add_argument("--store", default=None, help="the evidence store a deletion of passes checks its artifact against")
        ap.add_argument("--json", action="store_true")
        ap.add_argument("--apply", action="store_true", help=f"delete, and log each decision to <root>/{LOG}; without it nothing "
                                                             "is deleted or logged")
        ns = ap.parse_args(rest)
        if missing := [d for d in [ns.root, *ns.jobs_dir] if not Path(d).is_dir()]:
            print(f"research retention rm: {missing[0]} is not a directory", file=sys.stderr)
            return 2
        jobs = [d for d in DEFAULT_JOBS_DIRS if d.is_dir()] + [Path(d) for d in ns.jobs_dir]
        try:
            plan = rm(ns.paths, approved_by=ns.approved_by, ref=ns.ref, root=ns.root, apply=ns.apply, by=ns.by,
                      in_use=JobScripts(jobs, ns.root), now=utc_now, store_root=ns.store)
        except RecordError as e:
            print(f"research retention rm: {e}", file=sys.stderr)
            return 2
        done = all(a["action"] == "delete" for a in plan["actions"])
        if ns.json:
            print(json.dumps(plan, indent=2))
            return 0 if done else 1
        mode = f"applying, logged to {Path(plan['root']) / LOG}" if ns.apply else "dry run (--apply deletes)"
        print(f"research retention rm: {mode}: approved by {plan['approved_by']} ({plan['ref']})")
        for a in plan["actions"]:
            verb = {"delete": "DELETE" if ns.apply else "would delete", "refuse": "REFUSED", "failed": "FAILED"}[a["action"]]
            print(f"{verb:<12} {_size(a['bytes']):>10}  {a['path']}  {a['reason']}" + (f"  ERROR {a['error']}" if a["error"] else ""))
        n = sum(1 for a in plan["actions"] if a["action"] == "delete")
        print(f"{'deleted' if ns.apply else 'would delete'} {n} of {len(plan['actions'])} path(s), {_size(plan['freed_bytes'])}"
              + ("" if ns.apply else "; nothing was deleted"))
        return 0 if done else 1
    print(f"research retention: unknown subcommand {sub!r} (have: report, gc, rm)", file=sys.stderr)
    return 2


__all__ = ["DEFAULT_JOBS_DIRS", "FIELDS", "JobScripts", "KIND", "POLICY", "PRIORITIES", "RUN_EXPIRY", "SCHEMA", "SIDECAR", "RecordError",
           "beside", "check", "covers", "find", "gc", "implicit", "keep", "keep_main", "load", "main", "problems", "record_at", "report", "scan",
           "sidecar", "store_check", "target", "write"]

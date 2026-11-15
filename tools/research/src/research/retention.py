"""Retention of a node's data (`research/retention/v0`): nothing under a node's /workspace is deleted without a record that allows it.

    research keep PATH --owner @LANE --expires DATE|never --priority low|normal|high|keep [--regenerate CMD] [--art ART | --preserve]
                       [--why TEXT] [--by URL] [--root R] [--store S]
    research keep show PATH [--root R]
    research retention report [--root R] [--depth N] [--json]
    research retention gc [--root R] [--policy default] [--high-pct 80] [--low-pct 75] [--force] [--max-paths 200] [--jobs-dir D]...
                          [--store S] [--unpreserved-runs] [--json] [--apply]
    research retention rm PATH... --approved-by @OWNER --ref REF [--by URL] [--root R] [--jobs-dir D]... [--store S] [--json] [--apply]

A record is a sidecar: `<dir>/.retention.json` for a directory, `<file>.retention.json` beside a file, so it travels with its data
and `cat` reads it. A directory its owner made read-only has its record beside it, `<dir>.retention.json`; one with both is invalid. It covers everything beneath its path: the walk never descends into a recorded path. Its fields are FIELDS:
`path` (relative to the root), `node`, `owner` (a lane handle), `by`, `created`, `expires` (UTC, or null for never), `priority`
(PRIORITIES), `preserved` (an `art:` id once the data is on the remote, or null), `regenerate` (how to rebuild it, or null), `why`.

A directory whose `.custody` marker (store.custody) says `preserved: true` and which has no sidecar has an implicit record
(`implicit`): owner its job.json's lane, else its campaign, else `runner`; `preserved` the marker's run record; `expires`
RUN_EXPIRY after the marker's `verified_utc`; priority `normal`. Nothing is written for it, and a sidecar always wins.

A finished or runner-dead run that custody did not preserve gets a low-priority retention record that expires
UNPRESERVED_RUN_EXPIRY (7 days) after the run directory's last write, so the collector takes it. That record is implicit too
(`implicit`, source UNPRESERVED), with the same owner; `preserved` and `regenerate` are null, since the directory is the run's only
copy. A directory has it only when all of these hold. It is a run directory: named as a run id, holding the job.json of that id
whose `run.runs_root` is the directory it lies in, and neither a launcher's `remote.json` nor the `preserved.json` of a fetched
copy (`custody.has_custody`). Its run has ended (`evict._run_state`): its status.json is final, or it was refused, or its status
is `running` and the pid it records, at the start time it records, is gone from this host's process table, read
only for a run its job.json places on this host and from a table that shows pid 1, so one that hides other users' processes
shows nobody dead. Custody did not preserve it: it has no `.custody`, or one that says `preserved: false`; a marker that can't be
read gives no record. Neither it nor a directory above it has a sidecar: an explicit record always wins, whatever its priority
or expiry. No directory above it is named as a run id: a run inside another run's directory is that run's data. Its last write
is the newest mtime of the directory and of every directory and file under it (`_last_write`: one scandir and lstat walk that
follows no symlink). The record never expires before the custody key its runner publishes with does (`<request
dir>/custody/cred.json`, which the runner deletes when it is done), so not while that runner may still be publishing, which it
does after the status is final. A run that waits in a slot line is `running`, and one still queued has no job.json, so neither
has a record. A run directory whose status, files or custody key can't be read has no record either: it is listed as invalid
with what could not be read, since whether it may go is unknown, and is never deleted.
gc deletes these runs only with --unpreserved-runs: without it a pass lists them apart and leaves them, so a machine's first pass
can be read before anything goes. It never takes one whose clone still stands (CLONES: `<runs root>/../scratch/<run id>`, where
`research run --cwd clone` ran it on a machine): what removes a run's clone decides that its run is over from the run directory's
launch.json and status.json, so the clone goes first and the run directory at a pass after; until then gc lists the run with that
reason. `research keep RUNDIR ...` before the 7 days are over keeps a run, and so does preserving it
(`research data custody RUN --publish`).

Policy `default` (POLICY) acts when the root's filesystem is over high-pct (or with --force) and deletes, lowest priority first and
then the oldest expiry, until it is under low-pct. A filesystem's use is the larger of its blocks' and its inodes' (`inode_usage`):
a disk out of inodes refuses every write as one out of blocks does, even with most of its blocks free. One pass deletes at most
--max-paths paths (MAX_PATHS; 0 lifts the cap), and the next takes the rest. A path qualifies only with a valid record (explicit or
implicit) recorded on this node that is expired, `low` or `normal`, and preserved or regenerable, or an unpreserved run's under
--unpreserved-runs; and no file in the job-script directories may name it (`JobScripts`). A preserved run's implicit record also
needs its marker to still vouch for the run dir (`custody.marker_problem`), and an unpreserved run goes only when no clone of
it stands and this user can write into every directory in it: one that can't go is left and named, and takes none of the pass's
--max-paths. A path holding a nested record is left whole. A dry run (the default) reads no remote.

A record's expiry is its owner's word that the data may go. The remote check (`store_check`, just before each deletion of a
preserved path) is there so that gc never deletes the only copy of data somebody still promises to keep. Its answer is one of
STORED. gc deletes on `verified` (the remote holds the artifact; with no remote to ask, as on a node, which holds no store key, the
replica this store's catalog verified when custody pushed it), on `expired` (the remote holds its manifest and has dropped its blobs
past its kind's time to live, `store.gc.TTL_DAYS`) and on `withdrawn` (every blob the remote lacks is under a `withdrawn` label,
`research data preserved`'s WITHDRAWN): there the store gave the copy up by its own rule, so nothing still promised is lost. It
keeps the path on `uncovered` (the artifact doesn't hold every file now under the path at its size, so files custody trimmed never
count as stored, whatever the remote says), on `mismatch` (the remote's copy is not the artifact's, or a blob is absent that neither
rule explains), on `unknown` (the remote has no manifest for it: it never stored it) and on `unreachable` (the remote, or with none
this store, could not be asked), and an `unreachable` fails the pass (exit 1). Every --apply reports how many of each it found.

Neither command deletes a path that is, lies in or holds a Lean records tree (LEAN_RECORDS) whose `.lock` a run holds: it is
refused, naming the run its `.who` names, to retry when that run ends. Each deletion holds those trees' locks while it runs, as
`lean_audit.py --prune` does, so no run takes one meanwhile.

`rm` deletes the paths it is given, with their owner's yes, for a cleanup the policy doesn't reach. A path qualifies only when a
record covers it (its own or an ancestor's), that record and every record beneath it (sidecars and implicit ones) are valid, name
the --approved-by handle as owner and are not `keep`, no job script names it, and nothing in it is read-only or a mount. A sealed
run dir (store.sealed) is the one read-only path either command deletes: whole, by `sudo -n vy-seal-runs delete` once every check
here passed, which redoes custody's as root; a path inside one is refused. A path no
record covers has nobody to say yes, so it is never deleted. A path that is, holds or lies under a `passes` directory (PASSES; not a Python package of that name, nor anything inside a
venv, `_holds_passes`) also
needs its covering record's `preserved` artifact to hold that record's whole path, `verified` on the remote (`store_check`, just
before --apply deletes): a yes alone never deletes retained passes. Each record is read again just before --apply deletes, and every
decision of an --apply goes to the node's deletion log, `<root>/LOG`, with who deleted, the approver and --ref. A dry run (the
default) deletes and logs nothing.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shutil
import socket
import stat
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .disk import HOLD_PCT, RELEASE_PCT
from .store import custody as CU
from .store import evict as EV
from .store import ids
from .store import sealed as SL
from .telemetry.job import write_json_atomic

SCHEMA = "research/retention/v0"
SIDECAR = ".retention.json"
FIELDS = ("schema", "path", "node", "owner", "by", "created", "expires", "priority", "preserved", "regenerate", "why")
PRIORITIES = ("low", "normal", "high", "keep")
COLLECTABLE = ("low", "normal")
RUN_EXPIRY = timedelta(hours=48)
#: How long after its last write the directory of an ended run that custody did not preserve keeps its implicit record (module
#: docstring): the one number of that rule, which the record's `why`, gc's lines and the README state.
UNPRESERVED_RUN_EXPIRY = timedelta(days=7)
#: The source (`Found.source`, `Row.source`) of that record.
UNPRESERVED = "unpreserved"
#: What the report calls an implicit record, by its source.
IMPLICIT = {"custody": ".custody", UNPRESERVED: "unpreserved run"}
#: The directory beside a machine's runs root that holds its runs' clones, `<CLONES>/<run id>` (`clone.home`).
CLONES = "scratch"
#: The line the report and gc print above the unpreserved runs, which they list after every other path.
UNPRESERVED_HEAD = (f"unpreserved runs (ended, and custody did not preserve them: each directory is its run's only copy, and expires "
                    f"{UNPRESERVED_RUN_EXPIRY // timedelta(days=1)} d after its last write):")
DEFAULT_ROOT = Path("/workspace")
DEFAULT_JOBS_DIRS = (Path("/workspace/pouw/fill/queue"), Path("/workspace/pouw/fill/running"))
POLICY = "default/v0"
POLICIES = ("default",)
KIND = "node-data/v1"
LOG = Path("research/retention/deletions.jsonl")
PASSES = "passes"
LOG_SCHEMA = "research/retention-log/v1"
#: The paths one gc pass deletes at most (--max-paths): a pass that is wrong about what may go is wrong about this many.
MAX_PATHS = 200
#: What the remote check finds of a preserved path's stored copy (module docstring), in the order a pass counts them; gc deletes on
#: COLLECTED and keeps the path on the rest.
STORED = ("verified", "expired", "withdrawn", "uncovered", "mismatch", "unknown", "unreachable")
COLLECTED = STORED[:3]
RM_SCHEMA = "research/retention-rm/v0"
OWNER_RE = re.compile(r"^@[A-Za-z0-9][A-Za-z0-9._-]*$")
TS = "%Y-%m-%dT%H:%M:%SZ"
NAME_CHAR = r"[\w.-]"

disk_usage = shutil.disk_usage
#: The process table a recorded runner's death is read from (`_ended`).
PROC = Path("/proc")


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


def _last_write(path: Path) -> float:
    """The newest mtime of the directory `path` and of every directory and file under it, by os.scandir and lstat (no symlink is
    followed); OSError when any of it can't be read, so a directory never reads as older than it is."""
    newest, stack = os.lstat(path).st_mtime, [str(path)]
    while stack:
        with os.scandir(stack.pop()) as it:
            for e in it:
                newest = max(newest, e.stat(follow_symlinks=False).st_mtime)
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
    return newest


def _ended(run_dir: Path, host: Any) -> str | None:
    """The state the run of `run_dir` ended in (one of `evict.ENDED`), or None while nothing shows that it ended: a state that is
    not final, or a `running` whose runner is not known gone. A recorded runner is gone as `evict._run_state` decides it, and
    only where this host's process table can say (module docstring); `host` is job.json's. `evict.Unestablished` when its status
    can't be read."""
    state = EV._run_state(run_dir, PROC)
    if state == EV.LOST and (host != socket.gethostname() or not (PROC / "1" / "stat").is_file()):
        return None
    return state if state in EV.ENDED else None


def _unpreserved(d: Path, rel: str, job: dict, owner: str) -> dict | None:
    """The record of an ended run that custody did not preserve (module docstring), given that `.custody` doesn't say preserved;
    None for a directory that is not such a run's. RecordError when it is a run directory and what decides whether it may go
    can't be read: its status, something under it, or its runner's custody key."""
    run_id, runs_root, host = EV._get(job, "run", "id"), EV._get(job, "run", "runs_root"), EV._get(job, "environment", "hostname")
    if run_id != d.name or not ids.is_run_id(run_id) or not isinstance(host, str) or not host:
        return None
    if not isinstance(runs_root, str) or not os.path.isabs(runs_root) or os.path.realpath(runs_root) != os.path.realpath(d.parent):
        return None
    above = d.parents[:rel.count("/")]
    if os.path.lexists(d / CU.LAUNCHER_RECORD) or any(ids.is_run_id(a.name) or any(map(os.path.lexists, _places(a))) for a in above):
        return None
    try:
        if CU.has_custody(d)[0] or (state := _ended(d, host)) is None:
            return None
    except EV.Unestablished as e:
        raise RecordError(f"whether the run of {d} ended can't be read ({str(e).partition(': ')[2]})") from None
    except OSError as e:
        raise RecordError(f"whether the run of {d} ended can't be read ({e})") from None
    try:
        expires = datetime.fromtimestamp(math.ceil(_last_write(d)), UTC) + UNPRESERVED_RUN_EXPIRY
    except (OSError, OverflowError, ValueError) as e:
        raise RecordError(f"when {d} was last written can't be read ({e})") from None
    request = EV._get(job, "run", "remote", "request")
    if isinstance(request, str):
        key = Path(request).parent / CU.DIR_NAME / CU.CRED_FILE
        try:
            held = EV._quiet(lambda: os.lstat(key)) is not None
        except OSError as e:
            raise RecordError(f"whether the runner of {d} still holds its custody key can't be read ({e})") from None
        if held:
            key_expires = _utc_or_none(CU._key_expiry(key.parent))
            if key_expires is None:
                raise RecordError(f"the expiry of the custody key {key} can't be read, so whether the runner of {d} still publishes is unknown")
            expires = max(expires, key_expires)
    created = _utc_or_none(EV._get(job, "run", "created_utc"))
    ended = "its runner is gone" if state == EV.LOST else f"it ended {state}"
    return {"schema": SCHEMA, "path": rel, "node": host, "owner": owner, "by": None, "created": fmt_utc(created) if created else None,
            "expires": fmt_utc(expires), "priority": PRIORITIES[0], "preserved": None, "regenerate": None,
            "why": f"no run custody ({ended}): this directory is the run's only copy, and it expires "
                   f"{UNPRESERVED_RUN_EXPIRY // timedelta(days=1)} d after its last write"}


def _implicit(run_dir: Path, rel: str) -> tuple[dict, str] | None:
    """`implicit`'s record with its source: "custody" for a preserved run's, UNPRESERVED for an ended, unpreserved one's;
    `_unpreserved`'s RecordError."""
    d = Path(run_dir)
    m = _read_json(d / CU.MARKER)
    if m is None:
        # a marker that can't be read says nothing of what custody did; and only a directory named as a run can be an unpreserved one
        if os.path.lexists(d / CU.MARKER) or not ids.is_run_id(d.name):
            return None
    elif m.get("schema") != CU.SCHEMA or not isinstance(m.get("preserved"), bool):
        return None
    job = _read_json(d / "job.json") or {}
    owner = (EV._get(job, "ownership", "lane") or EV._get(job, "store", "campaign") or EV._get(job, "store", "conditions", "exec.campaign")
             or "runner")
    if m is None or not m["preserved"]:
        rec = _unpreserved(d, rel, job, owner)
        return (rec, UNPRESERVED) if rec else None
    verified = _utc_or_none(m.get("verified_utc"))
    created = _utc_or_none(EV._get(job, "run", "created_utc")) or verified
    rr = m.get("run_record")
    rec = {"schema": SCHEMA, "path": rel, "node": m.get("host"), "owner": owner, "by": None,
           "created": fmt_utc(created) if created else None, "expires": fmt_utc(verified + RUN_EXPIRY) if verified else None,
           "priority": "normal", "preserved": rr if ids.is_art(rr) else None, "regenerate": None,
           "why": f"run custody: the run record is on the remote, and the local copy expires {RUN_EXPIRY // timedelta(hours=1)} h after it"}
    return rec, "custody"


def implicit(run_dir: Path, rel: str) -> dict | None:
    """The record a run directory has without a sidecar (module docstring): a preserved run's, from its `.custody` marker, or an
    ended run's that custody did not preserve; None for any other directory. RecordError for a run directory whose status, files
    or custody key can't be read, so that whether it may go is unknown."""
    found = _implicit(run_dir, rel)
    return found[0] if found else None


@dataclass
class Found:
    path: Path
    record: dict | None
    source: str                  # "sidecar" | "custody" | UNPRESERVED
    error: str | None = None     # an invalid sidecar's problems, or what of an unpreserved run can't be read (then no record)


def record_at(path: Path, rel: str) -> Found | None:
    """The record of exactly `path` (its sidecar, else a directory's implicit one), or None. A sidecar that is invalid, and a run
    directory of which too little can be read to say whether it may go, is a Found with its `error` and no record."""
    p = Path(path)
    try:
        doc = load(p)
    except RecordError as e:
        return Found(p, None, "sidecar", str(e))
    if doc is not None:
        return Found(p, doc, "sidecar")
    if p.is_dir() and not p.is_symlink():
        try:
            found = _implicit(p, rel)
        except RecordError as e:
            return Found(p, None, UNPRESERVED, str(e))
        if found is not None:
            return Found(p, *found)
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
    source: str | None = None            # "sidecar" | "custody" | UNPRESERVED; None: no record
    error: str | None = None
    nested: list[str] = field(default_factory=list)
    rest: bool = False                   # an unowned row for the unrecorded rest of a partly recorded area
    dirs: int = 0                        # the directories among a recorded path's inodes, itself included

    def state(self, now: datetime) -> str:
        if self.error:
            return "invalid"
        if self.record is None:
            return "unowned"
        return "expired" if expired(self.record, now) else "recorded"

    def doc(self, now: datetime) -> dict[str, Any]:
        r = self.record or {}
        return {"path": self.path, "state": self.state(now), "bytes": self.bytes, "files": self.files,
                "inodes": self.files + self.dirs if self.source else None, "source": self.source,
                "owner": r.get("owner"), "expires": r.get("expires"), "priority": r.get("priority"), "preserved": r.get("preserved"),
                "regenerable": bool((r.get("regenerate") or "").strip()), "expired": bool(self.record) and expired(self.record, now),
                "rest": self.rest, "nested": self.nested, "error": self.error, "record": self.record}


def expired(record: dict, now: datetime) -> bool:
    e = _utc_or_none(record.get("expires"))
    return e is not None and e <= now


def measure(path: Path, errors: list[str]) -> tuple[int, int, int, list[Path]]:
    """(apparent bytes and count of every non-directory entry under `path`, the count of its directories with itself, the sidecars
    beneath its own), by os.scandir and lstat; symlinks are never followed."""
    try:
        st = os.lstat(path)
    except OSError as e:
        errors.append(f"{path}: {e.strerror or e}")
        return 0, 0, 0, []
    if not stat.S_ISDIR(st.st_mode):
        return st.st_size, 1, 0, []
    total, n, dirs, nested, stack = 0, 0, 0, [], [str(path)]
    while stack:
        d = stack.pop()
        dirs += 1
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
    return total, n, dirs, sorted(nested)


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
                b, n, dirs, nested = measure(Path(e.path), errors)
                rows.append(Row(r, b, n, found.record, found.source, found.error, [relative(x, root) for x in nested], dirs=dirs))
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


def unpreserved_runs(docs: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """The unpreserved runs among the rows `docs` (`Row.doc`), counted apart from every other path: the `expired` ones, past
    UNPRESERVED_RUN_EXPIRY after their last write, the `recorded` ones, still inside it, and the `invalid` ones, run directories
    with no record because too little of them can be read; each with its paths, bytes and inodes."""
    out = {state: {"paths": 0, "bytes": 0, "inodes": 0} for state in ("expired", "recorded", "invalid")}
    for d in docs:
        if d["source"] == UNPRESERVED:
            for k, v in (("paths", 1), ("bytes", d["bytes"]), ("inodes", d["inodes"])):
                out[d["state"]][k] += v
    return out


def report(root: Path | str = DEFAULT_ROOT, depth: int = 2, *, now: Callable[[], datetime] = utc_now,
           usage: Callable[[Path], Any] = shutil.disk_usage, node: str | None = None) -> dict[str, Any]:
    """`research retention report --json`: unowned (and invalid) areas first, then expired paths, then the rest, each by size.
    `unpreserved` counts the unpreserved runs apart (`unpreserved_runs`)."""
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
            "rows": docs, "totals": totals, "unpreserved": unpreserved_runs(docs), "errors": sc.errors}


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


@dataclass(frozen=True)
class Stored:
    """A remote check's answer for one preserved path: `state`, one of STORED, and `why` in words."""
    state: str
    why: str


def store_check(store_root: str | None = None, now: Callable[[], datetime] = utc_now) -> Callable[[str, Path], Stored]:
    """The default remote check, run just before a deletion: what became of the path's stored copy (STORED, module docstring).
    The artifact must hold the path's files (`covers`); then `preserved.check` in head mode says whether the remote holds it (remote
    sizes, and the single-part ETag against the local blob's MD5 where the blob is here, else a read-back; `LocalStore.verify`
    would read back every blob). Where it holds the manifest and lacks blobs, the store gave the copy up when that check's verdict is
    WITHDRAWN (it pulls the artifact's labels from the remote, and a `withdrawn` label covers every blob the remote lacks), or when
    the manifest is older at `now` than its kind's time to live; a blob absent that neither explains is a `mismatch`. Without a
    usable remote (a node holds no store key) it is `recorded` mode: the remote replica the store's catalog verified when custody
    pushed it, which is what the store eviction trusts too; an artifact that catalog doesn't vouch for is `unreachable`, since
    nothing here can be asked."""
    st = None

    def chk(art: str, path: Path) -> Stored:
        nonlocal st
        from .store import gc as G
        from .store import preserved as P
        from .store.local import LocalStore, NotFound, manifest_key, object_key

        try:
            st = st or LocalStore(store_root)
            r = st.remote
            nobody = (f"no remote can be asked from here, and the store at {st.root} doesn't vouch for it (on a node, --store or "
                      f"RESEARCH_STORE names the node's store, /workspace/research/store)")
            try:
                man = st.get_manifest(art)
            except NotFound as e:
                return Stored("unknown", str(e)) if r is not None else Stored("unreachable", f"{e}: {nobody}")
            if why := covers(man, path):
                return Stored("uncovered", why)
            rep = P.check(st, [art], mode="head" if r is not None else "recorded")
            if rep["preserved"]:
                return Stored("verified", "verified on the remote" if r is not None else
                              "verified on the remote when custody pushed it, by this store's catalog")
            if err := rep["targets"][0].get("error"):
                return Stored("unreachable", err)
            if rep["verdict"] == P.WITHDRAWN:
                return Stored("withdrawn", f"withdrawn from the remote: {st.withdrawn(art)}")
            # what fails besides withdrawn blobs: `preserved` lists those in the artifact's `withdrawn`, never as absent in its errors
            errs = (rep["artifacts"][art]["errors"] + [e for x in rep["not_preserved"] if x["kind"] == "labels" for e in x["errors"]]
                    or [rep.get("reason") or "not preserved"])
            if r is None:
                return Stored("unreachable", f"{errs[0]}: {nobody}")
            mkey = manifest_key(ids.hex_of(art))
            if f"{mkey}: absent" in errs:
                return Stored("unknown", f"{r.describe()} has no manifest for it, so it never stored it")
            dropped = {f"{object_key(h)}: absent" for h, _ in man.blobs()}
            if other := [e for e in errs if e not in dropped]:
                return Stored("mismatch", other[0])
            ttl, age = G.TTL_DAYS.get(man.kind), (now() - parse_utc(dict(r.list_times(mkey))[mkey])).total_seconds() / 86400
            if ttl is not None and age >= ttl:
                return Stored("expired", f"expired on the remote: a {man.kind} lives {ttl:g} d there, and its manifest is {age:.1f} d old")
            return Stored("mismatch", f"{errs[0]}, which neither a withdrawal nor its kind's time to live explains")
        except Exception as e:  # noqa: BLE001  a check that could not be made keeps the path and fails the pass
            return Stored("unreachable", f"{type(e).__name__}: {e}")

    return chk


@dataclass
class Action:
    row: Row
    action: str                  # "delete" | "skip"
    reason: str
    pct_after: float | None = None
    error: str | None = None
    stored: str | None = None    # the remote check's answer (STORED), where an --apply asked

    def doc(self, now: datetime) -> dict[str, Any]:
        return {**self.row.doc(now), "action": self.action, "reason": self.reason, "pct_after": self.pct_after, "error": self.error,
                "stored": self.stored}


def ineligible(row: Row, now: datetime, node: str) -> str | None:
    """Why the policy never deletes `row`'s path, from its record alone; None when it may."""
    rec = row.record
    if row.error:
        return f"no record: {row.error}" if row.source == UNPRESERVED else f"invalid record: {row.error}"
    if rec is None:
        return "no record: a path without one is never deleted (`research keep PATH ...` gives it one)"
    if rec["priority"] not in COLLECTABLE:
        return f"priority {rec['priority']}: never collected"
    if rec.get("expires") is None:
        return "never expires"
    if not expired(rec, now):
        return f"not expired: expires {rec['expires']}"
    if row.source != UNPRESERVED and not rec.get("preserved") and not (rec.get("regenerate") or "").strip():
        return "neither preserved nor regenerable: nothing would bring it back"
    if rec.get("node") and rec["node"] != node:
        return f"recorded on {rec['node']}, and this is {node}: `research keep` it here"
    if row.nested:
        return f"holds {len(row.nested)} nested record(s), e.g. {row.nested[0]}: a record covers everything beneath it, so remove those first"
    return None


def _why_deletable(row: Row, found: Stored | None) -> str:
    rec = row.record
    if row.source == UNPRESERVED:
        how = "the only copy of a run that custody did not preserve"
    elif rec.get("preserved"):
        how = f"preserved {rec['preserved']}" + (f" ({found.why})" if found else ", checked on the remote just before an --apply deletes it")
    else:
        how = "regenerable"
    return f"{rec['priority']}, expired {rec['expires']}, {how}"


def keeping(run_dir: Path) -> str:
    """The two ways its lane keeps an unpreserved run's directory, as gc prints them for a run it has not deleted."""
    return (f"to keep it: `research keep {run_dir} --owner @LANE --expires DATE --priority P`, or preserve it: `research data custody "
            f"{run_dir.name} --publish`")


def _clone_first(run_dir: Path) -> str | None:
    """Why gc leaves the unpreserved run of `run_dir` for now: its clone still stands (module docstring; a directory, never a
    link to one), or whether it does can't be read. None when it has no clone."""
    clone = run_dir.parent.parent / CLONES / run_dir.name
    try:
        st = EV._quiet(lambda: os.lstat(clone))
    except OSError as e:
        return f"whether its clone {clone} still stands can't be read ({e}), and a clone goes before its run's directory"
    if st is None or not stat.S_ISDIR(st.st_mode):
        return None
    return (f"its clone {clone} still stands, and the clone goes first: what removes a run's clone decides that its run is over from "
            f"this directory's launch.json and status.json, so this directory goes at a pass after the clone has gone")


def _read_only_dir(path: Path) -> str | None:
    for base, _dirs, _files in os.walk(path, onerror=_raise):
        if not os.access(base, os.W_OK | os.X_OK):
            return base
    return None


def _unwritable(path: Path) -> str | None:
    """Why this user can't remove the tree `path` whole (a directory in it that it can't write into, or can't read), or None."""
    try:
        ro = _read_only_dir(path)
    except OSError as e:
        return f"can't read all of it ({e}), so it can't be removed whole"
    return f"{ro} can't be written by this user, so the tree can't be removed whole: its owner makes it writable, or removes it" if ro else None


def _in_sealed(path: Path) -> str | None:
    if SL.sealed(Path(path).parent):
        return f"{path} is inside a sealed run dir: only the whole run dir is deleted"
    return None


#: A Lean records tree (`lean_changed.py --records`, `--try`, `--build`): a slot's checkout and its builds, `lean-records/build-I`
#: under the check cache (`lean_changed.RECORDS`), or that link's target on a larger volume. The run working in it holds its
#: `.lock` (a flock) and names itself in its `.who` (`lean_changed.hold_tree`). `lean_audit.prune_records` removes a tree only
#: while it holds that lock, and so do gc and rm (`_lean_locks`): a build whose tree goes from under it fails on a missing `.olean`.
LEAN_RECORDS, LEAN_LOCK, LEAN_WHO = "lean-records", ".lock", ".who"


def _lean_tree(d: Path) -> bool:
    """Whether `d` (or the directory a link `d` names) is a Lean records tree: it holds a `.lock`, and lies in `lean-records/`
    or holds a `.who`."""
    try:
        return (d / LEAN_LOCK).is_file() and (d.parent.name == LEAN_RECORDS or (d / LEAN_WHO).is_file())
    except OSError:
        return False


def lean_trees(path: Path, root: Path) -> list[Path]:
    """The Lean records trees `path` is or lies in (below `root`), else those it holds: a link to one counts, and the walk never
    follows a link or descends into a tree or CODE_DIRS. OSError when a directory can't be read."""
    p = Path(os.path.abspath(path))
    out = [d for d in (p, *p.parents) if Path(root) in d.parents and _lean_tree(d)]
    if out or not p.is_dir() or p.is_symlink():
        return out
    for base, dirs, _files in os.walk(p, onerror=_raise):
        dirs.sort()
        found = [d for d in dirs if _lean_tree(Path(base) / d)]
        out += [Path(base) / d for d in found]
        dirs[:] = [d for d in dirs if d not in found and d not in CODE_DIRS]
    return out


def _lean_holder(tree: Path) -> str:
    try:
        who = (tree / LEAN_WHO).read_text().strip()
    except OSError:
        who = ""
    return (f"{tree} is a Lean records tree, and {who or 'a run its .who does not name'} holds its lock ({tree / LEAN_LOCK}): "
            f"deleting it would crash that build; retry when the run ends")


@contextmanager
def _lean_locks(path: Path, root: Path) -> Iterator[str | None]:
    """Hold the lock of every Lean records tree `path` is, lies in or holds (`lean_trees`) while the block runs, so no run takes
    one meanwhile; yields None, or why not when a run holds one. OSError when a directory or a lock can't be read."""
    with ExitStack() as stack:
        held = None
        for tree in lean_trees(path, root):
            fd = os.open(tree / LEAN_LOCK, os.O_RDONLY)
            stack.callback(os.close, fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held = _lean_holder(tree)
                break
        yield held


def lean_in_use(path: Path, root: Path) -> str | None:
    """Why a Lean build is (or may be) working in `path` (`_lean_locks`), or None."""
    try:
        with _lean_locks(path, root) as held:
            return held
    except OSError as e:
        return f"can't read all of it ({e}), so whether a Lean build works in it is unknown"


def _delete(root: Path, path: Path) -> str | None:
    """Remove a recorded path with its sidecar; the error, or None. A tree holding a directory this process can't write into is
    left whole, since rmtree would stop partway; a sealed run dir (store.sealed) is deleted whole by vy-seal-runs, as root. The
    locks of the Lean records trees it is, lies in or holds are held throughout (`_lean_locks`), and one a run holds refuses it."""
    try:
        if root not in Path(os.path.abspath(path)).parents:
            return f"{path} is not under {root}"
        if os.path.ismount(path):
            return f"{path} is a mount point"
        if why := _in_sealed(path):
            return why
        with _lean_locks(path, root) as held:
            if held:
                return held
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
       remote_check: Callable[[str, Path], Stored] | None = None, in_use: Callable[[Path], str | None] | None = None,
       node: str | None = None, store_root: str | None = None,
       inodes: Callable[[Path], tuple[int, int]] | None = None, max_paths: int | None = None,
       unpreserved: bool = False) -> dict[str, Any]:
    """The policy (module docstring) over `root`: the plan, and with `apply` its deletions. `jobs_dirs` None means the
    DEFAULT_JOBS_DIRS that exist; `remote_check` None is `store_check(store_root, now)`; `in_use` None is `JobScripts(jobs_dirs)`;
    `inodes` (`inode_usage`) makes the filesystem's use the larger of its blocks' and its inodes', None its blocks' alone;
    `max_paths` caps the paths this pass deletes, None not at all; `unpreserved` (--unpreserved-runs) lets it delete the
    unpreserved runs, which it otherwise lists and leaves. The plan's `stored` counts the remote check's answers (STORED)
    over the paths an --apply asked about, `capped` the paths the cap left for the next pass, and `unpreserved` the unpreserved
    runs, apart from every other path (`unpreserved_runs`), with the `deleted` ones this pass deleted, or in a dry run would."""
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
                            "max_paths": max_paths, "acting": force or pct > high_pct, "actions": [], "freed_bytes": 0, "errors": [],
                            "stored": dict.fromkeys(STORED, 0), "capped": 0, "deletes_unpreserved": unpreserved,
                            "unpreserved": {**unpreserved_runs([]), "deleted": unpreserved_runs([])["expired"]}}
    if not plan["acting"]:
        plan["pct_after"] = plan["pct_before"]
        return plan
    if jobs_dirs is None:
        jobs_dirs = [d for d in DEFAULT_JOBS_DIRS if d.is_dir()]
    in_use = in_use or JobScripts(jobs_dirs, root)
    remote_check = remote_check or store_check(store_root, now)
    sc = scan(root, depth)
    plan["errors"] = sc.errors
    candidates, skips = [], []
    for row in sorted(sc.rows, key=lambda r: (-r.bytes, r.path)):
        why = ineligible(row, t, node)
        if why is None and row.source == "custody" and (problem := CU.marker_problem(root / row.path)):
            why = f"its custody marker no longer vouches for it: {problem}"
        if why is None and (u := in_use(root / row.path)):
            why = f"in use: {u}"
        if why is None and row.source == UNPRESERVED:
            why = _clone_first(root / row.path)
        if why is None and row.source == UNPRESERVED and not unpreserved:
            why = f"left: {_why_deletable(row, None)}, which only --unpreserved-runs deletes"
        if why:
            skips.append(Action(row, "skip", why))
        else:
            candidates.append(row)
    candidates.sort(key=lambda r: (PRIORITIES.index(r.record["priority"]), parse_utc(r.record["expires"]), r.path))
    acts, deletions = [], 0
    for row in candidates:
        p = root / row.path
        if pct < low_pct:
            acts.append(Action(row, "skip", f"not needed: the filesystem is at {pct:.1f}%, under {low_pct:g}%"))
            continue
        if max_paths is not None and deletions >= max_paths:
            plan["capped"] += 1
            acts.append(Action(row, "skip", f"left for the next pass: this one has its {max_paths} path(s) (--max-paths)"))
            continue
        fresh = record_at(p, row.path)
        why = found = None
        if fresh is None or fresh.record != row.record:
            why = "its record changed during gc: run gc again"
        elif u := in_use(p):
            why = f"in use: {u}"
        elif u := lean_in_use(p, root):
            why = f"in use: {u}"
        elif row.source == UNPRESERVED and (u := _unwritable(p)):
            why = u
        elif apply and row.record.get("preserved"):
            found = remote_check(row.record["preserved"], p)
            plan["stored"][found.state] += 1
            if found.state not in COLLECTED:
                why = f"preserved {row.record['preserved']} is not verified on the remote: {found.why}"
        if why:
            acts.append(Action(row, "skip", why, stored=found and found.state))
            continue
        deletions += 1
        if not apply:
            used -= row.bytes
            iused -= row.files + row.dirs
            pct = fullness()
            acts.append(Action(row, "delete", "would delete: " + _why_deletable(row, None), round(pct, 2)))
            plan["freed_bytes"] += row.bytes
            continue
        err = _delete(root, p)
        total, used, _ = usage(root)
        itotal, iused = inodes(root) if inodes else (0, 0)
        pct = fullness()
        acts.append(Action(row, "delete", "deleted: " + _why_deletable(row, found), round(pct, 2), err, found and found.state))
        if not err:
            plan["freed_bytes"] += row.bytes
    for a in acts + skips:
        if a.row.source == UNPRESERVED and not (apply and a.action == "delete" and not a.error):
            a.reason += f"; {keeping(root / a.row.path)}"
    plan["actions"] = [a.doc(t) for a in acts + skips]
    plan["unpreserved"] = {**unpreserved_runs(plan["actions"]),
                           "deleted": unpreserved_runs([a for a in plan["actions"] if a["action"] == "delete" and not a["error"]])["expired"]}
    plan["pct_after"] = round(pct, 2)
    return plan


# ---- rm -----------------------------------------------------------------------------------------------------------------------------------

def _handle(owner: Any) -> str:
    """An owner as a lane handle: implicit records name the job's lane without the @."""
    return "@" + str(owner or "").lstrip("@")


def beneath(path: Path, root: Path) -> list[Found]:
    """Every record strictly beneath `path`: each sidecar's, and the implicit record of each preserved `.custody` marker without one
    (an unpreserved run has no record beneath a recorded path or another run's directory, `_unpreserved`).
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
    if why := lean_in_use(path, root):
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
"""Directories that hold code, never a retained pass: not walked for PASSES (torch's and vLLM's `passes` packages sit in every
venv, and taken for passes they would refuse every approved tree holding one)."""


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
       node: str | None = None, log: Path | str | None = None, remote_check: Callable[[str, Path], Stored] | None = None,
       store_root: str | None = None) -> dict[str, Any]:
    """`research retention rm` (module docstring): the plan for `paths`, and with `apply` their deletions, every decision a line in the
    deletion log (`log`, default `root/LOG`). `in_use` None is `JobScripts` over the DEFAULT_JOBS_DIRS that exist; `remote_check` None
    is `store_check(store_root, now)`, read only by an --apply that deletes passes, which needs its answer `verified`. RecordError when --approved-by is not a handle or --ref
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
            act["bytes"], act["files"], _, _ = measure(p, errors)
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
                remote_check = remote_check or store_check(store_root, now)
                if (found := remote_check(art, top.path)).state != "verified":
                    act["error"] = f"it holds retained passes, and preserved {art} is not verified on the remote: {found.why}"
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
            f"{path}" + (f" (implicit: {IMPLICIT[d['source']]})" if d["source"] in IMPLICIT else "")
            + (f"  [{len(d['nested'])} nested]" if d["nested"] else ""))


def _apart(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`rows` without the unpreserved runs, and those: the report and gc print them after every other path, under UNPRESERVED_HEAD."""
    return [d for d in rows if d["source"] != UNPRESERVED], [d for d in rows if d["source"] == UNPRESERVED]


def _counted(c: dict[str, int]) -> str:
    return f"{c['paths']} run(s), {_size(c['bytes'])}, {c['inodes']} inode(s)"


def _unread(c: dict[str, int]) -> str:
    return f"; {_counted(c)} with no record, since what decides it can't be read" if c["paths"] else ""


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
    elif found.source == UNPRESERVED:
        doc["implicit"] = IMPLICIT[UNPRESERVED]
        print(f"research keep show: {keeping(found.path)}", file=sys.stderr)
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
        rows, runs = _apart(rep["rows"])
        for d in rows:
            print(_row_line(d))
        if runs:
            print(UNPRESERVED_HEAD)
            for d in runs:
                print(_row_line(d))
        tt = rep["totals"]
        print("totals: " + ", ".join(f"{k} {_size(v['bytes'])} in {v['paths']}" for k, v in tt.items() if k != "collectable" and v["paths"])
              + f"; collectable when the disk is full (expired, low or normal, preserved or regenerable): {_size(tt['collectable']['bytes'])} "
                f"in {tt['collectable']['paths']}")
        if runs:
            print(f"unpreserved runs, counted in the totals above: {_counted(rep['unpreserved']['expired'])} expired, which `research "
                  f"retention gc --unpreserved-runs` takes when the disk is full; {_counted(rep['unpreserved']['recorded'])} not yet expired"
                  + _unread(rep["unpreserved"]["invalid"]))
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
        ap.add_argument("--max-paths", type=int, default=MAX_PATHS, metavar="N", help="delete at most N paths in this pass and leave "
                        "the rest for the next (default %(default)s; 0: no cap)")
        ap.add_argument("--jobs-dir", action="append", default=[], metavar="D", help="more job-script directories (beside "
                        + " and ".join(map(str, DEFAULT_JOBS_DIRS)) + " when they exist): a path a file there names is in use")
        ap.add_argument("--store", default=None, metavar="ROOT", help="the evidence store that says what became of a preserved path's "
                        "stored copy (default $RESEARCH_STORE or ~/.research/store; on a node, the node's: /workspace/research/store)")
        ap.add_argument("--unpreserved-runs", action="store_true", help="take the unpreserved runs too: ended runs that custody did not "
                        f"preserve, each directory its run's only copy, expired {UNPRESERVED_RUN_EXPIRY // timedelta(days=1)} d after its "
                        "last write. Without it they are listed, apart, and left")
        ap.add_argument("--json", action="store_true")
        ap.add_argument("--apply", action="store_true", help="delete; without it nothing is deleted and no remote is read. Exit 1 "
                        "when a deletion failed or a stored copy could not be asked about")
        ns = ap.parse_args(rest)
        if ns.max_paths < 0:
            print(f"research retention gc: --max-paths {ns.max_paths}: give a count, or 0 for no cap", file=sys.stderr)
            return 2
        if missing := [d for d in [ns.root, *ns.jobs_dir] if not Path(d).is_dir()]:
            print(f"research retention gc: {missing[0]} is not a directory", file=sys.stderr)
            return 2
        jobs = [d for d in DEFAULT_JOBS_DIRS if d.is_dir()] + [Path(d) for d in ns.jobs_dir]
        try:
            plan = gc(ns.root, policy=ns.policy, high_pct=ns.high_pct, low_pct=ns.low_pct, force=ns.force, apply=ns.apply,
                      now=utc_now, usage=disk_usage, remote_check=store_check(ns.store, utc_now), in_use=JobScripts(jobs, ns.root),
                      inodes=inode_usage, max_paths=ns.max_paths or None, unpreserved=ns.unpreserved_runs)
        except ValueError as e:
            print(f"research retention gc: {e}", file=sys.stderr)
            return 2
        failed = [a for a in plan["actions"] if a["error"]]
        rc = 1 if failed or plan["stored"]["unreachable"] else 0
        if ns.json:
            print(json.dumps(plan, indent=2))
            return rc
        mode = "applying" if ns.apply else "dry run (--apply deletes)"
        head = f"research retention gc: policy {plan['policy']}, {mode}: {plan['root']} at {plan['pct_before']}%"
        if not plan["acting"]:
            print(f"{head}, not over {ns.high_pct:g}%: nothing to do (--force acts anyway)")
            return 0
        print(f"{head} (acts over {ns.high_pct:g}%{', forced' if ns.force else ''}; deletes until under {ns.low_pct:g}%)")
        def line(a: dict[str, Any]) -> str:
            verb = "DELETE" if a["action"] == "delete" and not a["error"] else "FAILED" if a["error"] else "skip"
            tail = f"  -> {a['pct_after']}%" if a["pct_after"] is not None else ""
            path = f"{a['path']}/* (unrecorded rest)" if a["rest"] else a["path"]
            return f"{verb:<7} {_size(a['bytes']):>10}  {path}  {a['reason']}{tail}" + (f"  ERROR {a['error']}" if a["error"] else "")

        rows, runs = _apart(plan["actions"])
        for a in rows:
            print(line(a))
        if runs:
            print(UNPRESERVED_HEAD)
            for a in runs:
                print(line(a))
        n = sum(1 for a in plan["actions"] if a["action"] == "delete" and not a["error"])
        print(f"{'deleted' if ns.apply else 'would delete'} {n} path(s), {_size(plan['freed_bytes'])}: {plan['pct_before']}% -> "
              f"{plan['pct_after']}%" + ("" if ns.apply else "; nothing was deleted") + (f"; {len(failed)} deletion(s) failed" if failed else "")
              + (f"; {plan['capped']} more path(s) left for the next pass (--max-paths {ns.max_paths})" if plan["capped"] else ""))
        if runs:
            u = plan["unpreserved"]
            took = (f"of which this pass {'deleted' if ns.apply else 'would delete'} {_counted(u['deleted'])}" if ns.unpreserved_runs
                    else "none of which goes without --unpreserved-runs")
            print(f"unpreserved runs, of the paths above: {_counted(u['expired'])} expired, {took}; {_counted(u['recorded'])} not yet expired"
                  + _unread(u["invalid"]))
        if ns.apply:
            count = lambda states: ", ".join(f"{plan['stored'][k]} {k}" for k in states)  # noqa: E731
            print(f"stored copies of the preserved paths it asked about: {count(COLLECTED)} (deleted); {count(STORED[len(COLLECTED):])} (kept)"
                  + ("; a copy that could not be asked about fails the pass" if plan["stored"]["unreachable"] else ""))
        for e in plan["errors"]:
            print(f"error: {e}", file=sys.stderr)
        return rc
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


__all__ = ["COLLECTED", "DEFAULT_JOBS_DIRS", "FIELDS", "JobScripts", "KIND", "MAX_PATHS", "POLICY", "PRIORITIES", "RUN_EXPIRY", "SCHEMA",
           "SIDECAR", "STORED", "UNPRESERVED", "UNPRESERVED_RUN_EXPIRY", "RecordError", "Stored", "beside", "check", "covers", "find", "gc",
           "implicit", "keep", "keep_main", "keeping", "load", "main", "problems", "record_at", "report", "scan", "sidecar", "store_check",
           "target", "unpreserved_runs", "write"]

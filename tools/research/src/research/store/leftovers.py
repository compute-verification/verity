"""What a node's work leaves behind that nothing else removes: the passes of `research data evict` beside evict.py's three.

Each pass is in `evict_src`'s idiom.  It fails closed: a condition it cannot establish removes nothing more and is the
report's `error`.  It starts no removal after its time limit (`budget_s`); the next pass goes on.  A dry run removes nothing
and says what it would.  The report counts what went (`deleted`, `deleted_inodes`, `deleted_bytes`, as removing it freed them:
a file with another link frees nothing) and what stayed by the kind its reason starts with (`kept_by`): `2:` in use (a live
process names it, a lock is held, a run has not ended), `recent:`, `run:` (a request's run dir exists), `retired:`,
`unknown:` (a record, a name or a layout the pass does not know), `budget:`, `error:`, `unestablished:`.  A path kept because
it is in use, unknown or failed is named with its reason, in the log and in `kept_first`.  What a live process names (cwd,
root, exe, open files, maps, command line, environment: `evict.live_texts`) is read when a pass first needs it and again once
`SCAN_S` old.

* **uv's editable shards and temporary build directories** (`evict_uv`).  uv keys an editable wheel by its source path
  (`<cache>/sdists-v<N>/editable/<digest>`), so every shipped source tree and every run clone leaves a shard per workspace
  member; a uv that was killed leaves its `<cache>/builds-v0/.tmp*`.  `uv cache prune` removes neither a shard (each holds a
  live revision) nor, without `--force`, anything while a uv runs (it waits for the cache's `.lock`, which every uv holds
  shared), and a prune by another uv version removes the buckets it does not know, so this pass does not call it.  A shard or
  build directory goes, oldest first, once nothing in it was written for `UV_KEEP_S` and no live process names it.  A shard
  goes under its own `.lock`, the exclusive flock uv takes before it reads or builds one (`CacheShard::lock`), taken without
  waiting, and in two steps: uv reads a cached wheel after it has released that lock, so the first step unlinks only the
  revision pointer (`revision.rev`), without which no uv finds the wheel and the next one to want it builds it again, under
  the lock, which is a write.  The first step stamps the shard with its time, and the second, once nothing has written the
  shard for `UV_KEEP_S` more, removes it, its `.lock` last.  A uv that waited on that `.lock` builds the shard again under
  it and leaves a shard without one: a pass makes it one, as uv does when it finds none, which counts as a write.  uv writes
  each file of a shard by rename, so two uv in one shard (the one that waited, and one that made a new `.lock`) cost a
  second build, not a broken wheel.  Only the buckets in `UV_SDISTS` are touched: their layout was read in uv's source, and
  a bucket of another version is reported, not guessed at.  The wheels the shards linked (`archive-v0`) stay: only
  `uv cache prune` tells which of them nothing links.
* **the store's lock files** (`evict_locks`): `<store>/locks/<name>.lock`, one empty file per key ever fetched
  (`fetchset.lock`).  One goes while this pass holds its flock, taken without waiting, once it is `LOCK_KEEP_S` old; the
  taker checks that the path still names the file it locked and takes the lock again if not, so no fetch runs beside
  another on an unlinked file.
* **request directories whose run directory is gone** (`evict_requests`): `<root>/requests/<run id>` holds the request, its
  inputs and its custody key, and nothing reads it once its run dir is deleted; without a run dir it also keeps the source
  tree it names (`evict.record_uses`).  A launch claims the run dir as its first act under `.launch.lock`, so a request goes
  under that lock, request.json first (a delivery that arrives later finds no request and claims nothing), once nothing in it
  was written for `REQUEST_KEEP_S`, or for as long as its launch may still deliver it (`remote.launch_limit`) when that is
  longer, with no run dir under the runs roots or the `runs_root` it names, and no live process naming its run.
* **the clones of runs that are over** (`evict_clones`): `<root>/scratch/<run id>`, which its runner removes and
  `clone.sweep` removes when the next clone is made there.  This pass removes the same clones (`clone.not_over`) on a node
  where none is, and keeps one whose run a live process still names.

A request or clone is a directory named like a run id (`ids.is_run_id`); any other entry stays, and is said.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Callable

from . import evict as EV
from . import ids
from .evict import Unestablished

UV_KEEP_S = 48 * 3600        # a shard or build directory written this recently may belong to a build under way or a tree in use
LOCK_KEEP_S = 3600           # a lock file made this recently belongs to fetches of one tree that are still taking it in turn
REQUEST_KEEP_S = 3600        # a request written this recently may be on its way to its launch
BUDGET_S = 4 * 60            # no removal of a pass starts after this long: with evict_src's, they stay inside the unit's `timeout 3000`
SCAN_S = 60                  # what live processes name is read again before a removal once the last reading is this old
LISTED = 20                  # how many paths a report names of those deleted, and of those kept with a reason to act on
LOUD = ("2", "error", "unknown", "unestablished")
UV_TAG = "CACHEDIR.TAG"
UV_SIGNATURE = "Signature: 8a477f597d28d172789f06886806bc55"
UV_LOCK = ".lock"
UV_POINTER = "revision.rev"
#: the source-distribution buckets whose shard layout this pass knows: `<bucket>/editable/<16 hex>/` holding `.lock`, the
#: revision pointer `revision.rev` and one directory per revision (uv 0.12 writes sdists-v9, uv 0.13 sdists-v10)
UV_SDISTS = ("sdists-v9", "sdists-v10")
UV_BUILDS = "builds-v0"
_SHARD = re.compile(r"[0-9a-f]{16}")
_BUILD = re.compile(r"\.tmp[A-Za-z0-9]{6}")
_UV_NAMED = re.compile(rf"(?:^|[/\0\s:=])(sdists-v[0-9]+/+editable/+[0-9a-f]{{16}}(?![0-9a-f])"
                       rf"|{UV_BUILDS}/+\.tmp[A-Za-z0-9]{{6}}(?![A-Za-z0-9]))")
"""A shard or build directory however a process spells its cache: one of that name in any cache keeps the one here."""
_RUN_NAMED = re.compile(rf"(?<![0-9A-Za-z])({ids.RUN_RE.pattern.strip('^$')})(?![0-9a-f])")
"""A run id anywhere in what a process names (its RESEARCH_RUN_ID, a path under its run dir, request or clone)."""


class _Held(Exception):
    """A lock this pass did not take; the message is why its path stays."""


class _Pass:
    """One pass's report, time limit and log lines."""

    def __init__(self, name: str, where: Path, dry_run: bool, budget_s: float, now: float | None, clock: Callable[[], float],
                 log: Callable[[str], None] | None):
        self.name, self.dry_run, self.budget_s, self.clock, self.t0 = name, dry_run, budget_s, clock, clock()
        self.t_wall = time.time() if now is None else now
        self.say = log or (lambda s: None)
        self.rep: dict[str, Any] = {"dir": str(where), "dry_run": dry_run, "found": 0, "deleted": 0, "deleted_inodes": 0,
                                    "deleted_bytes": 0, "kept": 0, "kept_by": {}, "deleted_first": [], "kept_first": [], "error": None}

    def now(self) -> float:
        return self.t_wall + self.clock() - self.t0

    def freed(self, inodes: int, nbytes: int) -> None:
        self.rep["deleted_inodes"] += inodes
        self.rep["deleted_bytes"] += nbytes

    def gone(self, path: Path, inodes: int, nbytes: int) -> None:
        self.rep["deleted"] += 1
        self.freed(inodes, nbytes)
        if len(self.rep["deleted_first"]) < LISTED:
            self.rep["deleted_first"].append(str(path))
        self.say(f"{self.name}: {'would delete' if self.dry_run else 'deleted'} {path} ({inodes} inodes, {nbytes / 1e6:.1f} MB)")

    def stay(self, path: Path | None, why: str, n: int = 1) -> None:
        kind = why.split(":", 1)[0]
        self.rep["kept"] += n
        self.rep["kept_by"][kind] = self.rep["kept_by"].get(kind, 0) + n
        if path is None:
            return
        if kind in LOUD and len(self.rep["kept_first"]) < LISTED:
            self.rep["kept_first"].append({"path": str(path), "why": why})
        if kind in LOUD or kind == "retired":
            self.say(f"{self.name}: kept {path}: {why}")

    def each(self, paths: list[Path], one: Callable[[Path], None]) -> str | None:
        """`one(path)` for each path until the time limit, after which the rest stay for the next pass.  A path whose lock is
        held, that changed under the pass or that failed stays with that reason; a condition that cannot be established stops
        the pass, and is returned."""
        self.rep["found"] += len(paths)
        for i, path in enumerate(paths):
            if self.clock() - self.t0 > self.budget_s:
                self.stay(None, f"budget: no removal starts after {self.budget_s / 60:g} min; the next pass goes on", len(paths) - i)
                break
            try:
                one(path)
            except _Held as e:
                self.stay(path, str(e))
            except FileNotFoundError:
                self.stay(path, "recent: it changed while it was read")
            except Unestablished as e:
                self.stay(None, f"unestablished: {e}", len(paths) - i)
                return str(e)
            except OSError as e:
                self.stay(path, f"error: {type(e).__name__}: {e}")
        return None

    def end(self, error: str | None = None) -> dict[str, Any]:
        r = self.rep
        r["error"] = error
        self.say(f"{self.name}: {'would delete' if self.dry_run else 'deleted'} {r['deleted']} of {r['found']} "
                 f"({r['deleted_bytes'] / 1e9:.1f} GB, {r['deleted_inodes']} inodes); kept {r['kept']} "
                 + " ".join(f"{c}:{n}" for c, n in sorted(r["kept_by"].items()))
                 + (f"; nothing more deleted: {error}" if error else ""))
        return r


class _Named:
    """What live processes name, by the key `pattern` captures: read at the first question and again once `SCAN_S` old."""

    def __init__(self, pattern: re.Pattern[str], what: str, under: Path, proc: Path, nginx_conf: Path, clock: Callable[[], float]):
        self.pattern, self.what, self.under, self.proc, self.nginx_conf, self.clock = pattern, what, under, proc, nginx_conf, clock
        self.at: float | None = None
        self.uses: dict[str, str] = {}

    def __call__(self, key: str) -> str | None:
        if self.at is None or self.clock() - self.at > SCAN_S:
            uses: dict[str, str] = {}
            for pid, _cwd, text in EV.live_texts(self.proc, self.under, self.nginx_conf):
                for m in self.pattern.finditer(text):
                    uses.setdefault(re.sub("/+", "/", m.group(1)), f"2: process {pid} ({EV.comm(self.proc, pid)}) names {self.what}")
            self.uses, self.at = uses, self.clock()
        return self.uses.get(key)


def _mtime(p: Path) -> float | None:
    """The mtime of `p` when it is a directory itself (not a link to one), else None."""
    try:
        st = EV._quiet(lambda: os.lstat(p))
    except OSError as e:
        raise Unestablished(f"2: cannot read {p}: {e}") from None
    return st.st_mtime if st is not None and stat.S_ISDIR(st.st_mode) else None


def _walk(path: Path) -> tuple[float, int, int]:
    """(newest mtime, inodes, bytes) of the tree at `path`: the last write anywhere in it, and what removing it frees."""
    st = os.lstat(path)
    newest, inodes, nbytes = st.st_mtime, 1, 0
    for dp, dns, fns in os.walk(path, onerror=EV._raise_oserror):
        for n in (*dns, *fns):
            s = os.lstat(os.path.join(dp, n))
            newest = max(newest, s.st_mtime)
            if stat.S_ISDIR(s.st_mode):
                inodes += 1
            elif s.st_nlink == 1:
                inodes, nbytes = inodes + 1, nbytes + s.st_size
    return newest, inodes, nbytes


def _unlink(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Unlink the file `path`; the (inodes, bytes) that frees: nothing while another link holds it."""
    st = os.lstat(path)
    if not dry_run:
        os.unlink(path)
    return (1, st.st_size) if st.st_nlink == 1 else (0, 0)


def _take(path: Path, create: bool = False) -> int:
    """A descriptor holding an exclusive flock on the regular file `path`, taken without waiting and only while `path` still
    names that file; `_Held` says why not.  Opened read-only (a flock needs no more) and without following a link."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC | (os.O_CREAT if create else 0), 0o666)
    try:
        held = os.fstat(fd)
        if not stat.S_ISREG(held.st_mode):
            raise _Held(f"unknown: its {path.name} is not a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise _Held(f"2: its lock is held ({path.name})") from None
        at = EV._quiet(lambda: os.lstat(path))
        if at is None or (at.st_dev, at.st_ino) != (held.st_dev, held.st_ino):
            raise _Held(f"2: its lock file was replaced while this pass took it ({path.name})")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _age(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


# ---- uv's editable shards and temporary build directories ------------------------------------------------------------------------------

def evict_uv(cache: Path, *, dry_run: bool = False, keep_s: float = UV_KEEP_S, budget_s: float = BUDGET_S,
             now: float | None = None, clock: Callable[[], float] = time.monotonic, proc: Path = Path("/proc"),
             log: Callable[[str], None] | None = None, nginx_conf: Path = EV.NGINX_CONF) -> dict[str, Any]:
    """Remove the editable shards and temporary build directories of the uv cache at `cache` that nothing wrote for `keep_s` and
    no live process names (module docstring), oldest first: a build directory outright, a shard under its `.lock` in two steps."""
    cache = Path(os.path.realpath(cache))
    p = _Pass("uv", cache, dry_run, budget_s, now, clock, log)
    named = _Named(_UV_NAMED, "it", cache, proc, nginx_conf, clock)
    found: list[tuple[float, Path]] = []
    other: list[tuple[Path, int]] = []
    try:
        for b in EV._entries(cache):
            if b == UV_BUILDS and _mtime(cache / b) is not None:
                found += [(t, cache / b / n) for n in EV._entries(cache / b)
                          if _BUILD.fullmatch(n) and (t := _mtime(cache / b / n)) is not None]
            ed = cache / b / "editable"
            if not re.fullmatch(r"sdists-v[0-9]+", b) or _mtime(cache / b) is None or _mtime(ed) is None:
                continue
            shards = [(t, ed / n) for n in EV._entries(ed) if _SHARD.fullmatch(n) and (t := _mtime(ed / n)) is not None]
            if b in UV_SDISTS:
                found += shards
            elif shards:
                other.append((ed, len(shards)))
        if (found or other) and not (EV._text(cache / UV_TAG) or "").startswith(UV_SIGNATURE):
            raise Unestablished(f"{cache} has no {UV_TAG} of a cache directory, so it is not taken for a uv cache")
    except Unestablished as e:
        return p.end(str(e))

    def idle(path: Path, newest: float) -> bool:
        if (age := p.now() - newest) < keep_s:
            p.stay(path, f"recent: written {_age(age)} ago (within {keep_s / 3600:g} h)")
        return age >= keep_s

    def one(path: Path) -> None:
        if not idle(path, os.lstat(path).st_mtime):
            return
        if why := named(str(path.relative_to(cache))):
            return p.stay(path, why)
        if path.parent.name == UV_BUILDS:
            newest, *size = _walk(path)
            if idle(path, newest):
                p.gone(path, *(size if dry_run else EV._remove_tree(path)))
            return
        try:
            lock = _take(path / UV_LOCK)
        except FileNotFoundError:
            if not dry_run:
                os.close(_take(path / UV_LOCK, create=True))
                os.utime(path, (p.now(), p.now()))
            return p.stay(path, f"recent: it had no {UV_LOCK}, which {'would be' if dry_run else 'is'} made as uv makes one; "
                                f"it goes once nothing has written it for {keep_s / 3600:g} h more")
        try:
            newest, *size = _walk(path)
            if not idle(path, newest):
                return
            modes = {n: os.lstat(path / n).st_mode for n in os.listdir(path) if n != UV_LOCK}
            if odd := sorted(n for n, m in modes.items() if not (stat.S_ISREG(m) if n == UV_POINTER else stat.S_ISDIR(m))):
                return p.stay(path, f"unknown: its {odd[0]} is neither the revision pointer nor a revision's directory")
            if UV_POINTER in modes and len(modes) > 1:
                p.freed(*_unlink(path / UV_POINTER, dry_run))
                if not dry_run:
                    os.utime(path, (p.now(), p.now()))
                return p.stay(path, f"retired: its revision pointer {'would be' if dry_run else 'is'} removed, so no uv reads its "
                                    f"wheel; it goes once nothing has written it for {keep_s / 3600:g} h more")
            if not dry_run:
                freed = [EV._remove_tree(path / n) if stat.S_ISDIR(m) else _unlink(path / n) for n, m in modes.items()]
                freed.append(_unlink(path / UV_LOCK))
                os.rmdir(path)
                size = [sum(f[0] for f in freed) + 1, sum(f[1] for f in freed)]
            p.gone(path, *size)
        finally:
            os.close(lock)

    error = p.each([path for _, path in sorted(found)], one)
    for ed, n in other:
        p.rep["found"] += n
        p.stay(ed, f"unknown: {ed.parent.name} is not a bucket whose layout this pass knows ({', '.join(UV_SDISTS)}): its {n} "
                   f"shards stay", n)
    return p.end(error)


# ---- the store's lock files ------------------------------------------------------------------------------------------------------------

def evict_locks(store_root: Path, *, dry_run: bool = False, keep_s: float = LOCK_KEEP_S, budget_s: float = BUDGET_S,
                now: float | None = None, clock: Callable[[], float] = time.monotonic,
                log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Unlink the lock files under `<store_root>/locks` (`fetchset.lock`'s: `<name>.lock`, empty) made more than `keep_s` ago, each
    while this pass holds its flock: one that is held stays.  Only a file named as the store names its locks
    (`fetchset.LOCK_NAME`) goes: another program's lock file, whose takers may not check what they hold, stays."""
    from . import fetchset

    d = Path(store_root) / "locks"
    p = _Pass("locks", d, dry_run, budget_s, now, clock, log)
    try:
        paths = [d / n for n in EV._entries(d) if n.endswith(".lock")]
    except Unestablished as e:
        return p.end(str(e))

    def one(path: Path) -> None:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or not fetchset.LOCK_NAME.fullmatch(path.name[:-len(".lock")]):
            return p.stay(path, "unknown: it is not a regular file named like one of the store's lock files")
        if (age := p.now() - st.st_mtime) < keep_s:
            return p.stay(path, f"recent: made {_age(age)} ago (within {keep_s / 3600:g} h)")
        lock = _take(path)
        try:
            if os.fstat(lock).st_size:
                return p.stay(path, "unknown: it is not empty, and the store's lock files hold no data")
            if not dry_run:
                os.unlink(path)
            p.gone(path, 1, 0)
        finally:
            os.close(lock)

    return p.end(p.each(paths, one))


# ---- request directories whose run directory is gone -----------------------------------------------------------------------------------

def _run_entries(d: Path) -> tuple[list[Path], list[Path]]:
    """The directories under `d` named like a run id, and the other directories there, which stay."""
    dirs = [d / n for n in EV._entries(d) if _mtime(d / n) is not None]
    return [x for x in dirs if ids.is_run_id(x.name)], [x for x in dirs if not ids.is_run_id(x.name)]


def _run_dir(homes: list[Path], name: str) -> Path | None:
    """The run dir `name` under one of `homes`, None when none has it; Unestablished when one cannot be read."""
    for home in homes:
        try:
            there = EV._quiet(lambda home=home: os.lstat(home / name))
        except OSError as e:
            raise Unestablished(f"2: cannot read {home / name}: {e}") from None
        if there is not None:
            return home / name
    return None


def evict_requests(requests_dir: Path, *, runs_dirs: list[Path] | None = None, dry_run: bool = False,
                   keep_s: float = REQUEST_KEEP_S, budget_s: float = BUDGET_S, now: float | None = None,
                   clock: Callable[[], float] = time.monotonic, proc: Path = Path("/proc"),
                   log: Callable[[str], None] | None = None, nginx_conf: Path = EV.NGINX_CONF) -> dict[str, Any]:
    """Remove each `<requests_dir>/<run id>` whose run dir is gone (module docstring).  `runs_dirs`: where run dirs are, by default
    `runs/` and `runs-aside/` beside `requests_dir`.  The directory a request's run dir would be in (the `runs_root` it names,
    else the first of `runs_dirs`) must be there: a run dir missing from a directory that is itself missing is not gone."""
    from .. import remote as RM

    requests_dir = Path(requests_dir)
    root = requests_dir.parent
    runs_dirs = [root / "runs", root / "runs-aside"] if runs_dirs is None else [Path(d) for d in runs_dirs]
    p = _Pass("requests", requests_dir, dry_run, budget_s, now, clock, log)
    named = _Named(_RUN_NAMED, "its run", requests_dir, proc, nginx_conf, clock)
    try:
        paths, others = _run_entries(requests_dir)
    except Unestablished as e:
        return p.end(str(e))

    def one(path: Path) -> None:
        req = EV._doc(EV._text(path / "request.json")) or {}
        own = Path(req["runs_root"]) if isinstance(req.get("runs_root"), str) and os.path.isabs(req["runs_root"]) else None
        if _mtime(own or runs_dirs[0]) is None:
            return p.stay(path, f"unknown: {own or runs_dirs[0]}, where its run dir would be, is not a directory")
        homes = [*runs_dirs, *([own] if own else [])]
        if run := _run_dir(homes, path.name):
            return p.stay(path, f"run: its run dir {run} exists")
        wait = req.get("mem_wait_s")
        keep = max(keep_s, RM.launch_limit(wait if isinstance(wait, (int, float)) and not isinstance(wait, bool) else None))
        if (age := p.now() - _walk(path)[0]) < keep:
            return p.stay(path, f"recent: written {_age(age)} ago (a launch may deliver it for {_age(keep)})")
        if why := named(path.name):
            return p.stay(path, why)
        made = not dry_run or os.path.lexists(path / RM.LAUNCH_LOCK)      # a dry run makes no lock file
        lock = _take(path / RM.LAUNCH_LOCK, create=not dry_run) if made else None
        try:
            if run := _run_dir(homes, path.name):
                return p.stay(path, f"run: its run dir {run} exists")
            size = _walk(path)[1:]
            if not dry_run:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(path / "request.json")
                EV._remove_tree(path)
            p.gone(path, *size)
        finally:
            if lock is not None:
                os.close(lock)

    error = p.each(paths, one)
    p.rep["found"] += len(others)
    for o in others:
        p.stay(o, "unknown: it is not named like a run id")
    return p.end(error)


# ---- the clones of runs that are over --------------------------------------------------------------------------------------------------

def evict_clones(scratch: Path, *, runs_root: Path | None = None, dry_run: bool = False, budget_s: float = BUDGET_S,
                 clock: Callable[[], float] = time.monotonic, proc: Path = Path("/proc"),
                 log: Callable[[str], None] | None = None, nginx_conf: Path = EV.NGINX_CONF) -> dict[str, Any]:
    """Remove each `<scratch>/<run id>` whose run, under `runs_root` (`runs/` beside `scratch`), is over (`clone.not_over`) and
    which no live process names, with `clone.remove`."""
    from .. import clone as CL

    scratch = Path(scratch)
    runs_root = scratch.parent / "runs" if runs_root is None else Path(runs_root)
    p = _Pass("clones", scratch, dry_run, budget_s, 0.0, clock, log)
    named = _Named(_RUN_NAMED, "its run", scratch, proc, nginx_conf, clock)
    try:
        paths, others = _run_entries(scratch)
    except Unestablished as e:
        return p.end(str(e))

    def one(path: Path) -> None:
        if why := CL.not_over(runs_root / path.name, proc) or named(path.name):
            return p.stay(path, why)
        size = _walk(path)[1:]
        if not dry_run and (why := CL.remove(path)):
            return p.stay(path, f"error: {why}")
        p.gone(path, *size)

    error = p.each(paths, one)
    p.rep["found"] += len(others)
    for o in others:
        p.stay(o, "unknown: it is not named like a run id")
    return p.end(error)

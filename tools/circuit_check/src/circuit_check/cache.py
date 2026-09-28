"""The result cache: a report is reused only when everything the run that produced it read is unchanged.

A cached run records, under the repository root: every file it opened for reading and every directory it listed (an audit
hook on ``open``, ``os.listdir`` and ``os.scandir``, in forked workers too), every path it only probed (``os.stat``: whether it
was a file, a directory or absent), the source file of every module it imported (Python may read a module's bytecode
without opening its source), and every library built in the tree that it loaded (``ctypes.dlopen``: by its bytes).  What it
cannot see (a program it starts: ``subprocess``, ``os.system``, ``exec``, ``posix_spawn``) makes the process's later results
uncacheable (`untraced`), so none is stored on reads the tracer missed.  Beside them the key holds the checker's own version (the
digest of this package's sources), the interpreter, the platform, the installed distributions, the request, and the
``VERITY_*`` / ``CIRCUIT_CHECK_*`` environment.  A later run with the same key reuses the report only if every recorded file
has the same sha256 and every recorded directory the same listing, so a hit means identical inputs.  The tree is git's
(`visible`): what git does not have (bytecode, a build inside the tree, as a concurrent step makes) counts as absent, being
derived from what it has.  Not seen: files a C extension reads with its own I/O (none here), and anything outside the
repository, which the interpreter, platform and distribution list stand for.

Entries are files ``<dir>/<version>/<sha256 of the entry>.json``; there is no expiry and no path or time shortcut.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

SCHEMA = "circuit-check-cache/v1"
_READS: dict[str, set[str]] = {"files": set(), "dirs": set(), "probes": set(), "libs": set()}
_ROOT: list[Path] = []
#: what the tracer cannot see: a program started from here, or a repository library loaded by the dynamic loader
UNTRACEABLE = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn")
_UNTRACED: list[str] = []
_INTERNAL: list[bool] = []


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_VISIBLE: dict[Path, tuple[frozenset[str], frozenset[str]] | None] = {}


def visible(root: Path) -> tuple[frozenset[str], frozenset[str]] | None:
    """(files, directories) of the tree as git has it: tracked and untracked-unignored in a checkout; in a shipped tree (no
    `.git`), the commit's tree from the machine's bare repository (`READY.json` names it).  None when neither is available."""
    root = Path(root).resolve()
    if root not in _VISIBLE:
        import subprocess

        if (root / ".git").exists():
            argv = ["git", "-C", str(root), "ls-files", "-z", "-co", "--exclude-standard"]
        elif (root / "READY.json").exists() and (root.parent.parent / "git" / "verity.git").exists():
            tree = json.loads((root / "READY.json").read_text())["tree"]
            argv = ["git", "--git-dir", str(root.parent.parent / "git" / "verity.git"), "ls-tree", "-r", "-z", "--name-only", tree]
        else:
            argv = None
        _INTERNAL.append(True)
        try:
            r = subprocess.run(argv, capture_output=True) if argv else None
        finally:
            _INTERNAL.pop()
        if r is None or r.returncode:
            _VISIBLE[root] = None
        else:
            files = frozenset(x.decode() for x in r.stdout.split(b"\0") if x)
            dirs = frozenset({"."} | {str(Path(f).parents[i]) for f in files for i in range(len(Path(f).parents) - 1)})
            _VISIBLE[root] = (files, dirs)
    return _VISIBLE[root]


def _listing(root: Path, rel: str) -> str:
    names = sorted(os.listdir(root / rel))
    if (v := visible(root)) is not None:                # what a build or a run adds inside the tree is not an input
        names = [n for n in names if os.path.normpath(os.path.join(rel, n)) in v[0] | v[1]]
    return hashlib.sha256("\0".join(names).encode()).hexdigest()


def _rel(path: Any) -> str | None:
    if isinstance(path, int) or not _ROOT:
        return None
    try:
        p = Path(os.fsdecode(path))
        p = p if p.is_absolute() else Path.cwd() / p
        rel = os.path.relpath(os.path.normpath(p), _ROOT[0])
    except (TypeError, ValueError):
        return None
    if rel.startswith("..") or "__pycache__" in rel.split(os.sep) or rel.split(os.sep)[0] == ".venv":
        return None
    return rel


def _hook(event: str, args: tuple) -> None:
    if event == "open":
        path, mode = args[0], args[1] if len(args) > 1 else "r"
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            return
        if isinstance(mode, str) or mode is None:
            if (rel := _rel(path)) is not None:
                _READS["files"].add(rel)
    elif event in ("os.listdir", "os.scandir"):
        if (rel := _rel(args[0] if args and args[0] is not None else ".")) is not None:
            _READS["dirs"].add(rel)
    elif event in UNTRACEABLE and not _INTERNAL:
        _UNTRACED.append(f"{event} {str(args[0] if args else '')[:120]}")
    elif event == "ctypes.dlopen" and args and args[0] and (rel := _rel(args[0])) is not None:
        _READS["libs"].add(rel)                        # a library built in the tree: an input by its bytes (`_digests`)


def _probing(fn):
    def stat(path, *a, **k):
        if (rel := _rel(path)) is not None:
            _READS["probes"].add(rel)
        return fn(path, *a, **k)
    return stat


def trace(root: Path) -> None:
    """Record what this process (and every process forked from it) reads under ``root`` from now on: opens and listings through
    the audit hook, existence probes (`Path.exists` and the like) through `os.stat` / `os.lstat`, which raise no audit event."""
    if not _ROOT:
        _ROOT.append(Path(root).resolve())
        sys.addaudithook(_hook)
        os.stat, os.lstat = _probing(os.stat), _probing(os.lstat)


def tracing() -> bool:
    return bool(_ROOT)


def untraced() -> list[str]:
    """What this process did that the tracer cannot see (a verdict after it is not stored)."""
    return list(_UNTRACED)


def reads() -> dict[str, list[str]]:
    """The files and directories read so far, the imported modules' sources included."""
    files = set(_READS["files"])
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and (rel := _rel(f)) is not None:
            files.add(rel)
    return {"files": sorted(files), "dirs": sorted(_READS["dirs"]), "probes": sorted(_READS["probes"] - files - _READS["dirs"]),
            "libs": sorted(_READS["libs"])}


def version() -> str:
    """The checker's own version: the digest of this package's sources."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(here.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(p.relative_to(here).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def key(request: dict) -> dict:
    from importlib import metadata

    _INTERNAL.append(True)                             # the tracer's own work (`platform` runs `uname`): no input of a check
    try:
        plat = platform.platform()
    finally:
        _INTERNAL.pop()
    dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    return {"schema": SCHEMA, "version": version(), "request": request, "python": sys.version, "platform": plat,
            "distributions": hashlib.sha256("\n".join(dists).encode()).hexdigest(),
            "environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("VERITY_", "CIRCUIT_CHECK_"))}}


def _kind(p: Path) -> str | None:
    return "dir" if p.is_dir() else "file" if p.exists() else None


def _digests(root: Path, got: dict[str, list[str]]) -> dict[str, dict[str, str | None]]:
    """Each recorded read as it is now.  With git's view of the tree (`visible`), a file or directory git does not have (bytecode,
    a build inside the tree) counts as absent: derived from what git has, it is not an input of its own."""
    v = visible(root)
    seen = (lambda r: True) if v is None else (lambda r: r in v[0] or r in v[1])

    def memo(kind: str, r: str, fn):
        k = (str(root), kind, r)                    # a run changes none of its inputs: each is read once per process
        if k not in _MEMO:
            _MEMO[k] = fn()
        return _MEMO[k]

    files = {r: memo("f", r, lambda r=r: _sha_file(root / r) if seen(r) and (root / r).is_file() else None) for r in got["files"]}
    dirs = {r: memo("d", r, lambda r=r: _listing(root, r) if seen(r) and (root / r).is_dir() else None) for r in got["dirs"]}
    probes = {r: memo("p", r, lambda r=r: _kind(root / r) if seen(r) else None) for r in got.get("probes", [])}
    # a library the dynamic loader mapped is git-ignored (built from its source here), so it counts by its bytes
    libs = {r: memo("l", r, lambda r=r: _sha_file(root / r) if (root / r).is_file() else None) for r in got.get("libs", [])}
    return {"files": files, "dirs": dirs, "probes": probes, "libs": libs}


_MEMO: dict[tuple[str, str, str], str | None] = {}


def _dir(cache: Path, k: dict) -> Path:
    return Path(cache) / k["version"] / hashlib.sha256(json.dumps(k, sort_keys=True, default=str).encode()).hexdigest()[:32]


def lookup(cache: Path, k: dict, root: Path) -> tuple[dict, str] | None:
    """(report, entry id) of an entry with key ``k`` whose every recorded read is unchanged under ``root``, else None."""
    d = _dir(cache, k)
    if not d.is_dir():
        return None
    for p in sorted(d.glob("*.json")):
        try:
            e = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if e.get("key") != k:
            continue
        rec = e["reads"]
        now = _digests(root, {k: list(v) for k, v in rec.items()})
        if now == rec:
            return e["report"], p.stem
    return None


def store(cache: Path, k: dict, root: Path, got: dict[str, list[str]], report: dict) -> str:
    e = {"key": k, "reads": _digests(root, got), "report": report}
    body = json.dumps(e, sort_keys=True, separators=(",", ":"), default=str)
    eid = hashlib.sha256(body.encode()).hexdigest()
    d = _dir(cache, k)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{eid}.tmp"
    tmp.write_text(body)
    os.replace(tmp, d / f"{eid}.json")
    return eid


__all__ = ["SCHEMA", "UNTRACEABLE", "key", "lookup", "reads", "store", "trace", "tracing", "untraced", "version"]

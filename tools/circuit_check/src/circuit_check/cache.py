"""The result cache: a report is reused only when everything the run that produced it read is unchanged.

A cached run records, under the repository root: every file it opened for reading and every directory it listed (an audit
hook on ``open``, ``os.listdir`` and ``os.scandir``, in forked workers too), every path it only probed (``os.stat``: whether it
was a file, a directory or absent), and the source file of every module it imported (Python may read a module's bytecode
without opening its source).  Beside them the key holds the checker's own version (the
digest of this package's sources), the interpreter, the platform, the installed distributions, the request, and the
``VERITY_*`` / ``CIRCUIT_CHECK_*`` environment.  A later run with the same key reuses the report only if every recorded file
has the same sha256 and every recorded directory the same listing, so a hit means identical inputs.  Not seen: files a C
extension reads with its own I/O (none here), and anything outside the repository, which the interpreter, platform and
distribution list stand for.

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
_READS: dict[str, set[str]] = {"files": set(), "dirs": set(), "probes": set()}
_ROOT: list[Path] = []


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _listing(p: Path) -> str:
    return hashlib.sha256("\0".join(sorted(os.listdir(p))).encode()).hexdigest()


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


def reads() -> dict[str, list[str]]:
    """The files and directories read so far, the imported modules' sources included."""
    files = set(_READS["files"])
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and (rel := _rel(f)) is not None:
            files.add(rel)
    return {"files": sorted(files), "dirs": sorted(_READS["dirs"]), "probes": sorted(_READS["probes"] - files - _READS["dirs"])}


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

    dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    return {"schema": SCHEMA, "version": version(), "request": request, "python": sys.version, "platform": platform.platform(),
            "distributions": hashlib.sha256("\n".join(dists).encode()).hexdigest(),
            "environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("VERITY_", "CIRCUIT_CHECK_"))}}


def _kind(p: Path) -> str | None:
    return "dir" if p.is_dir() else "file" if p.exists() else None


def _digests(root: Path, got: dict[str, list[str]]) -> dict[str, dict[str, str | None]]:
    files = {r: (_sha_file(root / r) if (root / r).is_file() else None) for r in got["files"]}
    dirs = {r: (_listing(root / r) if (root / r).is_dir() else None) for r in got["dirs"]}
    return {"files": files, "dirs": dirs, "probes": {r: _kind(root / r) for r in got.get("probes", [])}}


def lookup(cache: Path, k: dict, root: Path) -> tuple[dict, str] | None:
    """(report, entry id) of an entry with key ``k`` whose every recorded read is unchanged under ``root``, else None."""
    d = Path(cache) / k["version"]
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
    d = Path(cache) / k["version"]
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{eid}.tmp"
    tmp.write_text(body)
    os.replace(tmp, d / f"{eid}.json")
    return eid


__all__ = ["SCHEMA", "key", "lookup", "reads", "store", "trace", "version"]

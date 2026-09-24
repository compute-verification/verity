"""Which commit a pod's installed binary came from.

    python -m research.binprov record BINARY [--source DIR]   # at build time, right after `install`: writes BINARY.provenance.json
    python -m research.binprov show [DIR ...]                  # what a run would record (default $RESEARCH_BIN_DIRS or /workspace/bin)

Binaries sit at fixed paths (`/workspace/bin/veritor-zk-host-cuda`, `/workspace/bin/ligero-verify`) whatever commit built them,
so a run's source commit says nothing about the binary it called.  `record` writes, beside the binary, its sha256 and size and the
source it was built from: the shipped tree's `READY.json` (commit, tree) or a `pods sync` tree's `.research-source.json` (commit,
tree, dirty) found at or above `--source`, else `git rev-parse HEAD` there (and whether the worktree was dirty), else unknown.  `snapshot` -- run by `research run` on a machine into job.json
`run.binaries` -- hashes every regular file in the bin dirs and reports its provenance `recorded` (the file's sha256 is the one
recorded), `stale` (the binary changed after it was recorded: another build installed over it without recording), or `missing`.
Stdlib only, so a bootstrap can call it with `PYTHONPATH=<tree>/tools/research/src python3 -m research.binprov`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "research/binary-provenance/v1"
SUFFIX = ".provenance.json"
BIN_DIRS_ENV = "RESEARCH_BIN_DIRS"
DEFAULT_BIN_DIRS = ("/workspace/bin",)
READY_FILE = "READY.json"                     # a `research run --on --source` tree (remote.py)
SYNC_STAMP = ".research-source.json"          # a `research pods sync` tree (pods/connect.py; dirty trees allowed)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_identity(source: Path | None) -> dict[str, Any]:
    """{commit, tree, dirty, dir, via}: the shipped tree's READY.json at or above `source`, else git there, else via None."""
    if source is None:
        return {"commit": None, "tree": None, "dirty": None, "dir": None, "via": None}
    d = Path(source).resolve()
    for p in (d, *d.parents):
        for name in (READY_FILE, SYNC_STAMP):
            if (p / name).is_file():
                try:
                    doc = json.loads((p / name).read_text())
                except (OSError, ValueError):
                    continue
                return {"commit": doc.get("commit") or doc.get("identity"), "tree": doc.get("tree"),
                        "dirty": False if name == READY_FILE else doc.get("dirty"), "dir": str(p), "via": name}
    def git(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", "-C", str(d), *args], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    if commit:
        status = git("status", "--porcelain", "--untracked-files=no")
        return {"commit": commit, "tree": git("rev-parse", "HEAD^{tree}"), "dirty": bool(status) if status is not None else None,
                "dir": git("rev-parse", "--show-toplevel") or str(d), "via": "git"}
    return {"commit": None, "tree": None, "dirty": None, "dir": str(d), "via": None}


def record(binary: Path, source: Path | None = None) -> dict[str, Any]:
    """Write `<binary>.provenance.json` (atomically) and return it."""
    binary = Path(binary)
    doc = {"schema": SCHEMA, "path": str(binary.resolve()), "sha256": sha256_file(binary), "bytes": binary.stat().st_size,
           "source": source_identity(source), "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "host": socket.gethostname()}
    out = binary.with_name(binary.name + SUFFIX)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out)
    return doc


def bin_dirs() -> list[Path]:
    env = os.environ.get(BIN_DIRS_ENV)
    return [Path(x) for x in (env.split(":") if env else DEFAULT_BIN_DIRS) if x]


def snapshot(dirs: list[Path] | None = None) -> list[dict[str, Any]]:
    """One entry per regular file in the bin dirs: path, sha256, bytes, and `provenance` (`recorded` | `stale` | `missing`) with
    the recorded source when there is one."""
    out = []
    for d in bin_dirs() if dirs is None else dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.name.endswith(SUFFIX) or p.name.startswith("."):
                continue
            try:
                row: dict[str, Any] = {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
            except OSError as e:
                out.append({"path": str(p), "error": f"{type(e).__name__}: {e}"})
                continue
            prov = p.with_name(p.name + SUFFIX)
            try:
                doc = json.loads(prov.read_text())
            except (OSError, ValueError):
                doc = None
            if doc is None:
                row["provenance"] = "missing"
            else:
                row["provenance"] = "recorded" if doc.get("sha256") == row["sha256"] else "stale"
                row["source"] = doc.get("source")
                row["built_utc"] = doc.get("built_utc")
                if row["provenance"] == "stale":
                    row["recorded_sha256"] = doc.get("sha256")
            out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m research.binprov", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rc = sub.add_parser("record", help="write BINARY.provenance.json: sha256 + the source commit it was built from")
    rc.add_argument("binary", type=Path)
    rc.add_argument("--source", type=Path, default=None, help="the tree it was built from (default: the current directory)")
    sh = sub.add_parser("show", help="what `research run` records in job.json run.binaries")
    sh.add_argument("dirs", nargs="*", type=Path)
    ns = ap.parse_args(argv)
    if ns.cmd == "record":
        doc = record(ns.binary, ns.source or Path.cwd())
        src = doc["source"]
        print(f"provenance {ns.binary}: sha256 {doc['sha256'][:16]} from {src['commit'] or 'unknown source'}"
              + (" (dirty)" if src.get("dirty") else "") + (f" via {src['via']}" if src.get("via") else ""))
        return 0
    print(json.dumps(snapshot(ns.dirs or None), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

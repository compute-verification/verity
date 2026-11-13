#!/usr/bin/env python3
"""A Lake package with the packages of the tree it requires by path, for a job that builds it outside the checkout.

    python3 verity/protocols/flock/pod/lake_closure.py dirs PACKAGE           the package's directory, then each package its manifest gives
                                                                by path (the inherited ones too), one a line
    python3 verity/protocols/flock/pod/lake_closure.py key PACKAGE [PATH...]  16 hex digits over every file of those directories and of
                                                                PATHs (files or directories): the name of a cached build
    python3 verity/protocols/flock/pod/lake_closure.py copy PACKAGE DEST      copies those directories into DEST, each at its place
                                                                relative to the others, and prints PACKAGE's directory there

A copy of the package's directory alone does not build once it requires another by path, and a key over it alone misses
a change to what it requires. No `.lake` is read or copied. Stdlib only: pods run it before any environment exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

SKIP = (".lake", "__pycache__")


def closure(package: Path) -> list[Path]:
    """``package``, then the packages its manifest gives by path, sorted."""
    package = package.resolve()
    manifest = package / "lake-manifest.json"
    entries = json.loads(manifest.read_text()).get("packages", []) if manifest.exists() else []
    return [package, *sorted({(package / e["dir"]).resolve() for e in entries if e.get("type") == "path"} - {package})]


def files(path: Path) -> list[Path]:
    """Every file under ``path`` (itself when it is a file), build trees aside, sorted."""
    if path.is_file():
        return [path]
    out = []
    for root, dirs, names in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        out += [Path(root) / n for n in sorted(names)]
    return out


def top(dirs: list[Path]) -> Path:
    """The directory every one of ``dirs`` is placed under: their common parent."""
    return Path(os.path.commonpath([str(d) for d in dirs])) if len(dirs) > 1 else dirs[0].parent


def key(package: Path, extra: list[Path]) -> str:
    """A digest of each file's place under `top` and its bytes, over the closure and ``extra``."""
    dirs = closure(package)
    base = top(dirs)
    h = hashlib.sha256()
    for f in [f for d in dirs for f in files(d)]:
        h.update(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(base).as_posix()}\n".encode())
    for e in extra:
        for f in files(e.resolve()):
            h.update(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  +{f.relative_to(e.resolve().parent).as_posix()}\n".encode())
    return h.hexdigest()[:16]


def copy(package: Path, dest: Path) -> Path:
    """The closure under ``dest``; returns ``package``'s directory there."""
    dirs = closure(package)
    base = top(dirs)
    for d in dirs:
        shutil.copytree(d, dest / d.relative_to(base), ignore=shutil.ignore_patterns(*SKIP), dirs_exist_ok=True)
    return dest / dirs[0].relative_to(base)


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "dirs" and len(argv) == 2:
        print("\n".join(os.path.relpath(d) for d in closure(Path(argv[1]))))
    elif len(argv) >= 2 and argv[0] == "key":
        print(key(Path(argv[1]), [Path(a) for a in argv[2:]]))
    elif len(argv) == 3 and argv[0] == "copy":
        print(copy(Path(argv[1]), Path(argv[2])))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

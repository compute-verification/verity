#!/usr/bin/env python3
"""Where a path of an earlier layout is now, and where a path was, by the layout moves this tree records.

`tools/research/src/research/store/moves.json` lists each layout move's renamed paths, oldest first, and a move's generated
commit appends its own entry. So it says which moves have happened in this tree, where a map under `tools/move/` is on
`main` before its move is, and it holds every move, where a map holds one. Standard library only, and syntax Python 3.10
parses: `setup.sh` runs it with the machine's own `python3`.

    python3 tools/verity/lean/moved.py PATH       print the directory PATH names in this tree, from the repository's root:
                                                  where the recorded moves took it (`verity/Security` to `Security` to
                                                  `security`) when that is one, else PATH itself; exit 1 when neither is
    python3 tools/verity/lean/moved.py --carry    in this checkout, move each Lake package's build directory (`.lake`,
                                                  untracked, so a move leaves it at the package's former path) to where
                                                  the package is now; a package that has one already keeps it

A slot's tree gets the same from `lean_changed.py`, under the tree's lock (`carry`).
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
#: research's record of the layout moves: `{"moves": [{"paths": {old: new}}, ...]}`, a directory's entry ending in `/`
MOVES = "tools/research/src/research/store/moves.json"
LAKEFILES = ("lakefile.toml", "lakefile.lean")


def recorded(root: Path = ROOT) -> list[dict[str, str]]:
    """Each recorded move's `{old: new}`, oldest first; none where the tree has no record (a tree from before the first
    move). A record that doesn't read is an error: it is a tracked file, and guessing "no moves" would build cold."""
    p = root / MOVES
    if not p.is_file():
        return []
    doc = json.loads(p.read_text())
    moves = doc.get("moves") if isinstance(doc, dict) else None
    return [m["paths"] for m in moves or [] if isinstance(m, dict) and isinstance(m.get("paths"), dict)]


def through(path: str, paths: dict[str, str], back: bool = False) -> str | None:
    """``path`` through one move (``back``: where it was before it): the entry that names it, or the longest directory
    entry that holds it; None when the move doesn't know it."""
    best = None
    for old, new in paths.items():
        a, b = (new, old) if back else (old, new)
        if a.endswith("/") and b.endswith("/"):
            if (path + "/").startswith(a) and (best is None or len(a) > len(best[0])):
                best = (a, b)
        elif path == a and not b.endswith("/"):
            return b
    return None if best is None else (best[1] + (path + "/")[len(best[0]):]).rstrip("/")


def now(path: str, moves: list[dict[str, str]]) -> str:
    """Where ``path``, a path of some earlier layout, is after every recorded move."""
    for paths in moves:
        path = through(path, paths) or path
    return path


def former(path: str, moves: list[dict[str, str]]) -> list[str]:
    """Each earlier path of ``path``, the latest first."""
    out = []
    for paths in reversed(moves):
        was = through(path, paths, back=True)
        if was is not None and was != path:
            path = was
            out.append(path)
    return out


def resolve(path: str, root: Path = ROOT) -> str | None:
    """The directory ``path`` (from ``root``) names in this tree: where the recorded moves took it, when that is a
    directory, else itself when it is one. The moves come first: a machine that built a package before its move still has
    the package's old directory, holding only the build (`.lake`, which `carry` moves)."""
    p = posixpath.normpath(path)
    if p.startswith("..") or posixpath.isabs(p):
        return None
    q = now(p, recorded(root))
    return q if (root / q).is_dir() else p if (root / p).is_dir() else None


def carry(tree: Path, packages: list[str], moves: list[dict[str, str]], log=print, prune: bool = False) -> dict[str, str]:
    """In ``tree``, a checkout or a copy of one, move the build directory of each of ``packages`` (their paths now) that a
    move left at the package's former path to where the package is: Lake's input hashes don't depend on where a package
    is, so the modules built there stay built. A package that has a `.lake` already keeps it. A former directory the move
    emptied is removed; ``prune`` (a tree that holds only copies, a slot's) removes it with what is left in it, once no
    build directory is below it. Returns {former path: path now} of each one moved."""
    moved = {}
    for p in packages:
        dst = tree / p / ".lake"
        if os.path.lexists(dst):
            continue
        for f in former(p, moves):
            src = tree / f / ".lake"
            if src.is_dir() and not src.is_symlink():
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.rename(src, dst)
                moved[f] = p
                log(f"lean: {p}: its build directory moved with it from {f} (a layout move)")
                break
    for f in sorted(moved, key=len, reverse=True):
        d = tree / f
        if not d.is_dir() or d.is_symlink():
            continue
        if prune and not any(".lake" in dirs for _, dirs, _ in os.walk(d)):
            shutil.rmtree(d, ignore_errors=True)
        elif not any(d.iterdir()):                      # else it still holds something: untracked work, a nested build
            d.rmdir()
    return moved


def lake_packages(root: Path = ROOT) -> list[str] | None:
    """The directories of this checkout's tracked lakefiles; None where git can't list them."""
    r = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", *(f"*{n}" for n in LAKEFILES)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return sorted({posixpath.dirname(f) for f in r.stdout.split("\0") if posixpath.basename(f) in LAKEFILES})


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?", help="a directory, as some layout named it: from the working directory, or absolute")
    ap.add_argument("--carry", action="store_true", help="move this checkout's build directories left at former paths")
    args = ap.parse_args(argv)
    if args.carry == (args.path is not None):
        ap.error("give PATH or --carry")
    if args.carry:
        pkgs = lake_packages()
        if pkgs is None:
            print("moved.py: this checkout's files cannot be listed", file=sys.stderr)
            return 1
        carry(ROOT, pkgs, recorded())
        return 0
    rel = os.path.relpath(os.path.abspath(args.path), ROOT).replace(os.sep, "/")
    got = resolve(rel)
    if got is None:
        print(f"moved.py: {args.path} is not a directory of this tree, and no recorded layout move ({MOVES}) leads "
              f"from it to one", file=sys.stderr)
        return 1
    print(got)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

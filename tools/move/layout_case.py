#!/usr/bin/env python3
"""Fail a layout whose moved tree holds two paths that differ only by case.

    python3 tools/move/layout_case.py MAP... [--ref REF]

Applies each map's `[paths]`, in the order of the maps' names, to the files tracked at REF (default `origin/main`), and
prints every set of files or directories of the result that a case-insensitive file system holds as one: a checkout on
macOS merges them, so one of the two is lost or both are mixed. Exit 1 on any, with the paths and the files that land
under each; exit 0 on none. Two directories that collide are one finding, not one for each file below them.

`case_sets(paths)` is the check itself, for a caller that already has the moved tree's paths."""
from __future__ import annotations

import argparse
import collections
import os
import posixpath
import subprocess
import sys
import tomllib
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layout as L  # noqa: E402


def fold(path: str) -> str:
    """`path` as a case-insensitive file system compares it: case folded, and one Unicode normal form."""
    return unicodedata.normalize("NFD", path).casefold()


def case_sets(paths) -> list[list[str]]:
    """Each set of files or directories under `paths` (files, `/`-separated) that are spelled differently and fold to one
    path, at its shallowest: the directories of a set that differ are reported, never the paths below them."""
    spelled = collections.defaultdict(set)
    for p in paths:
        parts = p.split("/")
        for i in range(1, len(parts) + 1):
            q = "/".join(parts[:i])
            spelled[fold(q)].add(q)
    return sorted(sorted(s) for s in spelled.values() if len(s) > 1 and len({posixpath.dirname(q) for q in s}) == 1)


def moved(tracked, layouts) -> dict[str, str]:
    """{file tracked today: its path after every layout of `layouts`, each run on the tree the one before left}."""
    final = {p: p for p in tracked}
    for lay in layouts:
        for orig, cur in final.items():
            if (new := lay.file(cur)) is not None:
                final[orig] = new
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("maps", nargs="+", metavar="MAP", help="layout maps, run in the order of their names")
    ap.add_argument("--ref", default="origin/main", help="the commit whose files the maps move (default: origin/main)")
    a = ap.parse_args(argv)
    ls = subprocess.run(["git", "ls-tree", "-r", "--name-only", "-z", a.ref], capture_output=True, text=True)
    if ls.returncode:
        sys.exit(f"git ls-tree {a.ref}: {ls.stderr.strip()}")
    tracked = [p for p in ls.stdout.split("\0") if p]
    names = sorted(a.maps, key=os.path.basename)
    layouts = []
    for m in names:
        with open(m, "rb") as f:
            layouts.append(L.Layout(tomllib.load(f)))
    final = moved(tracked, layouts)
    sets = case_sets(final.values())
    for s in sets:
        print("differ only by case:")
        for q in s:
            under = sorted((new, old) for old, new in final.items() if new == q or new.startswith(q + "/"))
            new, old = under[0]
            how = "which stays" if all(n == o for n, o in under) else f"e.g. {old} -> {new}" if old != new else f"e.g. {new}"
            print(f"    {q}  ({len(under)} file{'s' if len(under) != 1 else ''}, {how})")
    n = sum(1 for old, new in final.items() if old != new)
    print(f"{len(sets)} set{'s' if len(sets) != 1 else ''} of paths that differ only by case in the tree {len(names)} map"
          f"{'s' if len(names) != 1 else ''} leave{'s' if len(names) == 1 else ''} of {a.ref} ({n} of {len(tracked)} files moved)"
          + (": give one of each set a row to another name" if sets else ""))
    return 1 if sets else 0


if __name__ == "__main__":
    sys.exit(main())

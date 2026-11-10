#!/usr/bin/env python3
"""The directories a layout move empties, and which of them the map empties row by row, so that a file a branch adds
there has no row and stays behind.  Git alone.

    emptied.py BASE MOVE MAP [NAME=COMMIT ...]

For each NAME=COMMIT: the files it has, once merged with BASE, under a directory emptied row by row that no row of the
map carries.  NAME!=COMMIT measures the branch as it stands, for one that cannot merge the base.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def files(commit: str) -> set[str]:
    return set(git("ls-tree", "-r", "--name-only", commit).split("\n")) - {""}


def dirs(paths: set[str]) -> set[str]:
    out = set()
    for p in paths:
        parts = p.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            out.add("/".join(parts[:i]) + "/")
    return out


def outermost(ds: set[str]) -> list[str]:
    return sorted(d for d in ds if not any(d != e and d.startswith(e) for e in ds))


def main(argv: list[str]) -> int:
    base, move, mp = argv[:3]
    rows = tomllib.loads(git("show", f"{base}:{mp}"))["paths"]
    dir_rows = [s for s in rows if s.endswith("/")]
    emptied = dirs(files(base)) - dirs(files(move))
    by_dir_row = {d for d in emptied if any(d.startswith(s) for s in dir_rows)}
    by_file = emptied - by_dir_row
    out = {"base": base, "move": move, "map": mp, "rows": len(rows), "directory_rows": len(dir_rows),
           "file_rows": len(rows) - len(dir_rows), "emptied": len(emptied), "emptied_outermost": len(outermost(emptied)),
           "emptied_by_a_directory_row": len(by_dir_row), "emptied_row_by_row": sorted(by_file),
           "emptied_row_by_row_outermost": outermost(by_file), "branches": {}}
    top = outermost(by_file)
    for name, commit in (a.split("=", 1) for a in argv[3:]):
        tree = commit
        if not name.endswith("!"):  # NAME!: the branch as it stands; else merged with the base, as `restack.py` does first
            tree = subprocess.run(["git", "merge-tree", "--write-tree", commit, base], capture_output=True, text=True,
                                  timeout=240).stdout.split("\n")[0]
        left = sorted(p for p in files(tree) if any(p.startswith(d) for d in top)
                      and p not in rows and not any(p.startswith(s) for s in dir_rows))
        out["branches"][name] = {"commit": commit, "tree": tree, "left_behind": len(left), "files": left}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""Does `git merge-tree` say what `git merge` does?  For each HEAD, in the clone this runs in (its work tree clean, the
merge drivers registered as the count's are): check HEAD out, `git merge --no-commit --no-ff BASE`, and compare the
outcome with `git --attr-source=HEAD merge-tree --write-tree HEAD BASE`: the files left in conflict, and when there are
none, the tree.

    real_merge.py BASE HEAD...
"""
from __future__ import annotations

import json
import subprocess
import sys


def git(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *a], capture_output=True, text=True)


def main(argv: list[str]) -> int:
    base, heads = argv[0], argv[1:]
    if git("status", "--porcelain").stdout.strip():
        raise SystemExit("the work tree is not clean")
    same = True
    for head in heads:
        p = subprocess.run(["git", f"--attr-source={head}", "merge-tree", "--write-tree", "--name-only", "--no-messages",
                            "-z", head, base], capture_output=True)
        fields = [s.decode() for s in p.stdout.split(b"\0") if s]
        tree, files = fields[0], sorted(set(fields[1:])) if p.returncode else []
        git("checkout", "-q", "--detach", head)
        m = git("merge", "--no-commit", "--no-ff", base)
        real = sorted(set(git("diff", "--name-only", "--diff-filter=U").stdout.split("\n")) - {""})
        real_tree = None if real else git("write-tree").stdout.strip()
        git("merge", "--abort")
        if git("status", "--porcelain").stdout.strip():
            git("reset", "-q", "--hard", head)
        row = {"head": head, "base": base, "merge_tree": {"files": files, "tree": None if files else tree},
               "git_merge": {"exit": m.returncode, "files": real, "tree": real_tree},
               "same_files": files == real, "same_tree": None if files or real else tree == real_tree}
        same = same and row["same_files"] and row["same_tree"] is not False
        print(json.dumps(row), flush=True)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

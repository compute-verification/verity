"""PORTCHK: add each record's own changes: the files its non-merge first-parent commits past origin/main change.

    python3 /tmp/portchk/postproc.py N...
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/agent/repos/verity")
OUT = Path("/tmp/portchk/out")


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout


def own(n: int) -> dict:
    p = OUT / f"{n}.json"
    rec = json.loads(p.read_text())
    head = rec["head"]
    commits = git("log", "--first-parent", "--no-merges", "--format=%H", f"origin/main..{head}").split()
    files = set()
    for c in commits:
        files |= set(git("diff-tree", "--no-commit-id", "--name-only", "-r", "--no-renames", c).split())
    tree = set(git("ls-tree", "-r", "--name-only", rec["tree_commit"]).splitlines()) if rec.get("tree_commit") else set()
    rec["own_commits"] = len(commits)
    rec["own"] = sorted(files)
    rec["own_lean"] = sorted(f for f in files if f.endswith(".lean") and f in tree)
    rec["own_lean_deleted"] = sorted(f for f in files if f.endswith(".lean") and f not in tree)
    rec["head_date"] = git("log", "-1", "--format=%cI", head).strip()
    p.write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    for a in sys.argv[1:]:
        r = own(int(a))
        print(a, "own", len(r["own"]), "own_lean", len(r["own_lean"]), "deleted", len(r["own_lean_deleted"]), "commits", r["own_commits"])

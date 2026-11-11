#!/usr/bin/env python3
"""`moved_merge.py WORKTREE PARTIAL ONTO MOVE`: restack.py's `moved` step by hand, for a mend: the three-way merge of PARTIAL
(a branch after the move's script) with ONTO over MOVE as the merge base, checked out at a new detached WORKTREE with the
conflict markers in its files.  Prints the conflicted files; `commit_mend.py` commits what is there once they are mended."""
import subprocess
import sys


def git(*args, cwd=None, ok=(0,)):
    p = subprocess.run(["git", *args], capture_output=True, cwd=cwd)
    if p.returncode not in ok:
        raise SystemExit(f"git {' '.join(args)}: {p.stderr.decode()}")
    return p


wt, partial, onto, move = sys.argv[1:]
p = git("merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", f"--merge-base={move}", partial, onto, ok=(0, 1))
fields = [s.decode() for s in p.stdout.split(b"\0") if s]
tree, files = fields[0], sorted(set(fields[1:])) if p.returncode else []
git("worktree", "add", "-q", "--detach", wt, partial)
git("read-tree", "-u", "--reset", tree, cwd=wt)
print(tree)
for f in files:
    print("conflicted", f)

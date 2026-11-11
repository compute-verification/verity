#!/usr/bin/env python3
"""carrier.py WORKTREE FIRST_PARENT MESSAGE COMMIT...: commit WORKTREE's index (the staged tree plus `cir223/`) with
FIRST_PARENT and every COMMIT as parents, so that a clone of the commit holds each of them; prints the commit. No ref moves.
"""
import os
import subprocess
import sys

ENV = dict(os.environ)


def git(*args, cwd, input=None):
    p = subprocess.run(["git", *args], capture_output=True, env=ENV, input=input, cwd=cwd)
    if p.returncode:
        raise SystemExit(f"git {' '.join(args)}: {p.stderr.decode()}")
    return p.stdout.decode().strip()


def main():
    wt, first, msg, *commits = sys.argv[1:]
    tree = git("write-tree", cwd=wt)
    parents, seen = [], set()
    for c in [first, *commits]:
        sha = git("rev-parse", f"{c}^{{commit}}", cwd=wt)
        if sha not in seen:
            seen.add(sha)
            parents += ["-p", sha]
    print(git("commit-tree", tree, *parents, "-F", "-", cwd=wt, input=(msg + "\n").encode()))


if __name__ == "__main__":
    main()

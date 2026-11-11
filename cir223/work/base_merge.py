#!/usr/bin/env python3
"""Merge each open head with the commit below the move, as restack.py's step `base` does (`git merge-tree`, ours the head,
theirs the base), run at the root of a checkout of that commit so the merge drivers and .gitattributes are the tree's before
the move. A clean merge becomes a commit in the object store (no ref moves); a conflicted one lists its files.

    base_merge.py BASE OUT.json PR=SHA...
"""
import json
import os
import subprocess
import sys

ENV = dict(os.environ)


def git(*args, ok=(0,), input=None):
    p = subprocess.run(["git", *args], capture_output=True, env=ENV, input=input)
    if p.returncode not in ok:
        raise SystemExit(f"git {' '.join(args)}: {p.stderr.decode()}")
    return p


def main():
    base, out, *pairs = sys.argv[1:]
    base = git("rev-parse", f"{base}^{{commit}}").stdout.decode().strip()
    rows = {}
    for pair in pairs:
        pr, sha = pair.split("=")
        head = git("rev-parse", f"{sha}^{{commit}}").stdout.decode().strip()
        row = {"pr": pr, "head": head, "base": base}
        if git("merge-base", "--is-ancestor", base, head, ok=(0, 1)).returncode == 0:
            row.update(status="holds-base", merged=head)
        else:
            p = git("merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", head, base, ok=(0, 1))
            fields = [s.decode() for s in p.stdout.split(b"\0") if s]
            tree, files = fields[0], sorted(set(fields[1:])) if p.returncode else []
            if files:
                row.update(status="conflicted", files=files, tree=tree)
            else:
                msg = f"Merge {base[:12]}, the layout move's base, into #{pr} at {head[:12]} (CIR-223 trial)\n"
                merged = git("commit-tree", tree, "-p", head, "-p", base, "-F", "-", input=msg.encode()).stdout.decode().strip()
                row.update(status="clean", merged=merged, tree=tree)
        rows[pr] = row
        print(pr, row["status"], row.get("merged", "")[:12], " ".join(row.get("files", [])), flush=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

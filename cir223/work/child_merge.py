#!/usr/bin/env python3
"""For each stack, merge the parent's base-merged head into the child (`git merge-tree`, ours the child, theirs the parent's
merge with the commit below the move), at the root of a checkout of that commit. A clean merge becomes a commit in the
object store with parents [child, parent's base merge]; no ref moves.

    child_merge.py BASE.json OUT.json CHILD=PARENT...
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
    base_json, out, *pairs = sys.argv[1:]
    rows = json.load(open(base_json))
    res = {}
    for pair in pairs:
        child, parent = pair.split("=")
        x, lp = rows[child]["head"], rows[parent].get("merged")
        row = {"child": child, "parent": parent, "head": x, "parent_merged": lp}
        if not lp:
            row.update(status="parent-conflicted")
        else:
            p = git("merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", x, lp, ok=(0, 1))
            fields = [s.decode() for s in p.stdout.split(b"\0") if s]
            tree, files = fields[0], sorted(set(fields[1:])) if p.returncode else []
            if files:
                row.update(status="conflicted", files=files)
            else:
                msg = (f"Merge #{parent}'s head with the layout move's base ({lp[:12]}) into #{child} at {x[:12]} "
                       "(CIR-223 trial)\n")
                merged = git("commit-tree", tree, "-p", x, "-p", lp, "-F", "-", input=msg.encode()).stdout.decode().strip()
                same = tree == rows[child].get("tree")
                row.update(status="clean", merged=merged, tree=tree, same_tree_as_own_base_merge=same)
        res[child] = row
        print(child, "on", parent, row["status"], row.get("merged", "")[:12], row.get("same_tree_as_own_base_merge"),
              " ".join(row.get("files", [])), flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

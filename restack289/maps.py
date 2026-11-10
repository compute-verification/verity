#!/usr/bin/env python3
"""VLLM-289 scratch: does each crossed result carry the branch's own change and nothing else?

    maps.py STAGED BASE OUT.json RUN...

For each clean result in the runs' out/results, from its bundle: the numstat of the branch's own change before (BASE to the
merge of BASE into the head; with no merge, the head against its merge base with BASE) against the result's over STAGED,
and whether the result descends from both the head and STAGED. Equal counts of files and lines say the move carried the
change; a few lines' difference is a sorted list both sides added to.
"""
import glob
import json
import os
import subprocess
import sys

staged, base, outp, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]


def git(*a):
    p = subprocess.run(["git", *a], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"git {a[:3]}: {p.stderr[-200:]}")
    return p.stdout


def stat(a, b):
    files = add = dele = 0
    for line in git("diff", "--numstat", "--no-renames", a, b).splitlines():
        x, y, _ = line.split("\t", 2)
        files, add, dele = files + 1, add + (int(x) if x.isdigit() else 0), dele + (int(y) if y.isdigit() else 0)
    return [files, add, dele]


def anc(a, b):
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b]).returncode == 0


res = {}
for r in runs:
    for f in sorted(glob.glob(os.path.expanduser(f"~/.research/runs/{r}/out/results/pr-*.json"))):
        d = json.load(open(f))
        if d.get("status") != "clean" or not (d.get("bundle") or {}).get("file"):
            continue
        git("fetch", "-q", os.path.join(os.path.dirname(os.path.dirname(f)), d["bundle"]["file"]), d["bundle"]["ref"])
        R, H, tip = d["result"], d["head"], d.get("tip")
        before = stat(base, tip) if tip and tip != H else stat(git("merge-base", H, base).strip(), H)
        after = stat(staged, R)
        ok = anc(H, R) and anc(staged, R)
        res[str(d["pr"])] = {"result": R, "head": H, "before": before, "after": after, "same": before == after, "descends": ok, "run": r}
        print(d["pr"], R[:12], "before", before, "after", after, "SAME" if before == after else "DIFFERS",
              "" if ok else "NOT-A-DESCENDANT", flush=True)
json.dump(res, open(outp, "w"), indent=1)

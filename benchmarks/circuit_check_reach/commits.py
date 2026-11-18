import collections
import json
import re
import subprocess
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "98e9c1de7be1"
TIP = sys.argv[2] if len(sys.argv) > 2 else "origin/interfaces"
files = open("/tmp/allreach.txt").read().split()


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout


rows = []
for line in git("log", "--no-merges", "--format=%H %cI", f"{BASE}..{TIP}", "--", *files).splitlines():
    h, t = line.split()
    touched = [f for f in git("show", "--name-only", "--format=", h).split() if f in files]
    ref = re.sub(r"[~^].*$", "", git("name-rev", "--name-only", "--refs=refs/remotes/origin/cursor/*", h).strip())
    rows.append({"c": h[:11], "t": t, "branch": ref.replace("remotes/origin/", ""), "files": touched})
json.dump(rows, open("/tmp/allreach_commits.json", "w"), indent=1)
print(dict(collections.Counter(r["branch"].rsplit("-", 1)[-1] for r in rows).most_common()))

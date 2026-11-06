"""tools/move/one14/conflicts.py LOW NEXT HEAD: what the queue's preflight (a) would say of HEAD on NEXT, from the checkout
it runs in: the files a merge of HEAD onto NEXT leaves conflicted (`research.queue.conflicting_files`: `git merge-tree`,
a conflict only in pins counting as none), and for each the members of NEXT above LOW (first parent) whose own merge
changed it (`research.queue.touched_by`), then the count and list of those members.

    git fetch -q origin checked next
    python3 tools/move/one14/conflicts.py origin/checked origin/next MOVE_COMMIT
"""
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

repo = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())
sys.path.insert(0, str(repo / "tools/research/src"))
from research import queue as Q  # noqa: E402


def sha(ref: str) -> str:
    return subprocess.run(["git", "rev-parse", ref], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


low, nxt, head = sys.argv[1], sha(sys.argv[2]), sha(sys.argv[3])
log = subprocess.run(["git", "log", "--first-parent", "--reverse", "--format=%H %s", f"{low}..{nxt}"], cwd=repo,
                     capture_output=True, text=True, check=True).stdout.splitlines()
members = []
for line in log:
    c, s = line.split(" ", 1)
    m = re.match(r"Queue: #(\d+) (.*)", s)
    members.append(SimpleNamespace(commit=c, pr=m.group(1) if m else c[:10], subject=m.group(2) if m else s))
files = Q.conflicting_files(repo, nxt, head)
print(f"{len(members)} members of next {nxt[:10]} above {low}; a merge of {head[:10]} onto it conflicts in {len(files)} files")
cache: dict = {}
by_file = {f: Q.touched_by(repo, members, [f], cache) for f in files}
for f in files:
    print(f"  {f}: {', '.join(by_file[f]) or '(no member above LOW)'}")
prs = {p for ps in by_file.values() for p in ps}
print(f"{len(prs)} members touch them:")
for m in members:
    if f"#{m.pr}" in prs:
        print(f"  #{m.pr} {m.subject[:100]}")

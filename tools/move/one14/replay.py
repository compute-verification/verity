"""tools/move/one14/replay.py LOW NEXT HEAD...: for each HEAD, next rebuilt on it as the queue builds it: each member of NEXT
above LOW (first parent, oldest first) merged in turn onto the tip so far by `research.queue.merged_tree`, with the tip's
merge drivers (`base_drivers`), a member a merge commit's second parent (its PR's head) and a drive commit its own change.
A member whose merge leaves a conflict is one the queue ejects; the rest stack. Run it from a checkout of the queue's tools.

    git fetch -q origin checked next
    python3 tools/move/one14/replay.py origin/checked origin/next origin/checked PART1 PART2
"""
import re
import subprocess
import sys
from pathlib import Path

repo = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())
sys.path.insert(0, str(repo / "tools/research/src"))
from research import queue as Q  # noqa: E402


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


low, nxt = sys.argv[1], git("rev-parse", sys.argv[2])
members = []
for line in git("log", "--first-parent", "--reverse", "--format=%H %P\t%s", f"{low}..{nxt}").splitlines():
    ids, s = line.split("\t", 1)
    c, *parents = ids.split()
    m = re.match(r"Queue: #(\d+) ", s)
    members.append((c, parents, f"#{m.group(1)}" if m else f"{c[:10]} ({s[:60]})"))
for head in sys.argv[3:]:
    tip = git("rev-parse", head)
    ejected = []
    for c, parents, name in members:
        if len(parents) == 2:
            with Q.base_drivers(repo, tip) as config:
                tree, files, _ = Q.merged_tree(repo, tip, parents[1], config=config)
            other = parents[1]
        else:
            r = subprocess.run(["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", f"--merge-base={parents[0]}",
                                tip, c], cwd=repo, capture_output=True, text=True)
            tree = r.stdout.split()[0] if r.returncode == 0 else None
            files = sorted({f for f in r.stdout.splitlines()[1:] if f}) if r.returncode == 1 else []
            other = c
        if tree is None:
            ejected.append((name, files))
            continue
        tip = git("commit-tree", tree, "-p", tip, "-p", other, "-m", f"replay {name}")
    print(f"{head}: {len(members)} members of next {nxt[:10]} above {low}; {len(ejected)} conflict")
    for name, files in ejected:
        print(f"  {name}: {', '.join(files)}")

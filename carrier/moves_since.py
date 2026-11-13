"""What `main` gained in moves since a carried head's own `main`, against that head's own change.

    python3 moves_since.py HEAD [MAIN_REF]

HEAD's own main is `git merge-base HEAD MAIN_REF` (default origin/main). For each commit after it on MAIN_REF with a
`Move:` or `Layout-move:` line: the files it touches (old and new paths) that HEAD changes over its own main, and, for a
layout move, the lines HEAD adds over its own main that name an old path of the move's map or its dotted form. Then
`git merge-tree` of MAIN_REF and HEAD. Exit 1 if any count is over 0 or the merge has a conflict. Reads git alone.
"""
import re
import subprocess
import sys
import tomllib


def git(*a, check=True):
    r = subprocess.run(["git", *a], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"git {' '.join(a)}: {r.stderr.strip()[:300]}")
    return r.stdout


head = sys.argv[1]
ref = sys.argv[2] if len(sys.argv) > 2 else "origin/main"
tip = git("rev-parse", ref).strip()
own = git("merge-base", head, tip).strip()
changed = set(git("diff", "--name-only", "--no-renames", own, head).split())
added = [x[1:] for x in git("diff", "--no-renames", "-U0", own, head).splitlines() if x.startswith("+") and not x.startswith("+++")]
print(f"head {head[:12]} holds main {own[:12]}; main is now {tip[:12]} ({len(git('rev-list', f'{own}..{tip}').split())} commits later); "
      f"the head changes {len(changed)} files and adds {len(added)} lines over its own main")
bad = 0
moves = []
for c in git("rev-list", "--reverse", f"{own}..{tip}").split():
    body = git("show", "-s", "--format=%B", c)
    m = re.search(r"^(Move|Layout-move): *(\S+)", body, re.M)
    if m:
        moves.append((c, m.group(1), m.group(2), body.splitlines()[0]))
if not moves:
    print("no move on main since the head's own main")
for c, kind, path, subject in moves:
    files = set(git("show", "--no-renames", "--name-only", "--format=", c).split())
    both = sorted(files & changed)
    line = f"{kind} {c[:12]} {path} ({subject[:70]}): touches {len(files)} paths, {len(both)} of them changed by the head"
    named = []
    if kind == "Layout-move":
        table = tomllib.loads(git("show", f"{c}:{path}")).get("paths", {})
        olds = set()
        for old in table:
            olds.add(old)
            if old.endswith(".py"):
                olds.add(old[:-3].replace("/", "."))
            elif old.endswith("/"):
                olds.add(old.rstrip("/").replace("/", "."))
        named = [(o, a[:120]) for a in added for o in olds if o in a]
        line += f"; the head's added lines name an old path of its map {len(named)} time(s)"
    print(line)
    for x in both[:8]:
        print("    both touch:", x)
    for o, a in named[:8]:
        print(f"    names {o}: {a}")
    bad += len(both) + len(named)
r = subprocess.run(["git", "merge-tree", "--write-tree", tip, head], capture_output=True, text=True)
print(f"git merge-tree {tip[:12]} {head[:12]}: exit {r.returncode}" + ("" if r.returncode == 0 else " CONFLICT: " + " / ".join(r.stdout.splitlines()[1:6])))
sys.exit(1 if bad or r.returncode else 0)

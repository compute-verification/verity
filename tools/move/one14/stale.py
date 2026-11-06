"""tools/move/one14/stale.py MAP LOW NEXT: the members of NEXT above LOW (first parent) whose own change adds a reference
to a path or module the move renames (MAP's `[paths]` and `[modules]`, and a from-import of a moved module from its old
package), or adds a file under a moved path: they merge cleanly onto the move and break after it."""
import re
import subprocess
import sys
import tomllib

mp, low, nxt = sys.argv[1:4]
m = tomllib.load(open(mp, "rb"))
paths, modules = m["paths"], m["modules"]


def git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, errors="replace", check=True).stdout


mods = sorted(modules, key=len, reverse=True)
mod_re = re.compile("|".join(re.escape(o) + r"(?![\w])" for o in mods))
path_re = re.compile("|".join(re.escape(o) + ("" if o.endswith("/") else r"(?![\w.-])") for o in sorted(paths, key=len, reverse=True)))
moved_names: dict[str, set[str]] = {}
for o in modules:
    pkg, _, name = o.rpartition(".")
    if o.count(".") == 4:
        moved_names.setdefault(pkg, set()).add(name)
from_re = re.compile(r"from (verity\.core\.protocols\.(?:certifier|pous|pouw)) import \(?([^)#\n]*)")
for line in git("log", "--first-parent", "--reverse", "--format=%H %s", f"{low}..{nxt}").splitlines():
    c, s = line.split(" ", 1)
    name = (re.match(r"Queue: (#\d+)", s) or [None, c[:10]])[1]
    hits = []
    for f in git("diff", "--name-only", "--diff-filter=A", f"{c}^1", c).split():
        if any(f.startswith(o) if o.endswith("/") else f == o for o in paths):
            hits.append(f"adds {f}")
    cur = None
    for ln in git("diff", "-U0", "--no-renames", f"{c}^1", c).splitlines():
        if ln.startswith("+++ "):
            cur = ln[6:]
            continue
        if not ln.startswith("+") or cur is None:
            continue
        text = ln[1:]
        found = [x.group(0) for x in mod_re.finditer(text)] + [x.group(0) for x in path_re.finditer(text)]
        for fm in from_re.finditer(text):
            got = {w.split(" as ")[0].strip() for w in fm.group(2).split(",")}
            found += [f"{fm.group(1)} import {n}" for n in sorted(got & moved_names.get(fm.group(1), set()))]
        if found:
            hits.append(f"{cur}: {', '.join(sorted(set(found)))}")
    if hits:
        print(f"{name} {s[len('Queue: ') + len(name) + 1:][:80] if s.startswith('Queue: ') else s[:80]}")
        for h in hits[:12]:
            print(f"    {h}")
        if len(hits) > 12:
            print(f"    ... {len(hits) - 12} more")

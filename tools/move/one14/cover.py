"""tools/move/one14/cover.py BASE_TREE HEAD_TREE MAP: for each suite, the moved files it covered at BASE that it doesn't
cover at HEAD at their new place (lost), those it newly covers (gained), and the `__init__.py`s of verity/ml/protocols it
lacks beside a moved file it covers (lacks), read with each tree's own `tools/verity/check/suites.py`."""
import importlib.util
import os
import subprocess
import sys
import tomllib


def suites_of(tree, name):
    cwd = os.getcwd()
    os.chdir(tree)
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(tree, "tools/verity/check/suites.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        pkgs, suites = m.members(), m.discover()
        return m, {s.name: (s, m.inputs_of(s, pkgs)) for s in suites}
    finally:
        os.chdir(cwd)


base, head, mp = sys.argv[1:4]
paths = tomllib.load(open(mp, "rb"))["paths"]
files = subprocess.run(["git", "ls-files"], cwd=base, capture_output=True, text=True, check=True).stdout.split()
moved = {}
for f in files:
    best = max((o for o in paths if f == o or (o.endswith("/") and f.startswith(o))), key=len, default=None)
    if best:
        moved[f] = paths[best] + f[len(best):] if best.endswith("/") else paths[best]
mb, sb = suites_of(base, "suites_base")
mh, sh = suites_of(head, "suites_head")
print(f"{len(moved)} moved files; {len(sb)} suites at base, {len(sh)} at head")
for name in sorted(set(sb) | set(sh)):
    if name not in sb or name not in sh:
        print(f"{name}: only at {'head' if name in sh else 'base'}")
        continue
    (s0, p0), (s1, p1) = sb[name], sh[name]
    c0, c1 = mb.dropped(s0, p0), mh.dropped(s1, p1)
    lost = [o for o, n in moved.items() if mb.covered(o, p0, c0) and not mh.covered(n, p1, c1)]
    gained = [o for o, n in moved.items() if not mb.covered(o, p0, c0) and mh.covered(n, p1, c1)]
    need = set()
    for n in moved.values():
        if mh.covered(n, p1, c1):
            need |= {"verity/ml/protocols/__init__.py", "/".join(n.split("/")[:4]) + "/__init__.py"}
    lacks = sorted(i for i in need if not mh.covered(i, p1, c1))
    if lost or gained or lacks:
        print(f"{name} ({s1.path}):")
        for tag, xs in (("lost", lost), ("gained", gained), ("lacks", lacks)):
            if xs:
                print(f"  {tag} {len(xs)}: {', '.join(sorted(xs)[:6])}{' ...' if len(xs) > 6 else ''}")

"""Import graph of every in-tree Lean package: longest chains by module count, lines, or measured seconds.

    python3 graph.py REPO [ROOT] [count|lines|TIMES.json|mods]

`mods` lists ROOT's in-tree import closure, one module a line (Extract.lean's MODS)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, str(ROOT / "tools/verity/lean"))
import headers  # noqa: E402


def packages():
    out = subprocess.run(["git", "ls-files", "*lakefile.toml"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    for f in out:
        if "tests/fixtures" in f:
            continue
        d = (ROOT / f).parent
        cfg = tomllib.loads((ROOT / f).read_text())
        yield cfg["name"], (d / cfg.get("srcDir", ".")).resolve(), d


def index():
    mods: dict[str, tuple[Path, str]] = {}
    for pkg, src, pdir in packages():
        files = subprocess.run(["git", "ls-files", "*.lean"], cwd=src, capture_output=True, text=True).stdout.split()
        for rel in files:
            p = (src / rel).resolve()
            if ".lake" in p.parts:
                continue
            # a file belongs to the package whose directory holds it most deeply
            name = rel[:-5].replace("/", ".")
            if src.name == "Security":
                top = name.split(".")[0]
                want = {"Proofs": "Proofs", "Properties": "Properties"}.get(top)
                if (want is None and pdir != src) or (want is not None and pdir.name != want):
                    continue
                pkg = {"Proofs": "security_proofs", "Properties": "security_properties"}.get(top, "security")
            prev = mods.get(name)
            if prev is None or len(str(pdir)) > len(str(prev[2])):
                mods[name] = (p, pkg, pdir)
    return mods


MODS = index()


@lru_cache(None)
def imports(m: str) -> tuple[str, ...]:
    src = MODS[m][0].read_text()
    imps = headers.imports(src) or []
    return tuple(headers.name(i) for i in imps if headers.name(i) in MODS)


@lru_cache(None)
def lines(m: str) -> int:
    return MODS[m][0].read_text().count("\n")


def longest(root: str, weight):
    memo: dict[str, tuple[float, list[str]]] = {}
    sys.setrecursionlimit(100000)

    def go(m):
        if m in memo:
            return memo[m]
        best = (0.0, [])
        for i in imports(m):
            r = go(i)
            if r[0] > best[0]:
                best = r
        res = (best[0] + weight(m), [m] + best[1])
        memo[m] = res
        return res

    return go(root)


def closure(root):
    seen, stack = set(), [root]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        stack.extend(imports(m))
    return seen


if __name__ == "__main__":
    root = sys.argv[2] if len(sys.argv) > 2 else "Proofs.Flock"
    mode = sys.argv[3] if len(sys.argv) > 3 else "count"
    if mode == "mods":
        print("\n".join(sorted(closure(root))))
        sys.exit(0)
    if mode == "count":
        w = lambda m: 1
    elif mode == "lines":
        w = lines
    else:
        times = json.loads(Path(mode).read_text())
        w = lambda m: times.get(m, 0.0)
    tot, path = longest(root, w)
    cl = closure(root)
    print(f"# root {root}: closure {len(cl)} modules, {sum(lines(m) for m in cl)} lines; longest by {mode}: {tot:.0f}, {len(path)} modules, {sum(lines(m) for m in path)} lines")
    for m in path:
        print(f"{lines(m):6d} {MODS[m][1]:22s} {m}")

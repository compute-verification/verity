"""Scratch, never to land: the retired-terms list between BASE and the checked-out tree, whole, and the lint's own sites in
pouw_cuda's sources.

    python sites.py BASE

It prints every key of the list that left, came or changed its count, with the keys that left matched to the keys that
came by git's renames, so what is left over is the list's own growth; then, for each file under a pouw_cuda/ directory,
each site the tree's lint finds in it, by line and term. It exits 0 when the growth is what the lint finds in three
row_table sources and nothing else."""
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

LIST = "tests/lint/allowlists/repo_retired_terms.json"
WANT = {"row_table.cu": 5, "row_table.cuh": 13, "row_table_host.cpp": 4}


def git(*a: str) -> str:
    return subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout


def load(text: str) -> Counter:
    c: Counter = Counter()
    for e in json.loads(text)["entries"]:
        c[e["file"]] += e["count"]
    return c


def main() -> int:
    base = sys.argv[1]
    a, b = load(git("show", f"{base}:{LIST}")), load(Path(LIST).read_text())
    ren = {}
    for line in git("-c", "diff.renameLimit=0", "diff", "-M50%", "--name-status", "--diff-filter=R", base, "HEAD").splitlines():
        _, old, new = line.split("\t")
        ren[old] = new
    print(f"{base[:12]}: {len(a)} keys, {sum(a.values())} in total; HEAD {git('rev-parse', 'HEAD')[:12]}: {len(b)} keys, "
          f"{sum(b.values())} in total; difference {sum(b.values()) - sum(a.values())}")
    gone, came = set(a) - set(b), set(b) - set(a)
    moved = {k: ren[k] for k in gone if ren.get(k) in came}
    print(f"keys that left: {len(gone)}; that came: {len(came)}; left and came as one renamed file: {len(moved)}, "
          f"{sum(1 for k, n in moved.items() if a[k] == b[n])} of them with the same count")
    for k, n in sorted(moved.items()):
        if a[k] != b[n]:
            print(f"  moved with another count: {k} {a[k]} -> {n} {b[n]}")
    left = sorted(gone - set(moved))
    new = sorted(came - set(moved.values()))
    for k in left:
        print(f"  left, and git pairs no key that came with it: {k} {a[k]}")
    for k in new:
        print(f"  came, and git pairs no key that left with it: {k} {b[k]}")
    changed = [k for k in sorted(set(a) & set(b)) if a[k] != b[k]]
    for k in changed:
        print(f"  same key, another count: {k} {a[k]} -> {b[k]}")
    # a file git reads as rewritten (under half the same) still pairs by its name and count
    pairs = []
    for k in list(left):
        twin = [n for n in new if Path(n).name == Path(k).name and b[n] == a[k]]
        if len(twin) == 1:
            pairs.append((k, twin[0]))
            left.remove(k)
            new.remove(twin[0])
    for k, n in pairs:
        print(f"  paired by name and count: {k} -> {n} ({a[k]})")
    growth = {Path(n).name: b[n] for n in new}
    print(f"the list's growth, after the pairs: {sorted((n, b[n]) for n in new)}; keys that only left: {left}")

    spec = importlib.util.spec_from_file_location("retired_terms", "tests/lint/test_repo_retired_terms.py")
    lint = importlib.util.module_from_spec(spec)
    sys.modules["retired_terms"] = lint
    sys.path[:0] = ["tests/lint", "tests"]
    spec.loader.exec_module(lint)
    found: dict[str, int] = {}
    terms: Counter = Counter()
    for f in sorted(git("ls-files", "*/pouw_cuda/*").split()):
        s = lint.sites(Path(f).read_text(encoding="utf-8", errors="replace"), f) if lint.holds(f) else []
        found[f] = len(s)
        print(f"{f}: {len(s)} site(s) by the tree's lint; the list allows {b.get(f, 0)}; held by the lint: {lint.holds(f)}")
        for line, term in s:
            terms[term] += 1
            print(f"    {Path(f).name}:{line} {term}")
    print(f"terms at those sites: {dict(terms)}")
    by_name = {Path(f).name: n for f, n in found.items() if n and Path(f).name != 'ncp2_fused.cu'}
    ok = growth == WANT and not left and not changed and by_name == WANT and all(a[k] == b[n] for k, n in moved.items())
    print("the growth is the three row_table sources' sites and nothing else:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

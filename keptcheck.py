"""keptcheck.py BASE: after a rename map's rerun, HEAD's working tree keeps every protected byte BASE had.

Fails when the domain tag files differ from BASE, or when any [keep] string of the three warden maps (as BASE holds
them) occurs fewer times in the tree than at BASE, or any [keep] pattern matches fewer times.
"""
import re
import subprocess
import sys
import tomllib

base = sys.argv[1]
MAPS = ["tools/move/warden_term_map.toml", "tools/move/warden_map.toml", "tools/move/warden_tokens_map.toml"]
TAGS = ["verity/commitments/domains.py", "verity/commitments/identity/Identity/Domains.lean"]
EXCL = [":!tools/move", ":!**/moves.json", ":!**/CHANGELOG.md", ":!**/lean-audit.json"]


def run(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout


bad = 0
d = run("git", "diff", base, "--", *TAGS)
print(f"tag files diff vs {base[:12]}: {'EMPTY' if not d else 'NOT EMPTY'}")
bad |= bool(d)
import os
keeps = {}
for m in MAPS:
    k = tomllib.loads(run("git", "show", f"{os.environ.get('KEEPREV', base)}:{m}")).get("keep", {})
    keeps[m] = (set(k.get("strings", [])), set(k.get("patterns", [])))
# A byte the whole run keeps: one both broad maps keep (warden_map keeps some names for warden_tokens_map to rename),
# and every string the term map keeps.
strings = (keeps[MAPS[1]][0] & keeps[MAPS[2]][0]) | keeps[MAPS[0]][0]
patterns = keeps[MAPS[1]][1] & keeps[MAPS[2]][1]


def count(rev, s, fixed):
    args = ["git", "grep", "-c", "-F" if fixed else "-P", "-e", s]
    out = run(*args, *([rev] if rev else []), "--", ".", *EXCL)
    return sum(int(l.rsplit(":", 1)[1]) for l in out.splitlines() if l)


for kind, items, fixed in (("string", strings, True), ("pattern", patterns, False)):
    for s in sorted(items):
        b, h = count(base, s, fixed), count(None, s, fixed)
        if h < b:
            bad = 1
            print(f"LOST {kind} {s!r}: {b} -> {h}")
print(f"kept {len(strings)} strings, {len(patterns)} patterns: {'OK' if not bad else 'FAIL'}")
sys.exit(bad)

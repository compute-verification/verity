#!/usr/bin/env python3
"""lock_text.py BASE HEAD: every line HEAD adds over BASE to a lock (a `lean-audit.json`, an `unread-caps.json`) that
carries, outside the fields the public export strips (`owner`, `issue`, `due`), a Linear id, a pull request's number, a
lane's handle or a date. Exit 1 when one does."""
import re, subprocess, sys

base, head = sys.argv[1:3]
out = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout
locks = [p for p in out("diff", "--name-only", base, head).split() if p.endswith(("lean-audit.json", "unread-caps.json"))]
STRIPPED = re.compile(r'^\s*"(owner|issue|due)"\s*:')
WHAT = {"a Linear id": re.compile(r"\b(?!SHA-|KEM-|DS-|RTX-|FP-)[A-Z]{2,6}-\d{1,5}\b"), "a pull request's number": re.compile(r"(?<![\w&])#\d{3,5}\b"),
        "a handle": re.compile(r"(?<![\w.])@[a-z][a-z0-9_-]+"),
        "a date": re.compile(r"\b(20\d\d-\d\d-\d\d|\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*( 20\d\d)?|"
                             r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]* \d{1,2})\b")}
hits = 0
for p in locks:
    added = [l[1:] for l in out("diff", "-U0", base, head, "--", p).splitlines() if l.startswith("+") and not l.startswith("+++")]
    kept = [l for l in added if not STRIPPED.match(l)]
    found = [(name, l.strip()) for l in kept for name, rx in WHAT.items() if rx.search(l)]
    print(f"{p}: {len(added)} added line(s), {len(added) - len(kept)} of them a stripped field (owner, issue, due); "
          f"{len(found)} of the rest carry an id, a number, a handle or a date")
    for name, l in found:
        hits += 1
        print(f"    {name}: {l[:260]}")
print(f"{len(locks)} lock(s) changed; {hits} line(s) to look at")
sys.exit(1 if hits else 0)

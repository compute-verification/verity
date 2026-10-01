#!/usr/bin/env python3
"""goals.md's Morning set -> a verity/overnight-numbers/v0 file, owner `morning-set`: per goal its text, latest status note and
checked state, keyed `<section>-<n>`. Node 1's publisher turns it into the panel verity/overnight-morning-set.
Usage: morning_set.py GOALS_MD > morning-set.json"""
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goals_notes import as_of, note_and_goal  # noqa: E402

SECTIONS = {"Circuits": "circuits", "Proofs": "proofs", "Compute accounting": "compute", "Memory accounting": "memory",
            "Infra": "infra", "PRs": "prs"}
SOURCE = "docs/goals.md Morning set"


def chunks(text: str, n: int = 200) -> list[str]:
    out = [""]
    for w in text.split():
        if out[-1] and len(out[-1]) + 1 + len(w) > n:
            out.append("")
        out[-1] = (out[-1] + " " + w).strip()[:n]
    return [c for c in out if c]


def main(path: str) -> int:
    mtime = datetime.fromtimestamp(os.stat(path).st_mtime, timezone.utc)
    stamp = mtime.strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = open(path, encoding="utf-8").read().splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("## Morning set"))
    except StopIteration:
        print("no '## Morning set' section", file=sys.stderr)
        return 1
    section, count, rows = None, {}, []
    for l in lines[start + 1:]:
        if l.startswith("## "):
            break
        if l.startswith("### "):
            section = SECTIONS.get(l[4:].split("(")[0].strip())
            continue
        m = re.match(r"- \[( |x)\] (.*)", l)
        if not m or section is None:
            continue
        note, goal = note_and_goal(m[2])
        while goal.startswith("("):
            older, rest = note_and_goal(goal)
            if older is None:
                break
            goal = rest
        count[section] = count.get(section, 0) + 1
        key = f"{section}-{count[section]}"
        t = as_of(note, mtime).strftime("%Y-%m-%dT%H:%M:%SZ") if note else None
        row = lambda metric, value, when: {"goal": key, "metric": metric, "value": value, "unit": None, "as_of": when, "source": SOURCE}  # noqa: E731
        rows += [row("goal" if i == 0 else "goal, continued", c, stamp) for i, c in enumerate(chunks(goal))]
        rows += [row("status" if i == 0 else "status, continued", c, t) for i, c in enumerate(chunks(note or ""))]
        rows.append(row("done", "yes" if m[1] == "x" else "no", t or stamp))
    print(json.dumps({"schema": "verity/overnight-numbers/v0", "owner": "morning-set", "updated_utc": stamp, "numbers": rows},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

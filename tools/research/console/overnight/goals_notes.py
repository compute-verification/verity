#!/usr/bin/env python3
"""goals.md's Overnight set -> verity/overnight-goal-notes/v0: each goal line's status note and checked state, filed under the
owner and goal key the console's Overnight view reads. Usage: goals_notes.py GOALS_MD > notes.json (exit 1 if a line can't be read)."""
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

SECTIONS = {"Proofs": "proofs", "Circuits": "circuits", "Compute accounting": "compute-accounting",
            "Memory accounting": "memory-accounting", "Infra": "infra", "PRs": "pr-captain", "Console": "console"}
# (section owner, phrase in the goal text) -> view key; keys from the console's 0625Z/0650Z requests, made up for panel-only goals
KEYS = [
    ("proofs", "Overhead against K", "overhead"), ("proofs", "GPU held per proof job", "gpu-held-per-job"),
    ("proofs", "#638 landed", "pr-638"),
    ("circuits", "Pure Boolean IR", "boolean-ir"), ("circuits", "coverage grid", "coverage"),
    ("circuits", "Prediction builder", "predictor"), ("circuits", "Llama-3.2-1B Build", "llama-1b-build"),
    ("circuits", "Commit holds a GPU", "commit-gpu-hold"), ("circuits", "Gemma-2's three batch-1", "gemma-stochastic"),
    ("compute-accounting", "Pearl-C stack", "pearl-c-stack"), ("compute-accounting", "W1 opcode inventory", "w1"),
    ("compute-accounting", "Pearl-C4", "pearl-c4"), ("compute-accounting", "Served decode hill-climb", "served-decode"),
    ("compute-accounting", "NCP with a speed target", "ncp"), ("compute-accounting", "non-Pearl design", "non-pearl"),
    ("compute-accounting", "Llama-3.1-70B FP8", "llama-70b-fp8"), ("compute-accounting", "Lower FP8", "lower-fp8-gamma"),
    ("memory-accounting", "Π₂", "pi2-call-time"), ("memory-accounting", "in-degree", "in-degree"),
    ("memory-accounting", "band decode", "band-decode"), ("memory-accounting", "P2 measured", "p2"),
    ("infra", "Held-but-idle", "held-idle"), ("infra", "--queue", "queue"), ("infra", "Quota cutover", "quota-cutover"),
    ("infra", "Hardened", "hardened"),
    ("pr-captain", "open PRs", "open-prs"),
    ("console", "morning view", "morning-view"),
]
PDT = timezone(timedelta(hours=-7))
TIME = re.compile(r"\b(1[0-2]|[1-9]):([0-5]\d) ?(AM|PM) P(?:D)?T\b")


NEXT = re.compile(r"\s+(\((1[0-2]|[1-9]):[0-5]\d ?(AM|PM) P(?:D)?T|[A-Z#`])")  # what follows a note: an older note, or the goal


def note_and_goal(text: str) -> tuple[str | None, str]:
    """`(note) goal` -> (note, goal); no leading note -> (None, text). The note ends at its balancing `)` when an older note or
    the goal text follows it; a note with an unbalanced parenthesis ends at the next `)` that is followed by one."""
    if not text.startswith("("):
        return None, text
    depth, start = 0, 1
    for i, ch in enumerate(text):
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth == 0:
            if NEXT.match(text, i + 1):
                return text[1:i].strip(), text[i + 1:].strip()
            start = i + 1
            break
    for i in range(start, len(text)):
        if text[i] == ")" and NEXT.match(text, i + 1):
            return text[1:i].strip(), text[i + 1:].strip()
    return None, text


def as_of(note: str, fallback: datetime) -> datetime:
    """The note's first Pacific time: 6–11 PM is 30 Sep, anything else 1 Oct (the overnight set's span); none: goals.md's mtime.
    A goal without a note gets no time at all, so any owner file outranks it."""
    m = TIME.search(note or "")
    if not m:
        return fallback
    h = int(m[1]) % 12 + (12 if m[3] == "PM" else 0)
    day = 30 if m[3] == "PM" and h >= 18 else 1
    return datetime(2026, 9 if day == 30 else 10, day, h, int(m[2]), tzinfo=PDT).astimezone(timezone.utc)


def main(path: str) -> int:
    mtime = datetime.fromtimestamp(os.stat(path).st_mtime, timezone.utc)
    lines = open(path, encoding="utf-8").read().splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("## Overnight set"))
    except StopIteration:
        print("no '## Overnight set' section", file=sys.stderr)
        return 1
    owner, goals, bad = None, [], []
    for l in lines[start + 1:]:
        if l.startswith("## "):
            break
        if l.startswith("### "):
            owner = SECTIONS.get(l[4:].split("(")[0].strip())
            continue
        m = re.match(r"- \[( |x)\] (.*)", l)
        if not m:
            continue
        note, goal = note_and_goal(m[2])
        while goal.startswith("("):  # older notes stacked after the latest one
            older, rest = note_and_goal(goal)
            if older is None:
                break
            goal = rest
        keys = [k for o, phrase, k in KEYS if o == owner and phrase in goal]
        if owner is None or len(keys) != 1:
            bad.append(f"{owner}: {goal[:80]!r} matches {keys}")
            continue
        goals.append({"owner": owner, "key": keys[0], "done": m[1] == "x", "goal": goal[:200], "note": note,
                      "as_of": as_of(note, mtime).strftime("%Y-%m-%dT%H:%M:%SZ") if note else None})
    for b in bad:
        print("unmatched goal line:", b, file=sys.stderr)
    print(json.dumps({"schema": "verity/overnight-goal-notes/v0", "source": "Project store docs/goals.md, Overnight set",
                      "updated_utc": mtime.strftime("%Y-%m-%dT%H:%M:%SZ"), "goals": goals}, ensure_ascii=False))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

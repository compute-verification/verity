#!/usr/bin/env python3
"""One owner's open heads across a cut, from `open_heads.py`'s result and the store's labels: what each does at the
cut's base, whether its restack gains a pins commit, which other open heads it contains, its mark and its latest
verdict, and the order to cross them in (a head after every open head it contains; among those free to go, marked heads
with a land verdict first, then by number).

    plan.py RESULT.json OWNER [--labels FILE]

FILE holds, for each head, a line `== #N SHA` and under it the `ready=` and `reviewed=` lines of
`research data labels pr:N@SHA` (default: asked now, one head at a time).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys


def labels_of(n: int, sha: str) -> list[str]:
    p = subprocess.run(["research", "data", "labels", f"pr:{n}@{sha}"], capture_output=True, text=True)
    return [ln for ln in p.stdout.splitlines() if re.match(r"^\S+  (ready|reviewed)=", ln)]


def state(lines: list[str]) -> tuple[bool, str]:
    """(marked, the latest verdict or '') of a head's `ready=` and `reviewed=` lines, oldest first."""
    ready = [m.group(1) for ln in lines if (m := re.match(r"^\S+  ready=(\w+)", ln))]
    seen = [m.group(1, 2) for ln in lines if (m := re.match(r"^\S+  reviewed=(\w+)  by (\S+)", ln))]
    return bool(ready) and ready[-1] == "true", f"{seen[-1][0]} ({seen[-1][1]})" if seen else ""


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("owner")
    ap.add_argument("--labels")
    a = ap.parse_args(argv)
    d = json.load(open(a.result))
    rows = {r["n"]: r for r in d["rows"] if r["owner"] == a.owner and not r.get("landed")}
    held: dict[int, list[str]] = {}
    if a.labels:
        n = None
        for ln in open(a.labels):
            if m := re.match(r"^== #(\d+) ", ln):
                n = int(m.group(1))
                held[n] = []
            elif n is not None and re.match(r"^\S+  (ready|reviewed)=", ln):
                held[n].append(ln)
    for n, r in rows.items():
        r["marked"], r["verdict"] = state(held[n] if n in held else labels_of(n, r["head"]))
        r["under"] = [c for c in r.get("contains", []) if c in rows and not rows[c].get("in_base")]
    todo = {n for n, r in rows.items() if not r.get("in_base")}
    order: list[int] = []
    while todo:
        free = [n for n in todo if not set(rows[n]["under"]) & todo] or sorted(todo)[:1]
        n = min(free, key=lambda n: (not (rows[n]["marked"] and rows[n]["verdict"].startswith("land")), n))
        order.append(n)
        todo.remove(n)
    base = d["summary"]["base"]
    print(f"base {base[:12]}, main {d['summary']['main'][:12]}, at {d['summary']['at_utc']}; {a.owner}: {len(rows)} open, "
          f"{len(order)} to cross, the base holds {[n for n, r in rows.items() if r.get('in_base')]}\n")
    print("| order | PR | head | mark, latest verdict | `git merge BASE` | pins commit | contains (open) |")
    print("|---|---|---|---|---|---|---|")
    for i, n in enumerate(order, 1):
        r = rows[n]
        if "error" in r or not r["base"].get("tree"):
            at = "not measured"
        elif r["base"]["files"]:
            at = "stops: " + ", ".join(f"`{f}`" for f in r["base"]["files"])
        else:
            at = "clean"
        mark = ("marked" if r["marked"] else "not marked") + ", " + (r["verdict"] or "no verdict")
        pins = ", ".join(p.rsplit("/", 1)[-1] for p in r.get("pins", {})) or "no"
        print(f"| {i} | #{n} | `{r['head'][:12]}` | {mark} | {at} | {pins} | "
              f"{', '.join('#%d' % c for c in r['under']) or 'none'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

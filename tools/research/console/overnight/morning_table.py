#!/usr/bin/env python3
"""Morning numbers: one row per overnight goal from node 1's published verity/overnight-<owner> panels (what the view reads).
Usage: morning_table.py DIR (DIR holds overnight-*.json, _last-run-node1.json and notes.json) > table.md"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PDT = timezone(timedelta(hours=-7))
SECTION = {"proofs": "Proofs", "circuits": "Circuits", "compute-accounting": "Compute accounting",
           "memory-accounting": "Memory accounting", "infra": "Infra", "pr-captain": "PRs", "console": "Console"}
PANELS = {"overhead": "verity/hillclimb-*", "pr-638": "verity/merge-trains", "gemma-stochastic": "verity/coverage-configs",
          "pearl-c-stack": "verity/merge-trains", "held-idle": "verity/node1-owner-hours, verity/node2-hours",
          "coverage": "verity/coverage-tally, -families, -causes"}


def pt(s):
    if not s:
        return ""
    t = datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(PDT)
    return t.strftime("%-I:%M %p").replace("AM", "AM").replace("PM", "PM")


def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def main(d: Path) -> None:
    notes = json.loads((d / "notes.json").read_text())["goals"]
    rows = {}
    for f in d.glob("overnight-*.json"):
        p = json.loads(f.read_text())
        rows[p["id"].split("overnight-", 1)[1]] = p["rows"]
    run = json.loads((d / "_last-run-node1.json").read_text())
    print(f"Published by node 1 at {pt(run['at'])} PDT ({run['published']} panels). One row per overnight goal, as the view reads it.\n")
    print("| Section | Goal | Done | Number | As of (PDT) | Source |\n|---|---|---|---|---|---|")
    for n in notes:
        owners = [n["owner"]] + (["console"] if n["key"] == "open-prs" else [])
        rs = [(o, r) for o in owners for r in rows.get(o, []) if r[0] == n["key"]]
        status = " ".join(r[2] for _, r in rs if r[1] in ("status", "status, continued") and r[2])
        other = "; ".join(f"{r[1]}: {r[2]}{' ' + r[3] if r[3] else ''}" for _, r in rs if r[1] not in ("status", "status, continued", "done"))
        number = "; ".join(x for x in (other, status) if x) or ""
        when = max((r[4] for _, r in rs if r[4]), default=None)
        srcs = sorted({("goals.md note" if (r[5] or "").startswith("docs/goals.md status") else f"{o}'s file") for o, r in rs})
        if n["key"] in PANELS:
            srcs.append(PANELS[n["key"]])
        done = next((r[2] for _, r in rs if r[1] == "done"), "yes" if n["done"] else "no")
        print(f"| {SECTION[n['owner']]} | {cell(n['goal'][:90])} | {done} | {cell(number)} | {pt(when)} | "
              f"{cell(', '.join(srcs))} |")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

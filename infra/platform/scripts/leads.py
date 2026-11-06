#!/usr/bin/env python3
"""Print the LEADS.json `swap all` reads, for the handles given, from what research says about each.

    leads.py HANDLE... [--dir cutover] [--budget USD] > LEADS.json

Run it in a Verity checkout with `research` on PATH, after each handle's inbox route names its platform agent id
(`research msg names open --as HANDLE --succeeds --holder $(npm run -s swap id HANDLE)`). For each handle it reads the
inbox (channel and ts) from `research msg names list --json`, and the channels and Linear team a Cursor lead subscribes
to from `research msg subscriptions --as HANDLE`. The doc is DIR/HANDLE/handoff.md, which must exist; the charter is
DIR/HANDLE/charter.md, written from the registry's remit when absent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REGISTRY = Path(os.environ.get("RESEARCH_SLACK_REGISTRY")
                or Path(__file__).resolve().parents[3] / ".agents" / "skills" / "using-slack" / "registry.json")
CHANNEL = re.compile(r'^subscribe_slack_channel\(channelId="([A-Z0-9]+)"', re.M)
TEAM = re.compile(r'^subscribe_linear_issue\(options=\{"scope": "team", "teamId": "([A-Z0-9]+)"\}', re.M)

CHARTER = """You are @{handle}: {remit}

Your checkout is verity/. Read verity/AGENTS.md, then the team, coordinating and using-slack skills in
verity/.agents/skills/, and work as a coordinator on Cursor does. Where they describe Cursor (subscriptions, timers,
Task subagents, the agent guard), the platform's instructions above take precedence.
"""


def research(*args: str) -> str:
    r = subprocess.run(["research", *args], capture_output=True, text=True)
    if r.returncode not in (0, 2):
        sys.exit(f"research {' '.join(args)}: exit {r.returncode}: {r.stderr.strip()[-300:]}")
    return r.stdout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("handles", nargs="+")
    p.add_argument("--dir", default="cutover", type=Path)
    p.add_argument("--budget", type=float, help="each lead's monthly budget in USD")
    a = p.parse_args()
    reg = {h["handle"]: h for h in json.loads(REGISTRY.read_text())["handles"]}
    routes = {r["name"]: r for r in json.loads(research("msg", "names", "list", "--json")) if not r.get("closed")}
    out, missing = [], []
    for handle in a.handles:
        if handle not in reg:
            sys.exit(f"@{handle} is not in {REGISTRY}")
        if handle not in routes:
            sys.exit(f"@{handle} has no open inbox route; open it first (research msg names open --as {handle} ...)")
        subs = research("msg", "subscriptions", "--as", handle)
        team = TEAM.search(subs)
        doc = a.dir / handle / "handoff.md"
        charter = a.dir / handle / "charter.md"
        if not doc.is_file():
            missing.append(str(doc))
        if not charter.is_file():
            charter.parent.mkdir(parents=True, exist_ok=True)
            charter.write_text(CHARTER.format(handle=handle, remit=reg[handle].get("remit") or "(no remit recorded)"))
        out.append({
            "lane": handle,
            "slackChannel": routes[handle]["channel"],
            "slackInbox": routes[handle]["inbox_ts"],
            **({"linearTeam": team[1]} if team else {}),
            "subscribe": CHANNEL.findall(subs),
            "charterFile": str(charter),
            "docFile": str(doc),
            **({"monthlyBudgetUsd": a.budget} if a.budget else {}),
        })
    if missing:
        sys.exit("no handoff at " + ", ".join(missing) + ": each Cursor lead writes its handoff first")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

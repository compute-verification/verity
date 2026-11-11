#!/usr/bin/env python3
"""One quarterback tick, from Linear only: open issues in the active projects that are overdue, due within 2 h, or on a
milestone due within 2 h; then coordinator liveness from a batch-fetch index.

Usage: PYTHONPATH=/workspace/tools/research/src qb_tick.py [RUN_DIR]
RUN_DIR is a batch-fetch-details directory (index.json). Flags ERROR/EXPIRED holders and non-IDLE holders with no message
activity for 45 min; fetch those with transcripts and run liveness.py before ringing.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from research.secret_hint import recover

recover()
from research.linear import Linear  # noqa: E402

now = datetime.now(timezone.utc)
# Linear due dates are the team's PST days: a date is late once it has ended in Los Angeles, not at UTC midnight.
local = now.astimezone(ZoneInfo("America/Los_Angeles"))
soon = local + timedelta(hours=2)
print(f"tick {now:%H:%MZ}")
lin = Linear()
projects = lin.gql('query{projects(first:50){nodes{id name status{type}}}}')["projects"]["nodes"]
active = [p for p in projects if (p.get("status") or {}).get("type") == "started"]
Q = ('query($id:String!){project(id:$id){issues(first:150,filter:{state:{type:{in:["unstarted","started"]}}}){nodes{'
     'identifier title dueDate state{name} team{key} projectMilestone{name targetDate}}}}}')
for p in active:
    rows = []
    for i in lin.gql(Q, {"id": p["id"]})["project"]["issues"]["nodes"]:
        m = i["projectMilestone"] or {}
        if (m.get("name") or "").startswith("X "):
            continue
        due = i["dueDate"]
        if due and due < f"{local:%Y-%m-%d}":
            rows.append(f"  LATE {i['identifier']} due {due} {i['state']['name']}: {i['title'][:70]}")
        elif due and due <= f"{soon:%Y-%m-%d}" and m.get("targetDate") and m["targetDate"] <= f"{soon:%Y-%m-%d}":
            rows.append(f"  DUE  {i['identifier']} {m['name'][:30]} {i['state']['name']}: {i['title'][:70]}")
    print(f"{p['name'][:60]}: {len(rows)} late or due today")
    for r in rows[:15]:
        print(r)

if len(sys.argv) > 1:
    idx = json.loads((Path(sys.argv[1]) / "index.json").read_text())
    for a in idx["agents"]:
        act = (a.get("lastMessageActivityAtMs") or 0) / 1000
        quiet = (now.timestamp() - act) / 60 if act else -1
        if a["status"] in ("ERROR", "EXPIRED"):
            print(f"  DOWN {a['name']} {a['bcId']} {a['status']}")
        elif a["status"] != "IDLE" and quiet > 45:
            print(f"  QUIET {a['name']} {a['bcId']} {a['status']} {quiet:.0f} min")
    print(f"  liveness: {len(idx['agents'])} holders read")

import subprocess as _sp
# Daniel, 4 Nov 6:23 AM PST: a goal's status must say where it stands now. A goal due within 24 h whose status is
# over 6 h old gets its owner asked, that tick, to restate it (once per goal per 6 h).
_gf = "/tmp/rot/goals_tick.json"
if _sp.run([sys.executable, "/tmp/rot/goalsapi.py", "get", _gf], capture_output=True, timeout=120).returncode == 0:
    for g in json.load(open(_gf)).get("goals", []):
        if g.get("status") == "done":
            continue
        try:
            d = datetime.fromisoformat(g["due"])
        except (KeyError, ValueError):
            continue
        d = d if d.tzinfo else d.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
        age = (now - datetime.fromisoformat(g["updatedAt"].replace("Z", "+00:00"))).total_seconds() / 3600
        if d - now < timedelta(hours=24) and age > 6:
            print(f"  STATUS-STALE {g['id']} ({g['owner']}, {g['status']}, due {g['due']}): updated {age:.1f} h ago"
                  "  <- ask its owner to restate status now")
else:
    print("  goals: fetch failed")
_pc = _sp.run([sys.executable, "/cursor/stores/self/bin/people_check.py"], capture_output=True, text=True, timeout=300)
for _ln in (_pc.stdout or "people: check failed: " + _pc.stderr[-200:]).splitlines():
    print(("  PEOPLE " if _pc.returncode else "  ") + _ln + ("  <- answer or route it first, this tick" if _pc.returncode else ""))

# Replies addressed to @top arrive as doorbells, not on the threads top subscribes to (infra's 18:52Z re-date, 3 Nov).
_db = _sp.run([sys.executable, "/cursor/stores/self/bin/doorbells.py"], capture_output=True, text=True, timeout=300)
print((_db.stdout or "  doorbells: check failed: " + _db.stderr[-200:]).rstrip())
if _db.returncode:
    print("  <- handle each DOORBELL this tick, then: python3 /cursor/stores/self/bin/doorbells.py --mark")

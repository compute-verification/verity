"""Admin hourly step (1) over a Goals API dump: python3 hourly_goals.py DUMP.json NOW_ISO."""
import json, re, sys
from datetime import datetime, timezone, timedelta
d = json.load(open(sys.argv[1])); now = datetime.fromisoformat(sys.argv[2].replace("Z", "+00:00"))
goals = {g["id"]: g for g in (d["goals"] if isinstance(d, dict) else d)}
def due(g):
    try: t = datetime.fromisoformat(g["due"])
    except Exception: return None
    return t if t.tzinfo else t.replace(tzinfo=timezone(timedelta(hours=-8)))
heads = {}
for g in goals.values():
    gd = due(g); st = g.get("status", "")
    if gd and gd < now and st not in ("done", "met", "complete", "completed"):
        print(f"PAST-DUE {g['owner']:20} {g['id']} due {g['due']} status {st}")
    for w in g.get("waitingOn", []) or []:
        wid = w if isinstance(w, str) else w.get("id") or w.get("goal")
        if wid in goals and due(goals[wid]) and gd and due(goals[wid]) > gd:
            print(f"LINK-LATER {g['owner']:20} {g['id']} ({g['due']}) waits on {wid} ({goals[wid]['due']})")
    if g.get("headline") and gd and gd <= now + timedelta(days=7):
        heads.setdefault(g["owner"], []).append(g["id"])
    desc = re.sub(r"\]\([^)]*\)", "]", g.get("description", ""))
    ids = sorted(set(i for i in re.findall(r"\b(?:[A-Z]{2,5}-\d{1,4}|#\d{3,5})\b", desc) if i != "SHA-512"))
    utc = re.findall(r"\b\d{1,2}:\d{2}Z\b|\bUTC\b", desc)
    rest = [f for f in ("Owner:", "Due:", "Status:", "**Owner", "**Due", "**Status") if f in desc]
    if ids or utc or rest:
        print(f"DESC {g['owner']:20} {g['id']}: ids {ids[:6]}{'…' if len(ids)>6 else ''} utc {utc[:3]} restated {rest}")
for o, hs in sorted(heads.items()):
    if len(hs) > 2: print(f"HEADLINES {o}: {len(hs)} due within a week: {hs}")
OPEN = ("done", "met", "complete", "completed")
waited = {(w.get("goal") if isinstance(w, dict) else w) for g in goals.values() for w in (g.get("waitingOn") or [])}
for g in goals.values():
    if g.get("status", "") in OPEN or g["id"] in waited:
        continue
    mine = {(w.get("goal") if isinstance(w, dict) else w) for w in (g.get("waitingOn") or [])}
    named = [p["id"] for p in goals.values() if p["id"] != g["id"] and p["id"] not in mine
             and f"(#{g['id']})" in (p.get("description") or "")]
    if g.get("headline") is False and named:
        print(f"UNLINKED {g['owner']:20} {g['id']}: {named[0]} names it as a part but doesn't wait on it")
for o in sorted({g["owner"] for g in goals.values()}):
    mine = [g for g in goals.values() if g["owner"] == o and g.get("status", "") not in OPEN]
    if mine and not any(due(g) and due(g) <= now + timedelta(hours=24) for g in mine):
        nxt = min(mine, key=lambda g: due(g) or now + timedelta(days=999))
        print(f"NOTHING-DUE-24H {o:20} next {nxt['id']} due {nxt['due']}")

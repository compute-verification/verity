"""A project's critical path at a glance: each open issue's state, due date, milestone, open blockers and last word, flagged
LATE (due passed), AT-RISK (milestone within 3 h and a blocker still open, or not started), QUIET (In Progress, nothing
for 2 h, or 12 h while its milestone is more than 24 h away). Usage: PYTHONPATH=/workspace/tools/research/src python3 path_check.py ["PROJECT PREFIX"] [--now ISO]"""
import sys
from datetime import datetime, timedelta, UTC

from research.linear import Linear
from research.secret_hint import recover

Q = ('query($id: String!) { project(id: $id) { name issues(first: 120, filter: {state: {type: {nin: ["completed", "canceled"]}}}) '
     '{ nodes { identifier title dueDate updatedAt startedAt '
     'state { name type } projectMilestone { name targetDate } '
     'comments(first: 1, orderBy: createdAt) { nodes { createdAt } } '
     'inverseRelations(first: 8) { nodes { type issue { identifier state { type } } } } } } } }')
TIME = __import__("re").compile(r"(\w{3}) (\d+) (\w{3}) (\d{2}):(\d{2})Z")
MONTHS = {m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}


def when(ms: dict | None, now: datetime) -> datetime | None:
    if not ms:
        return None
    if m := TIME.search(ms["name"]):
        return datetime(now.year, MONTHS[m[3]], int(m[2]), int(m[4]), int(m[5]), tzinfo=UTC)
    return datetime.fromisoformat(ms["targetDate"]).replace(tzinfo=UTC) + timedelta(hours=23, minutes=59) if ms.get("targetDate") else None


def ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def main() -> int:
    recover()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    now = datetime.fromisoformat(sys.argv[sys.argv.index("--now") + 1]) if "--now" in sys.argv else datetime.now(UTC)
    lin = Linear()
    p = lin.project(args[0] if args else "CI switch")
    data = lin.gql(Q, {"id": p["id"]})["project"]
    flagged = off = 0
    for i in sorted(data["issues"]["nodes"], key=lambda i: (i["projectMilestone"] or {}).get("name", "~")):
        if i["state"]["type"] in ("completed", "canceled"):
            continue
        if (i["projectMilestone"] or {}).get("name", "").startswith("X "):
            off += 1
            continue
        due = when(i["projectMilestone"], now)
        blockers = [r["issue"]["identifier"] for r in i["inverseRelations"]["nodes"]
                    if r["type"] == "blocks" and r["issue"]["state"]["type"] not in ("completed", "canceled")]
        last = max(filter(None, (ts((i["comments"]["nodes"] or [{}])[0].get("createdAt")), ts(i["startedAt"]))), default=None)
        flags = []
        if due and now > due:
            flags.append("LATE")
        elif due and due - now < timedelta(hours=3) and (blockers or i["state"]["type"] in ("unstarted", "backlog", "triage")):
            flags.append("AT-RISK")
        quiet_after = timedelta(hours=2 if not due or due - now < timedelta(hours=24) else 12)
        if i["state"]["name"] == "In Progress" and (not last or now - last > quiet_after):
            flags.append("QUIET")
        flagged += bool(flags)
        print(f"{' '.join(flags) or 'ok':12} {i['identifier']:9} {i['state']['name']:12} "
              f"{(due.strftime('%a %H:%MZ') if due else '-'):10} blockers {','.join(blockers) or '-':18} "
              f"last {(last.strftime('%H:%MZ') if last else '-'):6} {i['title'][:60]}")
    print(f"{data['name']}: {flagged} flagged at {now.strftime('%a %H:%MZ')}" + (f"; {off} off the path (X)" if off else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

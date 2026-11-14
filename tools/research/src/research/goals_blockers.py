"""`research goals blockers`: the open goals that say why they are off track (`blockers`), ranked by what waits on each.

A goal's goals above are the open goals, not parked, that wait on it through any number of `waitingOn` goal links; a done
goal waits on nothing, so a chain stops at it. The ranking is by the tier-1 goals among those above and the goal itself,
then by all the goals above, then by due. Each row says how many hours ago the goal was last written. With an earlier
dump, each row is `standing` (the same text then), `changed` or `new`, and a goal whose blocker is gone since is `cleared`:
done now, written with no blockers text, or no longer on the page. Texts compare exactly; the prose is never read.

Pure functions of the page's goals (`research goals show --json`) and a time. Stdlib only."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from statistics import median

from .goals import PACIFIC
from .goals_audit import _hours, _waits, due_at, parked, written

TEXT = 160


def _carries(g: dict) -> bool:
    return g.get("status") != "done" and bool(str(g.get("blockers") or "").strip())


def rows(goals: list[dict], now: datetime, owners: Iterable[str] = ()) -> list[dict]:
    """Every open goal, not parked, with a blockers text (of ``owners`` when any are given), ranked."""
    owners = set(owners)
    paused = parked(goals)
    live = {g["id"]: g for g in goals if g.get("status") != "done" and g["id"] not in paused}
    needed_by: dict[str, list[str]] = {}
    for g in live.values():
        for w in _waits(g):
            needed_by.setdefault(w, []).append(g["id"])
    out = []
    for gid, g in live.items():
        if not _carries(g) or owners and g.get("owner") not in owners:
            continue
        above: set[str] = set()
        todo = list(needed_by.get(gid, ()))
        while todo:
            k = todo.pop()
            if k != gid and k not in above:
                above.add(k)
                todo += needed_by.get(k, ())
        last = written(g)
        out.append({"id": gid, "owner": g.get("owner"), "status": g.get("status"), "due": g["due"],
                    "tier1": sorted(k for k in above | {gid} if live[k].get("tier") == 1), "above": len(above),
                    "hours": None if last is None else _hours(now - last), "blockers": g["blockers"]})
    return sorted(out, key=lambda r: (-len(r["tier1"]), -r["above"], due_at(r["due"]), r["id"]))


def since(current: list[dict], goals: list[dict], earlier: list[dict], owners: Iterable[str] = ()) -> dict:
    """Marks each of ``current`` (rows of ``goals``) `standing`, `changed` or `new` against the earlier dump, and names
    the goals of ``owners`` (all when none) that carried a blocker then and are cleared now."""
    owners = set(owners)
    then = {g["id"]: g["blockers"] for g in earlier if _carries(g) and (not owners or g.get("owner") in owners)}
    for r in current:
        r["since"] = "new" if r["id"] not in then else "standing" if then[r["id"]] == r["blockers"] else "changed"
    by = {g["id"]: g for g in goals}
    cleared = sorted(k for k in then if k not in by or not _carries(by[k]))
    counts = {m: sum(r["since"] == m for r in current) for m in ("standing", "new", "changed")}
    return counts | {"cleared": len(cleared), "cleared_ids": cleared}


def summary(current: list[dict]) -> dict:
    """How many rows there are, and the median and the oldest hours since written of those under a tier-1 goal or tier 1."""
    hours = [r["hours"] for r in current if r["tier1"] and r["hours"] is not None]
    return {"blockers": len(current), "under_tier1": sum(bool(r["tier1"]) for r in current),
            "median_hours": median(hours) if hours else None, "oldest_hours": max(hours, default=None)}


def _line(r: dict) -> str:
    text = " ".join(str(r["blockers"]).split())
    text = text if len(text) <= TEXT else text[:TEXT - 1] + "…"
    due = due_at(r["due"]).astimezone(PACIFIC).strftime("%a %-d %b")
    hours = "?" if r["hours"] is None else f"{r['hours']}h"
    mark = f"{r['since']:<9}" if "since" in r else ""
    return f"{len(r['tier1']):>3} {r['above']:>5}  {mark}{r['id']}  {r['owner']}  {r['status']}  due {due}  {hours}  {text}"


def render(shown: list[dict], total: dict, moved: dict | None = None, source: str = "") -> str:
    head = f"{'T1':>3} {'ABOVE':>5}  " + ("SINCE    " if moved is not None else "") + "GOAL  OWNER  STATUS  DUE  WRITTEN  BLOCKER"
    lines = [head, *map(_line, shown)] if shown else []
    if moved is not None:
        lines.append(f"since {source}: {moved['standing']} standing, {moved['new']} new, {moved['changed']} changed, "
                     f"{moved['cleared']} cleared" + (": " + ", ".join(moved["cleared_ids"]) if moved["cleared_ids"] else ""))
    age = (f", written a median {total['median_hours']:g} h and at most {total['oldest_hours']} h ago"
           if total["median_hours"] is not None else "")
    lines.append(f"research goals blockers: {total['blockers']} open goal(s) carry a blocker, showing {len(shown)}; "
                 f"{total['under_tier1']} under a tier-1 goal or tier 1{age}")
    return "\n".join(lines)

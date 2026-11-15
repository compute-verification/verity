"""`research goals audit`: what on the Goals page keeps a lead's plan from being read, by lead, each line with its fix.

  overdue    an open goal past its due
  stale      an open off-track goal, not parked, last written over STALE_H hours ago
  near       an open goal, not parked, due within NEAR_H hours and not written in the NEAR_H hours before its due
  unplaced   an open goal with no tier set that no goal waits on and that waits on no goal: tier 1 only by default
  order      an open goal due before an open goal it waits on, or waiting on a goal the page doesn't hold
  title      a title of more than TITLE_WORDS words
  no-goal    a lead named with --owner who owns no open goal

A date gets more precise as it nears (Daniel, 13 Nov): a day more than a week out, the day now expected inside a week,
and inside a day the hour the last step ends. So a goal is written again inside its last day, with that hour, a new day,
or `research goals confirm` when the hour stands; and an off-track goal says what is true today. A goal is written by
`set` and by `confirm`; one the page gives no `updatedAt` for is named by neither finding.

Pure functions of the page's goals (`research goals show --json`) and a time. Stdlib only."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta

from .goals import PACIFIC, _when

TITLE_WORDS = 10
STALE_H = 24
NEAR_H = 24
#: the goal whose subgoals are paused: each keeps the state it stopped in until its review
PARKED = "top/parked"
KINDS = ("overdue", "stale", "near", "unplaced", "order", "title", "no-goal")


def due_at(due: str) -> datetime:
    """A goal's due as a time: a day means its end in Pacific time, and a time carries its offset (Pacific without one)."""
    if "T" not in due:
        return datetime.combine(date.fromisoformat(due), time(23, 59, 59), PACIFIC)
    at = datetime.fromisoformat(due)
    return at if at.tzinfo else at.replace(tzinfo=PACIFIC)


def _hours(span: timedelta) -> int:
    return int(span.total_seconds() // 3600)


def _waits(goal: dict) -> list[str]:
    return [w["goal"] for w in goal.get("waitingOn") or () if isinstance(w, dict) and w.get("goal")]


def written(goal: dict) -> datetime | None:
    """When the goal was last written (UTC where it carries no offset); None where the page doesn't say."""
    if not goal.get("updatedAt"):
        return None
    at = datetime.fromisoformat(str(goal["updatedAt"]).replace("Z", "+00:00"))
    return at if at.tzinfo else at.replace(tzinfo=UTC)


def parked(goals: list[dict]) -> set[str]:
    """The parked goals' ids: PARKED and what it waits on, through other goals too, less whatever an open goal outside
    that also waits on. A goal a live goal still needs is live."""
    by = {g["id"]: g for g in goals}

    def under(start: Iterable[str]) -> set[str]:
        seen: set[str] = set()
        todo = list(start)
        while todo:
            gid = todo.pop()
            if gid in by and gid not in seen:
                seen.add(gid)
                todo += _waits(by[gid])
        return seen

    held = under([PARKED])
    return held - under(g["id"] for g in goals if g.get("status") != "done" and g["id"] not in held)


def findings(goals: list[dict], now: datetime, owners: Iterable[str] = ()) -> dict[str, dict]:
    """Each lead's open goals counted, and what the audit found on them, in KINDS' order: `{owner: {"open", "off_track",
    "findings": [{"kind", "goal", "text"}]}}` for every owner of an open goal and every one of ``owners``."""
    by = {g["id"]: g for g in goals}
    open_ = [g for g in goals if g.get("status") != "done"]
    needed = {w for g in open_ for w in _waits(g)}
    paused = parked(goals)
    out: dict[str, dict] = {o: {"open": 0, "off_track": 0, "findings": []} for o in owners}
    for g in open_:
        lead = out.setdefault(g.get("owner") or "nobody", {"open": 0, "off_track": 0, "findings": []})
        lead["open"] += 1
        lead["off_track"] += g.get("status") == "off-track"
    for owner, lead in out.items():
        mine = sorted((g for g in open_ if (g.get("owner") or "nobody") == owner), key=lambda g: due_at(g["due"]))
        found: list[tuple[str, str | None, str]] = []
        for g in mine:
            gid, at = g["id"], due_at(g["due"])
            if at < now:
                found.append(("overdue", gid, f"`{gid}` was due {_when(g['due'])}: give the date you now believe (`research goals set "
                              f"{gid} --due D`), or mark it done with its evidence (`research goals done {gid}`)"))
            last = None if gid in paused else written(g)
            if last and g.get("status") == "off-track" and now - last > timedelta(hours=STALE_H):
                found.append(("stale", gid, f"`{gid}` is off track and was last written {_hours(now - last)} hours ago: say what is "
                              f"true today and the one thing it waits on (`research goals set {gid} --blockers @why.md`, with "
                              f"`--due D` or `--status on-track` when a date now holds), or `research goals confirm {gid}` if every "
                              f"word still stands"))
            if last and now <= at <= now + timedelta(hours=NEAR_H) and last < at - timedelta(hours=NEAR_H):
                found.append(("near", gid, f"`{gid}` is due {_when(g['due'])} and was last written {_hours(at - last)} hours before "
                              f"that: set the hour you now expect it true (`research goals set {gid} --due YYYY-MM-DDTHH:MM-08:00`), "
                              f"or its new day, or `research goals confirm {gid}` if the hour stands"))
            if not g.get("tier") and gid not in needed and not _waits(g):
                found.append(("unplaced", gid, f"`{gid}` is tier 1 only because nothing waits on it: list it under its project (`research "
                              f"goals set PROJECT --wait goal:{gid}`, repeating the project's other links), or say it is a project "
                              f"(`research goals set {gid} --tier 1`)"))
            for k in _waits(g):
                other = by.get(k)
                if other is None:
                    found.append(("order", gid, f"`{gid}` waits on `{k}`, which the page doesn't hold: drop the link, or add the goal "
                                  f"(`research goals set {gid} --wait ...` replaces the list)"))
                elif other.get("status") != "done" and due_at(other["due"]) > at:
                    found.append(("order", gid, f"`{gid}` (due {_when(g['due'])}) waits on `{k}` (due {_when(other['due'])}): move one "
                                  f"date, or drop the link (`research goals set {gid} --wait ...` replaces the list)"))
            if (words := len(str(g.get("title") or "").split())) > TITLE_WORDS:
                found.append(("title", gid, f"`{gid}`'s title has {words} words: say what will be true in at most {TITLE_WORDS} "
                              f"(`research goals set {gid} --title T`)"))
        if not mine:
            found.append(("no-goal", None, "no open goal: the page says nothing of what this lead is working on; add the one to three "
                          "outcomes in hand (`research goals set ID --title T --status on-track --due D --owner HANDLE`)"))
        lead["findings"] = [{"kind": k, "goal": g, "text": t} for k, g, t in sorted(found, key=lambda f: KINDS.index(f[0]))]
    return dict(sorted(out.items()))


def tally(leads: dict[str, dict]) -> dict[str, int]:
    kinds = [f["kind"] for lead in leads.values() for f in lead["findings"]]
    return {k: kinds.count(k) for k in KINDS if k in kinds}


def render(leads: dict[str, dict]) -> str:
    lines = []
    for owner, lead in leads.items():
        lines.append(f"{owner}: {lead['open']} open, {lead['off_track']} off track" + ("" if lead["findings"] else ": nothing found"))
        lines += [f"  {f['kind']:<9} {f['text']}" for f in lead["findings"]]
    counts = tally(leads)
    hit = sum(1 for lead in leads.values() if lead["findings"])
    lines.append(f"research goals audit: {sum(counts.values())} finding(s) on {hit} of {len(leads)} lead(s)"
                 + (": " + ", ".join(f"{k} {n}" for k, n in counts.items()) if counts else ""))
    return "\n".join(lines)

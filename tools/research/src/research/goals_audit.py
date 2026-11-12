"""`research goals audit`: what on the Goals page keeps a lead's plan from being read, by lead, each line with its fix.

  overdue    an open goal past its due
  unplaced   an open goal with no tier set that no goal waits on and that waits on no goal: tier 1 only by default
  order      an open goal due before an open goal it waits on, or waiting on a goal the page doesn't hold
  title      a title of more than TITLE_WORDS words
  no-step    a lead whose open goals have none due in the next STEP_H hours
  no-goal    a lead named with --owner who owns no open goal

Pure functions of the page's goals (`research goals show --json`) and a time. Stdlib only."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta

from .goals import PACIFIC, _when

TITLE_WORDS = 8
STEP_H = 12
KINDS = ("overdue", "unplaced", "order", "title", "no-step", "no-goal")


def due_at(due: str) -> datetime:
    """A goal's due as a time: a day means its end in Pacific time, and a time carries its offset (Pacific without one)."""
    if "T" not in due:
        return datetime.combine(date.fromisoformat(due), time(23, 59, 59), PACIFIC)
    at = datetime.fromisoformat(due)
    return at if at.tzinfo else at.replace(tzinfo=PACIFIC)


def _waits(goal: dict) -> list[str]:
    return [w["goal"] for w in goal.get("waitingOn") or () if isinstance(w, dict) and w.get("goal")]


def findings(goals: list[dict], now: datetime, owners: Iterable[str] = ()) -> dict[str, dict]:
    """Each lead's open goals counted, and what the audit found on them, in KINDS' order: `{owner: {"open", "off_track",
    "findings": [{"kind", "goal", "text"}]}}` for every owner of an open goal and every one of ``owners``."""
    by = {g["id"]: g for g in goals}
    open_ = [g for g in goals if g.get("status") != "done"]
    needed = {w for g in open_ for w in _waits(g)}
    out: dict[str, dict] = {o: {"open": 0, "off_track": 0, "findings": []} for o in owners}
    for g in open_:
        lead = out.setdefault(g.get("owner") or "nobody", {"open": 0, "off_track": 0, "findings": []})
        lead["open"] += 1
        lead["off_track"] += g.get("status") == "off-track"
    soon = now + timedelta(hours=STEP_H)
    for owner, lead in out.items():
        mine = sorted((g for g in open_ if (g.get("owner") or "nobody") == owner), key=lambda g: due_at(g["due"]))
        found: list[tuple[str, str | None, str]] = []
        for g in mine:
            gid, at = g["id"], due_at(g["due"])
            if at < now:
                found.append(("overdue", gid, f"`{gid}` was due {_when(g['due'])}: give the date you now believe (`research goals set "
                              f"{gid} --due D`), or mark it done with its evidence (`research goals done {gid}`)"))
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
        if mine and not any(now <= due_at(g["due"]) <= soon for g in mine):
            found.append(("no-step", None, f"no open goal is due in the next {STEP_H} hours: write today's step as a goal of its own "
                          "(`research goals set STEP --tier 3 --due D ...`) and list it under the goal it serves (`--wait goal:STEP`)"))
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

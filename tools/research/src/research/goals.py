"""`research goals`: the Goals page (the console's /admin/goals), read and written with this agent's Cursor identity.

  research goals show [--json]               every goal by due date: when, status, title, tags, id, what it waits on
  research goals set ID [--title T] [--status on-track|off-track|done] [--due D] [--owner HANDLE] [--tag T ...]
                        [--tier N | --no-tier] [--wait goal:ID|ask:CARD ... | --no-wait]
                        [--description TEXT|@FILE] [--blockers TEXT|@FILE | --no-blockers]
                                             # adds the goal, or changes only the fields given
  research goals confirm ID                  writes it again unchanged: its owner still stands by it
  research goals done ID                     marks it done
  research goals remove ID
  research goals priority ID 1|2|3|clear     sets its priority (anyone may), or clears it back to the derived one
  research goals replace FILE.json           the whole page: exactly the goals in FILE (a JSON list)
  research goals audit [--owner HANDLE ...] [--from FILE.json] [--now TIME] [--json]
                                             # by lead: goals overdue, under no project, due before what they wait on
                                             # or titled past eight words, and a lead with nothing due in 12 hours;
                                             # each line names its fix; exit 1 when anything is found (goals_audit.py)

The plan is one graph of goals. A goal is an outcome with one owner: a headline for a title (three to six plain words,
no number; the writing-goals skill has the rule), a due date, a day meaning its end or a time with its offset such as
2026-11-05T10:00-08:00, a status (will it be true by then?), tags from the site's fixed list of areas, and a Markdown
description of what exactly will be true and where it stands, which never restates the fields. A goal under another is
its subgoal. The page derives how deep a goal sits from what waits on what: 1 when nothing waits on it, down to 3;
--tier, which keeps its old name, overrides that. What a goal waits on is links only: the narrowest goal that blocks
it, due before it, or an open ask (--wait goal:ID, --wait ask:CARD). An off-track goal says why: what it waits on, or
--blockers. The site checks every write and answers which field is wrong. --url points the command at another server;
the token's audience stays the route's.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .slack import RouteError, SlackError, _Identity, http_json, route_error

GOALS = "https://website-docs-sage.vercel.app/api/goals"   # the route, and its OIDC token's audience
PACIFIC = ZoneInfo("America/Los_Angeles")
STATUSES = ("on-track", "off-track", "done")
STORED_ONLY = ("updatedAt", "updatedBy", "wasDue", "priority", "prioritySet", "prioritySetBy", "prioritySetAt")


class Route:
    """The page's route. ``key`` is a site key with `events:read`, for a machine that holds no Cursor identity: it only
    reads. Without one every call carries this agent's identity."""

    def __init__(self, url: str = GOALS, key: str | None = None) -> None:
        self.base, self.key = url, key
        self._oidc = _Identity(GOALS)

    def call(self, body: dict | None = None) -> list[dict]:
        method = "GET" if body is None else "POST"
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if data else {}
        what = f"{method} {GOALS}"
        if self.key and body is not None:
            raise SlackError(f"{what}: a site key only reads the page; a write carries an agent's Cursor identity")

        def send(token: str) -> dict:
            return http_json(method, self.base, data, {"Authorization": f"Bearer {token}", **headers})
        try:
            out = send(self.key) if self.key else self._oidc.send(send)
        except urllib.error.HTTPError as e:
            if e.code == 401 and self.key:
                raise RouteError(f"{what}: the route rejected the site key (HTTP 401): it takes a live one with events:read", 401) from None
            if e.code == 400:
                try:
                    message = json.loads(e.read(65536)).get("message")
                except (ValueError, AttributeError):
                    message = "an answer that isn't the site's JSON"
                raise RouteError(f"{what}: refused: {message or 'a bad request'}", 400) from None
            raise RouteError(route_error(what, e), e.code) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise SlackError(f"{what}: the route is unreachable: {getattr(e, 'reason', e)}") from None
        if not isinstance(out, dict) or not isinstance(out.get("goals"), list):
            raise SlackError(f"{what}: the answer has no goals list")
        return out["goals"]


def _when(due: str) -> str:
    if "T" not in due:
        return datetime.fromisoformat(due).strftime("%a %-d %b") + ", end of day"
    return datetime.fromisoformat(due).astimezone(PACIFIC).strftime("%a %-d %b, %-I:%M %p")


def _text(value: str) -> str:
    """TEXT, or a file's contents for @FILE (@- reads stdin)."""
    if not value.startswith("@"):
        return value
    return sys.stdin.read() if value == "@-" else Path(value[1:]).read_text()


def _clean(goal: dict) -> dict:
    return {k: v for k, v in goal.items() if k not in STORED_ONLY}


def _find(goals: list[dict], gid: str) -> dict | None:
    return next((g for g in goals if g.get("id") == gid), None)


def show(goals: list[dict]) -> str:
    lines = []
    for g in goals:
        flag = "  [blocked]" if g.get("blockers") else ""
        was = f"  (was {_when(g['wasDue'])})" if g.get("wasDue") else ""
        tier = f"  (tier {g['tier']})" if g.get("tier") else ""
        waits = "".join(f"\n{'':<24} {'':<9}    waits on {next(iter(r.values()))}" for r in g.get("waitingOn", []))
        lines.append(f"{_when(g['due']):<24} {g['status']:<9}  {g['title']}  {' '.join(g.get('tags', []))}{tier}{flag}{was}\n"
                     f"{'':<24} {'':<9}  {g['id']}{waits}")
    return "\n".join(lines) if lines else "No goals yet."


def _link(value: str) -> dict:
    """goal:ID or ask:CARD, as a waitingOn link."""
    kind, _, target = value.partition(":")
    if kind not in ("goal", "ask") or not target:
        raise SlackError(f"--wait {value!r}: expected goal:ID or ask:CARD")
    return {kind: target}


def edited(current: dict | None, ns: argparse.Namespace) -> dict:
    """The goal `set` writes: the current one with the given fields changed, or a new one from them."""
    goal = _clean(current or {"id": ns.id})
    for field in ("title", "status", "due"):
        if getattr(ns, field) is not None:
            goal[field] = getattr(ns, field)
    if ns.tag:
        goal["tags"] = ns.tag
    if ns.owner is not None:
        goal["owner"] = ns.owner
    if ns.no_tier:
        goal.pop("tier", None)
    elif ns.tier is not None:
        goal["tier"] = ns.tier
    if ns.no_wait:
        goal.pop("waitingOn", None)
    elif ns.wait:
        goal["waitingOn"] = [_link(w) for w in ns.wait]
    if ns.description is not None:
        goal["description"] = _text(ns.description)
    if ns.no_blockers:
        goal.pop("blockers", None)
    elif ns.blockers is not None:
        goal["blockers"] = _text(ns.blockers)
    missing = [f for f in ("title", "status", "due") if not goal.get(f)]
    if missing:
        raise SlackError(f"goal {ns.id} is new, so it needs --{', --'.join(missing)}")
    goal.setdefault("tags", [])
    return goal


def main(argv: list[str], route: Route | None = None) -> int:
    p = argparse.ArgumentParser(prog="research goals", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=GOALS, help="the route (default: the console's)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("set")
    s.add_argument("id")
    s.add_argument("--title")
    s.add_argument("--status", choices=STATUSES)
    s.add_argument("--due")
    s.add_argument("--tag", action="append")
    s.add_argument("--owner", help="the lead that keeps it true, by handle")
    level = s.add_mutually_exclusive_group()
    level.add_argument("--tier", type=int, choices=range(1, 4), metavar="N",
                       help="how deep the goal sits, overriding what the plan derives: 1 a goal nothing waits on, 2 its "
                            "subgoal, 3 a subgoal of that")
    level.add_argument("--no-tier", action="store_true", help="the depth the plan derives from what needs it")
    s.add_argument("--description", help="TEXT, or @FILE")
    why = s.add_mutually_exclusive_group()
    why.add_argument("--blockers", help="why it's off track, when what it waits on doesn't say: TEXT, or @FILE")
    why.add_argument("--no-blockers", action="store_true")
    waits = s.add_mutually_exclusive_group()
    waits.add_argument("--wait", action="append", metavar="goal:ID|ask:CARD", help="what it waits on; repeat for each, replacing the list")
    waits.add_argument("--no-wait", action="store_true", help="it waits on nothing")
    for name in ("confirm", "done", "remove"):
        sub.add_parser(name).add_argument("id")
    pr = sub.add_parser("priority", help="set a goal's priority (anyone may), or clear it back to the derived one")
    pr.add_argument("id")
    pr.add_argument("level", choices=["1", "2", "3", "clear"])
    sub.add_parser("replace").add_argument("file")
    au = sub.add_parser("audit", help="by lead, what keeps the plan from being read, each line with its fix (goals_audit.py)")
    au.add_argument("--owner", action="append", default=[], metavar="HANDLE",
                    help="only this lead's goals, and a line when it owns no open goal; repeat for more (default: every owner)")
    au.add_argument("--from", dest="file", metavar="FILE.json", help="a saved `research goals show --json` instead of the page")
    au.add_argument("--now", metavar="TIME", help="the time to audit at, with its offset (default: now)")
    au.add_argument("--json", action="store_true")
    ns = p.parse_args(argv)
    route = route or Route(ns.url)
    try:
        if ns.cmd == "show":
            goals = route.call()
            print(json.dumps(goals, indent=1, ensure_ascii=False) if ns.json else show(goals))
            return 0
        if ns.cmd == "audit":
            from . import goals_audit

            goals = json.loads(Path(ns.file).read_text()) if ns.file else route.call()
            if not isinstance(goals, list):
                raise SlackError(f"{ns.file}: expected a JSON list of goals, as `research goals show --json` prints")
            now = datetime.fromisoformat(ns.now) if ns.now else datetime.now(UTC)
            if now.tzinfo is None:
                raise SlackError(f"--now {ns.now}: give its offset, as in 2026-11-05T10:00-08:00")
            leads = goals_audit.findings(goals, now, ns.owner)
            leads = {o: v for o, v in leads.items() if not ns.owner or o in ns.owner}
            print(json.dumps({"now": now.isoformat(), "leads": leads, "counts": goals_audit.tally(leads)}, indent=1, ensure_ascii=False)
                  if ns.json else goals_audit.render(leads))
            return 1 if goals_audit.tally(leads) else 0
        if ns.cmd == "replace":
            goals = json.loads(Path(ns.file).read_text())
            if not isinstance(goals, list):
                raise SlackError(f"{ns.file}: expected a JSON list of goals")
            out = route.call({"replace": [_clean(g) for g in goals]})
            print(f"research goals: the page holds {len(out)} goals")
            return 0
        if ns.cmd == "remove":
            route.call({"remove": ns.id})
            print(f"research goals: removed {ns.id}")
            return 0
        if ns.cmd == "priority":
            route.call({"priority": {"id": ns.id, "priority": None if ns.level == "clear" else int(ns.level)}})
            print(f"research goals: {ns.id}'s priority is {'derived again' if ns.level == 'clear' else ns.level}, set by this agent")
            return 0
        current = _find(route.call(), ns.id)
        if ns.cmd == "set":
            goal = edited(current, ns)
        elif current is None:
            raise SlackError(f"no goal {ns.id} (research goals show lists them)")
        else:
            goal = _clean(current) | ({"status": "done"} if ns.cmd == "done" else {})
        route.call({"set": goal})
        print(f"research goals: {ns.cmd} {ns.id}: {goal['status']}, due {_when(goal['due'])}")
        return 0
    except (SlackError, OSError, ValueError) as e:
        print(f"research goals: {e}", file=sys.stderr)
        return 1

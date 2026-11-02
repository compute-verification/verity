"""`research goals`: Daniel's Goals page (the console's /admin/goals), read and written with this agent's Cursor identity.

  research goals show [--json]               every goal by due date: when, status, title, tags, id
  research goals set ID [--title T] [--status on-track|off-track|done] [--due D] [--tag T ...]
                        [--description TEXT|@FILE] [--blockers TEXT|@FILE | --no-blockers]
                                             # adds the goal, or changes only the fields given
  research goals confirm ID                  writes it again unchanged: its owner still stands by it
  research goals done ID                     marks it done
  research goals remove ID
  research goals replace FILE.json           the whole page: exactly the goals in FILE (a JSON list)

A goal is an outcome Daniel would recognize and could check himself: a one-line title of at most eight words, a status
(will it be true by its due date?), a due date (2026-11-05, the end of that day, or 2026-11-05T10:00-08:00), lowercase
tags (demos, proofs, ci, infra, cleanup, accounting, pm), and Markdown for him: a description of what will be true and how
to check it, and blockers only while something blocks it (a link to "#<id>" opens that goal). The site checks every write
and answers which field is wrong. $RESEARCH_GOALS_URL points the command at another server (the tests' fake).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .slack import RouteError, SlackError, _Identity, http_json, route_error

GOALS = "https://website-docs-sage.vercel.app/api/goals"   # the route, and its OIDC token's audience
PACIFIC = ZoneInfo("America/Los_Angeles")
STATUSES = ("on-track", "off-track", "done")
STORED_ONLY = ("updatedAt", "updatedBy", "wasDue")


class Route:
    def __init__(self) -> None:
        self.base = os.environ.get("RESEARCH_GOALS_URL") or GOALS
        self._oidc = _Identity(GOALS)

    def call(self, body: dict | None = None) -> list[dict]:
        method = "GET" if body is None else "POST"
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if data else {}
        what = f"{method} {GOALS}"
        try:
            out = self._oidc.send(lambda t: http_json(method, self.base, data, {"Authorization": f"Bearer {t}", **headers}))
        except urllib.error.HTTPError as e:
            if e.code == 400:
                try:
                    message = json.loads(e.read(65536)).get("message")
                except (ValueError, AttributeError):
                    message = None
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
        lines.append(f"{_when(g['due']):<24} {g['status']:<9}  {g['title']}  {' '.join(g.get('tags', []))}{flag}{was}\n"
                     f"{'':<24} {'':<9}  {g['id']}")
    return "\n".join(lines) if lines else "No goals yet."


def edited(current: dict | None, ns: argparse.Namespace) -> dict:
    """The goal `set` writes: the current one with the given fields changed, or a new one from them."""
    goal = _clean(current or {"id": ns.id})
    for field in ("title", "status", "due"):
        if getattr(ns, field) is not None:
            goal[field] = getattr(ns, field)
    if ns.tag:
        goal["tags"] = ns.tag
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
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("set")
    s.add_argument("id")
    s.add_argument("--title")
    s.add_argument("--status", choices=STATUSES)
    s.add_argument("--due")
    s.add_argument("--tag", action="append")
    s.add_argument("--description", help="TEXT, or @FILE")
    blockers = s.add_mutually_exclusive_group()
    blockers.add_argument("--blockers", help="TEXT, or @FILE")
    blockers.add_argument("--no-blockers", action="store_true")
    for name in ("confirm", "done", "remove"):
        sub.add_parser(name).add_argument("id")
    sub.add_parser("replace").add_argument("file")
    ns = p.parse_args(argv)
    route = route or Route()
    try:
        if ns.cmd == "show":
            goals = route.call()
            print(json.dumps(goals, indent=1, ensure_ascii=False) if ns.json else show(goals))
            return 0
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

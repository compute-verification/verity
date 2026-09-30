"""The Scheduler's PR rows, as an in-memory model: what the site's `/api/prs` does (docs/pr-ownership-plan.md, slice 1).

Every PR has one row: its owner (a coordinator and a lane), the owner's blocker with a reference, and when the owner last touched
it.  The `prs-sync` cron (`sync`) lists the repository's PRs through the App.  A PR it hasn't seen gets a row, with the owner read
once from its body, by the first of:
    an `Owner: {coordinator}/{lane}` line      the coordinator must be in `coordinators`, else the line doesn't count
    a `Lane {name}` line                       the coordinator whose `lanes` list it, else DEFAULT_COORDINATOR
    a Cursor agent id (the PR's footer)        the coordinator whose `agents` list it, with no lane
A PR with none of these is unowned.  After that, the row wins over the body: `POST /api/prs/{n}` with `owner` changes it.  A new
head counts as owner activity, and a PR that's no longer open becomes `closed` or `merged`.

    GET  /api/prs            ?coordinator= ?lane= ?state= ?limit=    any valid token; open PRs, unless ?state= asks for closed or merged
    GET  /api/prs/{n}        the row, with its events                any valid token
    POST /api/prs/{n}        {owner?, blocker?, ref?, renew?}        prs:write; each is owner activity
    POST /api/prs/{n}/close  {reason}                                prs:write, acting as the owner's coordinator (UNOWNED_TO's for
                                                                     an unowned PR)

A coordinator acts through the token names its row lists, and every write is stamped with the token's name.  Closing goes through the
App: a comment `Closed by {coordinator} ({token}): {reason}`, then the PR closed.  Nothing here closes a PR by itself.

States, in this slice: `unowned`; `stale` after STALE_S without owner activity; `owned` otherwise; and `closed` or `merged`.  Stage 1.5's
reconcile refines `owned` into the queue's facts (held, not-ready, waiting, ready, in-train, failed), stops the clock while the queue
acts on a PR, and sends the clock's notices.  `pr_vectors.json` beside this pins it; `tests/test_jobs_vectors.py` runs it here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..queue import AGENT_RE, LANE_RE, OWNER_RE
from .model import Refused, Token, _limit
from .trains import TrainService

STALE_S = 86400
DEFAULT_COORDINATOR = "rc"            # a `Lane` line's lane that no coordinator lists is the research coordinator's
UNOWNED_TO = "rc"
BLOCKER_MAX, REF_MAX, REASON_MAX = 500, 200, 500
PR_STATES = ("unowned", "owned", "stale", "closed", "merged")
OWNER_VALUE = re.compile(r"([a-z0-9][a-z0-9-]*)/([A-Za-z0-9][A-Za-z0-9._-]*)")
WRITE_KEYS = ("owner", "blocker", "ref", "renew")


@dataclass
class PRRow:
    number: int
    head: str
    title: str
    coordinator: str | None
    lane: str | None
    owner_from: str | None             # body | lane | agent | set
    first_seen: float
    touched_at: float
    touched_by: str | None = None      # the token's name, or None for a push
    blocker: str | None = None
    ref: str | None = None
    blocker_by: str | None = None
    blocker_at: float | None = None
    closed: str | None = None          # closed | merged
    closed_at: float | None = None
    closed_by: str | None = None
    close_reason: str | None = None


@dataclass
class PRService(TrainService):
    #: name -> {"lanes": [...], "agents": [...], "tokens": [...]}: the `coordinators` table
    coordinators: dict[str, dict] = field(default_factory=dict)
    rows: dict[int, PRRow] = field(default_factory=dict)

    def _route(self, method, path, query, token, body, now):
        if method == "GET" and path == "/api/prs":
            self._auth_any(token)
            state = query.get("state")
            if state and state not in PR_STATES:
                raise Refused(400, f"state must be one of {', '.join(PR_STATES)}")
            rows = [r for r in self.rows.values() if (self.state(r, now) == state if state else r.closed is None)
                    and (not query.get("coordinator") or r.coordinator == query["coordinator"])
                    and (not query.get("lane") or r.lane == query["lane"])]
            return 200, {"prs": [self.view(r, now) for r in sorted(rows, key=lambda r: -r.number)[:_limit(query)]]}
        m = re.fullmatch(r"/api/prs/([^/]+)(/close)?", path)
        if m and method == "GET" and m.group(2) is None:
            self._auth_any(token)
            r = self._row(m.group(1))
            return 200, {**self.view(r, now), "events": [e for e in self.events if e["subject"] == f"pr:{r.number}"]}
        if m and method == "POST" and m.group(2) is None:
            return self.write(self._auth(token, "prs:write"), self._row(m.group(1)), body, now)
        if m and method == "POST":
            return self.close(self._auth(token, "prs:write"), self._row(m.group(1)), body, now)
        return super()._route(method, path, query, token, body, now)

    # -- the owner, from a body

    def owner_of(self, body: str) -> tuple[str | None, str | None, str | None]:
        """(coordinator, lane, where it came from), or (None, None, None) for an unowned PR."""
        for m in OWNER_RE.finditer(body):
            if m.group(1) in self.coordinators:
                return m.group(1), m.group(2), "body"
        if m := LANE_RE.search(body):
            lane = m.group(1)
            return next((c for c, row in self.coordinators.items() if lane in row.get("lanes", ())), DEFAULT_COORDINATOR), lane, "lane"
        for m in AGENT_RE.finditer(body):
            if c := next((c for c, row in self.coordinators.items() if m.group(0) in row.get("agents", ())), None):
                return c, None, "agent"
        return None, None, None

    def acting(self, t: Token) -> str | None:
        return next((c for c, row in self.coordinators.items() if t.name in row.get("tokens", ())), None)

    # -- the prs-sync cron

    def sync(self, now: float) -> list[int]:
        """New PRs get a row; a new head is owner activity; a PR no longer open is closed or merged.  Returns what changed."""
        changed = []
        for n, p in sorted(self.github.prs.items()):
            r = self.rows.get(n)
            if p["state"] == "open":
                if r is None:
                    c, lane, where = self.owner_of(p.get("body") or "")
                    self.rows[n] = PRRow(n, p["head"], p.get("title") or "", c, lane, where, now, now)
                    self._event(now, "pr.seen", f"pr:{n}", "cron", {"owner": _owner(c, lane), "from": where})
                elif r.closed is not None:
                    r.closed = r.closed_at = r.closed_by = r.close_reason = None
                    r.head, r.touched_at, r.touched_by = p["head"], now, None
                    self._event(now, "pr.reopened", f"pr:{n}", "cron", {"head": p["head"]})
                elif r.head != p["head"]:
                    r.head, r.touched_at, r.touched_by = p["head"], now, None
                    self._event(now, "pr.pushed", f"pr:{n}", "cron", {"head": p["head"]})
                else:
                    continue
            elif r is not None and r.closed is None:
                r.closed, r.closed_at = p["state"], now
                self._event(now, f"pr.{p['state']}", f"pr:{n}", "cron", {})
            else:
                continue
            changed.append(n)
        return changed

    # -- the routes

    def state(self, r: PRRow, now: float) -> str:
        if r.closed:
            return r.closed
        if r.coordinator is None:
            return "unowned"
        return "stale" if now - r.touched_at >= STALE_S else "owned"

    def view(self, r: PRRow, now: float) -> dict:
        return {"number": r.number, "head": r.head, "title": r.title, "state": self.state(r, now), "owner": _owner(r.coordinator, r.lane),
                "coordinator": r.coordinator, "lane": r.lane, "owner_from": r.owner_from, "blocker": r.blocker, "ref": r.ref,
                "blocker_by": r.blocker_by, "blocker_at": self.iso(r.blocker_at), "touched_at": self.iso(r.touched_at),
                "touched_by": r.touched_by, "first_seen_at": self.iso(r.first_seen), "closed_at": self.iso(r.closed_at),
                "closed_by": r.closed_by, "close_reason": r.close_reason}

    def _row(self, n: str) -> PRRow:
        r = self.rows.get(int(n)) if n.isdigit() else None
        if r is None:
            raise Refused(404, "no such PR: the site hasn't seen it open")
        return r

    def write(self, t: Token, r: PRRow, body: dict, now: float) -> tuple[int, dict]:
        if set(body) - set(WRITE_KEYS):
            raise Refused(400, f"the body may only name {', '.join(WRITE_KEYS)}")
        if not {"owner", "blocker", "renew"} & set(body):
            raise Refused(400, "name at least one of owner, blocker or renew")
        if "ref" in body and "blocker" not in body:
            raise Refused(400, "ref goes with blocker")
        if "renew" in body and body["renew"] is not True:
            raise Refused(400, "renew must be true")
        owner = body.get("owner")
        if "owner" in body:
            if not isinstance(owner, str) or not (m := OWNER_VALUE.fullmatch(owner)):
                raise Refused(400, "owner must be {coordinator}/{lane}")
            if m.group(1) not in self.coordinators:
                raise Refused(400, f"no coordinator {m.group(1)}")
        blocker, ref = body.get("blocker"), body.get("ref")
        if "blocker" in body and blocker is not None and not (isinstance(blocker, str) and 0 < len(blocker.strip()) <= BLOCKER_MAX):
            raise Refused(400, f"blocker must be null or 1 to {BLOCKER_MAX} characters")
        if ref is not None and not (isinstance(ref, str) and 0 < len(ref) <= REF_MAX):
            raise Refused(400, f"ref must be null or 1 to {REF_MAX} characters")
        if r.closed:
            raise Refused(409, f"#{r.number} is {r.closed}")
        n = f"pr:{r.number}"
        if "owner" in body:
            was = _owner(r.coordinator, r.lane)
            r.coordinator, r.lane = owner.split("/")
            r.owner_from = "set"
            self._event(now, "pr.owner", n, t.name, {"owner": owner, "was": was})
        if "blocker" in body:
            r.blocker, r.ref = (blocker.strip(), ref) if blocker is not None else (None, None)
            r.blocker_by, r.blocker_at = (t.name, now) if blocker is not None else (None, None)
            self._event(now, "pr.blocker", n, t.name, {"blocker": r.blocker, "ref": r.ref})
        if body.get("renew"):
            self._event(now, "pr.renewed", n, t.name, {})
        r.touched_at, r.touched_by = now, t.name
        return 200, self.view(r, now)

    def close(self, t: Token, r: PRRow, body: dict, now: float) -> tuple[int, dict]:
        reason = body.get("reason")
        if not (isinstance(reason, str) and 0 < len(reason.strip()) <= REASON_MAX):
            raise Refused(400, f"reason is needed, 1 to {REASON_MAX} characters: why it's closed (superseded by #M, …)")
        if r.closed:
            raise Refused(409, f"#{r.number} is {r.closed}")
        acting = self.acting(t)
        if acting is None:
            raise Refused(403, f"the token {t.name} acts for no coordinator")
        owner = r.coordinator or UNOWNED_TO
        if acting != owner:
            raise Refused(403, f"#{r.number} is {owner}'s to close; take it first with `research pr own {r.number} {acting}/{{lane}}`")
        p = self.github.prs[r.number]
        if p["state"] != "open":
            raise Refused(409, f"#{r.number} is {p['state']} on GitHub")
        self.github.comments.append({"pr": r.number, "body": f"Closed by {acting} ({t.name}): {reason.strip()}"})
        p["state"] = "closed"
        r.closed, r.closed_at, r.closed_by, r.close_reason = "closed", now, t.name, reason.strip()
        r.touched_at, r.touched_by = now, t.name
        self._event(now, "pr.closed", f"pr:{r.number}", t.name, {"reason": r.close_reason, "as": acting})
        return 200, self.view(r, now)


def _owner(coordinator: str | None, lane: str | None) -> str | None:
    return None if coordinator is None else f"{coordinator}/{lane}" if lane else coordinator

"""Asks as Linear issues (the trial from 8 Oct): `research msg ask` (and `send --to` one handle) files the ask in its owner's
team, the registry's `linear` key, and rings on Slack with the issue's link; `close` and `close --declined` move the issue to done or
canceled.

Credentials come from the environment only. LINEAR_CLIENT_ID and LINEAR_CLIENT_SECRET are an OAuth app's client-credentials
grant (never the authorization-code flow): an `app` actor, so each issue and comment shows as filed by the asking handle
(`createAsUser`). Without them, LINEAR_API_KEY is a person's key. Either way the acting user is unsubscribed from each issue
after filing it and after each comment (a comment subscribes its author again), so a person's inbox stays quiet. Either way an issue carries a `from:<handle>` label and
is never assigned or delegated: an issue assigned to the Cursor app starts a new agent instead of waking the lead. Stdlib only."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

API = "https://api.linear.app/graphql"
TOKEN_URL = "https://api.linear.app/oauth/token"
SCOPES = "read,write"   # always the same: a grant with other scopes revokes every live token of the app
EARLY_S = 3600          # a cached token is refetched this long before it expires
TITLE_MAX = 80

Transport = Callable[[str, bytes, dict], dict]


class LinearError(RuntimeError):
    pass


def _http(url: str, data: bytes, headers: dict, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _why(e: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(e.read() or b"{}")
    except ValueError:
        return f"HTTP {e.code}"
    msgs = [x.get("message", "?") for x in body.get("errors", [])] or [body.get("error_description") or body.get("error") or ""]
    return f"HTTP {e.code}: {'; '.join(m for m in msgs if m)}".rstrip(": ")


def token_cache() -> Path:
    return Path(os.environ.get("RESEARCH_LINEAR_TOKEN_CACHE") or Path.home() / ".research" / "linear-token.json")


def title(body: str) -> str:
    line = next((s.strip() for s in body.splitlines() if s.strip()), "(no text)")
    return line if len(line) <= TITLE_MAX else line[:TITLE_MAX - 1].rstrip() + "\u2026"


class Linear:
    def __init__(self, env: dict | None = None, transport: Transport | None = None, now: Callable[[], float] = time.time,
                 cache: Path | None = None):
        env = os.environ if env is None else env
        self.client, self.secret, self.key = env.get("LINEAR_CLIENT_ID"), env.get("LINEAR_CLIENT_SECRET"), env.get("LINEAR_API_KEY")
        self.app = bool(self.client and self.secret)
        if not self.app and not self.key:
            raise LinearError("no Linear credentials: LINEAR_CLIENT_ID and LINEAR_CLIENT_SECRET (the app), or LINEAR_API_KEY")
        self.http, self.now, self.cache = transport or _http, now, cache or token_cache()

    @property
    def mode(self) -> str:
        return "app (client credentials)" if self.app else "personal key"

    def _token(self, fresh: bool) -> str:
        tag = hashlib.sha256(self.client.encode()).hexdigest()[:16]
        if not fresh:
            try:
                c = json.loads(self.cache.read_text())
                if c["client"] == tag and c["scope"] == SCOPES and c["expires_at"] - EARLY_S > self.now():
                    return c["access_token"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
        basic = base64.b64encode(f"{self.client}:{self.secret}".encode()).decode()
        form = urllib.parse.urlencode({"grant_type": "client_credentials", "scope": SCOPES}).encode()
        try:
            r = self.http(TOKEN_URL, form, {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"})
        except urllib.error.HTTPError as e:
            raise LinearError(f"Linear refused the app's client-credentials grant ({_why(e)}): check LINEAR_CLIENT_ID, "
                              f"LINEAR_CLIENT_SECRET and that the app allows client credentials") from None
        token = r.get("access_token")
        if not isinstance(token, str) or not token:
            raise LinearError("Linear's client-credentials grant returned no access_token")
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.cache, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"client": tag, "scope": SCOPES, "access_token": token,
                       "expires_at": self.now() + float(r.get("expires_in") or 0)}, f)
        return token

    def gql(self, query: str, variables: dict | None = None) -> dict:
        body = json.dumps({"query": query, "variables": variables or {}}).encode()
        for fresh in (False, True):
            auth = f"Bearer {self._token(fresh)}" if self.app else self.key
            try:
                r = self.http(API, body, {"Authorization": auth, "Content-Type": "application/json"})
                break
            except urllib.error.HTTPError as e:
                if e.code == 401 and self.app and not fresh:     # a cached token Linear no longer takes (revoked, rotated)
                    continue
                raise LinearError(f"Linear: {_why(e)}") from None
        if r.get("errors"):
            raise LinearError("Linear: " + "; ".join(x.get("message", "?") for x in r["errors"]))
        return r["data"]

    def team(self, name: str) -> dict:
        """The team named `name` (a handle), or whose key it is."""
        q = "query($n: String!) { teams(filter: {or: [{name: {eq: $n}}, {key: {eq: $n}}]}) { nodes { id key name } } }"
        nodes = self.gql(q, {"n": name})["teams"]["nodes"]
        if not nodes:
            raise LinearError(f"no Linear team named {name!r}")
        return nodes[0]

    def label(self, name: str) -> str:
        nodes = self.gql("query($n: String!) { issueLabels(filter: {name: {eq: $n}}) { nodes { id } } }", {"n": name})["issueLabels"]["nodes"]
        if nodes:
            return nodes[0]["id"]
        made = self.gql("mutation($i: IssueLabelCreateInput!) { issueLabelCreate(input: $i) { issueLabel { id } } }",
                        {"i": {"name": name}})
        return made["issueLabelCreate"]["issueLabel"]["id"]

    def _quiet(self, issue_id: str) -> None:
        try:
            self.gql("mutation($id: String!) { issueUnsubscribe(id: $id) { success } }", {"id": issue_id})
        except LinearError:
            if not self.app:                                  # a person's inbox must stay quiet; an app's has nobody to bother
                raise

    def file(self, team: str, sender: str, body: str) -> dict:
        """The ask as an issue in `team`, from `sender`: {id, identifier, url}."""
        i = {"teamId": self.team(team)["id"], "title": title(body), "description": body, "labelIds": [self.label(f"from:{sender}")]}
        if self.app:
            i["createAsUser"] = sender
        issue = self.gql("mutation($i: IssueCreateInput!) { issueCreate(input: $i) { issue { id identifier url } } }",
                         {"i": i})["issueCreate"]["issue"]
        self._quiet(issue["id"])
        return issue

    def comment(self, issue_id: str, sender: str, body: str) -> None:
        i = {"issueId": issue_id, "body": body if self.app else f"@{sender}: {body}"}
        if self.app:
            i["createAsUser"] = sender
        self.gql("mutation($i: CommentCreateInput!) { commentCreate(input: $i) { success } }", {"i": i})
        self._quiet(issue_id)

    def move(self, issue_id: str, state: str) -> str:
        """Move the issue (its id, or an identifier such as `INF-12`) to its team's state named `state` (any case), else to the
        team's first state of that type (`completed`, `canceled`); that state's name. By name, `In Progress` is never the
        `started` state that happens to come first (In Review)."""
        issue = self.gql("query($id: String!) { issue(id: $id) { id team { states { nodes { id name type } } } } }",
                         {"id": issue_id})["issue"]
        nodes = issue["team"]["states"]["nodes"]
        to = (next((s for s in nodes if s["name"].lower() == state.lower()), None)
              or next((s for s in nodes if s["type"] == state), None))
        if to is None:
            raise LinearError(f"the issue's team has no state {state!r}: its states are {', '.join(s['name'] for s in nodes)}")
        self.gql("mutation($id: String!, $s: String!) { issueUpdate(id: $id, input: {stateId: $s}) { success } }",
                 {"id": issue["id"], "s": to["id"]})
        return to["name"]

    def issue_id(self, issue: str) -> str:
        """The id of the issue an identifier such as `INF-12` names."""
        return self.gql("query($id: String!) { issue(id: $id) { id } }", {"id": issue})["issue"]["id"]

    def issues(self, team: str, state: str | None = None) -> list[dict]:
        """The team's open issues, or those in the state named `state`, oldest first: {identifier, title, state: {name}}."""
        f = {"team": {"id": {"eq": self.team(team)["id"]}},
             "state": {"name": {"eqIgnoreCase": state}} if state else {"type": {"nin": ["completed", "canceled"]}}}
        q = "query($f: IssueFilter!) { issues(filter: $f, first: 250) { nodes { identifier title createdAt state { name } } } }"
        return sorted(self.gql(q, {"f": f})["issues"]["nodes"], key=lambda i: i["createdAt"])

    def subscribers(self, issue_id: str) -> list[str]:
        q = "query($id: String!) { issue(id: $id) { subscribers { nodes { name } } } }"
        return [n["name"] for n in self.gql(q, {"id": issue_id})["issue"]["subscribers"]["nodes"]]

    def authors(self, issue_id: str) -> tuple[str, list[str]]:
        """Who the issue and each of its comments show as filed by: an app's `createAsUser` name, else the user's."""
        q = ("query($id: String!) { issue(id: $id) { creator { name } botActor { name userDisplayName } "
             "comments { nodes { user { name } botActor { name userDisplayName } } } } }")
        i = self.gql(q, {"id": issue_id})["issue"]

        def who(x: dict) -> str:
            bot, user = x.get("botActor") or {}, x.get("creator") or x.get("user") or {}
            return f"{bot.get('userDisplayName')} (via {bot.get('name')})" if bot.get("userDisplayName") else user.get("name") or "?"
        return who(i), [who(c) for c in i["comments"]["nodes"]]

    def check(self, write_team: str | None = None, sender: str = "comms") -> tuple[list[str], bool]:
        """What these credentials reach, as lines, and whether they're fit for asks: no admin scope, and (with `write_team`)
        a test issue filed, commented on and canceled there."""
        d = self.gql("{ viewer { name } organization { urlKey } teams(first: 100) { nodes { name key } } }")
        lines = [f"credentials: {self.mode}, acting as {d['viewer']['name']} in {d['organization']['urlKey']}",
                 f"teams ({len(d['teams']['nodes'])}): " + ", ".join(f"{t['name']} {t['key']}"
                                                                     for t in sorted(d["teams"]["nodes"], key=lambda t: t["key"]))]
        try:
            self.gql("{ webhooks(first: 1) { nodes { id } } }")
            lines.append("admin: HAS ADMIN SCOPE (webhooks listed); asks need read and write only")
            ok = False
        except LinearError as e:
            ok = "admin" in str(e).lower()
            lines.append(f"admin: none ({e})" if ok else f"admin probe failed for another reason: {e}")
        if write_team:
            issue = self.file(write_team, sender, f"[{sender}] Linear check: a test issue, canceled at once")
            subs = self.subscribers(issue["id"])
            self.comment(issue["id"], sender, "a test comment")
            subs_after = self.subscribers(issue["id"])
            by, comments_by = self.authors(issue["id"])
            lines += [f"filed {issue['identifier']} {issue['url']} as {by} (subscribers: {', '.join(subs) or 'none'})",
                      f"commented as {', '.join(comments_by) or '?'} (subscribers: {', '.join(subs_after) or 'none'})",
                      f"moved to {self.move(issue['id'], 'canceled')}"]
        return lines, ok


def ask_issue(owner: str, sender: str, body: str, linear: Any = None) -> tuple[dict | None, str | None]:
    """`msg ask`'s issue: (issue, None), or (None, why) when it couldn't be filed, so the ask still goes out on Slack."""
    try:
        return (linear or Linear()).file(owner, sender, body), None
    except (LinearError, OSError, ValueError) as e:
        return None, str(e)

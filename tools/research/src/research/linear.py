"""Asks as Linear issues: `research msg ask` (and `send --to` one handle) files the ask in its owner's
team, the registry's `linear` key, and rings on Slack with the issue's link; `close` and `close --declined` move the issue to done or
canceled.

Credentials come from the environment only. LINEAR_CLIENT_ID and LINEAR_CLIENT_SECRET are an OAuth app's client-credentials
grant (never the authorization-code flow): an `app` actor, so each issue and comment shows as filed by the asking handle
(`createAsUser`). Without them, LINEAR_API_KEY is a person's key. Either way the acting user is unsubscribed from each issue
after filing it and after each comment (a comment subscribes its author again), so a person's inbox stays quiet. Either way an issue carries a `from:<handle>` label and
is never assigned or delegated: an issue assigned to the Cursor app starts a new agent instead of waking the lead.

`audit` reads every open issue once and names what keeps it off its board or stale (AUDIT_KINDS); `research msg linear audit`
prints it and top's hourly sweep rings each lane with its findings.  `edit` changes an issue's title, due date, description
or parent, `relate` and `unrelate` its relations, `update` posts a project update, `create_project` and `milestone` make
projects and milestones, all through the app, so nobody writes through the Linear MCP connector, which acts as Daniel.  None
of them sends a state. Stdlib only."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable

API = "https://api.linear.app/graphql"
TOKEN_URL = "https://api.linear.app/oauth/token"
SCOPES = "read,write"   # always the same: a grant with other scopes revokes every live token of the app
EARLY_S = 3600          # a cached token is refetched this long before it expires
TITLE_MAX = 80
OPEN_TYPES = ("triage", "unstarted", "started")                  # the audit's issues: Backlog is parked, not open work
TAKEN = ("Todo", "In Progress", "In Review")                     # taken up: it belongs on a board, with a date
QUIET_H = 24                                                     # In Progress with no comment and no start for this long
STALE_D = 7                                                      # Todo not updated or commented on for this long
AUDIT_PAGE = 25                                                  # first: 50 with the nested fields is "Query too complex"
AUDIT_CHILDREN = 50                                              # a parent's most recently updated children the audit reads
LANE_LABEL = "lane:"                                             # a handle with no team of its own marks its issues so
NO_LANE = "-"
AUDIT_KINDS = ("orphan", "milestone", "undated", "overdue", "quiet", "stale", "title", "landed", "lane")
RING_TAGS_RE = re.compile(r"(?:(?:<!subteam\^[A-Z0-9]+(?:\|[^>]*)?>|<@[A-Z0-9]+(?:\|[^>]*)?>|<!(?:here|channel)>"
                          r"|@[a-z0-9][a-z0-9._-]*)[\s,:;]*)+")
PAREN = r"(?:\s*\([^()\n]{1,30}\))?"                              # `(URGENT)`, `(sent 01:48Z)`
SIGNED_RE = re.compile(rf"[a-z][a-z0-9._-]*{PAREN}[\s,:;]+(?:\d{{1,2}}\s+[A-Z][a-z]{{2}}\s+)?\d{{1,2}}:\d{{2}}Z{PAREN}[\s,:;]*")
LOWER_WORD_RE = re.compile(r"[a-z][a-z'\u2019-]*(?![\w./`@])")   # `a`, `can't`; never `vLLM`, `hm96` or `tools/research`
DUE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
DATES, TEXTS = ("dueDate", "targetDate"), ("description", "content", "body")   # `_input`'s dates and markdown bodies
RELATE = {"blocks": "blocks", "blocked-by": "blocks", "related": "related", "duplicate": "duplicate"}   # the verb's word: Linear's type
SAYS = {"blocks": "blocks", "duplicate": "is a duplicate of", "related": "is related to", "similar": "is similar to"}
HEALTH = ("onTrack", "atRisk", "offTrack")
PAIR_Q = ("query($a: String!, $b: String!) { a: issue(id: $a) { id identifier "
          "relations { nodes { id type issue { id identifier } relatedIssue { id identifier } } } "
          "inverseRelations { nodes { id type issue { id identifier } relatedIssue { id identifier } } } } "
          "b: issue(id: $b) { id identifier } }")

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


def _capped(line: str) -> str:
    return line if len(line) <= TITLE_MAX else line[:TITLE_MAX - 1].rstrip() + "\u2026"


def first_line(body: str) -> str:
    """The body's first line, capped: what the Slack root of a filed ask carries."""
    return _capped(next((s.strip() for s in body.splitlines() if s.strip()), "(no text)"))


def unsigned(text: str) -> str:
    """`text` without its leading ring tags and the sender's signature (`@infra memory-accounting, 12:15Z, `)."""
    text = text.lstrip()
    if m := RING_TAGS_RE.match(text):
        text = text[m.end():]
    if m := SIGNED_RE.match(text):
        text = text[m.end():]
    return text


def title(body: str) -> str:
    """An issue's title from a Slack post's body: its first line without the leading ring tags and the sender's signature
    (`@infra memory-accounting, 12:15Z, an ask…` is `An ask…`), capitalized and capped; the first line as it is when nothing
    else is left."""
    line = next((s.strip() for s in unsigned(body).splitlines() if s.strip()), "")
    if not line:
        return first_line(body)
    return _capped(line[0].upper() + line[1:] if LOWER_WORD_RE.match(line) else line)


_title = title                                                   # `Linear.file`'s `title` argument shadows it


def owners(reg: dict) -> dict[str, str]:
    """Each Linear team key a handle owns (the Slack registry's `linear`), to its handle."""
    return {h["linear"]: h["handle"] for h in reg.get("handles", []) if h.get("linear")}


def _pick(nodes: list[dict], prefix: str, what: str) -> dict:
    """The one node whose name starts with `prefix` (any case), or whose name or first word is it when several do."""
    want = prefix.strip().lower()
    hits = [n for n in nodes if n["name"].lower().startswith(want)] if want else []
    if len(hits) > 1:
        hits = [n for n in hits if n["name"].lower() == want or n["name"].lower().split()[0] == want] or hits
    if not nodes:
        raise LinearError(f"there is no {what} to match {prefix!r}")
    if len(hits) != 1:
        raise LinearError(f"{what} {prefix!r} matches {len(hits) or 'none'} of: " + "; ".join(n["name"] for n in (hits or nodes)))
    return hits[0]


def _input(**given: tuple[str, str | None]) -> dict:
    """The input fields given, each `field=(flag, value)` and left out when the value is None, checked: a date (DATES) is
    YYYY-MM-DD or `none` (clears it), a markdown body (TEXTS) is not empty, a title or name is not blank (and is stripped)."""
    out: dict = {}
    for k, (flag, v) in given.items():
        if v is None:
            continue
        if k in DATES:
            bad = f"{flag} {v!r}: give a date as YYYY-MM-DD, or `none` to clear it"
            if v.lower() != "none":
                if not DUE_RE.fullmatch(v):
                    raise LinearError(bad)
                try:
                    datetime.strptime(v, "%Y-%m-%d")
                except ValueError:
                    raise LinearError(bad) from None
            out[k] = None if v.lower() == "none" else v
        elif k in TEXTS:
            if not v.strip():
                raise LinearError(f"{flag}: the file is empty")
            out[k] = v
        elif not v.strip():
            raise LinearError(f"{flag}: give the new {k}")
        else:
            out[k] = v.strip()
    return out


def said(i: dict) -> str:
    """What an input sets, as `field value, ...`, a markdown body by its length."""
    return ", ".join(f"{k} {f'{len(v)} characters' if k in TEXTS else v}" for k, v in i.items())


def _says(r: dict) -> str:
    """A relation as a sentence: `INF-1 blocks INF-2`."""
    return f"{r['issue']['identifier']} {SAYS.get(r['type'], r['type'])} {r['relatedIssue']['identifier']}"


def _epoch(stamp: str | None) -> float | None:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() if stamp else None


def tally(findings: list[dict]) -> dict[str, dict[str, int]]:
    """The findings' counts by lane, then by kind (in AUDIT_KINDS' order)."""
    out: dict[str, dict[str, int]] = {}
    for f in findings:
        out.setdefault(f["lane"], {}).setdefault(f["kind"], 0)
        out[f["lane"]][f["kind"]] += 1
    return {lane: {k: kinds[k] for k in AUDIT_KINDS if k in kinds} for lane, kinds in sorted(out.items())}


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

    def file(self, team: str, sender: str, body: str, parent: str | None = None, title: str | None = None) -> dict:
        """The ask as an issue in `team`, from `sender`: {id, identifier, url, project, projectMilestone}. With `parent` (an
        identifier such as CMP-92) it is that issue's sub-issue, on its project and milestone, so it lands on the board of the
        deliverable it serves; where Linear refuses the project (the team isn't in it), under the parent alone, said on stderr.
        `title` replaces the one derived from the body."""
        i = {"teamId": self.team(team)["id"], "title": title.strip() if title else _title(body), "description": body,
             "labelIds": [self.label(f"from:{sender}")]}
        if self.app:
            i["createAsUser"] = sender
        board = {}
        if parent:
            p = self.gql("query($id: String!) { issue(id: $id) { id project { id } projectMilestone { id } } }", {"id": parent})["issue"]
            i["parentId"] = p["id"]
            board = {k: x["id"] for k, x in (("projectId", p.get("project")), ("projectMilestoneId", p.get("projectMilestone"))) if x}
        q = ("mutation($i: IssueCreateInput!) { issueCreate(input: $i) { issue { id identifier url project { name } "
             "projectMilestone { name } } } }")
        try:
            issue = self.gql(q, {"i": {**i, **board}})["issueCreate"]["issue"]
        except LinearError as e:
            if not board:
                raise
            print(f"research msg: Linear refused {parent}'s project for this issue ({e}); filed under {parent} alone", file=sys.stderr)
            issue = self.gql(q, {"i": i})["issueCreate"]["issue"]
        self._quiet(issue["id"])
        return issue

    def projects(self) -> list[dict]:
        """The active projects (not completed or canceled), each with its milestones: {id, name, projectMilestones}."""
        q = ('{ projects(first: 100, filter: {status: {type: {nin: ["completed", "canceled"]}}}) '
             "{ nodes { id name projectMilestones(first: 50) { nodes { id name } } } } }")
        return self.gql(q)["projects"]["nodes"]

    def project(self, prefix: str) -> dict:
        """The one active project whose name starts with `prefix` (any case), as `projects` gives it. A prefix that matches
        none or several is refused, naming the candidates."""
        return _pick(self.projects(), prefix, "active project")

    def place(self, issue: str, project: str, milestone: str | None = None) -> dict:
        """Put the issue (an identifier such as CMP-92) in the active `project` (a prefix of its name), and with `milestone`
        in that project's milestone whose name starts with it (`M3`, `F2`): {issue, project, milestone}."""
        p = self.project(project)
        i, m = {"projectId": p["id"]}, None
        if milestone:
            m = _pick(p["projectMilestones"]["nodes"], milestone, f"milestone of {p['name']}")
            i["projectMilestoneId"] = m["id"]
        self.gql("mutation($id: String!, $i: IssueUpdateInput!) { issueUpdate(id: $id, input: $i) { success } }",
                 {"id": self.issue_id(issue), "i": i})
        return {"issue": issue, "project": p["name"], "milestone": m["name"] if m else None}

    def edit(self, issue: str, title: str | None = None, due: str | None = None, description: str | None = None,
             parent: str | None = None, send: bool = True) -> dict:
        """Change the issue's title, due date (YYYY-MM-DD, or `none` to clear it), description (replaced whole, as markdown) or
        parent (an identifier), at least one, through these credentials rather than a person's (the MCP connector acts as
        Daniel): the IssueUpdateInput it sends, which `send=False` only builds."""
        i = _input(title=("--title", title), dueDate=("--due", due), description=("--file", description))
        if parent is not None:
            i["parentId"] = self.issue_id(parent) if send else parent
        if not i:
            raise LinearError("give at least one of --title, --due, --file and --parent")
        if send:
            self.gql("mutation($id: String!, $i: IssueUpdateInput!) { issueUpdate(id: $id, input: $i) { success } }",
                     {"id": self.issue_id(issue), "i": i})
        return i

    def _pair(self, issue: str, other: str) -> tuple[dict, dict, list[dict]]:
        """The two issues ({id, identifier}) and the relations between them, either way; two names of one issue are refused."""
        d = self.gql(PAIR_Q, {"a": issue, "b": other})
        a, b = d["a"], d["b"]
        if a["id"] == b["id"]:
            raise LinearError(f"{issue} and {other} are one issue: name two different issues")
        rels = [r for r in a["relations"]["nodes"] + a["inverseRelations"]["nodes"]
                if {r["issue"]["id"], r["relatedIssue"]["id"]} == {a["id"], b["id"]}]
        return a, b, rels

    def relate(self, issue: str, kind: str, other: str) -> str:
        """Relate `issue` to `other` (identifiers) by `kind`, a key of RELATE (`blocked-by` is `other` blocks `issue`): the
        relation as a sentence. The same relation again is refused (`related` either way round). Linear itself moves an issue
        marked a duplicate to its Duplicate status, and the sentence says so."""
        if kind not in RELATE:
            raise LinearError(f"relate: {kind!r} is no relation: give one of {', '.join(RELATE)}")
        a, b, rels = self._pair(issue, other)
        t = RELATE[kind]
        x, y = (b, a) if kind == "blocked-by" else (a, b)
        new = {"issue": x, "type": t, "relatedIssue": y}
        if any(r["type"] == t and (r["issue"]["id"] == x["id"] or t == "related") for r in rels):
            raise LinearError(f"{_says(new)} already: there is nothing to add")
        self.gql("mutation($i: IssueRelationCreateInput!) { issueRelationCreate(input: $i) { success } }",
                 {"i": {"issueId": x["id"], "relatedIssueId": y["id"], "type": t}})
        return _says(new) + (f" (Linear moves {x['identifier']} to its Duplicate status)" if t == "duplicate" else "")

    def unrelate(self, issue: str, other: str) -> list[str]:
        """Delete every relation between the two issues, either way: each as a sentence. None is refused."""
        a, b, rels = self._pair(issue, other)
        if not rels:
            raise LinearError(f"{a['identifier']} and {b['identifier']} have no relation to remove")
        for r in rels:
            self.gql("mutation($id: String!) { issueRelationDelete(id: $id) { success } }", {"id": r["id"]})
        return [_says(r) for r in rels]

    def update(self, project: str, health: str | None, body: str | None) -> str:
        """Post a project update on the active `project` (a prefix of its name, as `place` takes it), with its health (one of
        HEALTH, any case) and markdown body: a line with its link."""
        h = {x.lower(): x for x in HEALTH}.get((health or "").lower())
        if not h:
            raise LinearError(f"--health{f' {health!r}' if health else ''}: give one of {', '.join(HEALTH)}")
        if body is None:
            raise LinearError("--body-file F: give the update's text as a markdown file")
        i = {"health": h, **_input(body=("--body-file", body))}
        p = self.project(project)
        u = self.gql("mutation($i: ProjectUpdateCreateInput!) { projectUpdateCreate(input: $i) { projectUpdate { url } } }",
                     {"i": {"projectId": p["id"], **i}})["projectUpdateCreate"]["projectUpdate"]
        return f"{p['name']}: update posted, {h} {u['url']}"

    def create_project(self, name: str, teams: list[str], lead: str | None = None, target: str | None = None,
                       content: str | None = None) -> str:
        """A new project named `name` in `teams` (keys or names), led by `lead` (a prefix of a Linear user's name: Linear's
        lead is a person, never a team or handle), with its target date and markdown body: a line with its link. A name some
        project has already (any case) is refused."""
        if not teams:
            raise LinearError("project create: give its team, --team KEY (more: repeat it)")
        name, opt = name.strip(), _input(targetDate=("--target", target), content=("--description-file", content))
        q = "query($n: String!) { projects(filter: {name: {eqIgnoreCase: $n}}) { nodes { name } } }"
        if same := self.gql(q, {"n": name})["projects"]["nodes"]:
            raise LinearError(f"project {same[0]['name']!r} exists already: pick another name, or work in that one")
        ts = [self.team(t) for t in teams]
        i, shown = {"name": name, "teamIds": [t["id"] for t in ts], **opt}, {"teams": " ".join(t["key"] for t in ts)}
        if lead:
            u = _pick(self.gql("{ users(first: 100) { nodes { id name } } }")["users"]["nodes"], lead, "Linear user")
            i["leadId"], shown["lead"] = u["id"], u["name"]
        p = self.gql("mutation($i: ProjectCreateInput!) { projectCreate(input: $i) { project { url } } }", {"i": i})
        return f"project {name} created: {said({**shown, **opt})} {p['projectCreate']['project']['url']}"

    def milestone(self, project: str, name: str, date: str | None = None, description: str | None = None,
                  rename: str | None = None, create: bool = False) -> str:
        """Make the milestone `name` in the active `project` (a prefix of its name), or (`create=False`) change the one whose
        name starts with `name`: its target date, markdown description and (editing) its name. A name another milestone of
        the project has (any case) is refused."""
        i = _input(name=("NAME", name) if create else ("--rename", rename), targetDate=("--date", date),
                   description=("--description-file", description))
        if not i:
            raise LinearError("milestone edit: give at least one of --date, --description-file and --rename")
        p = self.project(project)
        ms = p["projectMilestones"]["nodes"]
        m = None if create else _pick(ms, name, f"milestone of {p['name']}")
        if "name" in i and (taken := next((x for x in ms if x is not m and x["name"].lower() == i["name"].lower()), None)):
            raise LinearError(f"{p['name']} has milestone {taken['name']!r} already: pick another name, or edit that one "
                              f"(`research msg linear milestone edit PROJECT NAME`)")
        if m is None:
            self.gql("mutation($i: ProjectMilestoneCreateInput!) { projectMilestoneCreate(input: $i) { success } }",
                     {"i": {"projectId": p["id"], **i}})
            rest = {k: v for k, v in i.items() if k != "name"}
            return f"{p['name']} / {i['name']}: created" + (f", {said(rest)}" if rest else "")
        self.gql("mutation($id: String!, $i: ProjectMilestoneUpdateInput!) { projectMilestoneUpdate(id: $id, input: $i) { success } }",
                 {"id": m["id"], "i": i})
        return f"{p['name']} / {m['name']}: set {said(i)}"

    def unread(self, teams: list[str]) -> list[str]:
        """Those of `teams` (keys or names, any case) these credentials can't see, which the audit skips: Linear answers a
        filter on such a team with no issues, not an error."""
        nodes = self.gql("{ teams(first: 100) { nodes { key name } } }")["teams"]["nodes"]
        seen = {x.lower() for t in nodes for x in (t["key"], t["name"])}
        return [t for t in teams if t.lower() not in seen]

    def audit(self, owners: dict[str, str] | None = None, teams: list[str] | None = None, now: float | None = None) -> list[dict]:
        """One pass over the open issues (OPEN_TYPES), of `teams` (names or keys) or every team: each finding
        {issue, url, title, team, lane, state, kind, fix}, by lane, issue and kind (AUDIT_KINDS):
        - orphan: Todo, In Progress or In Review, in no project.  milestone: in a project with milestones, with none set.
        - undated: Todo, In Progress or In Review with no due date.  overdue: due before today (UTC).
        - quiet: In Progress with no comment and no start for QUIET_H (the last comment's and the start's times, not updatedAt,
          which bulk edits move).  stale: Todo neither updated nor commented on for STALE_D days (a comment leaves
          updatedAt as it was).  A parent counts its children's (AUDIT_CHILDREN of them) as its own: each one's updatedAt,
          which its start and state changes move, and its last comment.
        - title: it starts with ring tags or a sender's signature (`unsigned` would change it), or ends in "…": a Slack
          post's first line, not a deliverable.  An area prefix (`pous: …`) and a long title are fine.
        - landed: In Progress or In Review with GitHub PRs attached, every one merged or closed, and no comment since the
          last of them closed (its metadata's closedAt; an attachment's own updatedAt never moves): a comment naming the
          piece left answers it.
        - lane: its team has no handle (`owners`, team key to handle) and it has no `lane:<handle>` label.
        `lane` is the issue's `lane:` label, else its team's handle, else NO_LANE.  A team these credentials can't see
        (`unread`) has no issues here, so it adds no finding and no error."""
        now, owners = self.now() if now is None else now, owners or {}
        today = datetime.fromtimestamp(now, UTC).date().isoformat()
        f: dict = {"state": {"type": {"in": list(OPEN_TYPES)}}}
        if teams:
            f["team"] = {"or": [{"key": {"in": list(teams)}}, {"name": {"in": list(teams)}}]}
        q = ("query($f: IssueFilter!, $after: String) { issues(filter: $f, first: " + str(AUDIT_PAGE) + ", after: $after) { "
             "pageInfo { hasNextPage endCursor } nodes { identifier title url dueDate startedAt updatedAt createdAt "
             "state { name } team { key } project { id name } projectMilestone { id } labels { nodes { name } } "
             "comments(first: 1, orderBy: createdAt) { nodes { createdAt } } attachments { nodes { url sourceType metadata } } "
             f"children(first: {AUDIT_CHILDREN}, orderBy: updatedAt) {{ nodes {{ updatedAt "
             "comments(first: 1, orderBy: createdAt) { nodes { createdAt } } } } } } }")
        issues, after = [], None
        while True:
            page = self.gql(q, {"f": f, "after": after})["issues"]
            issues += page["nodes"]
            if not page["pageInfo"]["hasNextPage"]:
                break
            after = page["pageInfo"]["endCursor"]
        staged = {p["id"]: bool(p["projectMilestones"]["nodes"]) for p in self.projects()} if issues else {}
        out = []
        for i in issues:
            ident, state, key, project = i["identifier"], i["state"]["name"], i["team"]["key"], i.get("project")
            label = next((x["name"][len(LANE_LABEL):] for x in i["labels"]["nodes"] if x["name"].startswith(LANE_LABEL)), None)
            lane = label or owners.get(key) or NO_LANE
            prs = [a["metadata"] for a in i["attachments"]["nodes"]
                   if a.get("sourceType") == "github" and "/pull/" in (a.get("url") or "") and (a.get("metadata") or {}).get("status")]
            shut = [_epoch(p.get("closedAt") or p.get("mergedAt")) for p in prs]
            closed = max(shut) if shut and all(shut) else None
            last = max((t for t in (_epoch(c["createdAt"]) for c in i["comments"]["nodes"]) if t), default=None)
            kids = [_epoch(s) for c in (i.get("children") or {}).get("nodes", [])
                    for s in (c.get("updatedAt"), *(x["createdAt"] for x in c["comments"]["nodes"]))]
            kid = max((t for t in kids if t), default=None)
            heard = max((t for t in (last, _epoch(i.get("startedAt")), kid) if t), default=_epoch(i["createdAt"]))
            idle_d = (now - max(t for t in (_epoch(i["updatedAt"]), last, kid) if t)) / 86400
            nums = ", ".join(f"#{p.get('number')}" for p in prs[:5]) + (", ..." if len(prs) > 5 else "")
            found = {
                "orphan": state in TAKEN and not project
                and f"put it in its project and milestone (`research msg linear place {ident} PROJECT --milestone M`), "
                    "or move it to Backlog",
                "milestone": project and staged.get(project["id"]) and not i.get("projectMilestone")
                and f"set the milestone it serves (`research msg linear place {ident} \"{project['name']}\" --milestone M`)",
                "undated": state in TAKEN and not i.get("dueDate")
                and f"set the date it lands, its milestone's at the latest (`research msg linear edit {ident} --due YYYY-MM-DD`), "
                    "or move it to Backlog",
                "overdue": i.get("dueDate") and i["dueDate"] < today
                and f"due {i['dueDate']}: land it, or set the date it will land (`research msg linear edit {ident} --due "
                    "YYYY-MM-DD`) and comment why it moved",
                "quiet": state == "In Progress" and now - heard > QUIET_H * 3600
                and f"no comment or start for {(now - heard) / 3600:.0f} h: comment one line (what changed, the next step and "
                    "its date), or move it back to Todo",
                "stale": state == "Todo" and idle_d > STALE_D
                and f"not updated or commented on for {idle_d:.0f} days: plan it (a date and milestone) or cancel it with a one-line reason",
                "title": (unsigned(i["title"]) != i["title"].lstrip() or i["title"].rstrip().endswith("\u2026"))
                and "retitle it to the deliverable, with no handle, sender, time or trailing \u2026 "
                    f"(`research msg linear edit {ident} --title T`)",
                "landed": state in ("In Progress", "In Review") and prs and all(p["status"] in ("merged", "closed") for p in prs)
                and not (last and closed and last > closed)
                and f"every attached PR is merged or closed ({nums}): move it to Done if its done-condition holds, else to "
                    f"In Progress with a comment naming the piece left (`research msg linear comment {ident} TEXT`)",
                "lane": key not in owners and not label
                and f"team {key} has no handle: add `{LANE_LABEL}<owner>`",
            }
            out += [{"issue": ident, "url": i["url"], "title": i["title"], "team": key, "lane": lane, "state": state, "kind": k,
                     "fix": found[k]} for k in AUDIT_KINDS if found[k]]
        order = {k: n for n, k in enumerate(AUDIT_KINDS)}
        return sorted(out, key=lambda x: (x["lane"], x["team"], int(x["issue"].rsplit("-", 1)[1]), order[x["kind"]]))

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


def ask_issue(owner: str, sender: str, body: str, linear: Any = None, parent: str | None = None,
              title: str | None = None) -> tuple[dict | None, str | None]:
    """`msg ask`'s issue: (issue, None), or (None, why) when it couldn't be filed, so the ask still goes out on Slack."""
    try:
        return (linear or Linear()).file(owner, sender, body, parent=parent, title=title), None
    except (LinearError, OSError, ValueError) as e:
        return None, str(e)

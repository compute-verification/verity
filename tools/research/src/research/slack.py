"""`research msg`: messages between agents, on Slack (the conventions are
`.agents/skills/using-slack/SKILL.md`).

A message is a post in #agent-coordination, sent by a name: a lead's handle in the registry (@infra, @proofs, @lean, ...), which
`names open` gives an inbox thread. Only leads get names; a worker is reached through its lead. A thread is a conversation. Who a message reaches is one rule (`route_targets`): the names that
lead it (`@a @b ...`), an announcement's audience, on a reply the thread's starter or the names it addressed, and on a reply
under an announcement that names no one the owners of the PRs it names (`pr_watchers`). Each gets a doorbell in its inbox
thread, which its agent subscribes to. The footer carries the post's kind and flags.
Notes are records, never messages.

  research msg send --as NAME [--to @A @B ...] [--thread TS] [--kind WORD] [--quiet] [--urgent] (--text T | --file PATH)
                    [--verify CMD] [--ships URL#sha256=HEX | git:SHA]...     # prints `<ts> <permalink>`; --kind announce: an announcement
  research msg close --as NAME --thread TS [--link URL] [--note T] [--declined WHY]   # ✅ or ❌: a status line, and the reaction on an ask
  research msg read --thread TS [--since TS] | --since TS | --as NAME [--peek] [--urgent] [--json]  # --as: NAME's unread doorbells
  research msg names open|read|close|list [--as NAME] [--forward-to @NAME] [--succeeds] [--poster BC_ID]  # the names you answer to
  research msg traffic [--hours H] [--replay] [--json]             # the monitor: doorbells, pauses, dropped names, unanswered asks
  research msg traffic --latency [--hours H] [--json]              # per lane: asks, answered (Slack or Linear), median and p90 minutes
  research msg watch --as NAME [--every S] [--polls N]             # block until a new doorbell; the hook a wake script runs
  research msg pickup --thread TS                                  # 👀 on an ask you own
  research msg retract --as NAME --ts TS [--thread ROOT] [--why W] # blank a post NAME sent (needs SLACK_BOT_TOKEN)
  research msg roster --thread TS [--json]                         # an announcement's status lines; exit 0 when all are done
  research msg handles [--json] | subscriptions --as HANDLE        # the registry, holders from routes; the subscribe calls a holder makes
  research msg owners [PATH...]                                    # who owns each path (stdin: `git diff --name-only main... |`)
  research msg match --as HANDLE --event JSON|TEXT|@PATH|- [--why] # exit 0 addressed, 1 not, 2 error
  research msg verify-author --event JSON|@PATH|- [--why] [--authorizes]  # exit 0 only for our bot, a channel's bot or a named person;
                                                                   # --authorizes: only for a named human, who may authorize spend
  research msg deliver --ts TS [--thread T]                         # ring the doorbells for a post no client delivered
  research msg deliver --channel ask-the-team --event JSON|@PATH|-  # a person's post there: its doorbells name the channel
  research msg groups sync [--apply]                               # one user group per handle; a plan unless --apply
  research msg ask-daniel --as HANDLE --kind blocking|default --question Q --why W --option VALUE=LABEL... --recommend V
                          --default V --deadline '3:00 PM'|ISO (--page FILE.md --area AREA | --link NOTION_URL) [--wait]
                          # the ask's page in the Asks for Daniel database, and a card in #ask-daniel linking it; prints `<id> <ts> <permalink>`
  research msg ask-daniel status ID [--wait] [--json] | resolve ID --choice V [--note T]
  research msg ask-daniel pages [--as HANDLE] [--hours H] [--json]  # asks' pages edited lately: Status, Decision, newest comment

`ask`, `announce` and `reply` are the older spellings of `send`, kept while callers move.

#ask-daniel's cards go through the site's route (ASK_DANIEL, Cursor OIDC with the route as audience), not the Slack API; a route
that refuses the card's fields (HTTP 400) gets the old Approve/Deny shape `{title, text, link, requested_by}` instead. Each card
links its ask's page in the registry's Notion database (the ask-daniel channel's `pages`), where Daniel reads, comments and
decides; `--page` and `pages` reach it through Notion's API with $NOTION_API_KEY, and $RESEARCH_NOTION_URL points them at
another server (the tests' fake).

Every subcommand takes `--dry-run` (no request reaches Slack; each one it would send is printed, the token never),
`--registry PATH` (default $RESEARCH_SLACK_REGISTRY, else the checkout's) and `--channel KEY` (default agent-coordination);
a channel's id is $SLACK_CHANNEL_<KEY> (upper case, `-` as `_`) or the registry's, and `--channel-id` overrides both. The token
is $SLACK_BOT_TOKEN and is never printed; $SLACK_API_URL points the client at another Web API (the tests' fake). Stdlib only,
and it runs as a single file (`python3 slack.py ...`) from any checkout.

Without $SLACK_BOT_TOKEN (or with RESEARCH_SLACK_VIA=broker) every call goes through Verity's Slack broker instead: a POST of
`{"method", "args"}` to BROKER with this agent's Cursor OIDC token, answered with Slack's response unchanged. The broker holds the
bot token and allows only BROKER_METHODS; RESEARCH_SLACK_VIA=direct insists on the token.

The sender rings the doorbells right after posting; there is no server-side router (the website has no events route,
COMM-32). It routes by origin/main's registry (`routing_registry`) and the published routes (`Routes.refresh`), so an old
checkout or notes clone still rings a name added, reopened or closed since. Who holds a handle, who else posts as it and
which branch suffixes its PRs carry is its route's (`names open`), never the registry's. `match` is the older,
handle-only test of whether a post is addressed to HANDLE."""
from __future__ import annotations

import argparse
import base64
import html
import http.client
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

try:
    from .timefmt import both, slack_dates
except ImportError:                                                 # run as a single file: `python3 slack.py ...`
    def both(t: Any) -> str:
        return f"{datetime.fromtimestamp(float(t), timezone.utc):%H:%MZ}" if re.fullmatch(r"\d+(?:\.\d+)?", str(t)) else str(t)

    def slack_dates(text: str, now: Any = None) -> str:
        return text

SCHEMA = "verity/slack-registry/v1"
REGISTRY_REL = Path(".agents/skills/using-slack/registry.json")
REGISTRY_REFS = ("origin/main",)                                  # where a checkout older than the registry finds it
FRONT_DOOR = "agent-coordination"
ASKS_ON_LINEAR = True                                             # an ask is also an issue in its owner's team (`linear.py`)
LINEAR_LINK_RE = re.compile(r"\nLinear: <https://linear\.app/[^|>]+\|([A-Z][A-Z0-9]*-\d+)>")
LINEAR_KEY_RE = re.compile(r"^[A-Z][A-Z0-9]{1,6}$")
DEFAULT_API = "https://slack.com/api/"
BROKER = "https://website-docs-sage.vercel.app/api/agent-slack/call"      # also the OIDC token's audience
#: exactly what the broker's route allows: every Web API method this module calls but DIRECT_ONLY (the tests check the split)
BROKER_METHODS = ("auth.test", "chat.getPermalink", "chat.postMessage", "conversations.history", "conversations.replies",
                  "reactions.add", "usergroups.list")
DIRECT_ONLY = ("chat.update", "usergroups.create", "usergroups.disable", "usergroups.enable", "usergroups.update")   # `retract`, `groups sync --apply`: the token itself
BROKER_MAX_BODY = 65536                                                   # bytes of request JSON; the route refuses more (413)
ERROR_CODE_RE = re.compile(r"^[a-z0-9_]{1,64}$")
SCOPES_RE = re.compile(r"^[a-z0-9_.:]{1,64}(,[a-z0-9_.:]{1,64}){0,15}$")      # a not-ok answer's `needed`: chat:write,users:read
EVERY = "every handle"
DESCRIPTION_MAX = 140                                            # Slack's user-group description limit (description_too_long)
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
SUBTEAM_RE = re.compile(r"<!subteam\^([A-Z0-9]+)(?:\|@?([^>]*))?>")
DATE_TOKEN_RE = re.compile(r"<!date\^\d+\^[^|>]*\|([^>]*)>")          # `read` and `roster` show a date token as its fallback
ANNOUNCE = "\U0001f4e3"                                            # 📣
DONE, DECLINED = "\u2705", "\u274c"                                 # ✅ ❌
SHORTCODES = {ANNOUNCE: ":mega:", DONE: ":white_check_mark:", DECLINED: ":x:"}   # Slack stores a posted emoji as its shortcode
STATUS_RE = re.compile(rf"^((?:{DONE}|{SHORTCODES[DONE]}) done|(?:{DECLINED}|{SHORTCODES[DECLINED]}) declined)\b(?: \u2014 (.*?))?(?:\n|$)")
PIN_RE = re.compile(r"^(?:https?://\S+#sha256=[0-9a-f]{64}|git:[0-9a-f]{40})$")
SHIPS_CODE_RE = re.compile(r"\b(curl|wget|pip install|uv (?:pip|sync|tool)|git (?:pull|merge|checkout|fetch)|install|download|upgrade)\b", re.I)
SAFE_POINT = "Apply at your next safe point; don't interrupt running jobs."
EXPIRES_S = 2592000                                              # 30 days; a lapsed subscription wakes no one and says nothing
PAGE = 200
RATE_RETRIES, RATE_WAIT_MAX_S = 3, 30                         # a 429: retry this often, waiting at most this long each time
ASK_DANIEL = "https://website-docs-sage.vercel.app/api/agent-approvals"   # the cards' route, and its OIDC token's audience
ASK_CHANNEL = "ask-daniel"
PACIFIC = ZoneInfo("America/Los_Angeles")
KINDS = ("blocking", "default")
OPTION_RE = re.compile(r"^([a-z0-9][a-z0-9_-]{0,31})=(.+)$")
OPTIONS_MIN, OPTIONS_MAX = 2, 5
LABEL_MAX = 75                                                   # Slack's button text limit
DEADLINE_MAX = timedelta(days=7)
CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*([AaPp][Mm])$")
DECIDED = ("decided", "approved", "denied")                      # new cards, then old-shape Approve/Deny cards
NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2026-03-11"                                    # the first version whose pages take Notion-flavored Markdown
NOTION_HOSTS = ("notion.so", "www.notion.so", "app.notion.com")  # and any *.notion.site
AREAS = ("Claim or assumption", "Access or keys", "Machines or spend", "Process", "Research direction")   # the database's Area
PAGE_SECTIONS = ("## The problem", "## Background", "## Options", "## What happens next", "## Sources")
REPLY_CALLOUT = ('<callout icon="✍️" color="yellow_bg">\n'
                 "\t**Your reply.** Write below this line: a decision, a question or a change. "
                 "Top relays it to the lane, which answers under your text starting with its handle.\n"
                 "</callout>\n")                                 # top's asks relay reads the text after the bold line
SKILL_ASKING = ".agents/skills/asking-daniel/SKILL.md"
WAIT_EVERY_S = 30
WAIT_GRACE = timedelta(minutes=5)                                # the site's default cron runs every minute
ROUTES_DIR = "routes.d"                                          # in the notes: one `<name>.json` per inbox
ROUTES_RAW = "https://raw.githubusercontent.com/compute-verification/research-notes/main/routes.d/"
ROUTE_SCHEMA = "verity/slack-route/v1"
LEAD_RE = re.compile(r"^\s*(?:(?::[a-z0-9_+'-]+:|[\u2600-\u27bf\U0001f000-\U0001faff]\ufe0f?)\s*)?"   # one emoji may lead (`:mega: @a`)
                     r"((?:(?:@[a-z0-9][a-z0-9._-]*|<!subteam\^[A-Z0-9]+(?:\|@?[^>]*)?>)[\s,:]*)+)")
BARE_LEAD_RE = re.compile(r"^\s*([a-z0-9][a-z0-9._-]*):\s")
NAME_TOKEN_RE = re.compile(r"@([a-z0-9][a-z0-9._-]*)|<!subteam\^([A-Z0-9]+)(?:\|@?([^>]*))?>")
AGENT_ID_RE = re.compile(r"bc-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|bc-[0-9a-f]{8}"   # a cloud agent's (or its short form)
                         r"|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")                 # a desktop agent's, a bare UUID
QUIET = "(quiet)"                                                # ending a post: ring only the names it starts with
CAP_PER_HOUR = 20                                                # doorbells into one inbox from one author in one thread
DOORBELL_MAX = 600                                               # characters of the post a doorbell carries
POST_MAX = 3000                                                  # bytes: Slack splits a longer post into separate top-level messages
INBOX_HEAD = "Inbox for @"
PR_RE = re.compile(r"(?<![\w&/#])#(\d{3,6})\b")
BRANCH_RE = re.compile(r"\bcursor/([a-z0-9._-]*[a-z0-9])")
TRAIN_PREP_RE = re.compile(r"^train-prep-((?:\d+-)+)")                  # `cursor/train-prep-1585-1586-on-041c30713-4292`
RUN_ID_RE = re.compile(r"\br\d{8}-\d{6}-([0-9a-f]{4})\b|(?<![\w./#@-])([0-9a-f]{4})(?![\w/@-])")   # a run, or its last 4
HISTORY_S = 86400                                                # how far back a reply's thread counts for `pr_watchers`
GH, GH_REPO = "gh", "compute-verification/verity"                       # `pr_branches` asks gh for PR branches
TAG_SEP = " \u00b7 "
_TAG = r"(?:[a-z0-9][a-z0-9._-]*|\([a-z]+\))"
FOOTER_RE = re.compile(rf"\n_({_TAG}(?: \u00b7 {_TAG})*)_\s*$")                                  # `_ask · urgent_`
OLD_FOOTER_RE = re.compile(r"\n*_posted by <[^>\n]*>((?: \u00b7 [^\u00b7_\n]+)*)_\s*$")   # before 1 Oct 11:40 PDT
LINK_RE = re.compile(r"<((?:https?|mailto):[^|>\s]+)(?:\|([^>]*))?>")
FLAGS = ("urgent", QUIET)
SLACK_USER_RE = re.compile(r"^[UW][A-Z0-9]{2,}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
COLLABORATOR_ROLE = "collaborator: may ask, never authorizes spend, access or destructive changes"
UNREADABLE = ("missing_scope", "not_in_channel")                 # a channel the bot may post in but not read (#ask-the-team)


class SlackError(RuntimeError):
    pass


# --- registry ---------------------------------------------------------------------------------------------------------------

def registry_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    if env := os.environ.get("RESEARCH_SLACK_REGISTRY"):
        return Path(env)
    here = Path(__file__).resolve()
    for d in (Path.cwd(), *Path.cwd().parents, *here.parents):
        if (d / REGISTRY_REL).is_file():
            return d / REGISTRY_REL
    for ref in REGISTRY_REFS:                                       # a checkout whose tree predates the registry
        out = __import__("subprocess").run(["git", "show", f"{ref}:{REGISTRY_REL}"], capture_output=True)
        if out.returncode == 0 and out.stdout:
            p = Path(__import__("tempfile").gettempdir()) / f"verity-slack-registry-{re.sub(r'[^A-Za-z0-9]+', '-', ref)}.json"
            p.write_bytes(out.stdout)
            return p
    raise SlackError(f"no {REGISTRY_REL} here or above, nor at {' or '.join(REGISTRY_REFS)} in this checkout: run from your "
                     f"verity checkout after `git fetch origin main`, or pass --registry")


def origin_registry(path: Path, fetch: bool = True) -> dict | None:
    """The registry as origin/main has it at `path`'s place in its checkout (`git show`, after a best-effort fetch), or None
    where git can't answer (no clone, no remote, offline) or the file isn't JSON."""
    cwd = path.resolve().parent
    git = lambda *a: subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True, timeout=15,  # noqa: E731
                                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    try:
        if fetch:
            git("fetch", "-q", "origin", "main")
        out = git("show", f"origin/main:./{path.name}")
        reg = json.loads(out.stdout) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return reg if isinstance(reg, dict) else None


def origin_groups(path: Path) -> set[str]:
    """The user-group ids in origin/main's registry: an analysis checkout older than a sender's must not report the
    sender's group mentions as dropped (three false DROPPED lines on 14 Oct). Empty when git can't answer, which keeps
    `traffic`'s old behavior."""
    try:
        return {h["usergroup"] for h in (origin_registry(path) or {}).get("handles", []) if h.get("usergroup")}
    except (AttributeError, KeyError, TypeError):
        return set()


FRESH_EVERY_S = 1800                                             # a routed post fetches origin/main at most this often
ROUTED = ("send", "ask", "announce", "reply", "close", "deliver", "linear")   # the subcommands that ring doorbells


def routing_registry(reg: dict, path: Path, now: float | None = None) -> dict:
    """The registry a post routes by: origin/main's handles, then those only this checkout has (a branch adding one).  A
    sender on an old checkout otherwise rings no one for a handle added since (lean missed @console, proofs @hash; 18 Oct).
    It fetches origin/main at most once per FRESH_EVERY_S (a stamp file) and reads it with `git show`; where git can't
    answer, or origin/main's registry doesn't load, it is `reg`.  RESEARCH_SLACK_FRESH_CHECK=0 turns it off."""
    if os.environ.get("RESEARCH_SLACK_FRESH_CHECK") == "0":
        return reg
    stamp = Path(os.environ.get("RESEARCH_SLACK_SEEN") or Path.home() / ".research" / "slack-seen") / ".fresh-fetch"
    now = time.time() if now is None else now
    try:
        due = not stamp.is_file() or now - stamp.stat().st_mtime >= FRESH_EVERY_S
        if due:
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.touch()
            os.utime(stamp, (now, now))
    except OSError:
        due = False
    up = origin_registry(path, fetch=due)
    try:
        up = check_registry(up, f"origin/main's {path.name}") if up else None
    except SlackError:
        up = None
    if not up:
        return reg
    names = {h["handle"] for h in up["handles"]}
    return {**reg, **up, "handles": [*up["handles"], *(h for h in reg["handles"] if h["handle"] not in names)]}


def load_registry(path: Path) -> dict:
    try:
        reg = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SlackError(f"registry {path}: {e}") from e
    return check_registry(reg, str(path))


def check_registry(reg: dict, path: str) -> dict:
    if reg.get("schema") != SCHEMA:
        raise SlackError(f"registry {path}: schema {reg.get('schema')!r}, expected {SCHEMA!r}")
    handles, owned = set(), {}
    for h in reg.get("handles", []):
        name = h.get("handle", "")
        if not NAME_RE.match(name) or name in handles:
            raise SlackError(f"registry {path}: bad or duplicate handle {name!r}")
        handles.add(name)
        for k in ("remit", "charter"):
            if not isinstance(h.get(k), str) or not h[k]:
                raise SlackError(f"registry {path}: @{name} has no {k}")
        if len(h["remit"]) > DESCRIPTION_MAX:
            raise SlackError(f"registry {path}: @{name}'s remit is {len(h['remit'])} characters; its user group's description takes {DESCRIPTION_MAX}")
        for p in h.get("owns", []):
            if not isinstance(p, str) or not p or p.startswith("/") or ".." in p.split("/") or p in owned:
                raise SlackError(f"registry {path}: @{name} owns {p!r}: a path relative to the checkout that no other handle owns")
            owned[p] = name
        if "linear" in h and (not LINEAR_KEY_RE.match(str(h["linear"])) or any(o.get("linear") == h["linear"] for o in reg["handles"] if o is not h)):
            raise SlackError(f"registry {path}: @{name}'s linear {h['linear']!r}: a Linear team key no other handle has")
    chans = reg.get("channels") or {}
    if FRONT_DOOR not in chans:
        raise SlackError(f"registry {path}: no #{FRONT_DOOR}")
    names = set(chans)
    for key, ch in chans.items():
        subs = ch.get("subscribers")
        if not NAME_RE.match(key) or not ch.get("purpose") or not (subs == EVERY or (isinstance(subs, list) and set(subs) <= handles)):
            raise SlackError(f"registry {path}: #{key} needs a purpose and subscribers ({EVERY!r} or a list of handles)")
        for alias in ch.get("aliases", []):
            if not NAME_RE.match(alias) or alias in names:
                raise SlackError(f"registry {path}: #{key}'s alias {alias!r} is malformed or names another channel")
            names.add(alias)
    check_collaborators(reg, path, handles)
    return reg


def check_collaborators(reg: dict, path: str, handles: set[str]) -> None:
    """`collaborators`: people who may ask a lane for something, each `{name, slack_user, email (optional), note}`.  Their
    name (first name, lower case, as a post's author) is no handle's and no other person's, and their Slack id no one else's."""
    collabs = reg.get("collaborators", [])
    if not isinstance(collabs, list):
        raise SlackError(f"registry {path}: collaborators is a list of {{name, slack_user, email, note}}")
    taken = {person_name(p) for p in reg.get("humans", [])} | handles
    ids = {p.get("slack_user") for p in reg.get("humans", [])} - {None}
    for p in collabs:
        if not isinstance(p, dict) or set(p) - {"name", "slack_user", "email", "note"} or not {"name", "slack_user", "note"} <= set(p):
            raise SlackError(f"registry {path}: a collaborator is {{name, slack_user, email (optional), note}}, not {p!r}")
        if not isinstance(p["name"], str) or not p["name"].strip() or not isinstance(p["note"], str) or not p["note"].strip():
            raise SlackError(f"registry {path}: collaborator {p.get('name')!r} needs a name and a note")
        if not NAME_RE.match(name := person_name(p)) or name in taken:
            raise SlackError(f"registry {path}: collaborator {p['name']!r} posts as {name!r}, which a handle or another person has")
        if p["slack_user"] is not None and (not isinstance(p["slack_user"], str) or not SLACK_USER_RE.match(p["slack_user"])
                                            or p["slack_user"] in ids):
            raise SlackError(f"registry {path}: collaborator {p['name']!r}'s slack_user {p['slack_user']!r}: a Slack user id "
                             "(U...) no other person has, or null")
        if p.get("email") is not None and not (isinstance(p["email"], str) and EMAIL_RE.match(p["email"])):
            raise SlackError(f"registry {path}: collaborator {p['name']!r}'s email {p['email']!r} is not an address")
        taken.add(name)
        ids.add(p["slack_user"])


def person_name(p: dict) -> str:
    """A named person as a post's author: first name, lower case (`daniel`)."""
    words = str(p.get("name") or "").split()
    return words[0].lower() if words else ""


def people(reg: dict) -> list[tuple[dict, str]]:
    """The named people with their role: `human` (the registry's `humans`, who authorize) or `collaborator` (who may ask)."""
    return [(p, "human") for p in reg.get("humans", [])] + [(p, "collaborator") for p in reg.get("collaborators", [])]


def person(msg: dict | None, reg: dict) -> tuple[dict, str] | None:
    """The named person who typed `msg` (by Slack user id) and their role, or None.  An entry without an id is nobody."""
    if not msg or msg.get("bot_id"):
        return None
    return next(((p, role) for p, role in people(reg) if p.get("slack_user") and msg.get("user") == p["slack_user"]), None)


def people_names(reg: dict) -> set[str]:
    """The names the named people post as: they read Slack itself, so no doorbell rings them."""
    return {person_name(p) for p, _ in people(reg) if p.get("slack_user")}


def owner(reg: dict, path: str) -> str | None:
    """The handle that owns a checkout path: the one whose `owns` entry is its longest prefix (a directory ends in `/`)."""
    path = path.removeprefix("./")
    best = max(((p, h["handle"]) for h in reg["handles"] for p in h.get("owns", [])
                if path == p.rstrip("/") or path.startswith(p if p.endswith("/") else p + "/")),
               key=lambda ph: len(ph[0]), default=None)
    return best and best[1]


def entry(reg: dict, handle: str) -> dict:
    handle = handle.lstrip("@")
    for h in reg["handles"]:
        if h["handle"] == handle:
            return h
    raise SlackError(f"unknown handle @{handle}; the registry has " + ", ".join("@" + h["handle"] for h in reg["handles"]))


def subscribers(reg: dict, key: str) -> list[str]:
    subs = reg["channels"][key]["subscribers"]
    return [h["handle"] for h in reg["handles"]] if subs == EVERY else list(subs)


def _env(key: str) -> str:
    return "SLACK_CHANNEL_" + key.upper().replace("-", "_").replace(".", "_")


def resolve_key(reg: dict, key: str) -> str:
    """A channel's registry key, from its key or an old name it keeps as an alias (`approvals` for `ask-daniel`)."""
    return next((k for k, c in reg["channels"].items() if key in c.get("aliases", [])), key)


def channel_key(ns: argparse.Namespace, reg: dict) -> str:
    key = resolve_key(reg, (ns.channel or FRONT_DOOR).lstrip("#"))
    if key not in reg["channels"]:
        raise SlackError(f"no channel #{key} in the registry (it has {', '.join(sorted(reg['channels']))})")
    return key


def channel_id(ns: argparse.Namespace, reg: dict, key: str | None = None) -> str:
    key = key or channel_key(ns, reg)
    cid = (ns.channel_id if key == channel_key(ns, reg) else None) or os.environ.get(_env(key)) or reg["channels"][key].get("id")
    if not cid:
        raise SlackError(f"no channel id for #{key}: pass --channel-id, set {_env(key)}, or record its id in the registry")
    return cid


def key_of(reg: dict, cid: str) -> str | None:
    return next((k for k, c in reg["channels"].items() if cid in (c.get("id"), os.environ.get(_env(k)))), None)


def tag(h: dict) -> str:
    if h.get("slack_user"):
        return f"<@{h['slack_user']}>"                              # a person: a real mention, so Slack notifies them
    return f"<!subteam^{h['usergroup']}|@{h['handle']}>" if h.get("usergroup") else f"@{h['handle']}"


def human_entry(reg: dict, name: str) -> dict | None:
    """A named human as an addressee: `daniel` (first name, lower case), with the Slack user id that notifies them."""
    name = name.lstrip("@").lower()
    return next(({"handle": name, "slack_user": p["slack_user"]} for p in reg.get("humans", [])
                 if p.get("slack_user") and p["name"].split()[0].lower() == name), None)


# --- the Web API ------------------------------------------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("redirected; refusing to forward the bearer token")


def post(url: str, data: bytes, headers: dict[str, str], timeout: float = 30) -> dict:
    """The one HTTP exchange: POST `data`, parse the JSON answer. Raises urllib's errors for the caller to map."""
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout) as r:
        return json.load(r)


# The Cursor identity socket, as verity-github.py (https://website-docs-sage.vercel.app/agent-tools/verity-github.py, sha256
# 34ea37fb0126e20a22160649e502565d7b443d6bfd9feb438528bff3163a7886) reads it in `oidc_token()`.
class _UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path: str):
        super().__init__("cursor-agent", timeout=10)
        self.unix_path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.unix_path)


#: the waits before asking the identity socket again when a connection to it is refused, dropped or times out; a socket that
#: isn't there is not asked again (this is no Cursor agent)
OIDC_RETRY_S = (1, 4)


def oidc_token(aud: str, sleep: Callable[[float], Any] = time.sleep) -> str:
    """A Cursor OIDC token for this agent with audience `aud`, from the local cursor-agent socket. Asking again is safe: a token
    request changes nothing."""
    path = os.environ.get("CURSOR_AGENT_SOCKET", "/run/cursor/api.sock")
    hint = "the Slack broker needs is unavailable (not a Cursor agent? set SLACK_BOT_TOKEN)" if aud == BROKER else \
        f"{aud} needs is unavailable (only a Cursor agent has one)"
    if not os.path.exists(path):                     # checked first: connecting to a long path fails with a plain OSError
        raise SlackError(f"the Cursor identity socket {hint}")
    for wait in (*OIDC_RETRY_S, None):
        conn = _UnixHTTP(path)
        try:
            conn.request("POST", "/v1/tokens/oidc", json.dumps({"aud": aud}), {"Content-Type": "application/json"})
            r = conn.getresponse()
            status, body = r.status, r.read(32768)
            break
        except (OSError, http.client.HTTPException) as e:
            if wait is None or isinstance(e, FileNotFoundError):
                raise SlackError(f"the Cursor identity socket {hint}") from None
        finally:
            conn.close()
        sleep(wait)
    if status == 403:
        raise SlackError(f"Cursor refused this agent's identity ({aud} needs a Cursor agent's OIDC token)")
    if status != 200:
        raise SlackError(f"the Cursor identity service is unavailable (HTTP {status})")
    try:
        token = json.loads(body).get("token")
    except (ValueError, AttributeError):
        token = None
    if not isinstance(token, str) or not token or any(c.isspace() for c in token):
        raise SlackError("Cursor returned an invalid identity token")
    return token


class _Identity:
    """This agent's OIDC token for one audience, fetched when first needed and again when the server rejects a cached one.
    Cursor's tokens live 300 s, so a long-running client's (a relay, `ask-daniel --wait`) goes stale; a 401 means the server
    refused before acting, so the one retry can't post or file anything twice."""

    def __init__(self, aud: str):
        self.aud, self.token = aud, None

    def send(self, request: Callable[[str], Any]) -> Any:
        cached = self.token is not None
        self.token = self.token or oidc_token(self.aud)
        try:
            return request(self.token)
        except urllib.error.HTTPError as e:
            if e.code != 401 or not cached:
                raise
        self.token = oidc_token(self.aud)
        return request(self.token)


def via(token: str | None) -> str:
    """`direct` (the bot token, to Slack) or `broker`: the broker only without a token, or when RESEARCH_SLACK_VIA says so."""
    forced = os.environ.get("RESEARCH_SLACK_VIA", "")
    if forced not in ("", "direct", "broker"):
        raise SlackError(f"RESEARCH_SLACK_VIA={forced!r}: expected broker or direct")
    return forced or ("direct" if token else "broker")


def _code(e: urllib.error.HTTPError) -> str:
    """The broker's short error code (`{"error": "method_not_allowed"}`), or ''; nothing else of a response is echoed."""
    try:
        code = json.loads(e.read(4096)).get("error")
    except (OSError, ValueError, AttributeError):
        return ""
    return f" {code}" if isinstance(code, str) and ERROR_CODE_RE.match(code) else ""


def broker_error(method: str, e: urllib.error.HTTPError) -> str:
    why = f"HTTP {e.code}{_code(e)}"
    if e.code == 401:
        return f"{method}: the Slack broker rejected this agent's Cursor identity ({why}); set SLACK_BOT_TOKEN or ask @infra"
    if e.code == 403:
        return f"{method}: the Slack broker refused the call ({why}): it serves compute-verification/verity's agents and BROKER_METHODS only"
    if e.code == 413:
        return f"{method}: the request is larger than the broker takes ({why}); put the content in a file and link it"
    if e.code == 429:
        return f"{method}: the Slack broker is rate limiting ({why}; retry after {e.headers.get('Retry-After')} s)"
    if e.code >= 500:
        return f"{method}: the Slack broker is down or switched off ({why}); set SLACK_BOT_TOKEN or ask @infra"
    return f"{method}: the Slack broker answered {why}"


class Slack:
    """A minimal Web API client: form-encoded POSTs with the bot token (`direct`), or the same arguments as JSON to the broker
    with this agent's OIDC token (`broker`, see `via`). With `dry_run`, nothing is sent and no token is fetched: each call is
    printed (method and parameters, never a token) and answered with an empty `ok` response."""

    def __init__(self, token: str | None, *, base: str | None = None, dry_run: bool = False, log=None, transport=None):
        self.via = via(token)
        self.token = token if self.via == "direct" else None
        self.dry_run, self.log, self.transport, self._oidc = dry_run, log, transport or post, _Identity(BROKER)
        self.base = (base or os.environ.get("SLACK_API_URL") or DEFAULT_API).rstrip("/") + "/"
        if not dry_run and self.via == "direct" and not token:
            raise SlackError("SLACK_BOT_TOKEN is not set (a Cursor secret; see the using-slack skill), or pass --dry-run")

    def call(self, method: str, **params: Any) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        if self.dry_run:
            print(f"dry-run: {method} {json.dumps(params, sort_keys=True)}" + (" (via the broker)" if self.via == "broker" else ""),
                  file=self.log or sys.stderr)
            return {"ok": True, "dry_run": True}
        args = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in params.items()}
        for attempt in range(RATE_RETRIES + 1):
            try:
                out = self._broker(method, args) if self.via == "broker" else self.transport(
                    self.base + method, urllib.parse.urlencode(args).encode(),
                    {"Authorization": f"Bearer {self.token}", "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < RATE_RETRIES:                 # a burst of doorbells: wait as asked, then again
                    _sleep(min(float(e.headers.get("Retry-After") or 1), RATE_WAIT_MAX_S))
                    continue
                if self.via == "broker":
                    raise SlackError(broker_error(method, e)) from None
                hint = f" (rate limited; retry after {e.headers.get('Retry-After')} s)" if e.code == 429 else ""
                raise SlackError(f"{method}: HTTP {e.code}{hint}") from None
            except (urllib.error.URLError, OSError, ValueError) as e:
                where = "the Slack broker is unreachable" if self.via == "broker" else type(e).__name__
                raise SlackError(f"{method}: {where}: {getattr(e, 'reason', e)}") from None
        if not isinstance(out, dict):
            raise SlackError(f"{method}: the answer is not a JSON object")
        if not out.get("ok"):
            error, needed = out.get("error"), out.get("needed")     # echoed only when they are codes: an answer is not to be printed
            raise SlackError(f"{method}: {error if isinstance(error, str) and ERROR_CODE_RE.fullmatch(error) else 'not ok'}"
                             + (f" (needed scope {needed})" if isinstance(needed, str) and SCOPES_RE.fullmatch(needed) else ""))
        return out

    def _broker(self, method: str, args: dict[str, str]) -> dict:
        if method not in BROKER_METHODS:
            raise SlackError(f"{method}: not in BROKER_METHODS, so the broker would refuse it")
        body = json.dumps({"method": method, "args": args}).encode()
        if len(body) > BROKER_MAX_BODY:
            raise SlackError(f"{method}: {len(body)} bytes is more than the broker takes ({BROKER_MAX_BODY}); put the content in a file and link it")
        return self._oidc.send(lambda t: self.transport(BROKER, body, {"Authorization": f"Bearer {t}", "Content-Type": "application/json"}))

    def paged(self, method: str, key: str, limit: int | None = None, **params: Any) -> list[dict]:
        if method == "conversations.replies":
            return self._replies(key, limit, **params)
        items: list[dict] = []
        cursor = None
        while True:
            out = self.call(method, limit=PAGE, cursor=cursor, **params)
            items += out.get(key, [])
            cursor = (out.get("response_metadata") or {}).get("next_cursor")
            if not cursor or (limit and len(items) >= limit):
                return items[:limit] if limit else items

    def _replies(self, key: str, limit: int | None, **params: Any) -> list[dict]:
        """A thread's replies, paged by moving `oldest` past the last one seen: Slack's cursor for conversations.replies
        drops the message at each page boundary (a doorbell lost from a 204-reply inbox read, 18 Oct)."""
        items: list[dict] = []
        seen: set[str] = set()
        while True:
            out = self.call("conversations.replies", limit=PAGE, **params)
            page = [m for m in out.get(key, []) if m.get("ts") not in seen]
            seen.update(m.get("ts") for m in page)
            items += page
            cursor = (out.get("response_metadata") or {}).get("next_cursor")
            if not (out.get("has_more") or cursor) or (limit and len(items) >= limit):
                return items[:limit] if limit else items
            newest = max((m["ts"] for m in page if m.get("ts") and m["ts"] != params.get("ts")), key=float, default=None)
            if newest is not None and float(newest) > float(params.get("oldest") or 0):
                params.pop("cursor", None)
                params["oldest"] = newest
            elif cursor and cursor != params.get("cursor"):
                params["cursor"] = cursor                                # no progress by `oldest` (a tiny page): one by cursor
            else:
                return items[:limit] if limit else items

    def root(self, channel: str, thread: str) -> dict | None:
        msgs = self.call("conversations.replies", channel=channel, ts=thread, limit=1).get("messages", [])
        return next((m for m in msgs if m.get("ts") == thread), None)


def client(ns: argparse.Namespace) -> Slack:
    return Slack(os.environ.get("SLACK_BOT_TOKEN"), dry_run=ns.dry_run)


def _footer(tags: tuple[str, ...] = ()) -> str:
    """`_ask · urgent_`: the post's kind and flags, which `list` and `traffic` read.  The author is the post's username, and its holder is in the directory."""
    tags = tuple(t for t in tags if t)
    return f"\n\n_{TAG_SEP.join(tags)}_" if tags else ""


def plain(text: str) -> str:
    """A post as an agent should read it: no footer, Slack's markup undone (names, dates, links, escapes)."""
    text = OLD_FOOTER_RE.sub("", FOOTER_RE.sub("", text or ""))
    text = SUBTEAM_RE.sub(lambda m: "@" + (m[2] or m[1]).lstrip("@"), text)
    text = DATE_TOKEN_RE.sub(r"\1", text)
    text = LINK_RE.sub(lambda m: m[1] if not m[2] or m[2] == m[1] else f"{m[2]} ({m[1]})", text)
    return re.sub(r"\n{3,}", "\n\n", html.unescape(text)).strip()


def _pieces(text: str, limit: int = POST_MAX) -> list[str]:
    """`text` cut at line breaks (else at a space outside `<...>`) into pieces of at most `limit` bytes."""
    out = []
    while len(text.encode()) > limit:
        window = text.encode()[:limit].decode(errors="ignore")
        cut = window.rfind("\n")
        if cut < len(window) // 2:
            depth, cut = 0, len(window)
            for i, c in enumerate(window):
                depth = max(0, depth + (c == "<") - (c == ">"))
                if c == " " and depth == 0 and i:
                    cut = i
        out.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return [p for p in out if p] + [text]


def _post(api: Slack, ch: str, handle: str, text: str, thread: str | None = None, *,
          route: tuple[dict, Routes | None] | None = None, urgent: bool = False, tags: tuple[str, ...] = ()) -> str:
    """Post, print `<ts> <permalink>`, and with `route` (the registry and the routes) deliver it to whoever it reaches."""
    tags = tuple(tags) + (("urgent",) if urgent and "urgent" not in tags else ())
    head, *rest = _pieces(slack_dates(text))
    text = f"{head}{_footer(tags)}"
    if thread and not api.dry_run:                       # a reply's ts (a doorbell's `reply R`): its thread
        try:
            first = api.call("conversations.replies", channel=ch, ts=thread, limit=1).get("messages", [])
        except SlackError as e:                          # a channel the bot may post in but not read (#ask-the-team)
            if not any(w in str(e) for w in ("missing_scope", "not_in_channel")):
                raise
            first = []
        thread = (first[0].get("thread_ts") or first[0].get("ts") or thread) if first else thread
    out = api.call("chat.postMessage", channel=ch, thread_ts=thread, username=handle, text=text,
                   unfurl_links="false", unfurl_media="false")
    ts = out.get("ts") or "DRY-RUN"
    for i, piece in enumerate(rest, 2):                  # quiet, and led by its number, so the rest of a post wakes nobody
        api.call("chat.postMessage", channel=ch, thread_ts=thread or ts, username=handle, unfurl_links="false",
                 unfurl_media="false", text=f"({i}/{len(rest) + 1}) {piece}{_footer((QUIET,))}")
    try:
        link = api.call("chat.getPermalink", channel=ch, message_ts=ts).get("permalink") or "-"
    except SlackError:                                   # the post went out: a caller that retries on a raise would post it twice
        link = "-"
    print(f"{ts} {link}")
    if route and route[1] is not None and not api.dry_run:
        msg = {"text": text, "ts": ts, "thread_ts": thread, "bot_id": route[0].get("bot_id"), "username": handle}
        try:
            deliver(api, route[0], route[1], ch, msg)
        except SlackError as e:
            print(f"route: posted, but delivery stopped ({e}); the post itself stands: `research msg deliver --ts {ts}"
                  + (f" --thread {thread}" if thread else "") + "` rings what it missed", file=sys.stderr)
    return ts


# --- addressing and authorship ----------------------------------------------------------------------------------------------

def parse_event(raw: str) -> dict:
    """A Slack message event (bare, or wrapped as `{"event": ...}`), or plain text as `{"text": raw}`."""
    if raw == "-":
        raw = sys.stdin.read()
    elif raw.startswith("@") and Path(raw[1:]).is_file():
        raw = Path(raw[1:]).read_text()
    try:
        ev = json.loads(raw)
    except ValueError:
        return {"text": raw}
    if not isinstance(ev, dict):
        return {"text": raw}
    ev = ev.get("event", ev)
    if ev.get("subtype") == "message_changed" and isinstance(ev.get("message"), dict):
        ev = {"channel": ev.get("channel"), **ev["message"]}
    return ev


def by_handle(msg: dict | None, handle: str) -> bool:
    """Posted by this handle: a bot message under its username."""
    return bool(msg) and bool(msg.get("bot_id")) and msg.get("username") == handle


def mentions(text: str, h: dict, reg: dict) -> str | None:
    """Why `text` names the registry entry `h`, or None.  A user-group mention names its id's group, which Slack shows and
    notifies whatever the label says; the label counts only for an id no handle has (`lead_names` reads it so too)."""
    for gid, label in SUBTEAM_RE.findall(text or ""):
        if h.get("usergroup") and gid == h["usergroup"]:
            return f"user-group mention <!subteam^{gid}>"
        if label.lstrip("@") == h["handle"] and not any(o.get("usergroup") == gid for o in reg["handles"]):
            return f"user-group mention labelled @{h['handle']}"
    if re.search(rf"(?<![\w@./-])@{re.escape(h['handle'])}(?![\w-])", SUBTEAM_RE.sub(" ", text or "")):
        return f"plain-text @{h['handle']}"
    return None


def tagged(text: str, reg: dict) -> list[str]:
    return [h["handle"] for h in reg["handles"] if mentions(text, h, reg)]


def is_announcement(msg: dict | None) -> bool:
    return bool(msg) and (msg.get("text") or "").startswith((ANNOUNCE, SHORTCODES[ANNOUNCE]))


def to_every_handle(text: str) -> bool:
    """Whether an announcement's header addresses every handle: it says `to every handle`, or (an older header) names no
    audience.  A header whose audience resolves to no handle (a user group the registry lacks) is for no one, not all."""
    first = (text or "").split("\n", 1)[0]
    return " to " not in first or first.split(" to ", 1)[1].strip().startswith("every handle")


def audience(text: str, reg: dict) -> list[str]:
    """The handles an announcement's header tags after ` to ` (the header also names its author); see `to_every_handle`."""
    first = (text or "").split("\n", 1)[0]
    return tagged(first.split(" to ", 1)[1], reg) if " to " in first else []


def is_status(text: str) -> bool:
    return bool(STATUS_RE.match(text or ""))


def addressed(ev: dict, h: dict, reg: dict, key: str | None, api: Slack | None) -> str | None:
    """Why the post `ev` (in channel `key`, when known) is addressed to `h`, or None."""
    handle, text = h["handle"], ev.get("text", "")
    if by_handle(ev, handle):
        return None
    thread = ev.get("thread_ts")
    if not thread or thread == ev.get("ts"):
        if key and key != FRONT_DOOR and handle in subscribers(reg, key):
            return f"top-level post in #{key}, which @{handle} subscribes to"
        if why := mentions(text, h, reg):
            return why
        if is_announcement(ev) and to_every_handle(text):
            return "an announcement to every handle"
        return None
    if not is_status(text) and (why := mentions(text, h, reg)):
        return why
    if api is None:
        return None
    ch = ev.get("channel") or (channel_id_for(reg, key) if key else None)
    if not ch:
        raise SlackError("the event is a thread reply with no channel, and no --channel id is known")
    msgs = api.paged("conversations.replies", "messages", channel=ch, ts=thread)
    root = next((m for m in msgs if m.get("ts") == thread), None)
    if is_status(text):
        return f"a status line on @{handle}'s post" if by_handle(root, handle) else None
    started = f"thread {thread} (started {both(thread)})"
    if by_handle(root, handle) or any(by_handle(m, handle) for m in msgs if m.get("ts") != ev.get("ts")):
        return f"{started}: @{handle} posted in it"
    if root and mentions(root.get("text", ""), h, reg):
        return f"{started}: its root addresses @{handle}"
    if key and key != FRONT_DOOR and handle in subscribers(reg, key):
        return f"{started} in #{key}, which @{handle} subscribes to"
    return None


# --- routing: `@name` at the start of a post wakes that name's agent ---------------------------------------------------------
# The rule is `route_targets`, a pure function pinned by tests/slack_routing_vectors.json.  Delivery is a doorbell: the post itself (`doorbell`), as a reply in the
# target's inbox thread, which its agent subscribes to.

def ours(msg: dict | None, reg: dict) -> bool:
    """Posted by Verity's bot, the registry's `bot_id`: another app's bot may take a handle's username too.  With no
    `bot_id` recorded, no bot is ours."""
    return bool(msg) and bool(reg.get("bot_id")) and msg.get("bot_id") == reg["bot_id"]


def poster(msg: dict | None, reg: dict) -> str | None:
    """Who wrote `msg`, as a name: our bot's username, or `daniel`-style for a named human or a collaborator (first name,
    lower case)."""
    if not msg:
        return None
    if msg.get("bot_id"):
        return (msg.get("username") or None) if ours(msg, reg) else None
    return person_name(who[0]) if (who := person(msg, reg)) else None


def lead_names(text: str, reg: dict) -> list[str]:
    """The names a post addresses: the `@name`s and user-group mentions at its very start (after one emoji at most), before
    any other word, or a handle written bare as the first word with a colon (`lean: ...`)."""
    m = LEAD_RE.match(text or "")
    if not m and (bare := BARE_LEAD_RE.match(text or "")) and any(h["handle"] == bare[1] for h in reg["handles"]):
        return [bare[1]]
    names: list[str] = []
    for at, gid, label in NAME_TOKEN_RE.findall(m.group(1) if m else ""):
        known = next((h["handle"] for h in reg["handles"] if gid and h.get("usergroup") == gid), "")
        n = at.rstrip("._-") if at else known or (label or "").lstrip("@")     # Slack shows and notifies the id's group
        if n and n not in names:
            names.append(n)
    return names


UNROUTED_SKIP_RE = re.compile(r"```.*?```|`[^`\n]*`|<[^>\n]*>|^\s*>.*$", re.S | re.M)   # code, links, quoted lines
LATER_AT_RE = re.compile(r"(?<![\w.@/-])@([a-z0-9][a-z0-9._-]*)")
LATER_NAME_RE = re.compile(r"(?:^|(?<=[.!?;])[ \t]+)[ \t]*([a-z0-9][a-z0-9._-]*):\s", re.M)


def unrouted_names(body: str, reg: dict, author: str, rung: tuple[str, ...] = ()) -> list[str]:
    """The handles other than `author`, the `rung` ones and the lead's that a body names where no doorbell reaches: `@handle` after its
    lead, or `handle:` right after the lead, starting a later line or starting a sentence (only a post's start routes, `lead_names`)."""
    body = body or ""
    m = LEAD_RE.match(body)
    rest = UNROUTED_SKIP_RE.sub("", body[m.end() if m else 0:])
    found = [n.rstrip("._-") for n in LATER_AT_RE.findall(rest)] + LATER_NAME_RE.findall(rest)
    handles = {h["handle"] for h in reg["handles"]} - {author, *rung, *lead_names(body, reg)}
    return list(dict.fromkeys(n for n in found if n in handles))


def refuse_unrouted(body: str, reg: dict, author: str, rung: tuple[str, ...] = ()) -> None:
    if names := unrouted_names(body, reg, author, rung):
        raise SlackError(f"it names {' '.join('@' + n for n in names)} past its start, where nothing rings: only the names a "
                         "post starts with wake anyone. Move them to the start (--to, or the leading @names) to ring them, "
                         "or drop the @ (and don't start a line or sentence with `name:`) to mention them without a doorbell")


def header_names(text: str, reg: dict) -> list[str]:
    """The names an announcement's header (`📣 *Announcement* from @x to @a @b`) addresses; empty for `to every handle`."""
    first = (text or "").split("\n", 1)[0]
    return lead_names(first.split(" to ", 1)[1], reg) if " to " in first else []


def with_routes(reg: dict, routes: "Routes | None") -> dict:
    """`reg` with each handle's roster from its open inbox route: `holder` names the route's holder and `posters` (the
    other agents that post as it), and `branch_suffixes` are the route's.  The route is the record of both (`names open`
    writes it); the registry's copies are a fallback for readers without the routes (11 of 13 holders a generation stale
    on 15 Oct), kept before the route's.  Every handle gets a `holder`, empty when none is known.  Best effort: a route
    that can't be read adds nothing."""
    if routes is None:
        return reg
    routes.refresh([h["handle"] for h in reg["handles"] if h["handle"] not in routes.local and h["handle"] not in routes._fetched])
    handles = []
    for h in reg["handles"]:
        try:
            r = routes.get_route(h["handle"])
        except SlackError:
            r = None
        ids = [r["holder"], *r.get("posters", [])] if r and not r.get("closed") and r.get("holder") else []
        sfx = list(dict.fromkeys([*h.get("branch_suffixes", []), *(r.get("branch_suffixes", []) if ids else [])]))
        held = ", ".join(ids)
        holder = f"{h['holder']}; its inbox's holder ({held})" if h.get("holder") and held else h.get("holder") or held
        handles.append({**h, "holder": holder, **({"branch_suffixes": sfx} if sfx else {})})
    return {**reg, "handles": handles}


def _branch_suffix() -> str | None:
    """The last 4 characters of this checkout's `cursor/...-XXXX` branch: the suffix this agent's PRs carry."""
    try:
        b = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.fullmatch(r"cursor/.+-([0-9a-z]{4})", b)
    return m[1] if m else None


def branch_owners(branch: str, reg: dict) -> list[str]:
    """The handles whose agent made `branch`, in `reg` as `with_routes` folds it: its last 4 characters are the last 4 of
    a holder's agent id, or one of the handle's `branch_suffixes` (its earlier holders', and its holder's other branches)."""
    suffix = branch.rsplit("-", 1)[-1]
    return [h["handle"] for h in reg["handles"] if len(suffix) == 4 and
            suffix in [a[-4:] for a in AGENT_ID_RE.findall(h.get("holder") or "") if len(a) > 11] + h.get("branch_suffixes", [])]


def named_prs(text: str) -> tuple[list[str], list[str]]:
    """The PR numbers a post names (`#1585`, and those a train-prep branch's name lists) and the other `cursor/` branches it
    names, each in order."""
    nums, branches = [], []
    for n in PR_RE.findall(text or ""):
        nums.append(n)
    for b in BRANCH_RE.findall(text or ""):
        if m := TRAIN_PREP_RE.match(b):
            nums += m[1].rstrip("-").split("-")
        else:
            branches.append(b)
    return list(dict.fromkeys(nums)), list(dict.fromkeys(branches))


def run_ids(text: str) -> set[str]:
    """The run ids a post names, as their last 4 characters (`dd21` for r20261009-035518-dd21); a bare 4-character word counts
    only with a letter in it, so a year or a PR number doesn't."""
    return {a or b for a, b in RUN_ID_RE.findall(SUBTEAM_RE.sub(" ", text or "")) if a or re.search(r"[a-f]", b)}


def pr_watchers(msg: dict, reg: dict, history: list[dict] | tuple = (), branches: dict[str, str] | None = None) -> list[str]:
    """Who a reply under an announcement (the merge thread is one) wakes besides its addressees: the owners of each PR it
    names (`branch_owners` of the PR's branch, from `branches`: PR number -> head branch), and each handle that, in an earlier
    post of `history` (the thread's recent replies) naming one of its own PRs, named a run id this post names, such as the
    train its PR waits on.  A PR whose owner isn't known wakes whoever named it in `history`: unsure, it wakes."""
    branches = branches or {}

    def owners(text: str) -> tuple[list[str], list[str]]:
        nums, direct = named_prs(text)
        known, unknown = [], []
        for n in nums:
            got = branch_owners(branches[n], reg) if n in branches else []
            known += got
            unknown += [] if got else [n]
        for b in direct:
            known += branch_owners(b, reg)
        return known, unknown

    text = msg.get("text") or ""
    out, unknown = owners(text)
    ids, tied = run_ids(text), []
    for m in history:
        who, said = poster(m, reg), m.get("text") or ""
        if not who or is_status(said):
            continue
        if any(n in named_prs(said)[0] for n in unknown):
            out.append(who)
        if who in owners(said)[0] and ids & run_ids(said):
            tied.append(who)
    return list(dict.fromkeys(out + tied))


def route_targets(msg: dict, root: dict | None, reg: dict, inboxes: set[str] | frozenset = frozenset(),
                  history: list[dict] | tuple = (), branches: dict[str, str] | None = None) -> list[str]:
    """The names a post in #agent-coordination wakes, in order.  `msg` and `root` (its thread's root, None for a top-level
    post) are Slack messages; `inboxes` is every inbox thread's ts.  Nothing in an inbox thread routes; a status line wakes
    only the root's poster; otherwise the names at the start, an announcement's audience (every handle when it names
    none), and on a reply that names no one the root's poster or, when the poster replies, the names the root addressed.
    A quiet post wakes only the names at its start.  A reply under an announcement that names no one but its author, quiet
    or not, also wakes its `pr_watchers` that are handles (with `history` and `branches`, as there); one that names
    someone wakes only them, whatever PRs and runs its body mentions.  The author is never a target."""
    text, ts, thread = msg.get("text") or "", msg.get("ts"), msg.get("thread_ts")
    reply = bool(thread) and thread != ts
    if (thread or ts) in inboxes or text.startswith(INBOX_HEAD) or \
            (reply and root and root.get("bot_id") and (root.get("text") or "").startswith(INBOX_HEAD)):
        return []
    author, by = poster(msg, reg), poster(root, reg) if reply else None
    if author is None:
        return []                                                    # only our bot and the named people route
    if is_quiet(text):
        out = [] if is_status(text) else lead_names(text, reg)       # people write (quiet) for "no reply needed"
    elif reply and is_status(text):
        out = [by]
    else:
        out = lead_names(text, reg)
        if not reply and is_announcement(msg):
            out += [h["handle"] for h in reg["handles"]] if to_every_handle(text) else header_names(text, reg)
        if reply and set(out) - {author}:
            pass                                                     # a reply that names someone else is for them alone
        elif reply and by and by != author:
            out.append(by)
        elif reply and by:
            rt = (root or {}).get("text") or ""
            out += lead_names(rt, reg) + (header_names(rt, reg) if is_announcement(root) else [])
    if reply and is_announcement(root) and not is_status(text) and not set(lead_names(text, reg)) - {author}:
        handles = {h["handle"] for h in reg["handles"]}
        out += [n for n in pr_watchers(msg, reg, history, branches) if n in handles]
    seen: list[str] = []
    for n in out:
        if n and n != author and n not in seen:
            seen.append(n)
    return seen


def is_quiet(text: str) -> bool:
    """Whether a post carries the quiet marker: as a footer tag, or as `(quiet)` ending its text before any footer.
    `(quiet)` elsewhere is prose."""
    text = text or ""
    body = FOOTER_RE.sub("", OLD_FOOTER_RE.sub("", text)).rstrip()
    return QUIET in tags_of(text) or body.endswith(QUIET)


def tags_of(text: str) -> list[str]:
    """The tags a post's footer carries: `_ask · urgent_` (or, before 1 Oct, `_posted by <url|@x's agent> · ask · urgent_`)."""
    m = FOOTER_RE.search(text or "") or OLD_FOOTER_RE.search(text or "")
    return [t.strip() for t in m.group(1).split(TAG_SEP.strip()) if t.strip()] if m else []


def kind_of(msg: dict | None) -> str:
    """A post's kind: its footer's first tag that isn't a flag, else what its shape says (announce, status, message)."""
    text = (msg or {}).get("text") or ""
    kind = next((t for t in tags_of(text) if t not in FLAGS), None)
    return kind or ("announce" if is_announcement(msg) else "status" if is_status(text) else "message")


def notes_root() -> Path:
    return Path(os.environ.get("RESEARCH_NOTES") or Path.home() / ".research" / "notes").expanduser()


def _route_ok(r: Any, name: str) -> bool:
    return isinstance(r, dict) and r.get("schema") == ROUTE_SCHEMA and r.get("name") == name and isinstance(r.get("inbox_ts"), str)


RAW_HOST = "raw.githubusercontent.com"                           # the notes are private: `http_get` sends it the notes token


def http_get(url: str, timeout: float = 10) -> bytes | None:
    """The body at `url` (http(s), or a `file://` directory for tests and offline use), or None when there is none.  A
    github.com raw file is read with RESEARCH_NOTES_TOKEN when it is set: the notes repository is private, and without
    the token its raw files answer 404 as if they weren't there."""
    if url.startswith("file://"):
        p = Path(urllib.parse.unquote(url[7:]))
        return p.read_bytes() if p.is_file() else None
    headers = {}
    if urllib.parse.urlsplit(url).hostname == RAW_HOST and (token := os.environ.get("RESEARCH_NOTES_TOKEN")):
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return r.read(1 << 20)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise SlackError(f"{url}: HTTP {e.code}") from None
    except (urllib.error.URLError, OSError) as e:
        raise SlackError(f"{url}: {getattr(e, 'reason', e)}") from None


def _newer(local: dict | None, pub: dict | None) -> dict | None:
    """Of a name's two copies, the clone's and the published one, the one that says what is true now.  A later `opened`
    wins (a reopen or a succession stamps it anew: console's reopened inbox, 16 Oct); of one opening, a closed copy wins (a
    close comes after its open: @hash's forward to @kernels, 18 Oct); otherwise the clone's, with the published copy's
    `branch_suffixes` and `posters` too (raw.githubusercontent serves it minutes stale, and a holder elsewhere adds them)."""
    if not local or not pub:
        return local or pub
    if local.get("opened") and pub.get("opened") and local["opened"] != pub["opened"]:
        return pub if pub["opened"] > local["opened"] else local
    if bool(local.get("closed")) != bool(pub.get("closed")):
        return local if local.get("closed") else pub
    out = dict(local)
    for k in ("branch_suffixes", "posters"):
        if v := list(dict.fromkeys([*local.get(k, []), *pub.get(k, [])])):
            out[k] = v
    return out


class Routes:
    """The routes table, `routes.d/<name>.json` in the notes: the local clone's, and the published files, read from
    $RESEARCH_ROUTES_URL (default the notes repo's raw files, with RESEARCH_NOTES_TOKEN) for a name the clone lacks (a
    clone not pulled since that agent registered) or when asked to (`fresh`, `refresh`); of a name's two copies, the
    newer counts (`_newer`)."""

    def __init__(self, root: Path | None = None, base: str | None = None, get: Callable[[str], bytes | None] = http_get):
        self.root, self.get = (root or notes_root()), get
        self.base = (base or os.environ.get("RESEARCH_ROUTES_URL") or ROUTES_RAW).rstrip("/") + "/"
        self.local: dict[str, dict] = {}
        d = self.root / ROUTES_DIR
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            try:
                r = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if _route_ok(r, p.stem):
                self.local[p.stem] = r
        self._fetched: dict[str, dict | None] = {}
        self._unread: set[str] = set()                           # names whose published copy couldn't be read

    def _fetch(self, name: str) -> dict | None:
        raw = self.get(f"{self.base}{name}.json")
        try:
            r = json.loads(raw) if raw else None
        except ValueError:
            r = None
        self._fetched[name] = r if _route_ok(r, name) else None
        return self._fetched[name]

    def get_route(self, name: str, fresh: bool = False) -> dict | None:
        if not NAME_RE.match(name):
            return None
        if fresh or (name not in self.local and name not in self._fetched):
            self._fetch(name)
        return _newer(self.local.get(name), self._fetched.get(name))

    def refresh(self, names: list[str] | set[str]) -> None:
        """Read the published copy of each name now, side by side; one that can't be read keeps what was known."""
        names = [n for n in dict.fromkeys(names) if NAME_RE.match(n)]

        def one(name: str) -> None:
            try:
                self._fetch(name)
            except SlackError:
                self._unread.add(name)
        if len(names) == 1:
            one(names[0])
        elif names:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
            with ThreadPoolExecutor(max_workers=min(16, len(names))) as pool:
                list(pool.map(one, names))

    def current(self, name: str) -> dict | None:
        """`name`'s route as the published table has it, read once per table (`deliver` reads every name it starts from so):
        a copy that can't be read leaves the clone's, so a name the clone lacks then has none."""
        if not NAME_RE.match(name):
            return None
        if name not in self._fetched and name not in self._unread:
            self.refresh([name])
        return _newer(self.local.get(name), self._fetched.get(name))

    def all(self) -> dict[str, dict]:
        return {n: r for n in {**self.local, **self._fetched} if (r := _newer(self.local.get(n), self._fetched.get(n)))}

    def inboxes(self) -> set[str]:
        return {r["inbox_ts"] for r in self.all().values()}

    def resolve(self, name: str) -> tuple[str, dict | None]:
        """Follow a closed route's `forward_to` (at most FORWARD_HOPS hops), each name as published (`current`): the name that
        gets the doorbell, and its route.  No route at the end, or one still closed (`unreachable`), is a dead end, a loop or
        a chain too long: nobody reads its inbox."""
        r = self.current(name)
        for _ in range(FORWARD_HOPS):
            if not r or not r.get("closed") or not r.get("forward_to"):
                break
            name, r = r["forward_to"], self.current(r["forward_to"])
        return name, r


FORWARD_HOPS = 3


def unreachable(name: str, to: str, r: dict | None) -> str | None:
    """Why `Routes.resolve(name)`'s answer `(to, r)` rings no inbox, or None when `r` is open."""
    if not r:
        return "no inbox" if to == name else f"it forwards to @{to}, which has no inbox: see `research msg names list`"
    if not r.get("closed"):
        return None
    if r.get("forward_to"):
        return f"its forward_to loops or runs past {FORWARD_HOPS} hops (stopped at @{to}): see `research msg names list`"
    return "finished, with no forward_to: message its coordinator instead"


def self_agent_id() -> str | None:
    """This agent's bc id: from $RESEARCH_AGENT_URL, else the `cloud_agent_id` claim of its Cursor identity token."""
    if m := AGENT_ID_RE.search(os.environ.get("RESEARCH_AGENT_URL", "")):
        return m[0]
    try:
        part = oidc_token(BROKER).split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except (SlackError, IndexError, ValueError):
        return None
    m = AGENT_ID_RE.fullmatch(str(claims.get("cloud_agent_id") or ""))
    return m[0] if m else None


_CHANNEL = r"#[A-Za-z0-9][A-Za-z0-9._-]*"
#: groups: author, urgent, thread, reply; a post outside the front door names its channel before `thread` (`doorbell_channel`)
DOORBELL_RE = re.compile(rf"^from ([a-z0-9][a-z0-9._-]*)( \(URGENT\))?, (?:{_CHANNEL} )?thread (\d+\.\d+)(?:, reply (\d+\.\d+))?: ")
DOORBELL_CHANNEL_RE = re.compile(rf"^from [a-z0-9][a-z0-9._-]*(?: \(URGENT\))?, ({_CHANNEL}) thread \d")
CHANNEL_ID_RE = re.compile(r"^[CDG][A-Z0-9]{2,}$")


def _channel_flag(channel: str) -> str:
    return f"--channel-id {channel}" if CHANNEL_ID_RE.match(channel) else f"--channel {channel}"


def doorbell(author: str, thread: str, ts: str, text: str, urgent: bool = False, channel: str | None = None) -> str:
    """The post itself, so its reader can act without another read: `from infra, thread T[, reply R]: <text>`.  A post in a
    channel other than the front door names it (its registry key, else its id), so its reader answers there: `from daniel,
    #ask-the-team thread T: <text>`.  Never starts with `@`, so it can't route itself."""
    body = plain(text)
    if len(body) > DOORBELL_MAX:
        read = f"research msg read {_channel_flag(channel)} --thread {thread}" if channel else f"research msg read --thread {thread}"
        body = body[:DOORBELL_MAX].rstrip() + f"\u2026 (the rest: `{read}`)"
    where = (f"#{channel} " if channel else "") + f"thread {thread}" + (f", reply {ts}" if ts and ts != thread else "")
    return f"from {author}{' (URGENT)' if urgent else ''}, {where}: {body}"


def doorbell_channel(text: str) -> str | None:
    """The channel a doorbell names (`ask-the-team`), or None for the front door's."""
    return m[1][1:] if (m := DOORBELL_CHANNEL_RE.match(text or "")) else None


def is_doorbell(m: dict, reg: dict) -> bool:
    """A reply in an inbox thread that the router posted on Verity's bot.  Anything else there, a line someone typed or
    another app's bot, is no doorbell however it reads."""
    return m.get("username") == "router" and ours(m, reg)


def _from(text: str, author: str) -> bool:
    m = DOORBELL_RE.match(text)
    return bool(m) and (m[1] == author or (m[1] == "router" and f": paused @{author} " in text))


def _mentions_thread(text: str, thread: str, channel: str | None = None) -> bool:
    return bool(m := DOORBELL_RE.match(text)) and m[3] == thread and doorbell_channel(text) == channel


def _rang(text: str, ts: str, channel: str | None = None) -> bool:
    """Whether the doorbell `text` is for the post `ts` (in `channel`, None for the front door)."""
    return bool(m := DOORBELL_RE.match(text)) and (m[4] or m[3]) == ts and doorbell_channel(text) == channel


def pr_branches(nums: set[str], log=None) -> dict[str, str]:
    """PR number -> head branch, for `pr_watchers`: from the cache ($RESEARCH_PR_BRANCHES, default
    ~/.cache/research/pr-branches.json; a PR's branch never changes), and for numbers it lacks one `gh pr list`, whose answers
    join the cache.  Without gh, or when it fails, the numbers stay unknown and their PRs wake whoever named them."""
    path = Path(os.environ.get("RESEARCH_PR_BRANCHES") or Path.home() / ".cache" / "research" / "pr-branches.json")
    try:
        cache = json.loads(path.read_text())
    except (OSError, ValueError):
        cache = {}
    if nums - cache.keys():
        try:
            out = subprocess.run([GH, "pr", "list", "--repo", GH_REPO, "--state", "all",
                                  "--limit", "1000", "--json", "number,headRefName"], capture_output=True, text=True, timeout=60)
            if out.returncode:
                raise OSError(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else f"exit {out.returncode}")
            cache.update({str(p["number"]): p["headRefName"] for p in json.loads(out.stdout)})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, sort_keys=True))
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            print(f"route: PR owners unknown for {' '.join('#' + n for n in sorted(nums - cache.keys()))} (gh: {e}); "
                  "whoever named them in the thread is rung", file=log or sys.stderr)
    return {n: cache[n] for n in nums if n in cache}


def watch_context(api: Slack, ch: str, msg: dict, root: dict | None, now: float, log=None) -> tuple[list[dict], dict[str, str]]:
    """For a reply under an announcement, what `pr_watchers` reads: the thread's replies before it in the last HISTORY_S, and
    the branch of every PR they or it name.  Nothing for any other post."""
    thread = msg.get("thread_ts")
    if not thread or thread == msg.get("ts") or not is_announcement(root):
        return [], {}
    history = [m for m in api.paged("conversations.replies", "messages", channel=ch, ts=thread, oldest=f"{now - HISTORY_S:.6f}")
               if m.get("ts") not in (thread, msg.get("ts")) and float(m.get("ts") or 0) < float(msg.get("ts") or now)]
    nums = {n for m in [msg, *history] for n in named_prs(m.get("text") or "")[0]}
    return history, pr_branches(nums, log) if nums else {}


def deliver(api: Slack, reg: dict, routes: Routes, ch: str, msg: dict, *, urgent: bool = False,
            now: float | None = None, log=None) -> list[str]:
    """Deliver the post `msg`: ring the doorbell of each name it wakes.  One line per name saying
    what happened.  Every handle's published route and every name's the post starts with is read first, so a clone not
    pulled since a name opened, reopened, closed or changed holder rings what the published table says (`_newer`).  A post
    in another channel (#ask-the-team) routes the same way; its doorbells name the channel (`doorbell`), and where the bot
    can't read that channel (UNREADABLE) a reply's thread root counts as unknown."""
    log = log or sys.stderr
    front = channel_id_for(reg, FRONT_DOOR) or ch
    where = None if ch == front else (key_of(reg, ch) or ch)
    thread = msg.get("thread_ts") if msg.get("thread_ts") not in (None, msg.get("ts")) else None
    try:
        root = api.root(ch, thread) if thread else None
    except SlackError as e:
        if not where or not any(w in str(e) for w in UNREADABLE):
            raise
        root = None
    routes.refresh([*lead_names(msg.get("text", ""), reg), *(h["handle"] for h in reg["handles"])])
    reg = with_routes(reg, routes)
    inboxes = routes.inboxes()
    now = _now().timestamp() if now is None else now
    history, branches = watch_context(api, ch, msg, root, now, log)
    author = poster(msg, reg) or "?"
    urgent = urgent or "urgent" in tags_of(msg.get("text", ""))
    src, lines, missed = thread or msg.get("ts") or "", [], []
    if msg.get("bot_id") and not ours(msg, reg):
        lines.append("no doorbells: " + ("another app's bot posted it, not Verity's" if reg.get("bot_id") else
                                         "the registry records no bot_id (`research msg groups sync --apply` writes it)"))
    humans = people_names(reg)
    for name in route_targets(msg, root, reg, inboxes, history, branches):
        if name in humans:
            continue                                                 # people read Slack itself
        to, r = routes.resolve(name)
        if why := unreachable(name, to, r):
            h = next((h for h in reg["handles"] if h["handle"] == name), None)
            if h and h.get("linear"):
                lines.append(f"@{name}: no Slack inbox; reach it through its Linear team {h['linear']} (an ask to it files there)")
                continue
            known = h is not None
            lines.append(f"@{name}: no inbox" + (f"; its holder opens one with `research msg names open --as {name}`" if known else
                         " -- nobody will see this; check the name (`research msg names list`)") if not r and to == name
                         else f"@{name}: {why}")
            continue
        if to == author:
            continue
        box = ch if where is None else r.get("channel") or front    # an inbox is a thread in the front door
        try:
            rung = api.paged("conversations.replies", "messages", channel=box, ts=r["inbox_ts"], oldest=f"{now - 3600:.6f}")
            rung = [m for m in rung if m.get("ts") != r["inbox_ts"] and is_doorbell(m, reg)]
            if any(_rang(m.get("text") or "", msg.get("ts") or "", where) for m in rung):
                continue
            n = sum(_mentions_thread(m.get("text") or "", src, where) and _from(m.get("text") or "", author) for m in rung)
            if n >= CAP_PER_HOUR:                                        # a loop is one author; a roll call is many, once each
                if n == CAP_PER_HOUR:
                    if where is None:                                    # a person's channel gets no router notes
                        api.call("chat.postMessage", channel=ch, thread_ts=src, username="router", unfurl_links="false",
                                 text=f"routing @{author} to {to} paused here for the hour ({CAP_PER_HOUR} doorbells) {QUIET}")
                    api.call("chat.postMessage", channel=box, thread_ts=r["inbox_ts"], username="router", unfurl_links="false",
                             text=f"from router, {'#' + where + ' ' if where else ''}thread {src}: paused @{author} here "
                                  f"for the hour ({CAP_PER_HOUR} doorbells)")
                lines.append(f"@{to}: paused (cap of {CAP_PER_HOUR} an hour from @{author} in this thread)")
                continue
            api.call("chat.postMessage", channel=box, thread_ts=r["inbox_ts"], username="router", unfurl_links="false",
                     unfurl_media="false", text=doorbell(author, src, msg.get("ts") or src, msg.get("text", ""), urgent, where))
            lines.append(f"@{to}: rang" + (f" (for @{name}, finished)" if to != name else ""))
        except SlackError as e:                                      # the others still ring; `deliver` rings this one later
            lines.append(f"@{to}: NOT RUNG ({e})")
            missed.append(to)
    if missed:
        again = "research msg deliver " + (f"{_channel_flag(where)} " if where else "") + f"--ts {msg.get('ts')}" + \
            (f" --thread {thread}" if thread else "") + (" (or --event, where the bot can't read it)" if where else "")
        lines.append(f"{len(missed)} doorbell(s) didn't ring ({' '.join('@' + n for n in missed)}): `{again}` rings only the "
                     "ones still missing")
    for line in lines:
        print(f"route {line}", file=log)
    return lines


def channel_id_for(reg: dict, key: str) -> str | None:
    return os.environ.get(_env(key)) or reg["channels"][key].get("id")


def fetch_message(api: Slack, ch: str, ts: str, thread: str | None) -> dict | None:
    """The message itself, from Slack: a wake payload is not evidence of who wrote it."""
    if thread and thread != ts:
        msgs = api.call("conversations.replies", channel=ch, ts=thread, oldest=ts, latest=ts, inclusive="true",
                        limit=2).get("messages", [])                 # Slack puts the thread's root first, whatever `oldest` says
    else:
        msgs = api.call("conversations.history", channel=ch, oldest=ts, latest=ts, inclusive="true", limit=1).get("messages", [])
    return next((m for m in msgs if m.get("ts") == ts), None)


def author_allowed(msg: dict, reg: dict, key: str, names: set[str] | frozenset = frozenset()) -> str | None:
    """Why `msg`'s author is one of ours, or None: Verity's bot (the registry's `bot_id`, under a handle's username, a routed
    name in `names`, or the router), a bot the channel lists (`bots`: alerting), a named human (`humans`, by Slack user
    id) or a collaborator (`collaborators`, who may ask but authorizes nothing). An entry without an id allows nobody."""
    if msg.get("bot_id"):
        handles = {h["handle"] for h in reg["handles"]} | set(names) | {"router"}
        if reg.get("bot_id") and msg["bot_id"] == reg["bot_id"] and msg.get("username") in handles:
            return f"Verity's bot as @{msg['username']}"
        for b in reg["channels"][key].get("bots", []):
            if b.get("bot_id") and msg["bot_id"] == b["bot_id"]:
                return f"#{key}'s bot {b['name']} ({b['bot_id']})"
        return None
    if who := person(msg, reg):
        return f"{who[0]['name']} ({who[0]['slack_user']})" + (f", {COLLABORATOR_ROLE}" if who[1] == "collaborator" else "")
    return None


# --- subcommands ------------------------------------------------------------------------------------------------------------

def _body(ns: argparse.Namespace) -> str:
    if ns.text is not None:
        text = ns.text
    elif ns.file and ns.file != "-":
        text = Path(ns.file).read_text()
    else:
        text = sys.stdin.read()
    if not text.strip():
        raise SlackError("empty message")
    return text.rstrip()


def cmd_handles(ns: argparse.Namespace, reg: dict) -> int:
    """The registry, each handle with its holder and branch suffixes from its inbox route (`with_routes`)."""
    reg = with_routes(reg, routes_of(ns))
    if ns.json:
        print(json.dumps(reg, indent=2, ensure_ascii=False))
        return 0
    print(f"workspace {reg.get('workspace')}  bot {reg.get('bot_id') or 'id not recorded'}")
    for key, ch in reg["channels"].items():
        subs = ch["subscribers"] if ch["subscribers"] == EVERY else ", ".join("@" + s for s in ch["subscribers"])
        print(f"#{key:<20} {ch.get('id') or 'id not recorded':<12} subscribers: {subs}\n  {ch['purpose']}")
    for h in reg["handles"]:
        print(f"@{h['handle']:<10} {h.get('usergroup') or '-':<12} {h['holder'] or 'no holder: its inbox is not open'}\n"
              f"  {h['remit']}\n  charter: {h['charter']}")
    for p in reg.get("humans", []):
        print(f"human {p['name']} ({p.get('slack_user') or 'user id not recorded'})")
    for p in reg.get("collaborators", []):
        print(f"collaborator {p['name']} ({p.get('slack_user') or 'user id not recorded'}): {p['note']}")
    return 0


def cmd_owners(ns: argparse.Namespace, reg: dict) -> int:
    paths = ns.paths or sys.stdin.read().split()
    for p in paths:
        o = owner(reg, p)
        print(f"{'@' + o if o else '-':<20} {p}")
    return 0


def cmd_subscriptions(ns: argparse.Namespace, reg: dict) -> int:
    h = entry(reg, ns.handle)
    missing = False
    for key in reg["channels"]:
        if h["handle"] in subscribers(reg, key):
            cid = channel_id_for(reg, key)
            missing |= not cid
            print(f'subscribe_slack_channel(channelId="{cid or "<#" + key + " id not recorded>"}", topLevelOnly=true, '
                  f"expiresInSeconds={EXPIRES_S})   # #{key}")
    if key := h.get("linear"):
        print(f'subscribe_linear_issue(options={{"scope": "team", "teamId": "{key}"}}, expiresInSeconds={EXPIRES_S})   '
              f"# an issue filed into or changing state in @{h['handle']}'s Linear team, a human's included")
        print(f"# but no comment subscription on {key}: nearly every comment there is @{h['handle']}'s own, and it would wake "
              "it for each (every handle writes as one app; COMM-23)")
    print(f"# and subscribe_slack_thread(channelId, threadTs, expiresInSeconds={EXPIRES_S}) only for a thread a human may "
          f"answer in (@{h['handle']}'s #ask-daniel card, a thread a named human posted in); renew on every wake")
    print(f"# not for a thread between handles: the inbox already rings @{h['handle']} for each reply addressed to it, and "
          "never for its own posts, while a thread subscription also wakes it for each of its own replies (COMM-23)")
    return 2 if missing else 0


def routes_of(ns: argparse.Namespace) -> Routes:
    if getattr(ns, "_routes", None) is None:
        ns._routes = Routes()
    return ns._routes


def name_entry(ns: argparse.Namespace, reg: dict, name: str, person_ok: bool = False) -> dict:
    """A registry handle, or a name with an inbox that this checkout's registry predates: who may post (`--as`) and be
    addressed (`--to`)."""
    try:
        return entry(reg, name)
    except SlackError:
        if person_ok and (person := human_entry(reg, name)):
            return person
        name = name.lstrip("@")
        if routes_of(ns).get_route(name):
            return {"handle": name}
        raise SlackError(f"@{name} is neither a registry handle nor a name with an inbox (`research msg names list`); "
                         f"its agent opens one with `research msg names open --as {name}`") from None


def holder_ids(ns: argparse.Namespace, reg: dict, name: str) -> list[str]:
    """The agents that hold `name`: its open inbox route's holder and `posters` (`with_routes`)."""
    try:
        h = entry(reg, name)
    except SlackError:
        h = {"handle": name.lstrip("@")}
    return list(dict.fromkeys(AGENT_ID_RE.findall(with_routes({"handles": [h]}, routes_of(ns))["handles"][0]["holder"])))


def acting_as(ns: argparse.Namespace, reg: dict, name: str) -> dict:
    """`name_entry` for `--as NAME` on a command that posts or changes NAME's route: only NAME's holder (`holder_ids`; a
    short id such as `bc-8ece7cde` matches the full one) may use it, and an agent that can't tell its own id may not."""
    h = name_entry(ns, reg, name)
    ids, me = holder_ids(ns, reg, h["handle"]), self_agent_id()
    if me and any(me == a or (len(a) < len(me) and me.startswith(a)) for a in ids):
        return h
    if not me and ns.dry_run:
        print(f"dry-run: no agent id (set RESEARCH_AGENT_URL), so a real --as {h['handle']} would be refused", file=sys.stderr)
        return h
    held = f"held by {', '.join(ids)}" if ids else "held by no agent: it has no open inbox"
    raise SlackError(f"--as {h['handle']}: @{h['handle']} is {held}, and this agent is {me or 'unknown (set RESEARCH_AGENT_URL)'}; "
                     f"post as your own handle (a successor takes a name over with `research msg names open --as "
                     f"{h['handle']} --succeeds`, and its holder lets another agent post as it with `names open --as "
                     f"{h['handle']} --poster BC_ID`)")


def _route(ns: argparse.Namespace, reg: dict, ch: str) -> tuple[dict, Routes | None] | None:
    return (reg, routes_of(ns)) if ch == channel_id_for(reg, FRONT_DOOR) else None


def cmd_ask(ns: argparse.Namespace, reg: dict) -> int:
    h, owner = acting_as(ns, reg, ns.handle), name_entry(ns, reg, ns.to)
    if owner["handle"] == h["handle"]:
        raise SlackError("an ask goes to another handle")
    ch = channel_id(ns, reg, FRONT_DOOR)
    refuse_unrouted(body := _body(ns), reg, h["handle"], (owner["handle"],))
    body = linear_ask(ns, owner, h["handle"], body)
    _post(client(ns), ch, h["handle"], f"{tag(owner)} {body}", route=_route(ns, reg, ch), urgent=ns.urgent)
    return 0


def linear_ask(ns: argparse.Namespace, owner: dict, sender: str, body: str) -> str:
    """The ask filed as an issue in its owner's Linear team (`linear` in the registry), which holds the whole brief; what
    Slack carries is then its first line and the issue's link (COMM-15).  The body as it was when the owner has no team
    or Linear fails, so the ask still goes out on Slack in full."""
    if not ASKS_ON_LINEAR or ns.dry_run or not owner.get("linear"):
        return body
    from . import linear  # noqa: PLC0415

    issue, why = linear.ask_issue(owner["linear"], sender, body)
    if not issue:
        print(f"research msg: not filed on Linear ({why}); asking on Slack only", file=sys.stderr)
        return body
    print(f"linear: {issue['identifier']} {issue['url']}")
    return f"{linear.title(body)}\nLinear: <{issue['url']}|{issue['identifier']}>"


def linear_close(text: str, state_type: str) -> None:
    """Move the Linear issue an ask's root links (`linear_ask`) to its team's `state_type` state (`Linear.move`)."""
    if not (m := LINEAR_LINK_RE.search(text or "")):
        return
    from . import linear  # noqa: PLC0415

    try:
        print(f"linear: {m[1]} {linear.Linear().move(m[1], state_type)}")
    except (linear.LinearError, OSError, ValueError) as e:
        print(f"research msg: {m[1]} stays open on Linear ({e}); the status line stands", file=sys.stderr)


LINEAR_STATES = ("Backlog = filed, Todo = accepted, In Progress = someone is on it now, In Review = a PR is up, "
                 "Done = landed or answered")                    # what each state means for us (top, workflow thread, 15 Oct)


def cmd_linear(ns: argparse.Namespace, reg: dict) -> int:
    """`check` (what the credentials reach), `state ISSUE STATE`, `comment ISSUE TEXT --as HANDLE`, `list TEAM [--state S]`."""
    from . import linear  # noqa: PLC0415

    def need(value: str | None, what: str) -> str:
        if not value:
            raise SlackError(f"linear {ns.verb}: give {what}")
        return value
    try:
        lin = linear.Linear()
        if ns.verb == "check":
            lines, ok = lin.check(ns.write, name_entry(ns, reg, ns.handle or "comms")["handle"])
            print("\n".join(lines))
            return 0 if ok else 1
        if ns.verb == "list":
            for i in lin.issues(need(ns.target, "the TEAM (a handle or its key)"), ns.state):
                print(f"{i['identifier']}  {i['state']['name']}  {i['title']}")
            print(f"states: {LINEAR_STATES}")
            return 0
        issue = need(ns.target, "the ISSUE (e.g. INF-12)")
        if ns.verb == "state":
            print(f"{issue} {lin.move(issue, need(ns.value, 'the STATE by name (e.g. In Progress)'))}")
            return 0
        author, text = acting_as(ns, reg, need(ns.handle, "--as HANDLE, the comment's sender"))["handle"], need(ns.value, "the TEXT")
        refuse_unrouted(text, reg, author)
        info = lin.gql("query($id: String!) { issue(id: $id) { id identifier url team { key } } }", {"id": issue})["issue"]
        lin.comment(info["id"], author, text)
        print(f"{issue} commented")
        ring_comment(ns, reg, author, info, text)
        return 0
    except linear.LinearError as e:
        raise SlackError(str(e)) from None


def ring_comment(ns: argparse.Namespace, reg: dict, author: str, issue: dict, text: str) -> list[str]:
    """Ring the inbox of the issue's team's handle and of each name the comment starts with, as `send` would, so nobody
    polls Linear for comments (no one subscribes to them: every handle writes as one app; COMM-30).  The author is never
    rung.  One line per name."""
    owner = next((h["handle"] for h in reg["handles"] if h.get("linear") == issue["team"]["key"]), None)
    names = list(dict.fromkeys(([owner] if owner else []) + lead_names(text, reg)))
    routes, api, lines = routes_of(ns), client(ns), []
    ch = channel_id(ns, reg, FRONT_DOOR)
    body = plain(text)
    if len(body) > DOORBELL_MAX:
        body = body[:DOORBELL_MAX].rstrip() + "\u2026"
    for name in names:
        to, r = routes.resolve(name)
        if name == author or to == author:
            continue
        if why := unreachable(name, to, r):
            lines.append(f"@{name}: {why}; the comment is on {issue['identifier']}")
            continue
        api.call("chat.postMessage", channel=ch, thread_ts=r["inbox_ts"], username="router", unfurl_links="false",
                 unfurl_media="false", text=f"from {author}, Linear {issue['identifier']} ({issue['url']}): {body}")
        lines.append(f"@{to}: rang")
    for line in lines:
        print(f"route {line}", file=sys.stderr)
    return lines


def announcement(h: dict, to: list[dict], body: str, verify: str | None, ships: list[str]) -> str:
    for pin in ships:
        if not PIN_RE.match(pin):
            raise SlackError(f"--ships {pin!r}: pin code as URL#sha256=<64 hex> or git:<40-hex commit>")
    if not ships and SHIPS_CODE_RE.search(f"{body}\n{verify or ''}"):
        raise SlackError("the announcement fetches or installs code: pin it with --ships URL#sha256=HEX or git:SHA")
    audience = " ".join(tag(t) for t in to) if to else "every handle"
    lines = [f"{ANNOUNCE} *Announcement* from @{h['handle']} to {audience}", body]
    if verify:
        lines += [f"*{SAFE_POINT}*", f"*Verify:* `{verify}`"]
    for pin in ships:
        url, _, sha = pin.partition("#sha256=")
        lines.append(f"*Pinned:* commit `{pin[4:]}`" if pin.startswith("git:") else f"*Pinned:* {url} sha256 `{sha}` (don't run it if the checksum fails)")
    if verify:
        lines.append("*Close it with a status line:* `research msg close --thread <this ts>` (or `--declined WHY`)")
    return "\n".join(lines)


def _check_audience(to: list[dict], body: str, reg: dict) -> None:
    if not to and (lead := lead_names(body, reg)):
        raise SlackError(f"with no --to an announcement goes to every handle, but it starts with "
                         f"{' '.join('@' + n for n in lead)}: pass them as --to, or move them off the start")


def cmd_announce(ns: argparse.Namespace, reg: dict) -> int:
    h = acting_as(ns, reg, ns.handle)
    to = [name_entry(ns, reg, t, person_ok=True) for t in ns.to]
    ch = channel_id(ns, reg, FRONT_DOOR)
    _check_audience(to, body := _body(ns), reg)
    _post(client(ns), ch, h["handle"], announcement(h, to, body, ns.verify, ns.ships),
          route=_route(ns, reg, ch), urgent=ns.urgent)
    return 0


def cmd_send(ns: argparse.Namespace, reg: dict) -> int:
    """The one way to post: top-level (a new thread) or into `--thread`.  The `--to` names lead it, which is what routes it;
    `--kind` rides in the footer.  An announcement (`--kind announce`) gets its header,
    and with no `--to` it is for every handle."""
    h = acting_as(ns, reg, ns.handle)
    to = [name_entry(ns, reg, t, person_ok=True) for t in ns.to]
    kind = (ns.kind or ("ask" if to and not ns.thread else "")).strip().lower()
    if kind and not NAME_RE.match(kind) or kind in FLAGS:
        raise SlackError(f"--kind {ns.kind!r}: one lower-case word")
    if any(t["handle"] == h["handle"] for t in to):
        raise SlackError("--to names someone else")
    body = _body(ns)
    if kind != "announce" and not ns.thread and not to and is_announcement({"text": body}):
        raise SlackError(f"it starts with {ANNOUNCE}, which makes a top-level post an announcement to every handle: send it "
                         "with --kind announce (which checks its pins), or start it with something else")
    if to or kind != "announce":                                     # an announcement with no --to reaches every handle
        refuse_unrouted(body, reg, h["handle"], tuple(t["handle"] for t in to))
    if not to and (lead := lead_names(body, reg)) and set(lead) == {h["handle"]}:
        raise SlackError(f"it starts with @{h['handle']} only, and a post never wakes its author: name its addressee with --to")
    if kind == "announce":
        if ns.thread:
            raise SlackError("an announcement starts its own thread: drop --thread")
        _check_audience(to, body, reg)
        text = announcement(h, to, body, ns.verify, ns.ships)
    else:
        if ns.verify or ns.ships:
            raise SlackError("--verify and --ships go with --kind announce")
        if to and (m := LEAD_RE.match(body)) and set(lead_names(body, reg)) <= {t["handle"] for t in to}:
            body = body[m.end():]                                    # `--to @ci` with a body that starts `@ci` names it once
        if kind == "ask" and not ns.thread and len(to) == 1:
            body = linear_ask(ns, to[0], h["handle"], body)
        text = " ".join([*(tag(t) for t in to), body])
    ch = channel_id(ns, reg, FRONT_DOOR) if not (ns.thread or ns.channel) else channel_id(ns, reg)   # --channel: a machine's alert
    quiet = ns.quiet or kind == "fyi"                                # an fyi wants no reply, so it wakes only its --to names
    tags = tuple(t for t in (kind if kind not in ("announce", "reply") else "", QUIET if quiet else "") if t)
    _post(client(ns), ch, h["handle"], text, ns.thread, route=_route(ns, reg, ch), urgent=ns.urgent, tags=tags)
    return 0


def cmd_close(ns: argparse.Namespace, reg: dict) -> int:
    if ns.declined:
        return _status(ns, reg, f"{DECLINED} declined", "x", ns.declined)
    return _status(ns, reg, f"{DONE} done", "white_check_mark", " ".join(x for x in (ns.link, ns.note) if x))


def cmd_reply(ns: argparse.Namespace, reg: dict) -> int:
    h = acting_as(ns, reg, ns.handle)
    ch = channel_id(ns, reg)
    body = _body(ns)
    refuse_unrouted(body, reg, h["handle"])
    _post(client(ns), ch, h["handle"], body, ns.thread, route=_route(ns, reg, ch), urgent=ns.urgent,
          tags=(QUIET,) if ns.quiet else ())                         # a footer tag rides on the first piece of a long post
    return 0


def cmd_pickup(ns: argparse.Namespace, reg: dict) -> int:
    client(ns).call("reactions.add", channel=channel_id(ns, reg), timestamp=ns.thread, name="eyes")
    return 0


def _status(ns: argparse.Namespace, reg: dict, word: str, reaction: str, detail: str | None) -> int:
    """A status line in the thread; on an ask or an alert (one owner), also the reaction on its root. On an announcement the
    line alone carries it: every handle posts through one bot, which can place each reaction only once."""
    h = acting_as(ns, reg, ns.handle)
    api, ch = client(ns), channel_id(ns, reg)
    detail = " ".join((detail or "").split())
    _post(api, ch, h["handle"], f"{word}" + (f" \u2014 {detail}" if detail else ""), ns.thread, route=_route(ns, reg, ch))
    root = api.root(ch, ns.thread)
    if not is_announcement(root):
        api.call("reactions.add", channel=ch, timestamp=ns.thread, name=reaction)
        linear_close((root or {}).get("text", ""), "canceled" if reaction == "x" else "completed")
    return 0


def cmd_retract(ns: argparse.Namespace, reg: dict) -> int:
    """Blank a post that `--as` NAME sent: its text becomes `_retracted by @NAME_` (and `--why`), so its thread keeps its
    place.  Only NAME's holder may, only for NAME's own post, and only with the bot token: the broker allows no edits.  The
    doorbells it already rang keep their copy."""
    h, api = acting_as(ns, reg, ns.handle), client(ns)
    ch = channel_id(ns, reg)
    if not ns.dry_run:
        msg = fetch_message(api, ch, ns.ts, ns.thread)
        if not msg:
            raise SlackError(f"no post {ns.ts} in that channel" + ("" if ns.thread else " (a reply needs --thread ROOT)"))
        if (who := poster(msg, reg)) != h["handle"]:
            raise SlackError(f"{ns.ts} was posted by {'@' + who if who else 'someone else'}, not @{h['handle']}: "
                             "a name retracts only its own posts")
    text = f"_retracted by @{h['handle']}_" + (f": {' '.join(ns.why.split())}" if ns.why else "")
    api.call("chat.update", channel=ch, ts=ns.ts, text=text)
    print(f"retracted {ns.ts}")
    return 0


def cmd_roster(ns: argparse.Namespace, reg: dict) -> int:
    msgs = client(ns).paged("conversations.replies", "messages", channel=channel_id(ns, reg), ts=ns.thread)
    root = next((m for m in msgs if m.get("ts") == ns.thread), None)
    if not ns.dry_run and not is_announcement(root):
        raise SlackError(f"{ns.thread} is not an announcement")
    head = (root or {}).get("text", "")
    aud = [h["handle"] for h in reg["handles"]] if to_every_handle(head) else audience(head, reg)
    expected = [n for n in aud if n != (root or {}).get("username")]
    status: dict[str, dict] = {}
    for m in msgs:
        if m.get("ts") != ns.thread and m.get("username") in expected and ours(m, reg) and (mt := STATUS_RE.match(m.get("text", ""))):
            status[m["username"]] = {"status": mt.group(1).split()[1], "detail": DATE_TOKEN_RE.sub(r"\1", mt.group(2) or ""), "ts": m["ts"]}
    rows = [{"handle": n, **status.get(n, {"status": "missing"})} for n in expected]
    if ns.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for r in rows:
            print(f"{r['status']:<9} @{r['handle']}" + (f"  {r['detail']}" if r.get("detail") else ""))
    return 0 if all(r["status"] == "done" for r in rows) else 1


def cmd_match(ns: argparse.Namespace, reg: dict) -> int:
    h = entry(reg, ns.handle)
    ev = parse_event(ns.event)
    key = key_of(reg, ev["channel"]) if ev.get("channel") else channel_key(ns, reg)
    reply = ev.get("thread_ts") and ev.get("thread_ts") != ev.get("ts")
    api = client(ns) if reply and not by_handle(ev, h["handle"]) else None
    why = addressed(ev, h, reg, key, api)
    if ns.why:
        print(why or "not addressed", file=sys.stderr)
    return 0 if why else 1


def cmd_deliver(ns: argparse.Namespace, reg: dict) -> int:
    """Ring the doorbells for a post already in the channel that no client delivered: a person's typed post, or one whose
    sender stopped partway.  `--channel ask-the-team` (or --channel-id) delivers a post in another channel, whose doorbells
    name it.  `--event` (a watcher's Slack message event) gives the post: Slack's own copy where the bot can read the
    channel, else the event's (#ask-the-team: the bot lacks its history).  Safe to repeat: a name with this post's doorbell
    isn't rung again."""
    api, ev = client(ns), parse_event(ns.event) if ns.event else {}
    if ev.get("channel") and not (ns.channel or ns.channel_id):
        if not (key := key_of(reg, ev["channel"])):
            raise SlackError(f"the event's channel {ev['channel']} is not in the registry: pass --channel-id {ev['channel']}")
        ch = channel_id(ns, reg, key)
    else:
        ch = channel_id(ns, reg)
        if ev.get("channel") and ev["channel"] != ch:
            raise SlackError(f"the event was posted in {ev['channel']}, not {ch}")
    if ns.ts and ev.get("ts") and ns.ts != ev["ts"]:
        raise SlackError(f"--ts {ns.ts} is not the event's ts {ev['ts']}")
    if not (ts := ns.ts or ev.get("ts")):
        raise SlackError("deliver needs --ts, or an --event with its ts")
    thread = ns.thread or ev.get("thread_ts")
    copy = {k: ev[k] for k in ("ts", "thread_ts", "text", "user", "bot_id", "username") if ev.get(k)} if ev.get("text") else None
    try:
        msg = fetch_message(api, ch, ts, thread) if not api.dry_run else copy
    except SlackError as e:
        if not copy or not any(w in str(e) for w in UNREADABLE):
            raise
        print(f"deliver: {e} reading {ch}, so the event's own copy is delivered", file=sys.stderr)
        msg = copy
    if not msg:
        raise SlackError(f"no message {ts} in the channel")
    msg = {**msg, "thread_ts": msg.get("thread_ts") or thread}
    deliver(api, reg, routes_of(ns), ch, msg)
    return 0


def cmd_verify_author(ns: argparse.Namespace, reg: dict) -> int:
    """Fail closed: exit 0 only when the message, fetched from Slack by channel and ts, has an author the registry names;
    with --authorizes, only when that author is a named human (`humans`), the one who may authorize spend, access or keys,
    destructive changes or node changes (a collaborator may ask, never authorize).  With --dry-run the event's own fields
    are judged (a test of the allowlist, not of a real post)."""
    ev = parse_event(ns.event)
    key = (key_of(reg, ev["channel"]) if ev.get("channel") and not ns.channel else None) or channel_key(ns, reg)
    ch = channel_id(ns, reg, key)
    why = None
    if ev.get("channel") and ev["channel"] != ch:
        why_not = f"posted in {ev['channel']}, not #{key} ({ch})"
    elif not ev.get("ts"):
        why_not = "the event has no ts to look the message up by"
    else:
        msg = ev if ns.dry_run else fetch_message(client(ns), ch, ev["ts"], ev.get("thread_ts"))
        names = set()
        if msg and msg.get("bot_id") and msg.get("username") and not any(h["handle"] == msg["username"] for h in reg["handles"]):
            names = {msg["username"]} if routes_of(ns).get_route(msg["username"]) else set()
        why = author_allowed(msg, reg, key, names) if msg else None
        who = "@" + msg["username"] if msg and msg.get("username") else (msg or {}).get("user") or (msg or {}).get("bot_id")
        why_not = "no such message in the channel" if msg is None else f"author {who} is not in the registry"
        if why and ns.authorizes and (person(msg, reg) or (None, None))[1] != "human":
            why, why_not = None, (f"{why}; only a named human (the registry's humans) authorizes spend, access or keys, "
                                  "destructive changes or node changes")
    if ns.why:
        print(f"allowed: {why}" if why else f"not allowed: {why_not}", file=sys.stderr)
    return 0 if why else 1


def cmd_read(ns: argparse.Namespace, reg: dict) -> int:
    if ns.thread and ns.handle:
        raise SlackError("read: --thread reads one thread; it doesn't take --as")
    if not (ns.thread or ns.since or ns.handle):
        raise SlackError("read: one of --thread TS, --since TS or --as NAME is required")
    if ns.handle:
        name = ns.handle.lstrip("@")
        r = routes_of(ns).get_route(name, fresh=name not in routes_of(ns).local)
        if not r:
            raise SlackError(f"@{name} has no inbox; open one with `research msg names open --as {name}`")
        return inbox_read(ns, reg, r)
    api, ch = client(ns), channel_id(ns, reg)
    if ns.thread:
        since = {"oldest": ns.since} if ns.since else {}
        msgs = api.paged("conversations.replies", "messages", ns.limit, channel=ch, ts=ns.thread, **since)
        if ns.since:
            msgs = [m for m in msgs if float(m["ts"]) > float(ns.since)]     # Slack returns the thread's root whatever `oldest` says
    else:
        msgs = api.paged("conversations.history", "messages", ns.limit, channel=ch, oldest=ns.since)
        msgs = [m for m in msgs if m.get("ts") != ns.since][::-1]                   # oldest first; `oldest` is inclusive at the edge
    if ns.json:
        print(json.dumps(msgs, indent=2, ensure_ascii=False))
        return 0
    for m in msgs:
        thread = f" in {m['thread_ts']}" if m.get("thread_ts") and m.get("thread_ts") != m.get("ts") and not ns.thread else ""
        who = "@" + m["username"] if m.get("username") else poster(m, reg) or m.get("user") or m.get("bot_id") or "?"
        tags = [t for t in tags_of(m.get("text") or "") if t != "ask"]
        print(f"{m.get('ts')}{thread} {who}{' (' + ', '.join(tags) + ')' if tags else ''}: {plain(m.get('text', ''))}")
    return 0


# --- inboxes ----------------------------------------------------------------------------------------------------------------

def publish_route(root: Path, r: dict, message: str) -> str:
    """Write `routes.d/<name>.json` in the notes clone at `root` and push it: the line `notes sync` printed."""
    if not (root / ".git").exists():
        try:
            from . import notes_sync
            notes_sync.ensure_clone(root)
        except ImportError:
            pass
    if not (root / ".git").exists():
        raise SlackError(f"no notes clone at {root} (set RESEARCH_NOTES, or RESEARCH_NOTES_TOKEN to clone it); the route is:\n"
                         + json.dumps(r, indent=2))
    p = root / ROUTES_DIR / f"{r['name']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=2) + "\n")
    rel = f"{ROUTES_DIR}/{r['name']}.json"
    try:
        from . import notes_sync
    except ImportError:                                             # a single file: plain git
        for args in (["add", rel], ["commit", "-q", "-m", message, "--", rel], ["pull", "-q", "--rebase", "--autostash"], ["push", "-q"]):
            out = __import__("subprocess").run(["git", "-C", str(root), *args], capture_output=True, text=True)
            if out.returncode:
                raise SlackError(f"notes: git {args[0]} failed: {out.stderr.strip()[-300:]}")
        return f"notes: pushed {rel}"
    rep = notes_sync.sync(root, paths=[rel], message=message)
    line = notes_sync.render_sync(rep)
    if rep.get("rc", 0) and not rep.get("pushed"):
        raise SlackError(f"the route is written but not pushed: {line}")
    return line


def _now_stamp() -> str:
    return f"{_now():%Y-%m-%dT%H:%MZ}"


def cmd_inbox(ns: argparse.Namespace, reg: dict) -> int:
    routes = routes_of(ns)
    if ns.verb == "list":
        for h in reg["handles"]:                                    # a clone not pulled since the name opened lists it too
            if h["handle"] not in routes.local:
                routes.get_route(h["handle"])
        rows = sorted({**routes.local, **routes.all()}.values(), key=lambda r: r["name"])
        if ns.json:
            print(json.dumps(rows, indent=2))
            return 0
        for r in rows:
            state = f"closed {r['closed']}" + (f" -> @{r['forward_to']}" if r.get("forward_to") else "") if r.get("closed") else "open"
            print(f"@{r['name']:<28} {r['inbox_ts']:<18} {r.get('holder') or '?':<40} {state}" + (f"\n  {r['remit']}" if r.get("remit") else ""))
        return 0
    name = (ns.handle or "").lstrip("@")
    if not NAME_RE.match(name):
        raise SlackError(f"--as {ns.handle!r}: a name is lower case letters, digits, '.', '_' and '-'")
    if ns.verb == "open":
        if not any(h["handle"] == name for h in reg["handles"]):
            raise SlackError(f"@{name} is not a lead's handle: only leads get names, and workers are reached through their lead "
                             f"(Daniel, 2026-10-01); a new lead is added to {REGISTRY_REL} first")
        return inbox_open(ns, reg, routes, name)
    r = routes.get_route(name, fresh=name not in routes.local)
    if not r:
        raise SlackError(f"@{name} has no inbox; open one with `research msg names open --as {name}`")
    if ns.verb == "close":
        fwd = (ns.forward_to or "").lstrip("@") or None
        if fwd == name:
            raise SlackError(f"--forward-to @{name}: a name can't forward to itself, and its doorbells would ring a closed inbox; "
                             "forward to the handle that takes its work over, or close it with no --forward-to")
        if fwd and not any(h["handle"] == fwd for h in reg["handles"]) and not routes.get_route(fwd):
            raise SlackError(f"--forward-to @{fwd}: neither a registry handle nor a name with an inbox, so its doorbells would "
                             "ring no one; check the name (`research msg names list`) and forward to a handle or a name listed there")
        r = {**r, "closed": _now_stamp(), "forward_to": fwd}
        if not ns.dry_run:
            print(publish_route(routes.root, r, f"routes.d: close @{name}" + (f" -> @{r['forward_to']}" if r["forward_to"] else "")))
        print(f"@{name} closed" + (f"; its doorbells ring @{r['forward_to']}" if r["forward_to"] else ""))
        return 0
    return inbox_read(ns, reg, r)


def inbox_open(ns: argparse.Namespace, reg: dict, routes: Routes, name: str) -> int:
    """Open `name`'s inbox, or say it is open already.  An open inbox held by another agent is refused unless --succeeds names
    it: the successor takes the same thread over, unread doorbells and all.  The route is the handle's roster: its holder,
    the `posters` (--poster: other agents that post as it, kept across a succession) and the `branch_suffixes` its PRs
    carry (this checkout's branch's, and each earlier holder's)."""
    holder = ns.holder or (None if ns.dry_run else self_agent_id())
    if not holder and not ns.dry_run:
        raise SlackError("no agent id: pass --holder bc-... (or set RESEARCH_AGENT_URL)")
    for p in getattr(ns, "poster", None) or []:
        if not AGENT_ID_RE.fullmatch(p):
            raise SlackError(f"--poster {p!r}: an agent id (bc-<8 hex>, its full form, or a desktop agent's UUID)")
    old = routes.get_route(name, fresh=True)
    if old and (mine := routes.local.get(name)) and mine.get("holder") == old.get("holder"):
        old = {**old, **{k: v for k in ("branch_suffixes", "posters")
                         if (v := list(dict.fromkeys([*old.get(k, []), *mine.get(k, [])])))}}
    ch = channel_id(ns, reg, FRONT_DOOR)                           # the published copy lags the clone by minutes: keep both's lists
    posters = list(dict.fromkeys([*(old or {}).get("posters", []), *(getattr(ns, "poster", None) or [])]))
    if old and not old.get("closed") and old.get("holder") == holder:
        print(f"@{name} is open already: inbox {old['inbox_ts']}")
        new = {**old, **({"posters": posters} if posters else {})}
        if (s := _branch_suffix()) and s not in old.get("branch_suffixes", []):
            new["branch_suffixes"] = [*old.get("branch_suffixes", []), s]
        if new != old and not ns.dry_run:
            what = [*([f"branch suffix {s}"] if new.get("branch_suffixes") != old.get("branch_suffixes") else []),
                    *([f"posters {', '.join(posters)}"] if posters != old.get("posters", []) else [])]
            print(publish_route(routes.root, new, f"routes.d: @{name} holder's {' and '.join(what)}"))
        old = new
    else:
        if old and not old.get("closed") and not ns.succeeds:
            raise SlackError(f"@{name}'s inbox is held by {old.get('holder')}; a successor passes --succeeds")
        api = client(ns)
        if old:
            ts = old["inbox_ts"]
        else:
            ts = _post(api, ch, name, f"{INBOX_HEAD}{name}. Posts addressed to @{name} land here as replies; don't reply "
                       f"here, answer in the source thread.")
        remit = " ".join((ns.remit or (old or {}).get("remit") or "").split()) or None
        prev = [a[-4:] for a in AGENT_ID_RE.findall((old or {}).get("holder") or "") if len(a) > 11]
        sfx = list(dict.fromkeys([*(old or {}).get("branch_suffixes", []), *prev, *([s] if (s := _branch_suffix()) else [])]))
        r = {"schema": ROUTE_SCHEMA, "name": name, "holder": holder, "channel": ch, "inbox_ts": ts, "opened": _now_stamp(),
             "closed": None, "forward_to": None, **({"remit": remit} if remit else {}), **({"branch_suffixes": sfx} if sfx else {}),
             **({"posters": posters} if posters else {}),
             **({"succeeds": old.get("holder")} if old and old.get("holder") != holder else {})}
        if not ns.dry_run:
            print(publish_route(routes.root, r, f"routes.d: open @{name} for {holder}"))
        old = r
    print(f'subscribe_slack_thread(channelId="{ch}", threadTs="{old["inbox_ts"]}", expiresInSeconds={EXPIRES_S})   # @{name}\'s inbox; renew on every wake')
    return 0


def seen_path(name: str) -> Path:
    return Path(os.environ.get("RESEARCH_SLACK_SEEN") or Path.home() / ".research" / "slack-seen") / name


def unread(api: Slack, r: dict, since: str | None = None) -> list[dict]:
    """The doorbells after `since`, else after the last one this machine marked read."""
    p = seen_path(r["name"])
    seen = since or (p.read_text().strip() if p.is_file() else "0")
    msgs = api.paged("conversations.replies", "messages", channel=r.get("channel"), ts=r["inbox_ts"], oldest=seen)
    return [m for m in msgs if m.get("ts") not in (r["inbox_ts"], seen) and float(m["ts"]) > float(seen)]


def mark_read(name: str, msgs: list[dict]) -> None:
    """Only ever forward: a read from an older --since leaves the newer mark."""
    if msgs:
        p = seen_path(name)
        new = max((m["ts"] for m in msgs), key=float)
        if p.is_file() and float(p.read_text().strip() or 0) >= float(new):
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(new + "\n")


def urgent_doorbell(m: dict, reg: dict) -> bool:
    """A doorbell whose header says URGENT: `(URGENT)` in a post's body makes nothing urgent."""
    return is_doorbell(m, reg) and bool((d := DOORBELL_RE.match(m.get("text") or "")) and d[2])


def render_unread(name: str, msgs: list[dict], reg: dict) -> str:
    """One `- ` item per reply in the inbox, its later lines indented under it so a body can't pass for another doorbell;
    a reply the router didn't post says so."""
    if not msgs:
        return f"SLACK @{name}: nothing new"
    msgs = sorted(msgs, key=lambda m: (not urgent_doorbell(m, reg), float(m["ts"])))
    lines = [f"SLACK @{name}: {len(msgs)} new; answer with `research msg send --as {name} --thread T --text ...`"]
    for where in dict.fromkeys(c for m in msgs if is_doorbell(m, reg) and (c := doorbell_channel(m.get("text") or ""))):
        lines[0] += f"; one from #{where} is answered there, `... {_channel_flag(where)} --thread T`"
    for m in msgs:
        text = plain(m.get("text") or "")
        if not is_doorbell(m, reg):
            who = "@" + m["username"] if m.get("username") else m.get("user") or m.get("bot_id") or "?"
            text = f"(posted in the inbox by {who}, not a doorbell) {text}"
        lines.append("- " + re.sub(r"\n(?=.)", "\n  ", text))
    return "\n".join(lines)


def inbox_read(ns: argparse.Namespace, reg: dict, r: dict) -> int:
    r = {**r, "channel": r.get("channel") or channel_id(ns, reg, FRONT_DOOR)}
    api = client(ns)
    since = getattr(ns, "since", None)
    msgs = unread(api, r, since)
    if ns.json:
        print(json.dumps(msgs, indent=2, ensure_ascii=False))
    else:
        print(render_unread(r["name"], msgs, reg))
    if not ns.peek:
        mark_read(r["name"], msgs)
    return 1 if ns.urgent and not any(urgent_doorbell(m, reg) for m in msgs) else 0


def cmd_watch(ns: argparse.Namespace, reg: dict) -> int:
    """Block until a doorbell lands in NAME's inbox that neither `read --as` nor an earlier `watch` has seen; print it, exit
    0.  Marks nothing read: it is the hook a wake script runs (`while watch; do <wake the agent>; done`) for an agent that
    can't subscribe to its inbox thread.  Exit 1 after --polls polls with none."""
    name = ns.handle.lstrip("@")
    r = routes_of(ns).get_route(name, fresh=name not in routes_of(ns).local)
    if not r:
        raise SlackError(f"@{name} has no inbox")
    r = {**r, "channel": r.get("channel") or channel_id(ns, reg, FRONT_DOOR)}
    api, mark = client(ns), seen_path(f"{name}.watch")
    for i in range(ns.polls or 1 << 62):
        if i:
            _sleep(ns.every)
        last = float(mark.read_text().strip()) if mark.is_file() else 0.0
        new = [m for m in unread(api, r) if float(m["ts"]) > last]
        if new:
            print(render_unread(name, new, reg))
            mark_read(f"{name}.watch", new)
            return 0
    return 1


# --- traffic: the monitor ---------------------------------------------------------------------------------------------------

def comment_by(c: dict) -> str | None:
    """Who a Linear comment is from: the app's `createAsUser` name, else (a person's key, where every comment shows as that
    person) the handle its body opens with, `@handle: ` (`Linear.comment`), else the person."""
    if bot := (c.get("botActor") or {}).get("userDisplayName"):
        return bot
    if m := re.match(r"@([a-z0-9._-]+): ", c.get("body") or ""):
        return m[1]
    return (c.get("user") or {}).get("name")


def issue_taken(ident: str, asker: str | None, cache: dict[str, bool]) -> bool:
    """Whether the Linear issue `ident` was taken up there: it left Backlog (accepted, started, done or canceled), or
    someone other than `asker` commented on it.  False when Linear can't say, so a doubtful ask still shows."""
    if ident not in cache:
        from . import linear  # noqa: PLC0415
        q = "query($id: String!) { issue(id: $id) { state { type } comments { nodes { botActor { userDisplayName } user { name } body } } } }"
        try:
            i = linear.Linear().gql(q, {"id": ident})["issue"]
            who = [comment_by(c) for c in i["comments"]["nodes"]]
            cache[ident] = i["state"]["type"] != "backlog" or any(w and w != asker for w in who)
        except (linear.LinearError, OSError, KeyError, TypeError, ValueError):
            cache[ident] = False
    return cache[ident]


def _stamp_epoch(stamp: str | None) -> float | None:
    try:
        return datetime.strptime(stamp, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc).timestamp() if stamp else None
    except ValueError:
        return None


def replay(api: Slack, reg: dict, routes: Routes, ch: str, since: float, log=None) -> list[dict]:
    """Each post in `ch` since `since` whose doorbells differ from what `route_targets` says it should have rung: the
    names `missed` (expected, no doorbell in their inbox) and `extra` (a doorbell nothing called for).  A name is expected
    when its inbox existed then and wasn't closed yet (or forwards); a route closed and reopened inside the window can
    show false misses for the span it was closed, since routes keep no history."""
    posts, threads = [], {}
    for m in api.paged("conversations.history", "messages", channel=ch, oldest=f"{since - HISTORY_S:.6f}"):
        if (m.get("text") or "").startswith(INBOX_HEAD):
            continue
        if float(m["ts"]) > since:
            posts.append((m, None))
        if m.get("reply_count") and float(m.get("latest_reply") or 0) > since:
            threads[m["ts"]] = [x for x in api.paged("conversations.replies", "messages", channel=ch, ts=m["ts"],
                                                     oldest=f"{since - HISTORY_S:.6f}") if x.get("ts") != m["ts"]]
            posts += [(x, m) for x in threads[m["ts"]] if float(x["ts"]) > since]
    for n in {n for m, _ in posts for n in lead_names(m.get("text") or "", reg)}:
        routes.get_route(n)                                          # a name this clone lacks: its published route
    reg = with_routes(reg, routes)
    book, inboxes = routes.all(), routes.inboxes()
    by_inbox = {r["inbox_ts"]: n for n, r in sorted(book.items(), key=lambda kv: not kv[1].get("closed"))}
    rang: dict[str, set[str]] = {}
    for its, name in by_inbox.items():
        for m in api.paged("conversations.replies", "messages", channel=ch, ts=its, oldest=f"{since:.6f}"):
            if m.get("ts") != its and is_doorbell(m, reg) and (d := DOORBELL_RE.match(m.get("text") or "")):
                rang.setdefault(d[4] or d[3], set()).add(name)
    posts = [(m, root) for m, root in posts if (root or m)["ts"] not in inboxes and m["ts"] not in inboxes]
    humans = people_names(reg)

    def expected(name: str, t: float) -> str | None:
        to, r = routes.resolve(name)
        own = routes.get_route(name)
        if not own or float(own.get("inbox_ts") or 0) > t:
            return None
        closed = _stamp_epoch(own.get("closed"))
        if closed is None or closed > t:
            return name
        return to if r and not r.get("closed") else None

    out = []
    for m, root in posts:
        t, author = float(m["ts"]), poster(m, reg)
        history, branches = [], {}
        if root and is_announcement(root):
            history = [x for x in threads.get(root["ts"], []) if t - HISTORY_S <= float(x["ts"]) < t]
            nums = {n for x in [m, *history] for n in named_prs(x.get("text") or "")[0]}
            branches = pr_branches(nums, log) if nums else {}
        want = {e for n in route_targets(m, root, reg, inboxes, history, branches) if n not in humans and (e := expected(n, t))}
        got = rang.get(m["ts"], set())
        missed, extra = sorted(want - got - {author}), sorted(got - want)
        if missed or extra:
            out.append({"ts": m["ts"], "by": author, "missed": missed, "extra": extra,
                        "first": " ".join(plain(m.get("text") or "").split())[:80]})
    return out


def is_ask(m: dict) -> bool:
    """Whether a top-level post asks the names it rang: not an announcement, a status line, an inbox's head, quiet or an fyi."""
    text = m.get("text") or ""
    return not (is_announcement(m) or is_quiet(text) or text.startswith(INBOX_HEAD) or kind_of(m) in ("fyi", "status"))


ANSWER_REACTIONS = ("eyes", "white_check_mark", "x")
LINEAR_ANSWER_Q = ("query($id: String!) { issue(id: $id) { createdAt history(first: 100) { nodes { createdAt fromState { type } toState { type } } } "
                   "comments(first: 100) { nodes { createdAt user { name } botActor { userDisplayName } body } } } }")


def _iso_epoch(stamp: str) -> float:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def issue_answer(lin: Any, ident: str, asker: str | None) -> dict | None:
    """When the Linear issue `ident` was first taken up: its first move out of Backlog (by anyone: every write is the
    Verity Asks app's, so `by` is None), or the first comment by someone other than `asker`; {at, how, by}, or None when
    neither happened.
    An issue filed straight into another state shows that as a move out of Backlog at its creation, which is no answer."""
    i = lin.gql(LINEAR_ANSWER_Q, {"id": ident})["issue"]
    born = _iso_epoch(i["createdAt"])
    out = [{"at": _iso_epoch(h["createdAt"]), "how": "linear-state", "by": None}
           for h in i["history"]["nodes"]
           if (h.get("fromState") or {}).get("type") == "backlog" and (h.get("toState") or {}).get("type") not in (None, "backlog")
           and _iso_epoch(h["createdAt"]) > born]
    for c in i["comments"]["nodes"]:
        if (who := comment_by(c)) and who != asker:
            out.append({"at": _iso_epoch(c["createdAt"]), "how": "linear-comment", "by": who})
    return min(out, key=lambda a: a["at"], default=None)


def nearest_rank(xs: list[float], q: float) -> float | None:
    return sorted(xs)[max(math.ceil(q * len(xs)), 1) - 1] if xs else None


def ask_latency(api: Slack, reg: dict, routes: Routes, ch: str, since: float, now: float, stale_s: float,
                linear_client: Callable[[], Any] | None = None) -> dict:
    """Each ask posted in (since, now] that rang a lane's inbox, and how long the lane took to first answer it: the earliest of
    a reply in the ask's thread under a name of the lane's inbox (its own, or one it was renamed from), and (an ask linked
    to a Linear issue) the issue's first move out of Backlog or first comment by anyone but the asker (`issue_answer`), with
    who gave it (`answer_by`).  An ask with no timed answer that carries 👀, ✅ or ❌
    and rang only this lane counts as answered but untimed: Slack keeps no time for a reaction, and every handle reacts as
    one bot.  Answers after `now` don't count.  Per lane: asks, answered (by Slack, Linear or untimed), the nearest-rank
    median and p90 of the timed answers in minutes, the unanswered, and those unanswered for longer than `stale_s`."""
    tops: dict[str, dict[str, dict]] = {}                            # channel -> ts -> top-level post
    replies: dict[tuple[str, str], list[dict]] = {}
    asks, counted, names = [], set(), {}
    for r in routes.all().values():
        names.setdefault(r.get("inbox_ts"), set()).add(r["name"])  # a renamed lane answers under its old name too
    for r in sorted(routes.all().values(), key=lambda r: (bool(r.get("closed")), r["name"])):
        if not r.get("inbox_ts") or r["inbox_ts"] in counted:
            continue                                                 # a renamed name forwards into its successor's inbox
        counted.add(r["inbox_ts"])
        rch = r.get("channel") or ch
        if rch not in tops:
            tops[rch] = {m["ts"]: m for m in api.paged("conversations.history", "messages", channel=rch, oldest=f"{since:.6f}")
                         if float(m["ts"]) <= now}
        seen = set()
        for b in api.paged("conversations.replies", "messages", channel=rch, ts=r["inbox_ts"], oldest=f"{since:.6f}"):
            d = DOORBELL_RE.match(b.get("text") or "") if is_doorbell(b, reg) and b.get("ts") != r["inbox_ts"] else None
            if not d or (d[4] and d[4] != d[3]) or d[3] in seen:
                continue                                             # a reply's doorbell, or one `deliver` rang again
            root = tops[rch].get(d[3])
            if root and is_ask(root) and poster(root, reg) not in names[r["inbox_ts"]]:
                seen.add(d[3])
                asks.append({"lane": r["name"], "names": names[r["inbox_ts"]], "channel": rch, "root": root})
    rang: dict[tuple[str, str], int] = {}
    for a in asks:
        rang[(a["channel"], a["root"]["ts"])] = rang.get((a["channel"], a["root"]["ts"]), 0) + 1
    lin, issues, linear_errors = None, {}, []
    out = []
    for a in asks:
        root, lane, t0 = a["root"], a["lane"], float(a["root"]["ts"])
        key = (a["channel"], root["ts"])
        if root.get("reply_count") and key not in replies:
            replies[key] = api.paged("conversations.replies", "messages", channel=a["channel"], ts=root["ts"], oldest=root["ts"])
        found = [{"at": float(m["ts"]), "how": "slack-reply", "by": poster(m, reg)} for m in replies.get(key, [])
                 if m.get("ts") != root["ts"] and poster(m, reg) in a["names"] and t0 < float(m["ts"]) <= now]
        asker = poster(root, reg)
        if link := LINEAR_LINK_RE.search(root.get("text") or ""):
            if (link[1], asker) not in issues:
                from . import linear  # noqa: PLC0415
                try:
                    lin = lin or (linear_client or linear.Linear)()
                    issues[(link[1], asker)] = issue_answer(lin, link[1], asker)
                except (linear.LinearError, OSError, KeyError, TypeError, ValueError) as e:   # the ask still counts, from Slack
                    issues[(link[1], asker)] = None
                    linear_errors.append(f"{link[1]}: {e}")
            if (ans := issues[(link[1], asker)]) and ans["at"] <= now:
                found.append(ans)
        first = min(found, key=lambda x: x["at"], default=None)
        untimed = first is None and rang[key] == 1 and any(x.get("name") in ANSWER_REACTIONS for x in root.get("reactions", []))
        out.append({"lane": lane, "ts": root["ts"], "by": asker, "linear": link[1] if link else None,
                    "first": first["how"] if first else "reaction" if untimed else None, "answer_by": first["by"] if first else None,
                    "minutes": round(max(first["at"] - t0, 0) / 60, 1) if first else None,
                    "age_minutes": round((now - t0) / 60, 1)})
    lanes = []
    for lane in sorted({x["lane"] for x in out}):
        mine = [x for x in out if x["lane"] == lane]
        timed = [x["minutes"] for x in mine if x["minutes"] is not None]
        unanswered = [x for x in mine if not x["first"]]
        lanes.append({"lane": lane, "asks": len(mine), "answered": len(mine) - len(unanswered),
                      "slack": sum(x["first"] == "slack-reply" for x in mine),
                      "linear": sum((x["first"] or "").startswith("linear-") for x in mine),
                      "untimed": sum(x["first"] == "reaction" for x in mine),
                      "median_min": nearest_rank(timed, 0.5), "p90_min": nearest_rank(timed, 0.9),
                      "unanswered": len(unanswered), "stale": sum(x["age_minutes"] * 60 > stale_s for x in unanswered)})
    return {"since": since, "now": now, "lanes": lanes, "asks": out, "linear_errors": linear_errors}


def render_latency(rep: dict, hours: float) -> str:
    cols = ("asks", "answered", "slack", "linear", "untimed", "median_min", "p90_min", "unanswered", "stale")
    lines = [f"ask latency over the last {hours:g} h (to {both(rep['now'])}): minutes from an ask to its lane's first answer",
             f"  {'lane':<26}" + "".join(f"{c:>11}" for c in cols)]
    for x in rep["lanes"]:
        lines.append(f"  @{x['lane']:<25}" + "".join(f"{'-' if x[c] is None else x[c]:>11}" for c in cols))
    if rep["linear_errors"]:
        lines.append(f"  linear: {len(rep['linear_errors'])} issue(s) unread, counted from Slack alone: {rep['linear_errors'][0]}")
    return "\n".join(lines)


def cmd_traffic(ns: argparse.Namespace, reg: dict) -> int:
    """Doorbells per inbox (the most in any one hour), pauses, `@name`s nobody owns (dropped), and asks with no 👀 or reply
    after --stale-h hours, over the last --hours; with --latency, how fast each lane first answered its asks instead."""
    api, ch, routes = client(ns), channel_id(ns, reg, FRONT_DOOR), routes_of(ns)
    now = time.time() if ns.now is None else float(ns.now)
    since = now - ns.hours * 3600
    if ns.latency:
        rep = ask_latency(api, reg, routes, ch, since, now, ns.stale_h * 3600)
        print(json.dumps(rep, indent=2) if ns.json else render_latency(rep, ns.hours))
        return 0
    rows, counted = [], set()
    for r in sorted(routes.all().values(), key=lambda r: (bool(r.get("closed")), r["name"])):
        if not r.get("inbox_ts") or r["inbox_ts"] in counted:
            continue                                                 # a renamed name forwards into its successor's inbox
        counted.add(r["inbox_ts"])
        bells = [m for m in api.paged("conversations.replies", "messages", channel=r.get("channel") or ch, ts=r["inbox_ts"],
                                      oldest=f"{since:.6f}") if m.get("ts") != r["inbox_ts"] and is_doorbell(m, reg)]
        hours: dict[int, int] = {}
        for m in bells:
            hours[int(float(m["ts"]) // 3600)] = hours.get(int(float(m["ts"]) // 3600), 0) + 1
        rows.append({"name": r["name"], "doorbells": len(bells), "peak_per_hour": max(hours.values(), default=0),
                     "paused": sum(_from(m.get("text") or "", "router") for m in bells)})
    posts = api.paged("conversations.history", "messages", channel=ch, oldest=f"{since:.6f}")
    handles = {h["handle"] for h in reg["handles"]} | {poster({"user": p.get("slack_user")}, reg) for p in reg.get("humans", [])}
    dropped, stale, groups = [], [], {h.get("usergroup") for h in reg["handles"]}
    fresh: set[str] | None = None                                    # origin/main's group ids, fetched on the first unknown
    taken: dict[str, bool] = {}                                      # Linear issue -> left Backlog
    for m in posts:
        for n in lead_names(m.get("text") or "", reg):
            if not routes.get_route(n) and n not in handles:
                dropped.append({"name": n, "ts": m["ts"], "by": poster(m, reg)})
        lead = LEAD_RE.match(m.get("text") or "")
        for _, gid, label in NAME_TOKEN_RE.findall(lead.group(1) if lead else ""):
            if gid and not label and gid not in groups:
                if fresh is None:
                    fresh = origin_groups(registry_path(ns.registry))   # this checkout's registry may predate the group
                if gid not in fresh:                                 # a group no handle has: `lead_names` can't route it
                    dropped.append({"name": f"subteam^{gid}", "ts": m["ts"], "by": poster(m, reg)})
        ask = m.get("bot_id") and LEAD_RE.match(m.get("text") or "") and is_ask(m)  # older clients tagged an fyi without (quiet)
        answered = m.get("reply_count") or any(x.get("name") in ANSWER_REACTIONS for x in m.get("reactions", []))
        if ask and not answered and now - float(m["ts"]) > ns.stale_h * 3600:
            if not ((link := LINEAR_LINK_RE.search(m.get("text") or "")) and issue_taken(link[1], poster(m, reg), taken)):
                stale.append({"ts": m["ts"], "by": poster(m, reg), "to": lead_names(m.get("text") or "", reg)})
    replayed = replay(api, reg, routes, ch, since) if ns.replay else None
    if ns.json:
        print(json.dumps({"inboxes": rows, "dropped": dropped, "unanswered": stale,
                          **({"replay": replayed} if replayed is not None else {})}, indent=2))
        return 0
    print(f"traffic over the last {ns.hours:g} h (to {both(now)})")
    for x in rows:
        print(f"  @{x['name']:<28} {x['doorbells']:>4} doorbells, at most {x['peak_per_hour']} in an hour" +
              (f", {x['paused']} paused" if x["paused"] else ""))
    for d in dropped:
        print(f"  DROPPED @{d['name']}: {d['ts']} by @{d['by']} (no inbox and no handle)")
    for s in stale:
        print(f"  UNANSWERED {s['ts']} by @{s['by']} to {' '.join('@' + t for t in s['to']) or 'no one it can ring'} ({both(s['ts'])})")
    for x in replayed or []:
        what = "; ".join(f"{k} {' '.join('@' + n for n in x[k])}" for k in ("missed", "extra") if x[k])
        print(f"  REPLAY {x['ts']} by @{x['by']}: {what}: {x['first']}")
    if replayed is not None:
        print(f"  replayed: {len(replayed)} post(s) whose doorbells differ from route_targets")
    return 0


def cmd_groups_sync(ns: argparse.Namespace, reg: dict, path: Path) -> int:
    """One user group per registry handle (`handle` = its handle, `name` = its handle, `description` = its remit), created,
    re-enabled or updated, and a retired name's group (no handle, its inbox closed) disabled; with --apply the groups' ids, and the bot's own id when the registry lacks it, are written back into
    the registry, which the caller commits. Existing groups keep their members."""
    api = client(ns)
    write = ns.apply and not ns.dry_run
    if write and api.via == "broker":
        raise SlackError("groups sync --apply needs SLACK_BOT_TOKEN itself (the broker doesn't relay user-group writes): "
                         "run it on a VM that has the token")
    changed = False
    bot = api.call("auth.test").get("bot_id")
    if bot and reg.get("bot_id") != bot:
        print(f"record  bot {bot} in {path}")
        reg["bot_id"], changed = bot, True
    groups = api.call("usergroups.list", include_disabled="true").get("usergroups", [])
    live, by_id = {g.get("handle"): g for g in groups}, {g.get("id"): g for g in groups}
    for h in reg["handles"]:
        name, desc = h["handle"], h["remit"][:250]
        g = by_id.get(h.get("usergroup")) or live.get(name)          # by id first: a renamed handle keeps its group
        if g is None:
            print(f"create  @{name}")
            gid = (api.call("usergroups.create", name=name, handle=name, description=desc).get("usergroup") or {}).get("id") if write else None
        else:
            gid = g["id"]
            if g.get("date_delete"):
                print(f"enable  @{name} ({gid})")
                if write:
                    api.call("usergroups.enable", usergroup=gid)
            if g.get("handle") != name or g.get("name") != name or (g.get("description") or "") != desc:
                print(f"update  @{name} ({gid}): " + (f"handle @{g.get('handle')} -> @{name}" if g.get("handle") != name
                                                     else "name/description"))
                if write:
                    api.call("usergroups.update", usergroup=gid, name=name, handle=name, description=desc)
            elif not g.get("date_delete"):
                print(f"ok      @{name} ({gid})")
        if gid and h.get("usergroup") != gid:
            print(f"record  @{name} -> {gid} in {path}")
            h["usergroup"] = gid
            changed = True
    handles, ids = {h["handle"] for h in reg["handles"]}, {h.get("usergroup") for h in reg["handles"]}
    if extra := [g for n, g in live.items() if n not in handles and g.get("id") not in ids and not g.get("date_delete")]:
        routes = routes_of(ns)
        for g in extra:
            r = routes.get_route(g["handle"])
            if r and r.get("closed"):                                # a retired name: tagging it would ring nobody
                print(f"disable @{g['handle']} ({g['id']}): retired, its inbox closed {r['closed']}" +
                      (f" and forwards to @{r['forward_to']}" if r.get("forward_to") else ""))
                if write:
                    api.call("usergroups.disable", usergroup=g["id"])
    if write and changed:
        path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {path}: commit it")
    elif not write:
        print("plan only: rerun with --apply to change Slack and the registry")
    return 0


# --- #ask-daniel ------------------------------------------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def pacific(t: datetime) -> str:
    """How a person reads a time: `3:00 PM PDT, Wed 30 Sep`."""
    return t.astimezone(PACIFIC).strftime("%I:%M %p %Z, %a %d %b").lstrip("0")


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def since_ts(raw: str) -> str:
    """`--since`: a Slack ts as given, or an ISO time with its zone as one.  Slack answers a date passed as `oldest` with no
    messages, so a read that took one reported a quiet channel."""
    if re.fullmatch(r"\d+(\.\d+)?", raw):
        return raw
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        t = None
    if t is None or t.tzinfo is None:
        raise argparse.ArgumentTypeError(f"{raw!r}: give a Slack ts such as 1791876703.181769, or an ISO time with its zone "
                                         f"such as 2026-10-13T07:05:00Z")
    return f"{t.timestamp():.6f}"


def parse_deadline(raw: str, now: datetime) -> datetime:
    """`3:00 PM` (today, Pacific) or an ISO time with its zone, in UTC."""
    raw = raw.strip()
    if m := CLOCK_RE.match(raw):
        hour, minute = int(m[1]), int(m[2])
        if not (1 <= hour <= 12 and minute < 60):
            raise SlackError(f"--deadline {raw!r}: not a clock time")
        hour = hour % 12 + (12 if m[3].lower() == "pm" else 0)
        return now.astimezone(PACIFIC).replace(hour=hour, minute=minute, second=0, microsecond=0).astimezone(timezone.utc)
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        raise SlackError(f"--deadline {raw!r}: give `3:00 PM` (today, Pacific) or an ISO time such as 2026-09-30T23:00:00Z") from None
    if t.tzinfo is None:
        raise SlackError(f"--deadline {raw!r}: an ISO time needs its zone (Z, or an offset such as -07:00)")
    return t.astimezone(timezone.utc)


def card(ns: argparse.Namespace, reg: dict, now: datetime) -> dict:
    """The card's fields (`note:20260930T2045Z-handoff-from-infra-ask-daniel-card-fields`), checked as the route checks them."""
    need = {"--as": ns.handle, "--kind": ns.kind, "--question": ns.question, "--why": ns.why, "--recommend": ns.recommend,
            "--default": ns.default, "--deadline": ns.deadline, "--link or --page": ns.link or ns.page}
    if missing := [k for k, v in need.items() if not v]:
        raise SlackError(f"ask-daniel needs {', '.join(missing)}")
    if ns.link and ns.page:
        raise SlackError("--page or --link, not both: --page creates the ask's page, --link names one you made")
    h = entry(reg, ns.handle)
    question = ns.question.strip()
    if not question or "\n" in question:
        raise SlackError("--question is one line: one question per card, and the context goes behind --link")
    if not ns.why.strip():
        raise SlackError("--why is empty")
    options: list[dict] = []
    for raw in ns.option:
        m = OPTION_RE.match(raw)
        if not m or not m[2].strip() or len(m[2].strip()) > LABEL_MAX:
            raise SlackError(f"--option {raw!r}: give value=label, the value [a-z0-9_-] (at most 32), the label 1 to {LABEL_MAX} characters")
        if any(o["value"] == m[1] for o in options):
            raise SlackError(f"--option {m[1]!r} appears twice")
        options.append({"value": m[1], "label": m[2].strip()})
    if not OPTIONS_MIN <= len(options) <= OPTIONS_MAX:
        raise SlackError(f"a card has {OPTIONS_MIN} to {OPTIONS_MAX} options (--option value=label); this one has {len(options)}")
    values = [o["value"] for o in options]
    for flag, v in (("--recommend", ns.recommend), ("--default", ns.default)):
        if v not in values:
            raise SlackError(f"{flag} {v!r} is not an option ({', '.join(values)})")
    for o in options:
        if o["value"] == ns.recommend:
            o["recommended"] = True
    deadline = parse_deadline(ns.deadline, now)
    if deadline <= now:
        raise SlackError(f"--deadline {pacific(deadline)} has passed (it is {pacific(now)})")
    if deadline - now > DEADLINE_MAX:
        raise SlackError(f"--deadline {pacific(deadline)} is more than {DEADLINE_MAX.days} days away")
    if ns.link:
        if not ns.link.startswith("https://"):
            raise SlackError(f"--link {ns.link!r}: an https URL to the ask's page")
        host = urllib.parse.urlsplit(ns.link).hostname or ""
        if host not in NOTION_HOSTS and not host.endswith(".notion.site"):
            raise SlackError(f"--link {ns.link!r}: the ask's page in the Asks for Daniel database ({asks_db(reg)['database']}), "
                             f"written to the template in {SKILL_ASKING}; or pass the page's Markdown as --page FILE")
    else:
        if ns.area not in AREAS:
            raise SlackError(f"--page needs --area, one of: {', '.join(AREAS)}")
        check_page(ns.page)
    url = ns.agent_url or os.environ.get("RESEARCH_AGENT_URL")
    if not url and not ns.dry_run and (me := self_agent_id()):
        url = f"https://cursor.com/agents/{me}"
    if not url:
        raise SlackError("no --agent-url (or $RESEARCH_AGENT_URL): every card links the agent that asked it")
    return {"kind": ns.kind, "question": question, "why": ns.why.strip(), "options": options, "default": ns.default,
            "deadline": _iso(deadline), "link": ns.link, "requested_by": h["handle"], "agent_url": url}


def asks_db(reg: dict) -> dict:
    """The Asks for Daniel database: `{database, data_source}`, the ask-daniel channel's `pages` in the registry."""
    db = (reg.get("channels", {}).get(ASK_CHANNEL) or {}).get("pages") or {}
    if not db.get("database") or not db.get("data_source"):
        raise SlackError(f"the registry's #{ASK_CHANNEL} has no `pages` (the Asks for Daniel database's URL and data source id)")
    return db


def check_page(path: str) -> str:
    """An ask's page, as Markdown: the template's sections, in order (`SKILL_ASKING`)."""
    try:
        text = Path(path).read_text()
    except OSError as e:
        raise SlackError(f"--page {path}: {e.strerror or e}") from None
    at = [m.start() if (m := re.search(rf"(?m)^{re.escape(s)}[ \t]*$", text)) else -1 for s in PAGE_SECTIONS]
    if missing := [s for s, i in zip(PAGE_SECTIONS, at) if i < 0]:
        raise SlackError(f"--page {path} lacks {', '.join(missing)}: write it to the template in {SKILL_ASKING}")
    if at != sorted(at):
        raise SlackError(f"--page {path}: its sections go in the template's order: {', '.join(PAGE_SECTIONS)}")
    return text


def notion_error(what: str, e: urllib.error.HTTPError) -> str:
    try:
        body = json.load(e)
    except (ValueError, OSError):
        body = {}
    why = f"HTTP {e.code} {body.get('code', '')}".rstrip() + (f": {body['message']}" if body.get("message") else "")
    if e.code == 401:
        return f"{what}: Notion rejected $NOTION_API_KEY ({why})"
    if e.code == 404:
        return f"{what}: Notion can't see it ({why}): share the Asks for Daniel database with the token's connection (its ... menu > Connections)"
    return f"{what}: Notion answered {why}"


class Notion:
    """The Asks for Daniel database through Notion's API, with $NOTION_API_KEY: a Notion connection's token, shared on the
    database. $RESEARCH_NOTION_URL points it at another server (the tests' fake)."""

    def __init__(self):
        self.token = os.environ.get("NOTION_API_KEY")
        if not self.token:
            raise SlackError("no $NOTION_API_KEY (a Notion connection's token, shared on the Asks for Daniel database): write the "
                             "page with the Notion MCP and pass its URL as --link instead of --page")
        self.base = (os.environ.get("RESEARCH_NOTION_URL") or NOTION_API).rstrip("/")

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {self.token}", "Notion-Version": NOTION_VERSION,
                   **({"Content-Type": "application/json"} if data else {})}
        what = f"{method} {NOTION_API}{path.split('?')[0]}"
        try:
            out = http_json(method, self.base + path, data, headers)
        except urllib.error.HTTPError as e:
            raise SlackError(notion_error(what, e)) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise SlackError(f"{what}: Notion is unreachable: {getattr(e, 'reason', e)}") from None
        if not isinstance(out, dict):
            raise SlackError(f"{what}: the answer is not a JSON object")
        return out


def _text(s: str) -> dict:
    return {"rich_text": [{"text": {"content": s[:2000]}}]}


def page_request(body: dict, area: str, markdown: str, db: dict) -> dict:
    """POST /v1/pages: the ask's page in the database, its properties from the card's fields."""
    labels = {o["value"]: o["label"] for o in body["options"]}
    rec = next(o["label"] for o in body["options"] if o.get("recommended"))
    when = pacific(datetime.fromisoformat(body["deadline"]))
    unanswered = f"{labels[body['default']]} applies at {when}" if body["kind"] == "default" else f"Nothing happens; it waits for you (deadline {when})"
    return {"parent": {"type": "data_source_id", "data_source_id": db["data_source"]},
            "properties": {"Ask": {"title": [{"text": {"content": body["question"][:2000]}}]},
                           "Status": {"status": {"name": "Waiting on Daniel"}},
                           "Lane": {"select": {"name": body["requested_by"]}},
                           "Kind": {"select": {"name": body["kind"].capitalize()}},
                           "Area": {"select": {"name": area}},
                           "Recommendation": _text(rec), "If no answer": _text(unanswered),
                           "Deadline": {"date": {"start": body["deadline"]}},
                           "Lane checked": {"checkbox": True},
                           "Asked by": {"url": body["agent_url"]}},
            "markdown": with_reply(markdown), "markdown_version": "v2"}


def with_reply(markdown: str) -> str:
    """The page with Daniel's yellow reply callout just above `## The problem`, unless it has one."""
    if "**Your reply.**" in markdown:
        return markdown
    at = re.search(rf"(?m)^{re.escape(PAGE_SECTIONS[0])}[ \t]*$", markdown).start()
    return markdown[:at] + REPLY_CALLOUT + markdown[at:]


def prop_text(p: dict | None) -> str:
    """A page property's value as text."""
    if not p:
        return ""
    kind = p.get("type")
    v = p.get(kind)
    if kind in ("title", "rich_text"):
        return "".join(x.get("plain_text") or x.get("text", {}).get("content", "") for x in v or [])
    if kind in ("select", "status"):
        return (v or {}).get("name", "")
    if kind == "date":
        return (v or {}).get("start", "")
    if kind == "checkbox":
        return "yes" if v else "no"
    return v if isinstance(v, str) else ""


def cmd_pages(ns: argparse.Namespace, reg: dict) -> int:
    """The asks' pages edited in the last --hours, newest first: Status, Decision, and the newest comment."""
    db = asks_db(reg)
    where: list[dict] = [{"timestamp": "last_edited_time", "last_edited_time": {"on_or_after": _iso(_now() - timedelta(hours=ns.hours))}}]
    if ns.handle:
        where.append({"property": "Lane", "select": {"equals": entry(reg, ns.handle)["handle"]}})
    query = {"filter": {"and": where}, "sorts": [{"timestamp": "last_edited_time", "direction": "descending"}], "page_size": 100}
    if ns.dry_run:
        print(f"dry-run: POST {NOTION_API}/data_sources/{db['data_source']}/query {json.dumps(query, sort_keys=True)}", file=sys.stderr)
        return 0
    notion = Notion()
    rows = []
    for p in notion.call("POST", f"/data_sources/{db['data_source']}/query", query).get("results", []):
        props = p.get("properties", {})
        comments = notion.call("GET", f"/comments?block_id={urllib.parse.quote(p['id'])}&page_size=100").get("results", [])
        newest = comments[-1] if comments else None
        rows.append({"url": p.get("url"), "edited": p.get("last_edited_time"),
                     **{k.lower().replace(" ", "_"): prop_text(props.get(k)) for k in ("Ask", "Lane", "Status", "Decision", "Card ID", "Deadline")},
                     "comments": len(comments),
                     "newest_comment": newest and {"at": newest.get("created_time"), "text": prop_text({"type": "rich_text", "rich_text": newest.get("rich_text")})}})
    if ns.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    for r in rows:
        print(f"{r['status'] or '-':18} @{r['lane'] or '-':19} {r['ask']}  {r['url']}")
        if r["decision"]:
            print(f"    decision: {r['decision']}")
        if c := r["newest_comment"]:
            print(f"    {r['comments']} comment(s); newest {c['at']}: {c['text'][:300]}")
    return 0


def fallback(body: dict) -> dict:
    """The card in the route's old shape (`{title, text, link, requested_by}`, an Approve/Deny card), for a route that refuses
    the new fields: the question is the title, and the text carries the rest, its times in Pacific."""
    labels = {o["value"]: o["label"] for o in body["options"]}
    rec = next(o["value"] for o in body["options"] if o.get("recommended"))
    dflt = body["default"]
    lines = [f"*{body['kind'].upper()}*", f"*Why:* {body['why']}", "*Options:*",
             *(f"\u2022 `{o['value']}`: {o['label']}" + (" (recommended)" if o.get("recommended") else "") for o in body["options"]),
             f"*Default:* {labels[dflt]} (`{dflt}`)" + (", which applies at the deadline unless you object" if body["kind"] == "default"
                                                         else "; this card waits for your answer"),
             f"*Deadline:* {pacific(datetime.fromisoformat(body['deadline']))}",
             f"Approve takes the recommendation ({labels[rec]}) and Deny declines it; to choose another option, reply in the thread.",
             f"_asked by <{body['agent_url']}|@{body['requested_by']}'s agent>_"]
    return {"title": body["question"], "text": "\n".join(lines), "link": body["link"], "requested_by": body["requested_by"]}


class RouteError(SlackError):
    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def http_json(method: str, url: str, data: bytes | None, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.build_opener(_NoRedirect).open(req, timeout=30) as r:
        return json.load(r)


def route_error(what: str, e: urllib.error.HTTPError) -> str:
    why = f"HTTP {e.code}{_code(e)}"
    if e.code == 401:
        return f"{what}: the route rejected this agent's Cursor identity ({why}): it takes managed agents on compute-verification/verity only"
    if e.code == 403:
        return f"{what}: the route refused ({why}): only a card's own asker may resolve it"
    if e.code == 429:
        return f"{what}: the route is rate limiting ({why}; retry after {e.headers.get('Retry-After')} s)"
    if e.code >= 500:
        return f"{what}: the route is down ({why}); ask @top"
    return f"{what}: the route answered {why}"


class AskDaniel:
    """The #ask-daniel cards' route: JSON with this agent's Cursor OIDC token (audience ASK_DANIEL). $RESEARCH_ASK_DANIEL_URL
    points it at another server (the tests' fake); the audience stays the route's."""

    def __init__(self):
        self.base = (os.environ.get("RESEARCH_ASK_DANIEL_URL") or ASK_DANIEL).rstrip("/")
        self._oidc = _Identity(ASK_DANIEL)

    def call(self, method: str, path: str = "", body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        ctype = {"Content-Type": "application/json"} if data else {}
        what = f"{method} {ASK_DANIEL}{path}"
        try:
            out = self._oidc.send(lambda t: http_json(method, self.base + path, data, {"Authorization": f"Bearer {t}", **ctype}))
        except urllib.error.HTTPError as e:
            raise RouteError(route_error(what, e), e.code) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise SlackError(f"{what}: the route is unreachable: {getattr(e, 'reason', e)}") from None
        if not isinstance(out, dict):
            raise SlackError(f"{what}: the answer is not a JSON object")
        return out


def wait_for(get: Callable[[], dict], *, sleep: Callable[[float], None], every: float, polls: int) -> dict:
    """GET a card until it leaves `pending` (decided, or a blocking card overdue), at most `polls` times, `every` s apart."""
    for i in range(max(1, polls)):
        out = get()
        if out.get("status") != "pending" or i == polls - 1:
            return out
        sleep(every)
    return out


def ring_decision(ns: argparse.Namespace, reg: dict, cid: str, out: dict) -> None:
    """Once a card is decided, ring @top's inbox with its question, outcome and asker, so the decision reaches Daniel's
    log without a TOP issue filed for it (COMM-34).  Once per card: a ring already in the inbox isn't repeated."""
    if out.get("status") not in DECIDED or ns.dry_run:
        return
    to, r = routes_of(ns).resolve("top")
    if not r or r.get("closed"):
        return
    api, ch = client(ns), r.get("channel") or channel_id(ns, reg, FRONT_DOOR)
    mark = f"card `{cid}` decided"
    if any(is_doorbell(m, reg) and (m.get("text") or "").startswith(f"from ask-daniel, {mark}") for m in api.paged("conversations.replies", "messages", channel=ch, ts=r["inbox_ts"],
                                                             oldest=f"{_now().timestamp() - 14 * 86400:.6f}")):
        return
    question, asker, link = "", "", ""
    if out.get("ts") and (card_ch := out.get("channel") or channel_id_for(reg, ASK_CHANNEL)):
        try:
            card = fetch_message(api, card_ch, out["ts"], None) or {}
            blocks = json.dumps(card.get("blocks") or [])
            if m := re.search(r'"text": "(?:`[A-Z]+` )?\*(.+?)\*"', blocks):
                question = m[1]
            if m := re.search(r"asked by <[^|>]*\|([a-z0-9._-]+)>", blocks):
                asker = m[1]
            link = permalink(reg, card_ch, out["ts"])
        except SlackError:
            pass
    text = (f"from ask-daniel, {mark}: `{out.get('choice') or '-'}` ({out.get('source') or '-'}, {out.get('decided_at') or '-'})"
            + (f" on: {question}" if question else "") + (f"; asked by {asker}" if asker else "") + (f" {link}" if link else ""))
    api.call("chat.postMessage", channel=ch, thread_ts=r["inbox_ts"], username="router", unfurl_links="false", text=text)
    print(f"route @{to}: rang with the decision", file=sys.stderr)


def _show(out: dict, as_json: bool) -> int:
    if as_json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(" ".join(str(out.get(k) or "-") for k in ("status", "choice", "source", "decided_at")))
    return 0 if out.get("status") in DECIDED else 1


def permalink(reg: dict, ch: str, ts: str) -> str:
    return f"https://{reg['workspace']}/archives/{ch}/p{ts.replace('.', '')}"


def no_such_card(verb: str, cid: str) -> str:
    """The route answers 404 only for an ID with no card: a pending card's status is 200."""
    short = " The route takes the whole ID and looks up no prefix." if re.fullmatch(r"[0-9a-f]{4,12}", cid) else ""
    return (f"ask-daniel {verb} {cid}: HTTP 404: no card has that ID.{short} It is the UUID `ask-daniel` printed first when "
            "it posted the card, which its mention in the card's thread also ends with (`research msg read --channel ask-daniel "
            "--thread TS`), not the card's Slack ts; for a card posted any other way, record Daniel's thread reply as a ruling")


def cmd_ask_daniel(ns: argparse.Namespace, reg: dict) -> int:
    if ns.verb == "pages":
        return cmd_pages(ns, reg)
    if ns.verb:
        if not ns.id:
            raise SlackError(f"ask-daniel {ns.verb} needs the card's ID")
        path = "/" + urllib.parse.quote(str(ns.id), safe="")
        if ns.verb == "resolve":
            if not ns.choice:
                raise SlackError("ask-daniel resolve needs --choice VALUE: the option Daniel chose in the thread")
            body = {"choice": ns.choice, **({"note": ns.note} if ns.note else {})}
            if ns.dry_run:
                print(f"dry-run: POST {ASK_DANIEL}{path}/resolve {json.dumps(body, sort_keys=True)}", file=sys.stderr)
                return 0
            try:
                out = AskDaniel().call("POST", f"{path}/resolve", body)
                ring_decision(ns, reg, str(ns.id), out)
                return _show(out, ns.json)
            except RouteError as e:
                if e.status == 404:
                    raise SlackError(no_such_card("resolve", ns.id)) from None
                raise
        if ns.dry_run:
            print(f"dry-run: GET {ASK_DANIEL}{path}", file=sys.stderr)
            return 0
        route = AskDaniel()
        polls = math.ceil((DEADLINE_MAX + WAIT_GRACE).total_seconds() / ns.every) if ns.wait else 1
        try:
            out = wait_for(lambda: route.call("GET", path), sleep=_sleep, every=ns.every, polls=polls)
            ring_decision(ns, reg, str(ns.id), out)
            return _show(out, ns.json)
        except RouteError as e:
            if e.status == 404:
                raise SlackError(no_such_card("status", ns.id)) from None
            raise

    now = _now()
    body = card(ns, reg, now)
    acting_as(ns, reg, ns.handle)
    made: dict = {}
    if ns.page:
        req = page_request(body, ns.area, check_page(ns.page), asks_db(reg))
        if ns.dry_run:
            print(f"dry-run: POST {NOTION_API}/pages {json.dumps(req, sort_keys=True, ensure_ascii=False)}", file=sys.stderr)
            body["link"] = "https://app.notion.com/p/<the new page>"
        else:
            notion = Notion()
            made = notion.call("POST", "/pages", req)
            if not made.get("url") or not made.get("id"):
                raise SlackError("Notion's answer has no page url or id")
            body["link"] = made["url"]
            print(f"page {made['url']}", file=sys.stderr)
    old = fallback(body)
    if ns.dry_run:
        print(f"dry-run: POST {ASK_DANIEL} {json.dumps(body, sort_keys=True, ensure_ascii=False)}", file=sys.stderr)
        print(f"dry-run: on HTTP 400, the old shape: POST {ASK_DANIEL} {json.dumps(old, sort_keys=True, ensure_ascii=False)}", file=sys.stderr)
        return 0
    route = AskDaniel()
    fell_back = False
    try:
        try:
            out = route.call("POST", "", body)
        except RouteError as e:
            if e.status != 400:
                raise
            print(f"research msg: the route refused the card's fields ({e}); posting the old shape", file=sys.stderr)
            out, fell_back = route.call("POST", "", old), True
        cid, ts = out.get("id"), out.get("ts")
        if not cid or not ts:
            raise SlackError("the route's answer has no id or ts")
    except SlackError as e:
        if made:
            raise SlackError(f"{e}; the page is up at {made['url']}: rerun with --link {made['url']} in place of --page") from None
        raise
    ch = out.get("channel") or channel_id_for(reg, ASK_CHANNEL)
    print(f"{cid} {ts} {permalink(reg, ch, ts)}")
    if made:
        try:
            notion.call("PATCH", f"/pages/{made['id']}", {"properties": {"Card ID": _text(str(cid)), "Slack card": {"url": permalink(reg, ch, ts)}}})
        except SlackError as e:
            print(f"the card is up, but its page doesn't name it ({e}): set the page's Card ID and Slack card yourself", file=sys.stderr)
    deciders = " ".join(f"<@{p['slack_user']}>" for p in reg.get("humans", []) if p.get("slack_user"))
    when = pacific(datetime.fromisoformat(body["deadline"]))
    rec = next((o["label"] for o in body["options"] if o.get("recommended")), "")
    ping = (f"{deciders} a decision for you: {body['question']} I recommend: {rec}. "
            + (f"`{body['default']}` applies at {when} unless you choose." if body["kind"] == "default" else f"Blocking: nothing happens until you choose (by {when}).")
            + f" Read it, comment or decide on its page: {body['link']} (card `{cid}`)")
    try:                                                             # the card alone notifies nobody; a mention does
        client(ns).call("chat.postMessage", channel=ch, thread_ts=ts, username=body["requested_by"], text=ping, unfurl_links="false")
    except SlackError as e:
        print(f"the card is up, but the mention didn't post ({e}): mention Daniel on it yourself", file=sys.stderr)
    shape = "the old shape (the route predates the card's fields)" if fell_back else "the card's fields"
    print(f"posted as {shape}; now subscribe_slack_thread(channelId={ch}, threadTs={ts}, expiresInSeconds={EXPIRES_S})", file=sys.stderr)
    if fell_back and body["kind"] == "default":
        print(f"the old shape never applies a default: if Daniel hasn't answered by {pacific(datetime.fromisoformat(body['deadline']))}, "
              f"`{body['default']}` applies, and you record it", file=sys.stderr)
    if not ns.wait:
        return 0
    polls = math.ceil((datetime.fromisoformat(body["deadline"]) - now + WAIT_GRACE).total_seconds() / ns.every)
    out = wait_for(lambda: route.call("GET", f"/{urllib.parse.quote(str(cid), safe='')}"), sleep=_sleep, every=ns.every, polls=polls)
    ring_decision(ns, reg, str(cid), out)
    return _show(out, ns.json)


def main(argv: list[str], prog: str = "research msg") -> int:
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--dry-run", action="store_true", help="send nothing to Slack; print each request instead")
    c.add_argument("--registry", default=None, help="the registry (default $RESEARCH_SLACK_REGISTRY or the checkout's)")
    c.add_argument("--channel", default=None, metavar="KEY", help=f"a registry channel (default {FRONT_DOOR})")
    c.add_argument("--channel-id", default=None, metavar="ID", help="its Slack id (default $SLACK_CHANNEL_<KEY>, else the registry's)")

    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name: str, help: str, *, as_: bool = False, thread: bool = False, body: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[c], help=help)
        if as_:
            p.add_argument("--as", dest="handle", required=True, metavar="HANDLE", help="the handle you hold")
        if thread:
            p.add_argument("--thread", required=True, metavar="TS", help="the thread's root ts")
        if body:
            p.add_argument("--text", default=None)
            p.add_argument("--file", default=None, metavar="PATH", help="the body from PATH (`-` or neither: stdin)")
        return p

    cmd("handles", "the registry").add_argument("--json", action="store_true")
    cmd("owners", "the handle that owns each path (`-`: nobody)").add_argument("paths", nargs="*", metavar="PATH",
                                                                               help="checkout paths (none: read them from stdin)")
    cmd("subscriptions", "the subscribe calls this handle's holder makes", as_=True)
    p = cmd("send", "post a message: a new thread, or into --thread; the --to names lead it and get it", as_=True, body=True)
    p.add_argument("--to", nargs="+", action="extend", default=[], metavar="@NAME", help="who it's for: handles or names with an inbox")
    p.add_argument("--thread", default=None, metavar="TS", help="reply in this thread (default: start one)")
    p.add_argument("--kind", default=None, metavar="WORD",
                   help="a word the footer carries (ask, the default top-level with --to; announce; "
                        "fyi, which implies --quiet; ...)")
    p.add_argument("--verify", default=None, metavar="CMD", help="announce: a required action, and the command that shows it took")
    p.add_argument("--ships", action="append", default=[], metavar="URL#sha256=HEX|git:SHA", help="announce: code it runs, pinned")
    p.add_argument("--quiet", action="store_true", help="wake only the names it starts with: no reply needed")
    p.add_argument("--urgent", action="store_true", help="the doorbells say URGENT, and `read --as` lists them first")
    p = cmd("close", "close a thread you were asked in: a status line (done, or --declined WHY), and the root's reaction",
            as_=True, thread=True)
    p.add_argument("--declined", default=None, metavar="WHY")
    p.add_argument("--link", default=None, metavar="URL", help="the result: a PR, file or art: id")
    p.add_argument("--note", default=None)
    p = cmd("ask", "1:1: tag one handle or name in #agent-coordination, and ring its doorbell", as_=True, body=True)
    p.add_argument("--to", required=True, metavar="@OWNER")
    p.add_argument("--urgent", action="store_true", help="the doorbell says URGENT")
    p = cmd("linear", "Linear issues: `check` (what the credentials reach; exit 0 when fit for asks), `state ISSUE STATE` "
                      "(by name; " + LINEAR_STATES + "), `comment ISSUE TEXT --as HANDLE`, `list TEAM [--state S]` (open by default)")
    p.add_argument("verb", choices=("check", "state", "comment", "list"))
    p.add_argument("target", nargs="?", help="state, comment: the issue (INF-12); list: the team (a handle or its key)")
    p.add_argument("value", nargs="?", help="state: the state's name; comment: the text")
    p.add_argument("--state", default=None, metavar="NAME", help="list: only issues in this state (default: every open one)")
    p.add_argument("--write", default=None, metavar="TEAM", help="check: also file, comment on and cancel a test issue in TEAM")
    p.add_argument("--as", dest="handle", default=None, metavar="HANDLE",
                   help="comment: its sender (required); check: the test issue's sender (default comms)")
    p = cmd("announce", "1:some or 1:all in #agent-coordination", as_=True, body=True)
    p.add_argument("--to", nargs="+", action="extend", default=[], metavar="@HANDLE", help="the handles or names it's for (default: every handle)")
    p.add_argument("--verify", default=None, metavar="CMD", help="a required action: the command that shows it took")
    p.add_argument("--ships", action="append", default=[], metavar="URL#sha256=HEX|git:SHA", help="code it runs, pinned")
    p.add_argument("--urgent", action="store_true", help="the doorbells say URGENT")
    p = cmd("reply", "a post in a thread; `@name` at its start also rings that name's doorbell", as_=True, thread=True, body=True)
    p.add_argument("--quiet", action="store_true", help="ring only the names it starts with: no reply needed")
    p.add_argument("--urgent", action="store_true", help="the doorbells say URGENT")
    p = cmd("names", "the leads' names, each an inbox thread: open, read, close, list")
    p.add_argument("verb", choices=("open", "read", "close", "list"))
    p.add_argument("--as", dest="handle", default=None, metavar="NAME", help="the inbox's name: your handle")
    p.add_argument("--holder", default=None, metavar="BC_ID", help="open: the agent that holds it (default: this agent)")
    p.add_argument("--succeeds", action="store_true", help="open: take over an inbox another agent holds (a successor)")
    p.add_argument("--poster", action="append", default=[], metavar="BC_ID",
                   help="open: another agent that posts as this name (`--as`), such as a lane of its holder's; repeatable")
    p.add_argument("--remit", default=None, metavar="TEXT", help="open: what the name is for, one line (a successor keeps it)")
    p.add_argument("--forward-to", default=None, metavar="@NAME", help="close: whose doorbell rings instead")
    p.add_argument("--peek", action="store_true", help="read: don't mark them read")
    p.add_argument("--urgent", action="store_true", help="read: exit 1 unless an unread doorbell is URGENT")
    p.add_argument("--json", action="store_true")
    p = cmd("watch", "block until a new doorbell reaches NAME, print it, exit 0 (the hook for a wake script)", as_=True)
    p.add_argument("--every", type=float, default=20, metavar="S", help="seconds between polls (default 20)")
    p.add_argument("--polls", type=int, default=0, help="give up (exit 1) after this many polls (default: never)")
    p = cmd("traffic", "the monitor: doorbells per inbox, pauses, dropped names, unanswered asks")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--stale-h", type=float, default=2, help="an ask with no 👀, ✅, ❌ or reply after this many hours is unanswered")
    p.add_argument("--replay", action="store_true", help="also compare each post's doorbells with what route_targets says it rings")
    p.add_argument("--latency", action="store_true", help="instead: per lane, its asks, how many it answered (on Slack or Linear), "
                                                         "and the median and p90 minutes to its first answer")
    p.add_argument("--now", default=None, metavar="EPOCH", help="the window's end, in seconds since the epoch (default: now)")
    p.add_argument("--json", action="store_true")
    cmd("pickup", "\U0001f440 on an ask or alert you own", thread=True)
    p = cmd("retract", "blank a post your name sent (needs SLACK_BOT_TOKEN: the broker allows no edits)", as_=True)
    p.add_argument("--ts", required=True, help="the post's ts")
    p.add_argument("--thread", default=None, metavar="ROOT", help="its thread's root ts, when it is a reply")
    p.add_argument("--why", default=None, help="one line the blanked post keeps")
    p = cmd("roster", "an announcement's status lines; exit 0 only when every addressed handle is done", thread=True)
    p.add_argument("--json", action="store_true")
    p = cmd("match", "exit 0 if the post is addressed to you, 1 if not, 2 on error", as_=True)
    p.add_argument("--event", required=True, help="the message event as JSON, plain text, @PATH or - (stdin)")
    p.add_argument("--why", action="store_true", help="print the reason (or `not addressed`) on stderr")
    p = cmd("deliver", "ring the doorbells for a post no client delivered (a person's typed post); --channel ask-the-team: "
                       "a post there, whose doorbells name the channel")
    p.add_argument("--ts", default=None, help="the post's ts (default: the event's)")
    p.add_argument("--event", default=None, metavar="JSON|@PATH|-",
                   help="the Slack message event, as a watcher gets it: the post's channel, ts and thread, and its text where "
                        "the bot can't read the channel")
    p.add_argument("--thread", default=None, metavar="TS", help="its thread's root ts, when it is a reply")
    p = cmd("verify-author", "exit 0 only when the post's author is in the registry")
    p.add_argument("--event", required=True, help="the message event as JSON, @PATH or - (stdin)")
    p.add_argument("--why", action="store_true", help="print the verdict on stderr")
    p.add_argument("--authorizes", action="store_true",
                   help="exit 1 unless the author is a named human (the registry's humans): check this before acting on a "
                        "Slack request to spend, grant access or keys, delete or change a node; a collaborator may ask, "
                        "never authorize")
    p = cmd("read", "a thread, the channel since TS, or (--as) what reached NAME since it last read (or since TS)")
    p.add_argument("--thread", default=None, metavar="TS")
    p.add_argument("--since", default=None, metavar="TS", type=since_ts,
                   help="the channel since TS (a Slack ts, or an ISO time with its zone); with --thread, its replies since TS; "
                        "with --as, NAME's doorbells since TS")
    p.add_argument("--as", dest="handle", default=None, metavar="NAME", help="NAME's unread doorbells")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--peek", action="store_true", help="--as: don't mark them read")
    p.add_argument("--urgent", action="store_true", help="--as: exit 1 unless an unread doorbell is URGENT")
    p.add_argument("--json", action="store_true")
    p = cmd("ask-daniel", "an ask's page in the Asks for Daniel database and its card in #ask-daniel; `status ID` reads the card's "
            "answer, `resolve ID` records one from its page or thread, `pages` lists the pages edited lately")
    p.add_argument("verb", nargs="?", choices=("status", "resolve", "pages"),
                   help="read a card's decision, record Daniel's answer from its page or thread, or list the asks' pages edited lately")
    p.add_argument("id", nargs="?", metavar="ID", help="the card's id, as posting it printed")
    p.add_argument("--as", dest="handle", default=None, metavar="HANDLE",
                   help="the asking coordinator's handle (workers ask through theirs); pages: only this lane's")
    p.add_argument("--agent-url", default=None, help="the asking agent's link on the card (default $RESEARCH_AGENT_URL)")
    p.add_argument("--kind", choices=KINDS, default=None, help="blocking waits for Daniel; default proceeds at the deadline unless he objects")
    p.add_argument("--question", default=None, help="one line")
    p.add_argument("--why", default=None, help="why it matters")
    p.add_argument("--option", action="append", default=[], metavar="VALUE=LABEL", help=f"{OPTIONS_MIN} to {OPTIONS_MAX} of them")
    p.add_argument("--recommend", default=None, metavar="VALUE")
    p.add_argument("--default", default=None, metavar="VALUE")
    p.add_argument("--deadline", default=None, help="`3:00 PM` (today, Pacific) or an ISO time with its zone")
    p.add_argument("--page", default=None, metavar="FILE.md",
                   help=f"the ask's page as Notion-flavored Markdown, LaTeX in $`...`$ and $$ blocks, written to the template in "
                        f"{SKILL_ASKING}: created in the database (needs $NOTION_API_KEY), and the card links it")
    p.add_argument("--area", default=None, choices=AREAS, metavar="AREA", help=f"--page: one of {', '.join(AREAS)}")
    p.add_argument("--link", default=None, metavar="NOTION_URL", help="the ask's page, made with the Notion MCP, in place of --page")
    p.add_argument("--hours", type=float, default=24, metavar="H", help="pages: edited in the last H hours (default 24)")
    p.add_argument("--wait", action="store_true", help="poll until the card is decided (or overdue); exit 0 once decided")
    p.add_argument("--every", type=float, default=WAIT_EVERY_S, metavar="S", help=f"seconds between polls (default {WAIT_EVERY_S})")
    p.add_argument("--choice", default=None, metavar="VALUE", help="resolve: the option Daniel chose in the thread")
    p.add_argument("--note", default=None, help="resolve: his words, or a link to them")
    p.add_argument("--json", action="store_true", help="status: the route's whole answer; pages: each page's fields")
    p = sub.add_parser("groups", help="the handles' Slack user groups")
    gsub = p.add_subparsers(dest="gcmd", required=True)
    gsub.add_parser("sync", parents=[c], help="create or update a user group per handle; a plan unless --apply").add_argument("--apply", action="store_true")

    ns = ap.parse_args(argv)
    try:
        path = registry_path(ns.registry)
        reg = load_registry(path)
        if ns.cmd == "groups":
            return cmd_groups_sync(ns, reg, path)
        if ns.cmd in ROUTED or ns.cmd == "owners":                # who owns a path now is main's answer too
            reg = routing_registry(reg, path)
        return {"handles": cmd_handles, "owners": cmd_owners, "subscriptions": cmd_subscriptions, "ask": cmd_ask, "announce": cmd_announce, "reply": cmd_reply,
                "pickup": cmd_pickup, "roster": cmd_roster, "match": cmd_match,
                "verify-author": cmd_verify_author, "deliver": cmd_deliver, "read": cmd_read, "ask-daniel": cmd_ask_daniel, "names": cmd_inbox,
                "traffic": cmd_traffic, "send": cmd_send, "close": cmd_close,
                "watch": cmd_watch, "linear": cmd_linear, "retract": cmd_retract}[ns.cmd](ns, reg)
    except SlackError as e:
        print(f"{prog}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""`research msg`: messages between agents, on Slack (the conventions are
`.agents/skills/using-slack/SKILL.md`).

A message is a post in #agent-coordination, sent by a name: a lead's handle in the registry (@infra, @proofs, @lean, ...), which
`names open` gives an inbox thread. Only leads get names; a worker is reached through its lead. A thread is a conversation. Who a message reaches is one rule (`route_targets`): the names that
lead it (`@a @b ...`), an announcement's audience, and on a reply the thread's starter or the names it addressed (under an
announcement, only the names it starts with). Each gets a doorbell in its inbox
thread, which its agent subscribes to. The footer carries the post's kind and flags.
Notes are records, never messages.

  research msg send --as NAME [--to @A @B ...] [--thread TS] [--kind WORD] [--quiet] [--urgent] (--text T | --file PATH)
                    [--verify CMD] [--ships URL#sha256=HEX | git:SHA]...     # prints `<ts> <permalink>`; --kind announce: an announcement
  research msg ask --as NAME --to @OWNER (--text T | --file PATH) [--urgent]   # an issue in OWNER's Linear team, its tag and doorbell
  research msg ask --as NAME --to @daniel --kind K --question Q --why W --option V=L... --recommend V --default V --deadline D
                   --page FILE.md --area AREA                      # Daniel's: the ask's page and card, as `ask-daniel --page`
  research msg close --as NAME --thread TS [--link URL] [--note T] [--declined WHY]   # ✅ or ❌: a status line, and the reaction on an ask
  research msg linear check|state|comment|list|audit|place|edit ... # Linear through the app, never the MCP connector (`linear -h`)
  research msg linear relate ISSUE blocks|blocked-by|related|duplicate OTHER | unrelate ISSUE OTHER
  research msg linear update PROJECT --health onTrack|atRisk|offTrack --body-file F   # a project update
  research msg linear project create NAME --team KEY... [--lead USER] [--target D] [--description-file F]
  research msg linear milestone create|edit PROJECT NAME [--date D] [--description-file F] [--rename NEW]
  research msg read --thread TS [--since TS] | --since TS | --as NAME [--peek] [--urgent] [--json]  # --as: NAME's unread doorbells
  research msg names open|read|close|list [--as NAME] [--forward-to @NAME] [--succeeds] [--poster BC_ID]  # the names you answer to
  research msg traffic [--hours H] [--json]                        # the monitor: doorbells, pauses, dropped names, unanswered asks
  research msg traffic --latency [--hours H] [--json]              # per lane: asks, answered (by record), median and p90 minutes
  research msg pickup --thread TS                                  # 👀 on an ask you own
  research msg retract --as NAME --ts TS [--thread ROOT] [--why W] # blank a post NAME sent (needs SLACK_BOT_TOKEN)
  research msg roster --thread TS [--json]                         # an announcement's status lines; exit 0 when all are done
  research msg handles [--json] | subscriptions --as HANDLE        # the registry, holders from routes; the subscribe calls a holder makes
  research msg owners [PATH...]                                    # who owns each path (stdin: `git diff --name-only main... |`)
  research msg match --as HANDLE --event JSON|TEXT|@PATH|- [--why] # exit 0 addressed, 1 not, 2 error
  research msg verify-author --event JSON|@PATH|- [--why] [--authorizes]  # exit 0 only for our bot, a channel's bot or a named person;
                                                                   # --authorizes: only for a named human, who may authorize spend
  research msg deliver --ts TS [--thread T]                         # ring the doorbells for a post no client delivered
  research msg deliver --channel ask-the-team|chat-NAME --event JSON|@PATH|-  # a person's post there: its doorbells name the
                                                                   # channel; a chat's doorbells carry no text (`chat_targets`)
  research msg chat open NAME (--user U... | --email E) [--start-here URL] [--note T] [--cursor-bot U...]
                                                                   # a person's private #chat-NAME; the person goes to people.d
  research msg chat close NAME | chat list                         # archive it and mark it closed; each chat in people.d
  research msg chat expect NAME --email E [--note T]               # someone invited to the workspace: people.d, no channel
                                                                   # yet; the Slack events service opens it at their team_join
  research msg groups sync [--apply]                               # one user group per handle; a plan unless --apply
  research msg slack-app export|diff|apply [--app-id ID] [--manifest PATH]   # the app's manifest: live, against the repo's, update
  research msg ask-daniel --as HANDLE --kind blocking|default --question Q --why W --option VALUE=LABEL... --recommend V
                          --default V --deadline '3:00 PM'|ISO (--page FILE.md --area AREA | --link NOTION_URL) [--wait]
                          # the ask's page in the Asks for Daniel database, and a card in #ask-daniel linking it; prints `<id> <ts> <permalink>`
  research msg ask-daniel status ID [--wait] [--json] | resolve ID --choice V [--note T]
                          # --wait polls for at most 20 min: a card still pending then exits 79 with the status command (research/waitcap.py)
  research msg ask-daniel pages [--as HANDLE] [--hours H] [--json]  # asks' pages edited lately: Status, Decision, newest comment

An ask lives in one record, by its target: a handle's ask is an issue in its Linear team, and Daniel's is a page in the Asks
for Daniel database; `traffic` reads that record's state for whether, and `--latency` for when, it was answered.

`announce` and `reply` are the older spellings of `send`, kept while callers move.

#ask-daniel's cards go through the site's route (ASK_DANIEL, Cursor OIDC with the route as audience), not the Slack API. Each card
links its ask's page in the registry's Notion database (the ask-daniel channel's `pages`), where Daniel reads, comments and
decides; `--page` and `pages` reach it through Notion's API with $NOTION_API_KEY, and $RESEARCH_NOTION_URL points them at
another server (the tests' fake).

Every subcommand takes `--dry-run` (no request reaches Slack; each one it would send is printed, the token never),
`--registry PATH` (default $RESEARCH_SLACK_REGISTRY, else the checkout's) and `--channel KEY` (default agent-coordination);
a channel's id is $SLACK_CHANNEL_<KEY> (upper case, `-` as `_`) or the registry's, and `--channel-id` overrides both. The token
is $SLACK_BOT_TOKEN and is never printed; `slack-app` instead takes the app configuration token from its state file
(`app_token`), and its --dry-run reads the live app (export, validate) but changes nothing; $SLACK_API_URL points the client at another Web API (the tests' fake). Stdlib only,
and it runs as a single file (`python3 slack.py ...`) from any checkout.

Without $SLACK_BOT_TOKEN (or with RESEARCH_SLACK_VIA=broker) every call goes through Verity's Slack broker instead: a POST of
`{"method", "args"}` to BROKER with this agent's Cursor OIDC token, answered with Slack's response unchanged. The broker holds the
bot token and allows only BROKER_METHODS; RESEARCH_SLACK_VIA=direct insists on the token.

The sender rings the doorbells right after posting. A person's typed post in #ask-the-team or a chat channel is rung by the
Slack events service (`infra/slack-events/`), which imports this file and calls `deliver`; `deliver --channel KEY --event -`
does the same by hand. It routes by origin/main's registry
(`routing_registry`) and the published routes (`Routes.refresh`), so an old checkout or notes clone still rings a name added,
reopened or closed since. Who holds a handle, who else posts as it and which branch suffixes its PRs carry is its route's
(`names open`), never the registry's. `match` is the older, handle-only test of whether a post is addressed to HANDLE."""
from __future__ import annotations

import argparse
import base64
import fcntl
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
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

try:
    from . import waitcap
    from .timefmt import both, slack_dates
except ImportError:                                                 # run as a single file: `python3 slack.py ...`
    def both(t: Any) -> str:
        return f"{datetime.fromtimestamp(float(t), UTC):%H:%MZ}" if re.fullmatch(r"\d+(?:\.\d+)?", str(t)) else str(t)

    def slack_dates(text: str, now: Any = None) -> str:
        return text

    class waitcap:  # research/waitcap.py's cap, which a single file carries itself
        WAIT_CAP_S, STILL_GOING = 1200, 79

        @staticmethod
        def still_going(what: str, check: str, waited_s: float, end: str) -> str:
            return (f"research: {what} is still going; stopped waiting after {waited_s:g} s (no tool call may block more than 20 "
                    f"minutes). Nothing was cancelled. Check back with `{check}`. {end}")

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
#: `chat`: a collaborator's channel, its invites, pin and bookmark, which the broker doesn't relay
CHAT_METHODS = ("bookmarks.add", "conversations.archive", "conversations.create", "conversations.info", "conversations.invite",
                "conversations.inviteShared", "conversations.list", "conversations.unarchive", "pins.add", "users.list",
                "users.lookupByEmail")
DIRECT_ONLY = ("chat.update", "usergroups.create", "usergroups.disable", "usergroups.enable", "usergroups.update",
               *CHAT_METHODS)                             # `retract`, `groups sync --apply`, `chat`: the token itself
BROKER_MAX_BODY = 65536                                                   # bytes of request JSON; the route refuses more (413)
ERROR_CODE_RE = re.compile(r"^[a-z0-9_]{1,64}$")
SCOPES_RE = re.compile(r"^[a-z0-9_.:]{1,64}(,[a-z0-9_.:]{1,64}){0,15}$")      # a not-ok answer's `needed`: chat:write,users:read
POINTER_RE = re.compile(r"^/[A-Za-z0-9_/.:-]{0,120}$")                         # a JSON pointer into a manifest: /oauth_config/scopes/bot/3
EVERY = "every handle"
DESCRIPTION_MAX = 140                                            # Slack's user-group description limit (description_too_long)
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
SUBTEAM_RE = re.compile(r"<!subteam\^([A-Z0-9]+)(?:\|@?([^>]*))?>")
BARE_GROUP_RE = re.compile(r"@(S[0-9A-Z]{8,})\b")
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
TRANSIENT_RETRIES, TRANSIENT_WAIT_S = 3, 2.0                  # a read cut short, reset, timed out or a 5xx: 2, 4, then 8 s
READ_VERBS = ("get", "list", "history", "replies", "info", "members", "lookup", "test")
ASK_DANIEL = "https://website-docs-sage.vercel.app/api/agent-approvals"   # the cards' route, and its OIDC token's audience
ASK_CHANNEL = "ask-daniel"
DANIEL = "daniel"                                                # whose asks are pages: `traffic`'s lane for the cards
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
PAGE_SECTIONS = ("## The problem", "## Background", "## The design the rules want", "## Options", "## What happens next",
                 "## Sources")
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
WORKER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}")      # `linear comment --worker NAME`: shows as LANE/NAME
QUIET = "(quiet)"                                                # ending a post: ring only the names it starts with
CAP_PER_HOUR = 20                                                # doorbells into one inbox from one author in one thread
DOORBELL_MAX = 600                                               # characters of the post a doorbell carries
POST_MAX = 3000                                                  # bytes: Slack splits a longer post into separate top-level messages
INBOX_HEAD = "Inbox for @"
PR_RE = re.compile(r"(?<![\w&/#])#(\d{3,6})\b")
BRANCH_RE = re.compile(r"\bcursor/([a-z0-9._-]*[a-z0-9])")
TRAIN_PREP_RE = re.compile(r"^train-prep-((?:\d+-)+)")                  # `cursor/train-prep-1585-1586-on-041c30713-4292`
RUN_ID_RE = re.compile(r"\br\d{8}-\d{6}-([0-9a-f]{4})\b|(?<![\w./#@-])([0-9a-f]{4})(?![\w/@-])")   # a run, or its last 4
HISTORY_S = 86400                                                # how far back `traffic --volume` reads a thread before its window
TAG_SEP = " \u00b7 "
_TAG = r"(?:[a-z0-9][a-z0-9._-]*|\([a-z]+\))"
FOOTER_RE = re.compile(rf"\n_({_TAG}(?: \u00b7 {_TAG})*)_\s*$")                                  # `_ask · urgent_`
LINK_RE = re.compile(r"<((?:https?|mailto):[^|>\s]+)(?:\|([^>]*))?>")
FLAGS = ("urgent", QUIET)
SLACK_USER_RE = re.compile(r"^[UW][A-Z0-9]{2,}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
COLLABORATOR_ROLE = "collaborator: may ask, never authorizes spend, access or destructive changes"
COLLABORATOR_MAY = "ask a lane for something and request work within its budgets"
COLLABORATOR_NEVER = "spend, access or keys, destructive changes or node changes"
CHAT_PREFIX = "chat-"                                            # a collaborator's private channel, `chat open NAME`
CHAT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,74}$")          # Slack's channel names: 80 characters, lower case, no spaces
CHAT_WATCHER = "top"                                             # rung for a chat post routed to no one
PEOPLE_DIR = "people.d"                                          # in the notes, beside routes.d: one `<NAME>.json` per chat's person
PERSON_SCHEMA = "verity/slack-person/v1"
EXPECTED = "expected"                                            # a people.d record's status before its person joins Slack
NOTES_API = "https://api.github.com/repos/compute-verification/research-notes/contents/"   # where no clone is (the service)


class SlackError(RuntimeError):
    code: str | None = None                                       # a not-ok answer's error code, when it is one


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


GROUP_ID_RE = re.compile(r"^S[0-9A-Z]{8,}$")


def group_ids(h: dict) -> set[str]:
    """A handle's user-group ids: its current group and the ones it retired (a retired group still names it in old posts)."""
    return ({h["usergroup"]} if h.get("usergroup") else set()) | set(h.get("retired_usergroups", []))


def origin_groups(path: Path) -> set[str]:
    """The user-group ids in origin/main's registry: an analysis checkout older than a sender's must not report the
    sender's group mentions as dropped. Empty when git can't answer, which keeps
    `traffic`'s old behavior."""
    try:
        return {g for h in (origin_registry(path) or {}).get("handles", []) for g in group_ids(h)}
    except (AttributeError, KeyError, TypeError):
        return set()


FRESH_EVERY_S = 1800                                             # a routed post fetches origin/main at most this often
ROUTED = ("send", "ask", "announce", "reply", "close", "deliver", "linear", "ring-run")   # the subcommands that ring doorbells


def routing_registry(reg: dict, path: Path, now: float | None = None) -> dict:
    """The registry a post routes by: origin/main's handles, then those only this checkout has (a branch adding one).  A
    sender on an old checkout otherwise rings no one for a handle added since.
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
    return with_people({**reg, **up, "collaborators": up.get("collaborators", []),
                        "handles": [*up["handles"], *(h for h in reg["handles"] if h["handle"] not in names)]},
                       reg.get("chat_people", []))


def load_registry(path: Path, people: bool = True) -> dict:
    """The registry at `path`, checked; with `people`, also each open chat's person from the notes clone's people.d
    (`with_people`), which a command that writes the registry back leaves out."""
    try:
        reg = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SlackError(f"registry {path}: {e}") from e
    reg = check_registry(reg, str(path))
    return with_people(reg, ChatPeople().open()) if people else reg


EXAMPLES_MAX, EXAMPLE_CHARS = 5, 160                   # a handle's `examples`: the questions a collaborator might ask it


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
        ex = h.get("examples", [])
        if not isinstance(ex, list) or len(ex) > EXAMPLES_MAX or not all(isinstance(q, str) and 0 < len(q) <= EXAMPLE_CHARS for q in ex):
            raise SlackError(f"registry {path}: @{name}'s examples: at most {EXAMPLES_MAX} questions, each one line of at most "
                             f"{EXAMPLE_CHARS} characters, in a newcomer's words (what the chat router matches a post against)")
        lineage = h.get("lineage", [])
        if not isinstance(lineage, list) or not all(isinstance(a, str) and AGENT_ID_RE.fullmatch(a) for a in lineage):
            raise SlackError(f"registry {path}: @{name}'s lineage: the agent ids that held it before (bc-<8 hex> or its full "
                             "form), so a handover's predecessor stays known without a second id in `holder`")
        old = h.get("retired_usergroups", [])
        current = {o.get("usergroup") for o in reg["handles"]}
        if not isinstance(old, list) or any(not isinstance(g, str) or not GROUP_ID_RE.match(g) or g in current
                                            or any(g in o.get("retired_usergroups", []) for o in reg["handles"] if o is not h) for g in old):
            raise SlackError(f"registry {path}: @{name}'s retired_usergroups: Slack group ids (S...) that no handle has now and "
                             f"no other handle retired; mentions of them still route to @{name}, and groups sync never re-enables them")
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
    ext = reg.get("external_posters", [])
    if not isinstance(ext, list) or any(not isinstance(u, str) or not EXTERNAL_POSTER_RE.match(u) or u in handles
                                        or u == "router" for u in ext):
        raise SlackError(f"registry {path}: external_posters: the usernames under which something other than `research msg` "
                         "posts through Verity's bot (Daniel's desktop agent, Codex), so the Slack events service delivers "
                         "their posts; never a handle's or the router's")
    return reg


EXTERNAL_POSTER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,39}$")   # lower-cased, it's a doorbell's author


def external_post(msg: dict | None, reg: dict) -> bool:
    """Posted through Verity's bot by something other than `research msg` (`external_posters`), so no sender delivered it:
    the Slack events service does."""
    return ours(msg, reg) and (msg or {}).get("username") in reg.get("external_posters", [])


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


def chat_channels(reg: dict) -> dict[str, dict]:
    """Each open chat (`chat open`, people.d) as a registry channel: its id and its `person`.  No handle subscribes: the Slack
    events service hands each post to `deliver`, which rings the lanes it names, else CHAT_WATCHER."""
    return {r["channel"]["key"]: {"id": r["channel"]["id"], "person": r, "subscribers": [],
                                  "purpose": f"{r['name']}'s questions to the lanes (private)"}
            for r in reg.get("chat_people", [])}


def _person_ok(r: Any, name: str) -> bool:
    c = r.get("channel") if isinstance(r, dict) else None
    if not (isinstance(r, dict) and r.get("schema") == PERSON_SCHEMA and isinstance(r.get("name"), str) and bool(r["name"].strip())
            and isinstance(c, dict) and c.get("key") == CHAT_PREFIX + name):
        return False
    if is_expected(r):                                            # `chat expect`: no channel until they join
        return c.get("id") is None and isinstance(r.get("email"), str) and bool(EMAIL_RE.match(r["email"])) \
            and isinstance(r.get("expected"), str)
    return isinstance(c.get("id"), str) and bool(CHANNEL_ID_RE.match(c["id"])) and isinstance(r.get("opened"), str)


def is_expected(r: dict | None) -> bool:
    """A person `chat expect` published: invited to the workspace, their chat opened when they join (`team_join`)."""
    return bool(r) and r.get("status") == EXPECTED and not r.get("closed")


class ChatPeople:
    """people.d/<NAME>.json in the notes: the person of #chat-NAME (`chat open` writes it, `chat close` marks it closed).  The
    clone's, and for a name the clone lacks, or when asked (`fresh`), the published copy, read beside the routes' (the
    sibling `people.d/` of $RESEARCH_ROUTES_URL); of a name's two copies the newer counts (`_newer`), as for routes."""

    def __init__(self, root: Path | None = None, base: str | None = None, get: Callable[[str], bytes | None] | None = None):
        self.root, self.get = (root or notes_root()), get or http_get
        routes = (base or os.environ.get("RESEARCH_ROUTES_URL") or ROUTES_RAW).rstrip("/")
        self.base = routes.rsplit("/", 1)[0] + f"/{PEOPLE_DIR}/"
        self.local: dict[str, dict] = {}
        d = self.root / PEOPLE_DIR
        for p in sorted(d.glob("*.json")) if d.is_dir() else []:
            try:
                r = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if _person_ok(r, p.stem):
                self.local[p.stem] = r
        self._fetched: dict[str, dict | None] = {}

    def get_person(self, name: str, fresh: bool = False) -> dict | None:
        if not CHAT_NAME_RE.match(name):
            return None
        if fresh or name not in self.local:
            raw = self.get(f"{self.base}{name}.json")
            try:
                r = json.loads(raw) if raw else None
            except ValueError:
                r = None
            self._fetched[name] = r if _person_ok(r, name) else None
        return _newer(self.local.get(name), self._fetched.get(name))

    def all(self) -> dict[str, dict]:
        return {n: r for n in sorted({**self.local, **self._fetched}) if (r := _newer(self.local.get(n), self._fetched.get(n)))}

    def open(self) -> list[dict]:
        return [r for r in self.all().values() if not r.get("closed") and not is_expected(r)]

    def published(self) -> list[str]:
        """Every name people.d has, the clone's and the published ones: a `file://` directory's files, or the notes repo's
        through GitHub's contents API (`notes_api`; the raw host lists nothing).  Another base lists the clone's only."""
        names = set(self.local)
        if self.base.startswith("file://"):
            d = Path(urllib.parse.unquote(self.base[7:]))
            names |= {p.stem for p in d.glob("*.json")} if d.is_dir() else set()
        elif self.base == ROUTES_RAW.rstrip("/").rsplit("/", 1)[0] + f"/{PEOPLE_DIR}/":
            names |= {f["name"][:-5] for f in notes_api("GET", PEOPLE_DIR) or []
                      if isinstance(f, dict) and str(f.get("name", "")).endswith(".json")}
        return sorted(n for n in names if CHAT_NAME_RE.match(n))

    def expected(self) -> list[dict]:
        """The people `chat expect` published and who have no chat yet, each read as published now."""
        for n in self.published():
            self.get_person(n, fresh=True)
        return [r for r in self.all().values() if is_expected(r)]


def with_people(reg: dict, persons: list[dict]) -> dict:
    """`reg` with each open chat's person: `chat_people`, and in `collaborators` each who is not a named person already
    (a named human, or someone added by hand, by Slack id) and whose entry keeps the registry valid."""
    chats = {r["channel"]["key"]: r for r in persons if not r.get("closed") and not is_expected(r)}
    collabs = list(reg.get("collaborators", []))
    handles = {h["handle"] for h in reg["handles"]}
    for key, r in chats.items():
        p = {"name": r["name"], "slack_user": r.get("slack_user"), "note": r.get("note") or f"asks the lanes in #{key}",
             **({"email": r["email"]} if r.get("email") else {})}
        if p["slack_user"] and any(q.get("slack_user") == p["slack_user"] for q, _ in people({**reg, "collaborators": collabs})):
            continue
        try:
            check_collaborators({**reg, "collaborators": [*collabs, p]}, PEOPLE_DIR, handles)
        except SlackError:
            continue                                                  # a name or id someone else has: the chat stays, unrouted
        collabs.append(p)
    return {**reg, "collaborators": collabs, "chat_people": list(chats.values())}


def channels(reg: dict) -> dict[str, dict]:
    """The registry's channels and its collaborators' chat channels."""
    return {**reg["channels"], **chat_channels(reg)}


def subscribers(reg: dict, key: str) -> list[str]:
    subs = channels(reg)[key]["subscribers"]
    return [h["handle"] for h in reg["handles"]] if subs == EVERY else list(subs)


def _env(key: str) -> str:
    return "SLACK_CHANNEL_" + key.upper().replace("-", "_").replace(".", "_")


def resolve_key(reg: dict, key: str) -> str:
    """A channel's registry key, from its key or an old name it keeps as an alias (`approvals` for `ask-daniel`)."""
    return next((k for k, c in reg["channels"].items() if key in c.get("aliases", [])), key)


def channel_key(ns: argparse.Namespace, reg: dict) -> str:
    key = resolve_key(reg, (ns.channel or FRONT_DOOR).lstrip("#"))
    if key not in channels(reg) and key.startswith(CHAT_PREFIX):     # a chat the clone lacks: its published person
        try:
            r = ChatPeople().get_person(key[len(CHAT_PREFIX):])
        except SlackError:
            r = None
        if r and not r.get("closed") and not is_expected(r):
            reg.update(with_people(reg, [*reg.get("chat_people", []), r]))
    if key not in channels(reg):
        raise SlackError(f"no channel #{key} in the registry (it has {', '.join(sorted(channels(reg)))})")
    return key


def channel_id(ns: argparse.Namespace, reg: dict, key: str | None = None) -> str:
    key = key or channel_key(ns, reg)
    cid = (ns.channel_id if key == channel_key(ns, reg) else None) or os.environ.get(_env(key)) or channels(reg)[key].get("id")
    if not cid:
        raise SlackError(f"no channel id for #{key}: pass --channel-id, set {_env(key)}, or record its id in the registry")
    return cid


def key_of(reg: dict, cid: str) -> str | None:
    return next((k for k, c in channels(reg).items() if cid in (c.get("id"), os.environ.get(_env(k)))), None)


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


def transient(e: BaseException) -> bool:
    """A failure that a second identical request may not meet: a 5xx, a response cut short (IncompleteRead, a dropped
    connection), a reset or a timeout. A refusal (4xx), a DNS failure or an unparseable answer is not."""
    if isinstance(e, urllib.error.HTTPError):
        return e.code >= 500
    if isinstance(e, http.client.HTTPException):
        return True
    if isinstance(e, urllib.error.URLError):
        e = e.reason if isinstance(e.reason, BaseException) else e
    return isinstance(e, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError, http.client.HTTPException))


def idempotent(method: str) -> bool:
    """A read (`conversations.replies`, `users.list`, `chat.getPermalink`), which a retry can't repeat. A write cut short or timed
    out may already have been taken (a post would appear twice), so only reads are retried after a transient failure."""
    return method.rsplit(".", 1)[-1].startswith(READ_VERBS)


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

    def __init__(self, token: str | None, *, base: str | None = None, dry_run: bool = False, log=None, transport=None,
                 direct: bool = False):
        self.via = "direct" if direct else via(token)               # direct: the app configuration API, which the broker can't reach
        self.token = token if self.via == "direct" else None
        self.dry_run, self.log, self.transport, self._oidc = dry_run, log, transport or post, _Identity(BROKER)
        self.base = (base or os.environ.get("SLACK_API_URL") or DEFAULT_API).rstrip("/") + "/"
        if not dry_run and self.via == "direct" and not token and not direct:
            raise SlackError("SLACK_BOT_TOKEN is not set (a Cursor secret; see the using-slack skill), or pass --dry-run")

    def call(self, method: str, **params: Any) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        if self.dry_run:
            print(f"dry-run: {method} {json.dumps(params, sort_keys=True)}" + (" (via the broker)" if self.via == "broker" else ""),
                  file=self.log or sys.stderr)
            return {"ok": True, "dry_run": True}
        args = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in params.items()}
        rate = flaky = 0
        while True:
            try:
                out = self._broker(method, args) if self.via == "broker" else self.transport(
                    self.base + method, urllib.parse.urlencode(args).encode(),
                    {**({"Authorization": f"Bearer {self.token}"} if self.token else {}),
                     "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
                break
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code == 429 and rate < RATE_RETRIES:   # a burst of doorbells
                    rate += 1
                    _sleep(min(float(e.headers.get("Retry-After") or 1), RATE_WAIT_MAX_S))
                    continue
                if transient(e) and idempotent(method) and flaky < TRANSIENT_RETRIES:
                    flaky += 1
                    _sleep(TRANSIENT_WAIT_S * 2 ** (flaky - 1))
                    continue
                if isinstance(e, urllib.error.HTTPError):
                    if self.via == "broker":
                        raise SlackError(broker_error(method, e)) from None
                    hint = f" (rate limited; retry after {e.headers.get('Retry-After')} s)" if e.code == 429 else ""
                    raise SlackError(f"{method}: HTTP {e.code}{hint}") from None
                where = "the Slack broker is unreachable" if self.via == "broker" else type(e).__name__
                tries = f" (after {flaky + 1} tries)" if flaky else ""
                raise SlackError(f"{method}: {where}: {getattr(e, 'reason', e)}{tries}") from None
        if not isinstance(out, dict):
            raise SlackError(f"{method}: the answer is not a JSON object")
        if not out.get("ok"):
            error, needed = out.get("error"), out.get("needed")     # echoed only when they are codes: an answer is not to be printed
            code = error if isinstance(error, str) and ERROR_CODE_RE.fullmatch(error) else None
            at = [e["pointer"] for e in out.get("errors") or [] if isinstance(e, dict) and isinstance(e.get("pointer"), str)
                  and POINTER_RE.fullmatch(e["pointer"])][:10]                   # apps.manifest.validate: where the manifest is wrong
            err = SlackError(f"{method}: {code or 'not ok'}"
                             + (f" (needed scope {needed})" if isinstance(needed, str) and SCOPES_RE.fullmatch(needed) else "")
                             + (f" at {', '.join(at)}" if at else ""))
            err.code = code
            raise err
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
        drops the message at each page boundary."""
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
    """The command's client: `ns.api` where a caller brings its own (the Slack events service), else from the environment."""
    return getattr(ns, "api", None) or Slack(os.environ.get("SLACK_BOT_TOKEN"), dry_run=ns.dry_run)


def _footer(tags: tuple[str, ...] = ()) -> str:
    """`_ask · urgent_`: the post's kind and flags, which `list` and `traffic` read.  The author is the post's username, and its holder is in the directory."""
    tags = tuple(t for t in tags if t)
    return f"\n\n_{TAG_SEP.join(tags)}_" if tags else ""


HEADER_TIME_RE = re.compile(r"^\W{0,4}((?:(?:<[^>\s]+>|[@*_\w().,-]+) ){0,4}?)\*?([01]?\d|2[0-3]):([0-5]\d)Z\*?(?=[:,)]| \(| [-\u2014])")
DEADLINE_WORDS = {"by", "until", "till", "before", "after", "at", "from", "eta", "since", "to"}
STAMP_SLACK_S = 120                                              # a hand-typed time this close to the send time is left alone


def stamp_time(text: str, now: datetime) -> str:
    """A post whose first line gives a UTC time (`compute-research 13:07Z: ...`) more than STAMP_SLACK_S from when it is
    sent gets the real time beside it, `13:07Z (sent 13:22Z)`: hand-typed times run minutes off and misorder posts.  The
    typed time stays, since it may name a checkpoint rather than the moment."""
    m = HEADER_TIME_RE.match(text or "")
    if not m or any(w.strip("*_,()").lower() in DEADLINE_WORDS for w in m[1].split()):
        return text                                              # no header time, or a deadline (`... by 20:00Z:`)
    typed = int(m[2]) * 60 + int(m[3])
    real = now.astimezone(UTC)
    off = abs(typed - (real.hour * 60 + real.minute)) % (24 * 60)
    if min(off, 24 * 60 - off) * 60 <= STAMP_SLACK_S:
        return text
    return f"{text[:m.end()]} (sent {real:%H:%M}Z){text[m.end():]}"


def plain(text: str, reg: dict | None = None) -> str:
    """A post as an agent should read it: no footer, Slack's markup undone (names, dates, links, escapes).  With `reg`, a
    user group shows as its handle, a bare id copied into a doorbell included."""
    names = {g: h["handle"] for h in (reg or {}).get("handles", []) for g in group_ids(h)}
    text = FOOTER_RE.sub("", text or "")
    text = SUBTEAM_RE.sub(lambda m: "@" + (m[2] or names.get(m[1]) or m[1]).lstrip("@"), text)
    if names:
        text = BARE_GROUP_RE.sub(lambda m: "@" + names.get(m[1], m[1]), text)
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
    head, *rest = _pieces(slack_dates(stamp_time(text, _now())))
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
            deliver(api, route[0], route[1], ch, msg, sender=self_agent_id)
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
        if gid in group_ids(h):
            return f"user-group mention <!subteam^{gid}>"
        if label.lstrip("@") == h["handle"] and not any(gid in group_ids(o) for o in reg["handles"]):
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
    """Who wrote `msg`, as a name: our bot's username (an `external_posters` one in lower case, as a doorbell's author), or
    `daniel`-style for a named human or a collaborator (first name, lower case)."""
    if not msg:
        return None
    if msg.get("bot_id"):
        if not ours(msg, reg):
            return None
        name = msg.get("username") or None
        return name.lower() if name and external_post(msg, reg) else name
    return person_name(who[0]) if (who := person(msg, reg)) else None


def lead_names(text: str, reg: dict) -> list[str]:
    """The names a post addresses: the `@name`s and user-group mentions at its very start (after one emoji at most), before
    any other word, or a handle written bare as the first word with a colon (`lean: ...`)."""
    m = LEAD_RE.match(text or "")
    if not m and (bare := BARE_LEAD_RE.match(text or "")) and any(h["handle"] == bare[1] for h in reg["handles"]):
        return [bare[1]]
    names: list[str] = []
    for at, gid, label in NAME_TOKEN_RE.findall(m.group(1) if m else ""):
        known = next((h["handle"] for h in reg["handles"] if gid and gid in group_ids(h)), "")
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


def warn_unrouted(body: str, reg: dict, author: str, rung: tuple[str, ...] = ()) -> None:
    """Say which names a post mentions past its start, where nothing rings.  It posts anyway: refusing costs re-sends, and the
    sender learns here that those names weren't woken."""
    if names := unrouted_names(body, reg, author, rung):
        print(f"note: it names {' '.join('@' + n for n in names)} past its start, so they aren't rung: only the names a post "
              "starts with wake anyone. To ring them, move them to the start (--to, or the leading @names)", file=sys.stderr)


def header_names(text: str, reg: dict) -> list[str]:
    """The names an announcement's header (`📣 *Announcement* from @x to @a @b`) addresses; empty for `to every handle`."""
    first = (text or "").split("\n", 1)[0]
    return lead_names(first.split(" to ", 1)[1], reg) if " to " in first else []


def with_routes(reg: dict, routes: "Routes | None") -> dict:
    """`reg` with each handle's roster from its open inbox route: `holder` names the route's holder and `posters` (the
    other agents that post as it), and `branch_suffixes` are the route's.  The route is the record of both (`names open`
    writes it); the registry's copies are a fallback for readers without the routes, kept
    before the route's.  Every handle gets a `holder`, empty when none is known.  Best effort: a route
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


def route_targets(msg: dict, root: dict | None, reg: dict, inboxes: set[str] | frozenset = frozenset(),
                  author_rings: bool = False) -> list[str]:
    """The names a post in #agent-coordination wakes, in order.  `msg` and `root` (its thread's root, None for a top-level
    post) are Slack messages; `inboxes` is every inbox thread's ts.  Nothing in an inbox thread routes; a status line wakes
    only the root's poster; otherwise the names at the start, an announcement's audience (every handle when it names
    none), and on a reply that names no one the root's poster or, when the poster replies, the names the root addressed.
    A quiet post wakes only the names at its start.  A reply under an announcement wakes only the names it starts with,
    whatever PRs and runs its body names, and of its status lines only a decline wakes the announcer (`roster` reads the
    rest): otherwise the replies to an announcement to every handle are thousands of doorbells a week.  The author
    is never a target, unless
    `author_rings` (`rings_own_inbox`: another agent posting as the author's handle, such as the outgoing generation of a
    handover), when a post naming the author's handle is for it like a post naming anyone else."""
    text, ts, thread = msg.get("text") or "", msg.get("ts"), msg.get("thread_ts")
    reply = bool(thread) and thread != ts
    if (thread or ts) in inboxes or text.startswith(INBOX_HEAD) or \
            (reply and root and root.get("bot_id") and (root.get("text") or "").startswith(INBOX_HEAD)):
        return []
    author, by = poster(msg, reg), poster(root, reg) if reply else None
    if author is None:
        return []                                                    # only our bot and the named people route
    me = set() if author_rings else {author}
    if is_quiet(text):
        out = [] if is_status(text) else lead_names(text, reg)       # people write (quiet) for "no reply needed"
    elif reply and is_announcement(root):                            # its replies ring only the names they start with
        status = STATUS_RE.match(text)
        out = [by] if status and status[1].endswith("declined") else [] if status else lead_names(text, reg)
    elif reply and is_status(text):
        out = [by]
    else:
        out = lead_names(text, reg)
        if not reply and is_announcement(msg):
            out += [h["handle"] for h in reg["handles"]] if to_every_handle(text) else header_names(text, reg)
        if reply and set(out) - me:
            pass                                                     # a reply that names someone else is for them alone
        elif reply and by and by != author:
            out.append(by)
        elif reply and by:
            rt = (root or {}).get("text") or ""
            out += lead_names(rt, reg)
    seen: list[str] = []
    for n in out:
        if n and n not in me and n not in seen:
            seen.append(n)
    return seen


def chat_targets(msg: dict, thread: list[dict] | tuple, reg: dict) -> list[str]:
    """The names a post in a chat channel wakes, until `research msg router` routes them.  `thread` is the post's thread so
    far (empty for a top-level post).  The names it starts with; else, on a person's reply, the handles that posted in its
    thread; else, on a person's post, CHAT_WATCHER, who routes it by hand.  A bot's post (a lane handing off in the
    thread) and a quiet one wake only the names they start with.  The author is never a target."""
    author, text = poster(msg, reg), msg.get("text") or ""
    if author is None:
        return []
    names = [n for n in lead_names(text, reg) if n != author]
    if names or msg.get("bot_id") or is_quiet(text):
        return names
    handles = {h["handle"] for h in reg["handles"]}
    lanes = [m["username"] for m in thread if m.get("ts") != msg.get("ts") and ours(m, reg) and m.get("username") in handles]
    return list(dict.fromkeys(lanes)) or [CHAT_WATCHER]


def is_quiet(text: str) -> bool:
    """Whether a post carries the quiet marker: as a footer tag, or as `(quiet)` ending its text before any footer.
    `(quiet)` elsewhere is prose."""
    text = text or ""
    body = FOOTER_RE.sub("", text).rstrip()
    return QUIET in tags_of(text) or body.endswith(QUIET)


def tags_of(text: str) -> list[str]:
    """The tags a post's footer carries: `_ask · urgent_`."""
    m = FOOTER_RE.search(text or "")
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
    wins (a reopen or a succession stamps it anew); of one opening, a closed copy wins (a close comes after its open);
    otherwise the clone's, with the published copy's
    `branch_suffixes` and `posters` too (raw.githubusercontent serves it minutes stale, and a holder elsewhere adds them).
    A person `chat expect` published loses to a copy that opened or closed since (their chat opened when they joined), and
    of two such copies the later `expected` wins."""
    if not local or not pub:
        return local or pub
    if local.get("status") == EXPECTED or pub.get("status") == EXPECTED:
        last = lambda r: (max(str(r.get(k) or "") for k in ("expected", "opened", "closed")), r.get("status") != EXPECTED)  # noqa: E731
        return pub if last(pub) > last(local) else local
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
            from concurrent.futures import ThreadPoolExecutor
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


def same_agent(a: str, b: str) -> bool:
    """Two agent ids name one agent: equal, or one the short form (`bc-8ece7cde`) of the other."""
    short, full = sorted((a, b), key=len)
    return bool(short) and full.startswith(short)


def rings_own_inbox(routes: Routes | None, handle: str, sender: str | None) -> bool:
    """Whether a post as `handle` by the agent `sender` may ring `handle`'s own inbox: only when its open route's holder is
    another agent.  So the outgoing generation of a handover (`names open --succeeds`, which keeps it as a poster) or any
    other poster reaches the holder, and the holder is never woken by its own posts.  No sender, no route: never."""
    r = routes.current(handle) if routes is not None and sender else None
    return bool(r and not r.get("closed") and r.get("holder")) and not same_agent(r["holder"], sender or "")


_CHANNEL = r"#[A-Za-z0-9][A-Za-z0-9._-]*"
#: groups: author, urgent, thread, reply; a post outside the front door names its channel before `thread` (`doorbell_channel`)
DOORBELL_RE = re.compile(rf"^from ([a-z0-9][a-z0-9._-]*)( \(URGENT\))?, (?:{_CHANNEL} )?thread (\d+\.\d+)(?:, reply (\d+\.\d+))?: ")
DOORBELL_CHANNEL_RE = re.compile(rf"^from [a-z0-9][a-z0-9._-]*(?: \(URGENT\))?, ({_CHANNEL}) thread \d")
CHANNEL_ID_RE = re.compile(r"^[CDG][A-Z0-9]{2,}$")


def _channel_flag(channel: str) -> str:
    return f"--channel-id {channel}" if CHANNEL_ID_RE.match(channel) else f"--channel {channel}"


def doorbell(author: str, thread: str, ts: str, text: str, urgent: bool = False, channel: str | None = None,
             private: bool = False, reminder: str | None = None) -> str:
    """The post itself, so its reader can act without another read: `from infra, thread T[, reply R]: <text>`.  A post in a
    channel other than the front door names it (its registry key, else its id), so its reader answers there: `from daniel,
    #ask-the-team thread T: <text>`.  A `private` post's (a chat's) carries no text, since anyone in the front door reads
    the inboxes, only where to read it.  A `reminder` (the Slack events service's watchdog) leads the body with
    `reminder (WHY): `.  Never starts with `@`, so it can't route itself."""
    body = plain(text)
    if private:
        body = f"(a post in a private channel: research msg read {_channel_flag(channel or '?')} --thread {thread})"
    elif len(body) > DOORBELL_MAX:
        read = f"research msg read {_channel_flag(channel)} --thread {thread}" if channel else f"research msg read --thread {thread}"
        body = body[:DOORBELL_MAX].rstrip() + f"\u2026 (the rest: `{read}`)"
    where = (f"#{channel} " if channel else "") + f"thread {thread}" + (f", reply {ts}" if ts and ts != thread else "")
    return f"from {author}{' (URGENT)' if urgent else ''}, {where}: " + (f"reminder ({reminder}): " if reminder else "") + body


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


UNREADABLE = ("missing_scope", "not_in_channel")           # a token that may post in a channel but not read it


def _unreadable(e: SlackError) -> bool:
    return any(w in str(e) for w in UNREADABLE)


def deliver(api: Slack, reg: dict, routes: Routes, ch: str, msg: dict, *, urgent: bool = False,
            now: float | None = None, log=None, names: list[str] | None = None, reminder: str | None = None,
            sender: Callable[[], str | None] | None = None) -> list[str]:
    """Deliver the post `msg`: ring the doorbell of each name it wakes.  One line per name saying
    what happened.  Every handle's published route and every name's the post starts with is read first, so a clone not
    pulled since a name opened, reopened, closed or changed holder rings what the published table says (`_newer`).  A post
    in another channel (#ask-the-team, a chat channel) routes the same way; its doorbells name the channel (`doorbell`), and
    where the bot can't read that channel (UNREADABLE) a reply's thread root counts as unknown.  A post in a chat channel
    rings by `chat_targets`, and its doorbells, like those of a channel the registry doesn't know, carry no text.
    `names`, when given, are rung in place of the rule's: the Slack events service's decision for a post that named no one.
    A `reminder` (its watchdog's) rings again a name this post already rang, once per reason: only a doorbell for this post
    carrying the same reminder keeps it from ringing.  `sender` tells the agent that posted `msg` (`_post` passes
    `self_agent_id`), asked only for a post that would ring its author's own inbox, which it does when another agent
    holds it (`rings_own_inbox`)."""
    log = log or sys.stderr
    front = channel_id_for(reg, FRONT_DOOR) or ch
    where = None if ch == front else (key_of(reg, ch) or ch)
    chat = where in chat_channels(reg)
    private = where is not None and (chat or where not in channels(reg))
    thread = msg.get("thread_ts") if msg.get("thread_ts") not in (None, msg.get("ts")) else None
    notes: list[str] = []
    replies: list[dict] = []
    try:
        if thread and chat:
            replies = api.paged("conversations.replies", "messages", channel=ch, ts=thread)
            root = next((m for m in replies if m.get("ts") == thread), None)
        else:
            root = api.root(ch, thread) if thread else None
    except SlackError as e:                                          # posting still rings the names it starts with
        if not _unreadable(e):
            raise
        root = None
        notes.append(f"thread {thread} unreadable ({e}): only the names the post starts with ring")
    routes.refresh([*lead_names(msg.get("text", ""), reg), *(h["handle"] for h in reg["handles"])])
    reg = with_routes(reg, routes)
    inboxes = routes.inboxes()
    now = _now().timestamp() if now is None else now
    ch = front                                                       # from here on, `ch` is the inboxes' channel
    author = poster(msg, reg) or "?"
    urgent = urgent or "urgent" in tags_of(msg.get("text", ""))
    src, lines, missed = thread or msg.get("ts") or "", notes, []
    if msg.get("bot_id") and not ours(msg, reg):
        lines.append("no doorbells: " + ("another app's bot posted it, not Verity's" if reg.get("bot_id") else
                                         "the registry records no bot_id (`research msg groups sync --apply` writes it)"))
    humans = people_names(reg)
    own = False
    if names is not None:
        targets = names
    elif chat:
        targets = chat_targets(msg, replies, reg)
    else:
        targets = route_targets(msg, root, reg, inboxes, author_rings=True)
        own = author in targets and sender is not None and ours(msg, reg) and rings_own_inbox(routes, author, sender())
        targets = targets if own else route_targets(msg, root, reg, inboxes)
    for name in targets:
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
        if to == author and not own:
            continue
        try:
            try:
                rung = api.paged("conversations.replies", "messages", channel=ch, ts=r["inbox_ts"], oldest=f"{now - 3600:.6f}")
            except SlackError as e:                                  # unread, it rings unchecked: a repeat may ring twice
                if not _unreadable(e):
                    raise
                rung = []
            rung = [m for m in rung if m.get("ts") != r["inbox_ts"] and is_doorbell(m, reg)]
            if any(_rang(m.get("text") or "", msg.get("ts") or "", where)
                   and (not reminder or f": reminder ({reminder}): " in (m.get("text") or "")) for m in rung):
                continue
            n = sum(_mentions_thread(m.get("text") or "", src, where) and _from(m.get("text") or "", author) for m in rung)
            if n >= CAP_PER_HOUR:                                        # a loop is one author; a roll call is many, once each
                if n == CAP_PER_HOUR:
                    if where is None:                                    # a person's channel gets no router notes
                        api.call("chat.postMessage", channel=ch, thread_ts=src, username="router", unfurl_links="false",
                                 text=f"routing @{author} to {to} paused here for the hour ({CAP_PER_HOUR} doorbells) {QUIET}")
                    api.call("chat.postMessage", channel=ch, thread_ts=r["inbox_ts"], username="router", unfurl_links="false",
                             text=f"from router, {'#' + where + ' ' if where else ''}thread {src}: paused @{author} here "
                                  f"for the hour ({CAP_PER_HOUR} doorbells)")
                lines.append(f"@{to}: paused (cap of {CAP_PER_HOUR} an hour from @{author} in this thread)")
                continue
            api.call("chat.postMessage", channel=ch, thread_ts=r["inbox_ts"], username="router", unfurl_links="false",
                     unfurl_media="false", text=doorbell(author, src, msg.get("ts") or src, msg.get("text", ""), urgent, where,
                                                         private, reminder))
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
    return os.environ.get(_env(key)) or channels(reg)[key].get("id")


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
        for b in channels(reg)[key].get("bots", []):
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
    for key, ch in channels(reg).items():
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
    for key in channels(reg):
        if h["handle"] in subscribers(reg, key):
            cid = channel_id_for(reg, key)
            missing |= not cid
            print(f'subscribe_slack_channel(channelId="{cid or "<#" + key + " id not recorded>"}", topLevelOnly=true, '
                  f"expiresInSeconds={EXPIRES_S})   # #{key}")
    if key := h.get("linear"):
        print(f'subscribe_linear_issue(options={{"scope": "team", "teamId": "{key}"}}, expiresInSeconds={EXPIRES_S})   '
              f"# an issue filed into or changing state in @{h['handle']}'s Linear team, a human's included")
        print(f"# but no comment subscription on {key}: nearly every comment there is @{h['handle']}'s own, and it would wake "
              "it for each (every handle writes as one app)")
    print(f"# and subscribe_slack_thread(channelId, threadTs, expiresInSeconds={EXPIRES_S}) only for a thread a human may "
          f"answer in (@{h['handle']}'s #ask-daniel card, a thread a named human posted in); renew on every wake")
    print(f"# not for a thread between handles: the inbox already rings @{h['handle']} for each reply addressed to it, and "
          "never for its own posts, while a thread subscription also wakes it for each of its own replies")
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
    if me and any(same_agent(a, me) for a in ids):
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
    """A post routes in the front door and in a chat, where a lane hands off by starting a reply with another's name."""
    chats = {c["id"] for c in chat_channels(reg).values()}
    return (reg, routes_of(ns)) if ch == channel_id_for(reg, FRONT_DOOR) or ch in chats else None


CARD_FLAGS = ("--kind", "--question", "--why", "--option", "--recommend", "--default", "--deadline", "--page", "--area")


def ask_daniel_page(ns: argparse.Namespace, reg: dict) -> int:
    """`ask --to @daniel`: the ask's page in the Asks for Daniel database and its #ask-daniel card, as `ask-daniel --page`
    makes them, so the one `ask` verb files every ask in its target's store (a handle's Linear team, or Daniel's database)."""
    if ns.text is not None or ns.file is not None:
        raise SlackError(f"ask --to @{ns.to.lstrip('@')} is a page: give its Markdown as --page FILE.md (the template in "
                         f"{SKILL_ASKING}), not --text or --file")
    if ns.urgent:
        raise SlackError("a card has no --urgent: --kind blocking is for what can't proceed without him")
    if missing := [f for f in CARD_FLAGS if not getattr(ns, f[2:])]:
        raise SlackError(f"ask --to @{ns.to.lstrip('@')} is a page in the Asks for Daniel database and a card in #{ASK_CHANNEL} "
                         f"({SKILL_ASKING}): it needs {', '.join(missing)}; a page made with the Notion MCP goes out with "
                         f"`ask-daniel --link`")
    return cmd_ask_daniel(ns, reg)


def cmd_ask(ns: argparse.Namespace, reg: dict) -> int:
    if isinstance(ns.to, list):
        if len(ns.to) != 1:
            raise SlackError(f"--to {' --to '.join(ns.to)}: an ask goes to one owner, filed in its Linear team; for several, "
                             f"`research msg send --as ME --to {' '.join(ns.to)}`")
        ns.to = ns.to[0]
    if human_entry(reg, ns.to):
        return ask_daniel_page(ns, reg)
    h, owner = acting_as(ns, reg, ns.handle), name_entry(ns, reg, ns.to)
    if owner["handle"] == h["handle"]:
        raise SlackError("an ask goes to another handle")
    ch = channel_id(ns, reg, FRONT_DOOR)
    warn_unrouted(body := _body(ns), reg, h["handle"], (owner["handle"],))
    text = f"{tag(owner)} {linear_ask(ns, owner, h['handle'], body)}"
    check_length(reg, ch, text, POST_WORDS)
    _post(client(ns), ch, h["handle"], text, route=_route(ns, reg, ch), urgent=ns.urgent)
    return 0


def linear_ask(ns: argparse.Namespace, owner: dict, sender: str, body: str) -> str:
    """The ask filed as an issue in its owner's Linear team (`linear` in the registry), which holds the whole brief; what
    Slack carries is then its first line and the issue's link (with `--dry-run`, nothing filed, a stand-in link).  The body
    as it was when the owner has no team or Linear fails, so the ask still goes out on Slack in full."""
    if not ASKS_ON_LINEAR or not owner.get("linear"):
        return body
    from . import linear

    if ns.dry_run:
        print(f"dry-run: would file the ask in Linear team {owner['linear']}", file=sys.stderr)
        return f"{linear.first_line(body)}\nLinear: <https://linear.app|{owner['linear']}-?>"
    parent = getattr(ns, "parent", None)
    issue, why = linear.ask_issue(owner["linear"], sender, body, parent=parent, title=getattr(ns, "title", None))
    if not issue:
        print(f"research msg: not filed on Linear ({why}); asking on Slack only", file=sys.stderr)
        return body
    board = " / ".join(x["name"] for x in (issue.get("project"), issue.get("projectMilestone")) if x)
    print(f"linear: {issue['identifier']} {issue['url']}" + (f" under {parent}" if parent else "") + (f" on {board}" if board else ""))
    return f"{linear.first_line(body)}\nLinear: <{issue['url']}|{issue['identifier']}>"


def linear_close(text: str, state_type: str) -> None:
    """Move the Linear issue an ask's root links (`linear_ask`) to its team's `state_type` state (`Linear.move`)."""
    if not (m := LINEAR_LINK_RE.search(text or "")):
        return
    from . import linear

    try:
        print(f"linear: {m[1]} {linear.Linear().move(m[1], state_type)}")
    except (linear.LinearError, OSError, ValueError) as e:
        print(f"research msg: {m[1]} stays open on Linear ({e}); the status line stands", file=sys.stderr)


LINEAR_STATES = ("Backlog = filed, Todo = accepted, In Progress = someone is on it now, In Review = a PR is up, "
                 "Done = landed or answered")                    # what each state means for us


def render_audit(found: list[dict], counts: dict[str, dict[str, int]]) -> str:
    """The audit's findings by lane, one line each (`ISSUE kind: fix`), then their counts by lane and kind."""
    from .linear import AUDIT_KINDS

    lines, kinds = [], [k for k in AUDIT_KINDS if any(k in c for c in counts.values())]
    for lane in counts:
        mine = [f for f in found if f["lane"] == lane]
        lines.append(f"@{lane} ({len(mine)}):")
        lines += [f"  {f['issue']} {f['kind']}: {f['fix']}" for f in mine]
    lines.append(f"{len(found)} finding(s)" + (", by lane and kind:" if found else ""))
    if found:
        lines.append(f"  {'lane':<22}" + "".join(f"{k:>10}" for k in kinds) + f"{'total':>10}")
        for lane, c in [*counts.items(), ("total", {k: sum(x.get(k, 0) for x in counts.values()) for k in kinds})]:
            lines.append(f"  {('@' + lane) if lane != 'total' else lane:<22}" + "".join(f"{c.get(k, 0):>10}" for k in kinds)
                         + f"{sum(c.values()):>10}")
    return "\n".join(lines)


def cmd_linear(ns: argparse.Namespace, reg: dict) -> int:
    """`check` (what the credentials reach), `state ISSUE STATE`, `comment ISSUE TEXT --as HANDLE` (or `--worker NAME`, a
    lane's worker, which holds no handle: as LANE/NAME, ringing no one), `list TEAM [--state S]`, `audit [TEAM ...]
    [--lane HANDLE]` (exit 1 on any finding; a team the app can't read is named and skipped), `place ISSUE PROJECT
    [--milestone M]`, `edit ISSUE [--title T] [--due D|none] [--file BRIEF.md] [--parent ID]`, `relate ISSUE
    blocks|blocked-by|related|duplicate OTHER`, `unrelate ISSUE OTHER`, `update PROJECT --health H --body-file F`,
    `project create NAME --team KEY... [--lead USER] [--target D] [--description-file F]` and `milestone create|edit
    PROJECT NAME [--date D] [--description-file F] [--rename NEW]`. Each prints one line; a refusal exits 2."""
    from . import linear

    def need(value: str | None, what: str) -> str:
        if not value:
            raise SlackError(f"linear {ns.verb}: give {what}")
        return value
    value = " ".join(ns.value) or None
    if ns.worker and (ns.verb != "comment" or ns.handle):
        raise SlackError("--worker goes with `comment` alone, never with --as: --as HANDLE is for the handle's holder, "
                         "--worker NAME for a lane's worker, which holds no handle")
    if ns.worker and not WORKER_RE.fullmatch(ns.worker):
        raise SlackError(f"--worker {ns.worker!r}: a name of letters, digits, `.`, `_` and `-`, at most 40 (e.g. LINTOOLS)")
    if ns.dry_run and ns.verb not in ("check", "list", "audit", "edit", "state", "place", "comment"):
        raise SlackError(f"linear {ns.verb} has no --dry-run (it would write): leave it off, or try `edit`, `state`, "
                         "`place` or `comment`, which have one")
    try:
        brief = Path(ns.file).read_text() if ns.file else None
    except OSError as e:
        raise SlackError(f"--file {ns.file}: {e.strerror or e}") from None
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
        if ns.verb == "audit":
            keys = {h["handle"]: h["linear"] for h in reg["handles"] if h.get("linear")}
            teams = [keys.get(t.lstrip("@"), t) for t in ([ns.target] if ns.target else []) + ns.value]
            unread = lin.unread(teams or sorted(set(linear.owners(reg))))
            readable = [t for t in teams if t not in unread]
            if unread:
                print(f"linear audit: the app can't read team(s) {', '.join(unread)}, so their issues aren't audited",
                      file=sys.stderr)
            found = [] if teams and not readable else lin.audit(linear.owners(reg), readable or None,
                                                                None if ns.now is None else float(ns.now))
            if ns.lane:
                found = [f for f in found if f["lane"] == ns.lane.lstrip("@")]
            counts = linear.tally(found)
            print(json.dumps({"findings": found, "counts": counts, "unread": unread}, indent=2) if ns.json
                  else render_audit(found, counts))
            return 1 if found else 0
        if ns.verb == "update":
            print(lin.update(need(" ".join(filter(None, [ns.target, *ns.value])), "the PROJECT (a prefix of its name)"),
                             ns.health, brief))
            return 0
        if ns.verb in ("project", "milestone"):
            if ns.target not in ("create",) + (("edit",) if ns.verb == "milestone" else ()):
                raise SlackError("linear project: give `create NAME`" if ns.verb == "project"
                                 else "linear milestone: give `create` or `edit`, then PROJECT NAME")
            if ns.verb == "project":
                print(lin.create_project(need(value, "the project's NAME"), ns.team, ns.lead, ns.date, brief))
            elif len(ns.value) < 2:
                raise SlackError(f"linear milestone {ns.target}: give PROJECT (a prefix of its name, one word or quoted) and NAME")
            else:
                print(lin.milestone(ns.value[0], " ".join(ns.value[1:]), ns.date, brief, ns.rename, create=ns.target == "create"))
            return 0
        issue = need(ns.target, "the ISSUE (e.g. INF-12)")
        if ns.verb == "relate":
            if len(ns.value) != 2:
                raise SlackError(f"linear relate: give ISSUE {'|'.join(linear.RELATE)} OTHER")
            print(lin.relate(issue, *ns.value))
            return 0
        if ns.verb == "unrelate":
            print(f"{issue} unrelated: removed " + "; ".join(lin.unrelate(issue, need(value, "the OTHER issue"))))
            return 0
        if ns.verb == "state":
            state = need(value, 'the STATE by name (e.g. In Progress)')
            print(f"dry-run: would move {issue} to {state}" if ns.dry_run else f"{issue} {lin.move(issue, state)}")
            return 0
        if ns.verb == "place":
            if ns.dry_run:
                print(f"dry-run: would place {issue} in {need(value, 'the PROJECT (a prefix of its name)')}"
                      + (f" / {ns.milestone}" if ns.milestone else ""))
                return 0
            p = lin.place(issue, need(value, "the PROJECT (a prefix of its name)"), ns.milestone)
            print(f"{issue} placed in {p['project']}" + (f" / {p['milestone']}" if p["milestone"] else ""))
            return 0
        if ns.verb == "edit":
            i = lin.edit(issue, ns.title, ns.due, brief, ns.parent, send=not ns.dry_run)
            print(("dry-run: would set " if ns.dry_run else f"{issue} set ") + linear.said(i))
            return 0
        text = need(value, "the TEXT")
        if not ns.worker:
            author = acting_as(ns, reg, need(ns.handle, "--as HANDLE, the comment's sender (a lane's worker: --worker NAME)"))["handle"]
            warn_unrouted(text, reg, author)
        info = lin.gql("query($id: String!) { issue(id: $id) { id identifier url team { key } } }", {"id": issue})["issue"]
        if ns.worker:
            lane = linear.owners(reg).get(info["team"]["key"])
            if not lane:
                raise SlackError(f"--worker: {issue}'s team {info['team']['key']} has no handle in the registry, so there is "
                                 "no lane to comment as")
            author = f"{lane}/{ns.worker}"
        if ns.dry_run:
            print(f"dry-run: would comment on {issue} as {author}: {text}")
            return 0
        lin.comment(info["id"], author, text)
        if ns.worker:
            print(f"{issue} commented as {author}")
            if names := lead_names(text, reg):
                print(f"route: a worker's comment rings no one, so {', '.join('@' + n for n in names)} won't hear of it "
                      "from here: ask your lead to", file=sys.stderr)
            return 0
        print(f"{issue} commented")
        ring_comment(ns, reg, author, info, text)
        return 0
    except linear.LinearError as e:
        raise SlackError(str(e)) from None


def ring_comment(ns: argparse.Namespace, reg: dict, author: str, issue: dict, text: str) -> list[str]:
    """Ring the inbox of the issue's team's handle and of each name the comment starts with, as `send` would, so nobody
    polls Linear for comments (no one subscribes to them: every handle writes as one app).  The author is never
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
    check_length(reg, ch, text := announcement(h, to, body, ns.verify, ns.ships), ANNOUNCE_WORDS)
    _post(client(ns), ch, h["handle"], text, route=_route(ns, reg, ch), urgent=ns.urgent)
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
    if (getattr(ns, "parent", None) or getattr(ns, "title", None)) and not (kind == "ask" and not ns.thread and len(to) == 1):
        raise SlackError("--parent and --title go with an ask, which files an issue: a top-level send --to one handle")
    def own() -> bool:
        return rings_own_inbox(routes_of(ns), h["handle"], self_agent_id())
    if any(t["handle"] == h["handle"] for t in to) and not own():
        raise SlackError(f"--to names someone else (@{h['handle']}'s inbox is this agent's own)")
    body = _body(ns)
    if kind != "announce" and not ns.thread and not to and is_announcement({"text": body}):
        raise SlackError(f"it starts with {ANNOUNCE}, which makes a top-level post an announcement to every handle: send it "
                         "with --kind announce (which checks its pins), or start it with something else")
    if to or kind != "announce":                                     # an announcement with no --to reaches every handle
        warn_unrouted(body, reg, h["handle"], tuple(t["handle"] for t in to))
    if not to and (lead := lead_names(body, reg)) and set(lead) == {h["handle"]} and not own():
        raise SlackError(f"it starts with @{h['handle']} only, and a post never wakes its own inbox's holder: name its "
                         "addressee with --to")
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
    if not getattr(ns, "report", False):
        check_length(reg, ch, text, ANNOUNCE_WORDS if kind == "announce" else POST_WORDS)
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
    warn_unrouted(body, reg, h["handle"])
    if not getattr(ns, "report", False):
        check_length(reg, ch, body, POST_WORDS)
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
    sender stopped partway.  `--channel ask-the-team` or `chat-NAME` (or --channel-id) delivers a post in another channel,
    whose doorbells name it.  `--event` (a watcher's Slack message event) gives the post: Slack's own copy where the bot can read the
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
        if not copy or not _unreadable(e):
            raise
        print(f"deliver: {e} reading {ch}, so the event's own copy is delivered", file=sys.stderr)
        msg = copy
    if not msg:
        raise SlackError(f"no message {ts} in the channel")
    msg = {**msg, "thread_ts": msg.get("thread_ts") or thread}
    deliver(api, reg, routes_of(ns), ch, msg)
    return 0


def run_holder(reg: dict, routes: Routes, agent: str) -> str | None:
    """The handle whose open inbox route names `agent` as its holder or one of its posters, else whose registry `holder`
    names it; None when no handle does (a worker: its launcher passes --to)."""
    for r in routes.all().values():
        if not r.get("closed") and agent in (r.get("holder"), *r.get("posters", [])):
            return r["name"]
    return next((h["handle"] for h in reg["handles"] if agent in (h.get("holder") or "")), None)


def cmd_ring_run(ns: argparse.Namespace, reg: dict) -> int:
    """Ring the inbox of the lane that launched a run, once the run has ended: `from runs, run R rc N (AGENT): TEXT`.  The
    control pod calls it on each run's end (`request.json`'s submitted_by, the record's rc), so a coordinator that ended its
    turn while the run went on is woken by the agent watch, not by polling.  Once per run: a run already rung is skipped."""
    m = AGENT_ID_RE.search(ns.agent or "")
    agent = m[0] if m else (ns.agent or "")
    routes = routes_of(ns)
    name = (ns.to or "").lstrip("@") or run_holder(reg, routes, agent)
    if not name:
        raise SlackError(f"no handle holds {agent or '(no agent)'}: a worker's run names its launcher, so pass --to @LANE")
    to, r = routes.resolve(name)
    if why := unreachable(name, to, r):
        raise SlackError(f"@{name}: {why}")
    api, ch = client(ns), r.get("channel") or channel_id(ns, reg, FRONT_DOOR)
    head = f"from runs, run {ns.run} rc {ns.rc}"
    since = f"{_now().timestamp() - 2 * 86400:.6f}"
    if not api.dry_run and any(is_doorbell(m, reg) and (m.get("text") or "").startswith(head + " ")
                               for m in api.paged("conversations.replies", "messages", channel=ch, ts=r["inbox_ts"], oldest=since)):
        print(f"@{to}: run {ns.run} was rung already", file=sys.stderr)
        return 0
    body = plain(ns.text or "finished")[:DOORBELL_MAX]
    api.call("chat.postMessage", channel=ch, thread_ts=r["inbox_ts"], username="router", unfurl_links="false",
             unfurl_media="false", text=f"{head} ({agent or '-'}): {body}")
    print(f"route @{to}: rang for run {ns.run}", file=sys.stderr)
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
        at = datetime.fromtimestamp(float(m["ts"]), UTC).strftime("%H:%M:%SZ") if m.get("ts") else "?"
        print(f"{m.get('ts')} {at}{thread} {who}{' (' + ', '.join(tags) + ')' if tags else ''}: {plain(m.get('text', ''), reg)}")
    return 0


# --- inboxes ----------------------------------------------------------------------------------------------------------------

def publish_route(root: Path, r: dict, message: str, where: str = ROUTES_DIR, name: str | None = None) -> str:
    """Write `<where>/<name>.json` (default `routes.d/<its name>.json`; a chat's person goes to people.d) in the notes clone
    at `root` and push it: the line `notes sync` printed."""
    name = name or r["name"]
    if not (root / ".git").exists():
        try:
            from . import notes_sync
            notes_sync.ensure_clone(root)
        except ImportError:
            pass
    if not (root / ".git").exists():
        if os.environ.get("RESEARCH_NOTES_TOKEN"):                  # no git here (the Slack events service): GitHub's API
            rel = f"{where}/{name}.json"
            cur = notes_api("GET", rel)
            notes_api("PUT", rel, {"message": message, "branch": "main",
                                   "content": base64.b64encode((json.dumps(r, indent=2, ensure_ascii=False) + "\n").encode()).decode(),
                                   **({"sha": cur["sha"]} if isinstance(cur, dict) and cur.get("sha") else {})})
            return f"notes: pushed {rel} (GitHub's contents API)"
        raise SlackError(f"no notes clone at {root} (set RESEARCH_NOTES, or RESEARCH_NOTES_TOKEN to clone it); {where}/{name}.json "
                         "is:\n" + json.dumps(r, indent=2))
    p = root / where / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
    rel = f"{where}/{name}.json"
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
        raise SlackError(f"{rel} is written but not pushed: {line}")
    return line


def notes_api(method: str, rel: str, body: dict | None = None) -> Any:
    """GitHub's contents API on the notes repository, with RESEARCH_NOTES_TOKEN, for a caller with no clone (the Slack events
    service): the JSON answer, or None for a GET of a path that isn't there."""
    if not (token := os.environ.get("RESEARCH_NOTES_TOKEN")):
        raise SlackError("RESEARCH_NOTES_TOKEN is not set: without a notes clone, the notes are read and written with it")
    req = urllib.request.Request(NOTES_API + urllib.parse.quote(rel) + ("?ref=main" if method == "GET" else ""), method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                          "Content-Type": "application/json", "User-Agent": "research-msg"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        if e.code == 404 and method == "GET":
            return None
        raise SlackError(f"notes: GitHub {method} {rel}: HTTP {e.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise SlackError(f"notes: GitHub {method} {rel}: {getattr(e, 'reason', e)}") from None


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
            raise SlackError(f"@{name} is not a lead's handle: only leads get names, and workers are reached through their lead; "
                             f"a new lead is added to {REGISTRY_REL} first")
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
        text = plain(m.get("text") or "", reg)
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


# --- traffic: the monitor ---------------------------------------------------------------------------------------------------

def comment_by(c: dict) -> str | None:
    """Who a Linear comment is from: the app's `createAsUser` name, else (a person's key, where every comment shows as that
    person) the handle its body opens with, `@handle: ` (`Linear.comment`), else the person."""
    if bot := (c.get("botActor") or {}).get("userDisplayName"):
        return bot
    if m := re.match(r"@([a-z0-9._-]+): ", c.get("body") or ""):
        return m[1]
    return (c.get("user") or {}).get("name")


RECORD_UNREADABLE = "unreadable"                                     # the record of an ask's answer (Linear, Notion) couldn't be read


def issue_state(ident: str, asker: str | None, cache: dict[str, tuple[bool, str]]) -> tuple[bool, str]:
    """Whether the Linear issue `ident` was taken up there: it left Backlog (accepted, started, done or canceled), or
    someone other than `asker` commented on it; and the state that says so.  Not taken when Linear can't say, so a doubtful
    ask still shows."""
    if ident not in cache:
        from . import linear
        q = "query($id: String!) { issue(id: $id) { state { type } comments { nodes { botActor { userDisplayName } user { name } body } } } }"
        try:
            i = linear.Linear().gql(q, {"id": ident})["issue"]
            other = next((w for c in i["comments"]["nodes"] if (w := comment_by(c)) and w != asker), None)
            if i["state"]["type"] != "backlog":
                cache[ident] = (True, f"left Backlog ({i['state']['type']})")
            elif other:
                cache[ident] = (True, f"commented on by {other}")
            else:
                cache[ident] = (False, "in Backlog, no comment but the asker's")
        except (linear.LinearError, OSError, KeyError, TypeError, ValueError) as e:
            cache[ident] = (False, f"{RECORD_UNREADABLE} ({e})")
    return cache[ident]


PAGE_WAITING = "Waiting on Daniel"                               # an ask's page's Status until Daniel (or its deadline) moves it
URL_RE = re.compile(r"https://[^\s\"'<>|()\\]+")


def notion_page_id(url: str) -> str | None:
    """The page id a Notion URL ends with (`.../p/<32 hex>`, `.../Title-<32 hex>`, or the dashed UUID), or None."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if host not in NOTION_HOSTS and not host.endswith(".notion.site"):
        return None
    m = re.search(r"([0-9a-f]{32})$", parts.path.rsplit("/", 1)[-1].replace("-", "").lower())
    return m[1] if m else None


def page_link(m: dict) -> str | None:
    """The first Notion page a post links, in its text, blocks or attachments."""
    raw = " ".join([m.get("text") or "", json.dumps(m.get("blocks") or []), json.dumps(m.get("attachments") or [])])
    return next((u for u in URL_RE.findall(raw) if notion_page_id(u)), None)


def daniel_cards(api: Slack, reg: dict, since: float, now: float) -> list[dict]:
    """The #ask-daniel cards posted in (since, now] that link their ask's page (Notion's record of it), each `{ts, by, url,
    page}`; the page link is the card's, else its asker's mention in the thread (`ask-daniel`).  A card that links no page
    predates the Asks for Daniel database, and isn't counted."""
    if not (ch := channel_id_for(reg, ASK_CHANNEL)):
        return []
    out = []
    try:
        for m in api.paged("conversations.history", "messages", channel=ch, oldest=f"{since:.6f}"):
            if float(m["ts"]) > now or m.get("thread_ts") not in (None, m["ts"]):
                continue
            url, mention = page_link(m), None
            if not url and m.get("reply_count"):
                for r in api.paged("conversations.replies", "messages", channel=ch, ts=m["ts"]):
                    if r.get("ts") != m["ts"] and (url := page_link(r)):
                        mention = r
                        break
            if url:
                asker = re.search(r"asked by <[^|>]*\|([a-z0-9._-]+)>", json.dumps(m.get("blocks") or []) + (m.get("text") or ""))
                out.append({"ts": m["ts"], "by": asker[1] if asker else poster(mention or m, reg), "url": url, "page": notion_page_id(url)})
    except SlackError as e:
        print(f"research msg: #{ASK_CHANNEL} unread ({e}); Daniel's cards aren't counted", file=sys.stderr)
    return out


def page_state(notion: "Notion", page: str) -> dict:
    """An ask's page as its record says: `{status, answered, at}`.  Answered once its Status has left PAGE_WAITING (Daniel
    approved, decided otherwise or asked for a revision, or its default applied); `at` is then the page's last edit, the latest
    the Status can have moved, since Notion keeps no history of a property."""
    p = notion.call("GET", f"/pages/{page}")
    status = prop_text((p.get("properties") or {}).get("Status"))
    if not status:
        raise SlackError(f"page {page} has no Status")
    answered = status != PAGE_WAITING
    return {"status": status, "answered": answered,
            "at": _iso_epoch(p["last_edited_time"]) if answered and p.get("last_edited_time") else None}


class Pages:
    """Each card's page state, read once; the Notion client is made on the first read, and its absence (no $NOTION_API_KEY)
    is an error per page rather than for the whole report."""

    def __init__(self, client: Callable[[], "Notion"] | None = None):
        self.make, self.client, self.seen = client, None, {}

    def state(self, page: str) -> dict:
        if page not in self.seen:
            if not (self.make or os.environ.get("NOTION_API_KEY")):
                self.seen[page] = {"status": None, "answered": False, "at": None,
                                   "error": "no $NOTION_API_KEY (a Notion connection's token, shared on the Asks for Daniel database)"}
                return self.seen[page]
            try:
                self.client = self.client or (self.make or Notion)()
                self.seen[page] = page_state(self.client, page)
            except (SlackError, ValueError) as e:
                self.seen[page] = {"status": None, "answered": False, "at": None, "error": str(e)}
        return self.seen[page]


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
    An issue filed straight into another state shows that as a move out of Backlog at its creation, which times no answer:
    with nothing later, it is answered but untimed (`linear-filed`, like an issue filed already Done)."""
    i = lin.gql(LINEAR_ANSWER_Q, {"id": ident})["issue"]
    born = _iso_epoch(i["createdAt"])
    out = [{"at": _iso_epoch(h["createdAt"]), "how": "linear-state", "by": None}
           for h in i["history"]["nodes"]
           if (h.get("fromState") or {}).get("type") == "backlog" and (h.get("toState") or {}).get("type") not in (None, "backlog")
           and _iso_epoch(h["createdAt"]) > born]
    for c in i["comments"]["nodes"]:
        if (who := comment_by(c)) and who != asker:
            out.append({"at": _iso_epoch(c["createdAt"]), "how": "linear-comment", "by": who})
    if not out and any((h.get("fromState") or {}).get("type") == "backlog" and (h.get("toState") or {}).get("type") not in (None, "backlog")
                       for h in i["history"]["nodes"]):
        return {"at": born, "how": "linear-filed", "by": None}
    return min(out, key=lambda a: a["at"], default=None)


def nearest_rank(xs: list[float], q: float) -> float | None:
    return sorted(xs)[max(math.ceil(q * len(xs)), 1) - 1] if xs else None


def ask_latency(api: Slack, reg: dict, routes: Routes, ch: str, since: float, now: float, stale_s: float,
                linear_client: Callable[[], Any] | None = None, notion_client: Callable[[], Any] | None = None) -> dict:
    """Each ask posted in (since, now] that rang a lane's inbox, and each #ask-daniel card linking its page, and how long its
    first answer took, as the ask's record shows it (`record`, `ref`):
    - an ask linked to a Linear issue: the issue's first move out of Backlog or first comment by anyone but the asker
      (`issue_answer`), with who gave it (`answer_by`), or the addressed lane's own reply in the thread, whichever came first.
      An issue Linear can't read leaves the ask to Slack (`linear_errors`).
    - Daniel's card (lane `daniel`): its page's Status leaving PAGE_WAITING, at the page's last edit (`page_state`); `state`
      is the Status.  A page Notion can't read leaves the card unanswered (`notion_errors`).
    - any other ask: the first reply in its thread under a name of the lane's inbox (its own, or one it was renamed from).
      One with no reply that carries 👀, ✅ or ❌ and rang only this lane counts as answered but untimed: Slack keeps no time
      for a reaction, and every handle reacts as one bot.
    Only a lane or a named person asks: a machine's alert (`vy-monitors`) isn't an ask.  Answers after `now` don't count.  Per lane: asks, answered (by Slack, Linear, Notion or untimed), the nearest-rank
    median and p90 of the timed answers in minutes, the unanswered, and those unanswered for longer than `stale_s`."""
    tops: dict[str, dict[str, dict]] = {}                            # channel -> ts -> top-level post
    replies: dict[tuple[str, str], list[dict]] = {}
    asks, counted, names = [], set(), {}
    askers = {h["handle"] for h in reg["handles"]} | {poster({"user": p.get("slack_user")}, reg) for p in reg.get("humans", [])}
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
            if root and is_ask(root) and poster(root, reg) in askers and poster(root, reg) not in names[r["inbox_ts"]]:
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
        asker, record, ref, found = poster(root, reg), "slack", None, None
        if link := LINEAR_LINK_RE.search(root.get("text") or ""):
            if (link[1], asker) not in issues:
                from . import linear
                try:
                    lin = lin or (linear_client or linear.Linear)()
                    issues[(link[1], asker)] = {"answer": issue_answer(lin, link[1], asker)}
                except (linear.LinearError, OSError, KeyError, TypeError, ValueError) as e:   # the ask still counts, from Slack
                    issues[(link[1], asker)] = None
                    linear_errors.append(f"{link[1]}: {e}")
            if (read := issues[(link[1], asker)]) is not None:
                record, ref = "linear", link[1]
                found = [ans] if (ans := read["answer"]) and ans["at"] <= now else []
        if root.get("reply_count") and key not in replies:          # the addressee's own reply answers too, record or not
            replies[key] = api.paged("conversations.replies", "messages", channel=a["channel"], ts=root["ts"], oldest=root["ts"])
        found = (found or []) + [{"at": float(m["ts"]), "how": "slack-reply", "by": poster(m, reg)} for m in replies.get(key, [])
                                 if m.get("ts") != root["ts"] and poster(m, reg) in a["names"] and t0 < float(m["ts"]) <= now]
        first = min(found, key=lambda x: x["at"], default=None)
        untimed = (record == "slack" and first is None and rang[key] == 1
                   and any(x.get("name") in ANSWER_REACTIONS for x in root.get("reactions", [])))
        out.append({"lane": lane, "ts": root["ts"], "by": asker, "linear": link[1] if link else None, "record": record, "ref": ref,
                    "first": first["how"] if first else "reaction" if untimed else None, "answer_by": first["by"] if first else None,
                    "minutes": round(max(first["at"] - t0, 0) / 60, 1) if first and first["how"] != "linear-filed" else None,
                    "age_minutes": round((now - t0) / 60, 1)})
    pages, notion_errors = Pages(notion_client), []
    for c in daniel_cards(api, reg, since, now):
        t0, st = float(c["ts"]), pages.state(c["page"])
        if st.get("error"):
            notion_errors.append(f"{c['url']}: {st['error']}")
        done = st["answered"] and st["at"] is not None and st["at"] <= now
        out.append({"lane": DANIEL, "ts": c["ts"], "by": c["by"], "linear": None, "record": "notion", "ref": c["url"],
                    "state": st["status"], "first": "notion-status" if done else None, "answer_by": None,
                    "minutes": round(max(st["at"] - t0, 0) / 60, 1) if done else None, "age_minutes": round((now - t0) / 60, 1)})
    lanes = []
    for lane in sorted({x["lane"] for x in out}):
        mine = [x for x in out if x["lane"] == lane]
        timed = [x["minutes"] for x in mine if x["minutes"] is not None]
        unanswered = [x for x in mine if not x["first"]]
        lanes.append({"lane": lane, "asks": len(mine), "answered": len(mine) - len(unanswered),
                      "slack": sum(x["first"] == "slack-reply" for x in mine),
                      "linear": sum((x["first"] or "").startswith("linear-") and x["first"] != "linear-filed" for x in mine),
                      "notion": sum(x["first"] == "notion-status" for x in mine),
                      "untimed": sum(x["first"] in ("reaction", "linear-filed") for x in mine),
                      "median_min": nearest_rank(timed, 0.5), "p90_min": nearest_rank(timed, 0.9),
                      "unanswered": len(unanswered), "stale": sum(x["age_minutes"] * 60 > stale_s for x in unanswered)})
    return {"since": since, "now": now, "lanes": lanes, "asks": out, "linear_errors": linear_errors, "notion_errors": notion_errors}


def render_latency(rep: dict, hours: float) -> str:
    cols = ("asks", "answered", "slack", "linear", "notion", "untimed", "median_min", "p90_min", "unanswered", "stale")
    lines = [f"ask latency over the last {hours:g} h (to {both(rep['now'])}): minutes from an ask to the first answer its record "
             f"shows (Linear for a handle's linked ask, the Notion page's Status for Daniel's, else Slack)",
             f"  {'lane':<26}" + "".join(f"{c:>11}" for c in cols)]
    for x in rep["lanes"]:
        lines.append(f"  @{x['lane']:<25}" + "".join(f"{'-' if x[c] is None else x[c]:>11}" for c in cols))
    if rep["linear_errors"]:
        lines.append(f"  linear: {len(rep['linear_errors'])} issue(s) unread, counted from Slack alone: {rep['linear_errors'][0]}")
    if rep["notion_errors"]:
        lines.append(f"  notion: {len(rep['notion_errors'])} page(s) unread, counted unanswered: {rep['notion_errors'][0]}")
    return "\n".join(lines)


VOLUME_WORDS = 80                                                # a post longer than this is a brief: it belongs in its issue
VOLUME_THREADS = 10                                              # the busiest threads `traffic --volume` names
POST_WORDS, ANNOUNCE_WORDS = 60, 150                             # the coordinating skill's lengths ("How long, and where")
LENGTH_FIX = ("put the body in its Linear issue (`research msg linear comment ISSUE --file BODY.md --as ME`), or in a file, "
              "PR or note, and post a short line that links it")


def check_length(reg: dict, ch: str, text: str, limit: int) -> None:
    """A post `send`, `reply`, `announce` or `ask` makes in the front door, in words as `traffic --volume` counts them: past
    `limit` it goes out with a note on stderr, and past twice `limit` it is refused before anything is posted."""
    if ch != channel_id_for(reg, FRONT_DOOR) or (n := len(plain(text).split())) <= limit:
        return
    if n > 2 * limit:
        raise SlackError(f"{n} words, past twice the {limit} a post in #{FRONT_DOOR} may have, so nothing was posted: {LENGTH_FIX}")
    print(f"research msg: {n} words, past the {limit} a post in #{FRONT_DOOR} may have (posted; past {2 * limit} it is refused): "
          f"{LENGTH_FIX}", file=sys.stderr)


def volume(api: Slack, reg: dict, ch: str, since: float, now: float) -> dict:
    """Every post in `ch` in (since, now]: the top-level ones and the replies of each thread with one in the window (its root
    read back to HISTORY_S before it), inboxes aside, whose replies are doorbells that copy these posts.  Per poster (the post's
    `username`, a named person by the registry), by posts: its posts, their median, p90 and most words, and those over
    VOLUME_WORDS; the totals; and the VOLUME_THREADS threads with the most replies in the window."""
    posts, threads = [], []
    for root in api.paged("conversations.history", "messages", channel=ch, oldest=f"{since - HISTORY_S:.6f}"):
        if (root.get("text") or "").startswith(INBOX_HEAD):
            continue
        if since < float(root["ts"]) <= now:
            posts.append(root)
        if root.get("reply_count") and float(root.get("latest_reply") or 0) > since:
            replies = [m for m in api.paged("conversations.replies", "messages", channel=ch, ts=root["ts"], oldest=f"{since:.6f}")
                       if m.get("ts") != root["ts"] and since < float(m["ts"]) <= now]
            posts += replies
            if replies:
                threads.append({"ts": root["ts"], "by": poster(root, reg) or root.get("username") or root.get("user") or "?",
                                "replies": len(replies), "first": " ".join(plain(root.get("text") or "", reg).split())[:80]})
    by: dict[str, list[int]] = {}
    for m in posts:
        by.setdefault(poster(m, reg) or m.get("username") or m.get("user") or "?", []).append(len(plain(m.get("text") or "").split()))
    rows = [{"poster": who, "posts": len(w), "median": nearest_rank(w, 0.5), "p90": nearest_rank(w, 0.9), "max": max(w),
             "over": sum(x > VOLUME_WORDS for x in w)} for who, w in by.items()]
    words = [x for w in by.values() for x in w]
    return {"since": since, "now": now, "posters": sorted(rows, key=lambda r: (-r["posts"], r["poster"])),
            "totals": {"posts": len(words), "words": sum(words), "over": sum(x > VOLUME_WORDS for x in words),
                       "median": nearest_rank(words, 0.5)},
            "threads": sorted(threads, key=lambda t: (-t["replies"], t["ts"]))[:VOLUME_THREADS]}


def render_volume(rep: dict, hours: float) -> str:
    t = rep["totals"]
    lines = [f"volume in #{FRONT_DOOR} over the last {hours:g} h (to {both(rep['now'])}): {t['posts']} posts, {t['words']} words "
             f"(median {t['median'] if t['median'] is not None else '-'}), {t['over']} over {VOLUME_WORDS} words",
             f"  {'poster':<26}" + "".join(f"{c:>8}" for c in ("posts", "median", "p90", "max", f">{VOLUME_WORDS}"))]
    for r in rep["posters"]:
        lines.append(f"  @{r['poster']:<25}" + "".join(f"{r[c]:>8}" for c in ("posts", "median", "p90", "max", "over")))
    if rep["threads"]:
        lines.append("  threads with the most replies:")
        lines += [f"    {x['ts']} by @{x['by']}: {x['replies']} replies: {x['first']}" for x in rep["threads"]]
    return "\n".join(lines)


def cmd_traffic(ns: argparse.Namespace, reg: dict) -> int:
    """Doorbells per inbox (the most in any one hour), pauses, `@name`s nobody owns (dropped), and each ask with the record
    that says whether it was answered, over the last --hours: a handle's ask linked to a Linear issue by the issue
    (`issue_state`), Daniel's #ask-daniel card by its page's Status (`page_state`), and only an ask with neither by Slack (a
    reply, or 👀, ✅ or ❌).  An ask unanswered after --stale-h hours is listed with its record's state; one whose record can't
    be read (no Linear or Notion credentials) is listed apart as unchecked, so a missing key is one finding, not a false alert
    per ask.  With --latency, how fast each lane first answered its asks instead (`ask_latency`); with --volume, how
    much each poster wrote (`volume`)."""
    api, ch, routes = client(ns), channel_id(ns, reg, FRONT_DOOR), routes_of(ns)
    now = time.time() if ns.now is None else float(ns.now)
    since = now - ns.hours * 3600
    if ns.volume:
        rep = volume(api, reg, ch, since, now)
        print(json.dumps(rep, indent=2) if ns.json else render_volume(rep, ns.hours))
        return 0
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
    dropped, asks, groups = [], [], {g for h in reg["handles"] for g in group_ids(h)}
    fresh: set[str] | None = None                                    # origin/main's group ids, fetched on the first unknown
    taken: dict[str, tuple[bool, str]] = {}                          # Linear issue -> (taken, its state)
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
        if not (m.get("bot_id") and LEAD_RE.match(m.get("text") or "") and is_ask(m)):   # older clients tagged an fyi without (quiet)
            continue
        x = {"ts": m["ts"], "by": poster(m, reg), "to": lead_names(m.get("text") or "", reg)}
        if link := LINEAR_LINK_RE.search(m.get("text") or ""):
            answered, state = issue_state(link[1], x["by"], taken)
            answered = None if not answered and state.startswith(RECORD_UNREADABLE) else answered
            x.update(record="linear", ref=link[1], state=state)
        else:
            answered = bool(m.get("reply_count")) or any(r.get("name") in ANSWER_REACTIONS for r in m.get("reactions", []))
            x.update(record="slack", ref=None, state=("replied" if m.get("reply_count") else "reacted") if answered else "no reply or reaction")
        asks.append((x, answered))
    pages = Pages()
    for c in daniel_cards(api, reg, since, now):
        st = pages.state(c["page"])
        asks.append(({"ts": c["ts"], "by": c["by"], "to": [DANIEL], "record": "notion", "ref": c["url"],
                      "state": f"{RECORD_UNREADABLE} ({st['error']})" if st.get("error") else f"Status {st['status']}"},
                     None if st.get("error") else st["answered"]))
    answered_by = [x for x, ok in asks if ok]
    old_enough = [(x, ok) for x, ok in asks if now - float(x["ts"]) > ns.stale_h * 3600]
    stale = [x for x, ok in old_enough if ok is False]
    unchecked = [x for x, ok in old_enough if ok is None]
    if ns.json:
        print(json.dumps({"inboxes": rows, "dropped": dropped, "answered": answered_by, "unanswered": stale, "unchecked": unchecked}, indent=2))
        return 0
    print(f"traffic over the last {ns.hours:g} h (to {both(now)})")
    for x in rows:
        print(f"  @{x['name']:<28} {x['doorbells']:>4} doorbells, at most {x['peak_per_hour']} in an hour" +
              (f", {x['paused']} paused" if x["paused"] else ""))
    for d in dropped:
        print(f"  DROPPED @{d['name']}: {d['ts']} by @{d['by']} (no inbox and no handle)")
    if answered_by:
        n = {r: sum(x["record"] == r for x in answered_by) for r in ("linear", "notion", "slack")}
        print(f"  answered: {len(answered_by)} ask(s), by their record: {n['linear']} on Linear, {n['notion']} on Notion, "
              f"{n['slack']} on Slack (no record)")
    for s in stale:
        print(f"  UNANSWERED {s['ts']} by @{s['by']} to {' '.join('@' + t for t in s['to']) or 'no one it can ring'} ({both(s['ts'])}): "
              + {"linear": f"Linear {s['ref']} {s['state']}", "notion": f"Notion page {s['ref']} {s['state']}",
                 "slack": f"Slack, no record: {s['state']}"}[s["record"]])
    if unchecked:
        print(f"  UNCHECKED: {len(unchecked)} ask(s) whose record can't be read: {unchecked[0]['state']}")
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
    groups = api.call("usergroups.list", include_disabled="true", include_users="true").get("usergroups", [])
    retired = {g for h in reg["handles"] for g in h.get("retired_usergroups", [])}
    groups = [g for g in groups if g.get("id") not in retired]         # a retired group is never matched or touched again
    live, by_id = {g.get("handle"): g for g in groups}, {g.get("id"): g for g in groups}
    for h in reg["handles"]:
        name, desc = h["handle"], h["remit"][:250]
        g = by_id.get(h.get("usergroup")) or live.get(name)          # by id first: a renamed handle keeps its group
        if g is None:
            print(f"create  @{name}")
            gid = (api.call("usergroups.create", name=name, handle=name, description=desc).get("usergroup") or {}).get("id") if write else None
        else:
            gid = g["id"]
            if g.get("date_delete") and g.get("users"):                 # enabling it would notify its members again
                print(f"refuse  @{name} ({gid}): disabled with {len(g['users'])} member(s); a tag would notify them. Create an "
                      "empty group instead: rename this one aside and add it to the handle's retired_usergroups")
                continue
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


# --- chat channels: one private channel per person --------------------------------------------------------------------------

def chat_welcome(name: str) -> str:
    """The two lines `chat open` pins: whose the channel is, and where the answers come."""
    return (f"*This channel is {name.split()[0]}'s*, for questions and requests to Verity's agents: post a new message, and "
            "start it with a lane's name (`@circuits ...`) to ask that lane.\n"
            "*Agents answer in your post's thread*: reply there to follow up, one thread per question.")


def chat_channel(api: Slack, key: str, known: str | None) -> tuple[str, bool]:
    """#key's id, and whether it is new: the recorded channel `known` while Slack has it, else a new private channel, else the
    one of that name the bot is in (a channel `chat close` archived, unarchived again)."""
    ch = None
    if known:
        try:
            ch = api.call("conversations.info", channel=known).get("channel") or ({"id": known} if api.dry_run else None)
        except SlackError as e:
            if e.code != "channel_not_found":
                raise
    new = False
    if not ch:
        try:
            ch = api.call("conversations.create", name=key, is_private="true").get("channel") or {"id": f"<#{key}'s id>"}
            new = True
        except SlackError as e:
            if e.code != "name_taken":
                raise
            ch = next((c for c in api.paged("conversations.list", "channels", types="private_channel", exclude_archived="false")
                       if c.get("name") == key), None)
            if not ch:
                raise SlackError(f"#{key} exists, but the Agents bot isn't in it: a member there invites it (`/invite @Agents`), "
                                 "or pick another NAME") from None
    if ch.get("is_archived"):
        api.call("conversations.unarchive", channel=ch["id"])
    return ch["id"], new


def cursor_bot(members: list[dict]) -> str | None:
    """Cursor's app's bot user, which must be in a channel for `subscribe_slack_channel` to wake anyone there: a bot whose
    name or real name says cursor (one named exactly `cursor` first)."""
    bots = [m for m in members if m.get("is_bot") and not m.get("deleted") and "cursor" in
            f"{m.get('name', '')} {m.get('real_name', '')} {(m.get('profile') or {}).get('real_name', '')}".lower()]
    bots.sort(key=lambda m: (m.get("name") != "cursor", m.get("id", "")))
    return bots[0]["id"] if bots else None


def chat_invite(api: Slack, cid: str, uid: str) -> str:
    try:
        api.call("conversations.invite", channel=cid, users=uid)
    except SlackError as e:
        if e.code != "already_in_channel":
            raise
        return "already in"
    return "invited"


def connect_step(e: SlackError, email: str, key: str, reg: dict) -> str:
    """The one thing Daniel does when Slack refuses the bot's Slack Connect invite."""
    if e.code == "missing_scope":
        app = reg.get("slack_app_id")
        page = f"https://api.slack.com/apps/{app}/install-on-team" if app else "the install page `research msg slack-app apply` prints"
        return (f"approve the Agents app's scope conversations.connect:write: open {page} and click Allow "
                f"(Slack refused the bot's Slack Connect invite of {email}: missing_scope)")
    return (f"invite {email} to the workspace, or to #{key} yourself (open #{key}, click its name, choose Add people and enter "
            f"{email}; Slack refused the bot's Slack Connect invite: {e.code or e})")


def publish_person(api: Slack, r: dict, message: str) -> None:
    name = r["channel"]["key"][len(CHAT_PREFIX):]
    if api.dry_run:
        print(f"plan: publish {PEOPLE_DIR}/{name}.json to the notes: {json.dumps(r, ensure_ascii=False)}")
    else:
        print(publish_route(notes_root(), r, message, PEOPLE_DIR, name))


def chat_expect(ns: argparse.Namespace, reg: dict, book: ChatPeople) -> int:
    """`chat expect NAME --email E`: people.d/NAME.json with status `expected`, their email and no channel, which no command
    routes by.  A rerun updates the email or the note; a chat already open refuses."""
    key, email = CHAT_PREFIX + ns.name, (ns.email or "").strip()
    if not EMAIL_RE.match(email):
        raise SlackError("chat expect needs --email E: the address Daniel invited to the workspace, which their team_join carries")
    prev = book.get_person(ns.name, fresh=True)
    if prev and not prev.get("closed") and not is_expected(prev):
        raise SlackError(f"#{key} is open already ({prev['name']}): nobody to expect")
    clash = next((r for r in book.all().values() if r["channel"]["key"] != key and not r.get("closed")
                  and str(r.get("email") or "").lower() == email.lower()), None)
    if clash:
        raise SlackError(f"{email} is #{clash['channel']['key']}'s already (`research msg chat list`)")
    live = prev if is_expected(prev) else None
    rec = {"schema": PERSON_SCHEMA, "name": (live or {}).get("name") or ns.name.capitalize(), "slack_user": None,
           "email": email, "note": ns.note or (live or {}).get("note") or f"asks the lanes in #{key}", "status": EXPECTED,
           "channel": {"key": key, "id": None}, "expected": (live or {}).get("expected") or _now_stamp(), "opened": None,
           "closed": None}
    check_collaborators({**reg, "collaborators": [*reg.get("collaborators", []),
                                                  {k: rec[k] for k in ("name", "slack_user", "note", "email")}]},
                        PEOPLE_DIR, {h["handle"] for h in reg["handles"]})
    if rec == prev:
        print(f"{PEOPLE_DIR}/{ns.name}.json: unchanged")
    elif ns.dry_run:
        print(f"plan: publish {PEOPLE_DIR}/{ns.name}.json to the notes: {json.dumps(rec, ensure_ascii=False)}")
    else:
        print(publish_route(notes_root(), rec, f"chat expect {ns.name}", PEOPLE_DIR, ns.name))
    print(f"{rec['name']} ({email}) is expected: the Slack events service opens #{key} when they join the workspace "
          f"(team_join); until it is deployed, run `research msg chat open {ns.name} --email {email}` once they have")
    return 0


def cmd_chat(ns: argparse.Namespace, reg: dict) -> int:
    """`chat open NAME`: one person's private channel #chat-NAME, with the person and Cursor's bot in it, the two-line
    welcome pinned and --start-here bookmarked.  The person is published to the notes as people.d/NAME.json, from which
    every command takes them into `collaborators` (`with_people`): no registry change.  A rerun reuses the channel and the
    record (after a Slack Connect invite is accepted, it records their Slack id).  `close NAME` archives the channel and
    marks the record closed; `list` reads people.d.  `expect NAME --email E` publishes someone Daniel invited to the
    workspace, with no channel yet: the Slack events service runs `chat open` for them when they join (`team_join`).  `reg`
    is the registry without people.d."""
    book = ChatPeople()
    if ns.verb == "list":
        for n in list(book.local):
            book.get_person(n, fresh=True)
        rows = book.all()
        for r in rows.values():
            state = f"closed {r['closed']}" if r.get("closed") else f"expected since {r['expected']}, no channel yet" \
                if is_expected(r) else f"open since {r['opened']}"
            print(f"#{r['channel']['key']} ({r['channel']['id'] or 'no channel'}): {r['name']} "
                  f"({r.get('slack_user') or 'no Slack id yet'}, {r.get('email') or 'no email'}), {state}")
        if not rows:
            print(f"no chat channels in {book.root / PEOPLE_DIR}")
        return 0
    if not ns.name or not CHAT_NAME_RE.match(ns.name):
        raise SlackError(f"chat {ns.verb} needs NAME, in lower case letters, digits and dashes: the channel is #{CHAT_PREFIX}NAME")
    key = CHAT_PREFIX + ns.name
    if key in reg["channels"]:
        raise SlackError(f"#{key} is a registry channel, not a chat: pick another NAME")
    if ns.verb == "expect":
        return chat_expect(ns, reg, book)
    api = client(ns)
    if api.via == "broker" and not api.dry_run:
        raise SlackError("chat needs SLACK_BOT_TOKEN itself (the broker relays no channel or invite calls): run it where the "
                         "token is")
    prev = book.get_person(ns.name, fresh=True)
    plan = "plan: " if api.dry_run else ""
    if ns.verb == "close":
        if not prev or prev.get("closed"):
            raise SlackError(f"#{key} is not open (`research msg chat list`)")
        if is_expected(prev):
            print(f"{plan}{prev['name']} is no longer expected (#{key} was never opened)")
        else:
            try:
                api.call("conversations.archive", channel=prev["channel"]["id"])
            except SlackError as e:
                if e.code != "already_archived":
                    raise
            print(f"{plan}archived #{key} ({prev['channel']['id']})")
        publish_person(api, {**prev, "closed": _now_stamp()}, f"chat close {ns.name}")
        return 0

    if bool(ns.user) == bool(ns.email):
        raise SlackError("chat open needs the person: --user U... (their Slack id) or --email E (their address), one of them")
    if ns.user and not SLACK_USER_RE.match(ns.user):
        raise SlackError(f"--user {ns.user!r} is not a Slack user id (U...)")
    if ns.email and not EMAIL_RE.match(ns.email.strip()):
        raise SlackError(f"--email {ns.email!r} is not an address")
    if ns.start_here and not re.match(r"^https?://\S+$", ns.start_here):
        raise SlackError(f"--start-here {ns.start_here!r} is not an http(s) link")
    if ns.cursor_bot and not SLACK_USER_RE.match(ns.cursor_bot):
        raise SlackError(f"--cursor-bot {ns.cursor_bot!r} is not a Slack user id (U...)")
    members = api.paged("users.list", "members")
    if ns.user:
        user = next((m for m in members if m.get("id") == ns.user and not m.get("deleted")), None)
        if not user and not api.dry_run:
            raise SlackError(f"no Slack user {ns.user} in the workspace (users.list): check the id, or pass --email")
        user = user or {"id": ns.user}
    else:
        try:
            user = api.call("users.lookupByEmail", email=ns.email.strip()).get("user")
        except SlackError as e:
            if e.code != "users_not_found":
                raise
            user = None
    uid = (user or {}).get("id")
    email = ns.email.strip() if ns.email else ((user or {}).get("profile") or {}).get("email") or (prev or {}).get("email")
    live = prev if prev and not prev.get("closed") else None
    if live and live.get("slack_user") and uid and live["slack_user"] != uid:
        raise SlackError(f"#{key} is {live['name']}'s ({live['slack_user']}): `chat close {ns.name}` first, or pick another NAME")
    name = (user or {}).get("real_name") or ((user or {}).get("profile") or {}).get("real_name") or (live or {}).get("name") \
        or ns.name.capitalize()
    rec = {"schema": PERSON_SCHEMA, "name": name, "slack_user": uid or (live or {}).get("slack_user"), "email": email,
           "note": ns.note or (prev or {}).get("note") or f"asks the lanes in #{key}",
           "channel": {"key": key, "id": None}, "opened": (live or {}).get("opened") or _now_stamp(), "closed": None}
    human = any(p.get("slack_user") == rec["slack_user"] for p in reg.get("humans", []) if rec["slack_user"])
    if not human:
        p = {k: rec[k] for k in ("name", "slack_user", "note")} | ({"email": email} if email else {})
        check_collaborators({**reg, "collaborators": [*reg.get("collaborators", []), p]}, PEOPLE_DIR,
                            {h["handle"] for h in reg["handles"]})

    cid, new = chat_channel(api, key, (prev or {}).get("channel", {}).get("id"))
    rec["channel"]["id"] = cid
    print(f"{plan}#{key} {cid}: " + ("created, private" if new else "reused"))
    bot = ns.cursor_bot or cursor_bot(members)
    if bot:
        print(f"{plan}Cursor's bot {bot}: {chat_invite(api, cid, bot)}")
    elif api.dry_run:
        print("plan: invite Cursor's bot, the users.list bot whose name says cursor (or --cursor-bot U...)")
    else:
        print(f"Cursor's bot: no bot named cursor in users.list, so subscriptions to #{key} wake no one until it is in: rerun "
              "with --cursor-bot U...")
    daniel = None
    if uid:
        print(f"{plan}{name} {uid}: {chat_invite(api, cid, uid)}")
    elif api.dry_run:
        print(f"plan: invite {email} if users.lookupByEmail finds them in the workspace, else through Slack Connect "
              "(conversations.inviteShared)")
    else:
        try:
            api.call("conversations.inviteShared", channel=cid, emails=email)
            print(f"{email} is not in the workspace: Slack Connect invited them to #{key}. Once they accept, rerun this "
                  "command to record their Slack id (until then their posts ring no one)")
        except SlackError as e:
            daniel = connect_step(e, email, key, reg)

    if new or not live or live["channel"]["id"] != cid:
        out = api.call("chat.postMessage", channel=cid, username=CHAT_WATCHER, text=chat_welcome(name),
                       unfurl_links="false", unfurl_media="false")
        api.call("pins.add", channel=cid, timestamp=out.get("ts") or "DRY-RUN")
        print(f"{plan}pinned the welcome")
    if ns.start_here:
        api.call("bookmarks.add", channel_id=cid, title="Start here", type="link", link=ns.start_here)
        print(f"{plan}bookmarked {ns.start_here}")
    if rec != prev:
        publish_person(api, rec, f"chat open {ns.name}")
    else:
        print(f"{PEOPLE_DIR}/{ns.name}.json: unchanged")
    if human:
        print(f"{name} is a named human in the registry: their posts in #{key} route as theirs anywhere")
    if daniel:
        print(f"Daniel: {daniel}. Then rerun this command to record their Slack id; everything else is done")
    return 0


# --- the app's configuration ------------------------------------------------------------------------------------------------

MANIFEST_REL = REGISTRY_REL.with_name("slack-app.json")
APP_STATE_ENV, APP_SEED_ENV = "RESEARCH_SLACK_APP_STATE", "SLACK_APP_CONFIG_REFRESH_TOKEN"
#: the app configuration API: these take the configuration token (rotating it, the refresh token), never the bot's scopes
APP_METHODS = ("apps.manifest.export", "apps.manifest.update", "apps.manifest.validate", "tooling.tokens.rotate")
ROTATE_BEFORE_S = 600                                            # a configuration token lives 12 h; rotate this long before its exp
APP_TOKENS_PAGE = "https://api.slack.com/apps"                   # "Your App Configuration Tokens": where a person generates a pair
MANIFEST_IGNORED = ("_metadata",)                                # the manifest format's version, which export adds


def _epoch() -> float:
    return time.time()


def app_state_path() -> Path:
    return Path(os.environ.get(APP_STATE_ENV) or Path.home() / ".research" / "slack-app-token.json")


def _write_private(path: Path, data: dict) -> None:
    """`data` as `path`, readable by its owner only, replaced atomically and synced: a reader sees the old pair or the new."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    d = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(d)
    finally:
        os.close(d)


def _app_state(path: Path) -> dict:
    if not path.exists():
        if seed := os.environ.get(APP_SEED_ENV, "").strip():
            return {"refresh_token": seed}
        raise SlackError(f"no app configuration token: {path} is absent and {APP_SEED_ENV} is unset. A person generates a pair at "
                         f"{APP_TOKENS_PAGE} (Your App Configuration Tokens); set {APP_SEED_ENV} to its refresh token, which "
                         f"seeds the file once, or {APP_STATE_ENV} to the state file")
    if path.stat().st_mode & 0o077:
        raise SlackError(f"{path} is readable by others: `chmod 600 {path}` (it holds the app's configuration tokens)")
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SlackError(f"{path}: {type(e).__name__}; it should be {{token, refresh_token, exp}} as rotation writes it") from None
    if not isinstance(state, dict) or not isinstance(state.get("refresh_token"), str) or not state["refresh_token"]:
        raise SlackError(f"{path} has no refresh_token; move it aside and set {APP_SEED_ENV} to a new one")
    return state


def app_token(path: Path | None = None, *, transport=None) -> str:
    """The app configuration token from its state file (`app_state_path`), rotated through tooling.tokens.rotate first when it
    is within ROTATE_BEFORE_S of its exp. A refresh token works once, so the new pair is written before the new token is used,
    under a lock so that two processes never spend the same one; an absent file is seeded once from $SLACK_APP_CONFIG_REFRESH_TOKEN."""
    path = path or app_state_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _app_state(path)
        if isinstance(state.get("token"), str) and state["token"] and _epoch() < float(state.get("exp") or 0) - ROTATE_BEFORE_S:
            return state["token"]
        try:
            out = Slack(None, direct=True, transport=transport).call("tooling.tokens.rotate", refresh_token=state["refresh_token"])
        except SlackError as e:
            if e.code is None:                                     # no answer: whether Slack spent the token is unknown
                raise SlackError(f"{e}; if Slack refuses the refresh token on the next try, it was spent: see the using-slack skill") from None
            raise SlackError(f"tooling.tokens.rotate: {e.code}: Slack refused the refresh token in "
                             f"{path if path.exists() else '$' + APP_SEED_ENV}. A refresh token works once, so an earlier "
                             f"rotation or another copy of it spent it. A person generates a new pair at {APP_TOKENS_PAGE} "
                             f"(Your App Configuration Tokens); then move {path} aside and set {APP_SEED_ENV} to the new refresh token") from None
        token, refresh, exp = out.get("token"), out.get("refresh_token"), out.get("exp")
        if not (isinstance(token, str) and token and isinstance(refresh, str) and refresh and isinstance(exp, (int, float))):
            raise SlackError(f"tooling.tokens.rotate: the answer lacks a token, refresh_token or exp, so {path} is unchanged "
                             "and its refresh token may be spent")
        _write_private(path, {"token": token, "refresh_token": refresh, "exp": int(exp), "rotated": int(_epoch())})
        return token


def manifest_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    for d in (Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents):
        if (d / MANIFEST_REL).is_file():
            return d / MANIFEST_REL
    raise SlackError(f"no {MANIFEST_REL} here or above: run from your verity checkout, or pass --manifest")


def _leaves(v: Any, at: str = ""):
    """(path, member, value) for each leaf: a list of strings (scopes, events, URLs) is a set whose members are leaves."""
    if isinstance(v, dict) and v:
        for k in sorted(v):
            if not (at == "" and k in MANIFEST_IGNORED):
                yield from _leaves(v[k], f"{at}.{k}" if at else k)
    elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
        for x in sorted(set(v)):
            yield at, x, None
    elif isinstance(v, list) and v:
        for i, x in enumerate(v):
            yield from _leaves(x, f"{at}[{i}]")
    else:
        yield at, None, v


def manifest_diff(repo: dict, live: dict) -> list[str]:
    """The live manifest against the repo's, one line per key path: `+` only the repo has it, `-` only the live app, `~` both
    with different values (live -> repo). A set's member is `path: member`.  A key one side lacks and the other sets to
    false is no difference: Slack exports its false defaults (pkce_enabled, is_mcp_enabled, ...), and the file omits them."""
    a = {(p, m): v for p, m, v in _leaves(live)}
    b = {(p, m): v for p, m, v in _leaves(repo)}
    for x, y in ((a, b), (b, a)):
        for k in [k for k, v in x.items() if v is False and k not in y]:
            del x[k]
    show = lambda k, v: f"{k[0]}: {k[1]}" if k[1] is not None else f"{k[0]}: {json.dumps(v, ensure_ascii=False)}"  # noqa: E731
    out = []
    for k in sorted(a.keys() | b.keys(), key=lambda k: (k[0], k[1] or "")):
        if k not in a:
            out.append("+ " + show(k, b[k]))
        elif k not in b:
            out.append("- " + show(k, a[k]))
        elif a[k] != b[k]:
            out.append(f"~ {k[0]}: {json.dumps(a[k], ensure_ascii=False)} -> {json.dumps(b[k], ensure_ascii=False)}")
    return out


def manifest_scopes(m: dict) -> dict[str, set[str]]:
    scopes = (m.get("oauth_config") or {}).get("scopes") or {}
    return {k: set(scopes.get(k) or ()) for k in ("bot", "user")}


def cmd_slack_app(ns: argparse.Namespace, reg: dict) -> int:
    """`export` the live manifest, `diff` the repo's against it (exit 1 when they differ), or `apply` the repo's: validate,
    then update, refused in --dry-run. Scopes the repo adds take effect only once a person approves them on the install page."""
    app = ns.app_id or reg.get("slack_app_id")
    if not app:
        raise SlackError("no Slack app id: the registry's slack_app_id is null. Record it there (the App ID on the app's Basic "
                         f"Information page at {APP_TOKENS_PAGE}) or pass --app-id")
    if not re.fullmatch(r"A[A-Z0-9]{4,30}", app):
        raise SlackError(f"--app-id {app!r} is not a Slack app id (A...)")
    path = manifest_path(ns.manifest) if ns.acmd != "export" else None
    try:
        repo = json.loads(path.read_text()) if path else None
    except (OSError, ValueError) as e:
        raise SlackError(f"{path}: {e}") from None
    api = Slack(app_token(), direct=True)
    live = api.call("apps.manifest.export", app_id=app).get("manifest")
    if not isinstance(live, dict):
        raise SlackError("apps.manifest.export: the answer has no manifest")
    if ns.acmd == "export":
        print(json.dumps(live, indent=2, ensure_ascii=False))
        return 0
    lines = manifest_diff(repo, live)
    print(f"--- live (app {app})\n+++ repo ({path})")
    print("\n".join(lines) if lines else "no drift: the live manifest is the repo's")
    if ns.acmd == "diff":
        return 1 if lines else 0
    if not lines:
        return 0
    api.call("apps.manifest.validate", app_id=app, manifest=repo)
    print("validate: Slack accepts the repo's manifest")
    have, want = manifest_scopes(live), manifest_scopes(repo)
    moved = [f"{k} {s}{x}" for k in ("bot", "user") for s, xs in (("+", want[k] - have[k]), ("-", have[k] - want[k])) for x in sorted(xs)]
    approve = lambda why: (f"{why}: Slack requires a person to approve new scopes. Daniel: open "  # noqa: E731
                           f"https://api.slack.com/apps/{app}/install-on-team and click Allow")
    if ns.dry_run:
        print(f"dry-run: would call apps.manifest.update for {app} with {path}; nothing changed")
        if moved:
            print("then " + approve(f"scopes change ({', '.join(moved)})"))
        return 0
    out = api.call("apps.manifest.update", app_id=app, manifest=repo)
    print(f"updated {app} to {path}")
    if moved or out.get("permissions_updated"):
        print(approve(f"scopes change ({', '.join(moved)})" if moved else "Slack says the permissions changed"))
    return 0


# --- #ask-daniel ------------------------------------------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(UTC)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _clock() -> float:
    return time.monotonic()


def pacific(t: datetime) -> str:
    """How a person reads a time: `3:00 PM PDT, Wed 30 Sep`."""
    return t.astimezone(PACIFIC).strftime("%I:%M %p %Z, %a %d %b").lstrip("0")


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        return now.astimezone(PACIFIC).replace(hour=hour, minute=minute, second=0, microsecond=0).astimezone(UTC)
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        raise SlackError(f"--deadline {raw!r}: give `3:00 PM` (today, Pacific) or an ISO time such as 2026-09-30T23:00:00Z") from None
    if t.tzinfo is None:
        raise SlackError(f"--deadline {raw!r}: an ISO time needs its zone (Z, or an offset such as -07:00)")
    return t.astimezone(UTC)


def card(ns: argparse.Namespace, reg: dict, now: datetime) -> dict:
    """The card's fields, checked as the route checks them."""
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
                           "Status": {"status": {"name": PAGE_WAITING}},
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


def wait_for(get: Callable[[], dict], *, sleep: Callable[[float], None], every: float, polls: int,
             max_s: float = math.inf, clock: Callable[[], float] = time.monotonic) -> tuple[dict, bool]:
    """GET a card until it leaves `pending` (decided, or a blocking card overdue), at most `polls` times, `every` s apart, for at
    most `max_s` of `clock` time; and whether it stopped at `max_s` with the card still pending."""
    start = clock()
    for i in range(max(1, polls)):
        out = get()
        if out.get("status") != "pending" or i == polls - 1:
            return out, False
        if (left := max_s - (clock() - start)) <= 0:
            return out, True
        sleep(min(every, left))
    return out, False


def _await_card(ns: argparse.Namespace, reg: dict, route: "AskDaniel", cid: str, polls: int) -> int:
    """`ask-daniel ... --wait`: poll card `cid` until it is decided, `polls` polls or WAIT_CAP_S run out; a card still pending at
    the cap exits STILL_GOING with the command that reads it later (research.waitcap)."""
    out, capped = wait_for(lambda: route.call("GET", "/" + urllib.parse.quote(cid, safe="")), sleep=_sleep, every=ns.every,
                           polls=polls, max_s=waitcap.WAIT_CAP_S, clock=_clock)
    ring_decision(ns, reg, cid, out)
    rc = _show(out, ns.json)
    if not capped:
        return rc
    print(waitcap.still_going(f"card {cid} (pending)", f"research msg ask-daniel status {cid}", waitcap.WAIT_CAP_S,
                              end="End your turn now; a subscription to the card's thread (subscribe_slack_thread) wakes you "
                                  "when Daniel answers."), file=sys.stderr)
    return waitcap.STILL_GOING


def ring_decision(ns: argparse.Namespace, reg: dict, cid: str, out: dict) -> None:
    """Once a card is decided, ring @top's inbox with its question, outcome and asker, so the decision reaches Daniel's
    log without an issue filed for it.  Once per card: a ring already in the inbox isn't repeated."""
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
    return (f"ask-daniel {verb} {cid}: HTTP 404: no card has that ID. It is the UUID `ask-daniel` printed first when "
            "it posted the card (or a unique prefix of it), which its mention in the card's thread also ends with (`research msg "
            "read --channel ask-daniel --thread TS`), not the card's Slack ts; for a card posted any other way, record Daniel's thread reply as a ruling")


CARD_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
CARD_PREFIX_MIN, CARD_LOOKBACK_S = 6, 14 * 86400


def card_id(ns: argparse.Namespace, reg: dict, cid: str) -> str:
    """The card's whole ID from a unique prefix of it (posts show 8 hex): the first card in #ask-daniel's last
    CARD_LOOKBACK_S, newest first, whose thread's mention carries an ID starting with it.  A whole ID, or a dry run, is kept."""
    cid = (cid or "").strip().lower()
    if CARD_ID_RE.fullmatch(cid) or not re.fullmatch(rf"[0-9a-f-]{{{CARD_PREFIX_MIN},35}}", cid):
        return cid
    api = client(ns)
    if api.dry_run:
        return cid
    ch = channel_id(ns, reg, "ask-daniel")
    for root in api.paged("conversations.history", "messages", channel=ch, oldest=f"{_now().timestamp() - CARD_LOOKBACK_S:.6f}"):
        if not root.get("reply_count"):
            continue
        for m in api.paged("conversations.replies", "messages", channel=ch, ts=root["ts"], limit=3)[1:3]:
            if found := next((u for u in CARD_ID_RE.findall(m.get("text") or "") if u.startswith(cid)), None):
                return found
    raise SlackError(f"ask-daniel: no card in #ask-daniel's last {CARD_LOOKBACK_S // 86400} days has an ID starting {cid}; "
                     "pass the whole ID `ask-daniel` printed")


def cmd_ask_daniel(ns: argparse.Namespace, reg: dict) -> int:
    if ns.verb == "pages":
        return cmd_pages(ns, reg)
    if ns.verb:
        if not ns.id:
            raise SlackError(f"ask-daniel {ns.verb} needs the card's ID")
        ns.id = card_id(ns, reg, str(ns.id))
        path = "/" + urllib.parse.quote(str(ns.id), safe="")
        if ns.verb == "resolve":
            if not ns.choice:
                raise SlackError("ask-daniel resolve needs --choice VALUE: the option Daniel chose in the thread")
            provisional = ns.provisional or (ns.note or "").startswith("Provisionally accepted")
            body = {"choice": ns.choice, **({"note": ns.note} if ns.note else {}), **({"provisional": True} if provisional else {})}
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
        polls = math.ceil((DEADLINE_MAX + WAIT_GRACE).total_seconds() / ns.every) if ns.wait else 1
        try:
            return _await_card(ns, reg, AskDaniel(), str(ns.id), polls)
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
    if ns.dry_run:
        print(f"dry-run: POST {ASK_DANIEL} {json.dumps(body, sort_keys=True, ensure_ascii=False)}", file=sys.stderr)
        return 0
    route = AskDaniel()
    try:
        out = route.call("POST", "", body)
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
    print(f"posted; now subscribe_slack_thread(channelId={ch}, threadTs={ts}, expiresInSeconds={EXPIRES_S})", file=sys.stderr)
    if not ns.wait:
        return 0
    polls = math.ceil((datetime.fromisoformat(body["deadline"]) - now + WAIT_GRACE).total_seconds() / ns.every)
    return _await_card(ns, reg, route, str(cid), polls)


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
    p.add_argument("--report", action="store_true", help="a program's report (drive, the merge train), held to no length")
    p.add_argument("--to", nargs="+", action="extend", default=[], metavar="@NAME", help="who it's for: handles or names with an inbox")
    p.add_argument("--thread", default=None, metavar="TS", help="reply in this thread (default: start one)")
    p.add_argument("--kind", default=None, metavar="WORD",
                   help="a word the footer carries (ask, the default top-level with --to; announce; "
                        "fyi, which implies --quiet; ...)")
    p.add_argument("--verify", default=None, metavar="CMD", help="announce: a required action, and the command that shows it took")
    p.add_argument("--ships", action="append", default=[], metavar="URL#sha256=HEX|git:SHA", help="announce: code it runs, pinned")
    p.add_argument("--quiet", action="store_true", help="wake only the names it starts with: no reply needed")
    p.add_argument("--urgent", action="store_true", help="the doorbells say URGENT, and `read --as` lists them first")
    def issue_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--parent", default=None, metavar="ISSUE",
                       help="the ask's issue is a sub-issue of ISSUE (CMP-92), on its project and milestone")
        p.add_argument("--title", default=None, metavar="TEXT",
                       help="the issue's title (default: the body's first line, without its ring tags and signature)")
    issue_flags(p)
    p = cmd("close", "close a thread you were asked in: a status line (done, or --declined WHY), and the root's reaction",
            as_=True, thread=True)
    p.add_argument("--declined", default=None, metavar="WHY")
    p.add_argument("--link", default=None, metavar="URL", help="the result: a PR, file or art: id")
    p.add_argument("--note", default=None)
    def card_flags(p: argparse.ArgumentParser) -> None:
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

    p = cmd("ask", "1:1: an issue in the owner's Linear team, tagged in #agent-coordination with its doorbell; to @daniel, "
                   "the ask's page in the Asks for Daniel database and its #ask-daniel card (the card flags, as ask-daniel --page)",
            as_=True, body=True)
    p.add_argument("--to", required=True, action="append", metavar="@OWNER", help="the one owner (several: `send --to @a @b`)")
    p.add_argument("--urgent", action="store_true", help="the doorbell says URGENT")
    issue_flags(p)
    card_flags(p)
    p.set_defaults(verb=None, id=None, link=None, wait=False, every=WAIT_EVERY_S, json=False)
    p = cmd("linear", "Linear issues, relations, projects and milestones, through the app (never the Linear MCP connector, "
                      "which acts as Daniel): `check` (what "
                      "the credentials reach; exit 0 when fit for asks), `state ISSUE STATE` (by name; " + LINEAR_STATES + "), "
                      "`comment ISSUE TEXT --as HANDLE` (a lane's worker: `--worker NAME`), `list TEAM [--state S]` (open by "
                      "default), `audit [TEAM ...] [--lane HANDLE]` (each open issue off its board or stale, with its fix; exit "
                      "1 on any), `place ISSUE PROJECT [--milestone M]` (by name prefixes), `edit ISSUE [--title T] "
                      "[--due YYYY-MM-DD|none] [--file BRIEF.md] [--parent ID]`, `relate ISSUE blocks|blocked-by|related|"
                      "duplicate OTHER` (Linear moves a duplicate to its Duplicate status), `unrelate ISSUE OTHER`, `update "
                      "PROJECT --health onTrack|atRisk|offTrack --body-file F` (a project update), `project create NAME --team "
                      "KEY... [--lead USER] [--target YYYY-MM-DD] [--description-file F]`, `milestone create|edit PROJECT NAME "
                      "[--date YYYY-MM-DD] [--description-file F] [--rename NEW]`; none sends a state")
    p.add_argument("verb", choices=("check", "state", "comment", "list", "audit", "place", "edit", "relate", "unrelate", "update",
                                    "project", "milestone"))
    p.add_argument("target", nargs="?", help="state, comment, place, edit, relate, unrelate: the issue (INF-12); list, audit: the "
                                             "team (a handle or its key); update: the project; project, milestone: create or edit")
    p.add_argument("value", nargs="*", help="state: the state's name; comment: the text; place: the project; audit: more teams; "
                                            "relate: the kind and the other issue; unrelate: the other issue; project create: "
                                            "its name; milestone: the project (one word or quoted) and the milestone's name")
    p.add_argument("--state", default=None, metavar="NAME", help="list: only issues in this state (default: every open one)")
    p.add_argument("--write", default=None, metavar="TEAM", help="check: also file, comment on and cancel a test issue in TEAM")
    p.add_argument("--as", dest="handle", default=None, metavar="HANDLE",
                   help="comment: its sender, a handle you hold (or --worker); check: the test issue's sender (default comms)")
    p.add_argument("--worker", default=None, metavar="NAME",
                   help="comment: for a lane's workers, agents that hold no handle (never with --as): it shows as LANE/NAME, "
                        "LANE the handle of the issue's team, and rings no one")
    p.add_argument("--title", default=None, metavar="T", help="edit: the issue's new title")
    p.add_argument("--due", default=None, metavar="YYYY-MM-DD", help="edit: its due date (`none` clears it)")
    p.add_argument("--file", "--body-file", "--description-file", dest="file", default=None, metavar="F.md",
                   help="markdown: edit: the issue's new description; update: the update's body; project create, milestone: "
                        "the description")
    p.add_argument("--parent", default=None, metavar="ID", help="edit: make it a sub-issue of this issue (CMP-92)")
    p.add_argument("--health", default=None, metavar="H", help="update: onTrack, atRisk or offTrack")
    p.add_argument("--team", action="append", default=[], metavar="KEY", help="project create: a team it is in (repeat for more)")
    p.add_argument("--lead", default=None, metavar="USER",
                   help="project create: its lead, a prefix of a Linear user's name (a person: a lane is its --team)")
    p.add_argument("--date", "--target", dest="date", default=None, metavar="YYYY-MM-DD",
                   help="milestone: its target date; project create (--target): the project's (`none` clears it)")
    p.add_argument("--rename", default=None, metavar="NEW", help="milestone edit: the milestone's new name")
    p.add_argument("--lane", default=None, metavar="HANDLE", help="audit: only this lane's findings (`-`: those with no lane)")
    p.add_argument("--milestone", default=None, metavar="M", help="place: the project's milestone, a prefix of its name (M3)")
    p.add_argument("--json", action="store_true", help="audit: the findings and counts as JSON")
    p.add_argument("--now", default=None, metavar="EPOCH", help="audit: the time it audits at, in seconds since the epoch (default: now)")
    p = cmd("announce", "1:some or 1:all in #agent-coordination", as_=True, body=True)
    p.add_argument("--to", nargs="+", action="extend", default=[], metavar="@HANDLE", help="the handles or names it's for (default: every handle)")
    p.add_argument("--verify", default=None, metavar="CMD", help="a required action: the command that shows it took")
    p.add_argument("--ships", action="append", default=[], metavar="URL#sha256=HEX|git:SHA", help="code it runs, pinned")
    p.add_argument("--urgent", action="store_true", help="the doorbells say URGENT")
    p = cmd("reply", "a post in a thread; `@name` at its start also rings that name's doorbell", as_=True, thread=True, body=True)
    p.add_argument("--quiet", action="store_true", help="ring only the names it starts with: no reply needed")
    p.add_argument("--urgent", action="store_true", help="the doorbells say URGENT")
    p.add_argument("--report", action="store_true", help="a program's report (drive, the merge train), held to no length")
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
    p = cmd("traffic", "the monitor: doorbells per inbox, pauses, dropped names, unanswered asks")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--stale-h", type=float, default=2,
                   help="an ask its record doesn't show answered after this many hours is unanswered: its Linear issue still "
                        "in Backlog with no comment but the asker's, Daniel's page still Waiting on Daniel, or (no record) no "
                        "👀, ✅, ❌ or reply; one whose record can't be read is unchecked")
    p.add_argument("--latency", action="store_true", help="instead: per lane (and @daniel), its asks, how many its record shows "
                                                         "answered (Linear, Notion, or Slack where there's no record), and the "
                                                         "median and p90 minutes to the first answer")
    p.add_argument("--volume", action="store_true", help="instead: per poster in #agent-coordination, its posts and their median, "
                                                        f"p90 and most words, and those over {VOLUME_WORDS} words; and the "
                                                        "threads with the most replies")
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
    p = cmd("deliver", "ring the doorbells for a post no client delivered (a person's typed post); --channel ask-the-team "
                       "or chat-NAME: a post there, whose doorbells name the channel (a chat's carry no text; one naming no "
                       "one rings @top)")
    p.add_argument("--ts", default=None, help="the post's ts (default: the event's)")
    p.add_argument("--event", default=None, metavar="JSON|@PATH|-",
                   help="the Slack message event, as a watcher gets it: the post's channel, ts and thread, and its text where "
                        "the bot can't read the channel")
    p.add_argument("--thread", default=None, metavar="TS", help="its thread's root ts, when it is a reply")
    p = cmd("ring-run", "ring the launching lane's inbox when a run has ended (the control pod calls it)")
    p.add_argument("--run", required=True, metavar="RUN_ID")
    p.add_argument("--rc", required=True, help="the run's exit code, or its verdict word")
    p.add_argument("--agent", default=None, metavar="BC_ID|URL", help="the launching agent (request.json's submitted_by)")
    p.add_argument("--to", default=None, metavar="@LANE", help="the lane to ring, when the agent holds no handle (a worker)")
    p.add_argument("--text", default=None, help="one line: what finished, or where its result is")
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
    card_flags(p)
    p.add_argument("--link", default=None, metavar="NOTION_URL", help="the ask's page, made with the Notion MCP, in place of --page")
    p.add_argument("--hours", type=float, default=24, metavar="H", help="pages: edited in the last H hours (default 24)")
    p.add_argument("--wait", action="store_true",
                   help=f"poll until the card is decided (or overdue); exit 0 once decided, for at most {waitcap.WAIT_CAP_S} s (no tool "
                        f"call blocks longer than 20 minutes): a card still pending then exits {waitcap.STILL_GOING}, "
                        "naming `ask-daniel status ID` to check back")
    p.add_argument("--every", type=float, default=WAIT_EVERY_S, metavar="S", help=f"seconds between polls (default {WAIT_EVERY_S})")
    p.add_argument("--choice", default=None, metavar="VALUE", help="resolve: the option Daniel chose in the thread")
    p.add_argument("--note", default=None, help="resolve: his words, or a link to them")
    p.add_argument("--provisional", action="store_true",
                   help="resolve: the lane decided it under the standing rule, so the card says the lane decided, not "
                        "Daniel (implied by a --note starting 'Provisionally accepted')")
    p.add_argument("--json", action="store_true", help="status: the route's whole answer; pages: each page's fields")
    p = cmd("chat", "a person's private channel #chat-NAME: `open NAME` creates or reuses it, invites the person and Cursor's "
                    "bot, pins the welcome and publishes the person to the notes' people.d; `close NAME` archives it; `list`; "
                    "`expect NAME --email E` publishes someone invited to the workspace, whose chat opens when they join")
    p.add_argument("verb", choices=("open", "close", "list", "expect"))
    p.add_argument("name", nargs="?", metavar="NAME", help="open, close, expect: the channel is #chat-NAME")
    p.add_argument("--user", default=None, metavar="U...", help="open: the person's Slack id, in the workspace")
    p.add_argument("--email", default=None, metavar="E", help="open: the person's address (users.lookupByEmail, else Slack "
                   "Connect); expect: the address Daniel invited")
    p.add_argument("--start-here", default=None, metavar="URL", help="open: a link to bookmark in the channel as Start here")
    p.add_argument("--note", default=None, metavar="TEXT", help="open, expect: who they are (default: kept, or a stub)")
    p.add_argument("--cursor-bot", default=None, metavar="U...", help="open: Cursor's bot user (default: users.list's bot named cursor)")
    p = sub.add_parser("groups", help="the handles' Slack user groups")
    gsub = p.add_subparsers(dest="gcmd", required=True)
    gsub.add_parser("sync", parents=[c], help="create or update a user group per handle; a plan unless --apply").add_argument("--apply", action="store_true")
    p = sub.add_parser("slack-app", help=f"the Slack app's configuration: {MANIFEST_REL} against the live app")
    asub = p.add_subparsers(dest="acmd", required=True)
    for name, help in (("export", "print the live manifest as JSON"),
                       ("diff", "the repo's manifest against the live one, by key path; exit 1 when they differ"),
                       ("apply", "validate the repo's manifest, then update the app to it (--dry-run: validate and print the plan)")):
        q = asub.add_parser(name, parents=[c], help=help)
        q.add_argument("--app-id", default=None, metavar="ID", help="the app (default: the registry's slack_app_id)")
        q.add_argument("--manifest", default=None, metavar="PATH", help=f"the repo's manifest (default: the checkout's {MANIFEST_REL})")

    ns = ap.parse_args(argv)
    try:
        path = registry_path(ns.registry)
        reg = load_registry(path, people=ns.cmd not in ("groups", "chat"))     # these two never write people.d's into it
        if ns.cmd == "groups":
            return cmd_groups_sync(ns, reg, path)
        if ns.cmd == "slack-app":
            return cmd_slack_app(ns, reg)
        if ns.cmd == "chat":
            return cmd_chat(ns, reg)
        if ns.cmd in ROUTED or ns.cmd == "owners":                # who owns a path now is main's answer too
            reg = routing_registry(reg, path)
        return {"handles": cmd_handles, "owners": cmd_owners, "subscriptions": cmd_subscriptions, "ask": cmd_ask, "announce": cmd_announce, "reply": cmd_reply,
                "pickup": cmd_pickup, "roster": cmd_roster, "match": cmd_match,
                "verify-author": cmd_verify_author, "deliver": cmd_deliver, "ring-run": cmd_ring_run, "read": cmd_read, "ask-daniel": cmd_ask_daniel, "names": cmd_inbox,
                "traffic": cmd_traffic, "send": cmd_send, "close": cmd_close,
                "linear": cmd_linear, "retract": cmd_retract}[ns.cmd](ns, reg)
    except SlackError as e:
        print(f"{prog}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

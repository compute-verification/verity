"""`research msg` (also `research slack`): messages between agents, on Slack (the conventions are
`.agents/skills/using-slack/SKILL.md`).

A message is a post in #agent-coordination, sent by a name: a lead's handle in the registry (@infra, @proofs, @lean, ...), which
`names open` gives an inbox thread. Only leads get names; a worker is reached through its lead. A thread is a conversation. Who a message reaches is one rule (`deliveries`): the names that
lead it (`@a @b ...`), an announcement's audience, on a reply the thread's starter or the names it addressed, on a reply
under an announcement that names no one the owners of the PRs it names (`pr_watchers`), and every name whose subscriptions match it. Each delivery is a `wake` (a doorbell in that name's inbox thread, which its agent subscribes to)
or a `digest` (in that name's next `read --as`). The footer carries the post's kind and flags, which subscriptions filter on.
Notes are records, never messages.

  research msg send --as NAME [--to @A @B ...] [--thread TS] [--kind WORD] [--quiet] [--urgent] (--text T | --file PATH)
                    [--verify CMD] [--ships URL#sha256=HEX | git:SHA]...     # prints `<ts> <permalink>`; --kind announce: an announcement
  research msg close --as NAME --thread TS [--link URL] [--note T] [--declined WHY]   # ✅ or ❌: a status line, and the reaction on an ask
  research msg list [--as NAME] [--hours H] [--open] [--json]     # threads: when, kind, from, to, state
  research msg read --thread TS [--since TS] | --since TS | --as NAME [--peek] [--urgent] [--json]  # --as: unread doorbells, then the digest
  research msg names open|read|close|list [--as NAME] [--forward-to @NAME] [--succeeds]  # the names you answer to (`inbox` too)
  research msg subscribe --as NAME [--filter F [--deliver wake|digest] | --remove F]   # F: all, kind:K, from:@x, to:@x, thread:TS
  research msg traffic [--hours H] [--json]                        # the monitor: doorbells, pauses, dropped names, unanswered asks
  research msg watch --as NAME [--every S] [--polls N]             # block until a new doorbell; the hook a wake script runs
  research msg pickup --thread TS                                  # 👀 on an ask you own
  research msg retract --as NAME --ts TS [--thread ROOT] [--why W] # blank a post NAME sent (needs SLACK_BOT_TOKEN)
  research msg roster --thread TS [--json]                         # an announcement's status lines; exit 0 when all are done
  research msg handles [--json] | subscriptions --as HANDLE        # the registry; the subscribe calls a handle's holder makes
  research msg owners [PATH...]                                    # who owns each path (stdin: `git diff --name-only main... |`)
  research msg match --as HANDLE --event JSON|TEXT|@PATH|- [--why] # exit 0 addressed, 1 not, 2 error
  research msg verify-author --event JSON|@PATH|- [--why]          # exit 0 only for our bot, a channel's bot or a named human
  research msg deliver --ts TS [--thread T]                         # ring the doorbells for a post no client delivered
  research msg groups sync [--apply]                               # one user group per handle; a plan unless --apply
  research msg ask-daniel --as HANDLE --kind blocking|default --question Q --why W --option VALUE=LABEL... --recommend V
                          --default V --deadline '3:00 PM'|ISO --link URL [--wait]     # a card in #ask-daniel; prints `<id> <ts> <permalink>`
  research msg ask-daniel status ID [--wait] [--json] | resolve ID --choice V [--note T]

`ask`, `announce`, `reply`, `done` and `decline` are the older spellings of `send` and `close`, kept while callers move.

#ask-daniel's cards go through the site's route (ASK_DANIEL, Cursor OIDC with the route as audience), not the Slack API; a route
that refuses the card's fields (HTTP 400) gets the old Approve/Deny shape `{title, text, link, requested_by}` instead.

Every subcommand takes `--dry-run` (no request reaches Slack; each one it would send is printed, the token never),
`--registry PATH` (default $RESEARCH_SLACK_REGISTRY, else the checkout's) and `--channel KEY` (default agent-coordination);
a channel's id is $SLACK_CHANNEL_<KEY> (upper case, `-` as `_`) or the registry's, and `--channel-id` overrides both. The token
is $SLACK_BOT_TOKEN and is never printed; $SLACK_API_URL points the client at another Web API (the tests' fake). Stdlib only,
and it runs as a single file (`python3 slack.py ...`) from any checkout.

Without $SLACK_BOT_TOKEN (or with RESEARCH_SLACK_VIA=broker) every call goes through Verity's Slack broker instead: a POST of
`{"method", "args"}` to BROKER with this agent's Cursor OIDC token, answered with Slack's response unchanged. The broker holds the
bot token and allows only BROKER_METHODS; RESEARCH_SLACK_VIA=direct insists on the token.

Until the console site's events route delivers (`routes.d/router.json` says `"mode": "server"`), the sending CLI does: it rings
the doorbells right after posting.  In server mode each send first checks that the route answers (its `url`); when it
doesn't, the sender rings them itself and alerts the route's `owner`. `match` is the older, handle-only test of whether a post is addressed to HANDLE."""
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
DONE, DECLINED, TAKING = "\u2705", "\u274c", "\U0001f440"         # ✅ ❌ 👀
SHORTCODES = {ANNOUNCE: ":mega:", DONE: ":white_check_mark:", DECLINED: ":x:"}   # Slack stores a posted emoji as its shortcode
STATUS_RE = re.compile(rf"^((?:{DONE}|{SHORTCODES[DONE]}) done|(?:{DECLINED}|{SHORTCODES[DECLINED]}) declined)\b(?: \u2014 (.*?))?(?:\n|$)")
PIN_RE = re.compile(r"^(?:https?://\S+#sha256=[0-9a-f]{64}|git:[0-9a-f]{40})$")
SHIPS_CODE_RE = re.compile(r"\b(curl|wget|pip install|uv (?:pip|sync|tool)|git (?:pull|merge|checkout|fetch)|install|download|upgrade)\b", re.I)
SAFE_POINT = "Apply at your next safe point; don't interrupt running jobs."
EXPIRES_S = 2592000                                              # 30 days; a lapsed subscription wakes no one and says nothing
PAGE = 200
ROUTER_DOWN = "the router's events route doesn't answer"         # its owner's alert, by which the next sender sees it was sent
ROUTER_ALERT_S = 21600                                           # 6 hours: one alert per outage, again if it outlasts that
ROUTER_RECHECK_S = 300                                           # how stale a long-lived Routes' idea of the router may get
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
WAIT_EVERY_S = 30
WAIT_GRACE = timedelta(minutes=5)                                # the site's default cron runs every minute
ROUTES_DIR = "routes.d"                                          # in the notes: one `<name>.json` per inbox
ROUTES_RAW = "https://raw.githubusercontent.com/compute-verification/research-notes/main/routes.d/"
ROUTE_SCHEMA = "verity/slack-route/v1"
LEAD_RE = re.compile(r"^\s*(?:(?::[a-z0-9_+'-]+:|[\u2600-\u27bf\U0001f000-\U0001faff]\ufe0f?)\s*)?"   # one emoji may lead (`:mega: @a`)
                     r"((?:(?:@[a-z0-9][a-z0-9._-]*|<!subteam\^[A-Z0-9]+(?:\|@?[^>]*)?>)[\s,:]*)+)")
BARE_LEAD_RE = re.compile(r"^\s*([a-z0-9][a-z0-9._-]*):\s")
NAME_TOKEN_RE = re.compile(r"@([a-z0-9][a-z0-9._-]*)|<!subteam\^([A-Z0-9]+)(?:\|@?([^>]*))?>")
AGENT_ID_RE = re.compile(r"bc-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|bc-[0-9a-f]{8}")
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
DELIVER = ("wake", "digest")                                     # how a subscription delivers
FILTER_KEYS = ("kind", "from", "to", "thread")


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


def origin_groups(path: Path) -> set[str]:
    """The user-group ids in origin/main's registry (a best-effort fetch, then `git show`): an analysis checkout older than
    a sender's must not report the sender's group mentions as dropped (three false DROPPED lines on 14 Oct). Empty when git
    can't answer (no clone, no remote, offline), which keeps `traffic`'s old behavior."""
    try:
        cwd = path.resolve().parent
        subprocess.run(["git", "fetch", "origin", "main", "-q"], cwd=cwd, timeout=10, check=False, capture_output=True)
        out = subprocess.run(["git", "show", f"origin/main:./{path.name}"], cwd=cwd, timeout=10,
                             check=True, capture_output=True, text=True).stdout
        return {h["usergroup"] for h in json.loads(out).get("handles", []) if h.get("usergroup")}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return set()


def load_registry(path: Path) -> dict:
    try:
        reg = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SlackError(f"registry {path}: {e}") from e
    if reg.get("schema") != SCHEMA:
        raise SlackError(f"registry {path}: schema {reg.get('schema')!r}, expected {SCHEMA!r}")
    handles, owned = set(), {}
    for h in reg.get("handles", []):
        name = h.get("handle", "")
        if not NAME_RE.match(name) or name in handles:
            raise SlackError(f"registry {path}: bad or duplicate handle {name!r}")
        handles.add(name)
        for k in ("holder", "remit", "charter"):
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
    return reg


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
        try:
            out = self._broker(method, args) if self.via == "broker" else self.transport(
                self.base + method, urllib.parse.urlencode(args).encode(),
                {"Authorization": f"Bearer {self.token}", "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
        except urllib.error.HTTPError as e:
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
        items: list[dict] = []
        cursor = None
        while True:
            out = self.call(method, limit=PAGE, cursor=cursor, **params)
            items += out.get(key, [])
            cursor = (out.get("response_metadata") or {}).get("next_cursor")
            if not cursor or (limit and len(items) >= limit):
                return items[:limit] if limit else items

    def root(self, channel: str, thread: str) -> dict | None:
        msgs = self.call("conversations.replies", channel=channel, ts=thread, limit=1).get("messages", [])
        return next((m for m in msgs if m.get("ts") == thread), None)


def client(ns: argparse.Namespace) -> Slack:
    return Slack(os.environ.get("SLACK_BOT_TOKEN"), dry_run=ns.dry_run)


def _footer(tags: tuple[str, ...] = ()) -> str:
    """`_ask · urgent_`: what subscriptions filter on.  The author is the post's username, and its holder is in the directory."""
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
        first = api.call("conversations.replies", channel=ch, ts=thread, limit=1).get("messages", [])
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
        msg = {"text": text, "ts": ts, "thread_ts": thread, "bot_id": "self", "username": handle}
        try:
            deliver(api, route[0], route[1], ch, msg, link)
        except SlackError as e:
            print(f"route: posted, but no doorbell rang ({e}); the post itself stands", file=sys.stderr)
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


def mentions(text: str, h: dict) -> str | None:
    """Why `text` names the registry entry `h`, or None."""
    for gid, label in SUBTEAM_RE.findall(text or ""):
        if h.get("usergroup") and gid == h["usergroup"]:
            return f"user-group mention <!subteam^{gid}>"
        if label.lstrip("@") == h["handle"]:
            return f"user-group mention labelled @{h['handle']}"
    if re.search(rf"(?<![\w@./-])@{re.escape(h['handle'])}(?![\w-])", SUBTEAM_RE.sub(" ", text or "")):
        return f"plain-text @{h['handle']}"
    return None


def tagged(text: str, reg: dict) -> list[str]:
    return [h["handle"] for h in reg["handles"] if mentions(text, h)]


def is_announcement(msg: dict | None) -> bool:
    return bool(msg) and (msg.get("text") or "").startswith((ANNOUNCE, SHORTCODES[ANNOUNCE]))


def audience(text: str, reg: dict) -> list[str]:
    """The handles an announcement's header tags after ` to ` (the header also names its author); empty means every handle."""
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
        if why := mentions(text, h):
            return why
        if is_announcement(ev) and not audience(text, reg):
            return "an announcement to every handle"
        return None
    if not is_status(text) and (why := mentions(text, h)):
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
    if root and mentions(root.get("text", ""), h):
        return f"{started}: its root addresses @{handle}"
    if key and key != FRONT_DOOR and handle in subscribers(reg, key):
        return f"{started} in #{key}, which @{handle} subscribes to"
    return None


# --- routing: `@name` at the start of a post wakes that name's agent ---------------------------------------------------------
# The rule is `route_targets`, a pure function pinned by tests/slack_routing_vectors.json; the console site's events route
# implements the same rule against the same vectors.  Delivery is a doorbell: the post itself (`doorbell`), as a reply in the
# target's inbox thread, which its agent subscribes to.

def poster(msg: dict | None, reg: dict) -> str | None:
    """Who wrote `msg`, as a name: our bot's username, or `daniel`-style for a named human (first name, lower case)."""
    if not msg:
        return None
    if msg.get("bot_id"):
        return msg.get("username") or None
    return next((p["name"].split()[0].lower() for p in reg.get("humans", []) if p.get("slack_user") and msg.get("user") == p["slack_user"]), None)


def lead_names(text: str, reg: dict) -> list[str]:
    """The names a post addresses: the `@name`s and user-group mentions at its very start (after one emoji at most), before
    any other word, or a handle written bare as the first word with a colon (`lean: ...`)."""
    m = LEAD_RE.match(text or "")
    if not m and (bare := BARE_LEAD_RE.match(text or "")) and any(h["handle"] == bare[1] for h in reg["handles"]):
        return [bare[1]]
    names: list[str] = []
    for at, gid, label in NAME_TOKEN_RE.findall(m.group(1) if m else ""):
        n = at.rstrip("._-") if at else (label or "").lstrip("@") or next((h["handle"] for h in reg["handles"] if h.get("usergroup") == gid), "")
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


def branch_owners(branch: str, reg: dict) -> list[str]:
    """The handles whose agent made `branch`: its last 4 characters are the last 4 of the holder's agent id, or one of the
    handle's `branch_suffixes` (a holder whose PRs another agent of its own opens)."""
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
        return []                                                    # only our bot and the named humans route
    if is_quiet(text):
        out = [] if is_status(text) else lead_names(text, reg)       # people write (quiet) for "no reply needed"
    elif reply and is_status(text):
        out = [by]
    else:
        out = lead_names(text, reg)
        if not reply and is_announcement(msg):
            out += header_names(text, reg) or [h["handle"] for h in reg["handles"]]
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


def matches(filt: str, msg: dict, reg: dict, targets: list[str]) -> bool:
    """Whether `msg` passes a subscription filter: space-separated terms, all of which hold.  `all`; `kind:K`; `from:@x`;
    `to:@x` (x is among its targets); `thread:TS`.  An unknown term matches nothing, so a typo subscribes to nothing."""
    for term in (filt or "").split():
        key, _, val = term.partition(":")
        val = val.lstrip("@")
        if term == "all":
            continue
        if key == "kind" and kind_of(msg) == val or key == "from" and poster(msg, reg) == val or \
                key == "to" and val in targets or key == "thread" and (msg.get("thread_ts") or msg.get("ts")) == val:
            continue
        return False
    return bool((filt or "").split())


def deliveries(msg: dict, root: dict | None, reg: dict, book: dict[str, dict], inboxes: set[str] | frozenset = frozenset(),
               history: list[dict] | tuple = (), branches: dict[str, str] | None = None) -> list[dict]:
    """Who gets `msg`, and how: each of `route_targets` is woken (`wake`, why `addressed`); then each other name in `book`
    (name -> route) whose `subscriptions` match it, by its first matching subscription's `deliver` (`wake`: now; `digest`:
    when that name reads its digest).  A quiet post wakes only the names it starts with (or, naming no one, its `pr_watchers`), and still reaches
    digests.  Nothing in an inbox thread, and nothing by an unknown author, reaches anyone; the author never gets their own post."""
    targets = route_targets(msg, root, reg, inboxes, history, branches)
    out = [{"name": t, "deliver": "wake", "why": "addressed"} for t in targets]
    thread, text = msg.get("thread_ts") or msg.get("ts"), msg.get("text") or ""
    author = poster(msg, reg)
    if author is None or thread in inboxes or text.startswith(INBOX_HEAD):
        return out
    for name in sorted(book):
        r = book[name]
        if name == author or name in targets or r.get("closed"):
            continue
        for s in r.get("subscriptions") or []:
            mode = s.get("deliver", "digest")
            if mode in DELIVER and matches(s.get("filter", ""), msg, reg, targets) and not (mode == "wake" and is_quiet(text)):
                out.append({"name": name, "deliver": mode, "why": f"subscribes to {s['filter']}"})
                break
    return out


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


def route_answers(url: str, timeout: float = 10) -> str | None:
    """Why the router's events route at `url` doesn't answer (a 404, a 5xx, no answer at all), or None when it does: a
    deployed route answers an unsigned GET with a 405 or a 401."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout):
            return None
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}" if e.code == 404 or e.code >= 500 else None
    except (urllib.error.URLError, OSError) as e:
        return f"no answer ({getattr(e, 'reason', e)})"


class Routes:
    """The routes table, `routes.d/<name>.json` in the notes: the local clone's, and for a name it lacks (a clone not pulled
    since that agent registered) the published file, read from $RESEARCH_ROUTES_URL (default the notes repo's raw files,
    with RESEARCH_NOTES_TOKEN)."""

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
        self.router_down: tuple[str, str, str] | None = None

    def get_route(self, name: str, fresh: bool = False) -> dict | None:
        if not NAME_RE.match(name):
            return None
        if name in self.local and not fresh:
            return self.local[name]
        if name not in self._fetched or fresh:
            raw = self.get(f"{self.base}{name}.json")
            try:
                r = json.loads(raw) if raw else None
            except ValueError:
                r = None
            self._fetched[name] = r if _route_ok(r, name) else None
        return self._fetched[name] or self.local.get(name)

    def all(self) -> dict[str, dict]:
        return {**self.local, **{k: v for k, v in self._fetched.items() if v}}

    def server_delivers(self) -> bool:
        """Whether the console site's events route rings the doorbells: `routes.d/router.json` says `"mode": "server"` as
        published and in the clone, each that can be read, and its `url` answers.  Either saying otherwise means the sender
        rings them, so switching back to `cli` reaches every sender at once, a clone not yet pulled included; a route that
        doesn't answer leaves `router_down` = (url, why, owner) for `deliver` to alert its owner.  A process that holds one
        `Routes` for days (the monitors' relay, n1_alerts) reads all of this again every ROUTER_RECHECK_S."""
        now = _now().timestamp()
        if now - getattr(self, "_mode_at", -ROUTER_RECHECK_S) >= ROUTER_RECHECK_S:
            self._mode_at, self.router_down = now, None
            p, routers = self.root / ROUTES_DIR / "router.json", []
            for read in (lambda: self.get(f"{self.base}router.json"), lambda: p.read_bytes() if p.is_file() else None):
                try:
                    raw = read()
                    routers += [r if isinstance(r := json.loads(raw), dict) else {}] if raw else []
                except (OSError, SlackError, ValueError):
                    routers.append({})
            modes = [r.get("mode") for r in routers]
            self._mode = "server" if modes and all(m == "server" for m in modes) else "cli"
            url = next((r["url"] for r in routers if isinstance(r.get("url"), str)), None) if self._mode == "server" else None
            if url and (why := route_answers(url)):
                self._mode = "cli"
                self.router_down = (url, why, next((r["owner"] for r in routers if NAME_RE.match(str(r.get("owner")))), "top"))
        return self._mode == "server"

    def inboxes(self) -> set[str]:
        return {r["inbox_ts"] for r in self.all().values()}

    def resolve(self, name: str) -> tuple[str, dict | None]:
        """Follow a closed route's `forward_to` (at most 3 hops): the name that gets the doorbell, and its route."""
        r = self.get_route(name)
        for _ in range(3):
            if not r or not r.get("closed") or not r.get("forward_to"):
                break
            name, r = r["forward_to"], self.get_route(r["forward_to"])
        return name, r


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


DOORBELL_RE = re.compile(r"^from ([a-z0-9][a-z0-9._-]*)( \(URGENT\))?, thread (\d+\.\d+)(?:, reply (\d+\.\d+))?: ")


def doorbell(author: str, thread: str, ts: str, text: str, urgent: bool = False) -> str:
    """The post itself, so its reader can act without another read: `from infra, thread T[, reply R]: <text>`.  Never starts
    with `@`, so it can't route itself."""
    body = plain(text)
    if len(body) > DOORBELL_MAX:
        body = body[:DOORBELL_MAX].rstrip() + f"\u2026 (the rest: `research msg read --thread {thread}`)"
    where = f"thread {thread}" + (f", reply {ts}" if ts and ts != thread else "")
    return f"from {author}{' (URGENT)' if urgent else ''}, {where}: {body}"


def _from(text: str, author: str) -> bool:
    if m := DOORBELL_RE.match(text):
        return m[1] == author or (m[1] == "router" and f": paused @{author} " in text)
    return text.startswith((f"from {author}:", f"from {author} (", f"from router: paused {author} "))


def _mentions_thread(text: str, thread: str) -> bool:
    if m := DOORBELL_RE.match(text):
        return m[3] == thread
    return f"thread_ts={thread}" in text or f"/p{thread.replace('.', '')}" in text


def _rang(text: str, ts: str, link: str) -> bool:
    """Whether the doorbell `text` is for the post `ts` (at `link`)."""
    if m := DOORBELL_RE.match(text):
        return (m[4] or m[3]) == ts
    return link in text


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


def router_alert(api: Slack, ch: str, routes: Routes, author: str, src: str, ts: str, now: float) -> str:
    """Ring `router.json`'s owner, urgently, that its events route doesn't answer: once per ROUTER_ALERT_S in its inbox,
    whichever sender notices first.  The line `deliver` prints for it."""
    url, why, owner = routes.router_down or ("", "", "")
    to, r = routes.resolve(owner)
    down = f"{ROUTER_DOWN} ({why}), so this sender rang"
    if not r or to == author:
        return f"@{owner}: {down}" + ("" if r else "; no inbox to alert")
    seen = api.paged("conversations.replies", "messages", channel=ch, ts=r["inbox_ts"], oldest=f"{now - ROUTER_ALERT_S:.6f}")
    if any(ROUTER_DOWN in (m.get("text") or "") for m in seen):
        return f"@{to}: {down}; already alerted"
    api.call("chat.postMessage", channel=ch, thread_ts=r["inbox_ts"], username="router", unfurl_links="false",
             unfurl_media="false", text=doorbell("router", src, ts, f"{ROUTER_DOWN}: {url} answers {why}. Senders ring the "
                                                 "doorbells themselves meanwhile. Restore the route, or set routes.d/router.json's "
                                                 "mode to cli.", urgent=True))
    return f"@{to}: alerted, {down}"


def deliver(api: Slack, reg: dict, routes: Routes, ch: str, msg: dict, link: str, *, urgent: bool = False,
            now: float | None = None, log=None) -> list[str]:
    """Deliver the post `msg` (just posted at `link`): ring the doorbell of each name it wakes; a digest delivery waits for
    its reader.  One line per name saying what happened."""
    log = log or sys.stderr
    thread = msg.get("thread_ts") if msg.get("thread_ts") not in (None, msg.get("ts")) else None
    root = api.root(ch, thread) if thread else None
    for n in lead_names(msg.get("text", ""), reg):
        routes.get_route(n, fresh=n not in routes.local)            # a name new since the clone was pulled
    inboxes = routes.inboxes()
    now = _now().timestamp() if now is None else now
    history, branches = watch_context(api, ch, msg, root, now, log)
    if routes.server_delivers():
        names = route_targets(msg, root, reg, inboxes, history, branches)
        print(f"route: the router rings {' '.join('@' + n for n in names)}" if names else
              "route: this rings no one", file=log)
        return []
    author = poster(msg, reg) or "?"
    urgent = urgent or "urgent" in tags_of(msg.get("text", ""))
    src, lines = thread or msg.get("ts") or "", []
    if routes.router_down:
        lines.append(router_alert(api, ch, routes, author, src, msg.get("ts") or src, now))
    humans = {poster({"user": p.get("slack_user")}, reg) for p in reg.get("humans", [])}
    for d in deliveries(msg, root, reg, routes.all(), inboxes, history, branches):
        name = d["name"]
        if name in humans:
            continue                                                 # people read Slack itself
        if d["deliver"] == "digest":
            lines.append(f"@{name}: in its digest")
            continue
        to, r = routes.resolve(name)
        if not r or (r.get("closed") and not r.get("forward_to")):
            known = any(h["handle"] == name for h in reg["handles"])
            lines.append(f"@{name}: no inbox" + (" (top-level forwards it)" if known else
                         " -- nobody will see this; check the name (`research msg names list`)") if not r else
                         f"@{name}: finished, with no forward_to: message its coordinator instead")
            continue
        if to == author:
            continue
        rung = api.call("conversations.replies", channel=ch, ts=r["inbox_ts"], oldest=f"{now - 3600:.6f}", limit=PAGE).get("messages", [])
        rung = [m for m in rung if m.get("ts") != r["inbox_ts"]]
        if any(_rang(m.get("text") or "", msg.get("ts") or "", link) for m in rung):
            continue
        n = sum(_mentions_thread(m.get("text") or "", src) and _from(m.get("text") or "", author) for m in rung)
        if n >= CAP_PER_HOUR:                                        # a loop is one author; a roll call is many, once each
            if n == CAP_PER_HOUR:
                api.call("chat.postMessage", channel=ch, thread_ts=src, username="router", unfurl_links="false",
                         text=f"routing @{author} to {to} paused here for the hour ({CAP_PER_HOUR} doorbells) {QUIET}")
                api.call("chat.postMessage", channel=ch, thread_ts=r["inbox_ts"], username="router", unfurl_links="false",
                         text=f"from router, thread {src}: paused @{author} here for the hour ({CAP_PER_HOUR} doorbells)")
            lines.append(f"@{to}: paused (cap of {CAP_PER_HOUR} an hour from @{author} in this thread)")
            continue
        api.call("chat.postMessage", channel=ch, thread_ts=r["inbox_ts"], username="router", unfurl_links="false",
                 unfurl_media="false", text=doorbell(author, src, msg.get("ts") or src, msg.get("text", ""), urgent))
        lines.append(f"@{to}: rang" + (f" (for @{name}, finished)" if to != name else ""))
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
    name in `names`, or the router), a bot the channel lists (`bots`: alerting), or a named human (`humans`, by Slack user
    id). An entry without an id allows nobody."""
    if msg.get("bot_id"):
        handles = {h["handle"] for h in reg["handles"]} | set(names) | {"router"}
        if reg.get("bot_id") and msg["bot_id"] == reg["bot_id"] and msg.get("username") in handles:
            return f"Verity's bot as @{msg['username']}"
        for b in reg["channels"][key].get("bots", []):
            if b.get("bot_id") and msg["bot_id"] == b["bot_id"]:
                return f"#{key}'s bot {b['name']} ({b['bot_id']})"
        return None
    for p in reg.get("humans", []):
        if p.get("slack_user") and msg.get("user") == p["slack_user"]:
            return f"{p['name']} ({p['slack_user']})"
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
    if ns.json:
        print(json.dumps(reg, indent=2, ensure_ascii=False))
        return 0
    print(f"workspace {reg.get('workspace')}  bot {reg.get('bot_id') or 'id not recorded'}")
    for key, ch in reg["channels"].items():
        subs = ch["subscribers"] if ch["subscribers"] == EVERY else ", ".join("@" + s for s in ch["subscribers"])
        print(f"#{key:<20} {ch.get('id') or 'id not recorded':<12} subscribers: {subs}\n  {ch['purpose']}")
    for h in reg["handles"]:
        print(f"@{h['handle']:<10} {h.get('usergroup') or '-':<12} {h['holder']}\n  {h['remit']}\n  charter: {h['charter']}")
    for p in reg.get("humans", []):
        print(f"human {p['name']} ({p.get('slack_user') or 'user id not recorded'})")
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
              f"# an issue filed into or changing state in @{h['handle']}'s Linear team: the work item is its own doorbell")
        print(f'subscribe_linear_comment(options={{"scope": "team", "teamId": "{key}"}}, expiresInSeconds={EXPIRES_S})   '
              f"# a comment on an issue in @{h['handle']}'s Linear team")
    print(f"# and subscribe_slack_thread(channelId, threadTs, expiresInSeconds={EXPIRES_S}) for each thread @{h['handle']} "
          f"starts, replies in or picks up (not for a status line alone); renew on every wake")
    print("# but not an announcement's thread (the merge thread is one): there the doorbell rings for a post that names "
          f"@{h['handle']} at its start, or, naming no one, one of its PRs or a run it named beside one of its PRs; "
          "unsubscribe from any you hold")
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
    """The agents that hold `name`: the ids in its registry `holder`, and its open inbox's route holder (a successor's)."""
    try:
        ids = AGENT_ID_RE.findall(entry(reg, name).get("holder") or "")
    except SlackError:
        ids = []
    r = routes_of(ns).get_route(name.lstrip("@"))
    return list(dict.fromkeys(ids + ([r["holder"]] if r and r.get("holder") and not r.get("closed") else [])))


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
    held = f"held by {', '.join(ids)}" if ids else "held by no agent the registry or its route names"
    raise SlackError(f"--as {h['handle']}: @{h['handle']} is {held}, and this agent is {me or 'unknown (set RESEARCH_AGENT_URL)'}; "
                     f"post as your own handle (a successor takes a name over with `research msg names open --as "
                     f"{h['handle']} --succeeds`)")


def _route(ns: argparse.Namespace, reg: dict, ch: str) -> tuple[dict, Routes | None] | None:
    return (reg, routes_of(ns)) if ch == channel_id_for(reg, FRONT_DOOR) and not getattr(ns, "no_route", False) else None


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
    """The ask's body, filed as an issue in its owner's Linear team (`linear` in the registry) and ending with the issue's
    link; as it was when the owner has no team or Linear fails, so the ask still goes out on Slack."""
    if not (getattr(ns, "linear", False) or ASKS_ON_LINEAR) or ns.dry_run or not owner.get("linear"):
        return body
    from . import linear  # noqa: PLC0415

    issue, why = linear.ask_issue(owner["linear"], sender, body)
    if not issue:
        print(f"research msg: not filed on Linear ({why}); asking on Slack only", file=sys.stderr)
        return body
    print(f"linear: {issue['identifier']} {issue['url']}")
    return f"{body}\nLinear: <{issue['url']}|{issue['identifier']}>"


def linear_close(text: str, state_type: str) -> None:
    """Move the Linear issue an ask's root links (`linear_ask`) to its team's first `state_type` state."""
    if not (m := LINEAR_LINK_RE.search(text or "")):
        return
    from . import linear  # noqa: PLC0415

    try:
        print(f"linear: {m[1]} {linear.Linear().close(m[1], state_type)}")
    except (linear.LinearError, OSError, ValueError) as e:
        print(f"research msg: {m[1]} stays open on Linear ({e}); the status line stands", file=sys.stderr)


def cmd_linear(ns: argparse.Namespace, reg: dict) -> int:
    from . import linear  # noqa: PLC0415

    try:
        lines, ok = linear.Linear().check(ns.write, name_entry(ns, reg, ns.handle)["handle"])
    except linear.LinearError as e:
        raise SlackError(str(e)) from None
    print("\n".join(lines))
    return 0 if ok else 1


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
    `--kind` rides in the footer, where subscriptions filter on it.  An announcement (`--kind announce`) gets its header,
    and with no `--to` it is for every handle."""
    h = acting_as(ns, reg, ns.handle)
    to = [name_entry(ns, reg, t, person_ok=True) for t in ns.to]
    kind = (ns.kind or ("ask" if to and not ns.thread else "")).strip().lower()
    if kind and not NAME_RE.match(kind) or kind in FLAGS:
        raise SlackError(f"--kind {ns.kind!r}: one lower-case word")
    if any(t["handle"] == h["handle"] for t in to):
        raise SlackError("--to names someone else")
    body = _body(ns)
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
    ch = channel_id(ns, reg, FRONT_DOOR) if not ns.thread else channel_id(ns, reg)
    quiet = ns.quiet or kind == "fyi"                                # an fyi wants no reply, so it wakes only its --to names
    tags = tuple(t for t in (kind if kind not in ("announce", "reply") else "", QUIET if quiet else "") if t)
    _post(client(ns), ch, h["handle"], text, ns.thread, route=_route(ns, reg, ch), urgent=ns.urgent, tags=tags)
    return 0


def cmd_close(ns: argparse.Namespace, reg: dict) -> int:
    if ns.declined:
        return _status(ns, reg, f"{DECLINED} declined", "x", ns.declined)
    return _status(ns, reg, f"{DONE} done", "white_check_mark", " ".join(x for x in (ns.link, ns.note) if x))


def thread_state(m: dict) -> str:
    """An ask's state from its root alone: the owner's reaction, else whether anyone replied."""
    names = {x.get("name") for x in m.get("reactions", [])}
    return "done" if "white_check_mark" in names else "declined" if "x" in names else "seen" if "eyes" in names else \
        "answered" if m.get("reply_count") else "open"


def cmd_list(ns: argparse.Namespace, reg: dict) -> int:
    """Threads started in the front door over the last --hours, newest first: when, kind, from, to, state, replies.  With
    --as, only those its name started or that reach it."""
    api, ch, routes = client(ns), channel_id(ns, reg, FRONT_DOOR), routes_of(ns)
    since = _now().timestamp() - ns.hours * 3600
    inboxes, book, me = routes.inboxes(), routes.all(), (ns.handle or "").lstrip("@") or None
    rows = []
    for m in api.paged("conversations.history", "messages", channel=ch, oldest=f"{since:.6f}"):
        if m.get("ts") in inboxes or (m.get("text") or "").startswith(INBOX_HEAD):
            continue
        reach = deliveries(m, None, reg, book, inboxes)
        author = poster(m, reg)
        if me and author != me and me not in {d["name"] for d in reach}:
            continue
        state = "-" if is_announcement(m) else thread_state(m)
        if ns.open and state not in ("open", "seen", "answered", "-"):
            continue
        rows.append({"ts": m["ts"], "kind": kind_of(m), "from": author, "to": [d["name"] for d in reach if d["why"] == "addressed"],
                     "state": state, "replies": m.get("reply_count", 0), "first": " ".join(DATE_TOKEN_RE.sub(r"\1", m.get("text") or "").split("\n", 1)[0].split())[:100]})
    if ns.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    for r in rows:
        to = " ".join("@" + t for t in r["to"]) or "-"
        print(f"{r['ts']} {both(r['ts'])} {r['kind']:<9} @{r['from'] or '?'} -> {to}  [{r['state']}, {r['replies']} replies]  {r['first']}")
    if not rows:
        print(f"no threads over the last {ns.hours:g} h" + (f" for @{me}" if me else ""))
    return 0


def cmd_subscribe(ns: argparse.Namespace, reg: dict) -> int:
    """A name's subscriptions, in its route: list them, add `--filter F` (delivered by `--deliver`), or `--remove F`."""
    routes, name = routes_of(ns), ns.handle.lstrip("@")
    if ns.filter or ns.remove:
        acting_as(ns, reg, name)
    r = routes.get_route(name, fresh=name not in routes.local)
    if not r:
        raise SlackError(f"@{name} has no inbox; open one with `research msg names open --as {name}`")
    subs = list(r.get("subscriptions") or [])
    if ns.filter or ns.remove:
        filt = " ".join((ns.filter or ns.remove).split())
        if ns.filter and not all(t == "all" or t.partition(":")[0] in FILTER_KEYS and t.partition(":")[2].lstrip("@") for t in filt.split()):
            raise SlackError(f"--filter {filt!r}: terms are all, kind:K, from:@x, to:@x, thread:TS")
        subs = [s for s in subs if s.get("filter") != filt] + ([{"filter": filt, "deliver": ns.deliver}] if ns.filter else [])
        r = {**r, "subscriptions": subs}
        if not ns.dry_run:
            print(publish_route(routes.root, r, f"routes.d: @{name} " + (f"subscribes to {filt}" if ns.filter else f"drops {filt}")))
    for s in subs:
        print(f"@{name} {s.get('deliver', 'digest'):<6} {s['filter']}")
    if not subs:
        print(f"@{name} has no subscriptions (it gets what's addressed to it)")
    return 0


def cmd_reply(ns: argparse.Namespace, reg: dict) -> int:
    h = acting_as(ns, reg, ns.handle)
    ch = channel_id(ns, reg)
    body = _body(ns)
    refuse_unrouted(body, reg, h["handle"])
    body += f" {QUIET}" if ns.quiet else ""
    _post(client(ns), ch, h["handle"], body, ns.thread, route=_route(ns, reg, ch), urgent=ns.urgent)
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


def cmd_done(ns: argparse.Namespace, reg: dict) -> int:
    return _status(ns, reg, f"{DONE} done", "white_check_mark", " ".join(x for x in (ns.link, ns.note) if x))


def cmd_decline(ns: argparse.Namespace, reg: dict) -> int:
    return _status(ns, reg, f"{DECLINED} declined", "x", ns.why)


def cmd_roster(ns: argparse.Namespace, reg: dict) -> int:
    msgs = client(ns).paged("conversations.replies", "messages", channel=channel_id(ns, reg), ts=ns.thread)
    root = next((m for m in msgs if m.get("ts") == ns.thread), None)
    if not ns.dry_run and not is_announcement(root):
        raise SlackError(f"{ns.thread} is not an announcement")
    aud = audience((root or {}).get("text", ""), reg) or [h["handle"] for h in reg["handles"]]
    expected = [n for n in aud if n != (root or {}).get("username")]
    status: dict[str, dict] = {}
    for m in msgs:
        if m.get("ts") != ns.thread and m.get("username") in expected and m.get("bot_id") and (mt := STATUS_RE.match(m.get("text", ""))):
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
    """Ring the doorbells for a post already in the channel that no client delivered: a person's typed post, while the
    router isn't live.  Safe to repeat: a name with this post's doorbell isn't rung again."""
    api, ch = client(ns), channel_id(ns, reg, FRONT_DOOR)
    msg = fetch_message(api, ch, ns.ts, ns.thread)
    if not msg:
        raise SlackError(f"no message {ns.ts} in the channel")
    msg = {**msg, "thread_ts": msg.get("thread_ts") or ns.thread}
    link = api.call("chat.getPermalink", channel=ch, message_ts=ns.ts).get("permalink") or "-"
    routes = routes_of(ns)
    routes._mode = None                                              # the router may be the very thing missing
    deliver(api, reg, routes, ch, msg, link)
    return 0


def cmd_verify_author(ns: argparse.Namespace, reg: dict) -> int:
    """Fail closed: exit 0 only when the message, fetched from Slack by channel and ts, has an author the registry names.
    With --dry-run the event's own fields are judged (a test of the allowlist, not of a real post)."""
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
        r = {**r, "closed": _now_stamp(), "forward_to": (ns.forward_to or "").lstrip("@") or None}
        if not ns.dry_run:
            print(publish_route(routes.root, r, f"routes.d: close @{name}" + (f" -> @{r['forward_to']}" if r["forward_to"] else "")))
        print(f"@{name} closed" + (f"; its doorbells ring @{r['forward_to']}" if r["forward_to"] else ""))
        return 0
    return inbox_read(ns, reg, r)


def inbox_open(ns: argparse.Namespace, reg: dict, routes: Routes, name: str) -> int:
    """Open `name`'s inbox, or say it is open already.  An open inbox held by another agent is refused unless --succeeds names
    it: the successor takes the same thread over, unread doorbells and all."""
    holder = ns.holder or (None if ns.dry_run else self_agent_id())
    if not holder and not ns.dry_run:
        raise SlackError("no agent id: pass --holder bc-... (or set RESEARCH_AGENT_URL)")
    old = routes.get_route(name, fresh=True)
    ch = channel_id(ns, reg, FRONT_DOOR)
    if old and not old.get("closed") and old.get("holder") == holder:
        print(f"@{name} is open already: inbox {old['inbox_ts']}")
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
        r = {"schema": ROUTE_SCHEMA, "name": name, "holder": holder, "channel": ch, "inbox_ts": ts, "opened": _now_stamp(),
             "closed": None, "forward_to": None, **({"remit": remit} if remit else {}),
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


def render_unread(name: str, msgs: list[dict]) -> str:
    if not msgs:
        return f"SLACK @{name}: nothing new"
    msgs = sorted(msgs, key=lambda m: ("(URGENT)" not in (m.get("text") or ""), float(m["ts"])))
    lines = [f"SLACK @{name}: {len(msgs)} new; answer with `research msg send --as {name} --thread T --text ...`"]
    lines += ["- " + plain(m.get("text") or "") for m in msgs]
    return "\n".join(lines)


def inbox_read(ns: argparse.Namespace, reg: dict, r: dict) -> int:
    r = {**r, "channel": r.get("channel") or channel_id(ns, reg, FRONT_DOOR)}
    api = client(ns)
    since = getattr(ns, "since", None)
    msgs = unread(api, r, since)
    dig = digest(api, reg, routes_of(ns), r, since) if any(s.get("deliver", "digest") == "digest" for s in r.get("subscriptions") or []) else None
    if ns.json:
        print(json.dumps(msgs if dig is None else {"doorbells": msgs, "digest": dig}, indent=2, ensure_ascii=False))
    else:
        print(render_unread(r["name"], msgs))
        if dig is not None:
            print(render_digest(r["name"], dig, reg))
    if not ns.peek:
        mark_read(r["name"], msgs)
        mark_read(f"{r['name']}.digest", dig or [])
    return 1 if ns.urgent and not any("(URGENT)" in (m.get("text") or "") for m in msgs) else 0


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
            print(render_unread(name, new))
            mark_read(f"{name}.watch", new)
            return 0
    return 1


DIGEST_FIRST_H = 24                                              # a first digest reaches back this far
DIGEST_THREADS_D = 7                                             # replies count in threads started this recently


def digest(api: Slack, reg: dict, routes: Routes, r: dict, since_ts: str | None = None) -> list[dict]:
    """The posts since `since_ts`, else since `r`'s last digest, that its `digest` subscriptions match (what's addressed to it
    came as doorbells)."""
    p = seen_path(f"{r['name']}.digest")
    since = float(since_ts) if since_ts else float(p.read_text().strip()) if p.is_file() else _now().timestamp() - DIGEST_FIRST_H * 3600
    ch, inboxes, book = r["channel"], routes.inboxes(), {r["name"]: r}
    picked = []

    def take(m: dict, root: dict | None) -> None:
        if any(d["name"] == r["name"] and d["deliver"] == "digest" for d in deliveries(m, root, reg, book, inboxes)):
            picked.append(m)

    for m in api.paged("conversations.history", "messages", channel=ch, oldest=f"{since - DIGEST_THREADS_D * 86400:.6f}"):
        if float(m["ts"]) > since:
            take(m, None)
        if m.get("reply_count") and float(m.get("latest_reply") or 0) > since and m["ts"] not in inboxes:
            for x in api.paged("conversations.replies", "messages", channel=ch, ts=m["ts"], oldest=f"{since:.6f}"):
                if x.get("ts") != m["ts"] and float(x["ts"]) > since:
                    take(x, m)
    return sorted(picked, key=lambda m: float(m["ts"]))


def render_digest(name: str, msgs: list[dict], reg: dict) -> str:
    if not msgs:
        return f"DIGEST @{name}: nothing new"
    lines = [f"DIGEST @{name}: {len(msgs)} post(s) your subscriptions match"]
    for m in msgs:
        thread = f" in {m['thread_ts']}" if m.get("thread_ts") not in (None, m.get("ts")) else ""
        first = " ".join(plain(m.get("text") or "").split("\n", 1)[0].split())[:150]
        lines.append(f"- {m['ts']}{thread} @{poster(m, reg) or '?'} {kind_of(m)}: {first}")
    return "\n".join(lines)


# --- traffic: the monitor ---------------------------------------------------------------------------------------------------

def cmd_traffic(ns: argparse.Namespace, reg: dict) -> int:
    """Doorbells per inbox (the most in any one hour), pauses, `@name`s nobody owns (dropped), and asks with no 👀 or reply
    after --stale-h hours, over the last --hours."""
    api, ch, routes = client(ns), channel_id(ns, reg, FRONT_DOOR), routes_of(ns)
    now = time.time() if ns.now is None else float(ns.now)
    since = now - ns.hours * 3600
    rows, counted = [], set()
    for r in sorted(routes.all().values(), key=lambda r: (bool(r.get("closed")), r["name"])):
        if not r.get("inbox_ts") or r["inbox_ts"] in counted:
            continue                                                 # a renamed name forwards into its successor's inbox
        counted.add(r["inbox_ts"])
        bells = [m for m in api.paged("conversations.replies", "messages", channel=r.get("channel") or ch, ts=r["inbox_ts"],
                                      oldest=f"{since:.6f}") if m.get("ts") != r["inbox_ts"] and m.get("username") == "router"]
        hours: dict[int, int] = {}
        for m in bells:
            hours[int(float(m["ts"]) // 3600)] = hours.get(int(float(m["ts"]) // 3600), 0) + 1
        rows.append({"name": r["name"], "doorbells": len(bells), "peak_per_hour": max(hours.values(), default=0),
                     "paused": sum((m.get("text") or "").startswith(("from router: paused", "from router, thread")) for m in bells)})
    posts = api.paged("conversations.history", "messages", channel=ch, oldest=f"{since:.6f}")
    handles = {h["handle"] for h in reg["handles"]} | {poster({"user": p.get("slack_user")}, reg) for p in reg.get("humans", [])}
    dropped, stale, groups = [], [], {h.get("usergroup") for h in reg["handles"]}
    fresh: set[str] | None = None                                    # origin/main's group ids, fetched on the first unknown
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
        is_ask = m.get("bot_id") and LEAD_RE.match(m.get("text") or "") and not is_announcement(m) and \
            not is_quiet(m.get("text") or "") and kind_of(m) != "fyi"           # older clients tagged an fyi without (quiet)
        answered = m.get("reply_count") or any(x.get("name") in ("eyes", "white_check_mark", "x") for x in m.get("reactions", []))
        if is_ask and not answered and now - float(m["ts"]) > ns.stale_h * 3600:
            stale.append({"ts": m["ts"], "by": poster(m, reg), "to": lead_names(m.get("text") or "", reg)})
    if ns.json:
        print(json.dumps({"inboxes": rows, "dropped": dropped, "unanswered": stale}, indent=2))
        return 0
    print(f"traffic over the last {ns.hours:g} h (to {both(now)})")
    for x in rows:
        print(f"  @{x['name']:<28} {x['doorbells']:>4} doorbells, at most {x['peak_per_hour']} in an hour" +
              (f", {x['paused']} paused" if x["paused"] else ""))
    for d in dropped:
        print(f"  DROPPED @{d['name']}: {d['ts']} by @{d['by']} (no inbox and no handle)")
    for s in stale:
        print(f"  UNANSWERED {s['ts']} by @{s['by']} to {' '.join('@' + t for t in s['to']) or 'no one it can ring'} ({both(s['ts'])})")
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
    live = {g.get("handle"): g for g in api.call("usergroups.list", include_disabled="true").get("usergroups", [])}
    for h in reg["handles"]:
        name, desc = h["handle"], h["remit"][:250]
        g = live.get(name)
        if g is None:
            print(f"create  @{name}")
            gid = (api.call("usergroups.create", name=name, handle=name, description=desc).get("usergroup") or {}).get("id") if write else None
        else:
            gid = g["id"]
            if g.get("date_delete"):
                print(f"enable  @{name} ({gid})")
                if write:
                    api.call("usergroups.enable", usergroup=gid)
            if g.get("name") != name or (g.get("description") or "") != desc:
                print(f"update  @{name} ({gid}): name/description")
                if write:
                    api.call("usergroups.update", usergroup=gid, name=name, handle=name, description=desc)
            elif not g.get("date_delete"):
                print(f"ok      @{name} ({gid})")
        if gid and h.get("usergroup") != gid:
            print(f"record  @{name} -> {gid} in {path}")
            h["usergroup"] = gid
            changed = True
    handles = {h["handle"] for h in reg["handles"]}
    if extra := [g for n, g in live.items() if n not in handles and not g.get("date_delete")]:
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
            "--default": ns.default, "--deadline": ns.deadline, "--link": ns.link}
    if missing := [k for k, v in need.items() if not v]:
        raise SlackError(f"ask-daniel needs {', '.join(missing)}")
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
    if not ns.link.startswith("https://"):
        raise SlackError(f"--link {ns.link!r}: an https URL to the full context")
    url = ns.agent_url or os.environ.get("RESEARCH_AGENT_URL")
    if not url and not ns.dry_run and (me := self_agent_id()):
        url = f"https://cursor.com/agents/{me}"
    if not url:
        raise SlackError("no --agent-url (or $RESEARCH_AGENT_URL): every card links the agent that asked it")
    return {"kind": ns.kind, "question": question, "why": ns.why.strip(), "options": options, "default": ns.default,
            "deadline": _iso(deadline), "link": ns.link, "requested_by": h["handle"], "agent_url": url}


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
    return (f"ask-daniel {verb} {cid}: HTTP 404: no card has that ID. It is the UUID `ask-daniel` printed first when it posted "
            "the card, not the card's Slack ts; for a card posted any other way, record Daniel's thread reply as a ruling")


def cmd_ask_daniel(ns: argparse.Namespace, reg: dict) -> int:
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
                return _show(AskDaniel().call("POST", f"{path}/resolve", body), ns.json)
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
            return _show(wait_for(lambda: route.call("GET", path), sleep=_sleep, every=ns.every, polls=polls), ns.json)
        except RouteError as e:
            if e.status == 404:
                raise SlackError(no_such_card("status", ns.id)) from None
            raise

    now = _now()
    body = card(ns, reg, now)
    acting_as(ns, reg, ns.handle)
    old = fallback(body)
    if ns.dry_run:
        print(f"dry-run: POST {ASK_DANIEL} {json.dumps(body, sort_keys=True, ensure_ascii=False)}", file=sys.stderr)
        print(f"dry-run: on HTTP 400, the old shape: POST {ASK_DANIEL} {json.dumps(old, sort_keys=True, ensure_ascii=False)}", file=sys.stderr)
        return 0
    route = AskDaniel()
    fell_back = False
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
    ch = out.get("channel") or channel_id_for(reg, ASK_CHANNEL)
    print(f"{cid} {ts} {permalink(reg, ch, ts)}")
    deciders = " ".join(f"<@{p['slack_user']}>" for p in reg.get("humans", []) if p.get("slack_user"))
    when = pacific(datetime.fromisoformat(body["deadline"]))
    rec = next((o["label"] for o in body["options"] if o.get("recommended")), "")
    ping = (f"{deciders} a decision for you: {body['question']} I recommend: {rec}. "
            + (f"`{body['default']}` applies at {when} unless you choose." if body["kind"] == "default" else f"Blocking: nothing happens until you choose (by {when})."))
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
    return _show(wait_for(lambda: route.call("GET", f"/{urllib.parse.quote(str(cid), safe='')}"), sleep=_sleep, every=ns.every,
                          polls=polls), ns.json)


def main(argv: list[str], prog: str = "research msg") -> int:
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--dry-run", action="store_true", help="send nothing to Slack; print each request instead")
    c.add_argument("--registry", default=None, help="the registry (default $RESEARCH_SLACK_REGISTRY or the checkout's)")
    c.add_argument("--channel", default=None, metavar="KEY", help=f"a registry channel (default {FRONT_DOOR})")
    c.add_argument("--channel-id", default=None, metavar="ID", help="its Slack id (default $SLACK_CHANNEL_<KEY>, else the registry's)")

    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name: str, help: str, *, as_: bool = False, thread: bool = False, body: bool = False, url: bool = False,
            aliases: tuple[str, ...] = ()) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[c], help=help, aliases=list(aliases))
        if as_:
            p.add_argument("--as", dest="handle", required=True, metavar="HANDLE", help="the handle you hold")
        if thread:
            p.add_argument("--thread", required=True, metavar="TS", help="the thread's root ts")
        if body:
            p.add_argument("--text", default=None)
            p.add_argument("--file", default=None, metavar="PATH", help="the body from PATH (`-` or neither: stdin)")
        if url:
            p.add_argument("--agent-url", default=None, help="the asking agent's link on a card (default $RESEARCH_AGENT_URL); "
                                                                     "messages ignore it: their name is their author")
        return p

    cmd("handles", "the registry").add_argument("--json", action="store_true")
    cmd("owners", "the handle that owns each path (`-`: nobody)").add_argument("paths", nargs="*", metavar="PATH",
                                                                               help="checkout paths (none: read them from stdin)")
    cmd("subscriptions", "the subscribe calls this handle's holder makes", as_=True)
    p = cmd("send", "post a message: a new thread, or into --thread; the --to names lead it and get it", as_=True, body=True, url=True)
    p.add_argument("--to", nargs="+", action="extend", default=[], metavar="@NAME", help="who it's for: handles or names with an inbox")
    p.add_argument("--thread", default=None, metavar="TS", help="reply in this thread (default: start one)")
    p.add_argument("--kind", default=None, metavar="WORD",
                   help="a word the footer carries for subscriptions (ask, the default top-level with --to; announce; "
                        "fyi, which implies --quiet; ...)")
    p.add_argument("--verify", default=None, metavar="CMD", help="announce: a required action, and the command that shows it took")
    p.add_argument("--ships", action="append", default=[], metavar="URL#sha256=HEX|git:SHA", help="announce: code it runs, pinned")
    p.add_argument("--quiet", action="store_true", help="wake only the names it starts with: no reply needed (digests still get it)")
    p.add_argument("--urgent", action="store_true", help="the doorbells say URGENT, and `read --as` lists them first")
    p = cmd("close", "close a thread you were asked in: a status line (done, or --declined WHY), and the root's reaction",
            as_=True, thread=True, url=True)
    p.add_argument("--declined", default=None, metavar="WHY")
    p.add_argument("--link", default=None, metavar="URL", help="the result: a PR, file or art: id")
    p.add_argument("--note", default=None)
    p = cmd("list", "threads in the front door: when, kind, from, to, state")
    p.add_argument("--as", dest="handle", default=None, metavar="NAME", help="only those NAME started or that reach it")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--open", action="store_true", help="leave out the done and declined")
    p.add_argument("--json", action="store_true")
    p = cmd("subscribe", "a name's subscriptions: list, add --filter F, or --remove F", as_=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--filter", default=None, help="all, kind:K, from:@x, to:@x, thread:TS (space-separated terms all hold)")
    g.add_argument("--remove", default=None, metavar="FILTER")
    p.add_argument("--deliver", choices=DELIVER, default="digest", help="wake: a doorbell now; digest: in `read --as` (default)")
    p = cmd("ask", "1:1: tag one handle or name in #agent-coordination, and ring its doorbell", as_=True, body=True, url=True)
    p.add_argument("--to", required=True, metavar="@OWNER")
    p.add_argument("--urgent", action="store_true", help="the doorbell says URGENT")
    p.add_argument("--linear", action="store_true", help="also file it as an issue in the owner's Linear team, linked from the post")
    p = cmd("linear", "the Linear asks trial: `check` shows what the credentials reach, and is fit for asks (exit 0)")
    p.add_argument("verb", choices=("check",))
    p.add_argument("--write", default=None, metavar="TEAM", help="also file, comment on and cancel a test issue in TEAM (a handle)")
    p.add_argument("--as", dest="handle", default="comms", metavar="HANDLE", help="the test issue's sender (default comms)")
    p = cmd("announce", "1:some or 1:all in #agent-coordination", as_=True, body=True, url=True)
    p.add_argument("--to", nargs="+", action="extend", default=[], metavar="@HANDLE", help="the handles or names it's for (default: every handle)")
    p.add_argument("--verify", default=None, metavar="CMD", help="a required action: the command that shows it took")
    p.add_argument("--ships", action="append", default=[], metavar="URL#sha256=HEX|git:SHA", help="code it runs, pinned")
    p.add_argument("--urgent", action="store_true", help="the doorbells say URGENT")
    p = cmd("reply", "a post in a thread; `@name` at its start also rings that name's doorbell", as_=True, thread=True, body=True, url=True)
    p.add_argument("--quiet", action="store_true", help="ring only the names it starts with: no reply needed")
    p.add_argument("--urgent", action="store_true", help="the doorbells say URGENT")
    p = cmd("names", "the leads' names, each an inbox thread: open, read, close, list", url=True, aliases=("inbox",))
    p.add_argument("verb", choices=("open", "read", "close", "list"))
    p.add_argument("--as", dest="handle", default=None, metavar="NAME", help="the inbox's name: your handle")
    p.add_argument("--holder", default=None, metavar="BC_ID", help="open: the agent that holds it (default: this agent)")
    p.add_argument("--succeeds", action="store_true", help="open: take over an inbox another agent holds (a successor)")
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
    p.add_argument("--now", default=None, help=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true")
    cmd("pickup", "\U0001f440 on an ask or alert you own", thread=True)
    p = cmd("retract", "blank a post your name sent (needs SLACK_BOT_TOKEN: the broker allows no edits)", as_=True)
    p.add_argument("--ts", required=True, help="the post's ts")
    p.add_argument("--thread", default=None, metavar="ROOT", help="its thread's root ts, when it is a reply")
    p.add_argument("--why", default=None, help="one line the blanked post keeps")
    p = cmd("done", "\u2705 a status line, and the reaction on an ask or alert", as_=True, thread=True, url=True)
    p.add_argument("--link", default=None, metavar="URL", help="the result: a PR, file or art: id")
    p.add_argument("--note", default=None)
    cmd("decline", "\u274c a status line, and the reaction on an ask or alert", as_=True, thread=True, url=True).add_argument("--why", required=True)
    p = cmd("roster", "an announcement's status lines; exit 0 only when every addressed handle is done", thread=True)
    p.add_argument("--json", action="store_true")
    p = cmd("match", "exit 0 if the post is addressed to you, 1 if not, 2 on error", as_=True)
    p.add_argument("--event", required=True, help="the message event as JSON, plain text, @PATH or - (stdin)")
    p.add_argument("--why", action="store_true", help="print the reason (or `not addressed`) on stderr")
    p = cmd("deliver", "ring the doorbells for a post no client delivered (a person's typed post)")
    p.add_argument("--ts", required=True, help="the post's ts")
    p.add_argument("--thread", default=None, metavar="TS", help="its thread's root ts, when it is a reply")
    p = cmd("verify-author", "exit 0 only when the post's author is in the registry")
    p.add_argument("--event", required=True, help="the message event as JSON, @PATH or - (stdin)")
    p.add_argument("--why", action="store_true", help="print the verdict on stderr")
    p = cmd("read", "a thread, the channel since TS, or (--as) what reached NAME since it last read (or since TS)")
    p.add_argument("--thread", default=None, metavar="TS")
    p.add_argument("--since", default=None, metavar="TS", type=since_ts,
                   help="the channel since TS (a Slack ts, or an ISO time with its zone); with --thread, its replies since TS; "
                        "with --as, NAME's doorbells and digest since TS")
    p.add_argument("--as", dest="handle", default=None, metavar="NAME", help="NAME's unread doorbells, then its digest")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--peek", action="store_true", help="--as: don't mark them read")
    p.add_argument("--urgent", action="store_true", help="--as: exit 1 unless an unread doorbell is URGENT")
    p.add_argument("--json", action="store_true")
    p = cmd("ask-daniel", "a card for Daniel in #ask-daniel; `status ID` reads its answer, `resolve ID` records a thread answer", url=True)
    p.add_argument("verb", nargs="?", choices=("status", "resolve"), help="read a card's decision, or record Daniel's answer from its thread")
    p.add_argument("id", nargs="?", metavar="ID", help="the card's id, as posting it printed")
    p.add_argument("--as", dest="handle", default=None, metavar="HANDLE", help="the asking coordinator's handle (workers ask through theirs)")
    p.add_argument("--kind", choices=KINDS, default=None, help="blocking waits for Daniel; default proceeds at the deadline unless he objects")
    p.add_argument("--question", default=None, help="one line")
    p.add_argument("--why", default=None, help="why it matters")
    p.add_argument("--option", action="append", default=[], metavar="VALUE=LABEL", help=f"{OPTIONS_MIN} to {OPTIONS_MAX} of them")
    p.add_argument("--recommend", default=None, metavar="VALUE")
    p.add_argument("--default", default=None, metavar="VALUE")
    p.add_argument("--deadline", default=None, help="`3:00 PM` (today, Pacific) or an ISO time with its zone")
    p.add_argument("--link", default=None, metavar="URL", help="the full context")
    p.add_argument("--wait", action="store_true", help="poll until the card is decided (or overdue); exit 0 once decided")
    p.add_argument("--every", type=float, default=WAIT_EVERY_S, metavar="S", help=f"seconds between polls (default {WAIT_EVERY_S})")
    p.add_argument("--choice", default=None, metavar="VALUE", help="resolve: the option Daniel chose in the thread")
    p.add_argument("--note", default=None, help="resolve: his words, or a link to them")
    p.add_argument("--json", action="store_true", help="status: the route's whole answer")
    p = sub.add_parser("groups", help="the handles' Slack user groups")
    gsub = p.add_subparsers(dest="gcmd", required=True)
    gsub.add_parser("sync", parents=[c], help="create or update a user group per handle; a plan unless --apply").add_argument("--apply", action="store_true")

    ns = ap.parse_args(argv)
    try:
        path = registry_path(ns.registry)
        reg = load_registry(path)
        if ns.cmd == "groups":
            return cmd_groups_sync(ns, reg, path)
        return {"handles": cmd_handles, "owners": cmd_owners, "subscriptions": cmd_subscriptions, "ask": cmd_ask, "announce": cmd_announce, "reply": cmd_reply,
                "pickup": cmd_pickup, "done": cmd_done, "decline": cmd_decline, "roster": cmd_roster, "match": cmd_match,
                "verify-author": cmd_verify_author, "deliver": cmd_deliver, "read": cmd_read, "ask-daniel": cmd_ask_daniel, "names": cmd_inbox,
                "inbox": cmd_inbox, "traffic": cmd_traffic, "send": cmd_send, "close": cmd_close, "list": cmd_list,
                "subscribe": cmd_subscribe, "watch": cmd_watch, "linear": cmd_linear, "retract": cmd_retract}[ns.cmd](ns, reg)
    except SlackError as e:
        print(f"{prog}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""`research slack`: the coordination API on Slack (the conventions are `.agents/skills/using-slack/SKILL.md`).

Handles (Slack user groups: @infra, @proofs, ...) are roles, each held by one agent; only handle holders and named humans are on
Slack, and a coordinator relays to its own workers. Everything between handles happens in #agent-coordination: an ask tags one
handle, an announcement tags some (or none, for every handle), and each is one thread. #agent-alerts is machines to infra.
`.agents/skills/using-slack/registry.json` names the handles, channels and humans.

  research slack handles [--json]                                  # the registry
  research slack subscriptions --as HANDLE                         # the subscribe calls this handle's holder makes
  research slack ask --as HANDLE --to @OWNER [--text T | --file PATH]            # 1:1; prints `<ts> <permalink>`
  research slack announce --as HANDLE [--to @A @B ...] [--text T | --file PATH] [--verify CMD] [--ships URL#sha256=HEX | git:SHA]...
  research slack reply --as HANDLE --thread TS [--text T | --file PATH]
  research slack pickup --thread TS                                # 👀 on an ask you own
  research slack done --as HANDLE --thread TS [--link URL] [--note T]            # ✅: a status line, and the reaction on an ask
  research slack decline --as HANDLE --thread TS --why T           # ❌ the same way
  research slack roster --thread TS [--json]                       # an announcement's status lines; exit 0 when all are done
  research slack match --as HANDLE --event JSON|TEXT|@PATH|- [--why]            # exit 0 addressed, 1 not, 2 error
  research slack verify-author --event JSON|@PATH|- [--why]        # exit 0 only for our bot, a channel's bot or a named human
  research slack read --thread TS | --since TS [--limit N] [--json]
  research slack groups sync [--apply]                             # one user group per handle, and the bot's id; a plan unless --apply

Every subcommand takes `--dry-run` (no request reaches Slack; each one it would send is printed, the token never),
`--registry PATH` (default $RESEARCH_SLACK_REGISTRY, else the checkout's) and `--channel KEY` (default agent-coordination);
a channel's id is $SLACK_CHANNEL_<KEY> (upper case, `-` as `_`) or the registry's, and `--channel-id` overrides both. The token
is $SLACK_BOT_TOKEN and is never printed; $SLACK_API_URL points the client at another Web API (the tests' fake). Stdlib only.

Without $SLACK_BOT_TOKEN (or with RESEARCH_SLACK_VIA=broker) every call goes through Verity's Slack broker instead: a POST of
`{"method", "args"}` to BROKER with this agent's Cursor OIDC token, answered with Slack's response unchanged. The broker holds the
bot token and allows only BROKER_METHODS; RESEARCH_SLACK_VIA=direct insists on the token.

A post is addressed to HANDLE when it mentions the handle (its user group `<!subteam^ID>`, or plain `@handle`), is an
announcement that tags no handle, is top-level in a channel HANDLE subscribes to by name (#agent-alerts), or is a reply in a
thread whose root addressed HANDLE or where HANDLE has posted. A status line (✅/❌) is addressed only to the thread's author,
and a handle's own posts never are."""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "verity/slack-registry/v1"
REGISTRY_REL = Path(".agents/skills/using-slack/registry.json")
FRONT_DOOR = "agent-coordination"
DEFAULT_API = "https://slack.com/api/"
BROKER = "https://website-docs-sage.vercel.app/api/agent-slack/call"      # also the OIDC token's audience
#: every Web API method this module calls, and so exactly what the broker's route allows (the tests check the two agree)
BROKER_METHODS = ("auth.test", "chat.getPermalink", "chat.postMessage", "conversations.history", "conversations.replies",
                  "reactions.add", "usergroups.create", "usergroups.enable", "usergroups.list", "usergroups.update")
BROKER_MAX_BODY = 65536                                                   # bytes of request JSON; the route refuses more (413)
ERROR_CODE_RE = re.compile(r"^[a-z0-9_]{1,64}$")
EVERY = "every handle"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
SUBTEAM_RE = re.compile(r"<!subteam\^([A-Z0-9]+)(?:\|@?([^>]*))?>")
ANNOUNCE = "\U0001f4e3"                                            # 📣
DONE, DECLINED, TAKING = "\u2705", "\u274c", "\U0001f440"         # ✅ ❌ 👀
SHORTCODES = {ANNOUNCE: ":mega:", DONE: ":white_check_mark:", DECLINED: ":x:"}   # Slack stores a posted emoji as its shortcode
STATUS_RE = re.compile(rf"^((?:{DONE}|{SHORTCODES[DONE]}) done|(?:{DECLINED}|{SHORTCODES[DECLINED]}) declined)\b(?: \u2014 (.*?))?(?:\n|$)")
PIN_RE = re.compile(r"^(?:https?://\S+#sha256=[0-9a-f]{64}|git:[0-9a-f]{40})$")
SHIPS_CODE_RE = re.compile(r"\b(curl|wget|pip install|uv (?:pip|sync|tool)|git (?:pull|merge|checkout|fetch)|install|download|upgrade)\b", re.I)
SAFE_POINT = "Apply at your next safe point; don't interrupt running jobs."
EXPIRES_S = 259200
PAGE = 200


class SlackError(RuntimeError):
    pass


# --- registry ---------------------------------------------------------------------------------------------------------------

def registry_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    if env := os.environ.get("RESEARCH_SLACK_REGISTRY"):
        return Path(env)
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / REGISTRY_REL).is_file():
            return d / REGISTRY_REL
    return Path(__file__).resolve().parents[4] / REGISTRY_REL          # tools/research/src/research/slack.py -> the checkout


def load_registry(path: Path) -> dict:
    try:
        reg = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SlackError(f"registry {path}: {e}") from e
    if reg.get("schema") != SCHEMA:
        raise SlackError(f"registry {path}: schema {reg.get('schema')!r}, expected {SCHEMA!r}")
    handles = set()
    for h in reg.get("handles", []):
        name = h.get("handle", "")
        if not NAME_RE.match(name) or name in handles:
            raise SlackError(f"registry {path}: bad or duplicate handle {name!r}")
        handles.add(name)
        for k in ("holder", "remit", "charter"):
            if not isinstance(h.get(k), str) or not h[k]:
                raise SlackError(f"registry {path}: @{name} has no {k}")
    chans = reg.get("channels") or {}
    if FRONT_DOOR not in chans:
        raise SlackError(f"registry {path}: no #{FRONT_DOOR}")
    for key, ch in chans.items():
        subs = ch.get("subscribers")
        if not NAME_RE.match(key) or not ch.get("purpose") or not (subs == EVERY or (isinstance(subs, list) and set(subs) <= handles)):
            raise SlackError(f"registry {path}: #{key} needs a purpose and subscribers ({EVERY!r} or a list of handles)")
    return reg


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


def channel_key(ns: argparse.Namespace, reg: dict) -> str:
    key = (ns.channel or FRONT_DOOR).lstrip("#")
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
    return f"<!subteam^{h['usergroup']}|@{h['handle']}>" if h.get("usergroup") else f"@{h['handle']}"


# --- the Web API ------------------------------------------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("redirected; refusing to forward the bearer token")


def post(url: str, data: bytes, headers: dict[str, str]) -> dict:
    """The one HTTP exchange: POST `data`, parse the JSON answer. Raises urllib's errors for the caller to map."""
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.build_opener(_NoRedirect).open(req, timeout=30) as r:
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


def oidc_token(aud: str) -> str:
    """A Cursor OIDC token for this agent with audience `aud`, from the local cursor-agent socket."""
    conn = _UnixHTTP(os.environ.get("CURSOR_AGENT_SOCKET", "/run/cursor/api.sock"))
    try:
        conn.request("POST", "/v1/tokens/oidc", json.dumps({"aud": aud}), {"Content-Type": "application/json"})
        r = conn.getresponse()
        if r.status == 403:
            raise SlackError("Cursor refused this agent's identity (the broker needs a Cursor agent's OIDC token)")
        if r.status != 200:
            raise SlackError(f"the Cursor identity service is unavailable (HTTP {r.status})")
        token = json.loads(r.read(32768)).get("token")
    except (OSError, http.client.HTTPException, ValueError, AttributeError):
        raise SlackError("the Cursor identity socket the Slack broker needs is unavailable (not a Cursor agent? set SLACK_BOT_TOKEN)") from None
    finally:
        conn.close()
    if not isinstance(token, str) or not token or any(c.isspace() for c in token):
        raise SlackError("Cursor returned an invalid identity token")
    return token


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
        return f"{method}: the Slack broker refused the call ({why}): it serves danielreuter/verity's agents and BROKER_METHODS only"
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
        self.dry_run, self.log, self.transport, self._oidc = dry_run, log, transport or post, None
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
            raise SlackError(f"{method}: {out.get('error', 'not ok')}" + (f" (needed scope {out['needed']})" if out.get("needed") else ""))
        return out

    def _broker(self, method: str, args: dict[str, str]) -> dict:
        if method not in BROKER_METHODS:
            raise SlackError(f"{method}: not in BROKER_METHODS, so the broker would refuse it")
        body = json.dumps({"method": method, "args": args}).encode()
        if len(body) > BROKER_MAX_BODY:
            raise SlackError(f"{method}: {len(body)} bytes is more than the broker takes ({BROKER_MAX_BODY}); put the content in a file and link it")
        self._oidc = self._oidc or oidc_token(BROKER)
        return self.transport(BROKER, body, {"Authorization": f"Bearer {self._oidc}", "Content-Type": "application/json"})

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


def _footer(handle: str, url: str | None) -> str:
    url = url or os.environ.get("RESEARCH_AGENT_URL")
    if not url:
        raise SlackError("no --agent-url (or $RESEARCH_AGENT_URL): every post links the agent that wrote it")
    return f"_posted by <{url}|@{handle}'s agent>_"


def _post(api: Slack, ch: str, handle: str, text: str, url: str | None, thread: str | None = None) -> str:
    out = api.call("chat.postMessage", channel=ch, thread_ts=thread, username=handle, text=f"{text}\n\n{_footer(handle, url)}",
                   unfurl_links="false", unfurl_media="false")
    ts = out.get("ts") or "DRY-RUN"
    link = api.call("chat.getPermalink", channel=ch, message_ts=ts).get("permalink") or "-"
    print(f"{ts} {link}")
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
    if by_handle(root, handle) or any(by_handle(m, handle) for m in msgs if m.get("ts") != ev.get("ts")):
        return f"thread {thread}: @{handle} posted in it"
    if root and mentions(root.get("text", ""), h):
        return f"thread {thread}: its root addresses @{handle}"
    if key and key != FRONT_DOOR and handle in subscribers(reg, key):
        return f"thread {thread} in #{key}, which @{handle} subscribes to"
    return None


def channel_id_for(reg: dict, key: str) -> str | None:
    return os.environ.get(_env(key)) or reg["channels"][key].get("id")


def fetch_message(api: Slack, ch: str, ts: str, thread: str | None) -> dict | None:
    """The message itself, from Slack: a wake payload is not evidence of who wrote it."""
    if thread and thread != ts:
        msgs = api.call("conversations.replies", channel=ch, ts=thread, oldest=ts, latest=ts, inclusive="true", limit=1).get("messages", [])
    else:
        msgs = api.call("conversations.history", channel=ch, oldest=ts, latest=ts, inclusive="true", limit=1).get("messages", [])
    return next((m for m in msgs if m.get("ts") == ts), None)


def author_allowed(msg: dict, reg: dict, key: str) -> str | None:
    """Why `msg`'s author is one of ours, or None: Verity's bot (the registry's `bot_id`, under a handle's username), a bot the
    channel lists (`bots`: alerting), or a named human (`humans`, by Slack user id). An entry without an id allows nobody."""
    if msg.get("bot_id"):
        handles = {h["handle"] for h in reg["handles"]}
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


def cmd_subscriptions(ns: argparse.Namespace, reg: dict) -> int:
    h = entry(reg, ns.handle)
    missing = False
    for key in reg["channels"]:
        if h["handle"] in subscribers(reg, key):
            cid = channel_id_for(reg, key)
            missing |= not cid
            print(f'subscribe_slack_channel(channelId="{cid or "<#" + key + " id not recorded>"}", topLevelOnly=true, '
                  f"expiresInSeconds={EXPIRES_S})   # #{key}")
    print(f"# and subscribe_slack_thread(channelId, threadTs, expiresInSeconds={EXPIRES_S}) for each thread @{h['handle']} "
          f"starts, replies in or picks up (not for a status line alone); renew on every wake")
    return 2 if missing else 0


def cmd_ask(ns: argparse.Namespace, reg: dict) -> int:
    h, owner = entry(reg, ns.handle), entry(reg, ns.to)
    if owner["handle"] == h["handle"]:
        raise SlackError("an ask goes to another handle")
    _post(client(ns), channel_id(ns, reg, FRONT_DOOR), h["handle"], f"{tag(owner)} {_body(ns)}", ns.agent_url)
    return 0


def cmd_announce(ns: argparse.Namespace, reg: dict) -> int:
    h = entry(reg, ns.handle)
    to = [entry(reg, t) for t in ns.to]
    body = _body(ns)
    for pin in ns.ships:
        if not PIN_RE.match(pin):
            raise SlackError(f"--ships {pin!r}: pin code as URL#sha256=<64 hex> or git:<40-hex commit>")
    if not ns.ships and SHIPS_CODE_RE.search(f"{body}\n{ns.verify or ''}"):
        raise SlackError("the announcement fetches or installs code: pin it with --ships URL#sha256=HEX or git:SHA")
    audience = " ".join(tag(t) for t in to) if to else "every handle"
    lines = [f"{ANNOUNCE} *Announcement* from @{h['handle']} to {audience}", body]
    if ns.verify:
        lines += [f"*{SAFE_POINT}*", f"*Verify:* `{ns.verify}`"]
    for pin in ns.ships:
        url, _, sha = pin.partition("#sha256=")
        lines.append(f"*Pinned:* commit `{pin[4:]}`" if pin.startswith("git:") else f"*Pinned:* {url} sha256 `{sha}` (don't run it if the checksum fails)")
    lines.append(f"*Reply with a status line:* `research slack done --thread <this ts>` or `research slack decline --thread <this ts> --why ...`")
    _post(client(ns), channel_id(ns, reg, FRONT_DOOR), h["handle"], "\n".join(lines), ns.agent_url)
    return 0


def cmd_reply(ns: argparse.Namespace, reg: dict) -> int:
    h = entry(reg, ns.handle)
    _post(client(ns), channel_id(ns, reg), h["handle"], _body(ns), ns.agent_url, ns.thread)
    return 0


def cmd_pickup(ns: argparse.Namespace, reg: dict) -> int:
    client(ns).call("reactions.add", channel=channel_id(ns, reg), timestamp=ns.thread, name="eyes")
    return 0


def _status(ns: argparse.Namespace, reg: dict, word: str, reaction: str, detail: str | None) -> int:
    """A status line in the thread; on an ask or an alert (one owner), also the reaction on its root. On an announcement the
    line alone carries it: every handle posts through one bot, which can place each reaction only once."""
    h = entry(reg, ns.handle)
    api, ch = client(ns), channel_id(ns, reg)
    detail = " ".join((detail or "").split())
    _post(api, ch, h["handle"], f"{word}" + (f" \u2014 {detail}" if detail else ""), ns.agent_url, ns.thread)
    if not is_announcement(api.root(ch, ns.thread)):
        api.call("reactions.add", channel=ch, timestamp=ns.thread, name=reaction)
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
            status[m["username"]] = {"status": mt.group(1).split()[1], "detail": mt.group(2) or "", "ts": m["ts"]}
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
        why = author_allowed(msg, reg, key) if msg else None
        who = "@" + msg["username"] if msg and msg.get("username") else (msg or {}).get("user") or (msg or {}).get("bot_id")
        why_not = "no such message in the channel" if msg is None else f"author {who} is not in the registry"
    if ns.why:
        print(f"allowed: {why}" if why else f"not allowed: {why_not}", file=sys.stderr)
    return 0 if why else 1


def cmd_read(ns: argparse.Namespace, reg: dict) -> int:
    api, ch = client(ns), channel_id(ns, reg)
    if ns.thread:
        msgs = api.paged("conversations.replies", "messages", ns.limit, channel=ch, ts=ns.thread)
    else:
        msgs = api.paged("conversations.history", "messages", ns.limit, channel=ch, oldest=ns.since)
        msgs = [m for m in msgs if m.get("ts") != ns.since][::-1]                   # oldest first; `oldest` is inclusive at the edge
    if ns.json:
        print(json.dumps(msgs, indent=2, ensure_ascii=False))
        return 0
    for m in msgs:
        thread = f" [thread {m['thread_ts']}]" if m.get("thread_ts") and m.get("thread_ts") != m.get("ts") else ""
        who = "@" + m["username"] if m.get("username") else m.get("user") or m.get("bot_id") or "?"
        print(f"{m.get('ts')}{thread} {who}: {m.get('text', '')}")
    return 0


def cmd_groups_sync(ns: argparse.Namespace, reg: dict, path: Path) -> int:
    """One user group per registry handle (`handle` = its handle, `name` = its handle, `description` = its remit), created,
    re-enabled or updated; with --apply the groups' ids, and the bot's own id when the registry lacks it, are written back into
    the registry, which the caller commits. Existing groups keep their members."""
    api = client(ns)
    write = ns.apply and not ns.dry_run
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
    if write and changed:
        path.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {path}: commit it")
    elif not write:
        print("plan only: rerun with --apply to change Slack and the registry")
    return 0


def main(argv: list[str]) -> int:
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--dry-run", action="store_true", help="send nothing to Slack; print each request instead")
    c.add_argument("--registry", default=None, help="the registry (default $RESEARCH_SLACK_REGISTRY or the checkout's)")
    c.add_argument("--channel", default=None, metavar="KEY", help=f"a registry channel (default {FRONT_DOOR})")
    c.add_argument("--channel-id", default=None, metavar="ID", help="its Slack id (default $SLACK_CHANNEL_<KEY>, else the registry's)")

    ap = argparse.ArgumentParser(prog="research slack", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name: str, help: str, *, as_: bool = False, thread: bool = False, body: bool = False, url: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[c], help=help)
        if as_:
            p.add_argument("--as", dest="handle", required=True, metavar="HANDLE", help="the handle you hold")
        if thread:
            p.add_argument("--thread", required=True, metavar="TS", help="the thread's root ts")
        if body:
            p.add_argument("--text", default=None)
            p.add_argument("--file", default=None, metavar="PATH", help="the body from PATH (`-` or neither: stdin)")
        if url:
            p.add_argument("--agent-url", default=None, help="footer link to the posting agent (default $RESEARCH_AGENT_URL)")
        return p

    cmd("handles", "the registry").add_argument("--json", action="store_true")
    cmd("subscriptions", "the subscribe calls this handle's holder makes", as_=True)
    cmd("ask", "1:1: tag one handle in #agent-coordination", as_=True, body=True, url=True).add_argument("--to", required=True, metavar="@OWNER")
    p = cmd("announce", "1:some or 1:all in #agent-coordination", as_=True, body=True, url=True)
    p.add_argument("--to", nargs="+", default=[], metavar="@HANDLE", help="the handles it's for (default: every handle)")
    p.add_argument("--verify", default=None, metavar="CMD", help="a required action: the command that shows it took")
    p.add_argument("--ships", action="append", default=[], metavar="URL#sha256=HEX|git:SHA", help="code it runs, pinned")
    cmd("reply", "a post in a thread", as_=True, thread=True, body=True, url=True)
    cmd("pickup", "\U0001f440 on an ask or alert you own", thread=True)
    p = cmd("done", "\u2705 a status line, and the reaction on an ask or alert", as_=True, thread=True, url=True)
    p.add_argument("--link", default=None, metavar="URL", help="the result: a PR, file or art: id")
    p.add_argument("--note", default=None)
    cmd("decline", "\u274c a status line, and the reaction on an ask or alert", as_=True, thread=True, url=True).add_argument("--why", required=True)
    p = cmd("roster", "an announcement's status lines; exit 0 only when every addressed handle is done", thread=True)
    p.add_argument("--json", action="store_true")
    p = cmd("match", "exit 0 if the post is addressed to you, 1 if not, 2 on error", as_=True)
    p.add_argument("--event", required=True, help="the message event as JSON, plain text, @PATH or - (stdin)")
    p.add_argument("--why", action="store_true", help="print the reason (or `not addressed`) on stderr")
    p = cmd("verify-author", "exit 0 only when the post's author is in the registry")
    p.add_argument("--event", required=True, help="the message event as JSON, @PATH or - (stdin)")
    p.add_argument("--why", action="store_true", help="print the verdict on stderr")
    p = cmd("read", "channel or thread history, oldest first")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--thread", default=None, metavar="TS")
    g.add_argument("--since", default=None, metavar="TS")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("groups", help="the handles' Slack user groups")
    gsub = p.add_subparsers(dest="gcmd", required=True)
    gsub.add_parser("sync", parents=[c], help="create or update a user group per handle; a plan unless --apply").add_argument("--apply", action="store_true")

    ns = ap.parse_args(argv)
    try:
        path = registry_path(ns.registry)
        reg = load_registry(path)
        if ns.cmd == "groups":
            return cmd_groups_sync(ns, reg, path)
        return {"handles": cmd_handles, "subscriptions": cmd_subscriptions, "ask": cmd_ask, "announce": cmd_announce, "reply": cmd_reply,
                "pickup": cmd_pickup, "done": cmd_done, "decline": cmd_decline, "roster": cmd_roster, "match": cmd_match,
                "verify-author": cmd_verify_author, "read": cmd_read}[ns.cmd](ns, reg)
    except SlackError as e:
        print(f"research slack: {e}", file=sys.stderr)
        return 2

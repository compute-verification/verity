"""`research slack`: agents' side of Slack (the conventions are `.agents/skills/using-slack/SKILL.md`).

Handles are addresses (Slack user groups), channels are places, threads are conversations; the registry
(`.agents/skills/using-slack/registry.json`) names all three.

  research slack handles [--json]                                   # the registry: handles, channels, humans
  research slack post --as HANDLE [--channel KEY] [--thread TS] [--text T | --file PATH] [--agent-url URL]   # `<ts> <permalink>`
  research slack read [--channel KEY] --thread TS | --since TS [--limit N] [--json]
  research slack match --handle HANDLE --event JSON|TEXT|@PATH|- [--why]      # exit 0 addressed, 1 not, 2 error
  research slack pickup --channel KEY --thread TS                    # the owner's 👀 on a request
  research slack done --as HANDLE --channel KEY --thread TS --link URL   # ✅ on the request, and the result's link in its thread
  research slack announce --channel SERVICE --as OWNER --action TEXT --verify CMD [--ships URL#sha256=HEX | git:SHA]...
  research slack ack --channel KEY --thread TS --as NAME --status done|blocked [--note TEXT]
  research slack roster --channel KEY --thread TS [--json]           # acks against the registry; exit 0 only when all are done
  research slack verify-author --event JSON|@PATH|- [--channel KEY] [--why]   # exit 0 only for an allowlisted author
  research slack groups sync [--apply]                               # one user group per handle; a plan unless --apply

Every subcommand takes `--dry-run` (no request reaches Slack; each one it would send is printed, the token never) and
`--registry PATH` (default $RESEARCH_SLACK_REGISTRY, else the checkout's). `--channel KEY` is a registry channel
(`agent-coordination` by default where there is one); its id is $SLACK_CHANNEL_<KEY> (upper case, `-` as `_`) or the
registry's, and `--channel-id` overrides both. The token is $SLACK_BOT_TOKEN; a handle's top-level posts in a channel it owns
use the variable its `token_env` names, so that its bot is a distinct author. No token is ever printed. $SLACK_API_URL points
the client at another Web API (the tests' fake). Stdlib only.

A message is addressed to HANDLE when it mentions the handle's user group (`<!subteam^ID>` or `<!subteam^ID|@handle>`, the ID
from the registry), names `@handle` in plain text, or is a reply in a thread whose root mentions the handle or where the handle
has posted. The handle's own posts never are (an agent must not wake itself)."""
from __future__ import annotations

import argparse
import json
import os
import re
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
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
SUBTEAM_RE = re.compile(r"<!subteam\^([A-Z0-9]+)(?:\|@?([^>]*))?>")
ACK_RE = re.compile(r"^ACK (done|blocked)\b(?: \u2014 (.*?))?(?:\n|$)")
PIN_RE = re.compile(r"^(?:https?://\S+#sha256=[0-9a-f]{64}|git:[0-9a-f]{40})$")
SHIPS_CODE_RE = re.compile(r"\b(curl|wget|pip install|uv (?:pip|sync|tool)|git (?:pull|merge|checkout|fetch)|install|download|upgrade)\b", re.I)
SAFE_POINT = "Apply at your next safe point; don't interrupt running jobs."
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
        for k in ("owner", "remit", "charter"):
            if not isinstance(h.get(k), str) or not h[k]:
                raise SlackError(f"registry {path}: @{name} has no {k}")
    for key, ch in (reg.get("channels") or {}).items():
        if not NAME_RE.match(key) or not isinstance(ch, dict):
            raise SlackError(f"registry {path}: bad channel {key!r}")
        if ch.get("owner") is not None and ch["owner"] not in handles:
            raise SlackError(f"registry {path}: #{key}'s owner @{ch['owner']} is not a handle")
        for k in ("purpose", "subscribe"):
            if not ch.get(k):
                raise SlackError(f"registry {path}: #{key} has no {k}")
    return reg


def entry(reg: dict, handle: str) -> dict:
    handle = handle.lstrip("@")
    for h in reg["handles"]:
        if h["handle"] == handle:
            return h
    raise SlackError(f"unknown handle @{handle}; the registry has " + ", ".join("@" + h["handle"] for h in reg["handles"]))


def speaker(reg: dict, name: str) -> dict:
    """A registry handle's entry, or, for a worker (which has none), its lane name as a bare entry."""
    name = name.lstrip("@")
    if any(h["handle"] == name for h in reg["handles"]):
        return entry(reg, name)
    if not NAME_RE.match(name):
        raise SlackError(f"{name!r} is neither a registry handle nor a lane name (lowercase letters, digits, . _ -)")
    return {"handle": name}


def channel_key(ns: argparse.Namespace, reg: dict) -> str:
    key = (ns.channel or "").lstrip("#")
    chans = reg.get("channels") or {}
    if not key:
        raise SlackError("pass --channel (one of " + ", ".join(sorted(chans)) + ")")
    if key not in chans:
        raise SlackError(f"no channel #{key} in the registry (it has {', '.join(sorted(chans)) or 'none'})")
    return key


def channel_id(ns: argparse.Namespace, reg: dict) -> str:
    key = channel_key(ns, reg)
    env = "SLACK_CHANNEL_" + key.upper().replace("-", "_").replace(".", "_")
    cid = ns.channel_id or os.environ.get(env) or reg["channels"][key].get("id")
    if not cid:
        raise SlackError(f"no channel id for #{key}: pass --channel-id, set {env}, or record its id in the registry")
    return cid


# --- the Web API ------------------------------------------------------------------------------------------------------------

class Slack:
    """A minimal Web API client: form-encoded POSTs with a bearer token. With `dry_run`, nothing is sent: each call is printed
    (method and parameters, never the token) and answered with an empty `ok` response."""

    def __init__(self, token: str | None, *, token_env: str = "SLACK_BOT_TOKEN", base: str | None = None, dry_run: bool = False,
                 log=None):
        self.token, self.dry_run, self.log = token, dry_run, log
        self.base = (base or os.environ.get("SLACK_API_URL") or DEFAULT_API).rstrip("/") + "/"
        if not dry_run and not token:
            raise SlackError(f"{token_env} is not set (a Cursor secret; see the using-slack skill), or pass --dry-run")

    def call(self, method: str, **params: Any) -> dict:
        params = {k: v for k, v in params.items() if v is not None}
        if self.dry_run:
            print(f"dry-run: {method} {json.dumps(params, sort_keys=True)}", file=self.log or sys.stderr)
            return {"ok": True, "dry_run": True}
        body = urllib.parse.urlencode({k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in params.items()})
        req = urllib.request.Request(self.base + method, data=body.encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/x-www-form-urlencoded; charset=utf-8"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.load(r)
        except urllib.error.HTTPError as e:
            hint = f" (rate limited; retry after {e.headers.get('Retry-After')} s)" if e.code == 429 else ""
            raise SlackError(f"{method}: HTTP {e.code}{hint}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise SlackError(f"{method}: {type(e).__name__}: {e}") from None
        if not out.get("ok"):
            raise SlackError(f"{method}: {out.get('error', 'not ok')}" + (f" (needed scope {out['needed']})" if out.get("needed") else ""))
        return out

    def paged(self, method: str, key: str, limit: int | None = None, **params: Any) -> list[dict]:
        items: list[dict] = []
        cursor = None
        while True:
            out = self.call(method, limit=PAGE, cursor=cursor, **params)
            items += out.get(key, [])
            cursor = (out.get("response_metadata") or {}).get("next_cursor")
            if not cursor or (limit and len(items) >= limit):
                return items[:limit] if limit else items


def client(ns: argparse.Namespace, token_env: str = "SLACK_BOT_TOKEN") -> Slack:
    return Slack(os.environ.get(token_env), token_env=token_env, dry_run=ns.dry_run)


def _footer(handle: str, url: str | None) -> str:
    url = url or os.environ.get("RESEARCH_AGENT_URL")
    if not url:
        raise SlackError("no --agent-url (or $RESEARCH_AGENT_URL): every post links the agent that wrote it")
    return f"_posted by <{url}|@{handle}'s agent>_"


def _post(api: Slack, ch: str, name: str, text: str, thread: str | None = None) -> str:
    out = api.call("chat.postMessage", channel=ch, thread_ts=thread, username=name, text=text, unfurl_links="false", unfurl_media="false")
    ts = out.get("ts") or "DRY-RUN"
    link = api.call("chat.getPermalink", channel=ch, message_ts=ts).get("permalink") or "-"
    print(f"{ts} {link}")
    return ts


def _owner_check(reg: dict, key: str, handle: str, what: str) -> dict:
    """Top-level posts in a channel with an owner are the owner's (and named humans', who don't use this tool)."""
    ch = reg["channels"][key]
    if ch.get("owner") and handle != ch["owner"]:
        raise SlackError(f"#{key}: only its owner @{ch['owner']} {what}; everyone else replies in threads (--thread TS)")
    return ch


def _token_env(reg: dict, key: str, handle: str, top_level: bool) -> str:
    h = speaker(reg, handle)
    owns = reg["channels"][key].get("owner") == handle
    return h.get("token_env", "SLACK_BOT_TOKEN") if top_level and owns else "SLACK_BOT_TOKEN"


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


def by_handle(msg: dict, handle: str) -> bool:
    """Posted by this handle: a bot message under its username (`post --as`)."""
    return bool(msg.get("bot_id")) and msg.get("username") == handle


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


def addressed(ev: dict, h: dict, api: Slack | None, channel: str | None) -> str | None:
    """Why the message event `ev` is addressed to `h`, or None. Looks up the thread only when the message is a reply."""
    handle = h["handle"]
    if by_handle(ev, handle):
        return None
    if why := mentions(ev.get("text", ""), h):
        return why
    thread = ev.get("thread_ts")
    if not thread or thread == ev.get("ts") or api is None:
        return None
    ch = ev.get("channel") or channel
    if not ch:
        raise SlackError("the event is a thread reply with no channel, and no --channel is known")
    for m in api.paged("conversations.replies", "messages", channel=ch, ts=thread):
        if m.get("ts") == ev.get("ts"):
            continue
        if by_handle(m, handle):
            return f"thread {thread}: @{handle} posted in it"
        if m.get("ts") == thread and mentions(m.get("text", ""), h):
            return f"thread {thread}: its root addresses @{handle}"
    return None


def fetch_message(api: Slack, ch: str, ts: str, thread: str | None) -> dict | None:
    """The message itself, from Slack: a wake payload is not evidence of who wrote it."""
    if thread and thread != ts:
        msgs = api.call("conversations.replies", channel=ch, ts=thread, oldest=ts, latest=ts, inclusive="true", limit=1).get("messages", [])
    else:
        msgs = api.call("conversations.history", channel=ch, oldest=ts, latest=ts, inclusive="true", limit=1).get("messages", [])
    return next((m for m in msgs if m.get("ts") == ts), None)


def author_allowed(msg: dict, reg: dict, key: str) -> str | None:
    """Why `msg`'s author may post top-level in #key, or None: the channel owner's own bot (its handle's `bot_id`, under its
    username), a bot the channel lists in `authors.bots` (machine alerts), or a named human (`humans`, by Slack user id).
    An entry without an id allows nobody."""
    ch = reg["channels"][key]
    bots = list((ch.get("authors") or {}).get("bots", []))
    if ch.get("owner"):
        bots.append({"handle": ch["owner"], "bot_id": entry(reg, ch["owner"]).get("bot_id")})
    if msg.get("bot_id"):
        for b in bots:
            if b.get("bot_id") and msg["bot_id"] == b["bot_id"] and (not b.get("handle") or msg.get("username") == b["handle"]):
                return f"bot {b['bot_id']} (@{b.get('handle') or msg.get('username')})"
        return None
    for p in reg.get("humans", []):
        if p.get("slack_user") and msg.get("user") == p["slack_user"]:
            return f"{p['name']} ({p['slack_user']})"
    return None


# --- subcommands ------------------------------------------------------------------------------------------------------------

def cmd_handles(ns: argparse.Namespace, reg: dict) -> int:
    if ns.json:
        print(json.dumps(reg, indent=2))
        return 0
    print(f"workspace {reg.get('workspace')}")
    for key, ch in (reg.get("channels") or {}).items():
        own = f"owner @{ch['owner']}" if ch.get("owner") else "no owner"
        sub = ch["subscribe"]
        mode = f"topLevelOnly={str(bool(sub.get('top_level_only'))).lower()}" + ("; the owner in full" if sub.get("owner_full") else "")
        print(f"#{key:<20} {ch.get('id') or 'id not recorded':<12} {own}\n  {ch['purpose']}\n  subscribe: {sub.get('who')} ({mode})"
              f"\n  contract: {ch.get('contract') or 'not pinned yet'}")
    for h in reg["handles"]:
        print(f"@{h['handle']:<22} {h.get('usergroup') or '-':<12} {h['owner']}\n  {h['remit']}\n  charter: {h['charter']}")
    for p in reg.get("humans", []):
        print(f"human {p['name']} ({p.get('slack_user') or 'user id not recorded'})")
    return 0


def _body(ns: argparse.Namespace) -> str:
    if ns.text is not None:
        return ns.text
    if ns.file and ns.file != "-":
        return Path(ns.file).read_text()
    return sys.stdin.read()


def cmd_post(ns: argparse.Namespace, reg: dict) -> int:
    h = speaker(reg, ns.handle)
    key = channel_key(ns, reg)
    if not ns.thread:
        _owner_check(reg, key, h["handle"], "posts top-level")
    footer = _footer(h["handle"], ns.agent_url)
    text = _body(ns).rstrip()
    if not text:
        raise SlackError("empty message")
    api = client(ns, _token_env(reg, key, h["handle"], not ns.thread))
    _post(api, channel_id(ns, reg), h["handle"], f"{text}\n\n{footer}", ns.thread)
    return 0


def _who(m: dict) -> str:
    return "@" + m["username"] if m.get("username") else m.get("user") or m.get("bot_id") or "?"


def cmd_read(ns: argparse.Namespace, reg: dict) -> int:
    api, ch = client(ns), channel_id(ns, reg)
    if ns.thread:
        msgs = api.paged("conversations.replies", "messages", ns.limit, channel=ch, ts=ns.thread)
    else:
        msgs = api.paged("conversations.history", "messages", ns.limit, channel=ch, oldest=ns.since)
        msgs = [m for m in msgs if m.get("ts") != ns.since][::-1]                   # oldest first; `oldest` is inclusive at the edge
    if ns.json:
        print(json.dumps(msgs, indent=2))
        return 0
    for m in msgs:
        thread = f" [thread {m['thread_ts']}]" if m.get("thread_ts") and m.get("thread_ts") != m.get("ts") else ""
        print(f"{m.get('ts')}{thread} {_who(m)}: {m.get('text', '')}")
    return 0


def cmd_match(ns: argparse.Namespace, reg: dict) -> int:
    h = speaker(reg, ns.handle)
    ev = parse_event(ns.event)
    reply = ev.get("thread_ts") and ev.get("thread_ts") != ev.get("ts")
    needs_thread = reply and not mentions(ev.get("text", ""), h) and not by_handle(ev, h["handle"])
    api = client(ns) if needs_thread else None
    ch = ev.get("channel") or (channel_id(ns, reg) if needs_thread and not ev.get("channel") else None)
    why = addressed(ev, h, api, ch)
    if ns.why:
        print(why or "not addressed", file=sys.stderr)
    return 0 if why else 1


def cmd_pickup(ns: argparse.Namespace, reg: dict) -> int:
    client(ns).call("reactions.add", channel=channel_id(ns, reg), timestamp=ns.thread, name="eyes")
    return 0


def cmd_done(ns: argparse.Namespace, reg: dict) -> int:
    h = speaker(reg, ns.handle)
    api, ch = client(ns), channel_id(ns, reg)
    _post(api, ch, h["handle"], f":white_check_mark: {ns.link}\n\n{_footer(h['handle'], ns.agent_url)}", ns.thread)
    api.call("reactions.add", channel=ch, timestamp=ns.thread, name="white_check_mark")
    return 0


def cmd_verify_author(ns: argparse.Namespace, reg: dict) -> int:
    """Fail closed: exit 0 only when the message, fetched from Slack by channel and ts, was written by an author the channel
    allows. With --dry-run the event's own fields are judged (a test of the allowlist, not of a real post)."""
    ev = parse_event(ns.event)
    if not ns.channel and ev.get("channel"):
        ns.channel = next((k for k, c in (reg.get("channels") or {}).items()
                           if ev["channel"] in (c.get("id"), os.environ.get("SLACK_CHANNEL_" + k.upper().replace("-", "_")))), None)
    key, ch = channel_key(ns, reg), channel_id(ns, reg)
    why = None
    if ev.get("channel") and ev["channel"] != ch:
        why_not = f"posted in {ev['channel']}, not #{key} ({ch})"
    elif not ev.get("ts"):
        why_not = "the event has no ts to look the message up by"
    else:
        msg = ev if ns.dry_run else fetch_message(client(ns), ch, ev["ts"], ev.get("thread_ts"))
        why = author_allowed(msg, reg, key) if msg else None
        why_not = "no such message in the channel" if msg is None else f"author {_who(msg)} is not allowed to post top-level in #{key}"
    if ns.why:
        print(f"allowed: {why}" if why else f"not allowed: {why_not}", file=sys.stderr)
    return 0 if why else 1


def cmd_announce(ns: argparse.Namespace, reg: dict) -> int:
    h = entry(reg, ns.handle)
    key = channel_key(ns, reg)
    ch = _owner_check(reg, key, h["handle"], "announces")
    if not ch.get("owner"):
        raise SlackError(f"#{key} has no owner: announcements go in a service channel")
    for pin in ns.ships:
        if not PIN_RE.match(pin):
            raise SlackError(f"--ships {pin!r}: pin code as URL#sha256=<64 hex> or git:<40-hex commit>")
    if not ns.ships and SHIPS_CODE_RE.search(f"{ns.action}\n{ns.verify}"):
        raise SlackError("the action or its verify command fetches or installs code: pin it with --ships URL#sha256=HEX or git:SHA")
    lines = [f":rotating_light: *Required action* from @{h['handle']}", ns.action.strip(), f"*{SAFE_POINT}*", f"*Verify:* `{ns.verify}`"]
    for pin in ns.ships:
        url, _, sha = pin.partition("#sha256=")
        lines.append(f"*Pinned:* commit `{pin[4:]}`" if pin.startswith("git:") else f"*Pinned:* {url} sha256 `{sha}` (don't run it if the checksum fails)")
    lines.append(f"*Confirm in this thread:* `research slack ack --channel {key} --thread <this message's ts> --as <you> --status done|blocked [--note ...]`")
    lines.append(_footer(h["handle"], ns.agent_url))
    _post(client(ns, _token_env(reg, key, h["handle"], True)), channel_id(ns, reg), h["handle"], "\n".join(lines))
    return 0


def cmd_ack(ns: argparse.Namespace, reg: dict) -> int:
    name = speaker(reg, ns.handle)["handle"]
    note = " ".join((ns.note or "").split())
    text = f"ACK {ns.status}" + (f" \u2014 {note}" if note else "") + "\n\n" + _footer(name, ns.agent_url)
    _post(client(ns), channel_id(ns, reg), name, text, ns.thread)
    return 0


def cmd_roster(ns: argparse.Namespace, reg: dict) -> int:
    msgs = client(ns).paged("conversations.replies", "messages", channel=channel_id(ns, reg), ts=ns.thread)
    acks: dict[str, dict] = {}
    for m in msgs:
        if m.get("ts") == ns.thread or not m.get("bot_id") or not m.get("username"):
            continue
        if mt := ACK_RE.match(m.get("text", "")):
            acks[m["username"]] = {"status": mt.group(1), "note": mt.group(2) or "", "ts": m["ts"]}
    owner = reg["channels"][channel_key(ns, reg)].get("owner")
    expected = [h["handle"] for h in reg["handles"] if h["handle"] != owner]
    rows = [{"name": n, "registered": True, **acks.get(n, {"status": "missing"})} for n in expected]
    rows += [{"name": n, "registered": False, **a} for n, a in sorted(acks.items()) if n not in expected]
    if ns.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"{r['status']:<8} @{r['name']}{'' if r['registered'] else ' (not in the registry)'}" + (f"  {r['note']}" if r.get("note") else ""))
    return 0 if all(r["status"] == "done" for r in rows if r["registered"]) else 1


def cmd_groups_sync(ns: argparse.Namespace, reg: dict, path: Path) -> int:
    """One user group per registry handle (`handle` = its handle, `name` = its handle, `description` = its remit), created,
    re-enabled or updated; with --apply the groups' ids are written back into the registry, which the caller commits."""
    api = client(ns)
    live = {g.get("handle"): g for g in api.call("usergroups.list", include_disabled="true").get("usergroups", [])}
    write = ns.apply and not ns.dry_run
    changed = False
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
    def common(channel: str | None) -> argparse.ArgumentParser:
        c = argparse.ArgumentParser(add_help=False)
        c.add_argument("--dry-run", action="store_true", help="send nothing to Slack; print each request instead")
        c.add_argument("--registry", default=None, help="the registry (default $RESEARCH_SLACK_REGISTRY or the checkout's)")
        c.add_argument("--channel", default=channel, metavar="KEY", help="a registry channel" + (f" (default {channel})" if channel else ""))
        c.add_argument("--channel-id", default=None, metavar="ID", help="the channel's Slack id (default $SLACK_CHANNEL_<KEY>, else the registry's)")
        return c

    def agent_url(p: argparse.ArgumentParser) -> None:
        p.add_argument("--agent-url", default=None, help="footer link to the posting agent (default $RESEARCH_AGENT_URL)")

    front, anyc = common(FRONT_DOOR), common(None)
    ap = argparse.ArgumentParser(prog="research slack", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("handles", parents=[anyc], help="list the registry")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("post", parents=[front], help="post as a handle; prints `<ts> <permalink>`")
    p.add_argument("--as", dest="handle", required=True, help="your handle, or your lane name if you have none")
    p.add_argument("--thread", default=None, metavar="TS")
    p.add_argument("--text", default=None)
    p.add_argument("--file", default=None, metavar="PATH", help="the message body from PATH (`-` or neither: stdin)")
    agent_url(p)
    p = sub.add_parser("read", parents=[front], help="channel or thread history, oldest first")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--thread", default=None, metavar="TS")
    g.add_argument("--since", default=None, metavar="TS")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("match", parents=[front], help="exit 0 if the event is addressed to the handle, 1 if not, 2 on error")
    p.add_argument("--handle", required=True, help="your handle, or your lane name if you have none")
    p.add_argument("--event", required=True, help="the message event as JSON, plain text, @PATH or - (stdin)")
    p.add_argument("--why", action="store_true", help="print the reason (or `not addressed`) on stderr")
    p = sub.add_parser("pickup", parents=[front], help="react :eyes: on a request you own")
    p.add_argument("--thread", required=True, metavar="TS")
    p = sub.add_parser("done", parents=[front], help="react :white_check_mark: on a request and reply with the result's link")
    p.add_argument("--as", dest="handle", required=True)
    p.add_argument("--thread", required=True, metavar="TS")
    p.add_argument("--link", required=True, metavar="URL", help="the result: a file, PR or art: id")
    agent_url(p)
    p = sub.add_parser("announce", parents=[anyc], help="the channel owner's one required action, with its verify command and pinned code")
    p.add_argument("--as", dest="handle", required=True, help="the channel's owner")
    p.add_argument("--action", required=True, help="the single action every subscriber takes")
    p.add_argument("--verify", required=True, metavar="CMD", help="the command that shows the action took")
    p.add_argument("--ships", action="append", default=[], metavar="URL#sha256=HEX|git:SHA", help="code the action runs, pinned")
    agent_url(p)
    p = sub.add_parser("ack", parents=[anyc], help="confirm an announcement in its thread")
    p.add_argument("--thread", required=True, metavar="TS")
    p.add_argument("--as", dest="handle", required=True, help="your handle, or your lane name if you have none")
    p.add_argument("--status", required=True, choices=["done", "blocked"])
    p.add_argument("--note", default=None)
    agent_url(p)
    p = sub.add_parser("roster", parents=[anyc], help="who acked an announcement and who didn't; exit 0 only when every handle is done")
    p.add_argument("--thread", required=True, metavar="TS")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("verify-author", parents=[anyc], help="exit 0 only when the post's author may post top-level in the channel")
    p.add_argument("--event", required=True, help="the message event as JSON, @PATH or - (stdin)")
    p.add_argument("--why", action="store_true", help="print the verdict on stderr")
    p = sub.add_parser("groups", help="the handles' Slack user groups")
    gsub = p.add_subparsers(dest="gcmd", required=True)
    p = gsub.add_parser("sync", parents=[anyc], help="create or update a user group per handle; a plan unless --apply")
    p.add_argument("--apply", action="store_true")
    ns = ap.parse_args(argv)
    try:
        path = registry_path(ns.registry)
        reg = load_registry(path)
        if ns.cmd == "groups":
            return cmd_groups_sync(ns, reg, path)
        return {"handles": cmd_handles, "post": cmd_post, "read": cmd_read, "match": cmd_match, "pickup": cmd_pickup, "done": cmd_done,
                "announce": cmd_announce, "ack": cmd_ack, "roster": cmd_roster, "verify-author": cmd_verify_author}[ns.cmd](ns, reg)
    except SlackError as e:
        print(f"research slack: {e}", file=sys.stderr)
        return 2

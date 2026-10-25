"""The queue's consumer: one message is a Slack event (or an interactivity payload), routed by `research msg`'s own code.

It keeps no state of its own.  The queue's lease keeps a message invisible while it runs (the SDK renews it), the
`event_id` idempotency key drops Slack's retries, and Slack is the record:

- A person's post that names no one gets a routing line in its thread, posted by `router` before any doorbell, naming the
  lanes picked and carrying the post's ts and those names in its metadata (ROUTING_LINE).  A redelivery reads the thread
  first: finding the line, it rings the names the line gives, with no second pick; `deliver`'s once-per-post check skips
  the lanes already rung.  Where the bot can't read the channel, a redelivery picks again and may post a second line.
- A post routed by the names it starts with needs no line: its doorbells are its record.
- A post through Verity's bot by an `external_posters` name (no `research msg` sender delivered it), in
  `#agent-coordination` too, rings by the rule alone: no routing line, and nothing when it names no one.
- A `canary` (the watchdog's synthetic event, hourly) stamps canary_done_at and touches nothing else.
- A `team_join` of someone `chat expect` published opens their chat (`newcomer`)."""
from __future__ import annotations

import argparse
import re
from typing import Any

from . import router
from .deps import CANARY_DONE, DEAD, DEAD_TOPIC, LAST_DONE, LAST_SUCCESS, Deps, error

MAX_DELIVERIES = 5
ASK_CHANNELS = ("ask-the-team", "resource-accounting")          # a registry channel by these keys routes as #ask-the-team
ROUTING_LINE = "verity_routing_line"                             # the routing line's metadata event_type
START_HERE = "https://app.notion.com/p/3fe399515d9e812bab1ee1a74d901d7d"   # the newcomers' onboarding page
NO_MODEL = "sent to @top: no lane named; routing by model is off"
MODEL_DOWN = "router: model unavailable, sent to @top"
NO_VALID = "router: no valid name, sent to @top"
FAILED = f"router: routing this post failed after {MAX_DELIVERIES} tries; sent to @top"
PERSON_SUBTYPES = (None, "thread_broadcast", "file_share")       # a person's post; edits, deletions and joins are not


class Retry(Exception):
    """Deliver the message again after `delay` seconds (`subscriber.py` raises Vercel's RetryAfter for it)."""

    def __init__(self, delay: float, why: str):
        super().__init__(why)
        self.delay = delay


def backoff(delivery_count: int) -> int:
    return min(300, 5 * 2 ** max(1, delivery_count))


class Lines:
    """`deliver`'s log, one JSON line per routing line (names and outcomes, never text)."""

    def __init__(self, deps: Deps, event_id: str):
        self.deps, self.event_id, self.lines = deps, event_id, []

    def write(self, s: str) -> None:
        for line in s.splitlines():
            if line.strip():
                self.lines.append(line)
                self.deps.log("deliver", event_id=self.event_id, line=line)

    def flush(self) -> None:
        pass


def consume(deps: Deps, payload: dict, delivery_count: int) -> str:
    """Process one delivery; returns the decision.  Raises Retry for a delivery to come, and after MAX_DELIVERIES failures
    acknowledges instead: it dead-letters the message, says so in the thread and rings @top."""
    eid = str(payload.get("event_id") or "")
    if not eid:
        deps.log("dropped", why="no event_id")
        return "dropped"
    try:
        decision = process(deps, payload)
    except Exception as e:  # noqa: BLE001 - every failure retries, then dead-letters
        deps.log("failed", event_id=eid, delivery=delivery_count, error=error(e))
        if delivery_count < MAX_DELIVERIES:
            raise Retry(backoff(delivery_count), error(e)) from e
        dead_letter(deps, payload, delivery_count, e)
        deps.write(LAST_DONE, deps.clock())
        return "dead-lettered"
    deps.write(LAST_SUCCESS, deps.clock())
    deps.write(LAST_DONE, deps.clock())
    deps.log("done", event_id=eid, decision=decision)
    return decision


def process(deps: Deps, payload: dict) -> str:
    if payload.get("kind") == "interactivity":
        # TODO(comms): #ask-daniel's buttons. The decision is recorded by the website's /api/agent-approvals route, which
        # only a card's asker can reach (Cursor OIDC); `ring_decision` needs that record. README "Needs".
        deps.log("interactivity-unhandled", event_id=payload.get("event_id"), type=(payload.get("payload") or {}).get("type"))
        return "interactivity-unhandled"
    if payload.get("kind") == "canary":                          # the watchdog's, proving the queue has a working consumer
        deps.write(CANARY_DONE, deps.clock())
        return "canary"
    if payload.get("kind") != "event":
        return "ignored: unknown kind"
    ev = payload.get("event") or {}
    if ev.get("type") == "team_join":
        return newcomer(deps, ev)
    subtypes = (*PERSON_SUBTYPES, "bot_message") if deps.rs.external_post(ev, deps.registry) else PERSON_SUBTYPES
    if ev.get("type") != "message" or ev.get("subtype") not in subtypes or not ev.get("channel") or not ev.get("ts"):
        return f"ignored: {ev.get('type')}/{ev.get('subtype')}"
    return route(deps, payload["event_id"], ev)


def registry_for(deps: Deps, channel: str) -> dict:
    """The bundled registry with each chat person people.d publishes that this channel needs: Slack names the channel
    (`conversations.info`), and its person is the published `people.d/NAME.json` whose channel id it is."""
    rs = deps.rs
    reg = rs.with_people(deps.registry, rs.ChatPeople().open())
    if rs.key_of(reg, channel) is not None:
        return reg
    try:
        name = (deps.slack.call("conversations.info", channel=channel).get("channel") or {}).get("name") or ""
    except rs.SlackError as e:
        deps.log("channel-unknown", channel=channel, error=error(e))
        return reg
    if name.startswith(rs.CHAT_PREFIX):
        r = rs.ChatPeople().get_person(name[len(rs.CHAT_PREFIX):], fresh=True)
        if r and not r.get("closed") and r["channel"]["id"] == channel:
            reg = rs.with_people(reg, [*reg.get("chat_people", []), r])
    return reg


def message(ev: dict) -> dict:
    return {k: ev[k] for k in ("ts", "thread_ts", "text", "user", "bot_id", "username", "reactions") if ev.get(k)}


def is_line_for(m: dict, ts: str, reg: dict, rs: Any) -> bool:
    """Whether `m` is our routing line for the post `ts`: posted by `router` on Verity's bot, its metadata naming the post."""
    meta = m.get("metadata") or {}
    return m.get("username") == "router" and rs.ours(m, reg) and meta.get("event_type") == ROUTING_LINE \
        and (meta.get("event_payload") or {}).get("post") == ts


def line_names(m: dict, reg: dict) -> list[str]:
    """The names a routing line gives: its metadata's, else the `@name`s in its text; the handles among them, else @top."""
    handles = {h["handle"] for h in reg["handles"]}
    got = ((m.get("metadata") or {}).get("event_payload") or {}).get("names")
    if not isinstance(got, list):
        got = re.findall(r"@([a-z0-9][a-z0-9._-]*)", m.get("text") or "")
    return [n for n in dict.fromkeys(got) if isinstance(n, str) and n in handles] or ["top"]


def routing_line(deps: Deps, ch: str, thread: str, ts: str, reg: dict) -> dict | None:
    """The routing line already posted for the post `ts` (in the thread `thread`), or None, also where the bot can't read
    the channel."""
    rs = deps.rs
    try:
        msgs = deps.slack.paged("conversations.replies", "messages", channel=ch, ts=thread, include_all_metadata="true")
    except rs.SlackError as e:
        if not rs._unreadable(e):
            raise
        deps.log("routing-line-unreadable", channel=ch, error=error(e))
        return None
    return next((m for m in msgs if is_line_for(m, ts, reg, rs)), None)


def post_line(deps: Deps, ch: str, thread: str, ts: str, text: str, names: list[str]) -> None:
    deps.slack.call("chat.postMessage", channel=ch, thread_ts=thread, username="router", text=text,
                    metadata={"event_type": ROUTING_LINE, "event_payload": {"post": ts, "names": names}},
                    unfurl_links="false", unfurl_media="false")


def route(deps: Deps, eid: str, ev: dict) -> str:
    """A person's post in a chat or ask channel: `deliver`, as `research msg deliver --channel KEY --event` does.  An
    unaddressed one rings the names of its routing line: the one already in its thread, else a new one naming the
    model's picks, which in no-model mode is @top."""
    rs, api, ch = deps.rs, deps.slack, ev["channel"]
    reg = registry_for(deps, ch)
    key = rs.key_of(reg, ch)
    chat = key in rs.chat_channels(reg)
    msg = message(ev)
    external = rs.external_post(msg, reg)                          # no sender delivered it: it rings by the rule alone
    if key == rs.FRONT_DOOR and not external:
        return "front door: the sender delivers"
    if not chat and key not in ASK_CHANNELS and key != rs.FRONT_DOOR:
        return f"not routed here: {key or ch}"
    if rs.person(msg, reg) is None and not external:
        return "not a person's post"
    thread = msg.get("thread_ts") if msg.get("thread_ts") not in (None, msg["ts"]) else None
    text = msg.get("text") or ""
    replies: list[dict] = []
    if external:
        unaddressed = False
    elif chat:
        replies = api.paged("conversations.replies", "messages", channel=ch, ts=thread) if thread else []
        unaddressed = rs.chat_targets(msg, replies, reg) == [rs.CHAT_WATCHER] and rs.CHAT_WATCHER not in rs.lead_names(text, reg)
    else:
        try:
            root = api.root(ch, thread) if thread else None
        except rs.SlackError as e:
            if not rs._unreadable(e):
                raise
            root = None
        replies = [root] if root else []
        unaddressed = not rs.route_targets(msg, root, reg) and not rs.is_quiet(text)
    names, how = None, "rule"
    if unaddressed:
        found = routing_line(deps, ch, thread or msg["ts"], msg["ts"], reg)
        if found:
            names, how = line_names(found, reg), "its routing line"
        else:
            names, line = pick(deps, msg, replies, reg)
            post_line(deps, ch, thread or msg["ts"], msg["ts"], line, names)
            how = "picked"
        deps.log("unaddressed", event_id=eid, names=names, how=how)
    out = Lines(deps, eid)
    rs.deliver(api, reg, rs.Routes(), ch, msg, now=deps.clock(), log=out, names=names)
    if any("NOT RUNG" in s for s in out.lines):
        raise rs.SlackError("a doorbell didn't ring")                  # `deliver` again rings only the ones still missing
    return "unaddressed: " + " ".join("@" + n for n in names) if names else "rule"


def pick(deps: Deps, msg: dict, thread: list[dict], reg: dict) -> tuple[list[str], str]:
    """The names an unaddressed post rings and the routing line that says so: the model's valid handles (never @top), else
    @top."""
    try:
        got = router.route_unaddressed(msg, thread, reg)
    except Exception as e:  # noqa: BLE001 - a model failure sends the post to @top, never drops it
        deps.log("router-error", error=error(e))
        return ["top"], MODEL_DOWN
    if got is None:
        return ["top"], NO_MODEL
    handles = {h["handle"] for h in reg["handles"]} - {deps.rs.CHAT_WATCHER}
    names = [n for n in dict.fromkeys(got) if isinstance(n, str) and n in handles][:3]
    if not names:
        return ["top"], NO_VALID
    return names, "routed to " + " ".join("@" + n for n in names)


def newcomer(deps: Deps, ev: dict) -> str:
    """A `team_join`: the new member's email (`users.info`, scope users:read.email) against the people `chat expect`
    published.  One of them gets their chat, by `chat open`'s own path with START_HERE bookmarked; anyone else, nothing."""
    rs, user = deps.rs, ev.get("user")
    uid = user.get("id") if isinstance(user, dict) else user
    if not isinstance(uid, str) or not rs.SLACK_USER_RE.match(uid):
        return "ignored: team_join without a user"
    info = deps.slack.call("users.info", user=uid).get("user") or {}
    email = str((info.get("profile") or {}).get("email") or "").strip().lower()
    if info.get("is_bot") or info.get("deleted") or not email:
        return "team_join: not a person with an email"
    r = next((r for r in rs.ChatPeople().expected() if r["email"].strip().lower() == email), None)
    if not r:
        return "team_join: not expected"
    name = r["channel"]["key"][len(rs.CHAT_PREFIX):]
    rs.cmd_chat(argparse.Namespace(verb="open", name=name, user=uid, email=None, start_here=START_HERE, note=None,
                                   cursor_bot=None, dry_run=False, api=deps.slack), deps.registry)
    return f"team_join: opened #{r['channel']['key']}"


def dead_letter(deps: Deps, payload: dict, delivery_count: int, e: BaseException) -> None:
    """Park the message on DEAD_TOPIC and count it (best effort, in the cache health reads), then, best effort too, a
    failure line in its thread, which is its routing line from now on, and @top rung."""
    eid = str(payload["event_id"])
    deps.queue.send(DEAD_TOPIC, {"payload": payload, "error": type(e).__name__, "deliveries": delivery_count},
                    idempotency_key=eid)
    rec = deps.read(DEAD) or {"count": 0, "since": deps.clock()}
    deps.write(DEAD, {"count": int(rec.get("count") or 0) + 1, "since": rec.get("since")})
    deps.log("dead-lettered", event_id=eid, delivery=delivery_count, error=error(e))
    ev: dict[str, Any] = payload.get("event") or {}
    if payload.get("kind") != "event" or ev.get("type") != "message" or not ev.get("channel") or not ev.get("ts"):
        return
    rs, msg = deps.rs, message(ev)
    try:
        post_line(deps, ev["channel"], msg.get("thread_ts") or msg["ts"], msg["ts"], FAILED, ["top"])
    except Exception as x:  # noqa: BLE001 - best effort: Slack may be what failed
        deps.log("dead-letter-line-failed", event_id=eid, error=error(x))
    try:
        rs.deliver(deps.slack, registry_for(deps, ev["channel"]), rs.Routes(), ev["channel"], msg, now=deps.clock(),
                   log=Lines(deps, eid), names=["top"])
    except Exception as x:  # noqa: BLE001
        deps.log("dead-letter-ring-failed", event_id=eid, error=error(x))

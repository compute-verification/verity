"""The queue's consumer: one message is a Slack event (or an interactivity payload), routed by `research msg`'s own code.

Each `event_id` goes through a leased state machine in Redis: `processing:<owner>` set with NX and a LEASE_S expiry, which
a later delivery takes over once it has expired; `done`, kept DONE_S, written only after the doorbells and the thread line
are posted.  `deliver`'s once-per-post check stays the last guard against a double ring."""
from __future__ import annotations

import uuid
from typing import Any

from . import router
from .deps import DEAD_COUNT, DEAD_TOPIC, LAST_SUCCESS, NS, PENDING, Deps, error

LEASE_S = 120                                                    # past the consumer function's maxDuration (vercel.json)
DONE_S = 7 * 86400
MAX_DELIVERIES = 5
ASK_CHANNELS = ("ask-the-team", "resource-accounting")          # a registry channel by these keys routes as #ask-the-team
NO_MODEL = "sent to @top: no lane named; routing by model is off"
MODEL_DOWN = "router: model unavailable, sent to @top"
NO_VALID = "router: no valid name, sent to @top"
FAILED = f"router: routing this post failed after {MAX_DELIVERIES} tries; sent to @top"
PERSON_SUBTYPES = (None, "thread_broadcast", "file_share")       # a person's post; edits, deletions and joins are not


class Retry(Exception):
    """Deliver the message again after `delay` seconds (api/queue.py raises Vercel's RetryAfter for it)."""

    def __init__(self, delay: float, why: str):
        super().__init__(why)
        self.delay = delay


def backoff(delivery_count: int) -> int:
    return min(300, 5 * 2 ** max(1, delivery_count))


def event_key(event_id: str) -> str:
    return f"{NS}event:{event_id}"


def post_key(channel: str, ts: str) -> str:
    """The record that a person's post was routed: reconciliation enqueues a post without one."""
    return f"{NS}post:{channel}:{ts}"


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
    key, store = event_key(eid), deps.store
    state = store.get(key)
    if state in ("done", "dead"):
        store.zrem(PENDING, eid)
        deps.log("duplicate", event_id=eid, state=state)
        return "duplicate"
    owner = f"processing:{uuid.uuid4().hex}"
    if not store.claim(key, owner, LEASE_S):
        deps.log("lease-held", event_id=eid, delivery=delivery_count)
        raise Retry(LEASE_S, "another delivery holds the lease")       # never a failure: its holder may still finish
    try:
        decision = process(deps, payload)
    except Exception as e:  # noqa: BLE001 - every failure retries, then dead-letters
        store.release(key, owner)
        deps.log("failed", event_id=eid, delivery=delivery_count, error=error(e))
        if delivery_count < MAX_DELIVERIES:
            raise Retry(backoff(delivery_count), error(e)) from e
        dead_letter(deps, payload, delivery_count, e)
        store.set(key, "dead", DONE_S)
        store.zrem(PENDING, eid)
        return "dead-lettered"
    store.set(key, "done", DONE_S)
    store.zrem(PENDING, eid)
    store.set(LAST_SUCCESS, f"{deps.clock():.3f}")
    deps.log("done", event_id=eid, decision=decision)
    return decision


def process(deps: Deps, payload: dict) -> str:
    if payload.get("kind") == "interactivity":
        # TODO(comms): #ask-daniel's buttons. The decision is recorded by the website's /api/agent-approvals route, which
        # only a card's asker can reach (Cursor OIDC); `ring_decision` needs that record. README "Not built yet".
        deps.log("interactivity-unhandled", event_id=payload.get("event_id"), type=(payload.get("payload") or {}).get("type"))
        return "interactivity-unhandled"
    if payload.get("kind") != "event":
        return "ignored: unknown kind"
    ev = payload.get("event") or {}
    if ev.get("type") != "message" or ev.get("subtype") not in PERSON_SUBTYPES or not ev.get("channel") or not ev.get("ts"):
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


def route(deps: Deps, eid: str, ev: dict) -> str:
    """A person's post in a chat or ask channel: `deliver`, as `research msg deliver --channel KEY --event` does; an
    unaddressed one goes to the model's seam, which in no-model mode sends it to @top with a routing line in its thread."""
    rs, api, ch = deps.rs, deps.slack, ev["channel"]
    if deps.store.get(post_key(ch, ev["ts"])):
        return "already routed"                                         # Slack's event and reconciliation's, both delivered
    reg = registry_for(deps, ch)
    key = rs.key_of(reg, ch)
    chat = key in rs.chat_channels(reg)
    if key == rs.FRONT_DOOR:
        return "front door: the sender delivers"
    if not chat and key not in ASK_CHANNELS:
        return f"not routed here: {key or ch}"
    msg = message(ev)
    if rs.person(msg, reg) is None:
        return "not a person's post"
    thread = msg.get("thread_ts") if msg.get("thread_ts") not in (None, msg["ts"]) else None
    text = msg.get("text") or ""
    replies: list[dict] = []
    if chat:
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
    names, line = pick(deps, msg, replies, reg) if unaddressed else (None, None)
    out = Lines(deps, eid)
    rs.deliver(api, reg, rs.Routes(), ch, msg, now=deps.clock(), log=out, names=names)
    if any("NOT RUNG" in s for s in out.lines):
        raise rs.SlackError("a doorbell didn't ring")                  # `deliver` again rings only the ones still missing
    if line:
        api.call("chat.postMessage", channel=ch, thread_ts=thread or msg["ts"], username="router", text=line,
                 unfurl_links="false", unfurl_media="false")
    decision = "unaddressed: " + " ".join("@" + n for n in names) if names else "rule"
    deps.store.set(post_key(ch, msg["ts"]), decision, DONE_S)
    return decision


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


def dead_letter(deps: Deps, payload: dict, delivery_count: int, e: BaseException) -> None:
    """Park the message on DEAD_TOPIC (health counts it), then, best effort, a failure line in its thread and @top rung."""
    eid = str(payload["event_id"])
    deps.queue.send(DEAD_TOPIC, {"payload": payload, "error": type(e).__name__, "deliveries": delivery_count},
                    idempotency_key=eid)
    deps.store.incr(DEAD_COUNT)
    deps.log("dead-lettered", event_id=eid, delivery=delivery_count, error=error(e))
    ev: dict[str, Any] = payload.get("event") or {}
    if payload.get("kind") != "event" or not ev.get("channel") or not ev.get("ts"):
        return
    rs, api, msg = deps.rs, deps.slack, message(ev)
    try:
        api.call("chat.postMessage", channel=ev["channel"], thread_ts=msg.get("thread_ts") or msg["ts"], username="router",
                 text=FAILED, unfurl_links="false")
    except Exception as x:  # noqa: BLE001 - best effort: Slack may be what failed
        deps.log("dead-letter-line-failed", event_id=eid, error=error(x))
    try:
        rs.deliver(api, registry_for(deps, ev["channel"]), rs.Routes(), ev["channel"], msg, now=deps.clock(),
                   log=Lines(deps, eid), names=["top"])
    except Exception as x:  # noqa: BLE001
        deps.log("dead-letter-ring-failed", event_id=eid, error=error(x))

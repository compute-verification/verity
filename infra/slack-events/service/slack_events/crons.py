"""The two crons.  Both read Slack and keep no state of their own (the reminders they ring and the lines they post are
what they find next time); both stamp last_success_at in the cache.

- `watchdog` (every 5 minutes): a person's chat post that rang someone and has no 👀 and no lane reply after REMIND_S rings
  again; after ESCALATE_S with no lane reply it rings @top and says so in the thread; in the run that falls in DAILY_AT, a
  list of the chat threads still open goes to @top's inbox.  And `stall`: an event published STALL_S ago or more with no
  message done by the consumer since rings @comms (at most hourly), since health's other numbers can't show it; and
  `canary`, each hour, publishes a synthetic event that only the consumer can stamp done, so `stall` has one to judge.
- `reconcile` (every 10 minutes): each chat and ask channel's last hour.  A person's post that should ring someone and has
  neither a routing line nor a doorbell in any inbox is enqueued again, keyed on its channel and ts, so a second pass
  changes nothing.  The oldest such post is what health reports as `oldest_unprocessed_age_s`."""
from __future__ import annotations

from datetime import datetime, timezone

from .consumer import ASK_CHANNELS, is_line_for
from .deps import (
    CANARY_SENT,
    DEAD,
    LAST_DONE,
    LAST_ENQUEUED,
    LAST_SUCCESS,
    OLDEST,
    STALL_RUNG,
    TOPIC,
    Deps,
    error,
)

REMIND_S, ESCALATE_S = 15 * 60, 60 * 60
WINDOW_S = 86400                                                 # how far back a post counts as open
RECONCILE_S, SETTLE_S = 3600, 120                                # its window, and a post younger than SETTLE_S is in flight
OLDEST_TTL_S = 30 * 60                                           # reconcile's finding, unrefreshed for this long, is unknown
DAILY_AT = (16, 0)                                               # UTC hour and minute: 9 AM PDT; the run within 5 minutes after
EVERY_S = 5 * 60                                                 # the watchdog's schedule (vercel.json)
REMINDER = "no 👀 or lane reply in 15 min"
ESCALATION = "no lane reply in 60 min"
ESCALATED = "router: no lane has replied in 60 min; sent to @top"
DAILY_HEAD = "from watchdog, chat threads still open (daily):"
STALL_S, STALL_EVERY_S = 10 * 60, 60 * 60                         # waiting this long with nothing done: no working consumer
STALL_TO = "comms"                                               # who owns the service
STALL_HEAD = "from slack-events, the queue stalled:"
CANARY_EVERY_S = 60 * 60                                         # the synthetic event's period, so a quiet day still proves the drain


def chats(deps: Deps) -> tuple[dict, list[str]]:
    """The registry with every open chat's person, and their channel ids: the bot's private chat-* channels whose person
    people.d publishes."""
    rs = deps.rs
    people = rs.ChatPeople()
    persons = []
    for c in deps.slack.paged("conversations.list", "channels", types="private_channel", exclude_archived="true"):
        name = c.get("name") or ""
        if name.startswith(rs.CHAT_PREFIX) and (r := people.get_person(name[len(rs.CHAT_PREFIX):], fresh=True)) \
                and not r.get("closed") and r["channel"]["id"] == c.get("id"):
            persons.append(r)
    reg = rs.with_people(deps.registry, persons)
    return reg, [r["channel"]["id"] for r in reg.get("chat_people", [])]


def threads(deps: Deps, ch: str, oldest: float) -> list[list[dict]]:
    """Each thread of `ch` with a post since `oldest`, oldest first: its root, then its replies (a lone post is a thread
    of one)."""
    api, out = deps.slack, []
    for root in sorted(api.paged("conversations.history", "messages", channel=ch, oldest=f"{oldest - WINDOW_S:.6f}"),
                       key=lambda m: float(m["ts"])):
        if root.get("reply_count") or root.get("thread_ts") == root["ts"]:
            if float(root.get("latest_reply") or root["ts"]) < oldest:
                continue
            msgs = api.paged("conversations.replies", "messages", channel=ch, ts=root["ts"], include_all_metadata="true")
        elif float(root["ts"]) < oldest:
            continue
        else:
            msgs = [root]
        out.append(sorted(msgs, key=lambda m: float(m["ts"])))
    return out


def by_person(deps: Deps, m: dict, reg: dict) -> bool:
    return not m.get("bot_id") and m.get("subtype") is None and deps.rs.person(m, reg) is not None


def watchdog(deps: Deps) -> dict:
    rs, api, now = deps.rs, deps.slack, deps.clock()
    reg, ids = chats(deps)
    handles = {h["handle"] for h in reg["handles"]}
    counts, still_open = {"reminded": 0, "escalated": 0}, []
    for ch in ids:
        for thread in threads(deps, ch, now - WINDOW_S):
            root = thread[0]["ts"]
            for p in (m for m in thread if by_person(deps, m, reg)):
                age, later = now - float(p["ts"]), [m for m in thread if float(m["ts"]) > float(p["ts"])]
                if age < REMIND_S or any(rs.ours(m, reg) and m.get("username") in handles for m in later):
                    continue
                if not rs.chat_targets(p, [m for m in thread if m is not p], reg):
                    continue                                               # it rang no one (quiet, naming no one)
                still_open.append((ch, root, age))
                msg = {**{k: p[k] for k in ("ts", "text", "user") if k in p}, "thread_ts": p.get("thread_ts") or root}
                if age >= ESCALATE_S:
                    if not any(rs.ours(m, reg) and m.get("username") == "router" and (m.get("text") or "").startswith(ESCALATED)
                               for m in later):
                        lines = rs.deliver(api, reg, rs.Routes(), ch, msg, now=now, log=_Log(deps), names=["top"],
                                           reminder=ESCALATION)
                        if any("NOT RUNG" in s for s in lines):
                            continue                                       # the line would stop the next run's try
                        api.call("chat.postMessage", channel=ch, thread_ts=root, username="router", text=ESCALATED,
                                 unfurl_links="false")
                        counts["escalated"] += 1
                elif not any(r.get("name") == "eyes" for r in p.get("reactions") or []):
                    rs.deliver(api, reg, rs.Routes(), ch, msg, now=now, log=_Log(deps), reminder=REMINDER)
                    counts["reminded"] += 1
    if _daily_due(now):
        counts["daily"] = daily(deps, reg, still_open, now)
    counts["canary"] = canary(deps, now)
    counts["stalled"] = stall(deps, now)
    deps.write(LAST_SUCCESS, now)
    deps.log("watchdog", open=len(still_open), **counts)
    return {"open": len(still_open), **counts}


def canary(deps: Deps, now: float) -> bool:
    """Publish a synthetic event each CANARY_EVERY_S, stamped as an enqueue, so the stall check proves the consumer even on
    a day no person posts (no person posted in the people's channels for the drain's first two days)."""
    sent = deps.read(CANARY_SENT)
    if isinstance(sent, (int, float)) and now - sent < CANARY_EVERY_S:
        return False
    eid = f"canary-{int(now)}"
    try:
        deps.queue.send(TOPIC, {"kind": "canary", "event_id": eid, "source": "watchdog"}, idempotency_key=eid)
    except Exception as e:  # noqa: BLE001 - the next run tries again
        deps.log("canary-failed", error=error(e))
        return False
    deps.write(CANARY_SENT, now)
    deps.write(LAST_ENQUEUED, now)
    return True


def stall(deps: Deps, now: float) -> bool:
    """Ring @comms when the last event published is STALL_S old and the consumer has finished no message since (health
    looks clean while nothing is consumed).  At most every STALL_EVERY_S; an unknown time rings nothing."""
    enq, done, rung = (deps.read(k) for k in (LAST_ENQUEUED, LAST_DONE, STALL_RUNG))
    if not isinstance(enq, (int, float)) or now - enq < STALL_S or (isinstance(done, (int, float)) and done >= enq):
        return False
    if isinstance(rung, (int, float)) and now - rung < STALL_EVERY_S:
        return False
    rs = deps.rs
    to, r = rs.Routes().resolve(STALL_TO)
    if why := rs.unreachable(STALL_TO, to, r):
        deps.log("stall-unrung", to=STALL_TO, why=why)
        return False
    def at(t):
        return datetime.fromtimestamp(t, timezone.utc).strftime("%d %b %H:%MZ") if isinstance(t, (int, float)) else "unknown"
    deps.slack.call("chat.postMessage", channel=r.get("channel") or rs.channel_id_for(deps.registry, rs.FRONT_DOOR),
                    thread_ts=r["inbox_ts"], username="router", unfurl_links="false",
                    text=f"{STALL_HEAD} the last event published at {at(enq)}, the consumer's last message done at "
                         f"{at(done)}. Check /api/cron/drain and the deployment's logs (infra/slack-events).")
    deps.write(STALL_RUNG, now)
    deps.log("stalled", enqueued=enq, done=done)
    return True


def _daily_due(now: float) -> bool:
    t = datetime.fromtimestamp(now, timezone.utc)
    start = t.replace(hour=DAILY_AT[0], minute=DAILY_AT[1], second=0, microsecond=0).timestamp()
    return start <= now < start + EVERY_S


def daily(deps: Deps, reg: dict, still_open: list[tuple[str, str, float]], now: float) -> int:
    """The open threads, one line to @top's inbox, as `ring_decision` posts a decision there; once a day (a second run in
    the same window finds it)."""
    rs, api = deps.rs, deps.slack
    if not still_open:
        return 0
    to, r = rs.Routes().resolve(rs.CHAT_WATCHER)
    if not r or r.get("closed"):
        deps.log("daily-skipped", why=f"@{rs.CHAT_WATCHER} has no open inbox")
        return 0
    inbox = r.get("channel") or rs.channel_id_for(reg, rs.FRONT_DOOR)
    recent = api.paged("conversations.replies", "messages", channel=inbox, ts=r["inbox_ts"], oldest=f"{now - 3600:.6f}")
    if any(rs.ours(m, reg) and (m.get("text") or "").startswith(DAILY_HEAD) for m in recent):
        return 0
    threads_ = sorted({(ch, root): age for ch, root, age in still_open}.items())
    names = {c["id"]: k for k, c in rs.chat_channels(reg).items()}
    text = DAILY_HEAD + " " + "; ".join(f"#{names.get(ch, ch)} thread {root} ({age / 3600:.1f} h)" for (ch, root), age in threads_)
    api.call("chat.postMessage", channel=inbox, thread_ts=r["inbox_ts"], username="router", text=text, unfurl_links="false")
    return len(threads_)


def doorbells(deps: Deps, reg: dict, routes, oldest: float) -> set[tuple[str | None, str]]:
    """Every doorbell rung since `oldest`, as (the channel it names, the post's ts), from each inbox thread."""
    rs, api, out = deps.rs, deps.slack, set()
    front = rs.channel_id_for(reg, rs.FRONT_DOOR)
    for inbox in sorted(routes.inboxes()):
        try:
            msgs = api.paged("conversations.replies", "messages", channel=front, ts=inbox, oldest=f"{oldest:.6f}")
        except rs.SlackError as e:
            if not rs._unreadable(e):
                raise
            continue
        for m in msgs:
            if m.get("ts") != inbox and rs.is_doorbell(m, reg) and (d := rs.DOORBELL_RE.match(m.get("text") or "")):
                out.add((rs.doorbell_channel(m["text"]), d[4] or d[3]))
    return out


def should_ring(deps: Deps, p: dict, thread: list[dict], reg: dict, chat: bool, routes) -> bool:
    """Whether the consumer leaves a record of `p`: a routing line (an unaddressed post) or a doorbell for a lane it names
    that has an inbox.  A quiet post naming no one, or one naming only people, leaves none."""
    rs = deps.rs
    others = [m for m in thread if m is not p]
    if chat:
        targets = rs.chat_targets(p, others, reg)
    else:
        root = thread[0] if p.get("thread_ts") not in (None, p["ts"]) else None
        targets = rs.route_targets(p, root, reg)
        if not targets:
            return not rs.is_quiet(p.get("text") or "")
    humans, author = rs.people_names(reg), rs.poster(p, reg)
    return any(n not in humans and n != author and not rs.unreachable(n, *routes.resolve(n)) for n in targets)


def reconcile(deps: Deps) -> dict:
    rs, now = deps.rs, deps.clock()
    reg, ids = chats(deps)
    ids += [c["id"] for k, c in reg["channels"].items() if k in ASK_CHANNELS and c.get("id")]
    routes = rs.Routes()
    routes.refresh([h["handle"] for h in reg["handles"]])
    rung = doorbells(deps, reg, routes, now - RECONCILE_S)
    chat_ids = {c["id"] for c in rs.chat_channels(reg).values()}
    queued, unreadable, oldest = 0, [], None
    for ch in ids:
        try:
            found = threads(deps, ch, now - RECONCILE_S)
        except rs.SlackError as e:
            if not rs._unreadable(e):
                raise
            unreadable.append(ch)
            deps.log("reconcile-unreadable", channel=ch, error=error(e))
            continue
        where = rs.key_of(reg, ch) or ch
        for thread in found:
            for p in (m for m in thread if by_person(deps, m, reg)):
                if not now - RECONCILE_S <= float(p["ts"]) <= now or (where, p["ts"]) in rung \
                        or any(is_line_for(m, p["ts"], reg, rs) for m in thread) \
                        or not should_ring(deps, p, thread, reg, ch in chat_ids, routes):
                    continue
                oldest = p["ts"] if oldest is None or float(p["ts"]) < float(oldest) else oldest
                if float(p["ts"]) > now - SETTLE_S:
                    continue                                       # in flight: the consumer may not have reached it
                eid = f"reconcile:{ch}:{p['ts']}"
                ev = {**{k: p[k] for k in ("type", "ts", "thread_ts", "text", "user", "reactions") if k in p},
                      "type": "message", "channel": ch}
                deps.queue.send(TOPIC, {"kind": "event", "event_id": eid, "event": ev, "source": "reconcile"},
                                idempotency_key=eid)
                queued += 1
                deps.log("reconciled", event_id=eid, channel=ch, ts=p["ts"])
    if not unreadable:
        deps.write(OLDEST, {"ts": oldest, "at": now}, OLDEST_TTL_S)
    if deps.read(DEAD) is None:
        deps.write(DEAD, {"count": 0, "since": now})                # counted from here: an evicted count reads as unknown till now
    deps.write(LAST_SUCCESS, now)
    deps.log("reconcile", queued=queued, channels=len(ids), unreadable=len(unreadable))
    return {"queued": queued, "channels": len(ids), "unreadable": unreadable}


class _Log:
    def __init__(self, deps: Deps):
        self.deps = deps

    def write(self, s: str) -> None:
        for line in s.splitlines():
            if line.strip():
                self.deps.log("deliver", line=line)

    def flush(self) -> None:
        pass

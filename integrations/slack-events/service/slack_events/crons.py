"""The two crons.  Both read Slack and keep no state of their own (the reminders they ring and the lines they post are
what they find next time), and both stamp last_success_at.

- `watchdog` (every 5 minutes): a person's chat post that rang someone and has no 👀 and no lane reply after REMIND_S rings
  again; after ESCALATE_S with no lane reply it rings @top and says so in the thread; in the run that falls in DAILY_AT, a
  list of the chat threads still open goes to @top's inbox.
- `reconcile` (every 10 minutes): each chat and ask channel's last hour; a person's post the consumer has no record of and
  that has no routing line is enqueued again, keyed on its channel and ts, so a second pass changes nothing."""
from __future__ import annotations

from datetime import datetime, timezone

from .consumer import ASK_CHANNELS, post_key
from .deps import LAST_SUCCESS, PENDING, TOPIC, Deps, error

REMIND_S, ESCALATE_S = 15 * 60, 60 * 60
WINDOW_S = 86400                                                 # how far back a post counts as open
RECONCILE_S, SETTLE_S = 3600, 120                                # its window, and a post younger than SETTLE_S is in flight
DAILY_AT = (16, 0)                                               # UTC hour and minute: 9 AM PDT; the run within 5 minutes after
EVERY_S = 5 * 60                                                 # the watchdog's schedule (vercel.json)
REMINDER = "no 👀 or lane reply in 15 min"
ESCALATION = "no lane reply in 60 min"
ESCALATED = "router: no lane has replied in 60 min; sent to @top"
DAILY_HEAD = "from watchdog, chat threads still open (daily):"


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
            msgs = api.paged("conversations.replies", "messages", channel=ch, ts=root["ts"])
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
    deps.store.set(LAST_SUCCESS, f"{now:.3f}")
    deps.log("watchdog", open=len(still_open), **counts)
    return {"open": len(still_open), **counts}


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


def reconcile(deps: Deps) -> dict:
    rs, now = deps.rs, deps.clock()
    reg, ids = chats(deps)
    ids += [c["id"] for k, c in reg["channels"].items() if k in ASK_CHANNELS and c.get("id")]
    queued, unreadable = 0, []
    for ch in ids:
        try:
            found = threads(deps, ch, now - RECONCILE_S)
        except rs.SlackError as e:
            if not rs._unreadable(e):
                raise
            unreadable.append(ch)
            deps.log("reconcile-unreadable", channel=ch, error=error(e))
            continue
        for thread in found:
            for p in (m for m in thread if by_person(deps, m, reg)):
                if not now - RECONCILE_S <= float(p["ts"]) <= now - SETTLE_S or deps.store.get(post_key(ch, p["ts"])):
                    continue
                if any(rs.ours(m, reg) and m.get("username") == "router" and float(m["ts"]) > float(p["ts"]) for m in thread):
                    continue
                eid = f"reconcile:{ch}:{p['ts']}"
                ev = {**{k: p[k] for k in ("type", "ts", "thread_ts", "text", "user", "reactions") if k in p},
                      "type": "message", "channel": ch}
                deps.queue.send(TOPIC, {"kind": "event", "event_id": eid, "event": ev, "source": "reconcile"},
                                idempotency_key=eid)
                deps.store.zadd(PENDING, eid, now)
                queued += 1
                deps.log("reconciled", event_id=eid, channel=ch, ts=p["ts"])
    deps.store.set(LAST_SUCCESS, f"{now:.3f}")
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

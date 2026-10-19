"""The watchdog and reconciliation, against Slack as they find it."""
from __future__ import annotations

from datetime import datetime, timezone

from conftest import BOT, CHAT, DANIEL, LUKE, TEAM
from slack_events import crons
from slack_events.consumer import NO_MODEL, consume
from slack_events.crons import DAILY_HEAD, ESCALATE_S, ESCALATED, ESCALATION, REMIND_S, REMINDER, SETTLE_S
from slack_events.deps import LAST_SUCCESS, PENDING, TOPIC


def private(ts: str, reminder: str | None = None) -> str:
    return (f"from luke, #chat-luke thread {ts}: " + (f"reminder ({reminder}): " if reminder else "")
            + f"(a post in a private channel: research msg read --channel chat-luke --thread {ts})")


def routed(world, text: str, **kw) -> dict:
    post = world.slack.add(CHAT, text=text, user=LUKE, **kw)
    consume(world.deps, world.event(CHAT, post), 1)
    return post


def test_a_chat_post_with_no_eyes_and_no_lane_reply_rings_again_after_15_min_once(world):
    post = routed(world, "@circuits is the TP8 sweep done?")
    world.clock.t += REMIND_S - 1
    assert crons.watchdog(world.deps)["reminded"] == 0
    world.clock.t += 2
    assert crons.watchdog(world.deps)["reminded"] == 1
    world.clock.t += 300
    crons.watchdog(world.deps)
    assert world.rung("circuits") == [private(post["ts"]), private(post["ts"], REMINDER)] and world.rung("top") == []
    assert world.deps.store.get(LAST_SUCCESS) == f"{world.clock():.3f}"


def test_eyes_or_a_lane_reply_stops_the_reminder(world):
    eyed = routed(world, "@circuits is the TP8 sweep done?")
    eyed["reactions"] = [{"name": "eyes", "users": ["U0CURSOR"], "count": 1}]
    answered = routed(world, "@circuits and the Lean side?")
    world.slack.add(CHAT, text="it is", username="circuits", bot_id=BOT, thread_ts=answered["ts"])
    world.clock.t += REMIND_S + 60
    assert crons.watchdog(world.deps) == {"open": 1, "reminded": 0, "escalated": 0}
    assert world.rung("circuits") == [private(eyed["ts"]), private(answered["ts"])]


def test_after_60_min_with_no_lane_reply_it_rings_top_and_says_so_in_the_thread_once(world):
    post = routed(world, "@circuits is the TP8 sweep done?")
    post["reactions"] = [{"name": "eyes", "users": ["U0CURSOR"], "count": 1}]
    world.clock.t += ESCALATE_S + 1
    assert crons.watchdog(world.deps)["escalated"] == 1
    world.clock.t += 300
    assert crons.watchdog(world.deps)["escalated"] == 0
    assert world.rung("top") == [private(post["ts"], ESCALATION)]
    assert [m["text"] for m in world.slack.thread(CHAT, post["ts"]) if m.get("username") == "router"] == [ESCALATED]


def test_an_unaddressed_post_already_at_top_is_reminded_to_top(world):
    post = routed(world, "how do I read a PoUW receipt?")
    world.clock.t += REMIND_S + 1
    crons.watchdog(world.deps)
    assert world.rung("top") == [private(post["ts"]), private(post["ts"], REMINDER)]


def test_once_a_day_top_gets_the_list_of_open_threads(world):
    day = datetime.fromtimestamp(world.clock(), timezone.utc).replace(hour=crons.DAILY_AT[0], minute=0, second=0)
    world.clock.t = day.timestamp() - 2 * 3600
    post = routed(world, "@circuits is the TP8 sweep done?")
    post["reactions"] = [{"name": "eyes", "users": ["U0CURSOR"], "count": 1}]
    world.clock.t = day.timestamp() + 60
    assert crons.watchdog(world.deps)["daily"] == 1
    world.clock.t += 120
    assert crons.watchdog(world.deps)["daily"] == 0
    world.clock.t += crons.EVERY_S
    assert "daily" not in crons.watchdog(world.deps)
    daily = [t for t in world.rung("top") if t.startswith(DAILY_HEAD)]
    assert daily == [f"{DAILY_HEAD} #chat-luke thread {post['ts']} (2.0 h)"]


def test_reconciliation_enqueues_a_missed_post_once_and_the_consumer_routes_it(world):
    missed = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    asked = world.slack.add(TEAM, text="what is the TP8 status?", user=DANIEL)
    world.clock.t += SETTLE_S + 1
    young = world.slack.add(CHAT, text="still there?", user=LUKE)
    assert crons.reconcile(world.deps)["queued"] == 2
    assert crons.reconcile(world.deps)["queued"] == 2                      # sent again, and the queue keeps the first
    sent = world.deps.queue.on(TOPIC)
    assert [p["event_id"] for p in sent] == [f"reconcile:{CHAT}:{missed['ts']}", f"reconcile:{TEAM}:{asked['ts']}"]
    assert young["ts"] not in {p["event"]["ts"] for p in sent} and world.deps.store.zmin(PENDING) is not None
    for p in sent:
        consume(world.deps, p, 1)
    assert [m["text"] for m in world.slack.thread(CHAT, missed["ts"]) if m.get("username") == "router"] == [NO_MODEL]
    assert len(world.rung("top")) == 2 and world.deps.store.zmin(PENDING) is None
    world.clock.t += SETTLE_S + 1
    before = len(world.deps.queue.sent)
    assert crons.reconcile(world.deps)["queued"] == 1                      # only the young post, now settled
    assert len(world.deps.queue.sent) == before + 1


def test_reconciliation_skips_a_post_with_a_routing_line_and_a_routed_one(world):
    routed(world, "how do I read a PoUW receipt?")
    other = world.slack.add(CHAT, text="anyone?", user=LUKE)
    world.slack.add(CHAT, text=NO_MODEL, username="router", bot_id=BOT, thread_ts=other["ts"])
    world.clock.t += SETTLE_S + 1
    assert crons.reconcile(world.deps)["queued"] == 0
    assert world.deps.store.get(LAST_SUCCESS) == f"{world.clock():.3f}"


def test_reconciliation_outside_its_hour_leaves_old_posts(world):
    world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    world.clock.t += crons.RECONCILE_S + 1
    assert crons.reconcile(world.deps)["queued"] == 0

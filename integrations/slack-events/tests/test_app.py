"""The HTTP routes: Slack's signature, the synchronous publish and its 503, the health route and the crons' bearer."""
from __future__ import annotations

import hashlib
import hmac
import json
import urllib.parse

import pytest
from fastapi.testclient import TestClient

from conftest import BOT, CHAT, LUKE, SIGNING, T0
from slack_events import app as app_mod
from slack_events.deps import DEAD_COUNT, LAST_SUCCESS, PENDING, TOPIC


def client(world) -> TestClient:
    return TestClient(app_mod.create_app(lambda: world.deps))


def sign(body: bytes, ts: float, secret: str = SIGNING) -> dict:
    t = str(int(ts))
    return {"X-Slack-Request-Timestamp": t,
            "X-Slack-Signature": "v0=" + hmac.new(secret.encode(), f"v0:{t}:".encode() + body, hashlib.sha256).hexdigest()}


def post_event(c: TestClient, world, data: dict, *, ts: float | None = None, secret: str = SIGNING):
    body = json.dumps(data).encode()
    return c.post("/api/slack/events", content=body,
                  headers={"Content-Type": "application/json", **sign(body, world.clock() if ts is None else ts, secret)})


def callback(eid: str = "Ev1", **ev) -> dict:
    return {"type": "event_callback", "team_id": "T1", "event_id": eid,
            "event": {"type": "message", "channel": CHAT, "user": LUKE, "ts": f"{int(T0)}.000100", "text": "hello", **ev}}


def test_a_signed_event_is_published_keyed_on_its_event_id_and_answered_200(world):
    r = post_event(client(world), world, callback())
    assert r.status_code == 200 and r.json() == {"ok": True}
    [(topic, payload, key)] = world.deps.queue.sent
    assert (topic, key) == (TOPIC, "Ev1") and payload["kind"] == "event" and payload["event"]["channel"] == CHAT
    assert world.deps.store.zmin(PENDING) == T0


def test_url_verification_answers_the_challenge_and_publishes_nothing(world):
    r = post_event(client(world), world, {"type": "url_verification", "challenge": "abc123"})
    assert r.status_code == 200 and r.text == "abc123" and world.deps.queue.sent == []


@pytest.mark.parametrize("case,why", [("bad", "bad signature"), ("stale", "stale"), ("future", "stale"),
                                      ("unsigned", "unsigned"), ("no secret", "SLACK_SIGNING_SECRET is not set")])
def test_a_request_slack_did_not_sign_now_is_refused(world, case, why):
    c, data = client(world), callback()
    if case == "bad":
        r = post_event(c, world, data, secret="dummy-other-secret")
    elif case == "stale":
        r = post_event(c, world, data, ts=T0 - app_mod.MAX_SKEW_S - 1)
    elif case == "future":
        r = post_event(c, world, data, ts=T0 + app_mod.MAX_SKEW_S + 1)
    elif case == "unsigned":
        r = c.post("/api/slack/events", content=json.dumps(data).encode())
    else:
        world.deps.env = {}
        r = post_event(c, world, data)
    assert r.status_code == 401 and r.json() == {"error": why} and world.deps.queue.sent == []


def test_a_request_just_inside_the_skew_is_accepted(world):
    assert post_event(client(world), world, callback(), ts=T0 - app_mod.MAX_SKEW_S).status_code == 200


def test_a_signature_over_another_body_is_refused(world):
    body, other = json.dumps(callback()).encode(), json.dumps(callback(text="changed")).encode()
    r = client(world).post("/api/slack/events", content=other, headers=sign(body, T0))
    assert r.status_code == 401 and world.deps.queue.sent == []


def test_our_own_bot_posts_are_dropped_before_the_queue(world):
    r = post_event(client(world), world, callback(bot_id=BOT, username="circuits", user=None))
    assert r.status_code == 200 and r.json()["ignored"] == "our bot's post" and world.deps.queue.sent == []


def test_a_failed_publish_answers_503_so_slack_retries(world):
    world.deps.queue.fail = True
    r = post_event(client(world), world, callback())
    assert r.status_code == 503 and world.deps.queue.sent == []
    assert any(line["event"] == "publish-failed" and line["event_id"] == "Ev1" for line in world.logs())


def test_slacks_retry_of_an_event_enqueues_it_once(world):
    c = client(world)
    for _ in range(3):
        assert post_event(c, world, callback()).status_code == 200
    assert len(world.deps.queue.sent) == 1


def test_an_interactivity_payload_is_published_keyed_on_its_action(world):
    p = {"type": "block_actions", "trigger_id": "tr1", "actions": [{"action_id": "decide-a", "action_ts": "1800000000.1"}]}
    body = urllib.parse.urlencode({"payload": json.dumps(p)}).encode()
    c = client(world)
    for _ in range(2):
        r = c.post("/api/slack/interactivity", content=body,
                   headers={"Content-Type": "application/x-www-form-urlencoded", **sign(body, T0)})
        assert r.status_code == 200
    [(topic, payload, key)] = world.deps.queue.sent
    assert topic == TOPIC and key == "interactivity:decide-a:1800000000.1" and payload["payload"] == p
    bad = c.post("/api/slack/interactivity", content=body, headers=sign(body, T0, "dummy-other-secret"))
    assert bad.status_code == 401


def test_health_reports_the_queue_with_its_bearer_and_refuses_without(world):
    c, store = client(world), world.deps.store
    store.zadd(PENDING, "Ev1", T0 - 30)
    store.zadd(PENDING, "Ev2", T0 - 5)
    store.incr(DEAD_COUNT)
    store.set(LAST_SUCCESS, f"{T0 - 2:.3f}")
    r = c.get("/api/slack/health", headers={"Authorization": "Bearer dummy-health"})
    assert r.status_code == 200 and r.json() == {"oldest_unprocessed_age_s": 30.0, "dead_letter_count": 1,
                                                 "last_success_at": T0 - 2, "queue": "slack-events",
                                                 "dead_letter": "slack-events-dead"}
    assert c.get("/api/slack/health").status_code == 401
    assert c.get("/api/slack/health", headers={"Authorization": "Bearer dummy-wrong"}).status_code == 401
    world.deps.env = {}
    assert c.get("/api/slack/health", headers={"Authorization": "Bearer dummy-health"}).status_code == 503


def test_health_of_an_empty_queue(world):
    r = client(world).get("/api/slack/health", headers={"Authorization": "Bearer dummy-health"})
    assert r.json() == {"oldest_unprocessed_age_s": 0, "dead_letter_count": 0, "last_success_at": None,
                        "queue": "slack-events", "dead_letter": "slack-events-dead"}


@pytest.mark.parametrize("name", ["watchdog", "reconcile"])
def test_the_crons_run_only_with_vercels_cron_secret(world, name):
    c = client(world)
    assert c.get(f"/api/cron/{name}").status_code == 401
    r = c.get(f"/api/cron/{name}", headers={"Authorization": "Bearer dummy-cron"})
    assert r.status_code == 200 and world.deps.store.get(LAST_SUCCESS) == f"{T0:.3f}"

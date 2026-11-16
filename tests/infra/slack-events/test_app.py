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
from slack_events.deps import DEAD, LAST_DONE, LAST_ENQUEUED, LAST_SUCCESS, OLDEST, SLACK_HEARD, TOPIC


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
    assert world.deps.cache.get(LAST_ENQUEUED) == T0                         # the watchdog's stall check reads it


def test_url_verification_answers_the_challenge_and_publishes_nothing(world):
    r = post_event(client(world), world, {"type": "url_verification", "challenge": "abc123"})
    assert r.status_code == 200 and r.text == "abc123" and world.deps.queue.sent == []


def test_every_signed_request_stamps_slack_heard_at_ignored_ones_too_and_a_refused_one_never(world):
    """Reconciliation's silence check reads it: any signed request proves Slack still calls, whatever the service does next."""
    c = client(world)
    assert post_event(c, world, callback(), secret="dummy-other-secret").status_code == 401
    assert world.deps.cache.get(SLACK_HEARD) is None
    for data in ({"type": "url_verification", "challenge": "abc"}, callback(bot_id=BOT, username="circuits", user=None),
                 {"type": "app_rate_limited"}):
        world.clock.t += 60
        assert post_event(c, world, data).status_code == 200 and world.deps.cache.get(SLACK_HEARD) == world.clock()
    assert world.deps.queue.sent == []


def test_slacks_late_event_and_reconciliations_find_of_one_post_are_both_queued_and_it_rings_once(world):
    """Reconciliation queues under its own key, so a post Slack's event reaches late is queued twice; the second delivery
    finds the first's routing line and doorbell and adds neither."""
    from slack_events import crons
    from slack_events.consumer import consume
    post = world.slack.add(CHAT, text="how do I read a PoUW receipt?", user=LUKE)
    world.clock.t += crons.SETTLE_S + 1
    assert crons.reconcile(world.deps)["queued"] == 1
    data = {"type": "event_callback", "event_id": "EvLate", "event": {"type": "message", "channel": CHAT, **post}}
    assert post_event(client(world), world, data).status_code == 200
    assert [k for _, _, k in world.deps.queue.sent] == [f"reconcile:{CHAT}:{post['ts']}", "EvLate"]
    assert [consume(world.deps, p, 1) for p in world.deps.queue.on(TOPIC)] == ["unaddressed: @top"] * 2
    assert len(world.rung("top")) == 1 and len(world.lines(CHAT, post["ts"])) == 1


@pytest.mark.parametrize("who", ["person", "external"])
def test_a_post_whose_event_was_queued_and_lost_is_requeued_by_reconciliation_once(world, who):
    """Slack's event reached the queue and was then lost with no trace (no doorbell, no routing or dead-letter line).
    Under its own key reconciliation still queues the post, once while the queue keeps that key, and its delivery is
    the record."""
    from conftest import COORD
    from slack_events import crons
    from slack_events.consumer import consume
    ch, by = (CHAT, {"user": LUKE}) if who == "person" else (COORD, {"username": "Codex", "bot_id": BOT,
                                                                       "subtype": "bot_message"})
    post = world.slack.add(ch, text="@circuits is the TP8 sweep done?", **by)
    data = {"type": "event_callback", "event_id": "EvLost", "event": {"type": "message", "channel": ch, **post}}
    assert post_event(client(world), world, data).status_code == 200                  # queued, never consumed
    world.clock.t += crons.SETTLE_S + 1
    for _ in range(3):                                                     # every run finds it; the queue keeps one
        assert crons.reconcile(world.deps)["queued"] == 1
        world.clock.t += 600
    assert [k for _, _, k in world.deps.queue.sent] == ["EvLost", f"reconcile:{ch}:{post['ts']}"]
    assert consume(world.deps, world.deps.queue.on(TOPIC)[1], 1) == "rule" and len(world.rung("circuits")) == 1
    assert crons.reconcile(world.deps)["queued"] == 0                      # rung: the doorbell is the record


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
    r = post_event(client(world), world, callback(bot_id=BOT, username="Codex", user=None, subtype="bot_message"))
    assert r.status_code == 200 and "ignored" not in r.json() and len(world.deps.queue.sent) == 1   # no sender delivered it


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


def health(c: TestClient) -> dict:
    r = c.get("/api/slack/health", headers={"Authorization": "Bearer dummy-health"})
    assert r.status_code == 200
    body = r.json()
    assert body.pop("note") == app_mod.HEALTH_NOTE
    return body


def test_health_reports_what_the_cache_holds_with_its_bearer_and_refuses_without(world):
    c, cache = client(world), world.deps.cache
    cache.set(OLDEST, {"ts": f"{T0 - 30:.6f}", "at": T0 - 5})
    cache.set(DEAD, {"count": 1, "since": T0 - 86400})
    cache.set(LAST_SUCCESS, T0 - 2)
    cache.set(LAST_DONE, T0 - 600)
    cache.set(LAST_ENQUEUED, T0 - 60)
    cache.set(SLACK_HEARD, T0 - 7)
    assert health(c) == {"notes_token_source": "env", "notes_token_why": "no OIDC identity", "oldest_unprocessed_age_s": 30.0, "dead_letter_count": 1, "dead_letter_since": T0 - 86400,
                         "last_success_at": T0 - 2, "last_done_at": T0 - 600, "last_enqueued_at": T0 - 60,
                         "canary_sent_at": None, "canary_done_at": None, "slack_heard_at": T0 - 7,
                         "queue": "slack-events", "dead_letter": "slack-events-dead"}
    cache.set(OLDEST, {"ts": None, "at": T0})
    assert health(c)["oldest_unprocessed_age_s"] == 0.0
    assert c.get("/api/slack/health").status_code == 401
    assert c.get("/api/slack/health", headers={"Authorization": "Bearer dummy-wrong"}).status_code == 401
    world.deps.env = {}
    assert c.get("/api/slack/health", headers={"Authorization": "Bearer dummy-health"}).status_code == 503



def test_health_says_the_notes_token_comes_from_the_broker_once_it_mints_one(world, monkeypatch):
    rs = world.deps.rs
    monkeypatch.setattr(rs, "_notes_token", {})
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "dummy-oidc")
    answers = [OSError("down"), {"token": "dummy-minted", "expires_at": "2099-01-01T00:00:00Z"}]

    def http_json(method, url, data, headers):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a
    monkeypatch.setattr(rs, "http_json", http_json)
    c = client(world)
    h = health(c)                                                          # the broker refused: the old token still serves
    assert h["notes_token_source"] == "env" and h["notes_token_why"] == "the broker refused (down)"
    monkeypatch.setattr(rs, "_notes_token", {})
    h = health(c)
    assert h["notes_token_source"] == "broker" and h["notes_token_why"] is None and rs.notes_token() == "dummy-minted"


def test_the_identity_the_service_passes_in_reaches_the_broker_without_the_environment_variable(world, monkeypatch):
    rs, seen = world.deps.rs, []
    monkeypatch.setattr(rs, "_notes_token", {})
    monkeypatch.delenv("VERCEL_OIDC_TOKEN", raising=False)

    def http_json(method, url, data, headers):
        seen.append(headers["Authorization"])
        return {"token": "dummy-minted", "expires_at": "2099-01-01T00:00:00Z"}
    monkeypatch.setattr(rs, "http_json", http_json)
    monkeypatch.setattr(rs, "OIDC_TOKEN", lambda: "dummy-request-oidc")
    h = health(client(world))
    assert h["notes_token_source"] == "broker" and seen == ["Bearer dummy-request-oidc"]

    def no_identity():
        raise RuntimeError("no x-vercel-oidc-token header")
    monkeypatch.setattr(rs, "_notes_token", {})
    monkeypatch.setattr(rs, "OIDC_TOKEN", no_identity)
    h = health(client(world))
    assert h["notes_token_source"] == "env" and h["notes_token_why"] == "no OIDC identity (RuntimeError)" and len(seen) == 1

def test_a_cold_or_unreachable_cache_reads_as_unknown_not_as_down(world):
    unknown = {"notes_token_source": "env", "notes_token_why": "no OIDC identity", "oldest_unprocessed_age_s": None, "dead_letter_count": None, "dead_letter_since": None,
               "last_success_at": None, "last_done_at": None, "last_enqueued_at": None, "canary_sent_at": None,
               "canary_done_at": None, "slack_heard_at": None, "queue": "slack-events", "dead_letter": "slack-events-dead"}
    assert health(client(world)) == unknown
    world.deps.cache.set(LAST_SUCCESS, T0)
    world.deps.cache.fail = True
    assert health(client(world)) == unknown
    world.deps.cache.fail = False
    world.deps.cache.set(OLDEST, {"ts": f"{T0:.6f}", "at": T0}, {"ttl": 60})
    world.clock.t += 61                                                    # reconcile's finding, not refreshed: unknown
    assert health(client(world))["oldest_unprocessed_age_s"] is None


@pytest.mark.parametrize("name", ["watchdog", "reconcile"])
def test_the_crons_run_only_with_vercels_cron_secret(world, name):
    c = client(world)
    assert c.get(f"/api/cron/{name}").status_code == 401
    r = c.get(f"/api/cron/{name}", headers={"Authorization": "Bearer dummy-cron"})
    assert r.status_code == 200 and world.deps.cache.get(LAST_SUCCESS) == T0


def test_the_drain_runs_only_with_vercels_cron_secret_and_reports_what_it_handled(world, monkeypatch):
    from slack_events import subscriber

    async def drain(d):
        return {"handled": 2, "failed": 0}
    monkeypatch.setattr(subscriber, "drain", drain)
    c = client(world)
    assert c.get("/api/cron/drain").status_code == 401
    r = c.get("/api/cron/drain", headers={"Authorization": "Bearer dummy-cron"})
    assert r.status_code == 200 and r.json() == {"handled": 2, "failed": 0}

"""The HTTP side (`api/index.py`): Slack's Events and Interactivity requests, the health route and the three crons.

Slack's requests are checked (HMAC SHA-256 of `v0:{timestamp}:{raw body}` with SLACK_SIGNING_SECRET, at most MAX_SKEW_S
old), published to the queue synchronously, keyed on Slack's id, and answered 200; a publish that fails answers 503, so
Slack retries.  Nothing runs after the answer: the `drain` cron consumes the queue (`subscriber.py`)."""
from __future__ import annotations

import hashlib
import hmac
import json
import urllib.parse
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from . import crons, subscriber
from .deps import (CANARY_DONE, CANARY_SENT, DEAD, DEAD_TOPIC, LAST_DONE, LAST_ENQUEUED, LAST_SUCCESS, OLDEST, TOPIC, Deps,
                   error, production)

MAX_SKEW_S = 300
HEALTH_NOTE = ("from Vercel's Runtime Cache, best effort: null is unknown (a cold or evicted cache), not down; "
               "oldest_unprocessed_age_s is reconcile's last finding, dead_letter_count counts since dead_letter_since; "
               "last_done_at is the consumer's alone (last_success_at counts the crons too), and a last_enqueued_at "
               "after it means events wait; canary_* is the watchdog's hourly synthetic event and its consumption")


def signed(secret: str | None, ts: str | None, sig: str | None, body: bytes, now: float) -> str | None:
    """Why the request isn't Slack's, or None when it is.  No secret refuses everything."""
    if not secret:
        return "SLACK_SIGNING_SECRET is not set"
    if not ts or not sig:
        return "unsigned"
    try:
        age = abs(now - int(ts))
    except ValueError:
        return "bad timestamp"
    if age > MAX_SKEW_S:
        return "stale"
    want = "v0=" + hmac.new(secret.encode(), b"v0:" + ts.encode() + b":" + body, hashlib.sha256).hexdigest()
    return None if hmac.compare_digest(want, sig) else "bad signature"


def bearer(request: Request, secret: str | None) -> bool:
    got = request.headers.get("authorization") or ""
    return bool(secret) and hmac.compare_digest(got.encode(), f"Bearer {secret}".encode())


def create_app(deps: Callable[[], Deps] = production) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    async def verified(request: Request) -> tuple[Deps, bytes, JSONResponse | None]:
        d, body = deps(), await request.body()
        why = signed(d.env.get("SLACK_SIGNING_SECRET"), request.headers.get("x-slack-request-timestamp"),
                     request.headers.get("x-slack-signature"), body, d.clock())
        if why:
            d.log("refused", route=request.url.path, why=why)
            return d, body, JSONResponse({"error": why}, status_code=401)
        return d, body, None

    async def enqueue(d: Deps, payload: dict) -> JSONResponse:
        eid = payload["event_id"]
        try:
            await run_in_threadpool(d.queue.send, TOPIC, payload, idempotency_key=eid)
        except Exception as e:  # noqa: BLE001 - Slack retries a 5xx, keyed on the same id
            d.log("publish-failed", event_id=eid, error=error(e))
            return JSONResponse({"error": "publish failed"}, status_code=503)
        await run_in_threadpool(d.write, LAST_ENQUEUED, d.clock())
        d.log("enqueued", event_id=eid, kind=payload["kind"])
        return JSONResponse({"ok": True})

    @app.post("/api/slack/events")
    async def events(request: Request):
        d, body, refused = await verified(request)
        if refused:
            return refused
        try:
            data = json.loads(body)
        except ValueError:
            return JSONResponse({"error": "not JSON"}, status_code=400)
        if data.get("type") == "url_verification":
            return PlainTextResponse(str(data.get("challenge") or ""))
        ev, eid = data.get("event") or {}, data.get("event_id")
        if data.get("type") != "event_callback" or not eid:
            return JSONResponse({"ok": True, "ignored": data.get("type")})
        if ev.get("bot_id") and ev.get("bot_id") == d.registry.get("bot_id") and not d.rs.external_post(ev, d.registry):
            return JSONResponse({"ok": True, "ignored": "our bot's post"})
        return await enqueue(d, {"kind": "event", "event_id": eid, "event": ev, "team_id": data.get("team_id"),
                                 "source": "events"})

    @app.post("/api/slack/interactivity")
    async def interactivity(request: Request):
        d, body, refused = await verified(request)
        if refused:
            return refused
        try:
            p = json.loads(urllib.parse.parse_qs(body.decode()).get("payload", [""])[0])
        except ValueError:
            return JSONResponse({"error": "no payload"}, status_code=400)
        action = (p.get("actions") or [{}])[0]
        if action.get("action_id") and action.get("action_ts"):
            key = f"interactivity:{action['action_id']}:{action['action_ts']}"
        elif p.get("trigger_id"):
            key = f"interactivity:{p['trigger_id']}"
        else:
            return JSONResponse({"error": "no action id"}, status_code=400)
        return await enqueue(d, {"kind": "interactivity", "event_id": key, "payload": p, "source": "interactivity"})

    @app.get("/api/slack/health")
    async def health(request: Request):
        d = deps()
        token = d.env.get("HEALTH_TOKEN")
        if not token:
            return JSONResponse({"error": "HEALTH_TOKEN is not set"}, status_code=503)
        if not bearer(request, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        oldest, dead, last, done, enq, c_sent, c_done = [
            await run_in_threadpool(d.read, k)
            for k in (OLDEST, DEAD, LAST_SUCCESS, LAST_DONE, LAST_ENQUEUED, CANARY_SENT, CANARY_DONE)]
        age = None
        if isinstance(oldest, dict) and "ts" in oldest:
            age = round(max(0.0, d.clock() - float(oldest["ts"])), 3) if oldest["ts"] else 0.0
        dead = dead if isinstance(dead, dict) else {}
        return JSONResponse({"oldest_unprocessed_age_s": age, "dead_letter_count": dead.get("count"),
                             "dead_letter_since": dead.get("since"),
                             "last_success_at": float(last) if isinstance(last, (int, float)) else None,
                             "last_done_at": float(done) if isinstance(done, (int, float)) else None,
                             "last_enqueued_at": float(enq) if isinstance(enq, (int, float)) else None,
                             "canary_sent_at": float(c_sent) if isinstance(c_sent, (int, float)) else None,
                             "canary_done_at": float(c_done) if isinstance(c_done, (int, float)) else None,
                             "queue": TOPIC, "dead_letter": DEAD_TOPIC, "note": HEALTH_NOTE})

    def cron(name: str, run: Callable[[Deps], dict]):
        async def route(request: Request):
            d = deps()
            if not bearer(request, d.env.get("CRON_SECRET")):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            try:
                return JSONResponse(await run_in_threadpool(run, d))
            except Exception as e:  # noqa: BLE001 - the next run tries again; health's last_success_at goes stale
                d.log(f"{name}-failed", error=error(e))
                return JSONResponse({"error": error(e)}, status_code=500)
        app.add_api_route(f"/api/cron/{name}", route, methods=["GET"])

    cron("watchdog", crons.watchdog)
    cron("reconcile", crons.reconcile)

    @app.get("/api/cron/drain")
    async def drain(request: Request):
        d = deps()
        if not bearer(request, d.env.get("CRON_SECRET")):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            return JSONResponse(await subscriber.drain(d))
        except Exception as e:  # noqa: BLE001 - the next minute's run polls again
            d.log("drain-failed", error=error(e))
            return JSONResponse({"error": error(e)}, status_code=500)
    return app

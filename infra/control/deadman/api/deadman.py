"""The control pod's dead-man's switch (INF-131): a Vercel function that reads the heartbeat the watchers supervisor upserts into
Neon (`store_private.control_heartbeat`, one row per host) and says in #agent-alerts, naming @infra, when it has gone stale: the
pod, or the supervisor on it, stopped, and with it every watcher. It also reads the cloud watch's last pass from the same row
(`cloud_watch_t`), since the cloud watch can't watch itself.

    GET /api/deadman                                  {"ok", "host", "t", "age_s"}: 200 while fresh, 503 when stale (no details)
    GET /api/deadman  Authorization: Bearer $CRON_SECRET    the cron: the same check, plus each stale finding's post

Vercel's cron (vercel.json) calls it every EVERY_S and sends $CRON_SECRET. It keeps no state: a finding is posted on the first
call that sees it past its limit and again every REPEAT_S while it lasts, so a pod down for a day posts 1 + 24/REPEAT_H times.
Environment: STORE_PRIVATE_READER_URL (Neon's `store_private_reader` connection string; the function reads it with Neon's HTTP
SQL endpoint), SLACK_BOT_TOKEN, CRON_SECRET, and optionally DEADMAN_HOST (default vy-control-verity). Standard library only.
"""
from __future__ import annotations

import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler

EVERY_S = 600
STALE_S = 15 * 60               # the supervisor upserts every minute
CLOUD_WATCH_STALE_S = 30 * 60   # the cloud watch passes every few minutes
REPEAT_S = 6 * 3600
ALERTS = "C0C5Q30MCNA"          # #agent-alerts (.agents/skills/using-slack/registry.json)
INFRA = "<!subteam^S0C5APDSYF9>"
TABLE = "store_private.control_heartbeat"


def epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def neon_row(url: str, host: str) -> dict | None:
    """The heartbeat row for `host` ({"t", "body"}), through Neon's HTTP SQL endpoint; None when there is none."""
    db_host = urllib.parse.urlsplit(url).hostname
    req = urllib.request.Request(f"https://{db_host}/sql", method="POST",
                                 data=json.dumps({"query": f"SELECT t, body FROM {TABLE} WHERE host = $1", "params": [host]}).encode(),
                                 headers={"Neon-Connection-String": url, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        rows = json.load(r).get("rows") or []
    return rows[0] if rows else None


def findings(row: dict | None, now: float) -> list[tuple[str, float | None]]:
    """(what is stale, for how long past its limit) for each limit passed: the heartbeat, and the cloud watch's last pass. No
    row at all is stale but never posted (None): the table is created by the supervisor's first heartbeat."""
    if row is None:
        return [("no heartbeat row at all: the watchers supervisor has never published one", None)]
    t = epoch(row.get("t"))
    if t is None or now - t >= STALE_S:
        return [(f"the control pod's watchers heartbeat is {(now - (t or 0)) / 60:.0f} min old: the pod or its supervisor is down, "
                 "and with it every watcher on it", now - (t or 0) - STALE_S)]
    try:
        body = json.loads(row.get("body") or "{}")
    except ValueError:
        body = {}
    cw = epoch(body.get("cloud_watch_t"))
    if cw is not None and now - cw >= CLOUD_WATCH_STALE_S:
        return [(f"the cloud watch's last pass was {(now - cw) / 60:.0f} min ago while the pod's heartbeat is fresh: the cloud "
                 "watch is down", now - cw - CLOUD_WATCH_STALE_S)]
    return []


def due(past: float) -> bool:
    """Whether the cron call that finds a finding this far past its limit posts it: the first call, then one every REPEAT_S."""
    return past % REPEAT_S < EVERY_S


def post(token: str, text: str) -> None:
    req = urllib.request.Request("https://slack.com/api/chat.postMessage", method="POST",
                                 data=json.dumps({"channel": ALERTS, "text": text, "username": "deadman",
                                                  "unfurl_links": False}).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=20) as r:
        out = json.load(r)
    if not out.get("ok"):
        raise RuntimeError(f"chat.postMessage: {out.get('error')}")


def check(env: dict, now: float, cron: bool, read=neon_row, send=post) -> tuple[int, dict]:
    host = env.get("DEADMAN_HOST") or "vy-control-verity"
    try:
        row = read(env["STORE_PRIVATE_READER_URL"], host)
    except Exception as e:  # noqa: BLE001 (Neon unreadable is not the pod down: said, never posted)
        return 502, {"ok": False, "host": host, "error": f"heartbeat unreadable: {type(e).__name__}"}
    found = findings(row, now)
    t = epoch((row or {}).get("t"))
    out = {"ok": not found, "host": host, "t": (row or {}).get("t"), "age_s": round(now - t) if t else None}
    if cron:
        out["posted"] = []
        for what, past in found:
            if past is not None and due(past):
                send(env["SLACK_BOT_TOKEN"], f"{INFRA} *Dead-man's switch:* {what}. Reach it with `research pods ssh "
                                             f"vy-control-verity`; its boot starts the watchers (/data/boot/post_start.sh). "
                                             f"(infra/control/deadman)")
                out["posted"].append(what)
    return (200 if not found else 503), out


def authorized(header: str | None, secret: str | None) -> bool:
    return bool(secret) and hmac.compare_digest(header or "", f"Bearer {secret}")


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        env = dict(os.environ)
        cron = authorized(self.headers.get("Authorization"), env.get("CRON_SECRET"))
        try:
            code, out = check(env, time.time(), cron)
        except Exception as e:  # noqa: BLE001
            code, out = 500, {"ok": False, "error": type(e).__name__}
        body = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

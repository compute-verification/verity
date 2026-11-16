"""Deliver only the verified comms inbox to platform-agents. Slack credentials stay in this service."""
from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime
from typing import Any

from .deps import Deps, vercel_oidc

ENDPOINT = "COMMS_RELAY_URL"                                     # the platform's relay route: the project's variable holds it
SEAT = "808f8ee2-e60b-42e8-a960-9c7a8329cdfa"
RECEIPT = "verity_comms_relay_reply"
MAX_WAIT_S = 30 * 60
LINEAR_RING = re.compile(r"^from [a-z0-9][a-z0-9._-]*, Linear [A-Z][A-Z0-9]*-[0-9]+\b")


class Pending(Exception):
    """The model has the event; let the same queue delivery return for the answer."""


class Unsupported(Exception):
    """The platform owns this inbox but cannot safely handle this delivery."""


def candidate(d: Deps, ev: dict) -> bool:
    inbox = d.env.get("COMMS_RELAY_INBOX")
    if not (d.env.get(ENDPOINT) or "").startswith("https://"):
        return False
    if not inbox or ev.get("type") != "message" or ev.get("subtype") not in (None, "bot_message"):
        return False
    if inbox != f"{ev.get('channel')}/{ev.get('thread_ts')}" or ev.get("ts") == ev.get("thread_ts"):
        return False
    try:
        end = datetime.fromisoformat(d.env["COMMS_RELAY_UNTIL"].replace("Z", "+00:00"))
        if end.tzinfo is None or not re.fullmatch(r"\d{10}\.\d{6}", ev.get("ts", "")):
            return False
        if not end.timestamp() - 86400 <= float(ev["ts"]) <= d.clock() + 5 or not 0 < end.timestamp() - d.clock() <= 86400:
            return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    # A doorbell is a post from the router on our bot. Its author or shape alone is insufficient.
    text = ev.get("text") or ""
    return d.rs.is_doorbell(ev, d.registry) and bool(d.rs.DOORBELL_RE.match(text) or LINEAR_RING.match(text))


def exchange(url: str, data: dict) -> dict:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: Any) -> None:
            return None

    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={
        "Authorization": "Bearer " + vercel_oidc(), "Content-Type": "application/json"})
    with urllib.request.build_opener(NoRedirect()).open(req, timeout=10) as response:
        return json.load(response)


def forward(d: Deps, ev: dict) -> str:
    route = d.rs.Routes().get_route("comms", fresh=True)
    if not route or not route.get("holder") or route.get("closed") or (route.get("channel"), route.get("inbox_ts")) != (ev["channel"], ev["thread_ts"]):
        raise RuntimeError("cannot verify comms inbox")
    bell = d.rs.DOORBELL_RE.match(ev["text"])
    # A ring about another channel can contain a private pointer. Never follow it or
    # reply across that boundary. Leave its original doorbell in Slack for the holder.
    if not bell or d.rs.doorbell_channel(ev["text"]) not in (None, d.rs.FRONT_DOOR, ev["channel"]):
        if route["holder"] == SEAT:
            raise Unsupported("comms source outside the rehearsal")
        return "comms: Cursor holds inbox"
    source = bell[3]
    if route["holder"] != SEAT:
        reserved = (source == d.env.get("COMMS_RELAY_CANARY_THREAD") and bell[1] == "codex"
                    and "[comms relay test]" in ev["text"][bell.end():])
        if not reserved:
            return "comms: Cursor holds inbox"
    answer = exchange(d.env[ENDPOINT], {"channel": ev["channel"], "inbox": ev["thread_ts"], "ts": ev["ts"], "text": ev["text"]})
    status = answer.get("status")
    if status == "pending":
        raise Pending()
    if status == "silent":
        return "comms: no reply requested"
    if status == "check_receipt":
        messages = d.slack.paged("conversations.replies", "messages", channel=ev["channel"], ts=source,
                                 include_all_metadata="true", oldest=ev["ts"])
        for m in messages:
            meta = m.get("metadata") or {}
            if (d.rs.ours(m, d.registry) and m.get("username") == "comms" and meta.get("event_type") == RECEIPT
                    and (meta.get("event_payload") or {}).get("doorbell") == ev["ts"]):
                return "comms: reply already delivered"
        raise RuntimeError("comms reply claim has no Slack receipt")
    if status != "answer" or not isinstance(answer.get("reply"), str) or not answer["reply"].strip():
        raise RuntimeError("comms reply needs review")
    current = d.rs.Routes().get_route("comms", fresh=True)
    if not current or current.get("closed"):
        raise RuntimeError("cannot verify the comms holder before replying")
    if (current.get("holder"), current.get("inbox_ts"), current.get("channel")) != (route["holder"], ev["thread_ts"], ev["channel"]):
        raise Unsupported("holder changed before the reply was posted")
    # Only the trusted doorbell determines the destination. Treat model output as plain text.
    text = answer["reply"][:2900]
    d.slack.call("chat.postMessage", channel=ev["channel"], thread_ts=source, username="comms",
                 text=text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                 mrkdwn="false", link_names="false", parse="none", unfurl_links="false", unfurl_media="false",
                 metadata={"event_type": RECEIPT, "event_payload": {"doorbell": ev["ts"]}})
    return "comms: answered"

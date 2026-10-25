"""What the service runs against: the routing code, the registry, the queue, Slack, a best-effort cache and a clock.
`production()` builds them from the environment; the tests build fakes.  Logs carry ids, decisions and errors, never Slack
text.

Slack is the record: what was routed is the routing lines and doorbells the consumer posted, which it and the crons read
back.  The cache holds only what health reports, and losing it loses nothing but those numbers."""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from types import ModuleType
from typing import Any, Callable, Mapping, Protocol

from . import vendor

TOPIC = "slack-events"
DEAD_TOPIC = "slack-events-dead"
CACHE_NS = "slack-events"
LAST_SUCCESS = "last-success-at"                                 # when a delivery or a cron last finished
LAST_DONE = "last-done-at"                                       # when the consumer last finished a message (no cron writes it)
LAST_ENQUEUED = "last-enqueued-at"                               # when the app last published an event
STALL_RUNG = "stall-rung-at"                                     # when the watchdog last rang @comms for a stalled consumer
CANARY_SENT = "canary-sent-at"                                   # when the watchdog last published its synthetic event
CANARY_DONE = "canary-done-at"                                   # when the consumer last finished one
DEAD = "dead-letters"                                            # {"count", "since"}: dead letters since the record began
OLDEST = "oldest-unrouted"                                       # {"ts", "at"}: reconcile's oldest unrouted post, or ts None


class Cache(Protocol):
    """Vercel's Runtime Cache in production: regional and ephemeral, so a value may be missing at any time."""

    def get(self, key: str) -> Any: ...
    def set(self, key: str, value: Any, options: dict | None = None) -> None: ...


class Queue(Protocol):
    def send(self, topic: str, payload: dict, *, idempotency_key: str) -> None: ...


@dataclass
class Deps:
    rs: ModuleType                                                # research_slack: `tools/research/src/research/slack.py`
    registry: dict
    cache: Cache
    queue: Queue
    slack: Any                                                    # an `rs.Slack`
    clock: Callable[[], float]
    env: Mapping[str, str] = field(default_factory=dict)
    out: Any = None                                               # where log lines go (stdout)

    def log(self, event: str, **fields: Any) -> None:
        """One JSON line: ids, decisions and errors only.  Never pass Slack text here."""
        print(json.dumps({"event": event, **fields}, sort_keys=True, default=str), file=self.out or sys.stdout, flush=True)

    def read(self, key: str) -> Any:
        """The cached value, or None when it is missing or the cache fails: unknown, never an error."""
        try:
            return self.cache.get(key)
        except Exception as e:  # noqa: BLE001 - best effort by design
            self.log("cache-read-failed", key=key, error=error(e))
            return None

    def write(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        try:
            self.cache.set(key, value, {"ttl": ttl_s} if ttl_s else None)
        except Exception as e:  # noqa: BLE001
            self.log("cache-write-failed", key=key, error=error(e))


def error(e: BaseException) -> str:
    """An exception for the log: its type, and the message only for `research msg`'s own errors, which carry method names
    and Slack's error codes, never a post's text."""
    rs = vendor.slack()
    return f"{type(e).__name__}: {e}" if isinstance(e, rs.SlackError) else type(e).__name__


class VercelQueue:
    """Vercel Queues' synchronous client.  A duplicate idempotency key is the message already queued: Slack's retry of an
    event, or reconciliation finding a post it enqueued before."""

    def send(self, topic, payload, *, idempotency_key):
        from vercel.queue import DuplicateIdempotencyKeyError  # noqa: PLC0415
        from vercel.queue.sync import send  # noqa: PLC0415

        try:
            send(topic, payload, idempotency_key=idempotency_key)
        except DuplicateIdempotencyKeyError:
            pass


def runtime_cache() -> Cache:
    from vercel.cache import RuntimeCache  # noqa: PLC0415

    return RuntimeCache(namespace=CACHE_NS)


@lru_cache(maxsize=1)
def production() -> Deps:
    """The deployment's: SLACK_BOT_TOKEN posts directly (never through the broker), the registry bundled at build, Vercel
    Queues and Runtime Cache; routes.d and people.d from the published notes (RESEARCH_ROUTES_URL, RESEARCH_NOTES_TOKEN)."""
    rs, env = vendor.slack(), os.environ
    if not env.get("SLACK_BOT_TOKEN"):
        raise RuntimeError("SLACK_BOT_TOKEN is not set: the service posts and reads with Verity's bot, never the broker")
    return Deps(rs=rs, registry=vendor.registry(), cache=runtime_cache(), queue=VercelQueue(),
                slack=rs.Slack(env["SLACK_BOT_TOKEN"], direct=True), clock=time.time, env=env)

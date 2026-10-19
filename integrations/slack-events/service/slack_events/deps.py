"""What the service runs against: the routing code, the registry, a small Redis, the queue, Slack and a clock.  `production()`
builds them from the environment; the tests build fakes.  Logs carry ids, decisions and errors, never Slack text."""
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
NS = "slack-events:"                                              # every Redis key's prefix
PENDING = NS + "pending"                                          # event id -> when it was enqueued, until processed
DEAD_COUNT = NS + "dead-letter-count"
LAST_SUCCESS = NS + "last-success-at"


class Store(Protocol):
    """The Redis operations the service uses (Upstash's REST client in production)."""

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_s: int | None = None) -> None: ...
    def claim(self, key: str, value: str, ttl_s: int) -> bool: ...          # SET NX EX
    def release(self, key: str, value: str) -> None: ...                   # DEL only while it still holds `value`
    def incr(self, key: str) -> int: ...
    def zadd(self, key: str, member: str, score: float) -> None: ...
    def zrem(self, key: str, member: str) -> None: ...
    def zmin(self, key: str) -> float | None: ...                           # the lowest score, or None when empty


class Queue(Protocol):
    def send(self, topic: str, payload: dict, *, idempotency_key: str) -> None: ...


@dataclass
class Deps:
    rs: ModuleType                                                # research_slack: `tools/research/src/research/slack.py`
    registry: dict
    store: Store
    queue: Queue
    slack: Any                                                    # an `rs.Slack`
    clock: Callable[[], float]
    env: Mapping[str, str] = field(default_factory=dict)
    out: Any = None                                               # where log lines go (stdout)

    def log(self, event: str, **fields: Any) -> None:
        """One JSON line: ids, decisions and errors only.  Never pass Slack text here."""
        print(json.dumps({"event": event, **fields}, sort_keys=True, default=str), file=self.out or sys.stdout, flush=True)


def error(e: BaseException) -> str:
    """An exception for the log: its type, and the message only for `research msg`'s own errors, which carry method names
    and Slack's error codes, never a post's text."""
    rs = vendor.slack()
    return f"{type(e).__name__}: {e}" if isinstance(e, rs.SlackError) else type(e).__name__


class UpstashStore:
    """Upstash Redis over REST: UPSTASH_REDIS_REST_URL and _TOKEN, or the KV_REST_API_URL and _TOKEN that Vercel's
    Upstash integration sets."""

    RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"

    def __init__(self, env: Mapping[str, str]):
        from upstash_redis import Redis  # noqa: PLC0415

        url = env.get("UPSTASH_REDIS_REST_URL") or env.get("KV_REST_API_URL")
        token = env.get("UPSTASH_REDIS_REST_TOKEN") or env.get("KV_REST_API_TOKEN")
        if not url or not token:
            raise RuntimeError("no Redis: set UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN (or add the Upstash integration)")
        self.r = Redis(url=url, token=token, allow_telemetry=False)

    def get(self, key):
        v = self.r.get(key)
        return None if v is None else str(v)

    def set(self, key, value, ttl_s=None):
        self.r.set(key, value, ex=ttl_s)

    def claim(self, key, value, ttl_s):
        return bool(self.r.set(key, value, nx=True, ex=ttl_s))

    def release(self, key, value):
        self.r.eval(self.RELEASE, keys=[key], args=[value])

    def incr(self, key):
        return int(self.r.incr(key))

    def zadd(self, key, member, score):
        self.r.zadd(key, {member: score})

    def zrem(self, key, member):
        self.r.zrem(key, member)

    def zmin(self, key):
        got = self.r.zrange(key, 0, 0, withscores=True)
        if not got:
            return None
        first = got[0]
        return float(first[1] if isinstance(first, (list, tuple)) else got[1])


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


@lru_cache(maxsize=1)
def production() -> Deps:
    """The deployment's: SLACK_BOT_TOKEN posts directly (never through the broker), the registry bundled at build, Upstash
    and Vercel Queues; routes.d and people.d from the published notes (RESEARCH_ROUTES_URL, RESEARCH_NOTES_TOKEN)."""
    rs, env = vendor.slack(), os.environ
    if not env.get("SLACK_BOT_TOKEN"):
        raise RuntimeError("SLACK_BOT_TOKEN is not set: the service posts and reads with Verity's bot, never the broker")
    return Deps(rs=rs, registry=vendor.registry(), store=UpstashStore(env), queue=VercelQueue(),
                slack=rs.Slack(env["SLACK_BOT_TOKEN"], direct=True), clock=time.time, env=env)

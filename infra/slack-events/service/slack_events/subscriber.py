"""The queue's consumer, in poll mode: the `drain` cron (`/api/cron/drain`, every minute) claims the topic's messages as
CONSUMER_GROUP and hands each to `on_message`, until the topic is empty or the run's time is spent.  Entering a delivery
renews its lease while the handler runs and acknowledges it on a clean exit; a `Retry` from `consumer.consume` becomes
RetryAfter, which makes the message visible again after its delay.

Push mode needs Vercel's Python framework build, which compiles a `[[tool.vercel.subscribers]]` entry into a
queue-triggered function; this project's build (`framework: null`, `api/index.py`) never does, so a push subscriber
would consume nothing.  Polling needs no build support: it authenticates as `send` does."""
from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import anyio
from vercel.queue import Message, RetryAfter, poll

from . import consumer, deps

CONSUMER_GROUP = "slack-events-drain"
BATCH = 10                                                       # the most one poll may claim
LEASE_S = 120
BUDGET_S = 45.0                                                  # of the function's 60 s: stop claiming after this


async def on_message(message: Message[dict[str, Any]]) -> None:
    try:
        await anyio.to_thread.run_sync(consumer.consume, deps.production(), message.payload, message.metadata.delivery_count)
    except consumer.Retry as r:
        raise RetryAfter(r.delay, reason=str(r)) from None


async def drain(d: deps.Deps, receive: Callable[..., AsyncIterator[Any]] = poll,
                clock: Callable[[], float] = time.monotonic) -> dict:
    """Claim and handle messages until a poll comes back empty or BUDGET_S has passed.  A handler that fails past its
    Retry (it shouldn't: `consume` dead-letters instead) leaves its message for redelivery and the drain goes on."""
    start, handled, failed = clock(), 0, 0
    while clock() - start < BUDGET_S:
        got = 0
        async for delivery in receive(deps.TOPIC, CONSUMER_GROUP, limit=BATCH, lease_duration=LEASE_S):
            got += 1
            try:
                async with delivery as message:
                    await on_message(message)
                handled += 1
            except Exception as e:  # noqa: BLE001 - unacknowledged, it comes back
                failed += 1
                d.log("drain-handler-failed", error=deps.error(e))
        if not got:
            break
    if handled or failed:
        d.log("drained", handled=handled, failed=failed)
    return {"handled": handled, "failed": failed}

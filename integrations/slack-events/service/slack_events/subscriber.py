"""The queue's subscriber, as Vercel documents a Python one: `pyproject.toml`'s `[[tool.vercel.subscribers]]` names this
function, and Vercel's build compiles it into a queue-triggered function, with no `vercel.json` trigger.  The SDK accepts
each delivery, renews its lease while the handler runs and acknowledges it when the handler returns; a `Retry` from
`consumer.consume` becomes RetryAfter, which makes the message visible again after its delay."""
from __future__ import annotations

from typing import Any

import anyio
from vercel.queue import Message, RetryAfter, subscribe

from . import consumer, deps


@subscribe(topic=deps.TOPIC)
async def on_message(message: Message[dict[str, Any]]) -> None:
    try:
        await anyio.to_thread.run_sync(consumer.consume, deps.production(), message.payload, message.metadata.delivery_count)
    except consumer.Retry as r:
        raise RetryAfter(r.delay, reason=str(r)) from None

"""Vercel's queue consumer: the `slack-events` topic's push deliveries (vercel.json's `experimentalTriggers`) come here, and
`accept_and_handle` hands each to `on_message`, which runs `consumer.consume`.  A `Retry` becomes Vercel's RetryAfter; a
message `consume` returns from is acknowledged."""
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from vercel.queue import Message, RetryAfter, subscribe  # noqa: E402
from vercel.queue.sync import accept_and_handle  # noqa: E402

from slack_events import consumer, deps  # noqa: E402


@subscribe(topic=deps.TOPIC)
def on_message(message: Message[dict[str, Any]]) -> None:
    try:
        consumer.consume(deps.production(), message.payload, message.metadata.delivery_count)
    except consumer.Retry as r:
        raise RetryAfter(r.delay, reason=str(r)) from None


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.post("/{path:path}")
async def push(request: Request, path: str = "") -> Response:
    await run_in_threadpool(accept_and_handle, await request.body(), dict(request.headers))
    return Response(status_code=200)

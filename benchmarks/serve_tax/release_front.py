"""The release front of an arm on the certified tick that vLLM's client sends (`serve_tax`'s ``prod-all-on``): a stand-in for
the network attester's release side on the node, between the client and the server's `tick_plugin.TickFeed`.

It numbers each completion or chat request in the order it comes and puts it in the bucket of its arrival
(`corpus.StandIn`), in the release headers, and streams the watermark of those buckets to the server's
``POST /verity/watermark`` from its start to its stop, one line each bucket. So admission is the certified tick's, while
no answer is paced: the network attester's egress pacing is off. Every other request passes unnumbered. Each request
goes to the server on a connection of its own (``Connection: close`` both ways), so a client that keeps its
connections open is still numbered request by request."""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "benchmarks/network_e2e/traces", ROOT / "adapters/vllm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from corpus import StandIn  # noqa: E402 (the tree's, put on sys.path above)
from verity_vllm.pipeline.serve_admit import BUCKET_HEADER, RELEASE_HEADER, RUN_HEADER, WATERMARK_PATH  # noqa: E402
from verity_vllm.pipeline.tick_plugin import SAMPLED_PATHS  # noqa: E402

#: the headers the front writes itself, dropped from what it relays
HOP = frozenset({"connection", "keep-alive", RELEASE_HEADER, RUN_HEADER, BUCKET_HEADER})


def _rewritten(head: bytes, added: list[tuple[str, str]]) -> bytes:
    first, *fields = head.rstrip(b"\r\n").split(b"\r\n")
    kept = [f for f in fields if f.split(b":", 1)[0].strip().lower().decode("latin-1") not in HOP]
    more = [f"{k}: {v}".encode("latin-1") for k, v in [*added, ("Connection", "close")]]
    return b"\r\n".join([first, *kept, *more]) + b"\r\n\r\n"


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(1 << 16):
        writer.write(data)
        await writer.drain()


class ReleaseFront:
    """The front before the server on loopback port ``upstream``, in buckets of ``bucket_s`` from its start. ``clock`` and
    ``sleep`` are the event loop's (seconds); `summary` is what it released, whether the watermark held, and how many
    answers a side's hang-up cut short (``cut``, where any was)."""

    def __init__(self, upstream: int, bucket_s: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.upstream, self.clock, self.sleep = upstream, clock, sleep
        self.stand = StandIn(bucket_s, clock())
        self.stopping: asyncio.Event | None = None
        self.summary: dict = {"run": self.stand.run, "released": 0, "lines": 0, "watermark": None}

    def released(self, head: bytes) -> list[tuple[str, str]]:
        """The release headers of a request ``head``: a number and a bucket for a completion or chat request, none else."""
        method, target = head.split(b" ", 2)[:2]
        if method != b"POST" or target.split(b"?", 1)[0].decode("latin-1") not in SAMPLED_PATHS:
            return []
        n, bucket = self.stand.release(self.clock())
        self.summary["released"] = n + 1
        return [(RELEASE_HEADER, str(n)), (RUN_HEADER, self.stand.run), (BUCKET_HEADER, str(bucket))]

    async def connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            writer.close()
            return
        try:
            up_r, up_w = await asyncio.open_connection("127.0.0.1", self.upstream)
        except OSError:
            writer.write(b"HTTP/1.1 502 Upstream Unreachable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            writer.close()
            return
        up_w.write(_rewritten(head, self.released(head)))
        body = asyncio.create_task(_pipe(reader, up_w))
        try:
            writer.write(_rewritten(await up_r.readuntil(b"\r\n\r\n"), []))
            await _pipe(up_r, writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            self.summary["cut"] = self.summary.get("cut", 0) + 1
        finally:
            body.cancel()
            for w in (up_w, writer):
                w.close()

    async def watermark(self) -> None:
        """The watermark, one line as each bucket closes, until the stop and the line of the last release's bucket."""
        assert self.stopping is not None
        r, w = await asyncio.open_connection("127.0.0.1", self.upstream)
        w.write(f"POST {WATERMARK_PATH} HTTP/1.1\r\nHost: 127.0.0.1\r\n{RUN_HEADER}: {self.stand.run}\r\n"
                "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n".encode())
        while not (self.stopping.is_set() and self.stand.done(len(self.stand.buckets))):
            await self.sleep(max(0.0, self.stand.due() - self.clock()))
            while self.stand.due() <= self.clock():
                line = self.stand.close()
                w.write(b"%x\r\n%s\r\n" % (len(line), line))
                self.summary["lines"] += 1
            await w.drain()
        w.write(b"0\r\n\r\n")
        await w.drain()
        status = (await r.readline()).split(b" ", 2)
        self.summary["watermark"] = int(status[1]) if len(status) > 1 and status[1].isdigit() else None
        w.close()

    async def serve(self, listening: Callable[[int], None], stop: asyncio.Event) -> dict:
        """Listen on a free loopback port (``listening`` gets it) and stream the watermark until ``stop``; the summary."""
        self.stopping = stop
        server = await asyncio.start_server(self.connection, "127.0.0.1", 0)
        mark = asyncio.create_task(self.watermark())
        listening(server.sockets[0].getsockname()[1])
        try:
            await mark
        except Exception as e:  # noqa: BLE001 -- the gate opens on a broken watermark; the summary says why
            self.summary["error"] = f"{type(e).__name__}: {e}"
        finally:
            server.close()
        return self.summary


class Running:
    """A `ReleaseFront` on a thread of its own: `port` is where it listens, `stop` ends its watermark and gives its summary."""

    def __init__(self, upstream: int, bucket_s: float) -> None:
        port: Future[int] = Future()
        self.loop = asyncio.new_event_loop()
        self.front = ReleaseFront(upstream, bucket_s)
        self.stop_event = asyncio.Event()
        self.done: Future[dict] = Future()

        def run() -> None:
            try:
                self.done.set_result(self.loop.run_until_complete(self.front.serve(port.set_result, self.stop_event)))
            except BaseException as e:  # noqa: BLE001 -- handed to the caller's port or stop
                (port if not port.done() else self.done).set_exception(e)

        self.thread = threading.Thread(target=run, name="release-front", daemon=True)
        self.thread.start()
        self.port = port.result()

    def stop(self) -> dict:
        self.loop.call_soon_threadsafe(self.stop_event.set)
        try:
            return self.done.result()
        finally:
            self.thread.join()
            self.loop.close()

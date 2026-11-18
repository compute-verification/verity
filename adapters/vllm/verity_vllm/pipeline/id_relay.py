"""The network attester's id egress relay: a process of the server's side, on its host, between the network attester and
a server that answers in HTTP, which takes the HTTP off the server's answers to the network attester's id requests.

* **An id request** (a connection's first request, exactly `workload.id_request` in the network attester's P + 9-byte
  cells, its release headers aside: `ids_protocol.id_request_fields`) goes to the server as the request ``request_of``
  builds of its fields, on a connection of its own, under the release headers it came with. Of the server's answer only
  the token ids come back, as ``stream_ids`` reads them, each one id event written as its chunk arrives, then the trailer
  once ``stream_ids`` ends. A stream ``stream_ids`` breaks off (`Broken`: the server refused the request, sent an error or
  text without its ids, or broke off) ends as `ids_protocol.IdsProtocol`'s failed stream does: its ids so far, no trailer
  and none of the server's text, so the network attester fails the connection. Once the network attester's side of the
  connection has ended (closed, or broken), the relay closes its connection to the server, whatever the server has sent,
  so the server sees the close as it would on a connection of its own. What the network attester sends after its
  request is read and dropped.
* **Every other connection** (a model listing, the watermark's ``POST /verity/watermark``, a probe's HTTP) is relayed
  byte for byte, both ways, until both sides have closed.

``log`` takes one record per id request when its connection ends: the release's number, run and bucket (``n``,
``run``, ``bucket``), the prompt's and the output's id counts (``prompt``, ``ids``), ``ended`` (the trailer was sent;
else ``why``, which is `GONE` where the network attester's side ended first), ``at`` (the seconds from the relay's start
to the request's arrival, whole) and ``ms`` (each id event's write after the arrival, in milliseconds, then the
trailer's). It holds no id and no text. A connection's egress cell i is due i // r buckets after its first, so the delay
a connection needed after its release is the most of ``ms[i]`` less (i // r) buckets, over its events and its trailer.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from verity.protocols.resource_accounting.network.prover.ids_protocol import ID_PAYLOAD, id_request_fields
from verity.protocols.resource_accounting.network.prover.workload import TRAILER, id_event
from verity.protocols.resource_accounting.network.verifier import release as REL

__all__ = ["BLANK", "BODY_BYTES", "COMPLETIONS", "GONE", "READ_BYTES", "Broken", "Front", "RequestOf", "StreamIds", "serve"]

READ_BYTES = 1 << 16
#: the longest body the relay reads whole to tell an id request: a connection whose request says more is relayed
BODY_BYTES = 1 << 26
COMPLETIONS = b"POST /v1/completions HTTP/1.1"
BLANK = b"\r\n\r\n"
#: an id request's ``why`` where its connection ended before the server's stream of it did
GONE = "the network attester's side ended before the server's stream"

#: the server's copy of an id request: its fields, the server's ``host`` and the release headers it came with
RequestOf = Callable[[dict[str, Any], bytes, bytes], bytes]
#: each chunk's token ids, in order, of the server's answer read from its connection; `Broken` where it doesn't end whole
StreamIds = Callable[[asyncio.StreamReader], AsyncIterator[list[int]]]


class Broken(Exception):
    """Why the server's stream of an id request ends without its trailer."""


async def _first(reader: asyncio.StreamReader) -> bytes:
    """A connection's first bytes, as far as telling an id request takes: a ``POST /v1/completions`` that states its
    length in at most `BODY_BYTES` whole, head and body; of anything else its head, or what came of it."""
    try:
        head = await reader.readuntil(BLANK)
    except asyncio.IncompleteReadError as e:
        return e.partial
    except asyncio.LimitOverrunError:
        return await reader.read(READ_BYTES)
    line, *fields = head[:-len(BLANK)].split(b"\r\n")
    lengths = [f.partition(b":")[2].strip() for f in fields if f.partition(b":")[0].strip().lower() == b"content-length"]
    if line != COMPLETIONS or len(lengths) != 1 or not lengths[0].decode("latin-1").isdecimal() or int(lengths[0]) > BODY_BYTES:
        return head
    try:
        return head + await reader.readexactly(int(lengths[0]))
    except asyncio.IncompleteReadError as e:
        return head + e.partial


async def _pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    """`src`'s bytes to `dst` until `src` ends or either breaks, then `dst`'s write side closed."""
    with contextlib.suppress(ConnectionError):
        while data := await src.read(READ_BYTES):
            dst.write(data)
            await dst.drain()
    with contextlib.suppress(OSError):
        if dst.can_write_eof():
            dst.write_eof()


async def _ended(reader: asyncio.StreamReader) -> None:
    """Returns once `reader`'s side has ended or broken; what it sends before that is read and dropped."""
    with contextlib.suppress(OSError):
        while await reader.read(READ_BYTES):
            pass


async def _close(writer: asyncio.StreamWriter | None) -> None:
    if writer is not None:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()


class Front:
    """The relay (the module docstring's) before the server at ``upstream``, for id events of ``payload`` bytes: `handle`
    serves one connection, its task kept in ``handlers`` until it ends. ``log`` takes each id request's record, and
    ``clock`` reads the seconds its times are in."""

    def __init__(self, upstream: tuple[str, int], payload: int = ID_PAYLOAD, *, request_of: RequestOf,
                 stream_ids: StreamIds, log: Callable[[dict[str, Any]], None] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        id_event(0, payload)
        self.upstream, self.payload, self.log, self.clock = upstream, payload, log, clock
        self.request_of, self.stream_ids = request_of, stream_ids
        self.origin = clock()
        self.host = f"{upstream[0]}:{upstream[1]}".encode()
        self.handlers: set[asyncio.Task[Any]] = set()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # asyncio holds a connection's handler only while it watches the client's side. Once that side has ended, a
        # handler that still waits on the server (a relayed connection's, for the rest of the answer) is held by nothing
        # but itself, and a collection would destroy it: the answer cut.
        task = asyncio.current_task()
        assert task is not None                          # asyncio starts each connection's handler as a task
        self.handlers.add(task)
        task.add_done_callback(self.handlers.discard)
        try:
            first = await _first(reader)
            arrived = self.clock()
            fields = id_request_fields(first, self.payload)
            if fields is not None:
                await self.ids(first.partition(BLANK)[0] + BLANK, fields, reader, writer, arrived)
            elif first:
                await self.relay(first, reader, writer)
        except OSError:
            pass                                         # the client's connection broke, or the server's refused: closed below
        finally:
            await _close(writer)

    async def relay(self, first: bytes, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """A connection that isn't an id request's: `first` and everything after it to the server, and the server's bytes
        back, until both directions have ended."""
        back, server = await asyncio.open_connection(*self.upstream)
        try:
            server.write(first)
            await asyncio.gather(_pump(reader, server), _pump(back, writer))
        finally:
            await _close(server)

    async def ids(self, head: bytes, fields: dict[str, Any], reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                  arrived: float) -> None:
        """An id request (its head and fields, whole at `arrived`) answered in the id form from the server's stream of
        it, until that stream ends or `reader`'s side, the network attester's, does: either way the server's connection
        is closed."""
        rel = REL.release_of(head)
        n, run, bucket = rel if rel is not None else (None, None, None)
        rec: dict[str, Any] = {"n": n, "run": run.hex() if run else None, "bucket": bucket, "prompt": len(fields["prompt"]),
                               "ids": 0, "ended": False, "why": None, "at": round(arrived - self.origin, 3), "ms": []}

        def wrote() -> None:
            rec["ms"].append(round((self.clock() - arrived) * 1000, 1))

        server = None

        async def answer() -> None:
            nonlocal server
            back, server = await asyncio.open_connection(*self.upstream)
            server.write(self.request_of(fields, self.host, b"" if rel is None else REL.release_head(*rel)))
            async with contextlib.aclosing(self.stream_ids(back)) as chunks:
                async for ids in chunks:
                    for t in ids:
                        writer.write(id_event(t, self.payload))
                        wrote()
                    rec["ids"] += len(ids)
                    await writer.drain()
            writer.write(TRAILER)
            wrote()
            await writer.drain()
            rec["ended"] = True

        stream, gone = asyncio.ensure_future(answer()), asyncio.ensure_future(_ended(reader))
        try:
            await asyncio.wait({stream, gone}, return_when=asyncio.FIRST_COMPLETED)
            if stream.done():
                stream.result()
            else:
                rec["why"] = GONE
        except Broken as e:
            rec["why"] = str(e)
        except OSError as e:
            rec["why"] = f"a connection broke ({type(e).__name__})"
            raise
        finally:
            for task in (stream, gone):
                task.cancel()
            await asyncio.gather(stream, gone, return_exceptions=True)
            await _close(server)
            if self.log is not None:
                self.log(rec)


async def serve(front: Front, listen: tuple[str, int], *, announce: Callable[[tuple[str, int]], None],
                stop: asyncio.Event) -> None:
    """`front` on `listen` until `stop` is set; `announce` takes the address it listens on."""
    server = await asyncio.start_server(front.handle, *listen, limit=1 << 22, backlog=256)
    announce(tuple(server.sockets[0].getsockname()[:2]))
    async with server:
        await stop.wait()

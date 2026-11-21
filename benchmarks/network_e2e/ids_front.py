"""The network attester's id egress relay, started by path: a process of the server's side, on its host, between the
network attester and vLLM's own OpenAI server, which takes the HTTP off the server's answers to the network attester's id
requests.

    python benchmarks/network_e2e/ids_front.py --upstream HOST:PORT [--listen 127.0.0.1:18746] [--id-payload 503] [--log FILE]

* **An id request** (a connection's first request, exactly `release.id_request` in the network attester's P + 9-byte
  cells, its release headers aside: `release.id_request_fields`) goes to the server as the request ``request_of``
  builds of its fields, on a connection of its own, under the release headers it came with. Of the server's answer only
  the token ids come back, as ``stream_ids`` reads them, each one id event written as its chunk arrives, then the trailer
  once ``stream_ids`` ends. A stream ``stream_ids`` breaks off (`Broken`: the server refused the request, sent an error or
  text without its ids, or broke off) ends as ``serve-openai --protocol ids``'s failed stream does: its ids so far, no trailer
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
Every other connection that breaks, before it is relayed or while it is, leaves one record: ``broken``, the side whose
connection broke (``client`` or ``server``), ``error``, the exception's class name, and ``at``, the seconds from the
relay's start to the break.

The relay reads the server through two readers: an id request's copy for the server (`server_request`, a
``POST /v1/completions`` of its fields with ``return_token_ids``, under the release headers it came with), and the
server's streamed chunks (`token_ids`, through ``data: [DONE]``, `Broken` where the stream breaks).

Once the relay listens it writes one line, ``{"listening": [HOST, PORT]}``, on stdout; it serves until SIGTERM or
SIGINT. ``--log`` appends its records, one JSON line each.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from pathlib import Path
from typing import Any

from verity import hooks as H
from verity import release as REL
from verity.release import ID_PAYLOAD, TRAILER, id_event, id_request_fields

__all__ = ["BLANK", "BODY_BYTES", "COMPLETIONS", "GONE", "READ_BYTES", "Broken", "Front", "RequestOf", "StreamIds", "address",
           "chunk_ids", "main", "run", "serve", "server_request", "token_ids"]

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


#: why the server's stream of an id request ends without its trailer: the workload's readers' error
Broken = H.Broken


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
    """`src`'s bytes to `dst` until `src` ends, then `dst`'s write side closed; a break of either raises."""
    while data := await src.read(READ_BYTES):
        dst.write(data)
        await dst.drain()
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
    serves one connection, its task kept in ``handlers`` until it ends. ``log`` takes the module docstring's records, and
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
            if isinstance(fields, dict):
                await self.ids(first.partition(BLANK)[0] + BLANK, fields, reader, writer, arrived)
            elif first:
                await self.relay(first, reader, writer)
        except OSError as e:
            # the client's transport is down once its connection broke; else the break was the server's
            if self.log is not None:
                self.log({"broken": "client" if writer.transport.is_closing() else "server", "error": type(e).__name__,
                          "at": round(self.clock() - self.origin, 3)})
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
        is closed. A malformed release header (`release.release_of`'s refusal) refuses the request: its record's ``why``
        is the refusal, and the connection closes with no answer and no server connection."""
        try:
            rel, refused = REL.release_of(head), None
        except ValueError as e:
            rel, refused = None, str(e)
        n, run, bucket = rel if rel is not None else (None, None, None)
        rec: dict[str, Any] = {"n": n, "run": run.hex() if run else None, "bucket": bucket, "prompt": len(fields["prompt"]),
                               "ids": 0, "ended": False, "why": refused, "at": round(arrived - self.origin, 3), "ms": []}
        if refused is not None:
            if self.log is not None:
                self.log(rec)
            return

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


async def run(upstream: tuple[str, int], listen: tuple[str, int], payload: int = ID_PAYLOAD, *, server_request: RequestOf,
              stream_token_ids: StreamIds, log: Callable[[dict[str, Any]], None] | None = None, stop: asyncio.Event,
              announce: Callable[[tuple[str, int]], None]) -> None:
    """The network attester's id relay as a process: a `Front` before the server at `upstream`, served on `listen` until
    `stop` is set; `ValueError` for a `payload` below an id event, before it listens."""
    front = Front(upstream, payload, request_of=server_request, stream_ids=stream_token_ids, log=log)
    await serve(front, listen, announce=announce, stop=stop)


def address(text: str) -> tuple[str, int]:
    """``HOST:PORT`` as (host, port)."""
    host, sep, port = text.rpartition(":")
    if not (sep and host and port.isdecimal()):
        raise SystemExit(f"an address is HOST:PORT, not {text!r}")
    return host, int(port)


def server_request(fields: dict[str, Any], host: bytes, release: bytes) -> bytes:
    """The server's copy of an id request of `fields`: a ``POST /v1/completions`` of the fields, streamed, with
    ``return_token_ids``, on a connection it closes, under `release`, the release headers it came with."""
    body = json.dumps(fields | {"stop": [], "stream": True, "return_token_ids": True}, separators=(",", ":")).encode()
    return (COMPLETIONS + b"\r\nhost: " + host + b"\r\ncontent-type: application/json\r\n" + release
            + b"content-length: %d\r\nconnection: close\r\n\r\n" % len(body) + body)


def chunk_ids(data: bytes) -> list[int]:
    """The token ids of one streamed completion chunk (its ``data:`` JSON), every choice's ``token_ids`` in order;
    `Broken` for an error event, and for a choice whose text comes without its ids or whose ids aren't u32s."""
    chunk = json.loads(data)
    if not isinstance(chunk, dict) or "error" in chunk:
        raise Broken("the server sent an error event")
    ids: list[int] = []
    for choice in chunk.get("choices") or ():
        got = choice.get("token_ids") if isinstance(choice, dict) else ...
        if got is None and not choice.get("text"):
            continue
        if not (isinstance(got, list) and all(type(t) is int and 0 <= t < 1 << 32 for t in got)):
            raise Broken("the server sent a chunk without its token ids")
        ids += got
    return ids


async def _body(reader: asyncio.StreamReader, chunked: bool) -> AsyncIterator[bytes]:
    while True:
        if chunked:
            size = int((await reader.readuntil(b"\r\n")).split(b";", 1)[0], 16)
            if size == 0:
                return
            yield (await reader.readexactly(size + 2))[:-2]
        else:
            data = await reader.read(READ_BYTES)
            if not data:
                return
            yield data


async def token_ids(reader: asyncio.StreamReader) -> AsyncGenerator[list[int], None]:
    """Each chunk's token ids (`chunk_ids`) of a streamed ``/v1/completions`` response read from `reader`, through
    ``data: [DONE]``; `Broken` where the response isn't a 200 whose events go on to ``[DONE]``."""
    try:
        status, *lines = (await reader.readuntil(BLANK))[:-len(BLANK)].decode("latin-1").split("\r\n")
        code = status.split(" ")[1:2]
        if code != ["200"]:
            raise Broken(f"the server answered {code[0] if code and code[0].isdecimal() else 'no status'}")
        headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines)}
        buf = b""
        async for data in _body(reader, "chunked" in headers.get("transfer-encoding", "").lower()):
            *events, buf = (buf + data).replace(b"\r\n", b"\n").split(b"\n\n")
            for event in events:
                for line in event.split(b"\n"):
                    if line.startswith(b"data:"):
                        if line[5:].strip() == b"[DONE]":
                            return
                        yield chunk_ids(line[5:])
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError, ValueError) as e:
        raise Broken(f"the server's stream broke off ({type(e).__name__})") from e
    raise Broken("the server's stream ended before [DONE]")



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="the network attester's id relay before vLLM's own OpenAI server")
    p.add_argument("--upstream", required=True, help="the server's HOST:PORT: vLLM's own OpenAI server")
    p.add_argument("--listen", default="127.0.0.1:18746", help="the HOST:PORT the relay listens on")
    p.add_argument("--id-payload", type=int, default=ID_PAYLOAD, dest="id_payload",
                   help="each id event's bytes, the network attester's cell payload B - 9 (default 503: the scan's B = 512)")
    p.add_argument("--log", default=None, help="append the relay's records to this file (JSON lines)")
    opts = p.parse_args(argv)
    upstream, listen = address(opts.upstream), address(opts.listen)
    with contextlib.ExitStack() as stack:
        out = stack.enter_context(Path(opts.log).open("a", buffering=1, encoding="utf-8")) if opts.log else None

        def log(rec: dict[str, Any]) -> None:
            assert out is not None
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")

        async def serving() -> None:
            stop = asyncio.Event()
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_running_loop().add_signal_handler(sig, stop.set)

            def announce(at: tuple[str, int]) -> None:
                sys.stdout.write(json.dumps({"listening": list(at)}) + "\n")
                sys.stdout.flush()
            try:
                await run(upstream, listen, opts.id_payload, server_request=server_request, stream_token_ids=token_ids,
                          log=log if out is not None else None, stop=stop, announce=announce)
            except ValueError as e:
                raise SystemExit(f"--id-payload: {e}") from e

        asyncio.run(serving())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""`verity-vllm ids-front`: the network certifier's id form in front of vLLM's own OpenAI server.

    verity-vllm ids-front --upstream HOST:PORT [--listen 127.0.0.1:18746] [--id-payload 503] [--log FILE]

The network certifier's id egress form takes from the server only the scan's id events: no HTTP head, one
`io_tie.id_event` of P bytes an output id, then `io_tie.TRAILER`. `serve-openai --protocol ids` writes them itself
(`serve_protocol.IdsProtocol`). vLLM's own server, which serves the tensor-parallel workloads under the certified tick
(`tick_plugin`), answers in HTTP, and a middleware can't take a response's head off. This relay does it for that server:
a process of the server's side, on its host, between the network certifier and the server.

* **An id request** (a connection's first request, exactly `active.id_request` in the certifier's P + 9-byte cells, its
  release headers aside: `serve_protocol.id_request_fields`) goes to the server as a ``POST /v1/completions`` of the same
  fields with ``return_token_ids``, on a connection of its own, under the release headers it came with, which the tick's
  feed reads. Of the server's streamed chunks only the token ids come back, each one id event written as its chunk
  arrives, then the trailer at ``data: [DONE]``. A stream that doesn't get there (the server refused the request, sent an
  error event or text without its ids, or broke off) ends as `IdsProtocol`'s failed stream does: its ids so far, no
  trailer and none of the server's text, so the network certifier fails the connection.
* **Every other connection** (``GET /v1/models``, the watermark's ``POST /verity/watermark``, a probe's HTTP) is relayed
  byte for byte, both ways, until both sides have closed.

Once it listens it writes one line, ``{"listening": [HOST, PORT]}``, on stdout, and serves until SIGTERM or SIGINT.
``--log`` appends one JSON line per id request when its connection ends: the release's number, run and bucket
(``n``, ``run``, ``bucket``), the prompt's and the output's id counts (``prompt``, ``ids``), ``ended`` (the trailer was
sent; else ``why``), ``at`` (the seconds from the front's start to the request's arrival, whole) and ``ms`` (each id
event's write after the arrival, in milliseconds, then the trailer's). It holds no id and no text. A connection's
egress cell i is due i // r buckets after its first, so the delay a connection needed after its release is the most of
``ms[i]`` less (i // r) buckets, over its events and its trailer.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import sys
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.protocols.certifier.verifier import io_tie as IT
from verity.protocols.certifier.verifier import release as REL

from verity_vllm.config import UsageError, option
from verity_vllm.pipeline.serve_protocol import ID_PAYLOAD, id_request_fields

__all__ = ["BODY_BYTES", "Broken", "Front", "Options", "address", "chunk_ids", "main", "serve", "server_request",
           "token_ids"]

READ_BYTES = 1 << 16
#: the longest body the front reads whole to tell an id request: a connection whose request says more is relayed
BODY_BYTES = 1 << 26
COMPLETIONS = b"POST /v1/completions HTTP/1.1"
BLANK = b"\r\n\r\n"


class Broken(Exception):
    """Why the server's stream of an id request ends without its trailer."""


def address(text: str) -> tuple[str, int]:
    """``HOST:PORT`` as (host, port)."""
    host, sep, port = text.rpartition(":")
    if not (sep and host and port.isdigit()):
        raise UsageError(f"an address is HOST:PORT, not {text!r}")
    return host, int(port)


def server_request(head: bytes, fields: dict[str, Any], host: bytes) -> bytes:
    """The server's copy of an id request of `fields` whose head is `head`: a ``POST /v1/completions`` of the fields,
    streamed, with ``return_token_ids``, on a connection it closes, under `head`'s release headers
    (`release.release_head` of `release.release_of`)."""
    body = json.dumps(fields | {"stop": [], "stream": True, "return_token_ids": True}, separators=(",", ":")).encode()
    rel = REL.release_of(head)
    return (COMPLETIONS + b"\r\nhost: " + host + b"\r\ncontent-type: application/json\r\n"
            + (b"" if rel is None else REL.release_head(*rel))
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


async def token_ids(reader: asyncio.StreamReader) -> AsyncIterator[list[int]]:
    """Each chunk's token ids (`chunk_ids`) of a streamed ``/v1/completions`` response read from `reader`, through
    ``data: [DONE]``; `Broken` where the response isn't a 200 whose events go on to ``[DONE]``."""
    try:
        status, *lines = (await reader.readuntil(BLANK))[:-len(BLANK)].decode("latin-1").split("\r\n")
        code = status.split(" ")[1:2]
        if code != ["200"]:
            raise Broken(f"the server answered {code[0] if code and code[0].isdigit() else 'no status'}")
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
    if line != COMPLETIONS or len(lengths) != 1 or not lengths[0].isdigit() or int(lengths[0]) > BODY_BYTES:
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


async def _close(writer: asyncio.StreamWriter | None) -> None:
    if writer is not None:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()


class Front:
    """The relay (the module docstring's) before the server at ``upstream``, for id events of ``payload`` bytes: `handle`
    serves one connection. ``log`` takes each id request's record, and ``clock`` reads the seconds its times are in."""

    def __init__(self, upstream: tuple[str, int], payload: int = ID_PAYLOAD, *,
                 log: Callable[[dict[str, Any]], None] | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        IT.id_event(0, payload)
        self.upstream, self.payload, self.log, self.clock = upstream, payload, log, clock
        self.origin = clock()
        self.host = f"{upstream[0]}:{upstream[1]}".encode()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            first = await _first(reader)
            arrived = self.clock()
            fields = id_request_fields(first, self.payload)
            if fields is not None:
                await self.ids(first.partition(BLANK)[0] + BLANK, fields, writer, arrived)
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

    async def ids(self, head: bytes, fields: dict[str, Any], writer: asyncio.StreamWriter, arrived: float) -> None:
        """An id request (its head and fields, whole at `arrived`) answered in the id form from the server's stream of
        it."""
        rel = REL.release_of(head)
        n, run, bucket = rel if rel is not None else (None, None, None)
        rec: dict[str, Any] = {"n": n, "run": run.hex() if run else None, "bucket": bucket, "prompt": len(fields["prompt"]),
                               "ids": 0, "ended": False, "why": None, "at": round(arrived - self.origin, 3), "ms": []}

        def wrote() -> None:
            rec["ms"].append(round((self.clock() - arrived) * 1000, 1))

        server = None
        try:
            back, server = await asyncio.open_connection(*self.upstream)
            server.write(server_request(head, fields, self.host))
            async with contextlib.aclosing(token_ids(back)) as chunks:
                async for ids in chunks:
                    for t in ids:
                        writer.write(IT.id_event(t, self.payload))
                        wrote()
                    rec["ids"] += len(ids)
                    await writer.drain()
            writer.write(IT.TRAILER)
            wrote()
            await writer.drain()
            rec["ended"] = True
        except Broken as e:
            rec["why"] = str(e)
        except OSError as e:
            rec["why"] = f"a connection broke ({type(e).__name__})"
            raise
        finally:
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


@dataclass(frozen=True, kw_only=True)
class Options:
    upstream: str = option("--upstream", help="the server's HOST:PORT: vLLM's own OpenAI server")
    listen: str = option("--listen", default="127.0.0.1:18746", help="the HOST:PORT the network certifier connects to")
    id_payload: int = option("--id-payload", type=int, default=ID_PAYLOAD,
                             help="each id event's bytes, the network certifier's cell payload B − 9 (default 503: the scan's "
                                  "B = 512)")
    log: str | None = option("--log", default=None, help="append each id request's record to this file (JSON lines)")


def main(opts: Options) -> int:
    upstream, listen = address(opts.upstream), address(opts.listen)
    with contextlib.ExitStack() as stack:
        out = stack.enter_context(Path(opts.log).open("a", buffering=1, encoding="utf-8")) if opts.log else None

        def log(rec: dict[str, Any]) -> None:
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
        try:
            front = Front(upstream, opts.id_payload, log=log if out is not None else None)
        except ValueError as e:
            raise UsageError(f"--id-payload: {e}") from e

        async def run() -> None:
            stop = asyncio.Event()
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_running_loop().add_signal_handler(sig, stop.set)

            def announce(at: tuple[str, int]) -> None:
                sys.stdout.write(json.dumps({"listening": list(at)}) + "\n")
                sys.stdout.flush()
            await serve(front, listen, announce=announce, stop=stop)

        asyncio.run(run())
    return 0

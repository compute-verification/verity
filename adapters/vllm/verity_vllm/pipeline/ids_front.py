"""`verity-vllm ids-front`: the launcher of the processes schemes start beside vLLM's own OpenAI server.

    verity-vllm ids-front --upstream HOST:PORT --protocols NAME[,NAME] [--listen 127.0.0.1:18746] [--id-payload 503]
        [--log FILE]

Each scheme ``--protocols`` names (`verity.protocols.run.schemes.get`) is attached to this process's
exchange, where it files the processes it starts beside the workload (`verity.hooks.StartBeside`), each bound to the
readers it names by what they do (`READERS`): ``server-request``, an input's copy for the server (`server_request`, a
``POST /v1/completions`` of its fields with ``return_token_ids``, under the release headers it came with), and
``stream-token-ids``, the reader of the server's streamed chunks (`token_ids`, through ``data: [DONE]``, `Broken` where
the stream breaks). Each process is called with ``--upstream``, ``--listen``, ``--id-payload`` and ``--log``.

Once a process listens it writes one line, ``{"listening": [HOST, PORT]}``, on stdout; all serve until SIGTERM or
SIGINT. ``--log`` appends each process's records, one JSON line each.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity import hooks as H
from verity.hooks import Broken
from verity.protocols.run import schemes

from verity_vllm import messages
from verity_vllm.config import UsageError, option

__all__ = ["READERS", "Options", "address", "chunk_ids", "launch", "main", "server_request", "token_ids"]

COMPLETIONS = b"POST /v1/completions HTTP/1.1"
BLANK = b"\r\n\r\n"
READ_BYTES = 1 << 16


def address(text: str) -> tuple[str, int]:
    """``HOST:PORT`` as (host, port)."""
    host, sep, port = text.rpartition(":")
    if not (sep and host and port.isdecimal()):
        raise UsageError(f"an address is HOST:PORT, not {text!r}")
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


#: the workload's readers a scheme's process names by what they do (`verity.hooks.StartBeside`)
READERS = {"server-request": server_request, "stream-token-ids": token_ids}


@dataclass(frozen=True, kw_only=True)
class Options:
    upstream: str = option("--upstream", help="the server's HOST:PORT: vLLM's own OpenAI server")
    protocols: str = option("--protocols", default="",
                            help="the application's protocols line, comma-separated: the schemes whose processes start")
    listen: str = option("--listen", default="127.0.0.1:18746", help="the HOST:PORT the processes listen on")
    id_payload: int = option("--id-payload", type=int, default=503,
                             help="each id event's bytes, the network attester's cell payload B − 9 (default 503: the scan's "
                                  "B = 512)")
    log: str | None = option("--log", default=None, help="append each process's records to this file (JSON lines)")


def launch(exchange: messages.Exchange, names: list[str]) -> list[tuple[str, Any]]:
    """Attach the schemes ``names`` to ``exchange`` with this server's `READERS`, and return the processes they start
    beside it, each by its name, bound to its readers; `UsageError` where none is registered or none starts one."""
    exchange.readers = READERS
    for name in names:
        try:
            scheme = schemes.get(name)
        except KeyError as e:
            raise UsageError(f"--protocols: {e.args[0]}") from None
        exchange.attach(scheme)
    started = exchange.beside()
    if not started:
        raise UsageError(f"--protocols {','.join(names) or '(none)'}: no scheme starts a process beside the server "
                         f"({H.StartBeside.__name__})")
    return started


def main(opts: Options) -> int:
    upstream, listen = address(opts.upstream), address(opts.listen)
    started = launch(messages.current(), [n for n in opts.protocols.split(",") if n])
    with contextlib.ExitStack() as stack:
        out = stack.enter_context(Path(opts.log).open("a", buffering=1, encoding="utf-8")) if opts.log else None

        def log(rec: dict[str, Any]) -> None:
            assert out is not None
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")

        async def run() -> None:
            stop = asyncio.Event()
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_running_loop().add_signal_handler(sig, stop.set)

            def announce(at: tuple[str, int]) -> None:
                sys.stdout.write(json.dumps({"listening": list(at)}) + "\n")
                sys.stdout.flush()
            try:
                await asyncio.gather(*(start()(upstream, listen, opts.id_payload, log=log if out is not None else None,
                                               stop=stop, announce=announce) for _, start in started))
            except ValueError as e:
                raise UsageError(f"--id-payload: {e}") from e

        asyncio.run(run())
    return 0

"""`verity-vllm ids-front`: the network attester's id relay (`id_relay`) in front of vLLM's own OpenAI server.

    verity-vllm ids-front --upstream HOST:PORT [--listen 127.0.0.1:18746] [--id-payload 503] [--log FILE]

The network attester's id egress form takes from the server only the scan's id events: no HTTP head, one
`workload.id_event` of P bytes an output id, then `workload.TRAILER`. `serve-openai --protocol ids` writes them itself
(`ids_protocol.IdsProtocol`). vLLM's own server, which serves the tensor-parallel workloads under the certified tick
(`tick_plugin`), answers in HTTP, and a middleware can't take a response's head off; `id_relay.Front` does it for that
server, and this module hands it what it needs of vLLM's: an id request's copy for the server (`server_request`, a
``POST /v1/completions`` of the same fields with ``return_token_ids``, which the tick's feed reads by its release
headers) and the reader of the server's streamed chunks (`token_ids`, through ``data: [DONE]``). vLLM's server aborts a
request on its connection's close, which the relay makes when the network attester's side ends; one the tick gate
dropped (`serve_admit`: a lost release's) gets no chunk: the tick's middleware ends it when its watermark line comes
(`tick_plugin.TickFeed`), and the relay's close ends it where that line never does.

Once it listens it writes one line, ``{"listening": [HOST, PORT]}``, on stdout, and serves until SIGTERM or SIGINT.
``--log`` appends `id_relay`'s record of each id request, one JSON line, when its connection ends.
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

from verity.protocols.resource_accounting.network.prover.ids_protocol import ID_PAYLOAD
from verity_vllm.config import UsageError, option
from verity_vllm.pipeline.id_relay import BLANK, COMPLETIONS, READ_BYTES, Broken, Front, serve

__all__ = ["Options", "address", "chunk_ids", "main", "server_request", "token_ids"]


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


@dataclass(frozen=True, kw_only=True)
class Options:
    upstream: str = option("--upstream", help="the server's HOST:PORT: vLLM's own OpenAI server")
    listen: str = option("--listen", default="127.0.0.1:18746", help="the HOST:PORT the network attester connects to")
    id_payload: int = option("--id-payload", type=int, default=ID_PAYLOAD,
                             help="each id event's bytes, the network attester's cell payload B − 9 (default 503: the scan's "
                                  "B = 512)")
    log: str | None = option("--log", default=None, help="append each id request's record to this file (JSON lines)")


def main(opts: Options) -> int:
    upstream, listen = address(opts.upstream), address(opts.listen)
    with contextlib.ExitStack() as stack:
        out = stack.enter_context(Path(opts.log).open("a", buffering=1, encoding="utf-8")) if opts.log else None

        def log(rec: dict[str, Any]) -> None:
            assert out is not None
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
        try:
            front = Front(upstream, opts.id_payload, request_of=server_request, stream_ids=token_ids,
                          log=log if out is not None else None)
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

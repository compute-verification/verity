"""`id_relay` before a scripted server, with a fake request builder and a fake stream reader: an id request's answer is
the reader's ids as id events then the trailer, a reader's `Broken` ends it without the trailer, and every other
connection is relayed byte for byte."""
from __future__ import annotations

import asyncio
import itertools
from typing import Any

import pytest
from verity.protocols.resource_accounting.network.prover import id_relay as R
from verity.protocols.resource_accounting.network.prover.workload import FRAME_HEADER, TRAILER, id_event, id_request
from verity.protocols.resource_accounting.network.verifier import release as REL

P = 140
FIELDS = {"model": "m", "user": "u", "prompt": [128000, 791, 4062], "max_tokens": 2, "temperature": 0.0, "top_p": 1.0, "seed": 0}
SCRIPT = b"7,8|9"


def request_of(fields: dict[str, Any], host: bytes, release: bytes) -> bytes:
    return b"COPY " + host + b" " + release + repr(sorted(fields.items())).encode() + b"\r\n\r\n"


async def stream_ids(reader: asyncio.StreamReader):
    data = await reader.read()
    if data != SCRIPT:
        raise R.Broken("not the script")
    for part in data.split(b"|"):
        yield [int(t) for t in part.split(b",")]


async def scripted(answer: bytes, seen: list[bytes]) -> asyncio.Server:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        seen.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(answer)
        await writer.drain()
        writer.close()
    return await asyncio.start_server(handle, "127.0.0.1", 0)


async def through(answer: bytes, request: bytes, *, eof: bool = False) -> tuple[bytes, list[bytes], list[dict]]:
    """`request` through a relay before a server that answers ``answer``; ``eof`` ends the client's side after it, which an
    id request's client doesn't, since the relay ends that stream on the client's end (`id_relay.GONE`)."""
    seen: list[bytes] = []
    log: list[dict] = []
    ticks = itertools.count(1)
    upstream = await scripted(answer, seen)
    front = R.Front(upstream.sockets[0].getsockname()[:2], P, request_of=request_of, stream_ids=stream_ids,
                    log=log.append, clock=lambda: float(next(ticks)))
    listening = await asyncio.start_server(front.handle, "127.0.0.1", 0)
    async with upstream, listening:
        reader, writer = await asyncio.open_connection(*listening.sockets[0].getsockname()[:2])
        writer.write(request)
        await writer.drain()
        if eof:
            writer.write_eof()
        got = await reader.read()
        writer.close()
        await asyncio.gather(*front.handlers)
    return got, seen, log


def released() -> bytes:
    return REL.with_release(id_request(**FIELDS, frame_bytes=P + FRAME_HEADER.size), 4, b"run", 9)


def test_an_id_request_is_answered_with_the_readers_ids_then_the_trailer():
    got, seen, log = asyncio.run(through(SCRIPT, released()))
    assert got == b"".join(id_event(t, P) for t in (7, 8, 9)) + TRAILER
    host = seen[0].split(b" ")[1]
    assert seen == [request_of(FIELDS, host, REL.release_head(4, b"run", 9))]
    assert [(r["n"], r["bucket"], r["ids"], r["ended"], r["why"]) for r in log] == [(4, 9, 3, True, None)]


def test_a_broken_stream_ends_without_the_trailer():
    got, _, log = asyncio.run(through(b"no ids", released()))
    assert got == b"" and [(r["ended"], r["why"]) for r in log] == [(False, "not the script")]


def test_any_other_connection_is_relayed_byte_for_byte():
    got, seen, log = asyncio.run(through(b"HTTP/1.1 200 OK\r\n\r\n{}", b"GET /v1/models HTTP/1.1\r\n\r\n", eof=True))
    assert got == b"HTTP/1.1 200 OK\r\n\r\n{}" and seen == [b"GET /v1/models HTTP/1.1\r\n\r\n"] and log == []


def test_a_payload_below_an_id_event_is_refused():
    with pytest.raises(ValueError, match="an id event is at least"):
        R.Front(("127.0.0.1", 1), 137, request_of=request_of, stream_ids=stream_ids)

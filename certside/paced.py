#!/usr/bin/env python3
"""`warden_serve.py fake` with its id-form answer streamed as a model streams it:

    python certside/paced.py [--ids-below N] fake --listen HOST:PORT --first-ms F --token-ms T --id-payload P

`warden_serve`'s own fake answers an id request whole, `--first-ms` after it, and its `--token-ms` is only its text
form's. Here the first id event leaves `--first-ms` after the request and each next one `--token-ms` after the one
before, by the clock rather than by a sleep's length, so a late wake-up doesn't push the rest back; then
`io_tie.TRAILER`, at once, and the close. The ids are `fake_ids`'s, the same; `/v1/models` and the watermark stream are
answered as the fake answers them. Without `--id-payload` this is `warden_serve.py fake` unchanged.

`--ids-below N` puts the ids below N in place of `fake_ids`'s 128,256, for a model whose vocabulary is larger: the
same count of them, `fake_tokens`'s, from the same hash of the request by the same step, modulo N. An id is a u32 in
the id form (`io_tie.id_event`), so N is at most 2^32.
"""

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "network_traces"))

import warden_serve  # noqa: E402

whole = warden_serve.fake_endpoint
#: `--ids-below`, or None: `fake_ids`'s own
IDS_BELOW: int | None = None


def ids_of(body: dict) -> list[int]:
    """`warden_serve.fake_ids`, its modulus `IDS_BELOW` when given."""
    if IDS_BELOW is None:
        return warden_serve.fake_ids(body)
    h = int.from_bytes(hashlib.sha256(json.dumps(body, sort_keys=True).encode()).digest()[:4], "big")
    return [(h + 7919 * k) % IDS_BELOW for k in range(warden_serve.fake_tokens(body))]


async def paced(reader: asyncio.StreamReader, writer, *, sleep=asyncio.sleep, first_s: float = 0.1,
                token_s: float = 0.02, id_payload: int | None = None) -> None:
    if id_payload is None:
        return await whole(reader, writer, sleep=sleep, first_s=first_s, token_s=token_s)
    try:
        head = await reader.readuntil(b"\r\n\r\n")
        if head.startswith(b"POST " + warden_serve.WATERMARK_PATH + b" "):
            while await reader.read(1 << 16):
                pass
            return
        if head.startswith(b"GET /v1/models "):
            doc = json.dumps({"object": "list", "data": [{"id": "fake", "object": "model"}]}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\nconnection: close\r\n"
                         b"content-length: %d\r\n\r\n%s" % (len(doc), doc))
            await writer.drain()
            return
        m = re.search(rb"(?im)^content-length:\s*(\d+)", head)
        body = json.loads(await reader.readexactly(int(m.group(1)))) if m else {}
        clock = asyncio.get_running_loop().time
        first = clock() + first_s
        for k, x in enumerate(ids_of(body)):
            await sleep(max(0.0, first + k * token_s - clock()))
            writer.write(warden_serve.IT.id_event(x, id_payload))
            await writer.drain()
        writer.write(warden_serve.IT.TRAILER)
        await writer.drain()
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError, ValueError):
        pass
    finally:
        writer.close()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--ids-below"]:
        if len(argv) < 2 or not argv[1].isdigit() or not 1 <= int(argv[1]) <= 1 << 32:
            raise SystemExit("paced.py --ids-below N ...: N from 1 to 2^32, the bound of the output ids")
        IDS_BELOW, argv = int(argv[1]), argv[2:]
    warden_serve.fake_endpoint = paced
    raise SystemExit(warden_serve.main(argv))

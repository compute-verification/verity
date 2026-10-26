"""The torch-loop stub (`torch_loop_stub.py`): a warden's id request, as the warden forwards it, is answered id-form (each
step's id event of P bytes, then the trailer, no HTTP head), and every other request as before (the network
certifier's shim's fixed and varied bodies, `/v1/models`), an id request in other cells among them."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from verity.core.protocols.certifier import active as A
from verity.ml.definitions.certifier import io_tie as IT
from verity.core.protocols.certifier.release import with_release

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import certifier_serve as cs  # noqa: E402
import torch_loop_stub as S  # noqa: E402

P = S.ID_PAYLOAD
MODEL = S.L.Model(h=S.L.H)
NAME = f"torch-loop/mlp-w8a8-h{S.L.H}"


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


def answer(request: bytes, steps: int = 2) -> bytes:
    """The stub's whole answer to `request`, on one connection."""
    async def main() -> bytes:
        reader, writer = asyncio.StreamReader(), _Writer()
        reader.feed_data(request)
        reader.feed_eof()
        await S.Endpoint(MODEL, steps)(reader, writer)
        return bytes(writer.data)
    return asyncio.run(main())


def http(body: dict, path: str = "/v1/completions") -> bytes:
    data = json.dumps(body).encode()
    return b"POST %s HTTP/1.1\r\nhost: s\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n%s" % (
        path.encode(), len(data), data)


def ids_request(**kw) -> bytes:
    fields = {"model": NAME, "user": "user-0", "prompt": [70001, 3], "max_tokens": 2, "seed": 12345} | kw
    return A.id_request(**fields, frame_bytes=P + A.HEADER.size, offset=0)


def status(data: bytes) -> int:
    return int(data.split(b" ", 2)[1])


def test_a_wardens_id_request_is_answered_id_form_with_each_steps_id_then_the_trailer():
    outputs = S.step_outputs(MODEL, 3, 2)
    want = b"".join(IT.id_event(S.scan_id(o), P) for o in outputs) + IT.TRAILER
    assert S.id_response(outputs, P) == want and len(want) == 2 * P + len(IT.TRAILER)
    assert [S.scan_id(o) for o in outputs] == [S.step_id(o) % (1 << 17) for o in outputs]
    for request in (ids_request(), with_release(ids_request(), 7, b"\x5a" * A.RUN_ID_BYTES), with_release(ids_request(), 0)):
        assert answer(request) == want
    assert answer(ids_request(max_tokens=5)) == want
    one = answer(ids_request(max_tokens=1))
    assert one == IT.id_event(S.scan_id(outputs[0]), P) + IT.TRAILER
    cells = [IT.data_cell(S.scan_id(o), 1, P + A.HEADER.size)[A.HEADER.size:] for o in outputs]
    assert b"".join(cells) == want[:2 * P]


def test_a_tampered_endpoint_changes_only_its_tamper_at_id_requests_answer():
    """``byte`` changes one output byte before its id, ``extra`` sends one id event more, and ``late`` holds back all
    but the first event (its hold is ``late_s``, 0 here); the other id requests are answered as before."""
    outputs = S.step_outputs(MODEL, 3, 2)
    honest = S.id_response(outputs, P)
    flipped = bytes([outputs[0][0] ^ 1]) + outputs[0][1:]
    want = {"byte": IT.id_event(S.scan_id(flipped), P) + honest[P:],
            "extra": honest[:2 * P] + IT.id_event(S.scan_id(outputs[1]), P) + IT.TRAILER, "late": honest}
    assert S.scan_id(flipped) != S.scan_id(outputs[0])

    class Chunks(_Writer):
        def __init__(self) -> None:
            super().__init__()
            self.chunks: list[bytes] = []

        def write(self, data: bytes) -> None:
            super().write(data)
            self.chunks.append(bytes(data))

    for kind in S.TAMPERS:
        endpoint = S.Endpoint(MODEL, 2, tamper=kind, tamper_at=1, late_s=0)

        async def main() -> list[Chunks]:
            got = []
            for _ in range(3):
                reader, writer = asyncio.StreamReader(), Chunks()
                reader.feed_data(ids_request())
                reader.feed_eof()
                await endpoint(reader, writer)
                got.append(writer)
            return got

        first, hit, after = asyncio.run(main())
        assert bytes(first.data) == bytes(after.data) == honest and len(first.chunks) == 1
        assert bytes(hit.data) == want[kind]
        assert hit.chunks == ([honest[:P], honest[P:]] if kind == "late" else [want[kind]])
    try:
        S.Endpoint(MODEL, 2, tamper="drop")
    except ValueError as e:
        assert "--tamper" in str(e)
    else:
        raise AssertionError("an unknown tamper is refused")


def test_an_id_request_the_loop_cant_serve_is_refused_over_http():
    for kw, why in (({"temperature": 0.7}, "greedy"), ({"top_p": 0.9}, "greedy"), ({"prompt": [16]}, "below 16"),
                    ({"max_tokens": 0}, "max_tokens >= 1")):
        data = answer(ids_request(**kw))
        assert status(data) == 400 and why in json.loads(data.partition(b"\r\n\r\n")[2])["error"]


def test_every_other_request_is_answered_as_before_an_id_request_in_other_cells_among_them():
    """The shim's fixed body (`certifier_serve.request_body`) streams each step's 2H output bytes as hex over HTTP;
    an id request built for another cell size, or at the active link's offset, isn't the warden's, so it gets the
    varied path's answer (400: it asks for no token ids), as does one with another head."""
    outputs = S.step_outputs(MODEL, 3, 2)
    data = answer(http(cs.request_body(NAME, 2, prompt=[3])))
    assert data.startswith(b"HTTP/1.1 200 OK\r\n") and b"transfer-encoding: chunked" in data
    events = [json.loads(x[6:]) for x in data.split(b"\r\n") if x.startswith(b"data: {")]
    assert [e["choices"][0]["text"] for e in events] == [o.hex() for o in outputs]
    assert [e["choices"][0]["token_ids"] for e in events] == [[S.step_id(o)] for o in outputs]
    assert data.endswith(b"data: [DONE]\n\n\r\n0\r\n\r\n")

    long = [70001] * 59 + [3]
    fields = {"model": NAME, "user": "user-0", "prompt": long, "max_tokens": 2, "seed": 12345}
    assert answer(ids_request(prompt=long)) == S.id_response(outputs, P)
    for other in (A.id_request(**fields, frame_bytes=4096, offset=0), A.id_request(**fields, frame_bytes=P + 9),
                  ids_request(prompt=long).replace(b"host: server", b"host: serves", 1)):
        assert other != ids_request(prompt=long)
        got = answer(other)
        assert status(got) == 400 and "return_token_ids" in json.loads(got.partition(b"\r\n\r\n")[2])["error"]

    models = answer(b"GET /v1/models HTTP/1.1\r\nhost: s\r\n\r\n")
    assert status(models) == 200 and json.loads(models.partition(b"\r\n\r\n")[2]) == {"data": [{"id": NAME}]}

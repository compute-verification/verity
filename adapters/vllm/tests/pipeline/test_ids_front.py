"""`verity-vllm ids-front` (`pipeline.ids_front`) on CPU: before the server whose framing is vLLM's (`PlainProtocol`), an id
request as the network certifier forwards it is answered byte for byte as ``--protocol ids`` answers it, and every other
connection as the server answers it; before a server of scripted bytes, the ids come in order however the stream is
framed, and a stream that doesn't reach ``[DONE]`` ends without the trailer or any of the server's text."""
from __future__ import annotations

import asyncio
import itertools
import json
import re
import socket
import threading
import urllib.request
from typing import Any

import pytest
from verity.protocols.certifier.verifier import io_tie as IT
from verity.protocols.certifier.verifier import release as REL

from tests.pipeline import test_serve_openai as served
from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import ids_front as F
from verity_vllm.pipeline import serve_protocol as proto

P = 140
FIELDS = {"model": "m", "user": "u", "prompt": [128000, 791, 4062], "max_tokens": 2, "temperature": 0.0, "top_p": 1.0, "seed": 0}
STREAM = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream; charset=utf-8\r\ntransfer-encoding: chunked\r\n\r\n"


def released(payload: int = P, n: int = 4, bucket: int = 9) -> bytes:
    """An id request in cells of ``payload`` + 9 bytes as the network certifier forwards it: release `n` of run ``run``,
    at `bucket`."""
    return REL.with_release(served.id_request(payload), n, b"run", bucket)


def exchange(address: tuple[str, int], request: bytes) -> bytes:
    """`request` on a connection of its own, and every byte that comes back before the other side closes."""
    with socket.create_connection(address) as c:
        c.sendall(request)
        got = b""
        while chunk := c.recv(65536):
            got += chunk
    return got


def chunked(*parts: bytes) -> bytes:
    """An HTTP chunked body of `parts`, each one chunk, and its end."""
    return b"".join(b"%x\r\n%s\r\n" % (len(p), p) for p in parts) + b"0\r\n\r\n"


def event(doc: Any, end: bytes = b"\n\n") -> bytes:
    return b"data: " + (doc if isinstance(doc, bytes) else json.dumps(doc).encode()) + end


def choice(ids: list[int] | None, text: str = "", **more: Any) -> dict:
    """A completion chunk of one choice, as vLLM's server streams it under ``return_token_ids``."""
    return {"id": "cmpl-1", "object": "text_completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": None, "stop_reason": None,
                         **({} if ids is None else {"token_ids": ids}), **more}]}


class Scripted(threading.Thread):
    """A server of scripted bytes: of each connection it keeps the request (its head and the body it states), sends
    ``answer`` and closes."""

    def __init__(self, answer: bytes, connections: int = 1) -> None:
        super().__init__(daemon=True)
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.address = self.sock.getsockname()[:2]
        self.answer, self.connections, self.requests = answer, connections, []
        self.start()

    def run(self) -> None:
        with self.sock:
            for _ in range(self.connections):
                c, _ = self.sock.accept()
                with c:
                    got = b""
                    while b"\r\n\r\n" not in got and (more := c.recv(65536)):
                        got += more
                    head, _, body = got.partition(b"\r\n\r\n")
                    stated = re.search(rb"content-length: (\d+)", head)
                    while len(body) < int(stated.group(1) if stated else 0) and (more := c.recv(65536)):
                        body += more
                    self.requests.append(head + b"\r\n\r\n" + body)
                    c.sendall(self.answer)


class Echo(threading.Thread):
    """A server that sends one connection's bytes back as they come, and closes when the client's side ends."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.address = self.sock.getsockname()[:2]
        self.start()

    def run(self) -> None:
        with self.sock:
            c, _ = self.sock.accept()
            with c:
                while data := c.recv(65536):
                    c.sendall(data)


class Fronted:
    """`ids_front.serve` on a thread's own loop, before the server at ``upstream``, its clock a counter (1, 2, 3, ...
    seconds a reading) and its records kept in ``log``."""

    def __init__(self, upstream: tuple[str, int], payload: int = P) -> None:
        ticks = itertools.count(1)
        self.log: list[dict] = []
        self.front = F.Front(tuple(upstream), payload, log=self.log.append, clock=lambda: float(next(ticks)))
        self.ready = threading.Event()
        self.address: tuple[str, int] | None = None

    def __enter__(self) -> Fronted:
        def announce(at: tuple[str, int]) -> None:
            self.address = at
            self.ready.set()

        async def main() -> None:
            self.loop, self.stop = asyncio.get_running_loop(), asyncio.Event()
            await F.serve(self.front, ("127.0.0.1", 0), announce=announce, stop=self.stop)

        self.thread = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
        self.thread.start()
        self.ready.wait()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join()

    def send(self, request: bytes) -> bytes:
        return exchange(self.address, request)


def test_an_id_request_is_its_fields_whatever_release_headers_it_came_with():
    """`serve_protocol.id_request_fields`: the fields of an id request in its P + 9-byte cells, bare or as the network
    certifier forwards it (the release, the run and the bucket); None in other cells, cut short, or changed by a byte."""
    for request in (served.id_request(P), released(), REL.with_release(served.id_request(P), 0)):
        assert proto.id_request_fields(request, P) == FIELDS
    assert proto.id_request_fields(released(), P + 1) is None and proto.id_request_fields(released()[:-1], P) is None
    assert proto.id_request_fields(released().replace(b'"stream":true', b'"stream":True'), P) is None
    assert proto.id_request_fields(b"", P) is None


def test_the_servers_copy_keeps_the_release_and_asks_for_the_token_ids():
    """`server_request`: a streamed POST of the id request's fields with ``return_token_ids``, no stop strings, on a
    connection the server closes, under the release headers the network certifier put on it; none where it put none."""
    head = released().partition(b"\r\n\r\n")[0] + b"\r\n\r\n"
    got_head, _, body = F.server_request(head, FIELDS, b"127.0.0.1:8000").partition(b"\r\n\r\n")
    assert json.loads(body) == FIELDS | {"stop": [], "stream": True, "return_token_ids": True}
    assert got_head == (b"POST /v1/completions HTTP/1.1\r\nhost: 127.0.0.1:8000\r\ncontent-type: application/json\r\n"
                        b"x-verity-release: 4\r\nx-verity-release-run: " + b"run".hex().encode() +
                        b"\r\nx-verity-release-bucket: 9\r\ncontent-length: %d\r\nconnection: close" % len(body))
    assert REL.release_of(got_head + b"\r\n\r\n") == (4, b"run", 9)
    bare = F.server_request(served.id_request(P).partition(b"\r\n\r\n")[0] + b"\r\n\r\n", FIELDS, b"h")
    assert b"x-verity" not in bare and REL.release_of(bare.partition(b"\r\n\r\n")[0] + b"\r\n\r\n") is None


@pytest.mark.parametrize("payload", [P, proto.ID_PAYLOAD])
def test_before_vllms_framing_an_id_request_is_answered_as_protocol_ids_answers_it(tmp_path, payload):
    """The front before the server whose framing is vLLM's (`PlainProtocol`, its usage chunk forced): the network
    certifier's id request comes back as ``--protocol ids`` sends it from the same engine, each sampled id one event of P
    bytes, then the trailer, with no HTTP head, no usage and no text. The record has the release, the counts and each
    write's time after the arrival, and no id."""
    with served.Serving(tmp_path, commit=False) as s, Fronted(s.http.address, payload) as f:
        raw = f.send(released(payload))
        assert f.send(served.id_request(payload)) == raw
    assert raw == b"".join(IT.id_event(t, payload) for t in (4853, 4884)) + IT.TRAILER
    ids = proto.IdsProtocol(True, payload=payload, base=proto.PlainProtocol(True))
    with served.Serving(tmp_path / "ids", commit=False, protocol=ids) as direct:
        assert direct.send(released(payload)) == raw
    assert f.log == [{"n": 4, "run": b"run".hex(), "bucket": 9, "prompt": 3, "ids": 2, "ended": True, "why": None, "at": 1.0,
                      "ms": [1000.0, 2000.0, 3000.0]},
                     {"n": None, "run": None, "bucket": None, "prompt": 3, "ids": 2, "ended": True, "why": None, "at": 5.0,
                      "ms": [1000.0, 2000.0, 3000.0]}]


def test_every_other_connection_gets_the_servers_own_answer(tmp_path):
    """Through the front, what isn't an id request in the front's cells reaches the server unchanged and its answer
    comes back whole: ``/v1/models``, ``/tokenize``, the certifier's shim's streamed POST (vLLM's chunks, with their
    token ids), and an id request in other cells, which the server answers in HTTP. None is recorded."""
    with served.Serving(tmp_path, commit=False) as s, Fronted(s.http.address, proto.ID_PAYLOAD) as f:
        url = f"http://{f.address[0]}:{f.address[1]}"
        models = json.loads(urllib.request.urlopen(url + "/v1/models").read())  # noqa: S310 (the test's own 127.0.0.1 endpoint)
        tokenize = json.dumps({"model": "m", "prompt": "a bb ccc"}).encode()
        req = urllib.request.Request(url + "/tokenize", tokenize, {"content-type": "application/json"})  # noqa: S310 (as above)
        tokens = json.loads(urllib.request.urlopen(req).read())  # noqa: S310 (the test's own 127.0.0.1 endpoint)
        shim = served.shims_request(s.http.address[1])
        through, direct = f.send(shim), s.send(shim)
        other = f.send(released(P))
    assert [m["id"] for m in models["data"]] == ["m"] and tokens["tokens"] == [1, 101, 102, 103]
    head, body = served.dechunked(through)
    assert head.startswith(b"HTTP/1.1 200 OK\r\n") and served.dechunked(direct)[0] == head
    chunks = [json.loads(e) for e in served.events(body)[:-1]]
    assert [t for c in chunks for ch in c["choices"] for t in ch["token_ids"]] == [4853, 4884]
    assert other.startswith(b"HTTP/1.1 200 OK\r\n") and IT.ID_EVENT_HEAD not in other
    assert f.log == []


def test_a_relayed_connection_streams_both_ways_until_both_sides_end():
    """A connection that isn't an id request's is a pipe: the client's bytes reach the server as they come, the
    server's come back as they come, and the client's end reaches the server, whose end closes the connection (the
    watermark's stream, which the network certifier writes for a whole run)."""
    echo = Echo()
    with Fronted(echo.address) as f, socket.create_connection(f.address) as c:
        sent = b"POST /verity/watermark HTTP/1.1\r\ntransfer-encoding: chunked\r\n\r\n"
        for part in (sent, b"4\r\n0 0\n\r\n", b"4\r\n1 2\n\r\n"):
            c.sendall(part)
            got = b""
            while len(got) < len(part):
                got += c.recv(65536)
            assert got == part
        c.shutdown(socket.SHUT_WR)
        assert c.recv(65536) == b""
    echo.join()


def test_the_ids_come_in_order_however_the_server_frames_its_stream():
    """The id form holds exactly the chunks' token ids in order, whatever else the stream carries: an event cut across
    two HTTP chunks, two events in one, events ended by CRLFs, a chunk of several ids, one of none with the finish, the
    usage chunk, a comment line, and the prompt's ids on the first chunk. The server's copy of the request is
    `server_request`'s."""
    first = choice([7], " a", prompt_token_ids=[1, 2, 3])
    whole = event(first) + event(choice([8, 9], " b c"), b"\r\n\r\n") + b": keep-alive\n\n" + \
        event(choice([], finish_reason="length")) + event({"id": "cmpl-1", "choices": [], "usage": {"completion_tokens": 3}}) + \
        event(b"[DONE]")
    for body in (chunked(whole[:11], whole[11:60], whole[60:]), chunked(whole), whole):
        head = STREAM if body is not whole else STREAM.replace(b"transfer-encoding: chunked\r\n", b"")
        server = Scripted(head + body)
        with Fronted(server.address) as f:
            assert f.send(released()) == b"".join(IT.id_event(t, P) for t in (7, 8, 9)) + IT.TRAILER
        server.join()
        assert server.requests == [F.server_request(released().partition(b"\r\n\r\n")[0] + b"\r\n\r\n", FIELDS,
                                                    b"%s:%d" % (server.address[0].encode(), server.address[1]))]
        assert [(r["ids"], r["ended"], r["why"]) for r in f.log] == [(3, True, None)]


REFUSED = b'HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: 27\r\n\r\n{"error":{"message":"no"}}\n'


@pytest.mark.parametrize("answer, sent, why", [
    (REFUSED, [], "the server answered 400"),
    (b"", [], "the server's stream broke off (IncompleteReadError)"),
    (STREAM + chunked(event(choice([7], " a")), event({"error": {"message": "the engine broke", "code": 500}}), event(b"[DONE]")),
     [7], "the server sent an error event"),
    (STREAM + chunked(event(choice(None, " a")), event(b"[DONE]")), [], "the server sent a chunk without its token ids"),
    (STREAM + chunked(event(choice([7, -1], " a b")), event(b"[DONE]")), [], "the server sent a chunk without its token ids"),
    (STREAM + chunked(event(choice([7], " a")), event(b"{not json"), event(b"[DONE]")), [7],
     "the server's stream broke off (JSONDecodeError)"),
    (STREAM + chunked(event(choice([7], " a")))[:-5], [7], "the server's stream broke off (IncompleteReadError)"),
    (STREAM + chunked(event(choice([7], " a"))), [7], "the server's stream ended before [DONE]"),
])
def test_a_stream_that_does_not_reach_done_ends_without_the_trailer(answer, sent, why):
    """Where the server refuses the request, breaks off, sends an error event, text without its ids, an id that isn't
    a u32 or an event that isn't JSON, or ends before ``[DONE]``, the front sends the ids that came before and closes:
    no trailer, and nothing of what the server said. The record says why."""
    server = Scripted(answer)
    with Fronted(server.address) as f:
        raw = f.send(released())
    server.join()
    assert raw == b"".join(IT.id_event(t, P) for t in sent) and IT.TRAILER not in raw
    assert [(r["ids"], r["ended"], r["why"]) for r in f.log] == [(len(sent), False, why)]


def test_a_server_that_is_not_there_closes_the_connection():
    """With nothing listening at the server's address, an id request's connection and any other's are closed with
    nothing sent; the id request's record says a connection broke."""
    with socket.create_server(("127.0.0.1", 0)) as gone:
        address = gone.getsockname()[:2]
    with Fronted(address) as f:
        assert f.send(released()) == b"" and f.send(b"GET /v1/models HTTP/1.1\r\n\r\n") == b""
    assert [(r["ended"], r["why"]) for r in f.log] == [(False, "a connection broke (ConnectionRefusedError)")]


def test_the_command_is_registered_with_the_scans_payload_as_its_default():
    opts = cli.parse("ids-front", ["--upstream", "127.0.0.1:18745"], environ={})
    assert (opts.upstream, opts.listen, opts.id_payload, opts.log) == ("127.0.0.1:18745", "127.0.0.1:18746", proto.ID_PAYLOAD, None)
    assert F.address("node-a:18745") == ("node-a", 18745)
    for bad in ("18745", "host:", ":18745", "host:port"):
        with pytest.raises(UsageError):
            F.address(bad)
    with pytest.raises(ValueError, match="an id event is at least"):
        F.Front(("127.0.0.1", 1), IT.ID_EVENT_BYTES - 1)

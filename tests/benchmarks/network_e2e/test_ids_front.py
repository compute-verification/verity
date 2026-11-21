"""The id relay (`benchmarks/network_e2e/ids_front.py`) on CPU: before a server of scripted bytes, an id request as the
network attester forwards it is answered as id events and the trailer however the stream is framed, a stream that doesn't
reach ``[DONE]`` ends without the trailer or any of the server's text, and every other connection is the server's own."""
from __future__ import annotations

import asyncio
import contextlib
import gc
import http.client
import importlib.util
import itertools
import json
import re
import socket
import threading
from pathlib import Path
from typing import Any

import pytest
from verity import release as REL

_SPEC = importlib.util.spec_from_file_location("ids_front", Path(__file__).resolve().parents[3] / "benchmarks" / "network_e2e" / "ids_front.py")
assert _SPEC is not None and _SPEC.loader is not None
F = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(F)

P = 140
FIELDS = {"model": "m", "user": "u", "prompt": [128000, 791, 4062], "max_tokens": 2, "temperature": 0.0, "top_p": 1.0, "seed": 0}
STREAM = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream; charset=utf-8\r\ntransfer-encoding: chunked\r\n\r\n"


def bare_request(payload: int = P) -> bytes:
    """An id request in cells of ``payload`` + 9 bytes without release headers."""
    return REL.id_request(**FIELDS, frame_bytes=payload + REL.FRAME_HEADER.size, offset=0)


def released(payload: int = P, n: int = 4, bucket: int = 9) -> bytes:
    """An id request in cells of ``payload`` + 9 bytes as the network attester forwards it: release `n` of run ``run``,
    at `bucket`."""
    return REL.with_release(bare_request(payload), n, b"run", bucket)


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


def request_on(c: socket.socket) -> bytes:
    """The request a connection brings: its head and the body it states."""
    got = b""
    while b"\r\n\r\n" not in got and (more := c.recv(65536)):
        got += more
    head, _, body = got.partition(b"\r\n\r\n")
    stated = re.search(rb"content-length: (\d+)", head)
    while len(body) < int(stated.group(1) if stated else 0) and (more := c.recv(65536)):
        body += more
    return head + b"\r\n\r\n" + body


class Scripted(threading.Thread):
    """A server of scripted bytes: of each connection it keeps the request (`request_on`), sends ``answer`` and closes."""

    requests: list[bytes]

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
                    self.requests.append(request_on(c))
                    c.sendall(self.answer)


class Silent(threading.Thread):
    """A server that takes one connection's request, sends ``answer`` and then nothing, as vLLM's own server does for a
    request the tick gate dropped: ``sent`` is set once ``answer`` is out, and the thread ends only when the other side
    has ended the connection, closed or reset."""

    def __init__(self, answer: bytes) -> None:
        super().__init__(daemon=True)
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.address = self.sock.getsockname()[:2]
        self.answer, self.sent = answer, threading.Event()
        self.requests: list[bytes] = []
        self.start()

    def run(self) -> None:
        with self.sock:
            c, _ = self.sock.accept()
            with c:
                self.requests.append(request_on(c))
                c.sendall(self.answer)
                self.sent.set()
                with contextlib.suppress(ConnectionError):
                    while c.recv(65536):
                        pass


class Paused(threading.Thread):
    """A server that takes one connection's request, sends ``first``, reads until the other side has ended its sending
    (``ended`` is set then), sends ``rest`` only once ``go`` is set, and closes."""

    def __init__(self, first: bytes, rest: bytes) -> None:
        super().__init__(daemon=True)
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.address = self.sock.getsockname()[:2]
        self.first, self.rest, self.ended, self.go = first, rest, threading.Event(), threading.Event()
        self.start()

    def run(self) -> None:
        with self.sock:
            c, _ = self.sock.accept()
            with c, contextlib.suppress(OSError):        # a front that has closed its side takes no more
                try:
                    request_on(c)
                    c.sendall(self.first)
                    while c.recv(65536):
                        pass
                finally:
                    self.ended.set()
                self.go.wait()
                c.sendall(self.rest)


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


def copy_of(head: bytes, host: bytes) -> bytes:
    """`server_request` of `FIELDS` under `head`'s release headers, as the relay hands them to it."""
    rel = REL.release_of(head)
    return F.server_request(FIELDS, host, b"" if rel is None else REL.release_head(*rel))


class Fronted:
    """`serve` on a thread's own loop, before the server at ``upstream``, its clock a counter (1, 2, 3, ...
    seconds a reading) and its records kept in ``log``, ``recorded`` released once a record."""

    address: tuple[str, int]                 # where the front listens, known once `__enter__` returns

    def __init__(self, upstream: tuple[str, int], payload: int = P) -> None:
        ticks = itertools.count(1)
        self.log: list[dict] = []
        self.recorded = threading.Semaphore(0)

        def log(rec: dict) -> None:
            self.log.append(rec)
            self.recorded.release()
        self.front = F.Front(tuple(upstream), payload, request_of=F.server_request, stream_ids=F.token_ids, log=log,
                             clock=lambda: float(next(ticks)))
        self.ready = threading.Event()

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

    def ask(self, method: str, path: str, body: bytes | None = None) -> Any:
        """The JSON the front answers an HTTP client's request with, on a connection of its own that the client asks to
        be closed; any status but 200 fails the test."""
        conn = http.client.HTTPConnection(*self.address)
        try:
            conn.request(method, path, body, {"connection": "close"} | ({"content-type": "application/json"} if body else {}))
            resp = conn.getresponse()
            assert resp.status == 200, (path, resp.status, resp.reason)
            return json.loads(resp.read())
        finally:
            conn.close()


def test_an_id_request_is_its_fields_whatever_release_headers_it_came_with():
    """`release.id_request_fields`: the fields of an id request in its P + 9-byte cells, bare or as the network
    attester forwards it (the release, the run and the bucket); why not, in other cells, cut short, or changed by a byte."""
    for request in (bare_request(P), released(), REL.with_release(bare_request(P), 0)):
        assert REL.id_request_fields(request, P) == FIELDS
    for other, p in ((released(), P + 1), (released()[:-1], P), (released().replace(b'"stream":true', b'"stream":True'), P),
                     (b"", P)):
        assert isinstance(REL.id_request_fields(other, p), str)


def test_the_servers_copy_keeps_the_release_and_asks_for_the_token_ids():
    """`server_request`: a streamed POST of the id request's fields with ``return_token_ids``, no stop strings, on a
    connection the server closes, under the release headers the network attester put on it; none where it put none."""
    head = released().partition(b"\r\n\r\n")[0] + b"\r\n\r\n"
    got_head, _, body = copy_of(head, b"127.0.0.1:8000").partition(b"\r\n\r\n")
    assert json.loads(body) == FIELDS | {"stop": [], "stream": True, "return_token_ids": True}
    assert got_head == (b"POST /v1/completions HTTP/1.1\r\nhost: 127.0.0.1:8000\r\ncontent-type: application/json\r\n"
                        b"x-verity-release: 4\r\nx-verity-release-run: " + b"run".hex().encode() +
                        b"\r\nx-verity-release-bucket: 9\r\ncontent-length: %d\r\nconnection: close" % len(body))
    assert REL.release_of(got_head + b"\r\n\r\n") == (4, b"run", 9)
    bare = copy_of(bare_request(P).partition(b"\r\n\r\n")[0] + b"\r\n\r\n", b"h")
    assert b"x-verity" not in bare and REL.release_of(bare.partition(b"\r\n\r\n")[0] + b"\r\n\r\n") is None


@pytest.mark.parametrize("payload", [P, REL.ID_PAYLOAD])
def test_an_id_request_is_answered_in_the_id_form_from_the_servers_stream(payload):
    """Before a server that streams each sampled id in its own chunk: the network attester's id request comes back as
    each id one event of P bytes, then the trailer, with no HTTP head, no usage and no text. The record has the release,
    the counts and each write's time after the arrival, and no id."""
    answer = STREAM + chunked(event(choice([4853], " a")), event(choice([4884], " b")), event(b"[DONE]"))
    server = Scripted(answer, connections=2)
    with Fronted(server.address, payload) as f:
        raw = f.send(released(payload))
        assert f.send(bare_request(payload)) == raw
    server.join()
    assert raw == b"".join(REL.id_event(t, payload) for t in (4853, 4884)) + REL.TRAILER
    assert f.log == [{"n": 4, "run": b"run".hex(), "bucket": 9, "prompt": 3, "ids": 2, "ended": True, "why": None, "at": 1.0,
                      "ms": [1000.0, 2000.0, 3000.0]},
                     {"n": None, "run": None, "bucket": None, "prompt": 3, "ids": 2, "ended": True, "why": None, "at": 5.0,
                      "ms": [1000.0, 2000.0, 3000.0]}]


def test_every_other_connection_gets_the_servers_own_answer():
    """Through the front, what isn't an id request in the front's cells reaches the server unchanged and its answer comes
    back whole: a model listing, a tokenization, the streamed POST of a client that brings its own request, and an id
    request in other cells. None is recorded."""
    answer = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 11\r\nconnection: close\r\n\r\n{\"ok\":true}\n"
    requests = [b"GET /v1/models HTTP/1.1\r\nhost: h\r\nconnection: close\r\n\r\n",
                b"POST /tokenize HTTP/1.1\r\nhost: h\r\ncontent-length: 10\r\nconnection: close\r\n\r\n0123456789",
                b"POST /v1/completions HTTP/1.1\r\nhost: h\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}",
                released(P)]
    server = Scripted(answer, connections=len(requests))
    with Fronted(server.address) as f:
        assert [f.send(r) for r in requests] == [answer] * len(requests)
    server.join()
    assert server.requests == requests
    assert f.log == []




def test_a_relayed_connection_streams_both_ways_until_both_sides_end():
    """A connection that isn't an id request's is a pipe: the client's bytes reach the server as they come, the
    server's come back as they come, and the client's end reaches the server, whose end closes the connection (the
    watermark's stream, which the network attester writes for a whole run)."""
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
            assert f.send(released()) == b"".join(REL.id_event(t, P) for t in (7, 8, 9)) + REL.TRAILER
        server.join()
        assert server.requests == [copy_of(released().partition(b"\r\n\r\n")[0] + b"\r\n\r\n",
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
    assert raw == b"".join(REL.id_event(t, P) for t in sent) and REL.TRAILER not in raw
    assert [(r["ids"], r["ended"], r["why"]) for r in f.log] == [(len(sent), False, why)]


@pytest.mark.parametrize("field, name", [(b"x-verity-release: 4x", "x-verity-release"),
                                         (b"x-verity-release: 4\r\nx-verity-release-run: zz", "x-verity-release-run"),
                                         (b"x-verity-release: 4\r\nx-verity-release-bucket: -1", "x-verity-release-bucket")])
def test_an_id_request_with_a_malformed_release_is_refused_by_name_and_the_front_goes_on(field, name):
    """A malformed release header on an id request (`release.release_of`'s refusal) is no release: the front answers
    nothing, opens no server connection for it and records the refusal's name as why; the next request is served."""
    request = bare_request(P).replace(b"\r\n", b"\r\n" + field + b"\r\n", 1)
    server = Scripted(STREAM + chunked(event(choice([7], " a")), event(b"[DONE]")))
    with Fronted(server.address) as f:
        assert f.send(request) == b""
        assert f.send(released()) == REL.id_event(7, P) + REL.TRAILER
    server.join()
    assert len(server.requests) == 1
    assert [(r["n"], r["ended"], r["why"] and r["why"].split(":")[0]) for r in f.log] == [(None, False, name), (4, True, None)]


@pytest.mark.parametrize("answer, sent", [
    (b"", []),                                                              # the server holds the request and says nothing
    (STREAM, []),                                                           # its 200 head and no chunk
    (STREAM + chunked(event(choice([7], " a")))[:-5], [7]),                 # one chunk, then no more
])
@pytest.mark.parametrize("half", [False, True])
def test_once_the_network_attesters_side_has_ended_the_front_ends_the_servers(answer, sent, half):
    """While the front waits on the server's stream of an id request (the server has sent nothing, its 200 head and no
    chunk, or a chunk and no more, as vLLM's own server does for a request the tick gate dropped), the network
    attester's side ends, closed or only its sending half. The front then closes its connection to the server, which
    is what tells that server to abort the request, and its own to the network attester, with the ids so far and no trailer;
    the record says why. Without it the server's thread, which waits for that close, never ends."""
    server = Silent(answer)
    with Fronted(server.address) as f:
        with socket.create_connection(f.address) as c:
            c.sendall(released())
            server.sent.wait()
            got = b""
            while len(got) < len(sent) * P:
                got += c.recv(65536)
            if half:
                c.shutdown(socket.SHUT_WR)
                assert c.recv(65536) == b""
        server.join()
        f.recorded.acquire()
    assert got == b"".join(REL.id_event(t, P) for t in sent)
    assert server.requests == [copy_of(released().partition(b"\r\n\r\n")[0] + b"\r\n\r\n",
                                       b"%s:%d" % (server.address[0].encode(), server.address[1]))]
    assert f.log == [{"n": 4, "run": b"run".hex(), "bucket": 9, "prompt": 3, "ids": len(sent), "ended": False, "why": F.GONE,
                      "at": 1.0, "ms": [1000.0] * len(sent)}]


def test_a_relayed_connection_whose_client_has_ended_its_sending_gets_the_servers_answer_whole():
    """Once a client has ended its sending, asyncio no longer watches its connection, so the handler that waits on the
    server for the rest of the answer is held by nothing but itself. A collection during that wait must not destroy
    it: the answer still comes whole."""
    first, rest = b"HTTP/1.1 200 OK\r\ncontent-length: 10\r\n\r\n12345", b"67890"
    server = Paused(first, rest)

    async def collect() -> None:
        gc.collect()

    try:
        with Fronted(server.address) as f, socket.create_connection(f.address) as c:
            c.sendall(b"GET /v1/models HTTP/1.1\r\n\r\n")
            c.shutdown(socket.SHUT_WR)
            got = b""
            while len(got) < len(first):
                got += c.recv(65536)
            server.ended.wait()                          # the front has read the client's end and passed it on
            asyncio.run_coroutine_threadsafe(collect(), f.loop).result()
            server.go.set()
            with contextlib.suppress(ConnectionError):
                while chunk := c.recv(65536):
                    got += chunk
    finally:
        server.go.set()
    server.join()
    assert got == first + rest


def test_a_server_that_is_not_there_closes_the_connection():
    """With nothing listening at the server's address, an id request's connection and any other's are closed with
    nothing sent; the id request's record says a connection broke, and the other's break record names the server."""
    with socket.create_server(("127.0.0.1", 0)) as gone:
        address = gone.getsockname()[:2]
    with Fronted(address) as f:
        assert f.send(released()) == b"" and f.send(b"GET /v1/models HTTP/1.1\r\n\r\n") == b""
    ids, other = f.log
    assert (ids["ended"], ids["why"]) == (False, "a connection broke (ConnectionRefusedError)")
    assert (other["broken"], other["error"], set(other)) == ("server", "ConnectionRefusedError", {"broken", "error", "at"})


def test_the_command_defaults_to_the_scans_payload_and_runs_the_relay_with_its_readers(monkeypatch, capsys):
    """The command runs `run` with its options and the server's own readers (`server_request`, `token_ids`), and writes
    the address the relay listens on."""
    called: list = []

    async def run(upstream, listen, payload, *, server_request, stream_token_ids, log, stop, announce):
        called.append((upstream, listen, payload, server_request, stream_token_ids, log))
        announce(listen)

    monkeypatch.setattr(F, "run", run)
    assert F.main(["--upstream", "127.0.0.1:18745"]) == 0
    assert called == [(("127.0.0.1", 18745), ("127.0.0.1", 18746), REL.ID_PAYLOAD, F.server_request, F.token_ids, None)]
    assert json.loads(capsys.readouterr().out) == {"listening": ["127.0.0.1", 18746]}
    assert F.address("node-a:18745") == ("node-a", 18745)
    for bad in ("18745", "host:", ":18745", "host:port"):
        with pytest.raises(SystemExit, match="HOST:PORT"):
            F.address(bad)


def test_a_broken_stream_is_the_hooks_broken():
    from verity import hooks as H
    with pytest.raises(H.Broken, match="error event"):
        F.chunk_ids(b'{"error": "x"}')
    assert F.Broken is H.Broken

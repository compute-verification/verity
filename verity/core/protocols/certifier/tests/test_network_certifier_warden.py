"""The active warden (`warden`): driven tick by tick on a `VirtualClock`, its clients in-memory connections and its server
a fake token port, it releases honest traffic exactly on its schedule, fails a late token closed, queues past its lanes
then refuses with 429, never lets a server byte or a token past `max_tokens` reach a client, renders exactly the text of
the ids, and puts the same number of same-size frames in the same buckets whatever the content."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import random

import pytest

from verity.core.protocols.certifier.active import VirtualClock
from verity.core.protocols.certifier.grid import Reason
from verity.core.protocols.certifier.warden import (CODES, DUMMY, FAIL, FINAL, HEAD, NEUTRAL, TOKEN, Malformed,
                                                    Refused, Renderer, Request, Schedule, Token, TokenPort,
                                                    Vocabulary, Warden, WardenParams, audit, canonical, parse_token,
                                                    read_record_frame, render)

BOS, EOS = 0, 1
PIECES = [b"<s>", b"</s>", b"a", b"b", b" ", b"\n", b"\xc3", b"\xa9", b"\xff", b'"', b"\\", b"STOP", b"x" * 40,
          "é".encode(), b"\x01", b"ST", b"OP"]
VOCAB = Vocabulary(tuple(PIECES), frozenset({BOS, EOS}), frozenset({EOS}))
A, B_, SPACE, NL, C3, A9, FF, QUOTE, BACKSLASH, STOP, LONG, E_ACUTE, CTRL, ST, OP = range(2, 17)
MODEL = "toy"
PARAMS = WardenParams(frame_bytes=512, slots=2, bucket_ms=100, delay=2, lanes=2, queue=2, window=16, max_tokens=8,
                      max_prompt=8, context=32, record_bytes=64, request_bytes=256)
SCHEDULE = Schedule(PARAMS, VOCAB, MODEL)
TAU = PARAMS.bucket_ms / 1000
FB = PARAMS.frame_bytes


def body(stream: bool = True, max_tokens: int = 8, prompt=(BOS, A), **extra) -> dict:
    return {"model": MODEL, "prompt": list(prompt), "max_tokens": max_tokens, "stream": stream} | extra


def http(doc: dict | str, path: str = "/v1/completions", method: str = "POST") -> bytes:
    data = (doc if isinstance(doc, str) else json.dumps(doc)).encode()
    return (f"{method} {path} HTTP/1.1\r\nhost: warden\r\ncontent-type: application/json\r\n"
            f"content-length: {len(data)}\r\n\r\n").encode() + data


class Server:
    """A fake token port: request q's ids (`ids[q]`), each yielded at once unless a gate in `gates[(q, position)]`
    holds it; `extra[q]` is what it sends past them. `at[q]` is where its stream stands: "gate" or "end"."""

    def __init__(self, ids: dict[int, list], gates: dict | None = None) -> None:
        self.ids, self.gates = ids, gates or {}
        self.at: dict[int, str] = {}
        self.requests: list[Request] = []
        self.changed = asyncio.Event()

    def _mark(self, ordinal: int, where: str) -> None:
        self.at[ordinal] = where
        self.changed.set()

    async def until(self, predicate) -> None:
        while not predicate():
            self.changed.clear()
            await self.changed.wait()

    async def stream(self, request: Request):
        self.requests.append(request)
        try:
            for p, item in enumerate(self.ids[request.ordinal]):
                gate = self.gates.get((request.ordinal, p))
                if gate is not None and not gate.is_set():
                    self._mark(request.ordinal, "gate")
                    await gate.wait()
                yield item if isinstance(item, Token) else Token(item, p, None)
        finally:
            self._mark(request.ordinal, "end")


class Client:
    """One client connection: the request in its reader, and every write with the bucket it was made in."""

    def __init__(self, warden: Warden, request: bytes) -> None:
        self.warden, self.writes, self.closed = warden, [], False
        self.transport = self
        self.reader = asyncio.StreamReader()
        self.reader.feed_data(request)
        self.reader.feed_eof()
        self.task = asyncio.create_task(warden.handle(self.reader, self))

    def write(self, data: bytes) -> None:
        assert not self.closed
        self.writes.append((self.warden.ticks, bytes(data)))

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def get_write_buffer_size(self) -> int:
        return 0

    @property
    def data(self) -> bytes:
        return b"".join(d for _, d in self.writes)

    def frames(self) -> list[tuple[int, int]]:
        """(bucket, size) of each B-byte frame written."""
        out = []
        for t, d in self.writes:
            assert len(d) % FB == 0
            out += [(t, FB)] * (len(d) // FB)
        return out


async def drive(server: Server, arrivals: dict[int, list], *, windows: int = 2, params: WardenParams = PARAMS,
                late: dict[int, float] | None = None, published: list | None = None):
    """Serve `arrivals[t]` (request bytes) just before bucket t's tick, every token the server has ready by each tick,
    for `windows` windows; `late[t]` puts tick t that many seconds past its boundary."""
    schedule = SCHEDULE if params == PARAMS else Schedule(params, VOCAB, MODEL)
    clock = VirtualClock(-TAU / 2)
    warden = Warden(schedule, server, origin=0.0, windows=windows, clock=clock, sleep=clock.sleep,
                    publish=None if published is None else lambda record, grid: published.append(record))
    running = asyncio.create_task(warden.run())
    clients: list[Client] = []
    for t in range(windows * params.window):
        for request in arrivals.get(t, ()):
            clients.append(Client(warden, request))
            await warden.until(lambda w, n=len(clients): w.ordinals + len(w.refused) == n)
        await server.until(lambda: all(s.request.ordinal in server.at for s in warden.active))
        clock.advance(t * TAU + (late or {}).get(t, 0))
        await warden.until(lambda w, t=t: w.ticks > t)
    await running
    await asyncio.gather(*(c.task for c in clients))
    return warden, clients


def run(server_ids, arrivals, **kw):
    async def go():
        server = Server(*server_ids) if isinstance(server_ids, tuple) else Server(server_ids)
        warden, clients = await drive(server, arrivals, **kw)
        return warden, clients, server
    return asyncio.run(go())


def chunked(data: bytes) -> tuple[bytes, bytes]:
    """(head, body) of a chunked HTTP/1.1 response, checking that it ends with the last chunk."""
    head, _, rest = data.partition(b"\r\n\r\n")
    out = bytearray()
    while True:
        size, _, rest = rest.partition(b"\r\n")
        n = int(size, 16)
        if n == 0:
            assert rest == b"\r\n"
            return head, bytes(out)
        out += rest[:n]
        assert rest[n:n + 2] == b"\r\n"
        rest = rest[n + 2:]


def events(data: bytes) -> list:
    """A streamed response's events: each `data:` event's JSON (or "[DONE]"), comments dropped."""
    head, payload = chunked(data)
    assert head.startswith(b"HTTP/1.1 200 OK") and b"text/event-stream" in head
    out = []
    for event in payload.split(b"\n\n"):
        if event.startswith(b"data: "):
            text = event[len(b"data: "):].decode().rstrip(" ")
            out.append(text if text == "[DONE]" else json.loads(text))
        else:
            assert event.strip(b" :") == b""
    return out


def streamed_text(data: bytes) -> tuple[str, str | None]:
    evs = events(data)
    assert evs[-1] == "[DONE]"
    choices = [e["choices"][0] for e in evs[:-1]]
    return "".join(c["text"] for c in choices), choices[-1]["finish_reason"]


def decoded(ids, max_tokens: int = 8, stop=()) -> str:
    """The text of `ids` the slow way: their bytes up to EOS or `max_tokens`, decoded at once, cut at the first stop."""
    data = b""
    for i in ids[:max_tokens]:
        if i == EOS:
            break
        data += VOCAB.render(i)
    text = data.decode("utf-8", "replace")
    hits = [text.find(s) for s in stop if s in text]
    return text[:min(hits)] if hits else text


# --- the schedule ----------------------------------------------------------------------------------------------------


def test_honest_traffic_is_released_exactly_on_schedule_and_audited():
    """Four requests, streamed and not, one finishing early at EOS, one at a stop string: every frame the client gets is
    B bytes in the bucket the layout puts it, the client's text is the ids' text, and the audit of the records accepts
    every window and reads back the same ids and text."""
    ids = {0: [A, B_, SPACE, A, B_, A, B_, A], 1: [A, EOS, A], 2: [A, ST, OP, B_], 3: [E_ACUTE, C3, A9, A, B_, EOS]}
    arrivals = {1: [http(body())], 2: [http(body(stream=False, max_tokens=4))],
                3: [http(body(stop=["STOP"]))], 4: [http(body(stream=False, max_tokens=6))]}
    warden, clients, server = run(ids, arrivals)
    admitted = {e["ordinal"]: e["admitted"] for e in warden.log}
    # two lanes: 1's ends at 2 + 4 and 0's at 1 + 6, so the queued 2 and 3 take the lanes at 7 and 8
    assert admitted == {0: 1, 1: 2, 2: 7, 3: 8}
    for q, client in enumerate(clients):
        request = server.requests[q]
        layout = SCHEDULE.layout(request.stream, request.max_tokens)
        assert client.frames() == [(admitted[q] + b, FB) for b in layout]
        text, finish, _ = render(VOCAB, request, ids[q])
        assert text == decoded(ids[q], request.max_tokens, request.stop)
        if request.stream:
            assert streamed_text(client.data) == (text, finish)
        else:
            doc = json.loads(chunked(client.data)[1])
            assert (doc["choices"][0]["text"], doc["choices"][0]["finish_reason"]) == (text, finish)
            assert doc["usage"]["completion_tokens"] == render(VOCAB, request, ids[q])[2]
    assert [e["finish"] for e in sorted(warden.log, key=lambda e: e["ordinal"])] == ["length", "stop", "stop", "stop"]
    assert max(warden.lateness) == 0
    ingress = [r for r in warden.records if r.link == "requests"]
    egress = [r for r in warden.records if r.link == "responses"]
    verdict = audit(SCHEDULE, ingress, egress)
    assert verdict["accepted"], verdict["windows"]
    assert [(r["ordinal"], r["admitted"], r["text"]) for r in verdict["requests"]] == \
        [(q, admitted[q], render(VOCAB, server.requests[q], ids[q])[0]) for q in range(4)]
    assert verdict["frames"] == sum(len(c.frames()) for c in clients)


def test_the_egress_record_is_a_frame_per_frame_sent_of_the_kind_its_role_allows():
    ids = {0: [A, EOS]}
    warden, (client,), _ = run(ids, {0: [http(body(max_tokens=4))]}, windows=1)
    rows = [r for r in warden.records if r.link == "responses"][0].grid
    got = [read_record_frame(f) for row in rows for f in row if f is not None]
    assert [(kind, k) for kind, _, k, _, _ in got] == \
        [(HEAD, 0), (TOKEN, 1), (TOKEN, 2), (DUMMY, 3), (DUMMY, 4), (FINAL, 5)]
    assert [token for kind, _, _, token, _ in got if kind == TOKEN] == [A, EOS]
    assert got[-1][4] == CODES["stop"]
    assert len(client.frames()) == len(got)


# --- fail closed -----------------------------------------------------------------------------------------------------


def test_a_late_token_fails_its_request_closed_and_the_record_says_late():
    """Token 3 isn't ready at its slot's tick: the client gets a marked error there, dummies after it and the stream's
    end at its scheduled bucket; the record's frame there is FAIL, late, and the audit refuses the window. The token
    arriving later changes nothing."""

    async def go():
        gate = asyncio.Event()
        server = Server({0: [A, B_, A, B_, A, B_, A, B_]}, {(0, 3): gate})
        warden, clients = await drive(server, {1: [http(body())]})
        gate.set()
        return warden, clients

    warden, (client,) = asyncio.run(go())
    (entry,) = warden.log
    assert (entry["failed"], entry["failed_at"], entry["finish"]) == ("late", 3, None)
    evs = events(client.data)
    assert [e["choices"][0]["text"] for e in evs[:3]] == ["a", "b", "a"]
    assert evs[3]["error"]["type"] == "WardenLate" and evs[3]["error"]["code"] == 504
    assert evs[4:] == ["[DONE]"]
    assert client.frames() == [(1 + b, FB) for b in SCHEDULE.layout(True, 8)]
    egress = [r for r in warden.records if r.link == "responses"]
    frames = [read_record_frame(f) for r in egress for row in r.grid for f in row if f is not None]
    assert [(kind, code) for kind, _, k, _, code in frames if k == 4] == [(FAIL, CODES["late"])]
    assert [kind for kind, *_ in frames[5:-1]] == [DUMMY] * 4
    verdict = audit(SCHEDULE, [r for r in warden.records if r.link == "requests"], egress)
    assert not verdict["accepted"]
    assert any("request 0 failed late" in p for w in verdict["windows"] for p in w["problems"])
    assert verdict["requests"][0]["failed"] == "late"


def test_an_unstreamed_late_request_gets_an_error_body_of_the_same_frames():
    async def go():
        server = Server({0: [A, B_]}, {(0, 1): asyncio.Event()})
        return await drive(server, {0: [http(body(stream=False, max_tokens=4))]}, windows=1)

    warden, (client,) = asyncio.run(go())
    doc = json.loads(chunked(client.data)[1])
    assert doc["error"]["type"] == "WardenLate" and "token 1" in doc["error"]["message"]
    assert client.frames() == [(b, FB) for b in SCHEDULE.layout(False, 4)]


def test_a_late_tick_faults_both_grids_window_and_the_audit_refuses_it():
    ids = {0: [A, B_, EOS]}
    warden, _, _ = run(ids, {1: [http(body())]}, late={5: 2 * 0.025})
    statuses = {(r.link, r.window): r.status.reason for r in warden.records}
    assert statuses == {("requests", 0): Reason.MONITOR_UNAVAILABLE, ("responses", 0): Reason.MONITOR_UNAVAILABLE,
                        ("requests", 1): None, ("responses", 1): None}
    verdict = audit(SCHEDULE, [r for r in warden.records if r.link == "requests"],
                    [r for r in warden.records if r.link == "responses"])
    assert [w["accepted"] for w in verdict["windows"]] == [False, True]


# --- admission -------------------------------------------------------------------------------------------------------


def test_over_capacity_queues_fifo_then_refuses_with_429():
    """Two lanes and a queue of two: two requests take the lanes, and of three more the next bucket two wait and are
    admitted in order as lanes free, and the third is refused with 429 and takes no ordinal; nothing admitted is
    dropped or reordered."""
    ids = {q: [A, B_, A, EOS] for q in range(4)}
    warden, clients, server = run(ids, {1: [http(body(max_tokens=4)) for _ in range(2)],
                                        2: [http(body(max_tokens=4)) for _ in range(3)]})
    assert warden.refused == [(429, None)]
    refused = clients[4].data
    assert refused.startswith(b"HTTP/1.1 429 Too Many Requests")
    assert json.loads(refused.partition(b"\r\n\r\n")[2])["error"]["type"] == "rate_limit_exceeded"
    log = sorted(warden.log, key=lambda e: e["ordinal"])
    end = 1 + SCHEDULE.layout(True, 4)[-1]
    assert [(e["ordinal"], e["admitted"]) for e in log] == [(0, 1), (1, 1), (2, end + 1), (3, end + 1)]
    assert [r.ordinal for r in server.requests] == [0, 1, 2, 3]
    for client in clients[:4]:
        assert streamed_text(client.data) == ("aba", "stop")


def test_admission_holds_every_slot_through_max_tokens():
    """A request finishing at its first token still holds its lane through the slots of its max_tokens."""
    warden, _, _ = run({0: [EOS], 1: [EOS], 2: [EOS]}, {0: [http(body()) for _ in range(2)], 1: [http(body())]})
    assert {e["ordinal"]: e["admitted"] for e in warden.log}[2] == 1 + SCHEDULE.layout(True, 8)[-1]


# --- nothing of the server reaches a client --------------------------------------------------------------------------


def test_extra_tokens_and_the_servers_finish_never_reach_the_client():
    """A server sending tokens past max_tokens, and finish words of its own, gives the client exactly the bytes an
    honest server's ids give: the warden reads ids only and stops at max_tokens."""
    honest = {0: [A, B_, A, B_]}
    noisy = {0: [Token(A, 0, "stop"), Token(B_, 1, "abort"), Token(A, 2, None), Token(B_, 3, "length"),
                 Token(STOP, 4, None), Token(LONG, 5, "stop")]}
    a = run(honest, {1: [http(body(max_tokens=4))]})
    b = run(noisy, {1: [http(body(max_tokens=4))]})
    assert a[1][0].data == b[1][0].data
    assert b[0].log[0]["extra"] == 1 and b[0].log[0]["finish"] == "length"


def test_a_token_port_line_with_anything_but_id_position_and_finish_is_malformed():
    assert parse_token(b'{"id": 5, "position": 0, "finish": null}\n') == Token(5, 0, None)
    for line in [b'{"id": 5, "position": 0, "finish": null, "text": "SECRET"}\n',
                 b'{"id": 5, "position": 0}\n', b'{"id": "5", "position": 0, "finish": null}\n',
                 b'{"id": 5, "position": 0, "finish": "SECRET"}\n', b'{"id": 5, "position": 0, "finish": null}',
                 b'{"id": 5, "position": 0, "finish": null}' + b" " * 300 + b"\n", b"SECRET\n",
                 b'{"id": true, "position": 0, "finish": null}\n']:
        with pytest.raises(Malformed):
            parse_token(line)


class PortWriter:
    def __init__(self) -> None:
        self.data, self.closed = bytearray(), False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def port(lines: list[bytes], limit: int = 256) -> tuple[TokenPort, PortWriter]:
    writer = PortWriter()

    async def connect():
        reader = asyncio.StreamReader(limit=limit)
        reader.feed_data(b"".join(lines))
        reader.feed_eof()
        return reader, writer

    return TokenPort("server", 0, connect=connect), writer


def test_the_token_port_sends_the_canonical_request_record_and_reads_token_lines():
    request = Request(7, (BOS, A), 0.7, 0.9, 42, 3, ("STOP",), True)
    tokens, writer = port([b'{"id": 2, "position": 0, "finish": null}\n',
                           b'{"id": 1, "position": 1, "finish": "stop"}\n'])

    async def go():
        return [t async for t in tokens.stream(request)]

    assert asyncio.run(go()) == [Token(2, 0, None), Token(1, 1, "stop")]
    assert json.loads(writer.data) == {"ordinal": 7, "prompt": [BOS, A], "temperature": 0.7, "top_p": 0.9,
                                       "seed": 42, "max_tokens": 3, "stop": ["STOP"]}
    assert writer.closed


def test_a_server_sending_bytes_beside_its_tokens_fails_the_request_and_none_reach_the_client():
    """The token port's server adds a field to its second line, then sends an over-long line on another request: each
    request fails closed as malformed at its first missing token, and no server byte is in what the clients got."""
    lines = {0: [b'{"id": 2, "position": 0, "finish": null}\n',
                 b'{"id": 3, "position": 1, "finish": null, "text": "SECRET"}\n'],
             1: [b'{"id": 2, "position": 0, "finish": null}\n', b'{"id": 3' + b" " * 400 + b"SECRET}\n"]}
    ports = {q: port(lines[q])[0] for q in lines}

    class Upstream(Server):
        async def stream(self, request):
            try:
                async for token in ports[request.ordinal].stream(request):
                    yield token
            finally:
                self._mark(request.ordinal, "end")

    async def go():
        return await drive(Upstream({}), {1: [http(body()), http(body(stream=False))]})

    warden, clients = asyncio.run(go())
    assert [(e["failed"], e["failed_at"], e["received"]) for e in sorted(warden.log, key=lambda e: e["ordinal"])] == \
        [("malformed", 1, 1), ("malformed", 1, 1)]
    for client in clients:
        assert b"SECRET" not in client.data
    assert events(clients[0].data)[1]["error"]["type"] == "WardenUpstreamMalformed"


def test_out_of_order_or_out_of_vocabulary_ids_are_malformed():
    for bad in ([Token(A, 0), Token(B_, 2)], [Token(A, 0), Token(len(PIECES), 1)]):
        warden, (client,), _ = run({0: bad}, {0: [http(body(max_tokens=4))]}, windows=1)
        assert warden.log[0]["failed"] == "malformed" and warden.log[0]["received"] == 1


# --- rendering -------------------------------------------------------------------------------------------------------


def test_rendered_text_is_the_ids_text():
    """UTF-8 split across tokens, an invalid byte, a byte left open at the end, specials, escapes and stop strings: the
    renderer's text, one slot at a time, is the ids' bytes decoded at once and cut at the first stop."""
    cases = [([E_ACUTE, C3, A9, A], ()), ([C3, A, A9, FF], ()), ([A, C3], ()), ([BOS, A, BOS, B_], ()),
             ([QUOTE, BACKSLASH, NL, CTRL], ()), ([A, ST, OP, B_], ("STOP",)), ([A, STOP], ("TO",)),
             ([A, B_, A, B_], ("ba", "b")), ([A, EOS, B_], ()), ([LONG, LONG], ("xx",))]
    for ids, stop in cases:
        request = Request(0, (BOS,), 1.0, 1.0, None, 8, stop, True)
        assert render(VOCAB, request, ids + [EOS])[0] == decoded(ids + [EOS], 8, stop), (ids, stop)
    rng = random.Random(1)
    for _ in range(500):
        ids = [rng.randrange(len(PIECES)) for _ in range(rng.randrange(1, 9))] + [EOS]
        stop = tuple(rng.sample(["a", "ab", "STOP", "\n", "é", "xb", "\ufffd"], rng.randrange(0, 2)))
        request = Request(0, (BOS,), 1.0, 1.0, None, 8, stop, True)
        text, finish, used = render(VOCAB, request, ids)
        assert text == decoded(ids, 8, stop), (ids, stop)
        full = decoded(ids, 8)
        assert finish == ("stop" if EOS in ids[:used] or text != full or any(s in full for s in stop) else "length")


def test_the_renderer_holds_back_what_could_begin_a_stop_and_releases_the_rest_on_close():
    r = Renderer(VOCAB, Request(0, (BOS,), 1.0, 1.0, None, 8, ("STOP",), True))
    assert [r.feed(i) for i in (A, B_, ST)] == ["", "", "a"]
    assert r.close() == "bST" and r.text == "abST"
    r = Renderer(VOCAB, Request(0, (BOS,), 1.0, 1.0, None, 8, ("STOP",), True))
    assert [r.feed(i) for i in (A, ST, OP)] == ["", "", "a"] and r.finish == "stop" and r.close() == ""


def test_multi_stop_strings_cut_at_the_earliest_complete_one():
    """Stops are checked after each token on the text so far, and the earliest one complete there wins, even when a
    longer stop that began earlier would have completed at a later token."""
    request = Request(0, (BOS,), 1.0, 1.0, None, 8, ("aSTb", "ST"), True)
    assert render(VOCAB, request, [A, ST, B_]) == ("a", "stop", 2)


# --- size and timing carry nothing -----------------------------------------------------------------------------------


def test_dummy_frames_are_indistinguishable_in_size_and_the_wire_is_a_function_of_admission_and_max_tokens():
    """The same arrivals, max_tokens and stream flags with different contents: full length, EOS at once, a stop string,
    the most text a slot carries, a late token. Every client gets the same frames, B bytes each, in the same buckets."""
    contents = [{0: [A] * 8, 1: [A] * 4}, {0: [EOS], 1: [EOS]}, {0: [A, ST, OP], 1: [STOP]},
                {0: [LONG] * 8, 1: [CTRL, LONG, LONG, LONG]}, ({0: [A] * 8, 1: [A] * 4}, {(0, 2): asyncio.Event()})]
    arrivals = {1: [http(body(stop=["STOP"]))], 3: [http(body(stream=False, max_tokens=4))]}
    wires = []
    for ids in contents:
        warden, clients, _ = run(ids, arrivals)
        wires.append([c.frames() for c in clients])
    assert all(w == wires[0] for w in wires)
    assert SCHEDULE.wire.dummy(True) != SCHEDULE.wire.dummy(False)
    assert {len(SCHEDULE.wire.dummy(s)) for s in (True, False)} == {len(SCHEDULE.wire.token(0, "a")), FB}


def test_every_slot_fits_the_most_text_a_token_renders():
    """A slot that releases a stop's hold-back, an open UTF-8 sequence's replacement and the vocabulary's widest token
    still fits one frame, and a frame too small for that is refused when the schedule is made."""
    worst = Request(0, (BOS,), 1.0, 1.0, None, 64, ("\x01" * 32,), True)
    r = Renderer(VOCAB, worst)
    texts = [r.feed(i) for i in [CTRL] * 31 + [C3, LONG, CTRL, C3]] + [r.close()]
    assert texts[32] == "\x01" * 31 + "\ufffd" + "x" * 9
    rng = random.Random(2)
    for _ in range(200):
        r = Renderer(VOCAB, Request(0, (BOS,), 1.0, 1.0, None, 64, (rng.choice(["\x01" * 32, "\"" * 32, "ab"]),), True))
        while r.finish is None:
            texts.append(r.feed(rng.randrange(2, len(PIECES))))
        texts.append(r.close())
    for text in texts:
        assert len(json.dumps(text, ensure_ascii=False)) - 2 <= SCHEDULE.frame_text
        assert len(SCHEDULE.wire.token(10 ** 9, text)) == FB
        assert len(SCHEDULE.wire.final(10 ** 9, text, "length")) == FB
    with pytest.raises(ValueError, match="raise frame_bytes"):
        Schedule(WardenParams(frame_bytes=256), VOCAB, MODEL)


# --- ingress ---------------------------------------------------------------------------------------------------------


def test_a_canonical_request_encodes_and_decodes_exactly():
    for request in [Request(0, (BOS, A), 1.0, 1.0, None, 8, (), False),
                    Request(2 ** 32 - 1, tuple(range(8)), 0.0, 0.5, 2 ** 63 - 1, 1, ("é", "x" * 32, "a", "b"), True)]:
        data = request.encode()
        assert Request.decode(data) == request
        with pytest.raises(ValueError):
            Request.decode(data + b"\0")
        with pytest.raises(ValueError):
            Request.decode(data[:-1])


def test_ingress_takes_only_the_canonical_fields():
    take = lambda doc, **kw: canonical(doc, 3, PARAMS, VOCAB, model=MODEL, **kw)
    assert take({"prompt": [BOS, A], "max_tokens": 8}) == Request(3, (BOS, A), 1.0, 1.0, None, 8, (), False)
    assert take(body(temperature=0, top_p=0.5, seed=9, stop="\n", user="u", n=1, best_of=None, top_k=-1,
                     min_p=0, presence_penalty=0, frequency_penalty=0.0, repetition_penalty=1, logit_bias={},
                     logprobs=None, echo=False, suffix=None)) == \
        Request(3, (BOS, A), 0.0, 0.5, 9, 8, ("\n",), True)
    assert take({"prompt": "ab", "max_tokens": 8}, encode=lambda s: [BOS, A, B_]).prompt == (BOS, A, B_)
    assert take(body(max_tokens=8, prompt=[BOS] * 8)).max_tokens == 8
    with pytest.raises(Refused, match="max_tokens"):
        take({"prompt": [BOS]})  # OpenAI's default of 16 is past this warden's cap of 8
    refusals = {"top_k": 5, "min_p": 0.1, "presence_penalty": 0.5, "frequency_penalty": 1, "repetition_penalty": 1.1,
                "logit_bias": {"2": 1}, "n": 2, "best_of": 2, "logprobs": 1, "echo": True, "suffix": "x",
                "unknown": 1}
    for name, value in refusals.items():
        with pytest.raises(Refused) as e:
            take(body() | {name: value})
        assert e.value.param == name and e.value.status == 400
    assert set(NEUTRAL) <= set(refusals)
    bad = [({"model": "other"}, 404, "model"), ({"prompt": "ab"}, 400, "prompt"), ({"prompt": []}, 400, "prompt"),
           ({"prompt": [len(PIECES)]}, 400, "prompt"), ({"prompt": [BOS] * 9}, 400, "prompt"),
           ({"max_tokens": 9}, 400, "max_tokens"), ({"max_tokens": 0}, 400, "max_tokens"),
           ({"max_tokens": True}, 400, "max_tokens"), ({"temperature": 2.5}, 400, "temperature"),
           ({"top_p": 0}, 400, "top_p"), ({"seed": -1}, 400, "seed"), ({"seed": 2 ** 63}, 400, "seed"),
           ({"stop": ["a"] * 5}, 400, "stop"), ({"stop": ["x" * 33]}, 400, "stop"), ({"stop": [""]}, 400, "stop"),
           ({"stream": 1}, 400, "stream"), ({"temperature": float("nan")}, 400, "temperature"),
           ({"n": True}, 400, "n")]
    for change, status, param in bad:
        with pytest.raises(Refused) as e:
            take(body() | change)
        assert (e.value.status, e.value.param) == (status, param), change
    params = dataclasses.replace(PARAMS, context=12)
    with pytest.raises(Refused, match="context"):
        canonical(body(max_tokens=8, prompt=[BOS] * 5), 0, params, VOCAB, model=MODEL)


def test_the_http_front_refuses_what_it_does_not_serve():
    async def go():
        clock = VirtualClock(0.0)
        warden = Warden(SCHEDULE, Server({}), origin=0.0, windows=1, clock=clock, sleep=clock.sleep)
        clients = [Client(warden, http(body(top_k=5))), Client(warden, http("{", )),
                   Client(warden, http(body(), path="/v1/chat/completions")),
                   Client(warden, http(body(), method="GET")), Client(warden, b"GET /v1/models HTTP/1.1\r\n\r\n"),
                   Client(warden, b"POST /v1/completions HTTP/1.1\r\ntransfer-encoding: chunked\r\n\r\n")]
        await asyncio.gather(*(c.task for c in clients))
        return warden, clients

    warden, clients = asyncio.run(go())
    statuses = [int(c.data.split(b" ")[1]) for c in clients]
    assert statuses == [400, 400, 404, 405, 200, 400]
    assert json.loads(clients[4].data.partition(b"\r\n\r\n")[2])["data"][0]["id"] == MODEL
    assert warden.ordinals == 0 and not warden.queue


def test_the_vocabulary_reads_a_byte_level_bpe_tokenizer_json():
    doc = {"model": {"type": "BPE", "vocab": {"a": 0, "Ġb": 1, "Ċ": 2, "Ã©": 3, "Ã": 4}},
           "decoder": {"type": "ByteLevel"},
           "added_tokens": [{"id": 5, "content": "<|begin_of_text|>", "special": True},
                            {"id": 6, "content": "<|end_of_text|>", "special": True}]}
    v = Vocabulary.from_tokenizer_json(doc, eos=[6])
    assert v.tokens == (b"a", b" b", b"\n", "é".encode(), b"\xc3", b"<|begin_of_text|>", b"<|end_of_text|>")
    assert v.special == {5, 6} and v.eos == {6}
    assert v.render(5) == b"" and v.cost == 6
    with pytest.raises(ValueError):
        Vocabulary.from_tokenizer_json(doc | {"decoder": {"type": "Metaspace"}}, eos=[6])


def test_the_parameters_digest_binds_the_parameters_the_model_and_the_vocabulary():
    other = Vocabulary(VOCAB.tokens[:-1] + (b"Q",), VOCAB.special, VOCAB.eos)
    digests = {SCHEDULE.digest, Schedule(dataclasses.replace(PARAMS, delay=3), VOCAB, MODEL).digest,
               Schedule(PARAMS, VOCAB, "other").digest, Schedule(PARAMS, other, MODEL).digest}
    assert len(digests) == 4 and all(len(d) == 64 for d in digests)

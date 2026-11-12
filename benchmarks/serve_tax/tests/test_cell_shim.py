"""The benchmark client's shim to the network certifier (`cell_shim`), against the certifier itself.

The network certifier (`warden.Warden`) runs tick by tick on a `VirtualClock`, in its id egress form with a response
cap, in front of an in-memory server that answers in the id form; the shim's HTTP clients and its connections to the
certifier are in memory too. Nothing waits on the host's clock: the test moves the clock and waits for each side to
have read what the other wrote."""
from __future__ import annotations

import asyncio
import dataclasses
import json

import cell_shim as C
import pytest
from verity.protocols.certifier.verifier import io_tie as IT
from verity.protocols.certifier.verifier.active import VirtualClock, id_request
from verity.protocols.certifier.verifier.warden import Warden, WardenParams, audit

PARAMS = WardenParams(cell_bytes=256, slots=1, bucket_ms=100, delay=4, lanes=2, queue=2, window=40, ingress_cells=8,
                      ingress_buckets=6, egress_cells=12, egress="id", max_tokens_cap=8)
TAU = PARAMS.bucket_ms / 1000
#: a body as vLLM's client sends one (its `async_request_openai_completions`, greedy and to its length)
BODY = {"model": "glm", "prompt": "four words of text", "repetition_penalty": 1.0, "max_tokens": 3, "logprobs": None,
        "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True, "temperature": 0.0}


def encode(text: str) -> list[int]:
    """A stand-in tokenizer: a token a word, its id the word's length."""
    return [len(w) for w in text.split()]


def post(body: dict | bytes, request_id: str | None = "chat@96-r1-0", line: bytes = b"POST /v1/completions") -> bytes:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return (line + b" HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            + (b"X-Request-Id: " + request_id.encode() + b"\r\n" if request_id else b"")
            + b"Content-Length: %d\r\n\r\n" % len(raw) + raw)


async def turn() -> None:
    """One turn of the event loop: every task that is ready runs before this one goes on."""
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    loop.call_soon(done.set_result, None)
    await done


async def until(predicate) -> None:
    while not predicate():
        await turn()


class Pipe:
    """One direction of an in-memory connection: what is written is its reader's bytes, and closing it their end."""

    def __init__(self) -> None:
        self.reader, self.closed, self.transport = asyncio.StreamReader(), False, self

    def write(self, data: bytes) -> None:
        if not self.closed:
            self.reader.feed_data(bytes(data))

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.reader.feed_eof()

    def is_closing(self) -> bool:
        return self.closed

    def get_write_buffer_size(self) -> int:
        return 0


class Sink:
    """An HTTP client's end: every write of the shim's with the bucket it was made in."""

    def __init__(self, warden: Warden) -> None:
        self.warden, self.writes, self.closed = warden, [], False

    def write(self, data: bytes) -> None:
        assert not self.closed
        self.writes.append((self.warden.ticks - 1, bytes(data)))

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def answer(self) -> tuple[int, list[tuple[int, bytes]] | dict, bool]:
        """(status, body, whole): a stream's events, each with its bucket, and whether its last chunk came; any other
        answer's JSON."""
        head, _, rest = self.writes[0][1].partition(b"\r\n\r\n")
        status = int(head.split()[1])
        if b"transfer-encoding: chunked" not in head:
            assert len(self.writes) == 1 and b"content-length: %d" % len(rest) in head
            return status, json.loads(rest), True
        events, whole = [], False
        for bucket, data in [(self.writes[0][0], rest), *self.writes[1:]]:
            while data:
                size, _, data = data.partition(b"\r\n")
                chunk, data = data[:int(size, 16)], data[int(size, 16) + 2:]
                assert not whole
                if chunk:
                    assert chunk.startswith(b"data: ") and chunk.endswith(b"\n\n")
                    events.append((bucket, chunk[6:-2]))
                whole = not chunk
        return status, events, whole


def as_vllm_reads(events: list[tuple[int, bytes]]) -> dict:
    """A stream as vLLM's client reads one: a message with choices is a chunk, whatever its text, and the first is the
    first token; the usage message gives the output's and the prompt's token counts."""
    chunks, out = [], {"output_tokens": None, "prompt_len": None}
    for bucket, event in events:
        if event == b"[DONE]":
            continue
        data = json.loads(event)
        if data.get("choices"):
            chunks.append(bucket)
        elif usage := data.get("usage"):
            out |= {"output_tokens": usage.get("completion_tokens"), "prompt_len": usage.get("prompt_tokens")}
    return out | {"success": bool(chunks), "chunks": chunks}


class IdServer:
    """An in-memory server behind the certifier, in the id form: to each request, once it has it whole, ``answer``'s
    ids for its body, one id event each, and with ``whole`` the trailer and the stream's end."""

    def __init__(self, payload: int, answer) -> None:
        self.payload, self.answer, self.conns = payload, answer, []

    async def connect(self):
        conn = IdConn(self)
        self.conns.append(conn)
        return conn.reader, conn


class IdConn(Pipe):
    def __init__(self, server: IdServer) -> None:
        super().__init__()
        self.server, self.received, self.fed, self.eof = server, b"", 0, False

    def write(self, data: bytes) -> None:
        self.received += data
        ids, whole = self.server.answer(json.loads(self.received.partition(b"\r\n\r\n")[2]))
        out = b"".join(IT.id_event(x, self.server.payload) for x in ids) + (IT.TRAILER if whole else b"")
        self.fed += len(out)
        self.reader.feed_data(out)
        if whole:
            self.eof = True
            self.reader.feed_eof()

    def close(self) -> None:
        self.closed = True


def counting(body: dict) -> tuple[list[int], bool]:
    """A whole answer: `max_tokens` ids, counting up from the prompt's first."""
    return [body["prompt"][0] + k for k in range(body["max_tokens"])], True


def settled(warden: Warden) -> bool:
    """Whether the certifier has read everything its clients and its server have written."""
    for c in warden.active:
        if c.problem is None and not c.ingress_ended and c.reader._buffer:
            return False
        if c.server is None and c.upstream is None:
            return False
        if c.server is not None and (c.bytes_out != c.server[1].fed or c.egress_ended != c.server[1].eof):
            return False
    return True


async def drive(arrivals: dict[int, list[bytes]], answer=counting, params: WardenParams = PARAMS):
    """Send ``arrivals[t]`` (each an HTTP request's bytes) to the shim just before bucket t's tick, one after the
    other, for one window; returns the certifier, the server, each request's `Sink` and the shim's records."""
    clock = VirtualClock(-TAU / 2)
    server = IdServer(params.payload, answer)
    warden = Warden(params, server.connect, origin=0.0, windows=1, clock=clock, sleep=clock.sleep)
    log, pipes, sinks, tasks = [], [], [], []

    async def connect():
        up, down = Pipe(), Pipe()
        pipes.append(down)
        tasks.append(asyncio.create_task(warden.handle(up.reader, down)))
        return down.reader, up

    shim = C.HttpShim(connect, params.cell_bytes, encode, log.append, clock=clock, origin=0.0)
    running = asyncio.create_task(warden.run())
    for t in range(params.window):
        for raw in arrivals.get(t, ()):
            reader, opened = asyncio.StreamReader(), len(pipes)
            reader.feed_data(raw)
            reader.feed_eof()
            sinks.append(Sink(warden))
            task = asyncio.create_task(shim.handle(reader, sinks[-1]))
            tasks.append(task)
            await until(lambda: task.done() or len(pipes) > opened)
            await until(lambda: warden.ordinals + warden.refused == len(pipes))
        await until(lambda: settled(warden))
        clock.advance(t * TAU)
        await warden.until(lambda w, t=t: w.ticks > t)
        await until(lambda: all(p.closed or len(p.reader._buffer) < params.cell_bytes for p in pipes))
    await running
    await asyncio.gather(*tasks)
    assert all(s.closed for s in sinks)
    return warden, server, sinks, log


def records(warden: Warden):
    return tuple([r for r in warden.records if r.link == link] for link in ("requests", "responses", "fails"))


def test_a_completion_goes_to_the_certifier_as_its_id_request_and_comes_back_a_stream_on_the_cells_buckets():
    """vLLM's client's completions, a text prompt and a prompt of token ids, each reach the server behind the
    certifier as the canonical id request of their fields; each output id comes back as one chunk, written in the
    bucket the certifier released its cell in (the delay D after the admission, then one a bucket), then the usage and
    ``[DONE]`` at the ``END``. vLLM's client reads each as a served request with its token counts. The shim's record of
    each connection is the cells the certifier certified: the audit of the certifier's records reads back the same
    request and response digests, and no prompt or id is in the record."""
    ids = BODY | {"prompt": [9, 8, 7, 6, 5], "max_tokens": 2, "seed": 4}
    warden, server, sinks, log = asyncio.run(drive({1: [post(BODY)], 2: [post(ids, request_id=None)]}))
    sent = [id_request(model="glm", user="chat@96-r1-0", prompt=[4, 5, 2, 4], max_tokens=3, temperature=0.0, top_p=1.0,
                       seed=0, frame_bytes=PARAMS.cell_bytes, offset=0),
            id_request(model="glm", user="client", prompt=[9, 8, 7, 6, 5], max_tokens=2, temperature=0.0, top_p=1.0,
                       seed=4, frame_bytes=PARAMS.cell_bytes, offset=0)]
    for conn, request in zip(server.conns, sent):
        assert conn.received.startswith(b"POST /v1/completions HTTP/1.1\r\n")
        assert conn.received.partition(b"\r\n\r\n")[2] == request.partition(b"\r\n\r\n")[2]
        assert "ignore_eos" not in json.loads(conn.received.partition(b"\r\n\r\n")[2])
    D = PARAMS.delay
    for sink, s, prompt, out in zip(sinks, (1, 2), (4, 5), ([4, 5, 6], [9, 10])):
        status, events, whole = sink.answer()
        assert status == 200 and whole
        chunks = [(b, json.loads(e)) for b, e in events[:-2]]
        assert [(b, c["choices"]) for b, c in chunks] == [
            (s + D + k, [{"index": 0, "text": "", "token_ids": [x], "finish_reason": None}]) for k, x in enumerate(out)]
        end, usage = events[-2][0], json.loads(events[-2][1])
        assert end == s + D + len(out) and usage["choices"] == [] and events[-1] == (end, b"[DONE]")
        assert usage["usage"] == {"prompt_tokens": prompt, "completion_tokens": len(out),
                                  "total_tokens": prompt + len(out)}
        assert as_vllm_reads(events) == {"success": True, "chunks": [s + D + k for k in range(len(out))],
                                         "output_tokens": len(out), "prompt_len": prompt}
    verdict = audit(PARAMS, *records(warden))
    assert verdict["accepted"], verdict["windows"]
    certified = {c["ordinal"]: c for c in verdict["connections"]}
    for record, s, request, prompt, out in zip(log, (1, 2), sent, (4, 5), ([4, 5, 6], [9, 10])):
        assert {"index", "at", "sent_ms", "request_bytes", "request_sha256", "ordinal", "outcome", "kinds",
                "arrivals_ms", "data_bytes", "data_sha256", "response"} <= set(record)       # warden_serve's keys
        assert (record["outcome"], record["status"], record["kinds"]) == ("end", 200, "p" * D + "d" * len(out) + "e")
        assert record["arrivals_ms"] == [(s + k) * PARAMS.bucket_ms for k in range(D + len(out) + 1)]
        assert record["response"]["ids"] == record["response"]["completion_tokens"] == len(out)
        assert (record["prompt_tokens"], record["max_tokens"], record["request_bytes"]) == (prompt, len(out),
                                                                                            len(request))
        got = certified[record["ordinal"]]
        assert (got["ingress_sha256"], got["egress_sha256"]) == (record["request_sha256"], record["data_sha256"])
        assert not {"prompt", "ids", "token_ids", "text"} & set(record)
    assert C.client_doc(log) == {"connections": log}
    assert [x["id"] for x in log] == ["chat@96-r1-0", None]


@pytest.mark.parametrize("change, why", [
    ({"stream": False}, "stream is true"),
    ({"temperature": 0.7}, "temperature is 0"),
    ({"temperature": None}, "temperature is 0"),
    ({"ignore_eos": False}, "ignore_eos is true"),
    ({"n": 2}, "has no n"),
    ({"stop": ["."]}, "has no stop"),
    ({"repetition_penalty": 1.1}, "has no repetition_penalty"),
    ({"logprobs": 5}, "has no logprobs"),
    ({"logit_bias": {"5": 1.0}, "top_k": 4}, "has no logit_bias, top_k"),
    ({"prompt": ["two", "prompts"]}, "one text or one list of token ids"),
    ({"prompt": ""}, "one text or one list of token ids"),
    ({"max_tokens": None}, "not an id request"),
    ({"model": "m" * 70}, "not an id request"),
])
def test_a_body_the_id_request_cannot_carry_is_refused_and_nothing_is_sent(change: dict, why: str):
    """A completion whose answer would differ from the id request's (a field the id request has no place for, a
    sampled or unstreamed one, one that stops at its end of sequence) gets a 400 that names why, and the certifier
    sees no connection of it; so does a body that isn't JSON, and any other path a 404."""
    warden, server, (sink, other, lost), log = asyncio.run(drive({1: [
        post(BODY | change), post(b"{not json"), post(BODY, line=b"GET /v1/models")]}))
    status, body, _ = sink.answer()
    assert status == 400 and why in body["error"]["message"]
    assert other.answer()[0] == 400 and lost.answer()[0] == 404
    assert (warden.ordinals, warden.refused, server.conns) == (0, 0, [])
    assert [(x["status"], "unsent" in x) for x in log] == [(400, True)] * 2 and C.client_doc(log) == {"connections": []}


def test_a_request_never_read_whole_is_recorded_and_nothing_is_sent():
    """A client that closes inside its request's head or its body, and a head with no end, each leave a line that says
    what happened: the certifier sees no connection of them, and they are none of the offline check's."""
    whole = post(BODY)
    head = whole[:whole.index(b"\r\n\r\n") + 4]
    warden, server, (in_head, in_body, endless), log = asyncio.run(drive({1: [
        whole[:20], head + whole[len(head):len(head) + 7], b"POST /v1/completions HTTP/1.1\r\nX-Junk: " + b"j" * 70000]}))
    assert in_head.writes == [] and in_body.writes == []
    status, body, _ = endless.answer()
    assert status == 400 and "longer than this listener reads" in body["error"]["message"]
    assert (warden.ordinals, warden.refused, server.conns) == (0, 0, [])
    assert [(x["index"], x["status"]) for x in log] == [(None, None), (None, None), (None, 400)]
    assert log[0]["unsent"] == "the client closed 20 bytes into its request's head"
    assert log[1]["unsent"] == f"the client closed 7 bytes into its request's body of {len(whole) - len(head)}"
    assert log[2]["unsent"].startswith("a request's head had no end in its first ")
    assert C.client_doc(log) == {"connections": []}


def test_a_connection_the_certifier_refuses_or_fails_is_an_error_or_a_broken_stream_never_a_short_answer():
    """With one lane and one place in the queue: a request that arrives with the queue taken is refused by the
    certifier (503), and one that waits for the lane is served once it is free, its wait in its first chunk's time. A
    request past the response cap, and one whose server sends nothing by its first cell's bucket, fail before any id
    (502). One whose server stops after two ids has sent them: its stream ends without its usage, ``[DONE]`` or last
    chunk, which no HTTP client reads as a whole response. The record says which each was."""
    def answer(body: dict) -> tuple[list[int], bool]:
        ids, _ = counting(body)
        return {"stalls": ([], False), "stops": (ids[:2], False)}.get(body["user"], (ids, True))

    one = dataclasses.replace(PARAMS, lanes=1, queue=1)
    body = BODY | {"max_tokens": 4}
    warden, server, sinks, log = asyncio.run(drive({
        1: [post(body, "first"), post(body, "second")],     # the second finds the first in the queue
        2: [post(body, "waits")],                           # the first has the lane: this one the queue
        20: [post(BODY | {"max_tokens": 9}, "over")], 22: [post(body, "stalls")], 28: [post(body, "stops")]}, answer,
        params=one))
    D = one.delay
    by = {x["id"]: (x, s.answer()) for x, s in zip(log_order(log, sinks), sinks)}
    assert [by[k][0]["outcome"] for k in ("first", "second", "waits", "over", "stalls", "stops")] == [
        "end", "refused", "end", "fail", "fail", "fail"]
    assert [by[k][1][0] for k in ("first", "second", "waits", "over", "stalls", "stops")] == [200, 503, 200, 502, 502,
                                                                                               200]
    assert "refused the connection" in by["second"][1][1]["error"]["message"]
    assert by["second"][0]["kinds"] == "r" and warden.refused == 1
    # the first holds the lane through its END's bucket, 1 + D + 4, and the one waiting is admitted at the next tick
    first, waits = as_vllm_reads(by["first"][1][1]), as_vllm_reads(by["waits"][1][1])
    assert first["chunks"][0] == 1 + D and waits["chunks"][0] == (1 + D + 4 + 1) + D and waits["output_tokens"] == 4
    assert by["over"][0]["kinds"] == "pf" and by["stalls"][0]["kinds"] == "p" * D + "f"
    status, events, whole = by["stops"][1]
    assert not whole and [json.loads(e)["choices"][0]["token_ids"] for _, e in events] == [[4], [5]]
    assert by["stops"][0]["kinds"] == "p" * D + "ddf" and by["stops"][0]["response"] == {
        "status": None, "ids": None, "complete": False, "events": 2, "text_bytes": 0, "finish_reason": None,
        "prompt_tokens": None, "completion_tokens": 2}


def log_order(log: list[dict], sinks: list[Sink]) -> list[dict]:
    """The shim's records in the order their requests were sent (a record is written when its connection ends)."""
    assert len(log) == len(sinks)
    return sorted(log, key=lambda x: x["index"])


def test_a_cell_that_is_not_an_id_event_is_none():
    P = PARAMS.payload
    assert [C.event_id(IT.id_event(x, P)) for x in (0, 42, (1 << 32) - 1)] == [0, 42, (1 << 32) - 1]
    event = IT.id_event(42, P)
    for bad in (event[:-1] + b"x", b"x" + event[1:], event.replace(b"42", b"4x"), bytes(P), IT.TRAILER.ljust(P)):
        assert C.event_id(bad) is None

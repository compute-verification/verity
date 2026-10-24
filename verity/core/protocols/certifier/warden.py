"""The active warden: the network certifier as the clients' OpenAI endpoint, retransmitting every response on a fixed schedule.

The warden runs on an auditor host (`trusted-nebius-host`; co-located with the GPU for tonight's demo, a named
assumption). It terminates the clients' HTTP, serves ``/v1/completions`` (streamed and unstreamed) and ``/v1/models``,
and talks to the serving engine only through its narrow token port (`TokenPort`), which takes one canonical request
record and answers token ids. So no server byte reaches a client: the warden renders every byte it sends from token ids
and its own public rules, and certifies what it saw and what it sent with the certifier's grid (`grid.Certifier`) and
certificates (`certificate.Issuer`, in the runner).

**Ingress** (`canonical`). A body names only `model`, `prompt` (text, which the registered tokenizer encodes, or token
ids), `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `stream` and `user` (read and dropped), or a field of
`NEUTRAL` at its neutral value. Anything else is refused (400, naming the field): `top_k` > 0, `min_p`, the penalties,
`logit_bias`, `logprobs`, `n` > 1 and `best_of` > 1 among them. A request the warden takes gets the next ingress
ordinal and is the canonical `Request`; its record (`Request.encode`, fixed places then the prompt ids) is the ingress
grid's frames (`active.request_frames`) in the bucket the request is admitted.

**Admission.** At most `lanes` requests are served at once. A request waits in a FIFO queue of at most `queue`, else it
is refused with 429, and the queue's head is admitted at the first bucket with a free lane. A lane is held from the
admission bucket a through the request's last frame, whose bucket is fixed at admission by `max_tokens`, so an admitted
request's egress slots through `max_tokens` are free by construction. Nothing admitted is dropped or reordered.

**The schedule** (`Schedule.layout`). Every frame on a client connection is B = `frame_bytes` bytes: an HTTP/1.1 chunk
of fixed size (`Wire`), after a response head padded to B. Request q's frames are, by index k:
- k = 0, the head, in bucket a;
- k = 1 + j for j < S = r·⌈max_tokens/r⌉, slot j, in bucket a + D + ⌊j/r⌋: token j as one SSE event (streamed) or
  JSON whitespace (unstreamed), or a dummy of the same size (an SSE comment, or whitespace) once the response has
  finished and for j ≥ max_tokens;
- the final segment from bucket a + D + S/r, r frames a bucket: one frame streamed (the finish event and ``[DONE]``),
  `Schedule.body_frames(max_tokens)` unstreamed (the completion's JSON, padded with whitespace to the most text
  `max_tokens` tokens can render), the last ending the chunked body.
So each connection's frame count, sizes and buckets are a function of (a, stream, max_tokens) and the public
parameters alone, and a is a function of the arrivals and the admission rule.

**Egress** (`Renderer`). Token j is rendered at its slot's tick from its id alone: the registered vocabulary's bytes
(`Vocabulary`; special ids render nothing), decoded incrementally as UTF-8 with replacement, stop strings applied on the
rendered text with a hold-back of the longest stop's length less one character (the text before the first stop is
released, the stop and everything after it never), the finish being the first of an EOS id ("stop"), a stop string
("stop") and the `max_tokens`-th token ("length"). The server's finish field is never read for it.

**Fail closed.** A token not received by its slot's tick fails the request at that slot: an error event (streamed) or
an error body (unstreamed) is the client's, every later slot a dummy, and the record's frame there is `FAIL` with code
late (or malformed, when the server's stream broke the port's format before it). The connection still ends at its
scheduled bucket. A tick that runs more than `slack` past its boundary faults both grids' windows
(`monitor-unavailable`), as the active certifier's do.

**The records.** The egress grid has r·lanes slots a bucket of `record_bytes`-byte frames, one per frame sent, in
(ordinal, index) order: `active.encode_frame(kind, ordinal, index ‖ token id ‖ code)`, kinds `HEAD`, `TOKEN`, `DUMMY`,
`FINAL` and `FAIL`. The wire's bytes are `Wire`'s public function of a request's record frames and its canonical
request, so the record is what the warden sent. The ingress grid has lanes·F_in slots of `request_bytes`-byte frames.
`audit` is the read of the records in the clear: the egress frames in every bucket are exactly the schedule's for the
admissions the ingress records show, and no request failed.

`PROTOCOL.md`'s "The active warden" has the leak bound. The module imports only the standard library and `verity`.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import math
import struct
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Protocol

from verity.ml.commitments.identity import canonical_json_bytes, tagged_sha512
from verity.core.protocols.certifier.active import (DATA, HEADER, Requests, decode_frame, encode_frame, precise_sleep,
                                                    request_frame_count, request_frames)
from verity.core.protocols.certifier.grid import Certifier, GridParams, Reason, Record, row_count

PARAMETERS_TAG = "verity-network-warden/parameters/v1"
VOCABULARY_TAG = "verity-network-warden/vocabulary/v1"
#: the egress record's frame kinds, beside `active`'s PAD, DATA and SALT
HEAD, TOKEN, DUMMY, FINAL, FAIL = 0x10, 0x11, 0x12, 0x13, 0x14
KIND_NAMES = {HEAD: "head", TOKEN: "token", DUMMY: "dummy", FINAL: "final", FAIL: "fail"}
#: a record frame's code: a final frame's finish, a fail frame's reason
CODES = {None: 0, "stop": 1, "length": 2, "late": 3, "malformed": 4}
NO_TOKEN = 0xFFFF_FFFF
RECORD_FRAME = struct.Struct(">IIB")  # index, token id, code
#: the canonical request's fixed places: ordinal, flags (1 seed, 2 stream), seed, temperature, top_p, max_tokens, stops
REQUEST_HEAD = struct.Struct(">IBQddIB")
MAX_STOPS, STOP_BYTES = 4, 32
#: what a body may name at its neutral value only (None too): each changes the sampler, the output or the response
NEUTRAL: dict[str, tuple] = {"n": (1,), "best_of": (1,), "top_k": (-1, 0), "min_p": (0,), "presence_penalty": (0,),
                             "frequency_penalty": (0,), "repetition_penalty": (1,), "logit_bias": ({},),
                             "logprobs": (), "echo": (False,), "suffix": ()}
TAKEN = ("model", "prompt", "max_tokens", "temperature", "top_p", "seed", "stop", "stream", "user")
#: OpenAI's completions default
DEFAULT_MAX_TOKENS = 16
#: a token port line's bound: `{"id": …, "position": …, "finish": …}` and a newline
TOKEN_LINE_BYTES = 256
FINISH_WORDS = frozenset({"stop", "length", "abort"})
LAST_CHUNK = b"0\r\n\r\n"
#: a chunk's size field: four hex digits, leading zeros, so every frame's is the same length
CHUNK_DIGITS = 4
#: the most a client connection may hold unread before the warden stops writing to it
WRITE_BACKLOG = 1 << 22
STATUS_TEXT = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed", 413: "Content Too Large",
               429: "Too Many Requests", 431: "Request Header Fields Too Large", 503: "Service Unavailable"}


@dataclass(frozen=True)
class WardenParams:
    """The schedule's public parameters, fixed before the run (`Schedule.digest` binds them)."""

    frame_bytes: int = 1024  # B: every frame on a client connection
    slots: int = 8  # r: a connection's slots a bucket
    bucket_ms: int = 100  # τ
    delay: int = 10  # D: buckets from a request's admission to its first token's
    lanes: int = 8  # requests served at once
    queue: int = 64  # requests waiting for a lane, past which 429
    window: int = 600  # T: buckets a record covers
    max_tokens: int = 256
    max_prompt: int = 2048
    context: int = 4096  # prompt and max_tokens together, the server's model length
    record_bytes: int = 64  # an egress record frame
    request_bytes: int = 4096  # an ingress record frame

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} is a positive integer, not {value!r}")
        if self.record_bytes < HEADER.size + RECORD_FRAME.size:
            raise ValueError(f"a record frame holds {HEADER.size + RECORD_FRAME.size} bytes")
        if self.frame_bytes > 16 ** CHUNK_DIGITS + 2 + CHUNK_DIGITS + 2:
            raise ValueError(f"a frame is one chunk of at most {16 ** CHUNK_DIGITS - 1} bytes")
        if self.max_prompt + 1 > self.context:
            raise ValueError("the context holds the longest prompt and a token")


@dataclass(frozen=True)
class Vocabulary:
    """The registered detokenizer's table: each id's bytes, the special ids (which render nothing) and the EOS ids."""

    tokens: tuple[bytes, ...]
    special: frozenset[int]
    eos: frozenset[int]

    def __post_init__(self) -> None:
        if not self.eos or not self.eos <= set(range(len(self.tokens))) or not self.special <= set(range(len(self.tokens))):
            raise ValueError("the EOS and special ids are some of the vocabulary's")

    @property
    def size(self) -> int:
        return len(self.tokens)

    def render(self, token: int) -> bytes:
        return b"" if token in self.special else self.tokens[token]

    @cached_property
    def cost(self) -> int:
        """The most JSON-escaped bytes one token's bytes can render to: 6 a control byte (``\\u00XX``), 2 a quote or
        backslash, 3 a byte past ASCII (U+FFFD at worst), 1 else."""
        return max(sum(6 if x < 0x20 else 2 if x in b'"\\' else 3 if x >= 0x80 else 1 for x in self.render(i))
                   for i in range(self.size))

    @cached_property
    def digest(self) -> bytes:
        parts = [struct.pack(">I", len(self.tokens))] + [struct.pack(">I", len(t)) + t for t in self.tokens]
        for ids in (self.special, self.eos):
            parts += [struct.pack(">I", len(ids))] + [struct.pack(">I", i) for i in sorted(ids)]
        return tagged_sha512(VOCABULARY_TAG, b"".join(parts))

    @classmethod
    def from_tokenizer_json(cls, doc: Mapping, eos: Sequence[int]) -> Vocabulary:
        """A byte-level BPE `tokenizer.json`'s table (Llama 3's): each vocabulary string mapped back through GPT-2's
        bytes-to-unicode table, each added token special (rendering nothing) or its content."""
        if doc.get("model", {}).get("type") != "BPE" or (doc.get("decoder") or {}).get("type") != "ByteLevel":
            raise ValueError("the warden reads a byte-level BPE tokenizer.json")
        printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
        to_byte = {chr(b): b for b in printable}
        to_byte |= {chr(256 + n): b for n, b in enumerate(b for b in range(256) if b not in printable)}
        table: dict[int, bytes] = {i: bytes(to_byte[c] for c in s) for s, i in doc["model"]["vocab"].items()}
        special = set()
        for t in doc.get("added_tokens", ()):
            table[t["id"]] = t["content"].encode()
            if t.get("special"):
                special.add(t["id"])
        if set(table) != set(range(len(table))):
            raise ValueError("the tokenizer's ids aren't 0 … N − 1")
        return cls(tuple(table[i] for i in range(len(table))), frozenset(special), frozenset(eos))


class Refused(Exception):
    """A request the warden refuses before it takes it: an OpenAI error response with `status`."""

    def __init__(self, message: str, status: int = 400, param: str | None = None,
                 kind: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status, self.param, self.kind = status, param, kind


@dataclass(frozen=True)
class Request:
    """The canonical request: what the warden certifies on ingress and sends the server's token port."""

    ordinal: int
    prompt: tuple[int, ...]
    temperature: float
    top_p: float
    seed: int | None
    max_tokens: int
    stop: tuple[str, ...]
    stream: bool

    def encode(self) -> bytes:
        """u32 ordinal ‖ u8 flags (1 seed, 2 stream) ‖ u64 seed ‖ f64 temperature ‖ f64 top_p ‖ u32 max_tokens ‖ u8
        number of stops ‖ `MAX_STOPS` × (u8 length ‖ `STOP_BYTES` bytes, zero-filled) ‖ u32 prompt length ‖ u32 ids."""
        flags = (self.seed is not None) | 2 * self.stream
        out = [REQUEST_HEAD.pack(self.ordinal, flags, self.seed or 0, self.temperature, self.top_p, self.max_tokens,
                                 len(self.stop))]
        for s in self.stop + ("",) * (MAX_STOPS - len(self.stop)):
            b = s.encode()
            out.append(bytes([len(b)]) + b.ljust(STOP_BYTES, b"\0"))
        out.append(struct.pack(f">I{len(self.prompt)}I", len(self.prompt), *self.prompt))
        return b"".join(out)

    @classmethod
    def decode(cls, data: bytes) -> Request:
        try:
            ordinal, flags, seed, temperature, top_p, max_tokens, n = REQUEST_HEAD.unpack_from(data)
            at, stops = REQUEST_HEAD.size, []
            for i in range(MAX_STOPS):
                length = data[at]
                stops.append(data[at + 1: at + 1 + length].decode())
                at += 1 + STOP_BYTES
            (count,) = struct.unpack_from(">I", data, at)
            prompt = struct.unpack_from(f">{count}I", data, at + 4)
        except (struct.error, IndexError, UnicodeDecodeError) as e:
            raise ValueError(f"not a canonical request: {e}") from None
        request = cls(ordinal, prompt, temperature, top_p, seed if flags & 1 else None, max_tokens, tuple(stops[:n]),
                      bool(flags & 2))
        if flags > 3 or n > MAX_STOPS or request.encode() != data:
            raise ValueError("not a canonical request")
        return request

    def port_record(self) -> dict:
        """The token port's request line: (ordinal, prompt ids, temperature, top_p, seed or none, max_tokens, stop)."""
        return {"ordinal": self.ordinal, "prompt": list(self.prompt), "temperature": self.temperature,
                "top_p": self.top_p, "seed": self.seed, "max_tokens": self.max_tokens, "stop": list(self.stop)}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical(body: Any, ordinal: int, params: WardenParams, vocabulary: Vocabulary, *, model: str,
              encode: Callable[[str], list[int]] | None = None) -> Request:
    """The canonical request of a ``/v1/completions`` body, or `Refused` naming the field."""
    if not isinstance(body, dict):
        raise Refused("the body is a JSON object")
    for key, value in body.items():
        if key in NEUTRAL:
            if value is not None and not any(value == v and isinstance(value, bool) == isinstance(v, bool)
                                             for v in NEUTRAL[key]):
                raise Refused(f"{key}: the warden serves only temperature, top_p, seed, max_tokens and stop, so "
                              f"{key} is refused unless {', '.join(map(repr, NEUTRAL[key])) or 'null'}", param=key)
        elif key not in TAKEN:
            raise Refused(f"{key}: not a field the warden takes", param=key)
    if body.get("model") not in (None, model):
        raise Refused(f"model: this warden serves {model!r}", status=404, param="model", kind="model_not_found")
    prompt = body.get("prompt")
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
        prompt = prompt[0]
    if isinstance(prompt, str):
        if encode is None:
            raise Refused("prompt: this warden takes the prompt as token ids", param="prompt")
        ids = list(encode(prompt))
    elif isinstance(prompt, list) and all(type(i) is int for i in prompt):
        ids = prompt
    else:
        raise Refused("prompt: one prompt, as text or token ids", param="prompt")
    if not 1 <= len(ids) <= params.max_prompt or not all(0 <= i < vocabulary.size for i in ids):
        raise Refused(f"prompt: 1 … {params.max_prompt} ids of the registered vocabulary", param="prompt")
    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    if type(max_tokens) is not int or not 1 <= max_tokens <= params.max_tokens:
        raise Refused(f"max_tokens: 1 … {params.max_tokens}", param="max_tokens")
    if len(ids) + max_tokens > params.context:
        raise Refused(f"the prompt and max_tokens exceed the context of {params.context} tokens", param="max_tokens")
    temperature = 1.0 if body.get("temperature") is None else body["temperature"]
    if not _number(temperature) or not 0 <= temperature <= 2:
        raise Refused("temperature: 0 … 2", param="temperature")
    top_p = 1.0 if body.get("top_p") is None else body["top_p"]
    if not _number(top_p) or not 0 < top_p <= 1:
        raise Refused("top_p: in (0, 1]", param="top_p")
    seed = body.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 1 << 63):
        raise Refused("seed: an integer in 0 … 2^63 − 1, or null", param="seed")
    stop = body.get("stop")
    stop = [] if stop is None else [stop] if isinstance(stop, str) else stop
    if not isinstance(stop, list) or len(stop) > MAX_STOPS or \
            not all(isinstance(s, str) and 1 <= len(s.encode()) <= STOP_BYTES for s in stop):
        raise Refused(f"stop: at most {MAX_STOPS} strings of 1 … {STOP_BYTES} UTF-8 bytes", param="stop")
    stream = body.get("stream", False)
    if stream is None:
        stream = False
    if not isinstance(stream, bool):
        raise Refused("stream: a boolean", param="stream")
    if body.get("user") is not None and not isinstance(body["user"], str):
        raise Refused("user: a string", param="user")
    return Request(ordinal, tuple(ids), float(temperature), float(top_p), seed, max_tokens, tuple(stop), stream)


class Renderer:
    """One response's text, rendered from its token ids alone, one token a slot (the module docstring's "Egress").
    `feed(id)` returns the text that token's slot releases, `close()` what the final frame releases; `finish` is
    "stop" or "length" once the response has finished, and `text` everything released."""

    def __init__(self, vocabulary: Vocabulary, request: Request) -> None:
        self.vocabulary, self.max_tokens, self.stop = vocabulary, request.max_tokens, request.stop
        self.hold = max(map(len, self.stop), default=1) - 1
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.pending = ""
        self.count = 0
        self.finish: str | None = None
        self.released: list[str] = []
        self._stopped = False

    def feed(self, token: int) -> str:
        if self.finish is not None:
            raise ValueError("the response has finished")
        self.count += 1
        if token in self.vocabulary.eos:
            self.finish = "stop"
            return ""
        out = self._release(self._decoder.decode(self.vocabulary.render(token)))
        if self.finish is None and self.count >= self.max_tokens:
            self.finish = "length"
        return out

    def close(self) -> str:
        return "" if self._stopped else self._release(self._decoder.decode(b"", final=True), final=True)

    def _release(self, piece: str, final: bool = False) -> str:
        s = self.pending + piece
        hits = [i for i in map(s.find, self.stop) if i >= 0]
        if hits:
            self.finish, self._stopped, self.pending, out = "stop", True, "", s[:min(hits)]
        else:
            cut = len(s) if final else max(0, len(s) - self.hold)
            self.pending, out = s[cut:], s[:cut]
        self.released.append(out)
        return out

    @property
    def text(self) -> str:
        return "".join(self.released)


def render(vocabulary: Vocabulary, request: Request, tokens: Sequence[int]) -> tuple[str, str | None, int]:
    """(text, finish, tokens used) of a response to `request` whose token ids are `tokens`: `Renderer`'s rule."""
    r = Renderer(vocabulary, request)
    for token in tokens:
        if r.finish is not None:
            break
        r.feed(token)
    if r.finish is not None:
        r.close()
    return r.text, r.finish, r.count


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _error(message: str, kind: str, code: int, param: str | None = None) -> bytes:
    return _json({"error": {"message": message, "type": kind, "param": param, "code": code}})


FAIL_ERRORS = {"late": ("token {j} was not ready at its deadline", "WardenLate", 504),
               "malformed": ("the server's token stream broke the port's format before token {j}",
                             "WardenUpstreamMalformed", 502)}


class Wire:
    """What the warden writes on a client connection: every frame `frame_bytes` long, a chunk of HTTP/1.1's chunked
    body after a padded head; streamed, each a server-sent event or comment padded with spaces, unstreamed whitespace
    then the completion's JSON padded with spaces."""

    def __init__(self, frame_bytes: int, model: str) -> None:
        self.frame_bytes, self.model = frame_bytes, _json(model)
        self.room = frame_bytes - CHUNK_DIGITS - 4
        self.last_room = self.room - len(LAST_CHUNK)
        self.done = b"data: [DONE]\n\n"

    def chunk(self, payload: bytes, last: bool = False) -> bytes:
        if len(payload) != (self.last_room if last else self.room):
            raise ValueError(f"a {len(payload)}-byte payload isn't a frame's")
        return b"%0*x\r\n" % (CHUNK_DIGITS, len(payload)) + payload + b"\r\n" + (LAST_CHUNK if last else b"")

    def head(self, stream: bool) -> bytes:
        kind = b"text/event-stream" if stream else b"application/json"
        head = (b"HTTP/1.1 200 OK\r\ncontent-type: " + kind + b"\r\ncache-control: no-cache\r\n"
                b"transfer-encoding: chunked\r\nconnection: close\r\nx-warden-pad: ")
        fill = self.frame_bytes - len(head) - 4
        if fill < 0:
            raise ValueError("a frame holds the response head")
        return head + b"0" * fill + b"\r\n\r\n"

    def _event(self, data: bytes, room: int, tail: bytes = b"") -> bytes:
        fill = room - len(b"data: ") - len(data) - 2 - len(tail)
        if fill < 0:
            raise ValueError(f"an event of {len(data)} bytes doesn't fit a frame")
        return b"data: " + data + b" " * fill + b"\n\n" + tail

    def _choice(self, ordinal: int, text: str, finish: str | None) -> bytes:
        return (b'{"id":"cmpl-%d","object":"text_completion","created":0,"model":' % ordinal + self.model +
                b',"choices":[{"index":0,"text":' + _json(text) + b',"logprobs":null,"finish_reason":' +
                _json(finish) + b"}]}")

    def token(self, ordinal: int, text: str) -> bytes:
        return self.chunk(self._event(self._choice(ordinal, text, None), self.room))

    def dummy(self, stream: bool) -> bytes:
        return self.chunk(b":" + b" " * (self.room - 3) + b"\n\n" if stream else b" " * self.room)

    def error(self, ordinal: int, code: str, j: int) -> bytes:
        message, kind, status = FAIL_ERRORS[code]
        return self.chunk(self._event(_error(message.format(j=j), kind, status), self.room))

    def final(self, ordinal: int, text: str, finish: str) -> bytes:
        return self.chunk(self._event(self._choice(ordinal, text, finish), self.last_room, self.done), last=True)

    def ended(self) -> bytes:
        """A failed stream's final frame: ``[DONE]`` after a comment."""
        return self.chunk(b":" + b" " * (self.last_room - 3 - len(self.done)) + b"\n\n" + self.done, last=True)

    def completion(self, ordinal: int, text: str, finish: str, prompt_tokens: int, completion_tokens: int) -> bytes:
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens}
        return (b'{"id":"cmpl-%d","object":"text_completion","created":0,"model":' % ordinal + self.model +
                b',"choices":[{"index":0,"text":' + _json(text) + b',"logprobs":null,"finish_reason":' +
                _json(finish) + b'}],"usage":' + _json(usage) + b"}")

    def body(self, data: bytes, frames: int) -> list[bytes]:
        """An unstreamed body's final segment: `data` padded with spaces to `frames` frames."""
        size = (frames - 1) * self.room + self.last_room
        if len(data) > size:
            raise ValueError(f"a body of {len(data)} bytes doesn't fit {frames} frames")
        data = data.ljust(size, b" ")
        cuts = [i * self.room for i in range(frames)] + [size]
        return [self.chunk(data[cuts[i]:cuts[i + 1]], last=i == frames - 1) for i in range(frames)]


class Schedule:
    """The warden's public schedule on one registered model: its parameters, the vocabulary its renderer reads and the
    model's name, the frames of a request's response (`layout`) and the two grids' parameters."""

    #: the most digits an ordinal or a usage count has
    DIGITS = 10

    def __init__(self, params: WardenParams, vocabulary: Vocabulary, model: str) -> None:
        self.params, self.vocabulary, self.model = params, vocabulary, model
        self.wire = w = Wire(params.frame_bytes, model)
        big = 10 ** self.DIGITS - 1
        hold = 6 * (STOP_BYTES - 1)
        self.frame_text = vocabulary.cost + 9 + hold  # the most escaped text one slot releases
        event = len(b"data: ") + 2
        token_room = w.room - event - len(w._choice(big, "", None))
        final_room = w.last_room - event - len(w._choice(big, "", "length")) - len(w.done)
        error_room = w.room - event - max(len(_error(m.format(j=big), k, s)) for m, k, s in FAIL_ERRORS.values())
        if min(token_room - self.frame_text, final_room - hold - 9, error_room) < 0:
            raise ValueError(f"a {params.frame_bytes}-byte frame can't hold one slot's text, up to {self.frame_text} "
                             "escaped bytes of this vocabulary: raise frame_bytes")
        self.body_overhead = len(w.completion(big, "", "length", big, big))
        w.head(True)
        self.egress_grid = GridParams(params.window, params.lanes * params.slots, params.record_bytes, params.bucket_ms)
        longest = REQUEST_HEAD.size + MAX_STOPS * (1 + STOP_BYTES) + 4 + 4 * params.max_prompt
        self.request_frames = request_frame_count(longest, params.request_bytes)
        self.ingress_grid = GridParams(params.window, params.lanes * self.request_frames, params.request_bytes,
                                       params.bucket_ms)

    def slots(self, max_tokens: int) -> int:
        """S: the token slots of a response, `max_tokens` rounded up to whole buckets."""
        return self.params.slots * -(-max_tokens // self.params.slots)

    def body_frames(self, max_tokens: int) -> int:
        """An unstreamed response's final frames: room for the JSON of the most text `max_tokens` tokens render."""
        size = self.body_overhead + max_tokens * self.vocabulary.cost + 9
        return 1 + max(0, -(-(size - self.wire.last_room) // self.wire.room))

    def layout(self, stream: bool, max_tokens: int) -> tuple[int, ...]:
        """Each frame's bucket, from the admission bucket: the head, the token slots, the final segment."""
        r, D, S = self.params.slots, self.params.delay, self.slots(max_tokens)
        final = 1 if stream else self.body_frames(max_tokens)
        return (0, *(D + j // r for j in range(S)), *(D + S // r + i // r for i in range(final)))

    def role(self, stream: bool, max_tokens: int, k: int) -> tuple[int, int]:
        """(HEAD, 0), (TOKEN, j) for slot j, or (FINAL, i) for the final segment's frame i."""
        S = self.slots(max_tokens)
        return (HEAD, 0) if k == 0 else (TOKEN, k - 1) if k <= S else (FINAL, k - 1 - S)

    @cached_property
    def digest(self) -> bytes:
        """The 64 bytes each record head's `parameters` carries: the parameters, the model and the vocabulary."""
        return tagged_sha512(PARAMETERS_TAG, canonical_json_bytes(
            asdict(self.params) | {"model": self.model, "vocabulary": self.vocabulary.digest.hex(),
                                   "max_stops": MAX_STOPS, "stop_bytes": STOP_BYTES}))


def record_frame(kind: int, ordinal: int, index: int, frame_bytes: int, token: int = NO_TOKEN, code: int = 0) -> bytes:
    return encode_frame(kind, ordinal, RECORD_FRAME.pack(index, token, code), frame_bytes)


def read_record_frame(frame: bytes) -> tuple[int, int, int, int, int]:
    """(kind, ordinal, index, token id, code) of an egress record frame."""
    kind, ordinal, payload = decode_frame(frame)
    if len(payload) != RECORD_FRAME.size or kind not in KIND_NAMES:
        raise ValueError("not a warden record frame")
    return (kind, ordinal, *RECORD_FRAME.unpack(payload))


@dataclass(frozen=True, slots=True)
class Token:
    """One message of the token port: a sampled id at its position, and the server's finish, which the warden logs
    and never renders."""

    id: int
    position: int
    finish: str | None = None


class Malformed(Exception):
    """A server message that isn't one token port line."""


def parse_token(line: bytes) -> Token:
    """One token port line, exactly ``{"id": int, "position": int, "finish": null | "stop" | "length" | "abort"}``."""
    if not line.endswith(b"\n") or len(line) > TOKEN_LINE_BYTES:
        raise Malformed("a token line is one JSON object and a newline, at most "
                        f"{TOKEN_LINE_BYTES} bytes")
    try:
        doc = json.loads(line)
    except ValueError:
        raise Malformed("a token line is JSON") from None
    if not isinstance(doc, dict) or set(doc) != {"id", "position", "finish"}:
        raise Malformed("a token line has exactly id, position and finish")
    i, p, f = doc["id"], doc["position"], doc["finish"]
    if type(i) is not int or type(p) is not int or i < 0 or p < 0 or not (f is None or f in FINISH_WORDS):
        raise Malformed("a token line's id and position are non-negative integers, its finish null or a reason")
    return Token(i, p, f)


class Upstream(Protocol):
    """The server's side: a stream of `Token`s for one canonical request. Cancelling the iteration aborts it."""

    def stream(self, request: Request) -> AsyncIterator[Token]: ...


class TokenPort:
    """The serving engine's token-stream port: one TCP connection a request, the request as one JSON line
    (`Request.port_record`), then one `parse_token` line a sampled token, in position order, until the server closes.
    The warden closes the connection to abort. `connect` opens it, `asyncio.open_connection` to (host, port) by
    default."""

    def __init__(self, host: str, port: int, *,
                 connect: Callable[[], Awaitable[tuple[asyncio.StreamReader, Any]]] | None = None) -> None:
        self.host, self.port = host, port
        self.connect = connect or (lambda: asyncio.open_connection(host, port, limit=TOKEN_LINE_BYTES))

    async def stream(self, request: Request) -> AsyncIterator[Token]:
        reader, writer = await self.connect()
        try:
            writer.write(_json(request.port_record()) + b"\n")
            await writer.drain()
            while True:
                try:
                    line = await reader.readline()
                except ValueError:
                    raise Malformed(f"a token line is at most {TOKEN_LINE_BYTES} bytes") from None
                if not line:
                    return
                yield parse_token(line)
        finally:
            writer.close()


async def read_request(reader: asyncio.StreamReader, body_bytes: int) -> tuple[str, str, bytes]:
    """(method, path, body) of one HTTP/1.1 request with a Content-Length body, or `Refused`."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.LimitOverrunError:
        raise Refused("the request head is too long", status=431) from None
    except asyncio.IncompleteReadError:
        raise Refused("the connection ended inside the request head") from None
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise Refused("not an HTTP/1.1 request line")
    headers = {}
    for line in lines[1:]:
        if line:
            name, sep, value = line.partition(":")
            if not sep:
                raise Refused("a header line is name: value")
            headers[name.strip().lower()] = value.strip()
    if "transfer-encoding" in headers:
        raise Refused("the warden reads a Content-Length body, not a chunked one")
    length = headers.get("content-length", "0")
    if not length.isdigit():
        raise Refused("content-length: a number")
    if int(length) > body_bytes:
        raise Refused(f"the body is over {body_bytes} bytes", status=413)
    try:
        body = await reader.readexactly(int(length))
    except asyncio.IncompleteReadError:
        raise Refused("the connection ended inside the body") from None
    return parts[0], parts[1].split("?")[0], body


def response(status: int, body: bytes) -> bytes:
    return (b"HTTP/1.1 %d %s\r\ncontent-type: application/json\r\ncontent-length: %d\r\nconnection: close\r\n\r\n"
            % (status, STATUS_TEXT.get(status, "Error").encode(), len(body)) + body)


@dataclass(eq=False)
class _Served:
    request: Request
    writer: Any
    done: asyncio.Future
    renderer: Renderer
    layout: tuple[int, ...]
    arrived: int
    admitted: int | None = None
    tokens: list[int] = field(default_factory=list)
    sent: int = 0
    malformed: str | None = None
    upstream_error: str | None = None
    extra: int = 0
    server_finish: str | None = None
    failed: str | None = None
    failed_at: int | None = None
    body: list[bytes] | None = None
    task: asyncio.Task | None = None


class Warden:
    """The active warden on one registered model (the module docstring): `handle(reader, writer)` serves one client
    connection, `run()` ticks the schedule from `origin`, the clock's time of bucket 0, for `windows` record windows,
    then cuts whatever is still served or queued. `upstream` is the server's token port (`TokenPort`).

    - `publish(record, params)` receives each window's records, ingress then egress, as they close (by default they
      are kept in `records`); `log` has one entry a request once its connection ends, `refused` each refusal's status
      and field, `lateness` each tick's in seconds;
    - `encode`, the registered tokenizer's text-to-ids (with its BOS), takes text prompts; without it a prompt is ids;
    - `until(predicate)` waits until the warden's state satisfies it, for a caller that drives a `VirtualClock`."""

    def __init__(self, schedule: Schedule, upstream: Upstream, *, origin: float, windows: int,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = precise_sleep,
                 publish: Callable[[Record, GridParams], None] | None = None,
                 encode: Callable[[str], list[int]] | None = None, slack: float = 0.025,
                 egress_link: str = "responses", ingress_link: str = "requests", body_bytes: int = 1 << 20) -> None:
        self.schedule, self.params, self.upstream = schedule, schedule.params, upstream
        self.origin, self.windows, self.clock, self.sleep, self.slack = origin, windows, clock, sleep, slack
        self.encode, self.body_bytes = encode, body_bytes
        self.egress = Certifier(egress_link, schedule.egress_grid)
        self.ingress = Certifier(ingress_link, schedule.ingress_grid)
        self.records: list[Record] = []
        self.publish = publish or (lambda record, params: self.records.append(record))
        self.queue: deque[_Served] = deque()
        self.active: list[_Served] = []
        self.ordinals = 0
        self.log: list[dict] = []
        self.refused: list[tuple[int, str | None]] = []
        self.lateness: list[float] = []
        self.ticks = 0
        self._changed = asyncio.Event()

    async def until(self, predicate: Callable[[Warden], bool]) -> None:
        while not predicate(self):
            self._changed.clear()
            await self._changed.wait()

    def _bucket(self) -> int:
        return math.ceil((self.clock() - self.origin) * 1000 / self.params.bucket_ms)

    async def handle(self, reader: asyncio.StreamReader, writer: Any) -> None:
        try:
            served = self._take(*await read_request(reader, self.body_bytes), writer)
            if served is not None:
                await served.done
        except Refused as e:
            self.refused.append((e.status, e.param))
            writer.write(response(e.status, _error(str(e), e.kind, e.status, e.param)))
        finally:
            writer.close()
            self._changed.set()

    def _take(self, method: str, path: str, body: bytes, writer: Any) -> _Served | None:
        if path == "/v1/models":
            if method != "GET":
                raise Refused("GET /v1/models", status=405)
            writer.write(response(200, _json({"object": "list", "data": [
                {"id": self.schedule.model, "object": "model", "created": 0, "owned_by": "verity"}]})))
            return None
        if path != "/v1/completions":
            raise Refused(f"the warden serves /v1/completions and /v1/models, not {path}", status=404)
        if method != "POST":
            raise Refused("POST /v1/completions", status=405)
        try:
            doc = json.loads(body)
        except ValueError:
            raise Refused("the body is JSON") from None
        request = canonical(doc, self.ordinals, self.params, self.schedule.vocabulary, model=self.schedule.model,
                            encode=self.encode)
        if len(self.queue) >= self.params.queue:
            raise Refused(f"{self.params.queue} requests are waiting for a lane", status=429,
                          kind="rate_limit_exceeded")
        self.ordinals += 1
        served = _Served(request, writer, asyncio.get_running_loop().create_future(),
                         Renderer(self.schedule.vocabulary, request),
                         self.schedule.layout(request.stream, request.max_tokens), self._bucket())
        self.queue.append(served)
        self._changed.set()
        return served

    async def run(self) -> None:
        tau = self.params.bucket_ms / 1000
        try:
            for t in range(self.windows * self.params.window):
                deadline = self.origin + t * tau
                await self.sleep(deadline - self.clock())
                late = self.clock() - deadline
                self.lateness.append(late)
                if late > self.slack:
                    self.ingress.fault(Reason.MONITOR_UNAVAILABLE)
                    self.egress.fault(Reason.MONITOR_UNAVAILABLE)
                self._tick(t)
                self.ticks += 1
                self._changed.set()
        finally:
            for served in [*self.active, *self.queue]:
                self._end(served, None, cut=True)
            self.active, self.queue = [], deque()
            self._changed.set()

    def _tick(self, t: int) -> None:
        while self.queue and len(self.active) < self.params.lanes:
            self._admit(self.queue.popleft(), t)
        for served in self.active:
            out = []
            while served.sent < len(served.layout) and served.admitted + served.layout[served.sent] == t:
                record, wire = self._frame(served, served.sent)
                self.egress.enqueue(record)
                out.append(wire)
                served.sent += 1
            if out:
                self._write(served, b"".join(out))
        for served in self.active:
            if served.sent == len(served.layout):
                self._end(served, t)
        self.active = [s for s in self.active if s.sent < len(s.layout)]
        self.ingress.tick()
        self.egress.tick()
        if (t + 1) % self.params.window == 0:
            self.publish(self.ingress.close_window(), self.schedule.ingress_grid)
            self.publish(self.egress.close_window(), self.schedule.egress_grid)

    def _admit(self, served: _Served, t: int) -> None:
        served.admitted = t
        self.active.append(served)
        size = self.params.request_bytes
        data = request_frames(served.request.encode(), size)
        for i in range(0, len(data), size):
            self.ingress.enqueue(data[i:i + size])
        served.task = asyncio.create_task(self._pump(served))

    async def _pump(self, served: _Served) -> None:
        """Read the server's tokens for one request. A token out of order or outside the vocabulary stops the read as
        malformed, and so does one past max_tokens (`extra`); what came before it is kept."""
        try:
            async for token in self.upstream.stream(served.request):
                if len(served.tokens) == served.request.max_tokens:
                    served.extra += 1
                    return
                if token.position != len(served.tokens) or not 0 <= token.id < self.schedule.vocabulary.size:
                    served.malformed = f"token at position {token.position}, id {token.id}"
                    return
                served.tokens.append(token.id)
                served.server_finish = token.finish
                self._changed.set()
        except Malformed as e:
            served.malformed = str(e)
        except OSError as e:
            served.upstream_error = f"{type(e).__name__}: {e}"
        finally:
            self._changed.set()

    def _abort(self, served: _Served) -> None:
        if served.task is not None:
            served.task.cancel()

    def _frame(self, served: _Served, k: int) -> tuple[bytes, bytes]:
        """Frame k of a response at its tick: its record frame and the bytes the client gets."""
        q, w, size = served.request, self.schedule.wire, self.params.record_bytes
        role, j = self.schedule.role(q.stream, q.max_tokens, k)
        if role == HEAD:
            return record_frame(HEAD, q.ordinal, k, size), w.head(q.stream)
        r = served.renderer
        if role == TOKEN:
            if served.failed or r.finish is not None or j >= q.max_tokens:
                return record_frame(DUMMY, q.ordinal, k, size), w.dummy(q.stream)
            if j < len(served.tokens):
                token = served.tokens[j]
                text = r.feed(token)
                if r.finish is not None:
                    self._abort(served)
                return record_frame(TOKEN, q.ordinal, k, size, token), w.token(q.ordinal, text) if q.stream \
                    else w.dummy(False)
            served.failed, served.failed_at = ("malformed" if served.malformed else "late"), j
            self._abort(served)
            return record_frame(FAIL, q.ordinal, k, size, code=CODES[served.failed]), \
                w.error(q.ordinal, served.failed, j) if q.stream else w.dummy(False)
        if served.body is None:
            # close() may meet a stop string in the held-back text, so the finish is read after it
            if served.failed:
                message, kind, status = FAIL_ERRORS[served.failed]
                served.body = [w.ended()] if q.stream else \
                    w.body(_error(message.format(j=served.failed_at), kind, status), self.schedule.body_frames(q.max_tokens))
            elif q.stream:
                served.body = [w.final(q.ordinal, r.close(), r.finish)]
            else:
                text = r.text + r.close()
                served.body = w.body(w.completion(q.ordinal, text, r.finish, len(q.prompt), r.count),
                                     self.schedule.body_frames(q.max_tokens))
        return record_frame(FINAL, q.ordinal, k, size, code=CODES[served.failed or r.finish]), served.body[j]

    def _write(self, served: _Served, data: bytes) -> None:
        writer = served.writer
        if writer.is_closing():
            return
        if writer.transport.get_write_buffer_size() > WRITE_BACKLOG:
            writer.close()
            return
        writer.write(data)

    def _end(self, served: _Served, t: int | None, cut: bool = False) -> None:
        self._abort(served)
        served.writer.close()
        if not served.done.done():
            served.done.set_result(None)
        q, r = served.request, served.renderer
        self.log.append({"ordinal": q.ordinal, "arrived": served.arrived, "admitted": served.admitted, "end": t,
                         "stream": q.stream, "prompt_tokens": len(q.prompt), "max_tokens": q.max_tokens,
                         "temperature": q.temperature, "seed": q.seed, "stop": list(q.stop),
                         "tokens": r.count, "received": len(served.tokens), "finish": r.finish,
                         "failed": served.failed, "failed_at": served.failed_at, "extra": served.extra,
                         "malformed": served.malformed, "upstream_error": served.upstream_error,
                         "server_finish": served.server_finish, "cut": cut})


def audit(schedule: Schedule, ingress: Sequence[Record], egress: Sequence[Record]) -> dict:
    """The read of a run's records in the clear (a rehearsal's; a deployment's is N2's, in ZK): the requests the
    ingress records admit, each in one bucket and in ordinal order, and in every bucket of the egress records exactly
    the frames the schedule puts there, of the kinds their roles allow, each response's token ids rendering to its
    finish at its last token frame (`render`). A window is accepted when both its records are complete, its frames
    are the schedule's and no request failed in it."""
    T = schedule.params.window
    verdicts: dict[int, list[str]] = defaultdict(list)
    for record in [*ingress, *egress]:
        if not record.status.complete:
            verdicts[record.window].append(f"{record.link}: {record.status.reason.value}")
    admitted: list[tuple[Request, int]] = []
    cutter, data, first = Requests(), defaultdict(bytearray), {}
    for record in sorted(ingress, key=lambda r: r.window):
        for b, row in enumerate(record.grid):
            t = record.window * T + b
            for frame in row[:row_count(row)]:
                kind, _, payload = decode_frame(frame)
                if kind != DATA:
                    verdicts[record.window].append(f"bucket {t}: an ingress frame of kind {kind}")
                    continue
                s, chunk, last = cutter.feed(payload)
                data[s] += chunk
                first.setdefault(s, t)
                if last:
                    try:
                        request = Request.decode(bytes(data.pop(s)))
                    except ValueError as e:
                        verdicts[record.window].append(f"bucket {t}: {e}")
                        continue
                    if first.pop(s) != t:
                        verdicts[record.window].append(f"bucket {t}: request {request.ordinal} spans buckets")
                    admitted.append((request, t))
    for n, (request, t) in enumerate(admitted):
        if request.ordinal != n:
            verdicts[t // T].append(f"bucket {t}: ordinal {request.ordinal} where {n} is next")
    expected: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for request, a in admitted:
        for k, offset in enumerate(schedule.layout(request.stream, request.max_tokens)):
            expected[a + offset].append((request.ordinal, k))
    frames: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    recorded = set()
    for record in sorted(egress, key=lambda r: r.window):
        recorded.add(record.window)
        for b, row in enumerate(record.grid):
            t = record.window * T + b
            try:
                got = [read_record_frame(f) for f in row[:row_count(row)]]
            except ValueError as e:
                verdicts[record.window].append(f"bucket {t}: {e}")
                continue
            if [(o, k) for _, o, k, _, _ in got] != expected.get(t, []):
                verdicts[record.window].append(f"bucket {t}: the frames aren't the schedule's")
            for kind, o, k, token, code in got:
                frames[o].append((t, kind, k, token, code))
    requests = []
    for request, a in admitted:
        o, mine = request.ordinal, frames.get(request.ordinal, [])
        kinds = [(kind, k, token, code) for _, kind, k, token, code in mine]
        ids = [token for kind, _, token, _ in kinds if kind == TOKEN]
        text, finish, used = render(schedule.vocabulary, request, ids)
        fails = [(t, code) for t, kind, _, _, code in mine if kind == FAIL]
        problem = None
        for kind, k, token, code in kinds:
            role, j = schedule.role(request.stream, request.max_tokens, k)
            allowed = {HEAD} if role == HEAD else {FINAL} if role == FINAL else {TOKEN, DUMMY, FAIL}
            if kind not in allowed:
                problem = f"frame {k} is a {KIND_NAMES[kind]} frame"
        slots = [kind for kind, k, _, _ in kinds if schedule.role(request.stream, request.max_tokens, k)[0] == TOKEN]
        live = len(ids) + len(fails)
        if slots[:live] != [TOKEN] * len(ids) + [FAIL] * len(fails) or any(kind != DUMMY for kind in slots[live:]) \
                or len(fails) > 1:
            problem = problem or "the token slots aren't tokens, then at most one fail, then dummies"
        elif not fails and len(slots) == schedule.slots(request.max_tokens) and (used != len(ids) or finish is None):
            problem = f"its {len(ids)} tokens don't end at their finish ({finish}, after {used})"
        final = [code for kind, _, _, code in kinds if kind == FINAL]
        if final and not fails and any(c != CODES[finish] for c in final):
            problem = problem or "its final frames don't carry its finish"
        if problem:
            verdicts[a // T].append(f"request {o}: {problem}")
        for t, code in fails:
            verdicts[t // T].append(f"request {o} failed {next(k for k, v in CODES.items() if v == code)}")
        requests.append({"ordinal": o, "admitted": a, "stream": request.stream, "prompt_tokens": len(request.prompt),
                         "max_tokens": request.max_tokens, "temperature": request.temperature, "top_p": request.top_p,
                         "seed": request.seed, "stop": list(request.stop), "ids": ids, "text": text,
                         "finish": None if fails else finish,
                         "failed": next((k for k, v in CODES.items() if fails and v == fails[0][1]), None)})
    windows = sorted(recorded | {r.window for r in ingress})
    out = [{"window": w, "accepted": not verdicts.get(w), "problems": verdicts.get(w, [])} for w in windows]
    return {"accepted": bool(out) and all(w["accepted"] for w in out), "windows": out, "requests": requests,
            "frames": sum(len(v) for v in frames.values())}

"""Bytes on a channel: a message's frame, and the parse of a frame as adversarial. What a message declares (its schema,
its kind, its type's registry) is `verity.channel`'s.

~~~text
frame     u32be len(head) ‖ u32be len(body) ‖ head ‖ body
head      canonical JSON (the commitments' rules: sorted keys, no whitespace, UTF-8, no floats) of one object,
          {"type": T, "v": V, ...fields}
body      the message's bytes, never base64
~~~

The frame is `spot.remote`'s, with a body. `encode` is a message's frame under a registry (`MessageTypes`), checked
against its schema; `decode` parses every frame as adversarial and refuses by name (`Refused.name`, one of `REFUSALS`):
a frame that isn't one (`malformed`), a head or body past its bound (`oversized`), a type the registry doesn't hold
(`unknown-type`), a version it doesn't (`unknown-version`), and fields other than the schema's (`schema`). The head must
be the canonical encoding of what it says, so one message has one frame. `FrameReader` cuts a byte stream into frames,
and `handshake` is the exchange that opens every channel between parties.
"""

from __future__ import annotations

import json
import struct
from typing import Any

from verity.channel import MAX_HEAD, Message, MessageTypes, Refused, canonical

WIRE_VERSION = 1
HEADER = struct.Struct(">II")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError("an object repeats a key")
    return dict(pairs)


def _no_float(s: str) -> Any:
    raise ValueError(f"a number with a fraction or an exponent ({s[:32]})")


def _no_constant(s: str) -> Any:
    raise ValueError(f"{s} is not JSON")


def parse(data: bytes) -> Any:
    """The canonical JSON `data` encodes; anything else (another encoding of it included) raises `Refused('malformed')`."""
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_float=_no_float,
                           parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise Refused("malformed", f"the head is not JSON ({e})") from None
    if canonical(value) != data:
        raise Refused("malformed", "the head is not in canonical form")
    return value


def frame(head: bytes, body: bytes = b"") -> bytes:
    return HEADER.pack(len(head), len(body)) + head + body


def split(data: bytes, max_head: int = MAX_HEAD, max_body: int = 1 << 32) -> tuple[bytes, bytes]:
    """A frame's head and body, its header checked against the bounds before anything else."""
    if len(data) < HEADER.size:
        raise Refused("malformed", f"a frame is at least {HEADER.size} bytes, not {len(data)}")
    nh, nb = HEADER.unpack_from(data)
    if nh > max_head or nb > max_body:
        raise Refused("oversized", f"a frame is a head of at most {max_head} bytes and a body of at most {max_body}, not "
                                   f"{nh} + {nb}")
    if len(data) != HEADER.size + nh + nb:
        raise Refused("malformed", f"a frame of {nh} + {nb} bytes is {len(data) - HEADER.size} bytes long")
    return data[HEADER.size:HEADER.size + nh], data[HEADER.size + nh:]


def encode(types: MessageTypes, msg: Message) -> bytes:
    """`msg`'s frame, checked against its schema in `types` as `decode` checks it."""
    head = msg.head()
    s = types.schema(msg.type, msg.version)
    if (why := s.why(head, msg.body)) is not None:
        raise Refused(*why)
    h = canonical(head)
    if len(h) > types.max_head:
        raise Refused("oversized", f"{msg.type}'s head is {len(h)} bytes, past {types.max_head}")
    return frame(h, bytes(msg.body))


def decode(types: MessageTypes, data: bytes) -> Message:
    """The message `data` frames, parsed as adversarial against `types`: each rule broken is refused by name
    (`Refused`)."""
    h, body = split(data, types.max_head, types.max_body)
    head = parse(h)
    if type(head) is not dict or type(head.get("type")) is not str:
        raise Refused("malformed", "a head is an object with a string type")
    t, v = head["type"], head.get("v")
    if t not in types.types:
        raise Refused("unknown-type", f"{t!r} is none of {sorted(types.types)}")
    if type(v) is not int:
        raise Refused("unknown-version", f"{t}'s version is an integer, not {v!r}")
    s = types.schema(t, v)
    if (why := s.why(head, body)) is not None:
        raise Refused(*why)
    return Message(t, {k: head[k] for k in s.fields}, bytes(body), v)


class FrameReader:
    """Frames out of a byte stream: `feed` the bytes as they arrive, `next` each whole frame. A header past the bounds
    is refused as soon as it arrives (`oversized`), before its body is buffered."""

    def __init__(self, max_head: int = MAX_HEAD, max_body: int = 1 << 32) -> None:
        self.max_head, self.max_body, self._buf = max_head, max_body, bytearray()

    def feed(self, data: bytes) -> None:
        self._buf += data

    @property
    def pending(self) -> int:
        return len(self._buf)

    def next(self) -> bytes | None:
        if len(self._buf) < HEADER.size:
            return None
        nh, nb = HEADER.unpack_from(self._buf)
        if nh > self.max_head or nb > self.max_body:
            raise Refused("oversized", f"a frame is a head of at most {self.max_head} bytes and a body of at most "
                                       f"{self.max_body}, not {nh} + {nb}")
        n = HEADER.size + nh + nb
        if len(self._buf) < n:
            return None
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out


def frames(data: bytes, max_head: int = MAX_HEAD, max_body: int = 1 << 32) -> list[bytes]:
    """Every frame of `data`, which holds whole frames back to back."""
    r = FrameReader(max_head, max_body)
    r.feed(data)
    out = []
    while (f := r.next()) is not None:
        out.append(f)
    if r.pending:
        raise Refused("malformed", f"{r.pending} bytes after the last whole frame")
    return out


__all__ = ["HEADER", "WIRE_VERSION", "FrameReader", "decode", "encode", "frame", "frames", "parse", "split"]

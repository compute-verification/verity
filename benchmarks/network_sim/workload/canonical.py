"""The canonical framing of the network certifier's links that the workload frames its messages in: requests and data
in `DATA` frames on `CANONICAL_STREAM`, and responses in canonical events of at most `TOKEN_BYTES` bytes each."""

from __future__ import annotations

import base64

from verity.protocols.accounting.certifier.verifier.id_form import EVENT_END, EVENT_HEAD
from verity.protocols.accounting.certifier.verifier.link_frame import DATA, HEADER, LENGTH, encode_frame

CANONICAL_STREAM = 0
TOKEN_BYTES = 64
EVENT_TEXT = 4 * -(-TOKEN_BYTES // 3)
EVENT_TAIL = b'"},"finish_reason":null}]}' + EVENT_END
EVENT_BYTES = len(EVENT_HEAD) + EVENT_TEXT + len(EVENT_TAIL)


def event(data: bytes) -> bytes:
    """The canonical event of at most `TOKEN_BYTES` bytes: `EVENT_HEAD`, their base64 padded with NUL to `EVENT_TEXT`
    characters, `EVENT_TAIL`; `EVENT_BYTES` long whatever the data."""
    if len(data) > TOKEN_BYTES:
        raise ValueError(f"an event carries at most {TOKEN_BYTES} bytes, not {len(data)}")
    return EVENT_HEAD + base64.b64encode(data).ljust(EVENT_TEXT, b"\0") + EVENT_TAIL


def event_placement(frame_bytes: int) -> tuple[int, int, bytes, bytes]:
    """The egress template's (p, s, prefix, suffix): an event's frame is its header and `EVENT_HEAD`, the base64 text
    at p, `EVENT_TAIL` at s = p + `EVENT_TEXT`, and zero fill."""
    head = HEADER.pack(DATA, CANONICAL_STREAM, EVENT_BYTES) + EVENT_HEAD
    if len(head) + EVENT_TEXT + len(EVENT_TAIL) > frame_bytes:
        raise ValueError(f"a {frame_bytes}-byte frame has no room for an event")
    return len(head), len(head) + EVENT_TEXT, head, EVENT_TAIL


def data_frames(data: bytes, frame_bytes: int) -> bytes:
    """Bytes on a link: `DATA` frames of B − 9 bytes on `CANONICAL_STREAM`, the last zero-filled, one for no bytes."""
    room = frame_bytes - HEADER.size
    if room < 1:
        raise ValueError(f"a {frame_bytes}-byte frame has no room for data")
    return b"".join(encode_frame(DATA, CANONICAL_STREAM, data[i : i + room].ljust(room, b"\0"), frame_bytes)
                    for i in range(0, max(1, len(data)), room))


def request_frames(request: bytes, frame_bytes: int) -> bytes:
    """A request on the link: its length (u32be) and its bytes as `data_frames`."""
    if frame_bytes - HEADER.size < LENGTH.size:
        raise ValueError(f"a {frame_bytes}-byte frame has no room for a request's length")
    return data_frames(LENGTH.pack(len(request)) + request, frame_bytes)


def request_frame_count(length: int, frame_bytes: int) -> int:
    """The frames `request_frames` puts a request of `length` bytes in."""
    return -(-(LENGTH.size + length) // (frame_bytes - HEADER.size))


def request_placement(frame_bytes: int) -> tuple[int, int, bytes, bytes]:
    """The ingress template's (p, s, prefix, suffix), every `data_frames` frame's: its header, then B − 9 bytes of
    data at p = 9 (a request's length and bytes), zero past them."""
    return HEADER.size, frame_bytes, HEADER.pack(DATA, CANONICAL_STREAM, frame_bytes - HEADER.size), b""


def event_body(event: bytes) -> bytes:
    """What the proxy frames of an event: its lines after the first `id:` line (`event_position`'s), so neither the
    position nor anything before it, such as an HTTP response head, is recorded."""
    lines = event.split(b"\n")
    for i, line in enumerate(lines):
        if line.startswith(b"id:"):
            return b"\n".join(lines[i + 1 :])
    return event

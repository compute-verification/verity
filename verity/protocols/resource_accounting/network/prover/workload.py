"""The network attester's messages between the workload and the prover: the committed rows the served-rows tie reads,
and the reported advice the certifier's prover computes from the workload's scheduling. The parties are SPOT's
(`spot.prover.workload`), declared again here because this place does not load SPOT's, and the one list of every
protocol's messages is `tests/test_workload_messages.py`'s.

~~~text
workload -> prover     CommittedRows, ReportedAdvice
~~~

The forms the workload and the network attester agree on are declared here for the workload: the link frame's header
(`FRAME_HEADER`), the canonical id request (`id_request`), and a response's id events (`id_event`) and its last bytes
(`TRAILER`). The verifier keeps its own (`link_frame.HEADER`, `id_form.id_request`, `io_tie.id_event`, `io_tie.TRAILER`)
and reads nothing of this module; `test_network_attester_id_request_writers.py` holds each pair equal.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence

from verity.channel import Body, Schema
from verity.protocol import Channel, Role

__all__ = ["CHANNELS", "FRAME_HEADER", "MESSAGES", "PARTIES", "TRAILER", "id_event", "id_request"]

WORKLOAD = "workload"
PROVER = "prover"
PARTIES = (Role(WORKLOAD, "developer"), Role(PROVER, "developer"))
#: A body's bound: the wire's u32 length.
BODY = Body(max=(1 << 32) - 1)

MESSAGES = (
    Schema("CommittedRows", 1, "private", body=BODY),
    Schema("ReportedAdvice", 1, "private", body=BODY),
)

CHANNELS = (
    Channel(WORKLOAD, PROVER, ("CommittedRows", "ReportedAdvice")),
)

#: a link frame's header: kind, stream and the payload's length (the verifier's `link_frame.HEADER`)
FRAME_HEADER = struct.Struct(">BII")

_DIGITS = 10
_HEAD = (b"POST /v1/completions HTTP/1.1\r\nhost: server\r\ncontent-type: application/json\r\n"
         b"content-length:")
_NAME_BYTES = 64
_EVENT_END = b"\n\n"
_ID_EVENT_HEAD = (b'data: {"object":"chat.completion.chunk","created":0,"model":"verity","choices":[{"index":0,'
                  b'"token_id":')
_ID_EVENT_TAIL = b',"finish_reason":null}]}' + _EVENT_END
_ID_EVENT_BYTES = len(_ID_EVENT_HEAD) + _DIGITS + len(_ID_EVENT_TAIL)
#: a response's last bytes, after its id events (the verifier's `io_tie.TRAILER`)
TRAILER = b"data: [DONE]" + _EVENT_END


def _u32(x: int, what: str) -> bytes:
    if type(x) is not int or not 0 <= x < 1 << 32:
        raise ValueError(f"{what} is a u32, not {x!r}")
    return b"%*d" % (_DIGITS, x)


def _ratio(x: float, what: str) -> bytes:
    s = b"%8.6f" % x if isinstance(x, (int, float)) and not isinstance(x, bool) else b""
    if not (s and 0 <= x < 10 and float(s) == x):
        raise ValueError(f"{what} is a number in [0, 10) of at most six decimals, not {x!r}")
    return s


def _name(s: str, what: str) -> bytes:
    raw = json.dumps(s).encode() if isinstance(s, str) else b""
    if not raw or len(raw) > _NAME_BYTES:
        raise ValueError(f"{what} is a string of at most {_NAME_BYTES} bytes as JSON, not {s!r}")
    return raw.ljust(_NAME_BYTES)


def id_request(*, model: str, user: str, prompt: Sequence[int], max_tokens: int, temperature: float = 0.0,
               top_p: float = 1.0, seed: int = 0, frame_bytes: int) -> bytes:
    """The canonical `/v1/completions` request of a token-id prompt in the network attester's `frame_bytes`-byte
    cells, which carry the request's bytes alone: every field at a place fixed by the prompt's length and the frame
    size, none straddling a frame. The verifier's `id_form.id_request` at ``offset=0`` writes the same bytes and
    refuses the same fields (`test_network_attester_id_request_writers.py`)."""
    if not prompt or isinstance(prompt, (str, bytes)):
        raise ValueError("an id request's prompt is at least one token id")
    pieces = [b'{"max_tokens":' + _u32(max_tokens, "max_tokens"),
              b',"temperature":' + _ratio(temperature, "temperature"),
              b',"top_p":' + _ratio(top_p, "top_p"), b',"seed":' + _u32(seed, "a seed"),
              b',"stop":[],"stream":true', b',"model":' + _name(model, "a model"),
              b',"user":' + _name(user, "a user"), b',"prompt":',
              *((b"," if k else b"[") + _u32(t, "a prompt id") for k, t in enumerate(prompt)), b"]}"]
    room = frame_bytes - FRAME_HEADER.size
    at = len(_HEAD) + _DIGITS + 4
    if at > room or max(map(len, pieces)) > room:
        raise ValueError(f"a {frame_bytes}-byte frame has no room for an id request's head or a field")
    parts = []
    for p in pieces:
        gap = -at % room if at // room != (at + len(p) - 1) // room else 0
        parts.append(b" " * gap + p)
        at += gap + len(p)
    raw = b"".join(parts)
    return _HEAD + b"%*d" % (_DIGITS, len(raw)) + b"\r\n\r\n" + raw


def id_event(token_id: int, size: int = _ID_EVENT_BYTES) -> bytes:
    """The canonical event of one token id in `size` bytes: its head, the id right-aligned in 10 decimal characters,
    then its tail with spaces (JSON whitespace) ahead of its blank line to fill `size`. The verifier's `io_tie.id_event`
    writes the same bytes and refuses the same arguments (`test_network_attester_id_request_writers.py`)."""
    if type(token_id) is not int or not 0 <= token_id < 1 << 32:
        raise ValueError(f"a token id is a u32, not {token_id!r}")
    if type(size) is not int or size < _ID_EVENT_BYTES:
        raise ValueError(f"an id event is at least {_ID_EVENT_BYTES} bytes, not {size!r}")
    tail = _ID_EVENT_TAIL[:-len(_EVENT_END)] + b" " * (size - _ID_EVENT_BYTES) + _EVENT_END
    return _ID_EVENT_HEAD + b"%10d" % token_id + tail

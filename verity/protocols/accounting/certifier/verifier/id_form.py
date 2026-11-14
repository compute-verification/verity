"""The id form's fixed bytes: the canonical event's head and end, which an id event's are cut from, and the canonical
id request's head and the places of its first fields, with the readers of a field and of `max_tokens`, and
`id_request`, which writes the request these places are of."""

from __future__ import annotations

import json
from collections.abc import Sequence

from verity.protocols.accounting.certifier.verifier.link_frame import HEADER, LENGTH

EVENT_END = b"\n\n"
EVENT_HEAD = (b'data: {"object":"chat.completion.chunk","created":0,"model":"verity","choices":[{"index":0,'
              b'"delta":{"base64":"')
ID_DIGITS = 10
#: an id request's HTTP head, up to its length (`id_request`)
ID_REQUEST_HEAD = (b"POST /v1/completions HTTP/1.1\r\nhost: server\r\ncontent-type: application/json\r\n"
                   b"content-length:")
#: the room of an id request's `model` and `user`, each its JSON string padded with spaces
ID_NAME_BYTES = 64
#: a prompt id's place in an id request: `[` or `,`, then the id in `ID_DIGITS` characters
ID_FIELD_BYTES = 1 + ID_DIGITS
#: the body's offset in an id request at offset 0: `ID_REQUEST_HEAD`, the content-length in `ID_DIGITS`, CRLF CRLF
ID_BODY_AT = len(ID_REQUEST_HEAD) + ID_DIGITS + 4
#: the body's first field, `max_tokens`, its value in `ID_DIGITS` characters at `MAX_TOKENS_AT`
MAX_TOKENS_KEY = b'{"max_tokens":'
MAX_TOKENS_AT = ID_BODY_AT + len(MAX_TOKENS_KEY)


def u32_field(field: bytes) -> int | None:
    """The u32 an id request's `ID_DIGITS` characters hold as `_u32_field` writes one (spaces, then decimal digits
    without a leading zero, below 2^32), or None for any other bytes."""
    s = field.lstrip(b" ")
    if len(field) == ID_DIGITS and s.isdigit() and (s == b"0" or s[:1] != b"0") and int(s) < 1 << 32:
        return int(s)
    return None


def id_request_max_tokens(head: bytes) -> int | None:
    """The `max_tokens` of an id request at offset 0 (the network certifier's cells), read from its first bytes at the
    place `id_request` fixes for it: `ID_REQUEST_HEAD`, a u32 content-length, CRLF CRLF and `MAX_TOKENS_KEY`, then the
    u32; None where any of them isn't there."""
    if head[:len(ID_REQUEST_HEAD)] != ID_REQUEST_HEAD or \
            u32_field(head[len(ID_REQUEST_HEAD):len(ID_REQUEST_HEAD) + ID_DIGITS]) is None or \
            head[ID_BODY_AT - 4:MAX_TOKENS_AT] != b"\r\n\r\n" + MAX_TOKENS_KEY:
        return None
    return u32_field(head[MAX_TOKENS_AT:MAX_TOKENS_AT + ID_DIGITS])


def _u32_field(x: int, what: str) -> bytes:
    if type(x) is not int or not 0 <= x < 1 << 32:
        raise ValueError(f"{what} is a u32, not {x!r}")
    return b"%*d" % (ID_DIGITS, x)


def _ratio_field(x: float, what: str) -> bytes:
    s = b"%8.6f" % x if isinstance(x, (int, float)) and not isinstance(x, bool) else b""
    if not (s and 0 <= x < 10 and float(s) == x):
        raise ValueError(f"{what} is a number in [0, 10) of at most six decimals, not {x!r}")
    return s


def _name_field(s: str, what: str) -> bytes:
    raw = json.dumps(s).encode() if isinstance(s, str) else b""
    if not raw or len(raw) > ID_NAME_BYTES:
        raise ValueError(f"{what} is a string of at most {ID_NAME_BYTES} bytes as JSON, not {s!r}")
    return raw.ljust(ID_NAME_BYTES)


def id_request(*, model: str, user: str, prompt: Sequence[int], max_tokens: int, temperature: float = 0.0,
               top_p: float = 1.0, seed: int = 0, frame_bytes: int, offset: int = LENGTH.size) -> bytes:
    """A canonical `/v1/completions` request of a token-id prompt, whose every field is at a place fixed by the
    prompt's length and the frame size alone, in the link frames `request_frames` puts it in: `ID_REQUEST_HEAD`, its
    body's length in `ID_DIGITS` characters (HTTP's optional whitespace before it), in the first frame, then the body,
    streamed, with no stop strings: `max_tokens` and `seed` in `ID_DIGITS` characters, `temperature` and `top_p` in
    eight (six decimals), `model` and `user` each in `ID_NAME_BYTES`, and each prompt id in its own `ID_FIELD_BYTES`.
    No field straddles a frame: spaces (JSON's whitespace) before one that would move it, with its key, to the next.
    At 4096-byte frames every field but the prompt's is in the first frame. `offset` is the request's first byte's
    place in its first frame's payload: LENGTH's 4 bytes on the active certifier's link, 0 in the network certifier's
    cells, which carry the bytes alone."""
    if offset not in (0, LENGTH.size):
        raise ValueError(f"an id request starts its first frame's payload at 0 or {LENGTH.size}, not {offset!r}")
    if not prompt or isinstance(prompt, (str, bytes)):
        raise ValueError("an id request's prompt is at least one token id")
    pieces = [b'{"max_tokens":' + _u32_field(max_tokens, "max_tokens"),
              b',"temperature":' + _ratio_field(temperature, "temperature"),
              b',"top_p":' + _ratio_field(top_p, "top_p"), b',"seed":' + _u32_field(seed, "a seed"),
              b',"stop":[],"stream":true', b',"model":' + _name_field(model, "a model"),
              b',"user":' + _name_field(user, "a user"), b',"prompt":',
              *((b"," if k else b"[") + _u32_field(t, "a prompt id") for k, t in enumerate(prompt)), b"]}"]
    room = frame_bytes - HEADER.size
    at = offset + len(ID_REQUEST_HEAD) + ID_DIGITS + 4
    if at > room or max(map(len, pieces)) > room:
        raise ValueError(f"a {frame_bytes}-byte frame has no room for an id request's head or a field")
    parts = []
    for p in pieces:
        gap = -at % room if at // room != (at + len(p) - 1) // room else 0
        parts.append(b" " * gap + p)
        at += gap + len(p)
    raw = b"".join(parts)
    return ID_REQUEST_HEAD + b"%*d" % (ID_DIGITS, len(raw)) + b"\r\n\r\n" + raw

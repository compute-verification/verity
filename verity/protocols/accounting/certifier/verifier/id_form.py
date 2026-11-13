"""The id form's fixed bytes: the canonical event's head and end, which an id event's are cut from, and the canonical
id request's head and the places of its first fields, with the readers of a field and of `max_tokens`. `active.id_request`
writes the request these places are of."""

from __future__ import annotations

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

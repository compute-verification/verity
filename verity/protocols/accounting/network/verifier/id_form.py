"""The places of the canonical id request's first fields (`verity.release.id_request`, which writes it), with the
readers of a field and of `max_tokens`."""

from __future__ import annotations

from verity.release import EVENT_END, EVENT_HEAD, ID_DIGITS, ID_NAME_BYTES, ID_REQUEST_HEAD, id_request

#: a prompt id's place in an id request: `[` or `,`, then the id in `ID_DIGITS` characters
ID_FIELD_BYTES = 1 + ID_DIGITS
#: the body's offset in an id request at offset 0: `ID_REQUEST_HEAD`, the content-length in `ID_DIGITS`, CRLF CRLF
ID_BODY_AT = len(ID_REQUEST_HEAD) + ID_DIGITS + 4
#: the body's first field, `max_tokens`, its value in `ID_DIGITS` characters at `MAX_TOKENS_AT`
MAX_TOKENS_KEY = b'{"max_tokens":'
MAX_TOKENS_AT = ID_BODY_AT + len(MAX_TOKENS_KEY)


def u32_field(field: bytes) -> int | None:
    """The u32 an id request's `ID_DIGITS` characters hold as `verity.release.id_request` writes one (spaces, then decimal digits
    without a leading zero, below 2^32), or None for any other bytes."""
    s = field.lstrip(b" ")
    if len(field) == ID_DIGITS and s.isdigit() and (s == b"0" or s[:1] != b"0") and int(s) < 1 << 32:
        return int(s)
    return None


def id_request_max_tokens(head: bytes) -> int | None:
    """The `max_tokens` of an id request at offset 0 (the network attester's cells), read from its first bytes at the
    place `id_request` fixes for it: `ID_REQUEST_HEAD`, a u32 content-length, CRLF CRLF and `MAX_TOKENS_KEY`, then the
    u32; None where any of them isn't there."""
    if head[:len(ID_REQUEST_HEAD)] != ID_REQUEST_HEAD or \
            u32_field(head[len(ID_REQUEST_HEAD):len(ID_REQUEST_HEAD) + ID_DIGITS]) is None or \
            head[ID_BODY_AT - 4:MAX_TOKENS_AT] != b"\r\n\r\n" + MAX_TOKENS_KEY:
        return None
    return u32_field(head[MAX_TOKENS_AT:MAX_TOKENS_AT + ID_DIGITS])


__all__ = ["EVENT_END", "EVENT_HEAD", "ID_BODY_AT", "ID_DIGITS", "ID_FIELD_BYTES", "ID_NAME_BYTES", "ID_REQUEST_HEAD",
           "MAX_TOKENS_AT", "MAX_TOKENS_KEY", "id_request", "id_request_max_tokens", "u32_field"]

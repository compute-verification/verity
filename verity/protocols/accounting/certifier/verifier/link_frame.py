"""The link's frame: its header, the kinds both certifiers' frames have, and its codec. A link frame is B bytes: the
header (kind, stream, the payload's length), the payload, then zero fill. The network certifier's cells are such
frames."""

from __future__ import annotations

import struct

HEADER = struct.Struct(">BII")
PAD, DATA, SALT = 0, 1, 3


def encode_frame(kind: int, stream: int, payload: bytes, frame_bytes: int) -> bytes:
    """One B-byte link frame: the header, the payload, then zero fill."""
    if HEADER.size + len(payload) > frame_bytes:
        raise ValueError(f"a {len(payload)}-byte payload doesn't fit a {frame_bytes}-byte frame")
    return HEADER.pack(kind, stream, len(payload)) + payload + bytes(frame_bytes - HEADER.size - len(payload))


def decode_frame(frame: bytes) -> tuple[int, int, bytes]:
    """(kind, stream, payload) of a link frame; padding is (`PAD`, 0, b"")."""
    kind, stream, n = HEADER.unpack_from(frame)
    return kind, stream, bytes(frame[HEADER.size : HEADER.size + n])

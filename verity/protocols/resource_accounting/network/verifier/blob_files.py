"""A file of blobs, each behind its length: what a certifier's records and their salts are kept in on disk."""

from __future__ import annotations

import struct
from pathlib import Path


def write_blobs(path: Path, blob: bytes) -> None:
    with open(path, "ab") as fh:
        fh.write(struct.pack(">I", len(blob)) + blob)


def read_blobs(path: Path) -> list[bytes]:
    data, out, i = path.read_bytes() if path.exists() else b"", [], 0
    while i < len(data):
        (n,) = struct.unpack_from(">I", data, i)
        out.append(data[i + 4 : i + 4 + n])
        i += 4 + n
    return out

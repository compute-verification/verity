"""What one `json.dumps` call writes, encoded a bounded number of list items per call.

A served window's binding map has up to about a million entries, and it is encoded and hashed on the seal's thread beside the
serving engine: one `json.dumps` over it is one C call that holds the GIL for seconds, and the engine's thread waits that long
for its next step. Encoded `CHUNK` items at a time, the engine's thread
takes the GIL between them. The pieces concatenate to the bytes the one call writes.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

CHUNK = 1024


def _separators(kw: Mapping[str, Any]) -> tuple[str, str]:
    if kw.get("indent") is not None:
        raise ValueError("json_chunks: no indent (an indented list's items are not encoded independently)")
    return tuple(kw.get("separators") or (", ", ": "))  # type: ignore[return-value]


def list_pieces(items: Sequence[Any], *, chunk: int | None = None, **kw: Any) -> Iterator[str]:
    """``json.dumps(list(items), **kw)`` in pieces, ``chunk`` (default `CHUNK`) items a piece."""
    item, _ = _separators(kw)
    enc = json.JSONEncoder(**kw).encode
    chunk = int(chunk or CHUNK)
    yield "["
    for i in range(0, len(items), chunk):
        yield (item if i else "") + item.join(enc(x) for x in items[i:i + chunk])
    yield "]"


def dict_pieces(doc: Mapping[str, Any], key: str, *, chunk: int | None = None, **kw: Any) -> Iterator[str]:
    """``json.dumps({k: v for k, v in doc.items() if k != key} | {key: doc[key]}, **kw)`` in pieces: the other keys whole, then
    ``key``'s list by `list_pieces`. No ``sort_keys`` (``key`` comes last)."""
    if kw.get("sort_keys"):
        raise ValueError("json_chunks: dict_pieces writes its key last, which sort_keys would move")
    item, colon = _separators(kw)
    enc = json.JSONEncoder(**kw).encode
    head = {k: v for k, v in doc.items() if k != key}
    yield (enc(head)[:-1] + item if head else "{") + enc(key) + colon
    yield from list_pieces(doc[key], chunk=chunk, **kw)
    yield "}"


def sha256_hex(pieces: Iterable[str]) -> str:
    """The sha256 of the pieces' UTF-8 bytes, back to back."""
    h = hashlib.sha256()
    for p in pieces:
        h.update(p.encode())
    return h.hexdigest()

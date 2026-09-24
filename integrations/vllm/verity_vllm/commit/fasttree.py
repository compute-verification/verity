"""Packed-identity Merkle construction with the frozen ``verity-vllm`` framing and a pluggable hash backend.

Same preimages as :mod:`verity_vllm.commit.hashing` (so the SHA-256 backend reproduces the production roots bit for bit):

    leaf = h( u32be(len(tag)) || tag || u64be(len(id)) || id || u64be(len(dtype)) || dtype || u64be(len(shape)) || shape || u64be(len(v)) || v )
    node = h( u32be(len(tag)) || tag || u64be(32) || left || u64be(32) || right )
    lift = h( u32be(len(tag)) || tag || u64be(32) || child )

"Packed identity": the per-leaf header (everything before the value) is built ONCE from the pre-resolved
(canonical_id, dtype, shape) triple and kept as bytes; the hot path is one ``h(header + len || value)`` per leaf
with no regex parse, no dict of ids, and leaves already in tree order.  The fold is a list-of-bytes loop that calls
the backend directly.  ``memoryview`` values are accepted (zero-copy from a memmap).

Backends: ``sha256`` (hashlib/OpenSSL), ``blake3`` (the ``blake3`` wheel, single-threaded per call),
``noop`` (builds the framing bytes, returns a constant 32-byte digest: the "surrounding pipeline intact, no hash
arithmetic" ablation).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

from verity_vllm.commit import hashing

ZERO32 = bytes(32)
U64_32 = (32).to_bytes(8, "big")


def _tag_prefix(tag: str) -> bytes:
    t = tag.encode("utf-8")
    return len(t).to_bytes(4, "big") + t


LEAF_PREFIX = _tag_prefix(hashing.LEAF_TAG)
NODE_PREFIX = _tag_prefix(hashing.NODE_TAG) + U64_32
LIFT_PREFIX = _tag_prefix(hashing.LIFT_TAG) + U64_32


class _Noop:
    """hashlib-shaped object that consumes bytes without hashing them."""

    __slots__ = ("n",)

    def __init__(self, data: bytes | memoryview = b"") -> None:
        self.n = len(data)

    def update(self, data: bytes | memoryview) -> None:
        self.n += len(data)

    def digest(self) -> bytes:
        return ZERO32


def backend(name: str) -> Callable[..., Any]:
    if name == "sha256":
        return hashlib.sha256
    if name == "noop":
        return _Noop
    if name == "blake3":
        import blake3

        def _b3(data: bytes | memoryview = b"") -> Any:
            return blake3.blake3(bytes(data) if isinstance(data, memoryview) else data, max_threads=1)

        return _b3
    raise ValueError(name)


def leaf_header(canonical_id: str, dtype: str, shape: Sequence[int]) -> bytes:
    c = canonical_id.encode("utf-8")
    d = dtype.encode("utf-8")
    s = hashing.shape_text(shape).encode("utf-8")
    return LEAF_PREFIX + len(c).to_bytes(8, "big") + c + len(d).to_bytes(8, "big") + d + len(s).to_bytes(8, "big") + s


def leaf_digest(header: bytes, value: bytes | memoryview, new: Callable[..., Any] = hashlib.sha256) -> bytes:
    h = new(header)
    h.update(len(value).to_bytes(8, "big"))
    h.update(value)
    return h.digest()


def fold(level: list[bytes], new: Callable[..., Any] = hashlib.sha256) -> bytes:
    """Fold leaf digests (already in tree order) to the root; identical shape rule to ``MerkleTree``."""
    if not level:
        return hashing.empty_root()
    while len(level) > 1:
        nxt = [new(NODE_PREFIX + level[i] + U64_32 + level[i + 1]).digest() for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(new(LIFT_PREFIX + level[-1]).digest())
        level = nxt
    return level[0]


def fold_levels(level: list[bytes], new: Callable[..., Any] = hashlib.sha256) -> list[list[bytes]]:
    levels = [level]
    while len(level) > 1:
        nxt = [new(NODE_PREFIX + level[i] + U64_32 + level[i + 1]).digest() for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(new(LIFT_PREFIX + level[-1]).digest())
        level = nxt
        levels.append(level)
    return levels


def tree_root(headers: Sequence[bytes], values: Sequence[bytes | memoryview], new: Callable[..., Any] = hashlib.sha256) -> bytes:
    """headers/values in tree order -> root (one leaf hash per value, then the fold)."""
    return fold([leaf_digest(h, v, new) for h, v in zip(headers, values, strict=True)], new)


def chunk_layout(tree_id: str, buf: bytes | memoryview, chunk: int, dtype: str = "uint8") -> tuple[list[bytes], list[memoryview]]:
    """Contiguous ``chunk``-byte leaves over one buffer; leaf id ``<tree_id>/chunk=<k>`` binds the offset range."""
    mv = memoryview(buf)
    headers, values = [], []
    k = 0
    for off in range(0, len(mv), chunk):
        v = mv[off : off + chunk]
        headers.append(leaf_header(f"{tree_id}/chunk={k}", dtype, (len(v),)))
        values.append(v)
        k += 1
    return headers, values


def opening_bytes(leaf_count: int, leaf_nbytes: int, id_nbytes: int = 0) -> dict[str, int]:
    """Authentication bytes to open one leaf: the leaf bytes + one 32-byte sibling per level (lifts carry none)."""
    depth = 0
    w = leaf_count
    while w > 1:
        depth += 1
        w = (w + 1) // 2
    return {"depth": depth, "path_bytes_max": 32 * depth, "leaf_bytes": leaf_nbytes, "id_bytes": id_nbytes,
            "total_max": 32 * depth + leaf_nbytes + id_nbytes}

"""The one hash function of the POC and its tags.

Byte layout of ``H(tag, *parts)`` (SHA-256 over a single stream)::

    u32be(len(tag)) || tag || for each part: u64be(len(part)) || part

- ``tag`` is the ASCII/UTF-8 bytes of a tag string such as ``"verity-vllm/leaf/v1"``.
- ``u32be`` is a 4-byte big-endian unsigned length; ``u64be`` an 8-byte one.
- Parts are raw bytes.  Strings are UTF-8 encoded before they become parts.
- There is no frame prefix other than the tag; the tag is the domain separator.

The tags and their parts (all frozen, contract section 5):

==============================  ==============================================================
tag                             parts, in order
==============================  ==============================================================
``verity-vllm/leaf/v1``         canonical_id (UTF-8), dtype (UTF-8), shape text (UTF-8), value
``verity-vllm/node/v1``         left (32 bytes), right (32 bytes)
``verity-vllm/lift/v1``         child (32 bytes) -- an odd single child, never duplicated
``verity-vllm/ids/v1``          the sorted canonical ids of a tree, one UTF-8 part each
``verity-vllm/empty/v1``        (no parts) -- the root of a tree with zero leaves
``verity-vllm/json/v1``         canonical JSON bytes of a document (manifest, description...)
``verity-vllm/challenge/v1``    the bound digests and the sample of a challenge (see relations)
==============================  ==============================================================

*Shape text* is ``"[" + ",".join(str(d) for d in shape) + "]"`` with no spaces; a scalar is
``"[]"``.  *dtype* is the torch dtype name without the ``torch.`` prefix (``"bfloat16"``), see
:mod:`verity_vllm.dtypes`.  *value* is the exact little-endian device bytes of the tensor in
C order: BF16 is 2 bytes per element, FP8 1 byte, packed INT4 exactly as stored.  No conversion
before hashing.

Reference implementation in Python (what W1 can copy verbatim)::

    def H(tag: str, *parts: bytes) -> bytes:
        h = hashlib.sha256()
        t = tag.encode("utf-8")
        h.update(len(t).to_bytes(4, "big")); h.update(t)
        for p in parts:
            h.update(len(p).to_bytes(8, "big")); h.update(p)
        return h.digest()
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

LEAF_TAG = "verity-vllm/leaf/v1"
NODE_TAG = "verity-vllm/node/v1"
LIFT_TAG = "verity-vllm/lift/v1"
IDS_TAG = "verity-vllm/ids/v1"
EMPTY_TAG = "verity-vllm/empty/v1"
JSON_TAG = "verity-vllm/json/v1"
CHALLENGE_TAG = "verity-vllm/challenge/v1"

DIGEST_BYTES = 32

FRAME_DESCRIPTION = "sha256(u32be(len(tag)) || tag || (u64be(len(part)) || part)*)"


def H(tag: str, *parts: bytes) -> bytes:  # noqa: N802 - the contract names it H
    """SHA-256 over the tag and the length-prefixed parts (layout in the module docstring)."""
    digest = hashlib.sha256()
    encoded_tag = tag.encode("utf-8")
    digest.update(len(encoded_tag).to_bytes(4, "big"))
    digest.update(encoded_tag)
    for part in parts:
        if not isinstance(part, bytes | bytearray | memoryview):
            raise TypeError(f"hash parts must be bytes, got {type(part).__name__}")
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.digest()


def shape_text(shape: Sequence[int]) -> str:
    """The canonical textual shape used inside the leaf hash: ``[4,8]``, scalar ``[]``."""
    dims = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError(f"shape dims must be non-negative ints, got {shape!r}")
        dims.append(str(dim))
    return "[" + ",".join(dims) + "]"


def parse_shape_text(text: str) -> tuple[int, ...]:
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"malformed shape text {text!r}")
    inner = text[1:-1]
    if inner == "":
        return ()
    dims = []
    for piece in inner.split(","):
        if not piece.isdigit() or (len(piece) > 1 and piece[0] == "0"):
            raise ValueError(f"malformed shape text {text!r}")
        dims.append(int(piece))
    return tuple(dims)


def leaf_hash(canonical_id: str, dtype: str, shape: Sequence[int], value: bytes) -> bytes:
    """``H("verity-vllm/leaf/v1", canonical_id, dtype, shape, raw little-endian bytes)``."""
    return H(
        LEAF_TAG,
        canonical_id.encode("utf-8"),
        dtype.encode("utf-8"),
        shape_text(shape).encode("utf-8"),
        value,
    )


def node_hash(left: bytes, right: bytes) -> bytes:
    _check_digest(left, "left")
    _check_digest(right, "right")
    return H(NODE_TAG, left, right)


def lift_hash(child: bytes) -> bytes:
    _check_digest(child, "child")
    return H(LIFT_TAG, child)


def ids_digest(sorted_ids: Iterable[str]) -> bytes:
    """Digest of a tree's canonical ids in tree (code-point) order, so an opening cannot be
    replayed under another tree with the same root width."""
    return H(IDS_TAG, *(cid.encode("utf-8") for cid in sorted_ids))


def empty_root() -> bytes:
    return H(EMPTY_TAG)


def canonical_json_bytes(document: Any) -> bytes:
    """The one JSON serialisation whose sha256 defines ``profile_id`` and document digests:
    sorted keys, no whitespace, ASCII-escaped, UTF-8, no trailing newline."""
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def json_digest(document: Any) -> bytes:
    """``H("verity-vllm/json/v1", canonical_json_bytes(document))``."""
    return H(JSON_TAG, canonical_json_bytes(document))


def profile_id(manifest: Any) -> str:
    """First 12 hex of the sha256 of the manifest's canonical JSON bytes (contract section 2.1).

    The capture writer stores ``manifest.json`` as exactly those bytes so that
    ``sha256(file) == sha256(canonical_json_bytes(manifest))``.
    """
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()[:12]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_digest(value: bytes, name: str) -> None:
    if not isinstance(value, bytes | bytearray | memoryview) or len(value) != DIGEST_BYTES:
        raise ValueError(f"{name} must be a 32-byte digest")

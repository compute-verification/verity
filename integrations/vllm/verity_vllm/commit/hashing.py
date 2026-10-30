"""The one hash function of the POC and its tags.

``H``, the leaf/node/lift/ids/empty tags and their rules are core's (:mod:`verity.ml.commitments.leaves`); :func:`ids_digest`
here hashes the ids in the order given (callers pass them already in tree order).

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
``verity-vllm/challenge/v1``    the bound digests and the sample of a challenge
==============================  ==============================================================

*Shape text* is ``"[" + ",".join(str(d) for d in shape) + "]"`` with no spaces; a scalar is
``"[]"``.  *dtype* is the torch dtype name without the ``torch.`` prefix (``"bfloat16"``).  *value*
is the exact little-endian device bytes of the tensor in C order: BF16 is 2 bytes per element, FP8
1 byte, packed INT4 exactly as stored.  No conversion before hashing.

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
from collections.abc import Iterable
from typing import Any

from verity.ml.commitments.leaves import (
    DIGEST_BYTES,
    EMPTY_TAG,
    IDS_TAG,
    LEAF_TAG,
    LIFT_TAG,
    NODE_TAG,
    H,
    empty_root,
    leaf_hash,
    lift_hash,
    node_hash,
    parse_shape_text,
    shape_text,
)

__all__ = ["CHALLENGE_TAG", "DIGEST_BYTES", "EMPTY_TAG", "FRAME_DESCRIPTION", "IDS_TAG", "JSON_TAG", "LEAF_TAG", "LIFT_TAG", "NODE_TAG", "H",
           "canonical_json_bytes", "empty_root", "ids_digest", "json_digest", "leaf_hash", "lift_hash", "node_hash", "parse_shape_text",
           "profile_id", "sha256_hex", "shape_text"]

JSON_TAG = "verity-vllm/json/v1"
CHALLENGE_TAG = "verity-vllm/challenge/v1"

FRAME_DESCRIPTION = "sha256(u32be(len(tag)) || tag || (u64be(len(part)) || part)*)"


def ids_digest(sorted_ids: Iterable[str]) -> bytes:
    """Digest of a tree's canonical ids in tree (code-point) order, so an opening cannot be
    replayed under another tree with the same root width."""
    return H(IDS_TAG, *(cid.encode("utf-8") for cid in sorted_ids))


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

"""The commitment engine this prototype REUSES (imported, not forked) and the small pieces built on top of it.

From `verity_vllm` (production + C0/C1 commitment engine):
  * `commit.path_shape / commit.fold_path`         -- the production opening verifier's core (shape rule + fold)
  * `commit_bench.fasttree` (`fold`, `fold_levels`, `NODE_PREFIX`, `LIFT_PREFIX`) -- C1 packed fold, production node/lift framing
  * `commit_bench.semantic.pos_leaf`               -- the C0 position-identified leaf: SHA-256("verity/pos-leaf/v0" || u64 nbytes || value)
  * `commit_bench.stream.StreamingMerkle`          -- O(log n) streaming frontier with a tracked authentication path (replay-to-open)

Added here: the context-binding root wrapper (the analogue of C0's `semantic_root`), a hash-call counter for the cost
tables, and `path_in_levels` (production `MerkleTree.path` over already-folded levels).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

_POC = Path(__file__).resolve().parents[2] / "vllm-poc"
if _POC.is_dir() and str(_POC) not in sys.path:  # the engine lives in the vllm-poc tree
    sys.path.insert(0, str(_POC))

from verity_vllm.commit.merkle import fold_path, path_shape, tree_depth  # noqa: E402
from verity_vllm.commit import fasttree as ft  # noqa: E402
from verity_vllm.commit.semantic_layout import POS_LEAF_TAG, pos_leaf  # noqa: E402
from verity_vllm.commit.stream_merkle import StreamingMerkle  # noqa: E402

__all__ = ["fold_path", "path_shape", "tree_depth", "ft", "pos_leaf", "POS_LEAF_TAG", "StreamingMerkle",
           "ROOT_TAG", "bind_root", "CountingHash", "path_in_levels", "sha_blocks", "LEAF_FRAME_BYTES", "NODE_PREIMAGE_BYTES"]

ROOT_TAG = b"verity/fa2c/root/v0"
LEAF_FRAME_BYTES = len(POS_LEAF_TAG) + 8          # 26 B of framing around the value bytes
NODE_PREIMAGE_BYTES = len(ft.NODE_PREFIX) + 32 + 8 + 32   # 103 B (2 SHA blocks)
LIFT_PREIMAGE_BYTES = len(ft.LIFT_PREFIX) + 32     # 63 B (1 block)


def sha_blocks(message_bytes: int) -> int:
    """SHA-256 compressions for a message of that length (padding + 8-byte length)."""
    return (message_bytes + 8) // 64 + 1


def layout_digest(kind: str, params: dict[str, Any]) -> bytes:
    return hashlib.sha256(json.dumps({"kind": kind, **params}, sort_keys=True, separators=(",", ":")).encode()).digest()


def bind_root(program_digest: bytes, ctx_digest: bytes, geo_digest: bytes, layout_dig: bytes, n_words: int, merkle_root: bytes) -> bytes:
    """R = SHA-256(ROOT_TAG || program_digest || ctx_digest || geo_digest || layout_digest || u64 N || merkle_root).

    Same shape as C0's `semantic_root` (program, static template, per-record context, size, tree root): a root cannot be
    replayed for another Program, request, step, layer, geometry or layout, and N pins the tree shape."""
    for d in (program_digest, ctx_digest, geo_digest, layout_dig, merkle_root):
        if len(d) != 32:
            raise ValueError("32-byte digests expected")
    return hashlib.sha256(ROOT_TAG + program_digest + ctx_digest + geo_digest + layout_dig + n_words.to_bytes(8, "big") + merkle_root).digest()


class CountingHash:
    """hashlib-compatible constructor that counts digests and bytes (for the cost tables); wraps `hashlib.sha256`."""

    def __init__(self) -> None:
        self.calls = 0
        self.bytes = 0
        self.blocks = 0

    def __call__(self, data: bytes | memoryview = b"") -> Any:
        return _Counted(self, data)

    def reset(self) -> None:
        self.calls = self.bytes = self.blocks = 0


class _Counted:
    __slots__ = ("c", "h", "n")

    def __init__(self, counter: CountingHash, data: bytes | memoryview) -> None:
        self.c, self.h, self.n = counter, hashlib.sha256(data), len(data)

    def update(self, data: bytes | memoryview) -> None:
        self.h.update(data)
        self.n += len(data)

    def digest(self) -> bytes:
        self.c.calls += 1
        self.c.bytes += self.n
        self.c.blocks += sha_blocks(self.n)
        return self.h.digest()


def path_in_levels(levels: Sequence[Sequence[bytes]], index: int) -> tuple[bytes | None, ...]:
    """Sibling per level for leaf ``index`` given the folded levels (`fasttree.fold_levels`); production `MerkleTree.path`."""
    out: list[bytes | None] = []
    cursor = index
    for level in levels[:-1]:
        if cursor == len(level) - 1 and len(level) % 2 == 1:
            out.append(None)
        else:
            out.append(level[cursor ^ 1])
        cursor //= 2
    return tuple(out)


def fold_digests(digests: Sequence[bytes], new: Callable[..., Any] = hashlib.sha256) -> bytes:
    """Root over already-hashed children (subtree roots folded with the production node/lift rule)."""
    return ft.fold(list(digests), new)

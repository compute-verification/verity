"""The commitment engine the hidden-state committers REUSE (imported, not forked) and the small pieces built on top of it.

From `verity_vllm` (production + C0/C1 commitment engine):
  * `commit.path_shape / commit.fold_path`         -- the production opening verifier's core (shape rule + fold)
  * `commit_bench.fasttree` (`fold`, `fold_levels`, `NODE_PREFIX`, `LIFT_PREFIX`) -- C1 packed fold, production node/lift framing
  * `commit_bench.semantic.pos_leaf`               -- the C0 position-identified leaf: SHA-256("verity/pos-leaf/v0" || u64 nbytes || value)

Added here: the context-binding root wrapper (the analogue of C0's `semantic_root`) and `path_in_levels` (production
`MerkleTree.path` over already-folded levels).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity_vllm.commit.merkle import fold_path, path_shape
from verity_vllm.commit import fasttree as ft
from verity_vllm.commit.semantic_layout import pos_leaf

__all__ = ["fold_path", "path_shape", "ft", "pos_leaf", "ROOT_TAG", "bind_root", "path_in_levels"]

ROOT_TAG = b"verity/fa2c/root/v0"


def bind_root(program_digest: bytes, ctx_digest: bytes, geo_digest: bytes, layout_dig: bytes, n_words: int, merkle_root: bytes) -> bytes:
    """R = SHA-256(ROOT_TAG || program_digest || ctx_digest || geo_digest || layout_digest || u64 N || merkle_root).

    Same shape as C0's `semantic_root` (program, static template, per-record context, size, tree root): a root cannot be
    replayed for another Program, request, step, layer, geometry or layout, and N pins the tree shape."""
    for d in (program_digest, ctx_digest, geo_digest, layout_dig, merkle_root):
        if len(d) != 32:
            raise ValueError("32-byte digests expected")
    return hashlib.sha256(ROOT_TAG + program_digest + ctx_digest + geo_digest + layout_dig + n_words.to_bytes(8, "big") + merkle_root).digest()


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

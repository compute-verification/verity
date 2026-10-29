"""The lead's commitment of one call's words and of its A (PoUW v2, ``blc4a_t64_e7_s1``): the bytes each leaf holds, the
inline frame-v3-sha512 leaves and the subtree below each instance's keyed root.  ``native_rows.cu``'s ``lead_*`` kernels hash
the same bytes (``tests/commit/test_lead_rows.py`` holds this module to ``ssn.cu``'s export layout and to core's ``merkle``).

**The committed words.**  The served GEMM exports, at each of its S k32 steps, a 64 x BN block of s32 words a (row tile rt,
column tile ct, step t), block ``(rt * N / BN + ct) * S + t`` of the export, in wgmma's fragment order: word (i, j, t) is at
``ni * 512 + w * 128 + g * 16 + q * 4 + eh * 2 + el`` of its block, with ``i = 64 rt + 16 w + 8 eh + g`` and
``j = BN ct + 8 ni + 2 q + el``.  So each 16-row x 8-column MMA tile (ni, w) of a block is 128 contiguous words, 512 bytes.  A
call of m rows and n columns commits the tiles of its ``pad_rows(m)`` x ``pad_cols(n)`` words (``layout.h``; Lean's
``padRows``, ``padCols``), and the kernel's other rows and columns are computed and not committed.  The committed stream is the
export's tiles in their order (rt, ct, t, ni, w) with every uncommitted tile skipped, each word 4 bytes little-endian as the
kernel stores it: ``pad_rows(m) * pad_cols(n) * S`` words.

**The leaves.**  A call's stream is cut into ``CHUNK_WORDS``-word chunks (256 KiB; the last is short), and A_j's row-major bytes
(its m served rows of K int8) into ``A_SEGMENT``-byte segments (the last is short).  Each value is an inline frame-v3-sha512
leaf at its own position, rank = position, ``dom.leaf(r, r, schema, value)``: its first block holds the rank, so equal values at
two positions share no compression.  An instance's leaves are core's ``tree_levels`` under ``instance_domain``.  A forward's
calls concatenate their chunks into one words instance, chunks never spanning calls (``words_values``).

**The keyed root** replaces the subtree's root as the instance's public digest; :func:`keyed_root` is its stub.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels

__all__ = ["A_SCHEMA", "A_SEGMENT", "BINDING_TAG", "CHUNK_WORDS", "ROW_TILE", "TILE_COLS", "TILE_ROWS", "TILE_WORDS",
           "WORDS_SCHEMA", "WORDS_POSITION", "Subtree", "a_position", "a_segments", "chunks", "committed_tiles",
           "instance_domain", "keyed_root", "pad_cols", "pad_rows", "stream", "subtree", "words_position", "words_values"]

TILE_ROWS, TILE_COLS = 16, 8
TILE_WORDS = TILE_ROWS * TILE_COLS
ROW_TILE = 64
CHUNK_WORDS = 65536
A_SEGMENT = 1024
WORDS_SCHEMA = "pouw/lead/words/v1"
A_SCHEMA = "pouw/lead/a/v1"
BINDING_TAG = b"pouw/lead/instance/v1"
WORDS_POSITION = 1 << 63


def pad_rows(m: int) -> int:
    return TILE_ROWS * -(-m // TILE_ROWS)


def pad_cols(n: int) -> int:
    return TILE_COLS * -(-n // TILE_COLS)


def _tile_mask(m: int, n: int, S: int, bn: int) -> np.ndarray:
    """Over the export's tiles (rt, ct, t, ni, w): whether each is committed."""
    if m <= 0 or n <= 0 or S <= 0 or bn <= 0 or bn % TILE_COLS:
        raise ValueError(f"a call of {m} x {n} at {S} steps over {bn}-column tiles")
    nrt, nct, nni = -(-m // ROW_TILE), -(-n // bn), bn // TILE_COLS
    rows = (4 * np.arange(nrt)[:, None] + np.arange(4)[None, :]) < pad_rows(m) // TILE_ROWS
    cols = (nni * np.arange(nct)[:, None] + np.arange(nni)[None, :]) < pad_cols(n) // TILE_COLS
    return np.broadcast_to(rows[:, None, None, None, :] & cols[None, :, None, :, None], (nrt, nct, S, nni, 4))


def committed_tiles(m: int, n: int, S: int, bn: int) -> np.ndarray:
    """The export word offset of each committed tile, in stream order."""
    mask = _tile_mask(m, n, S, bn)
    return np.flatnonzero(mask.reshape(-1)).astype(np.int64) * TILE_WORDS


def stream(export: np.ndarray, m: int, n: int, S: int, bn: int) -> bytes:
    """A call's committed words from the kernel's export of its ``64 ceil(m / 64)`` x ``bn ceil(n / bn)`` computed words."""
    mask = _tile_mask(m, n, S, bn)
    words = np.ascontiguousarray(export, dtype="<i4").reshape(-1)
    if words.size != mask.size * TILE_WORDS:
        raise ValueError(f"an export of {words.size} words, the call's tiles hold {mask.size * TILE_WORDS}")
    return words.reshape(*mask.shape, TILE_WORDS)[mask].tobytes()


def chunks(words: bytes) -> list[bytes]:
    """A call's committed stream in ``CHUNK_WORDS``-word chunks, the last short."""
    size = 4 * CHUNK_WORDS
    return [words[i:i + size] for i in range(0, len(words), size)]


def words_values(calls: Sequence[bytes]) -> list[bytes]:
    """A words instance's leaf values: each call's chunks, in call order."""
    return [c for words in calls for c in chunks(words)]


def a_segments(a: bytes) -> list[bytes]:
    """A_j's row-major bytes in ``A_SEGMENT``-byte segments, the last short."""
    return [a[i:i + A_SEGMENT] for i in range(0, len(a), A_SEGMENT)]


def a_position(forward: int, calls: int, call: int) -> int:
    """A_j's instance position: the call's position in the window."""
    if not 0 <= call < calls or forward < 0 or forward * calls + call >= WORDS_POSITION:
        raise ValueError(f"call {call} of {calls} in forward {forward}")
    return forward * calls + call


def words_position(forward: int) -> int:
    if not 0 <= forward < WORDS_POSITION:
        raise ValueError(f"forward {forward}")
    return WORDS_POSITION + forward


def instance_domain(salt_w: bytes, position: int, count: int) -> CommitmentDomain:
    """An instance's tree domain: ``count`` leaves, bound to the window's live salt and the instance's position."""
    if len(salt_w) != 64 or not 0 <= position < 1 << 64:
        raise ValueError("a 64-byte salt and a u64 position")
    binding = hashlib.sha512(BINDING_TAG + salt_w + position.to_bytes(8, "big")).digest()
    return CommitmentDomain(binding, -1, RangeIndexedDomain(0, count), hash="sha512")


@dataclass(frozen=True)
class Subtree:
    """An instance's leaf values and tree levels (leaves first, padded to a power of two)."""

    dom: CommitmentDomain
    schema: str
    values: tuple[bytes, ...]
    levels: tuple[tuple[bytes, ...], ...]

    @property
    def root(self) -> bytes:
        """The tree's root, which the keyed root replaces."""
        return self.levels[-1][0]

    @property
    def depth(self) -> int:
        return merkle_depth(self.dom.count)


def subtree(dom: CommitmentDomain, schema: str, values: Sequence[bytes]) -> Subtree:
    if not values:
        raise ValueError("an instance has at least one leaf")
    vs = tuple(bytes(v) for v in values)
    return Subtree(dom, schema, vs, tree_levels(dom, [dom.leaf(r, r, schema, v) for r, v in enumerate(vs)]))


def keyed_root(ks_w: bytes, salt_w: bytes, sub: Subtree) -> bytes:
    """STUB: the instance's keyed root d = SHA-512(ks_w || salt_w || M_top), M_top the subtree root's frame-v3 message."""
    raise NotImplementedError("the keyed root waits on its byte layout (verity.ml.commitments.keyed)")

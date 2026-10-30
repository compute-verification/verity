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

**The keyed root.**  An instance's public digest is its keyed root d = SHA-512(ks_w || salt_w || M_top)
(``verity.ml.commitments.keyed``), in the subtree root's place.  A window's instances are A_j at f calls + j, one a call, and the
forward's words at 2^63 + f, one a forward.  :class:`WindowKeying` keys them in the order d's hiding needs.  The window's call count
is fixed when it is made.  keyLeaf (ks_w's hm96 leaf) and the weights' root are recorded before salt_w arrives.  Each position
then gets one d.  It refuses everything else (:class:`verity_vllm.errors.Refused`).
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from verity.ml.commitments import hm96, keyed
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels
from verity_vllm.errors import Refused

__all__ = ["A_LEAF_TAG", "A_SEGMENT", "BINDING_TAG", "CHUNK_WORDS", "KEY_SALT_BYTES", "ROW_TILE", "TILE_COLS", "TILE_ROWS",
           "TILE_WORDS", "WORDS_LEAF_TAG", "WORDS_POSITION", "Subtree", "WindowKeying", "a_position", "a_segments", "chunks",
           "committed_tiles", "instance_domain", "key_leaf", "keyed_root", "pad_cols", "pad_rows", "stream", "subtree",
           "words_position", "words_values"]

TILE_ROWS, TILE_COLS = 16, 8
TILE_WORDS = TILE_ROWS * TILE_COLS
ROW_TILE = 64
CHUNK_WORDS = 65536
A_SEGMENT = 1024
WORDS_LEAF_TAG = "pouw/lead/words/v1"
A_LEAF_TAG = "pouw/lead/a/v1"
BINDING_TAG = b"pouw/lead/instance/v1"
WORDS_POSITION = 1 << 63
KEY_SALT_BYTES = hm96.SHA512.salt_bytes


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
    """A_j's instance position, f calls + j: injective over the window's calls and below every words position."""
    if not 0 <= call < calls or forward < 0 or forward * calls + call >= WORDS_POSITION:
        raise Refused(f"call {call} of {calls} in forward {forward}")
    return forward * calls + call


def words_position(forward: int) -> int:
    """The forward's words instance position, 2^63 + f."""
    if not 0 <= forward < WORDS_POSITION:
        raise Refused(f"forward {forward}")
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
    """The instance's public digest d = SHA-512(ks_w || salt_w || M_top), M_top the subtree root's frame-v3 message."""
    return keyed.digest(ks_w, salt_w, keyed.top_message(sub.dom, sub.schema, sub.values, sub.levels))


def key_leaf(ks_w: bytes, y0: bytes) -> bytes:
    """keyLeaf: ks_w's hm96-sha512 tree leaf under the salt y0 (``KEY_SALT_BYTES`` OS bytes)."""
    H = hm96.SHA512
    return H.tree_leaf(H.default_key, H.commit_string(H.default_key, ks_w, y0))


class WindowKeying:
    """One window's keyed digests.  ``calls`` (calls a forward) is fixed here.  :meth:`register` gives what the challenger
    records: the weights' root, keyLeaf and ``calls``.  :meth:`open` then takes salt_w.  Then :meth:`a_root` and
    :meth:`words_root` key one digest at each position.  ks_w and y0 are fresh OS bytes unless given, and ks_w is never
    published."""

    def __init__(self, calls: int, *, ks_w: bytes | None = None, y0: bytes | None = None) -> None:
        if type(calls) is not int or not 1 <= calls < WORDS_POSITION:
            raise Refused(f"a window of {calls} calls a forward")
        ks_w = os.urandom(keyed.KEY_BYTES) if ks_w is None else bytes(ks_w)
        y0 = os.urandom(KEY_SALT_BYTES) if y0 is None else bytes(y0)
        if len(ks_w) != keyed.KEY_BYTES or len(y0) != KEY_SALT_BYTES:
            raise Refused(f"ks_w is {keyed.KEY_BYTES} bytes and y0 {KEY_SALT_BYTES}")
        self._calls, self._ks_w, self._key_leaf = calls, ks_w, key_leaf(ks_w, y0)
        self._weights: bytes | None = None
        self._salt_w: bytes | None = None
        self._digests: dict[int, bytes] = {}

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def key_leaf(self) -> bytes:
        return self._key_leaf

    def register(self, weights_root: bytes) -> tuple[bytes, bytes, int]:
        """The window's registration, recorded with the challenger before it draws salt_w."""
        if self._weights is not None:
            raise Refused("a window is registered once")
        self._weights = bytes(weights_root)
        return self._weights, self._key_leaf, self._calls

    def open(self, salt_w: bytes) -> None:
        """Take the window's live salt, drawn after its registration was recorded."""
        if self._weights is None:
            raise Refused("salt_w before keyLeaf, the weights' root and the call count are recorded")
        if self._salt_w is not None:
            raise Refused("a second salt_w in the window")
        if len(salt_w) != keyed.KEY_BYTES:
            raise Refused(f"salt_w is {keyed.KEY_BYTES} bytes")
        self._salt_w = bytes(salt_w)

    def a_root(self, forward: int, call: int, a: bytes) -> bytes:
        """root(A_j): the keyed digest of the call's A, in ``A_SEGMENT``-byte segments, at f calls + j."""
        return self._keyed(a_position(forward, self._calls, call), A_LEAF_TAG, a_segments(a))

    def words_root(self, forward: int, calls: Sequence[bytes]) -> bytes:
        """The forward's words root: the keyed digest of its calls' committed streams, in call order, at 2^63 + f."""
        if len(calls) > self._calls:
            raise Refused(f"{len(calls)} calls' words in a window of {self._calls} calls a forward")
        return self._keyed(words_position(forward), WORDS_LEAF_TAG, words_values(calls))

    def _keyed(self, position: int, schema: str, values: list[bytes]) -> bytes:
        if self._salt_w is None:
            raise Refused("a keyed digest before salt_w")
        if position in self._digests:
            raise Refused(f"a second keyed digest at position {position} in the window")
        d = keyed_root(self._ks_w, self._salt_w, subtree(instance_domain(self._salt_w, position, len(values)), schema, values))
        self._digests[position] = d
        return d

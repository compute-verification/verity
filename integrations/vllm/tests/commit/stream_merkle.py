"""Streaming Merkle construction with O(log n) resident state, and replay-to-open (CPERF 2026-09-17 addendum).

Same frozen framing and the same pairing/lift rule as :class:`verity_vllm.commit.merkle.MerkleTree` (so the streaming root
equals the production root whenever the leaves are streamed in the production tree order), but the builder never
holds the leaves: it keeps ONE pending digest per level (the classic Merkle frontier) plus two counters per level.

    b = StreamingMerkle()
    for header, value in producer(): b.add(header, value)      # leaf digest computed and folded in immediately
    root = b.finish()                                          # O(log n) state at every moment before this line

Replay-to-open: nothing but ``(root, leaf_count)`` is retained.  To open leaf ``i`` later, the deterministic producer
is re-run and streamed through :class:`StreamingMerkle` with ``track=i``; the builder records the sibling of the
tracked node at every level as pairs form (still O(log n) state) and returns a path that
:func:`verity_vllm.commit.merkle.verify_opening` accepts against the retained root.

Tree order: production sorts canonical ids (``identity.tree_sorted``); a stream can only reduce online in ARRIVAL
order, so either (a) the producer emits in tree order (true for an RU whose op list is fixed by the template:
``ids_sorted`` is known before any value exists) or (b) the frame is versioned so that tree order = emission order.
This module does not decide that; :func:`stream_in_tree_order` shows (a) for a stored RU.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from tests.commit import fasttree as ft
from verity_vllm.commit import scheme


class StreamingMerkle:
    """Online Merkle reduction: ``add`` per leaf, ``finish`` for the root; resident state = one digest per level."""

    __slots__ = ("new", "pending", "count", "leaf_count", "track", "path", "track_pos", "_done")

    def __init__(self, new: Callable[..., Any] = hashlib.sha256, track: int | None = None) -> None:
        self.new = new
        self.pending: list[bytes | None] = []   # one unpaired node per level (the frontier)
        self.count: list[int] = []              # nodes seen so far per level (gives every node's index)
        self.leaf_count = 0
        self.track = track                      # leaf index whose authentication path we collect (or None)
        self.path: list[bytes | None] = []      # sibling per level for the tracked leaf (None = lifted)
        self.track_pos: list[int] = []          # tracked node's index per level (index >> level)
        self._done = False

    # -- state accounting ------------------------------------------------------------------------------------------
    def resident_digests(self) -> int:
        return sum(1 for p in self.pending if p is not None) + sum(1 for p in self.path if p is not None)

    def resident_bytes(self) -> int:
        return 32 * self.resident_digests() + 16 * len(self.count)

    # -- construction ----------------------------------------------------------------------------------------------
    def add(self, header: bytes, value: bytes | memoryview) -> None:
        self.add_digest(ft.leaf_digest(header, value, self.new))

    def add_digest(self, digest: bytes) -> None:
        if self._done:
            raise RuntimeError("finished")
        if self.track is not None and self.leaf_count == self.track:
            self.track_pos = [self.track]
        self.leaf_count += 1
        self._push(0, digest)

    def _ensure(self, level: int) -> None:
        while len(self.pending) <= level:
            self.pending.append(None)
            self.count.append(0)

    def _tracked_at(self, level: int) -> int | None:
        return self.track_pos[level] if (self.track is not None and len(self.track_pos) > level) else None

    def _push(self, level: int, node: bytes) -> None:
        """Insert ``node`` as the next element of ``level``; pair it with the pending one if there is one."""
        self._ensure(level)
        idx = self.count[level]
        self.count[level] += 1
        left = self.pending[level]
        if left is None:
            self.pending[level] = node
            return
        # pair (idx-1, idx) -> parent at level+1 with index idx//2
        t = self._tracked_at(level)
        if t is not None and t in (idx - 1, idx):
            self.path.append(node if t == idx - 1 else left)
            self.track_pos.append(idx // 2)
        self.pending[level] = None
        self._push(level + 1, self.new(scheme.NODE_PREFIX + left + scheme.U64_32 + node).digest())

    def finish(self) -> bytes:
        """Apply the lifts for the odd tails, bottom-up, exactly as the batch fold does; returns the root."""
        if self._done:
            raise RuntimeError("finished")
        self._done = True
        if self.leaf_count == 0:
            return scheme.empty_root()
        carry: bytes | None = None
        level = 0
        top = len(self.pending) + 2
        while level <= top + 1:
            self._ensure(level)
            pend = self.pending[level]
            if carry is not None:
                # the carry (a lift or a pair from below) is the LAST element of this level
                idx = self.count[level]
                self.count[level] += 1
                if pend is not None:
                    t = self._tracked_at(level)
                    if t is not None and t in (idx - 1, idx):
                        self.path.append(carry if t == idx - 1 else pend)
                        self.track_pos.append(idx // 2)
                    node = self.new(scheme.NODE_PREFIX + pend + scheme.U64_32 + carry).digest()
                    self.pending[level] = None
                    carry = node
                    level += 1
                    top += 1
                    continue
                pend, carry = carry, None
            if pend is None:
                level += 1
                continue
            if self.count[level] == 1:
                return pend  # a level with exactly one element in total is the root
            t = self._tracked_at(level)  # pend is the odd trailing element: lifted
            if t is not None and t == self.count[level] - 1:
                self.path.append(None)
                self.track_pos.append(t // 2)
            carry = self.new(scheme.LIFT_PREFIX + pend).digest()
            self.pending[level] = None
            level += 1
        raise RuntimeError("no root")  # pragma: no cover


# -- helpers for the demo / test -------------------------------------------------------------------------------------


def stream_root(leaves: Iterable[tuple[bytes, bytes | memoryview]], new: Callable[..., Any] = hashlib.sha256) -> tuple[bytes, int, int]:
    """Root, leaf count and the maximum resident bytes observed during the stream."""
    b = StreamingMerkle(new)
    peak = 0
    for h, v in leaves:
        b.add(h, v)
        peak = max(peak, b.resident_bytes())
    return b.finish(), b.leaf_count, peak


def replay_to_open(producer: Callable[[], Iterator[tuple[bytes, bytes | memoryview]]], index: int,
                   new: Callable[..., Any] = hashlib.sha256) -> tuple[bytes, int, list[bytes | None], bytes]:
    """Re-run ``producer`` and return (root, leaf_count, path for ``index``, the leaf's value bytes)."""
    b = StreamingMerkle(new, track=index)
    value: bytes | None = None
    for i, (h, v) in enumerate(producer()):
        if i == index:
            value = bytes(v)
        b.add(h, v)
    root = b.finish()
    if value is None:
        raise IndexError(index)
    return root, b.leaf_count, b.path, value


def stream_in_tree_order(reader: Any, rid: str, step: int) -> tuple[bytes, int, int]:
    """A stored RU streamed in the production tree order (ids known from the index before the values): root,
    leaves, peak resident bytes of the builder.  Root equals ``commitments.json``'s."""
    from verity_vllm.commit import identity

    ru = reader.ru(rid, step)
    by = {e.canonical_id: e for e in ru.index.entries}
    ids = identity.tree_sorted(list(by))
    import numpy as np

    mm = memoryview(np.memmap(ru.dir / "values.bin", dtype=np.uint8, mode="r"))

    def gen() -> Iterator[tuple[bytes, memoryview]]:
        for c in ids:
            e = by[c]
            yield ft.leaf_header(c, e.dtype, e.shape), mm[e.offset : e.offset + e.nbytes]

    return stream_root(gen())

"""Vectorized ``hm96`` hiding leaves (``verity.ml.commitments.hm96``, spec ``PROTOCOL.md`` beside it): one instance's tree leaves
over many unsalted leaf digests at once.  ``tests/test_hiding.py`` pins it to the reference and its vectors.

Each leaf gets fresh salt bytes from the OS generator (one call per ``leaves`` call), which the caller retains: a leaf opens
with its value and its salt.  ``M_key * y`` gathers the instance's byte tables (per salt byte position, the XOR of the columns
its set bits select) with numpy over all the leaves at once.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence

import numpy as np

from verity.ml.commitments import hm96

NAME = hm96.NAME
SALT_BYTES = hm96.SALT_BYTES
_BATCH = 4096


class HidingLeaves:
    """The vectorized path of one hm96 instance (``hm96.SHA256`` by default; ``hm96.SHA512`` for a window's rows) under ``key`` (the
    instance's pinned key by default)."""

    def __init__(self, key: bytes | None = None, instance: hm96.Instance = hm96.SHA256) -> None:
        self.instance = inst = instance
        self.key = inst.check_key(inst.default_key if key is None else key)
        self.digest = inst.scheme_digest(self.key)
        self._d, self._y = inst.digest_bytes, inst.salt_bytes
        flat = b"".join(t.to_bytes(self._d, "little") for table in inst.byte_tables(self.key) for t in table)
        self._tables = np.frombuffer(flat, dtype=np.uint64).reshape(self._y, 256, self._d // 8)
        self._salt_h = hashlib.new(inst.hash, inst.salt_prefix)
        self._leaf_h = hashlib.new(inst.hash, inst.leaf_prefix(self.key))

    def masks(self, salts: bytes, n: int) -> np.ndarray:
        """``M_key * y`` of ``n`` concatenated salts, as ``(n, digest bytes)``."""
        y = np.frombuffer(salts, dtype=np.uint8).reshape(n, self._y)
        out = np.zeros((n, self._d // 8), dtype=np.uint64)
        for s in range(0, n, _BATCH):
            yt = np.ascontiguousarray(y[s:s + _BATCH].T)
            acc = out[s:s + _BATCH]
            for p in range(self._y):
                acc ^= self._tables[p][yt[p]]
        return out.view(np.uint8).reshape(n, self._d)

    def leaves_with(self, inner: Sequence[bytes], salts: bytes) -> list[bytes]:
        """The tree leaves of the unsalted digests ``inner`` under the given concatenated salts."""
        n, d, y = len(inner), self._d, self._y
        if len(salts) != n * y:
            raise ValueError(f"{n} leaves take {n * y} salt bytes, not {len(salts)}")
        if not n:
            return []
        b = (np.frombuffer(b"".join(inner), dtype=np.uint8).reshape(n, d) ^ self.masks(salts, n)).tobytes()
        ys, bs = memoryview(salts), memoryview(b)
        out = []
        for i in range(n):
            h = self._salt_h.copy()
            h.update(ys[i * y:(i + 1) * y])
            g = self._leaf_h.copy()
            g.update(bs[i * d:(i + 1) * d])
            g.update(h.digest())
            out.append(g.digest())
        return out

    def leaves(self, inner: Sequence[bytes]) -> tuple[list[bytes], bytes]:
        """Fresh salts for every leaf of ``inner``: the tree leaves and the concatenated salts (the stored bytes)."""
        salts = os.urandom(len(inner) * self._y)
        return self.leaves_with(inner, salts), salts

    def leaf(self, inner_digest: bytes, salt: bytes) -> bytes:
        """One leaf through core's reference (the verifier's path)."""
        return self.instance.tree_leaf(self.key, self.instance.commit_string(self.key, inner_digest, salt))

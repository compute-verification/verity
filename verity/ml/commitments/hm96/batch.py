"""hm96 tree leaves for many values at once, the mask in numpy: the bytes of `Instance.tree_leaf` of `Instance.commit_string`
(the plain reference beside it), for a committer of many rows that holds no numpy itself (PoUW's registered weights,
``verity.core.protocols.pouw.weights``)."""

from __future__ import annotations

import functools
import hashlib

import numpy as np

from . import Instance, _byte_tables

__all__ = ["masks", "tree_leaves"]


@functools.lru_cache(maxsize=4)
def _tables(inst: Instance, key: bytes) -> np.ndarray:
    """`Instance.byte_tables` as [salt byte, value, digest word] u64 words, each column XOR little-endian."""
    d = inst.digest_bytes
    raw = b"".join(v.to_bytes(d, "little") for t in _byte_tables(inst, inst.check_key(key)) for v in t)
    return np.frombuffer(raw, "<u8").reshape(inst.salt_bytes, 256, d // 8)


def masks(inst: Instance, key: bytes, salts: bytes) -> bytes:
    """``M_key * y`` of each salt y (`Instance.mask`), concatenated; ValueError unless `salts` is whole salts."""
    if len(salts) % inst.salt_bytes:
        raise ValueError(f"{len(salts)} bytes are not whole {inst.name} salts of {inst.salt_bytes} bytes")
    t = _tables(inst, bytes(key))
    y = np.frombuffer(salts, np.uint8).reshape(-1, inst.salt_bytes)
    out = np.zeros((y.shape[0], t.shape[2]), np.uint64)
    for p in range(inst.salt_bytes):
        out ^= t[p][y[:, p]]
    return out.astype("<u8").tobytes()


def tree_leaves(inst: Instance, key: bytes, digests: bytes, salts: bytes) -> list[bytes]:
    """``tree_leaf(key, commit_string(key, x, y))`` for each unsalted digest x of `digests` and salt y of `salts`, in order;
    ValueError unless they are as many whole digests as salts."""
    d, s = inst.digest_bytes, inst.salt_bytes
    n = len(digests) // d
    if len(digests) != n * d or len(salts) != n * s:
        raise ValueError(f"{len(digests)} digest bytes and {len(salts)} salt bytes are not as many {d}- and {s}-byte items")
    b = (np.frombuffer(digests, np.uint8) ^ np.frombuffer(masks(inst, key, salts), np.uint8)).tobytes()
    salt_h, leaf_h = hashlib.new(inst.hash, inst.salt_prefix), hashlib.new(inst.hash, inst.leaf_prefix(key))
    out = []
    for i in range(n):
        c = salt_h.copy()
        c.update(salts[s * i:s * (i + 1)])
        h = leaf_h.copy()
        h.update(b[d * i:d * (i + 1)])
        h.update(c.digest())
        out.append(h.digest())
    return out

"""Merkle multi-opening (shared authentication paths) for a set of leaves of a power-of-two Blake3 tree whose leaves
are ``blake3(row words, little-endian u32)`` and nodes ``blake3(left || right)`` -- the tree ``verity/ml/kernels/hash_gpu/
merkle.commit`` builds on the device (bit-exact with the host ``blake3`` package, as ``ligero/merkle.verify_path``
relies on).  ``plan`` lists the sibling nodes the verifier cannot recompute, level by level in sorted order; the
prover gathers exactly those digests from the device tree in one copy; ``verify`` consumes them in the same order.
"""
from __future__ import annotations

import numpy as np

try:
    import blake3 as _blake3

    def H(data: bytes) -> bytes:
        return _blake3.blake3(data).digest()
except ImportError:  # pragma: no cover
    import hashlib

    def H(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()


def plan(indices, depth: int) -> list[tuple[int, int]]:
    cur = sorted(set(int(i) for i in indices))
    out: list[tuple[int, int]] = []
    for level in range(depth):
        cur_set = set(cur)
        nxt: list[int] = []
        for i in cur:
            if (i ^ 1) not in cur_set:
                out.append((level, i ^ 1))
            parent = i >> 1
            if not nxt or nxt[-1] != parent:
                nxt.append(parent)
        cur = nxt
    return out


def plan_levels(indices, depth: int) -> list[tuple[int, np.ndarray]]:
    """``plan`` grouped by level, vectorised: [(level, sibling indices (int64, sorted))] -- the same digests in the same
    order as ``plan`` (a missing sibling i ^ 1 sorts with i, so per-level order is the sorted order of the siblings)."""
    cur = np.unique(np.asarray(indices, dtype=np.int64))
    out: list[tuple[int, np.ndarray]] = []
    for level in range(depth):
        sib = cur ^ 1
        miss = sib[~np.isin(sib, cur, assume_unique=True)]
        if len(miss):
            out.append((level, np.sort(miss)))
        cur = np.unique(cur >> 1)
    return out


def gather(tree, indices, depth: int) -> bytes:
    """Concatenated 32-byte sibling digests per ``plan`` from a device tree (``levels[k]`` = (count, 8) uint32)."""
    import cupy as cp

    pl = plan(indices, depth)
    if not pl:
        return b""
    parts = []
    by_level: dict[int, list[int]] = {}
    for lv, i in pl:
        by_level.setdefault(lv, []).append(i)
    digests: dict[tuple[int, int], bytes] = {}
    for lv, idxs in by_level.items():
        arr = cp.asnumpy(tree.levels[lv][cp.asarray(np.asarray(idxs, dtype=np.int64))]).astype("<u4")
        for i, row in zip(idxs, arr):
            digests[(lv, i)] = row.tobytes()
    for key in pl:
        parts.append(digests[key])
    return b"".join(parts)


def leaf_digest(words: np.ndarray) -> bytes:
    return H(np.ascontiguousarray(np.asarray(words, dtype=np.int64).astype(np.uint32)).tobytes())


def verify(root: bytes, leaves: dict[int, bytes], siblings: bytes, depth: int) -> bool:
    """``leaves``: index -> leaf digest.  ``siblings``: the prover's bytes in ``plan`` order."""
    pl = plan(list(leaves.keys()), depth)
    if len(siblings) != 32 * len(pl):
        return False
    pos = 0
    cur = dict(leaves)
    for _level in range(depth):
        nxt: dict[int, bytes] = {}
        for i in sorted(cur):
            if (i >> 1) in nxt:
                continue
            sib = i ^ 1
            if sib in cur:
                s = cur[sib]
            else:
                s = siblings[pos:pos + 32]
                pos += 32
            l, r = (cur[i], s) if i % 2 == 0 else (s, cur[i])
            nxt[i >> 1] = H(l + r)
        cur = nxt
    return pos == len(siblings) and cur.get(0) == root

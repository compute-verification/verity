"""CPU references (bit-exact targets) for ``merkle.commit``: ``hashlib`` SHA-256, the ``blake3`` package, and the
pure-Python Poseidon2 of ``poseidon2_ref``.  Small inputs only (the Poseidon2 reference is pure Python)."""
from __future__ import annotations

import hashlib

import numpy as np

from . import poseidon2_ref as p2

CHUNK_ROWS = 256


def column_bytes(m: np.ndarray, j: int) -> bytes:
    """The bytes hashed for column ``j``: its elements as little-endian u32, row order."""
    return np.ascontiguousarray(m[:, j]).astype("<u4").tobytes()


def _blake3(data: bytes) -> bytes:
    import blake3

    return blake3.blake3(data).digest()


def leaf(m: np.ndarray, j: int, hash: str, p2inst: p2.Poseidon2 | None = None) -> bytes:
    if hash == "sha256":
        return hashlib.sha256(column_bytes(m, j)).digest()
    if hash == "blake3":
        return _blake3(column_bytes(m, j))
    if hash == "sha256-chunked":
        col = m[:, j]
        level = [hashlib.sha256(np.ascontiguousarray(col[i:i + CHUNK_ROWS]).astype("<u4").tobytes()).digest()
                 for i in range(0, len(col), CHUNK_ROWS)]
        while len(level) > 1:
            nxt = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level) - 1, 2)]
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        return level[0]
    if hash.startswith("poseidon2"):
        inst = p2inst or p2.instance(int(hash.split("-")[1]))
        return np.array(inst.leaf_blocks([int(x) for x in m[:, j]]), dtype="<u4").tobytes()
    raise ValueError(hash)


def node(left: bytes, right: bytes, hash: str, p2inst: p2.Poseidon2 | None = None) -> bytes:
    if hash in ("sha256", "sha256-chunked"):
        return hashlib.sha256(left + right).digest()
    if hash == "blake3":
        return _blake3(left + right)
    inst = p2inst or p2.instance(int(hash.split("-")[1]))
    lw = np.frombuffer(left, dtype="<u4").tolist()
    rw = np.frombuffer(right, dtype="<u4").tolist()
    return np.array(inst.compress(lw, rw), dtype="<u4").tobytes()


def tree(m: np.ndarray, hash: str, p2inst: p2.Poseidon2 | None = None) -> list[list[bytes]]:
    """All levels (leaves first) of the commitment of ``m`` (``(R, N)`` uint32) under ``hash``."""
    levels = [[leaf(m, j, hash, p2inst) for j in range(m.shape[1])]]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        nxt = [node(cur[i], cur[i + 1], hash, p2inst) for i in range(0, len(cur) - 1, 2)]
        if len(cur) % 2:
            nxt.append(cur[-1])
        levels.append(nxt)
    return levels


def verify_path(leaf_digest: bytes, index: int, path: list[bytes | None], root: bytes, hash: str,
                p2inst: p2.Poseidon2 | None = None) -> bool:
    d = leaf_digest
    for sib in path:
        if sib is not None:
            d = node(d, sib, hash, p2inst) if index % 2 == 0 else node(sib, d, hash, p2inst)
        index >>= 1
    return d == root

"""Merkle commitment to the columns of the codeword matrix U (m x n): leaf c = H(column c as m little-endian uint32),
internal = H(left || right).  Host hashing: Blake3 (``blake3`` package) when importable, else SHA-256 (hashlib) --
both PLACEHOLDERS for the GPU Merkle of ``backends/shared/hash_gpu``; the hash is named in the proof header.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os

import numpy as np

try:
    import blake3 as _blake3

    def _h(data: bytes) -> bytes:
        return _blake3.blake3(data).digest()

    HASH_NAME = "blake3 (host, placeholder)"
except ImportError:                                   # pragma: no cover
    def _h(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()

    HASH_NAME = "sha256 (host, placeholder)"

__all__ = ["HASH_NAME", "commit", "open_columns", "verify_path", "hash_bytes"]


def hash_bytes(data: bytes) -> bytes:
    return _h(data)


def leaves_of(U_cols: np.ndarray, threads: int | None = None) -> list[bytes]:
    """U_cols: (n, m) uint32 (column-major view of U) -> n leaf digests."""
    n = U_cols.shape[0]
    raw = np.ascontiguousarray(U_cols.astype(np.uint32))
    threads = threads or min(32, (os.cpu_count() or 4))
    if n < 256 or threads <= 1:
        return [_h(raw[i].tobytes()) for i in range(n)]
    chunks = np.array_split(np.arange(n), threads)

    def work(idx):
        return [_h(raw[i].tobytes()) for i in idx]

    with ThreadPoolExecutor(threads) as ex:
        parts = list(ex.map(work, chunks))
    return [d for part in parts for d in part]


def build_tree(leaves: list[bytes]) -> list[list[bytes]]:
    levels = [leaves]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([_h(cur[i] + cur[i + 1]) for i in range(0, len(cur), 2)])
    return levels


def commit(U_cols: np.ndarray) -> tuple[bytes, list[list[bytes]]]:
    """Root and the tree (all levels, leaves first)."""
    levels = build_tree(leaves_of(U_cols))
    return levels[-1][0], levels


def open_columns(levels: list[list[bytes]], cols: list[int]) -> list[list[bytes]]:
    """Authentication path (sibling per level) for each column."""
    paths = []
    for c in cols:
        path = []
        i = c
        for lvl in levels[:-1]:
            path.append(lvl[i ^ 1])
            i >>= 1
        paths.append(path)
    return paths


def verify_path(root: bytes, col: int, column: np.ndarray, path: list[bytes]) -> bool:
    h = _h(np.ascontiguousarray(column.astype(np.uint32)).tobytes())
    i = col
    for sib in path:
        h = _h(h + sib) if i % 2 == 0 else _h(sib + h)
        i >>= 1
    return h == root

"""hm96 tree leaves for many values at once, the mask in numpy: the bytes of `Instance.tree_leaf` of `Instance.commit_string`
(the plain reference beside it), for a committer of many rows that holds no numpy itself (PoUW's registered weights,
``verity.core.protocols.pouw.weights``); and `salts`, the stream every hm96-sha512 salt of a committed row is drawn from."""

from __future__ import annotations

import functools
import hashlib

from collections.abc import Sequence

import numpy as np

from verity.ml.crypto import chacha20

from . import SHA512, Instance, _byte_tables

__all__ = ["SALT_BLOCKS", "masks", "salts", "tree_leaves"]

#: ChaCha20 blocks an hm96-sha512 salt
SALT_BLOCKS = SHA512.salt_bytes // chacha20.BLOCK_BYTES
#: rows a pass of `salts`
_ROWS = 1 << 16


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


def salts(key: bytes, nonce: int, rows: Sequence[int] | np.ndarray, first: int = 0) -> bytes:
    """The hm96-sha512 salt stream: each of ``rows``' salt in order, concatenated (``SALT_BLOCKS`` ChaCha20 blocks each), row
    j's the blocks ``first + 3j`` .. ``first + 3j + 2`` under the 32-byte ``key`` at ``nonce`` (`chacha20.blocks`, C-Flock's
    ``hm96_salt`` at ``first = 0``); ValueError on another key size, a nonce or a row's block outside [0, 2^64)."""
    at = np.asarray(rows)
    if at.ndim != 1 or (at.size and at.dtype.kind not in "iu"):
        raise ValueError(f"rows of {at.dtype} {at.shape}: a sequence of row indices")
    if not 0 <= first < 1 << 64:
        raise ValueError("ChaCha20's block counter is 64-bit")
    if not at.size:
        return chacha20.blocks(key, nonce, np.zeros(0, np.uint64))
    if int(at.min()) < 0 or first + SALT_BLOCKS * int(at.max()) + SALT_BLOCKS > 1 << 64:
        raise ValueError("ChaCha20's block counter is 64-bit: a row's salt blocks fall outside it")
    at = at.astype(np.uint64)
    step = np.uint64(first) + np.arange(SALT_BLOCKS, dtype=np.uint64)
    return b"".join(chacha20.blocks(key, nonce, (np.uint64(SALT_BLOCKS) * at[lo:lo + _ROWS, None] + step).ravel())
                    for lo in range(0, at.size, _ROWS))

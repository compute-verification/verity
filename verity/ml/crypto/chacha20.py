"""ChaCha20's keystream in numpy, many blocks at once: Bernstein's block function (20 rounds) with a 64-bit block counter in
words 12 and 13 and a 64-bit nonce in words 14 and 15, the layout C-Flock's ``chacha20_block``
(``verity/ml/flock/cuda/sha512.cuh``) draws hm96 leaf salts with.  RFC 8439's 32-bit counter and 96-bit nonce are the same
four words read another way.  The plain reference is ``verity.core.protocols.pouw.audit.chacha20_block``, whose tests hold
this to it, and ``audit.hm96_salt`` is `hm96_salts`' (row j's salt, blocks 3j .. 3j + 2).
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import numpy as np

__all__ = ["BLOCK_BYTES", "SALT_BLOCKS", "hm96_salts", "keystream"]

BLOCK_BYTES = 64
#: blocks an hm96 leaf salt: row j's are blocks 3j .. 3j + 2 (`audit.hm96_salt`)
SALT_BLOCKS = 3
_SIGMA = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)
_ROUND = ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),
          (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14))
_M32 = np.uint64(0xFFFFFFFF)
#: rows a pass of `hm96_salts`
_ROWS = 1 << 16


def _key(key: bytes, nonce: int) -> None:
    if len(key) != 32:
        raise ValueError(f"a ChaCha20 key is 32 bytes, not {len(key)}")
    if not 0 <= nonce < 1 << 64:
        raise ValueError("ChaCha20's nonce and block counter are 64-bit")


def keystream(key: bytes, nonce: int, first: int, blocks: int) -> bytes:
    """Blocks ``first`` .. ``first + blocks - 1`` of the stream under the 32-byte `key` and `nonce`, concatenated
    (``BLOCK_BYTES`` each); ValueError on another key size, a nonce or counter outside [0, 2^64)."""
    _key(key, nonce)
    if first < 0 or blocks < 0 or first + blocks > 1 << 64:
        raise ValueError("ChaCha20's nonce and block counter are 64-bit")
    if not blocks:
        return b""
    return _blocks(key, nonce, np.uint64(first) + np.arange(blocks, dtype=np.uint64))


def hm96_salts(key: bytes, nonce: int, rows: Sequence[int] | np.ndarray) -> bytes:
    """Each of ``rows``' hm96 salt in order, concatenated (``SALT_BLOCKS * BLOCK_BYTES`` bytes each): row j's
    `audit.hm96_salt(key, nonce, j)`, blocks 3j .. 3j + 2, every row's at once; ValueError on a row whose blocks fall outside
    [0, 2^64)."""
    _key(key, nonce)
    at = np.asarray(rows)
    if at.ndim != 1 or (at.size and at.dtype.kind not in "iu"):
        raise ValueError(f"rows of {at.dtype} {at.shape}: a sequence of row indices")
    if not at.size:
        return b""
    if int(at.min()) < 0 or int(at.max()) > ((1 << 64) - SALT_BLOCKS) // SALT_BLOCKS:
        raise ValueError("ChaCha20's block counter is 64-bit: a row's salt blocks fall outside it")
    at = at.astype(np.uint64)
    step = np.arange(SALT_BLOCKS, dtype=np.uint64)
    return b"".join(_blocks(key, nonce, (np.uint64(SALT_BLOCKS) * at[lo:lo + _ROWS, None] + step).ravel())
                    for lo in range(0, at.size, _ROWS))


def _blocks(key: bytes, nonce: int, ctr: np.ndarray) -> bytes:
    """The blocks at counters ``ctr`` (u64), in its order, concatenated."""
    blocks = ctr.size
    st = [np.full(blocks, w, np.uint32) for w in (*_SIGMA, *struct.unpack("<8I", key))]
    st += [(ctr & _M32).astype(np.uint32), (ctr >> np.uint64(32)).astype(np.uint32),
           np.full(blocks, nonce & 0xFFFFFFFF, np.uint32), np.full(blocks, nonce >> 32, np.uint32)]
    x = [w.copy() for w in st]
    for _ in range(10):
        for a, b, c, d in _ROUND:
            for p, q, r, n in ((a, b, d, 16), (c, d, b, 12), (a, b, d, 8), (c, d, b, 7)):
                x[p] += x[q]
                x[r] ^= x[p]
                x[r] = (x[r] << np.uint32(n)) | (x[r] >> np.uint32(32 - n))
    return np.stack([x[i] + st[i] for i in range(16)], axis=1).astype("<u4").tobytes()

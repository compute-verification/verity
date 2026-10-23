"""H1's served tables registered at serving scale (VLLM-76): `registered.Registered.commit` of rows of 16-bit words as
``hm96-sha512/row/v2`` bit rows, row j salted `audit.hm96_salt` of the run's key at the value's nonce, with the same
salts, ``b || c`` strings, tree and entry, computed from the rows' NumPy words instead of Python ints.

* **The rows' digests.** Row j's bit row is its words little-endian, so ``x_j = SHA-512(sha512_row_v2_prefix(16 W) ||
  words[j])`` is hashed straight from the array's memory, rows split among threads (`hashlib` releases the GIL on a
  buffer of 2 KiB or more, which every H1 row is: a z/y record is 6 KiB, an A or B row K + 4 bytes).
* **The salts.** ChaCha20 blocks 3j, 3j + 1, 3j + 2 under the key at the nonce, the block function `audit.chacha20_block`
  (64-bit counter in words 12 and 13, 64-bit nonce in 14 and 15), every row's at once over NumPy.
* **The commit strings.** ``b = x ⊕ M y`` gathers hm96's byte tables over every row (`hiding.HidingLeaves.masks`), ``c``
  and the tree leaf hash from the instance's one-block prefixes, and the frame-v3-sha512 leaves, nodes and pads from
  their tag's and domain's constant prefix (`merkle._hash`'s message), so a row costs a few short SHA-512 calls.
* **What it holds.** The ``b || c`` strings and the tree's levels, 64 bytes a node. The salts are drawn again on read
  (`Salts`), and the rows are kept only with ``keep_rows`` (`Rows`): a window's call tables and its own ``b/<K>`` are
  rebuilt by the prover from what it keeps (`pouw_h1_prover.keep_call`, its weights), so `Served` keeps none of their
  rows and a window holds a few hundred bytes a row (its ``b || c`` and about two tree nodes, as Python bytes), where
  Python ints held some 36 bytes a word. The period's `Weights` keeps its rows, two bytes a word, for `Weights.matches`.

`commit` returns a `registered.Registered` whose ``rows`` and ``salts`` are those sequences; everything `Served`,
`ServedTree.of`, `registered.verify_read` and the record read of it (``root``, ``levels``, ``bcs``, ``entry()``,
``path``) is what `Registered.commit` gives over the same rows and salts (``tests/protocol_options/
test_pouw_h1_registered.py`` holds it to core's on small windows).
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any

import numpy as np

from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import merkle_depth
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix

from verity_vllm.commit.hiding import HidingLeaves

_INST = hm96.SHA512
_FRAME = b"veritor/protocol/merkle/frame/v3\0"
_SIGMA = np.frombuffer(b"expand 32-byte k", dtype="<u4")
_QR = ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15), (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13),
       (3, 4, 9, 14))
#: ChaCha20 blocks a pass, rows a digest task
_BLOCKS, _ROWS = 1 << 16, 1 << 11


@cache
def _pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max(1, min(16, os.cpu_count() or 1)), thread_name_prefix="h1-registered")


@cache
def _hiding() -> HidingLeaves:
    return HidingLeaves(instance=_INST)


def _rotl(v: np.ndarray, n: int) -> np.ndarray:
    return (v << np.uint32(n)) | (v >> np.uint32(32 - n))


def salts(key: bytes, nonce: int, n: int) -> np.ndarray:
    """Rows 0 .. n - 1's salts ``(n, 192)`` u8: row j's `audit.hm96_salt(key, nonce, j)`."""
    if len(key) != 32 or not 0 <= nonce < 1 << 64 or 3 * n > 1 << 64:
        raise ValueError("a 32-byte key, a u64 nonce and at most 2^64 blocks")
    out = np.empty((3 * n, 16), "<u4")
    for lo in range(0, 3 * n, _BLOCKS):
        m = min(_BLOCKS, 3 * n - lo)
        ctr = np.arange(lo, lo + m, dtype=np.uint64)
        s = np.empty((16, m), np.uint32)
        s[0:4] = _SIGMA[:, None]
        s[4:12] = np.frombuffer(key, "<u4")[:, None]
        s[12], s[13] = (ctr & np.uint64(0xFFFFFFFF)).astype(np.uint32), (ctr >> np.uint64(32)).astype(np.uint32)
        s[14], s[15] = nonce & 0xFFFFFFFF, nonce >> 32
        x = s.copy()
        for _ in range(10):
            for a, b, c, d in _QR:
                x[a] += x[b]; x[d] = _rotl(x[d] ^ x[a], 16)  # noqa: E702
                x[c] += x[d]; x[b] = _rotl(x[b] ^ x[c], 12)  # noqa: E702
                x[a] += x[b]; x[d] = _rotl(x[d] ^ x[a], 8)  # noqa: E702
                x[c] += x[d]; x[b] = _rotl(x[b] ^ x[c], 7)  # noqa: E702
        x += s
        out[lo:lo + m] = x.T
    return out.view(np.uint8).reshape(n, 3 * 64)


def _digests(words: np.ndarray, lo: int, hi: int, head: Any, out: np.ndarray) -> None:
    for j in range(lo, hi):
        h = head.copy()
        h.update(words[j])
        out[j] = np.frombuffer(h.digest(), np.uint8)


def row_digests(words: np.ndarray) -> np.ndarray:
    """Each row's ``x``, ``(n, 64)`` u8: `registered.row_digest(row, 16, SCHEMA_V2)` of ``words`` ``(n, W)`` u16."""
    words = np.ascontiguousarray(words, "<u2")
    n, w = words.shape
    head, out = hashlib.sha512(sha512_row_v2_prefix(16 * w)), np.empty((n, 64), np.uint8)
    rows = words.view(np.uint8).reshape(n, 2 * w)
    list(_pool().map(lambda lo: _digests(rows, lo, min(lo + _ROWS, n), head, out), range(0, n, _ROWS)))
    return out


def _part(p: bytes) -> bytes:
    return len(p).to_bytes(8, "big") + p


def _uint(v: int) -> bytes:
    return _part(v.to_bytes(max(1, (v.bit_length() + 7) // 8), "big"))


class _Frame:
    """`merkle._hash` under one frame-v3-sha512 domain, each tag's constant prefix hashed once."""

    def __init__(self, domain_id: bytes) -> None:
        def head(tag: bytes) -> Any:
            return hashlib.sha512(_FRAME + len(tag).to_bytes(4, "big") + tag + _part(domain_id))
        self.leaf, self.node, self.pad = head(b"leaf"), head(b"node"), head(b"pad")

    def levels(self, leaves: list[bytes]) -> tuple[tuple[bytes, ...], ...]:
        """`merkle.tree_levels` of ``leaves``."""
        level = leaves + [self._h(self.pad, _uint(r)) for r in range(len(leaves), 1 << merkle_depth(len(leaves)))]
        out = [tuple(level)]
        while len(level) > 1:
            d = _uint(len(out) - 1)
            level = [self._h(self.node, d + _uint(i // 2) + _part(level[i]) + _part(level[i + 1]))
                     for i in range(0, len(level), 2)]
            out.append(tuple(level))
        return tuple(out)

    @staticmethod
    def _h(head: Any, rest: bytes) -> bytes:
        h = head.copy()
        h.update(rest)
        return h.digest()


class Salts(Sequence):
    """A registered value's salts, drawn again on read: item j is `audit.hm96_salt(key, nonce, j)`."""

    def __init__(self, key: bytes, nonce: int, n: int) -> None:
        self.key, self.nonce, self.n = bytes(key), int(nonce), int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, j: Any) -> Any:
        if isinstance(j, slice):
            return [self[i] for i in range(*j.indices(self.n))]
        if not -self.n <= j < self.n:
            raise IndexError(j)
        return hm96_salt(self.key, self.nonce, j % self.n)


class Rows(Sequence):
    """A registered value's rows: item j row j's words as ints (a slice, the words ``(·, W)`` u16), when kept; else every
    read refused, the prover rebuilding them from what it keeps (`pouw_h1_prover`)."""

    def __init__(self, words: np.ndarray | None, n: int) -> None:
        self.words, self.n = words, int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, j: Any) -> Any:
        if self.words is None:
            raise LookupError("this registered value's rows aren't kept here: the prover rebuilds them from its own "
                              "(pouw_h1_prover.keep_call, its weights)")
        if isinstance(j, slice):
            return self.words[j]
        return tuple(int(v) for v in self.words[j])


def commit(program: str, value: str, words: np.ndarray, key: bytes, nonce: int, keep_rows: bool = False,
           **binding: Any) -> G.Registered:
    """`registered.Registered.commit(program, value, rows, salts, schema=SCHEMA_V2, **binding)` of ``words`` ``(n, W)``
    u16, ``rows`` their rows and ``salts`` row j's `audit.hm96_salt(key, nonce, j)`. ``binding``: the domain's other
    arguments (``window``, ``produced``). ``keep_rows``: hold the rows (`Rows`), else none."""
    words = np.asarray(words)
    if words.ndim != 2 or not words.shape[0] or not words.shape[1] or words.dtype != np.uint16:
        raise ValueError("a registered value is at least one row of 16-bit words, every row the same width")
    n, w = words.shape
    dom = G.domain(program, value, w, n, G.WORD_BITS, **binding)
    y = salts(key, nonce, n)
    b = (row_digests(words) ^ _hiding().masks(y.tobytes(), n)).tobytes()
    ys = y.tobytes()
    salt_h, leaf_h = hashlib.sha512(_INST.salt_prefix), hashlib.sha512(_INST.leaf_prefix(_INST.default_key))
    frame, schema = _Frame(dom.domain_id), _part(G.SCHEMA_V2.encode())
    bcs, leaves = [], []
    for j in range(n):
        h = salt_h.copy()
        h.update(ys[192 * j:192 * (j + 1)])
        bc = b[64 * j:64 * (j + 1)] + h.digest()
        g = leaf_h.copy()
        g.update(bc)
        u = _uint(j)
        bcs.append(bc)
        leaves.append(frame._h(frame.leaf, u + u + schema + _part(g.digest())))
    return G.Registered(program=program, value=value, words=int(w), word_bits=G.WORD_BITS,
                        rows=Rows(np.ascontiguousarray(words) if keep_rows else None, n), salts=Salts(key, nonce, n),
                        bcs=tuple(bcs), levels=frame.levels(leaves), schema=G.SCHEMA_V2, **binding)


__all__ = ["Rows", "Salts", "commit", "row_digests", "salts"]

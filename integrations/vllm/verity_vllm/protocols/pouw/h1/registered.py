"""H1's served tables registered at serving scale: `registered.Registered.commit` of rows of 16-bit words as
``hm96-sha512/row/v2`` bit rows, row j salted `audit.hm96_salt` of the run's key at the value's nonce, with the same
salts, ``b || c`` strings, tree and entry, computed from the rows' NumPy words instead of Python ints.

* **The rows' digests.** Row j's bit row is its words little-endian, so ``x_j = SHA-512(sha512_row_v2_prefix(16 W) ||
  words[j])`` is hashed straight from the array's memory, rows split among threads (`hashlib` releases the GIL on a
  buffer of 2 KiB or more, which every H1 row is: a z/y record is 6 KiB, an A or B row K + 4 bytes).
* **The salts.** ChaCha20 blocks 3j, 3j + 1, 3j + 2 under the key at the nonce, every row's at once
  (`crypto.chacha20.keystream`, `audit.chacha20_block`'s layout).
* **The commit strings.** ``b = x ⊕ M y`` gathers hm96's byte tables over every row (`hm96.batch.masks`), ``c``
  and the tree leaf hash from the instance's one-block prefixes, and the frame-v3-sha512 leaves, nodes and pads from
  their tag's and domain's constant prefix (`merkle._hash`'s message), so a row costs a few short SHA-512 calls.
* **What it holds.** The ``b || c`` strings and the tree's levels, 64 bytes a node. The salts are sampled again on read
  (`Salts`), and the rows are kept only with ``keep_rows`` (`Rows`): a window's call tables and its own ``b/<K>`` are
  rebuilt by the prover from what it keeps (`pouw_h1_prover.keep_call`, its weights), so `Served` keeps none of their
  rows and a window holds a few hundred bytes a row (its ``b || c`` and about two tree nodes, as Python bytes), where
  Python ints held some 36 bytes a word. The period's `Weights` keeps its rows, two bytes a word, for `Weights.matches`.

* **On the GPU**: where this process has CUDA, `commit` hashes the same strings, leaves and levels on the device
  (`pouw_h1_registered_device`, `pouw_cuda/registered_rows.cu`), a thread a row and a node, and only the domain id and
  the result's bytes touch the host (`path`, `MODE`), and `public_salts` derives a public table's salts there. The
  NumPy path above is the reference the device is held to
  (``tests/protocols/test_pouw_h1_registered_device.py``) and the fallback everywhere else.

`commit` returns a `registered.Registered` whose ``rows`` and ``salts`` are those sequences; everything `Served`,
`ServedTree.of`, `registered.verify_read` and the record read of it (``root``, ``levels``, ``bcs``, ``entry()``,
``path``) is what `Registered.commit` gives over the same rows and salts (``tests/protocols/
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
from verity.ml.commitments.hm96 import batch
from verity.ml.commitments.merkle import merkle_depth
from verity.ml.commitments.rowleaf import (ROW_SEG_BYTES, sha512_row_seg_prefix, sha512_row_seg_segment_prefix,
                                           sha512_row_seg_segments, sha512_row_v2_prefix)
from verity.ml.crypto import chacha20

_INST = hm96.SHA512
_FRAME = b"veritor/protocol/merkle/frame/v3\0"
#: `commit`'s path (`path`): ``"auto"``, ``"gpu"`` or ``"numpy"``, set by the caller that configures the run
MODE = "auto"
#: rows a salt pass, rows a digest task
_SALTED, _ROWS = 1 << 16, 1 << 11


@cache
def _pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max(1, min(16, os.cpu_count() or 1)), thread_name_prefix="h1-registered")


def salts(key: bytes, nonce: int, n: int) -> bytes:
    """Rows 0 .. n - 1's salts, concatenated: row j's `audit.hm96_salt(key, nonce, j)`."""
    return b"".join(chacha20.keystream(key, nonce, 3 * lo, 3 * min(_SALTED, n - lo)) for lo in range(0, n, _SALTED))


def _digests(words: np.ndarray, lo: int, hi: int, head: Any, out: np.ndarray) -> None:
    for j in range(lo, hi):
        h = head.copy()
        h.update(words[j])
        out[j] = np.frombuffer(h.digest(), np.uint8)


def _seg_digests(rows: np.ndarray, lo: int, hi: int, heads: list[Any], outer: Any, out: np.ndarray) -> None:
    for j in range(lo, hi):
        o = outer.copy()
        for s, h in enumerate(heads):
            h = h.copy()
            h.update(rows[j, ROW_SEG_BYTES * s:ROW_SEG_BYTES * (s + 1)])
            o.update(h.digest())
        out[j] = np.frombuffer(o.digest(), np.uint8)


def row_digests(words: np.ndarray, schema: str = G.SCHEMA_V2) -> np.ndarray:
    """Each row's ``x``, ``(n, 64)`` u8: `registered.row_digest(row, 16, schema)` of ``words`` ``(n, W)`` u16, ``schema``
    ``SCHEMA_V2`` or ``SCHEMA_SEG``."""
    words = np.ascontiguousarray(words, "<u2")
    n, w = words.shape
    out, rows = np.empty((n, 64), np.uint8), words.view(np.uint8).reshape(n, 2 * w)
    if schema == G.SCHEMA_V2:
        head = hashlib.sha512(sha512_row_v2_prefix(16 * w))
        list(_pool().map(lambda lo: _digests(rows, lo, min(lo + _ROWS, n), head, out), range(0, n, _ROWS)))
        return out
    if schema != G.SCHEMA_SEG:
        raise ValueError(f"rows of {schema!r}: {G.SCHEMA_V2} or {G.SCHEMA_SEG}")
    segs = sha512_row_seg_segments(16 * w)
    heads = [hashlib.sha512(sha512_row_seg_segment_prefix(j, len(segs), m)) for j, m in enumerate(segs)]
    outer = hashlib.sha512(sha512_row_seg_prefix(16 * w))
    list(_pool().map(lambda lo: _seg_digests(rows, lo, min(lo + _ROWS, n), heads, outer, out), range(0, n, _ROWS)))
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
    """A registered value's salts, drawn again on read: item j is `audit.hm96_salt(key, nonce, j)`; `at` draws many at
    once."""

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

    def at(self, positions: Sequence[int] | np.ndarray) -> bytes:
        """Items ``positions`` (each in [0, n)) concatenated, sampled in one pass (`crypto.chacha20.hm96_salts`)."""
        at = np.asarray(positions)
        if at.size and (int(at.min()) < 0 or int(at.max()) >= self.n):
            raise IndexError(f"a position outside the value's {self.n} rows")
        return chacha20.hm96_salts(self.key, self.nonce, at)


class Rows(Sequence):
    """A registered value's rows: item j row j's words as ints (a slice, the words ``(·, W)`` u16; `at`, any rows' words),
    when kept; else every read refused, the prover rebuilding them from what it keeps (`pouw_h1_prover`)."""

    def __init__(self, words: np.ndarray | None, n: int) -> None:
        self.words, self.n = words, int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, j: Any) -> Any:
        if isinstance(j, slice):
            return self._kept()[j]
        return tuple(int(v) for v in self._kept()[j])

    def at(self, positions: Sequence[int] | np.ndarray) -> np.ndarray:
        """Rows ``positions`` (each in [0, n)) as their words ``(·, W)`` u16, gathered at once."""
        at = np.asarray(positions, dtype=np.int64)
        if at.size and (int(at.min()) < 0 or int(at.max()) >= self.n):
            raise IndexError(f"a position outside the value's {self.n} rows")
        return self._kept()[at]

    def _kept(self) -> np.ndarray:
        if self.words is None:
            raise LookupError("this registered value's rows aren't kept here: the prover rebuilds them from its own "
                              "(pouw_h1_prover.keep_call, its weights)")
        return self.words


def path() -> str:
    """Where `commit` hashes, `MODE`'s choice: ``"gpu"`` (`pouw_h1_registered_device`, refused where it can't run),
    ``"numpy"``, or by default ``"auto"``, the GPU where this process has CUDA through torch and the kernels build for its
    device, else NumPy. Both register the same bytes; NumPy is the reference."""
    mode = MODE
    if mode not in ("auto", "gpu", "numpy"):
        raise ValueError(f"MODE={mode!r}: auto, gpu or numpy")
    if mode != "auto":
        return mode
    from verity_vllm.protocols.pouw.h1 import registered_device as D

    return "gpu" if D.available() else "numpy"


def commit(program: str, value: str, words: np.ndarray, key: bytes, nonce: int, keep_rows: bool = False,
           row_salts: bytes | None = None, schema: str = G.SCHEMA_V2, **binding: Any) -> G.Registered:
    """`registered.Registered.commit(program, value, rows, salts, schema=schema, **binding)` of ``words`` ``(n, W)``
    u16, ``rows`` their rows and ``salts`` row j's `audit.hm96_salt(key, nonce, j)`, or with ``row_salts`` (n salts
    concatenated, each used once) those, ``key`` and ``nonce`` unread. ``schema``: ``SCHEMA_V2``, or ``SCHEMA_SEG`` (an
    embedding table's rows, hashed with NumPy wherever `path` says). ``binding``: the domain's other arguments
    (``window``, ``produced``, ``session``). ``keep_rows``: hold the rows (`Rows`), else none. Hashed where `path` says."""
    words = np.asarray(words)
    if words.ndim != 2 or not words.shape[0] or not words.shape[1] or words.dtype != np.uint16:
        raise ValueError("a registered value is at least one row of 16-bit words, every row the same width")
    if schema not in (G.SCHEMA_V2, G.SCHEMA_SEG):
        raise ValueError(f"rows of {schema!r}: {G.SCHEMA_V2} or {G.SCHEMA_SEG}")
    n, w = words.shape
    dom = G.domain(program, value, w, n, G.WORD_BITS, **binding)
    given = None if row_salts is None else bytes(row_salts)
    fixed = None if given is None else tuple(given[192 * j:192 * (j + 1)] for j in range(n))
    if fixed is not None and (len(given) != 192 * n or len(set(fixed)) != n):
        raise ValueError(f"one {_INST.salt_bytes}-byte salt per row, never reused")
    if schema == G.SCHEMA_V2 and path() == "gpu":
        from verity_vllm.protocols.pouw.h1 import registered_device as D

        bcs, levels = D.device_commit(words, key, nonce, given, dom.domain_id)
    else:
        bcs, levels = _numpy(words, key, nonce, given, dom.domain_id, schema)
    return G.Registered(program=program, value=value, words=int(w), word_bits=G.WORD_BITS,
                        rows=Rows(np.ascontiguousarray(words) if keep_rows else None, n),
                        salts=Salts(key, nonce, n) if fixed is None else fixed,
                        bcs=bcs, levels=levels, schema=schema, **binding)


def public_salts(head: bytes, first: int, n: int) -> bytes:
    """Positions ``first`` .. ``first + n - 1``'s public row salts concatenated, each ``SHA-512(head || u64be(position) ||
    u8(i))`` for i = 0, 1, 2 (`core_check.public_row_salt`, ``head`` its domain, context length and context), on the
    device where `path` says, else with `hashlib`."""
    if n < 1 or first < 0 or first + n > 1 << 64:
        raise ValueError(f"{n} public salts from position {first}: at least one, every position a u64")
    if path() == "gpu":
        from verity_vllm.protocols.pouw.h1 import registered_device as D

        return D.device_public_salts(head, first, n)
    return b"".join(hashlib.sha512(head + u.to_bytes(8, "big") + bytes([i])).digest()
                    for u in range(first, first + n) for i in range(_INST.salt_bytes // 64))


def _numpy(words: np.ndarray, key: bytes, nonce: int, given: bytes | None, domain_id: bytes,
           row_schema: str = G.SCHEMA_V2) -> tuple[tuple[bytes, ...], tuple[tuple[bytes, ...], ...]]:
    """`commit`'s ``b || c`` strings and tree levels from NumPy and `hashlib`: the reference."""
    n = words.shape[0]
    ys = salts(key, nonce, n) if given is None else given
    b = (row_digests(words, row_schema)
         ^ np.frombuffer(batch.masks(_INST, _INST.default_key, ys), np.uint8).reshape(n, 64)).tobytes()
    salt_h, leaf_h = hashlib.sha512(_INST.salt_prefix), hashlib.sha512(_INST.leaf_prefix(_INST.default_key))
    frame, schema = _Frame(domain_id), _part(row_schema.encode())
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
    return tuple(bcs), frame.levels(leaves)


__all__ = ["MODE", "Rows", "Salts", "commit", "path", "public_salts", "row_digests", "salts"]

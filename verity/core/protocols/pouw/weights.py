"""An H1 period's registered weights (the model registry): each contraction K's ``<prefix>b/<K>``, the B
rows of the window Program's linears of that K, and `weights_root`, their fold, the weights' root a period-open signs.

* **A B row** (`row_bytes`) is one output row of a linear: its K int8 codes, then the 4 bytes of its f32 scale s_w,
  little-endian (`h1zy.ROW`): `b_words(K)` u16 words, as a ``hm96-sha512/row/v2`` bit row is its bytes.
* **``<prefix>b/<K>``** is the rows of K's linears, linear after linear in the window's order, committed as
  `registered.Registered.commit` commits them: under `registered.domain` of the circuit's SHA-512, the name, `b_words(K)`
  words and Σn leaves, binding no window (a value held across windows), in ``hm96-sha512/row/v2``, row j's salt
  `audit.hm96_salt` of the registration's 32-byte key at `B_NONCE` plus K's place among the Ks, ascending. The auditor
  hands that key to every lane serving the model (`model-key`), so it salts these rows only, never a window's A or z/y:
  those stay under the prover's own key.
* `weights_roots` computes each K's root streaming, in memory independent of the row count: it cuts K's rows into aligned
  chunks of at most `CHUNK_BYTES` (a power of two rows each), and each chunk's subtree root, pads past Σn included, is
  folded in order as `tree_levels` folds the leaves. The caller's process copies each row into one of ``2 * workers + 1``
  shared slots of a chunk each, which `workers` forked processes read in place, so the auditor's host holds about
  ``(2 * workers + 1) * CHUNK_BYTES`` and `workers` processes: the default, `DEFAULT_WORKERS`, fits a challenger host
  of 4 vCPU and 15 GB. A chunk's salts are ChaCha20's keystream in numpy (`verity.ml.crypto.chacha20`, the
  blocks `audit.hm96_salt` concatenates) and its hm96 leaves `verity.ml.commitments.hm96.batch`'s. Its roots are
  `Registered.commit`'s for any `workers`, and `entry` is the registration's entry (`Registered.entry`), which credit's
  `_weights` checks.
* `weights_root` is credit's fold of the roots: `tagged` of ``verity/pouw/h1/weights/v1`` and each ``K, root`` by
  ascending K.
* **Chunk rows.** Past `CHUNK_MAX` codes a row of K is `chunks` (K) = n leaves, leaf n·r + c chunk c's K/n codes and
  the row's s_w (`leaf_words` (K) words), and credit's `_weights` reads ``<prefix>b/<K>`` as n·Σn such leaves.
  `weights_roots` and `entry` still commit whole rows, so at such a K their entry is not yet the one credit accepts.
"""
from __future__ import annotations

import functools
import hashlib
import itertools
import mmap
import multiprocessing
import os
from collections import deque
from typing import Any, Callable, Iterable, Mapping, Sequence

from verity.core.protocols.pouw.protocol import tagged
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.commitments import domains, hm96
from verity.ml.commitments.hm96 import batch as hm96_batch
from verity.ml.commitments.merkle import merkle_depth
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix
from verity.ml.crypto import chacha20

#: an H1 window's weights' root: each K's registered ``…/b/<K>`` root
WEIGHTS_TAG = domains.POUW_H1_WEIGHTS
#: ``b/<K>``'s first salt nonce, past every round's and call's tables in the prover's salt stream (vllm's
#: ``REQUESTS_NONCE + 2^42 + 2^40``), so a key that salts both never repeats a (nonce, leaf)
B_NONCE = (1 << 61) + (1 << 42) + (1 << 40)
SCHEMA = G.SCHEMA_V2
_INST = hm96.SHA512
#: a chunk's rows are at most this many bytes, and from 2^4 to 2^14 rows (fewer only when K's tree is smaller)
CHUNK_BYTES = 1 << 23
#: `weights_roots`'s processes unless the caller says: memory ~ (2 * DEFAULT_WORKERS + 1) * CHUNK_BYTES and the processes
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))
#: hm96_salt's ChaCha20 blocks per leaf (a 192-byte salt)
_SALT_BLOCKS = _INST.salt_bytes // chacha20.BLOCK_BYTES


def b_words(k: int) -> int:
    """A B row's 16-bit words: its K int8 codes, then the 4 bytes of its f32 s_w."""
    return (k + 5) // 2


#: the largest K an H1 lane is proved at whole, and a chunk's largest: K = 4096's lane fills C-Flock's 2^27 block
CHUNK_MAX = 4096
#: a chunk is whole blocks of its rows' and columns' noise streams (`circuit.seeds.BLOCK_BITS`, 512 bits)
CHUNK_ALIGN = 512


def chunks(k: int) -> int:
    """The chunks an H1 lane at contraction ``k`` is proved in (`h1zy.lane_statements`), and so the chunk rows a table at
    ``k`` commits each row as (`h1zy.chunk_leaves`): 1, the whole lane, at K <= `CHUNK_MAX`; past it the least n >= 2
    whose chunks K / n are at most `CHUNK_MAX` entries and a multiple of `CHUNK_ALIGN`. ValueError at a K with no such n.
    `PouwCheck.Chunks.chunks` is its Lean twin."""
    k = int(k)
    if k < 1:
        raise ValueError(f"K = {k}: a lane is proved only at a positive K")
    if k <= CHUNK_MAX:
        return 1
    for n in range(-(-k // CHUNK_MAX), k // CHUNK_ALIGN + 1):
        if k % (n * CHUNK_ALIGN) == 0:
            return n
    raise ValueError(f"K = {k}: no n splits it into chunks of at most {CHUNK_MAX} entries that are whole {CHUNK_ALIGN}-bit "
                     f"noise blocks")


def leaf_words(k: int) -> int:
    """A leaf's 16-bit words at K: one chunk's K / `chunks` (K) codes, then the row's f32 s_w."""
    return b_words(k // chunks(k))


def value(prefix: str, k: int) -> str:
    return f"{prefix}b/{k}"


def row_bytes(codes: bytes, scale: bytes) -> bytes:
    """A B row's bit row: its K int8 codes' bytes, then its f32 s_w's 4 bytes little-endian; ValueError on an odd K or a
    scale that isn't 4 bytes."""
    if len(codes) % 2 or not codes:
        raise ValueError(f"K = {len(codes)}: a row's codes are whole u16 words")
    if len(scale) != 4:
        raise ValueError(f"an s_w is an f32's 4 bytes, not {len(scale)}")
    return bytes(codes) + bytes(scale)


def row_words(row: bytes) -> list[int]:
    """A B row's u16 words, as `Registered.commit` takes them: its bytes little-endian in pairs."""
    return [row[i] | row[i + 1] << 8 for i in range(0, len(row), 2)]


def ks(linears: Sequence[tuple[int, int]]) -> list[int]:
    """The Ks of the window's linears (``(n, K)`` each), ascending; ValueError on an odd K or an empty linear."""
    if not linears or any(type(n) is not int or type(k) is not int or n < 1 or k < 2 or k % 2 for n, k in linears):
        raise ValueError("the window's linears are (n, K), n >= 1 and K a positive even number each")
    return sorted({k for _n, k in linears})


def chunk_bits(width: int, depth: int) -> int:
    """log2 of a chunk's rows for rows of `width` bytes in a tree of `depth` levels."""
    bits = max(4, min(14, (CHUNK_BYTES // width).bit_length() - 1))
    return min(bits, depth)


@functools.lru_cache(maxsize=8)
def _domain(circuit: str, name: str, words: int, leaves: int):
    return G.domain(circuit, name, words, leaves)


def _subtree(spec: tuple, ci: int, rows: bytes) -> bytes:
    """Chunk `ci`'s subtree root: its rows' leaves (`rows` their bytes, possibly none), then pads to 2^bits leaves."""
    circuit, name, words, total, key, nonce, width, bits = spec
    dom, j0, n = _domain(circuit, name, words, total), ci << bits, len(rows) // width
    level = []
    if n:
        row_h = hashlib.sha512(sha512_row_v2_prefix(8 * width))
        digests = bytearray()
        for i in range(n):
            h = row_h.copy()
            h.update(rows[width * i:width * (i + 1)])
            digests += h.digest()
        salts = chacha20.keystream(key, nonce, _SALT_BLOCKS * j0, _SALT_BLOCKS * n)
        tl = hm96_batch.tree_leaves(_INST, _INST.default_key, bytes(digests), salts)
        level = [dom.leaf(j0 + i, j0 + i, SCHEMA, tl[i]) for i in range(n)]
    level += [dom.pad(j) for j in range(j0 + n, j0 + (1 << bits))]
    g = j0
    for d in range(bits):
        level = [dom.node(d, (g >> 1) + i // 2, level[i], level[i + 1]) for i in range(0, len(level), 2)]
        g >>= 1
    return level[0]


class _Fold:
    """The root over 2^m subtree roots of level `bits`, taken in index order, as `tree_levels` folds above them."""

    def __init__(self, dom, bits: int):
        self.dom, self.bits, self.count, self.stack = dom, bits, 0, []

    def push(self, root: bytes) -> None:
        level, idx = self.bits, self.count
        self.count += 1
        while self.stack and self.stack[-1][0] == level:
            _, left = self.stack.pop()
            idx >>= 1
            root = self.dom.node(level, idx, left, root)
            level += 1
        self.stack.append((level, root))

    def root(self) -> bytes:
        if len(self.stack) != 1:
            raise RuntimeError(f"{self.count} subtrees aren't a power of two")
        return self.stack[0][1]


#: each `weights_roots` call's chunk slots, shared anonymous maps its forked workers inherit
_SLOTS: dict[int, list[mmap.mmap]] = {}
_CALLS = itertools.count()


def _slot_subtree(spec: tuple, ci: int, slot: tuple[int, int], nbytes: int) -> bytes:
    """A worker's `_subtree` of chunk `ci`, its rows the first `nbytes` of slot ``(call, index)``."""
    call, index = slot
    return _subtree(spec, ci, memoryview(_SLOTS[call][index])[:nbytes])


def _shape(linears, k: int) -> tuple[list[int], int, int, int, int]:
    """K's linears, Σn, row width, tree depth and chunk bits."""
    mine = [w for w, (_n, kw) in enumerate(linears) if kw == k]
    total, width = sum(linears[w][0] for w in mine), 2 * b_words(k)
    depth = merkle_depth(total)
    return mine, total, width, depth, chunk_bits(width, depth)


def weights_roots(circuit: str, prefix: str, linears: Sequence[tuple[int, int]], rows: Callable[[int], Iterable[bytes]],
                  key: bytes, *, progress: Callable[[int, int], None] | None = None,
                  workers: int | None = None) -> dict[int, bytes]:
    """Each K's ``<prefix>b/<K>`` root: `linears` the window's linears in its order, ``(n, K)`` each; ``rows(w)`` linear w's
    n rows, each `row_bytes` (any bytes-like object); `key` the registration's 32-byte salt key; `workers` the processes
    that commit chunks (`DEFAULT_WORKERS` when None; at 1, or where processes can't fork, the caller's own alone). Equal to
    `Registered.commit`'s root over the same rows and salts. ValueError on a row of another width, a linear of another
    row count, a key not 32 bytes or `workers` < 1."""
    if len(key) != 32:
        raise ValueError(f"a salt key is 32 bytes, not {len(key)}")
    workers = DEFAULT_WORKERS if workers is None else workers
    if type(workers) is not int or workers < 1:
        raise ValueError(f"workers is a positive integer, not {workers!r}")
    order = ks(linears)
    slot_bytes = max((1 << s[4]) * s[2] for s in (_shape(linears, k) for k in order))
    if workers == 1 or "fork" not in multiprocessing.get_all_start_methods():
        local = [memoryview(bytearray(slot_bytes))]
        return {k: _k_root(circuit, prefix, linears, rows, key, i, k, local, None, None, progress)
                for i, k in enumerate(order)}
    call = next(_CALLS)
    _SLOTS[call] = [mmap.mmap(-1, slot_bytes) for _ in range(2 * workers + 1)]
    pool = None
    try:
        pool = multiprocessing.get_context("fork").Pool(workers)
        views = [memoryview(m) for m in _SLOTS[call]]
        try:
            return {k: _k_root(circuit, prefix, linears, rows, key, i, k, views, pool, call, progress)
                    for i, k in enumerate(order)}
        finally:
            for v in views:
                v.release()
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        for m in _SLOTS.pop(call):
            m.close()


def _k_root(circuit, prefix, linears, rows, key, i, k, slots, pool, call, progress) -> bytes:
    mine, total, width, depth, bits = _shape(linears, k)
    spec = (circuit, value(prefix, k), b_words(k), total, bytes(key), B_NONCE + i, width, bits)
    fold, pending, free = _Fold(_domain(*spec[:4]), bits), deque(), list(range(len(slots)))[::-1]

    def settle() -> None:
        result, index = pending.popleft()
        fold.push(result.get())
        if index is not None:
            free.append(index)

    def submit(ci: int, index: int | None, nbytes: int) -> None:
        if pool is None:
            fold.push(_subtree(spec, ci, slots[0][:nbytes] if nbytes else b""))
            return
        slot = None if index is None else (call, index)
        pending.append((pool.apply_async(_slot_subtree if slot else _subtree,
                                         (spec, ci, slot, nbytes) if slot else (spec, ci, b"")), index))

    def take() -> int:
        while not free:
            settle()
        return free.pop()

    per, ci, j = (1 << bits) * width, 0, 0
    index, off = take(), 0
    for w in mine:
        got = 0
        for row in rows(w):
            if len(row) != width:
                raise ValueError(f"linear {w}'s row {got} is {len(row)} bytes, not K = {k}'s {width}")
            slots[index][off:off + width] = memoryview(row).cast("B")
            off += width
            if off == per:
                submit(ci, index, off)
                ci += 1
                index, off = (take(), 0) if pool is not None else (index, 0)
            if progress is not None and j % 65536 == 0:
                progress(k, j)
            got += 1
            j += 1
        if got != linears[w][0]:
            raise ValueError(f"linear {w} gave {got} rows, not its n = {linears[w][0]}")
    if off:
        submit(ci, index, off)
        ci += 1
    elif pool is not None:
        free.append(index)
    for c in range(ci, (1 << depth) >> bits):
        submit(c, None, 0)
        while len(pending) > 2 * len(slots):
            settle()
    while pending:
        settle()
    return fold.root()


def entry(circuit: str, prefix: str, k: int, n: int, root: bytes) -> dict[str, Any]:
    """``<prefix>b/<K>``'s registration entry (`Registered.entry`) of Σn = `n` rows with root `root`."""
    v = value(prefix, k)
    return {"value": v, "root": root.hex(), "leaves": n, "words": b_words(k),
            "domain": G.domain(circuit, v, b_words(k), n).domain_id.hex()}


def weights_root(roots: Mapping[int, bytes]) -> bytes:
    """The weights' root: `tagged` of `WEIGHTS_TAG` and each K's ``…/b/<K>`` root, ``K, root`` by ascending K."""
    return tagged(WEIGHTS_TAG, *(p for k in sorted(roots) for p in (k, roots[k])))


__all__ = ["B_NONCE", "CHUNK_ALIGN", "CHUNK_BYTES", "CHUNK_MAX", "DEFAULT_WORKERS", "SCHEMA", "WEIGHTS_TAG", "b_words",
           "chunk_bits", "chunks", "entry", "ks", "leaf_words", "row_bytes", "row_words", "value", "weights_root", "weights_roots"]

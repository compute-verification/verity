"""The block pool between two windows as one row, and its root: the row `CompileSlice_v6` carries from window to
window (`verity.protocols.certifier.prover.compile_slice_v6`), and the one a serving stack exports from its prefix cache.

**The row**, field-major, each element least significant bit first: the free queue's ``head`` and ``tail`` (NW bits
each, NW = bitlen(NB − 1)), then per block b = 1 … NB − 1 its links ``prev``, ``next``, ``cprev``, ``cnext`` (the free
queue's and those among the blocks cached under one key, 0 for none), ``mlast`` (the key's last copy where b is its first
copy, else 0) and ``key``, the 512-bit BlockKey_v1 key b is cached under (`gates.block_key`, `gates.rowleaf.Digest`
layout, 0 for none). Block b's fields are at index b − 1, and block 0, the free queue's sentinel, has none. It holds no
count, owner or stamp: a pool between windows holds no block.

**The root** is the row's unsalted `Sha512RowSeg_v1` digest (`rowleaf.sha512_row_seg_digest`). The empty pool, every
block free in order and nothing cached (`empty`), has the root `genesis`: a chain's first window opens it, and so does
the window after a failed one.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

from verity.commitments import rowleaf as RL
from verity.commitments.gates import rowleaf as RG

__all__ = ["LINKS", "NOKEY", "PoolRow", "decode", "empty", "encode", "fields", "genesis", "root", "row_bytes", "width"]

LINKS = ("prev", "next", "cprev", "cnext")
NOKEY = bytes(64)


class PoolRow(NamedTuple):
    """The pool between windows, block b's fields at index b − 1."""
    head: int
    tail: int
    prev: tuple[int, ...]
    next: tuple[int, ...]
    cprev: tuple[int, ...]
    cnext: tuple[int, ...]
    mlast: tuple[int, ...]
    key: tuple[bytes, ...]


def width(nb: int) -> int:
    """NW, a block number's bits in a pool of ``nb`` blocks (the null block's included)."""
    if nb < 2:
        raise ValueError(f"a pool of {nb} blocks has none to cache in")
    return max((nb - 1).bit_length(), 1)


def fields(nb: int) -> list[tuple[str, int, int]]:
    """The row's fields (name, count, bits) for a pool of ``nb`` blocks."""
    NW, n = width(nb), nb - 1
    return [("head", 1, NW), ("tail", 1, NW)] + [(f, n, NW) for f in LINKS + ("mlast",)] + [("key", n, 512)]


def empty(nb: int) -> PoolRow:
    """The empty pool of ``nb`` blocks: every block free, in order, nothing cached."""
    n = nb - 1
    width(nb)
    return PoolRow(1, n, tuple(range(n)), tuple(b + 1 if b < n else 0 for b in range(1, nb)), (0,) * n, (0,) * n,
                   (0,) * n, (NOKEY,) * n)


def _bits(v: int, w: int) -> list[int]:
    if v < 0 or v >> w:
        raise ValueError(f"{v} doesn't fit {w} bits")
    return [(v >> j) & 1 for j in range(w)]


def encode(row: PoolRow) -> list[int]:
    """The row's bits (`fields`), its pool's NB one more than its blocks."""
    nb = len(row.prev) + 1
    NW = width(nb)
    if any(len(getattr(row, f)) != nb - 1 for f in LINKS + ("mlast", "key")):
        raise ValueError("a pool row's fields hold one element a block")
    out = _bits(row.head, NW) + _bits(row.tail, NW)
    for f in LINKS + ("mlast",):
        out += [b for v in getattr(row, f) for b in _bits(v, NW)]
    return out + [b for k in row.key for b in RG.digest_bits(k)]


def decode(nb: int, bits: Sequence[int]) -> PoolRow:
    """The row of a pool of ``nb`` blocks whose bits (`encode`) are ``bits``."""
    fs, at, got = fields(nb), 0, {}
    if len(bits) != sum(n * w for _, n, w in fs):
        raise ValueError(f"a pool row of {nb} blocks holds {sum(n * w for _, n, w in fs)} bits, not {len(bits)}")
    for name, n, w in fs:
        elems = [list(bits[at + j * w: at + (j + 1) * w]) for j in range(n)]
        at += n * w
        got[name] = tuple(RG.digest_bytes(e) for e in elems) if name == "key" else \
            tuple(sum(int(b) << j for j, b in enumerate(e)) for e in elems)
    return PoolRow(got["head"][0], got["tail"][0], *(got[f] for f in LINKS + ("mlast", "key")))


def row_bytes(row: PoolRow) -> tuple[bytes, int]:
    """``rowleaf.bit_row((encode(row), 1))``, the row's bytes and bits, packed in one integer rather than a bit at a
    time: the row's elements as binary strings, last first, are its bits most significant first."""
    nb = len(row.prev) + 1
    NW = width(nb)
    if any(len(getattr(row, f)) != nb - 1 for f in LINKS + ("mlast", "key")):
        raise ValueError("a pool row's fields hold one element a block")
    for f in ("head", "tail", *LINKS, "mlast"):
        vs = getattr(row, f)
        for v in (vs,) if isinstance(vs, int) else (min(vs), max(vs)):
            if v < 0 or v >> NW:
                raise ValueError(f"{v} doesn't fit {NW} bits")
    if any(len(k) != 64 for k in row.key):
        raise ValueError(f"a digest is 64 bytes, not {next(len(k) for k in row.key if len(k) != 64)}")
    fmt = f"0{NW}b"
    parts = [format(int.from_bytes(k[i:i + 8], "big"), "064b") for k in reversed(row.key) for i in range(56, -8, -8)]
    for f in reversed(LINKS + ("mlast",)):
        parts += [format(v, fmt) for v in reversed(getattr(row, f))]
    s = "".join(parts) + format(row.tail, fmt) + format(row.head, fmt)
    return int(s, 2).to_bytes((len(s) + 7) // 8, "little"), len(s)


def root(row: PoolRow) -> bytes:
    """The row's root: its unsalted `Sha512RowSeg_v1` digest."""
    return RL.sha512_row_seg_digest(*row_bytes(row))


def genesis(nb: int) -> bytes:
    """The empty pool's root."""
    return root(empty(nb))

"""The Commit's ``rows-blake3`` research arm: :mod:`step_rows`'s construction with keyed BLAKE3 in every hash position (the hash
loop's sibling arm, ``benchmarks/hash/loop``).  It never serves and nothing opens or verifies it: it measures what the committer's step
costs when SHA-512's compressions are BLAKE3's.  The format is the hypothetical hm96-blake3 of the hash lane's intpipe
(``benchmarks/hash/intpipe/hm96_ref.py``, ``Prim("blake3k")``): the same tags, salts, domain and tree as hm96-sha512, sizes scaled to
BLAKE3's 32-byte digest and 64-byte block, H = BLAKE3 keyed by the public ``HASH_KEY``:

  x       ``H(Q(n_bits) || row)``, Q the ``sha512/row/v2`` tag and u64be n_bits zero padded to 64 bytes
  y       ``H(Ks || u64be(pos) || u64be(r))`` for r < 4, concatenated                                      128 bytes, Ks 32 bytes
  b || c  ``b = x ^ M_key y`` (the key 160 bytes from hm96-sha512's key label under H), ``c = H(salt_prefix || y)``
  h       ``H(leaf_prefix || b || c)``, leaf_prefix = the leaf tag and H(key), zero padded to 64 bytes
  leaf    ``H(frame "leaf" did uint(pos) uint(pos) "hm96-sha512/row/v2" h)``; pads and nodes framed as frame-v3's, under H

``did`` is H of the step domain's frame-v3-sha512 domain message (:func:`step_rows.domain`'s).  ``native_rows_b3.cu`` hashes the same
bytes into 64-byte slots (the digest, then 32 zero bytes), so the committer's windows and trees are :mod:`step_rows`'s; the step root
and run root stay SHA-512 over the 32-byte tree root.
"""
from __future__ import annotations

import functools
from collections.abc import Sequence

from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import CommitmentDomain, domain_message
from verity.ml.commitments.rowleaf import SHA512_ROW_V2_TAG
from verity.ml.crypto import blake3 as b3
from verity_vllm.commit import step_rows

NAME = "rows-blake3"
HASH_KEY = b3.blake3(b"verity/hash-intpipe/hm96-blake3k/hash-key/v1 (hypothetical)")
DIGEST_BYTES, BLOCK = 32, 64
SALT_BYTES = 2 * DIGEST_BYTES + 64
KEY_BYTES = DIGEST_BYTES
HM96_KEY_BYTES = 3 * DIGEST_BYTES + 64
T_BYTES = 256 * HM96_KEY_BYTES
ROW_BYTES = step_rows.ROW_BYTES
ROOT_BYTES = step_rows.ROOT_BYTES
SCHEMA = step_rows.SCHEMA
SLOT = 64
_FRAME = b"veritor/protocol/merkle/frame/v3\0"
_TAGS = step_rows.INST
domain = step_rows.domain
run_root = step_rows.run_root


def h(data: bytes) -> bytes:
    return b3.keyed(HASH_KEY, data)


def _pad(b: bytes) -> bytes:
    return b.ljust(BLOCK, b"\0")


@functools.cache
def key() -> bytes:
    raw = bytearray(b"".join(h(_TAGS.key_label + i.to_bytes(4, "big")) for i in range(-(-HM96_KEY_BYTES // DIGEST_BYTES)))[:HM96_KEY_BYTES])
    raw[-1] &= (1 << (8 * DIGEST_BYTES + 8 * SALT_BYTES - 1 - 8 * (HM96_KEY_BYTES - 1))) - 1
    return bytes(raw)


def row_prefix(n_bits: int) -> bytes:
    return _pad(SHA512_ROW_V2_TAG + int(n_bits).to_bytes(8, "big"))


SALT_PREFIX = _pad(_TAGS.salt_tag)


@functools.cache
def leaf_prefix() -> bytes:
    return _pad(_TAGS.leaf_tag + h(key()))


@functools.cache
def _t_rows() -> tuple[int, ...]:
    k = int.from_bytes(key(), "little")
    rows = [0] * 256
    for v in range(1, 256):
        rows[v] = rows[v & (v - 1)] ^ (k >> ((v & -v).bit_length() - 1))
    return tuple(rows)


def t_table() -> bytes:
    """``T_v`` = the XOR of ``(K >> s)`` over the set bits ``s`` of ``v``: 256 rows of 160 bytes, little-endian (the kernel's mask
    table, read from global memory)."""
    return b"".join(r.to_bytes(HM96_KEY_BYTES, "little") for r in _t_rows())


def did(dom: CommitmentDomain) -> bytes:
    return h(domain_message(dom.binding, dom.owner, dom.positions.identity_sha512(), dom.count, "sha512"))


def salt(ks: bytes, pos: int) -> bytes:
    if len(ks) != KEY_BYTES:
        raise ValueError(f"a {NAME} salt key is {KEY_BYTES} bytes")
    return b"".join(h(ks + int(pos).to_bytes(8, "big") + r.to_bytes(8, "big")) for r in range(SALT_BYTES // DIGEST_BYTES))


def mask(y: bytes) -> bytes:
    rows, acc = _t_rows(), 0
    for i, v in enumerate(y):
        acc ^= rows[v] >> (8 * i)
    return (acc & ((1 << (8 * DIGEST_BYTES)) - 1)).to_bytes(DIGEST_BYTES, "little")


def hiding_leaf(row: bytes, y: bytes, n_bits: int) -> tuple[bytes, bytes]:
    """``(b || c, h)`` for a row of ``n_bits`` bits and its salt."""
    x = h(row_prefix(n_bits) + row)
    bc = bytes(p ^ q for p, q in zip(x, mask(y), strict=True)) + h(SALT_PREFIX + y)
    return bc, h(leaf_prefix() + bc)


def _uint(v: int) -> bytes:
    return v.to_bytes(max(1, (v.bit_length() + 7) // 8), "big")


def _message(tag: bytes, *parts: bytes) -> bytes:
    return _FRAME + len(tag).to_bytes(4, "big") + tag + b"".join(len(p).to_bytes(8, "big") + p for p in parts)


def leaf(d: bytes, pos: int, hl: bytes) -> bytes:
    return h(_message(b"leaf", d, _uint(pos), _uint(pos), SCHEMA.encode(), hl))


def pad(d: bytes, rank: int) -> bytes:
    return h(_message(b"pad", d, _uint(rank)))


def node(d: bytes, level: int, index: int, left: bytes, right: bytes) -> bytes:
    return h(_message(b"node", d, _uint(level), _uint(index), left, right))


def levels(dom: CommitmentDomain, rows: Sequence[bytes], ks: bytes, n_bits: int) -> tuple[tuple[bytes, ...], ...]:
    """Every level of the step's tree (leaves first, padded to a power of two), 32-byte digests."""
    d, n = did(dom), len(rows)
    width = 1 << max(0, (n - 1).bit_length())
    lv = [[leaf(d, i, hiding_leaf(r, salt(ks, i), n_bits)[1]) for i, r in enumerate(rows)] + [pad(d, r) for r in range(n, width)]]
    for level in range(width.bit_length() - 1):
        cur = lv[-1]
        lv.append([node(d, level, i, cur[2 * i], cur[2 * i + 1]) for i in range(len(cur) // 2)])
    return tuple(tuple(x) for x in lv)


def slot(digest: bytes) -> bytes:
    """A digest as the kernels store it: 64 bytes, the digest then zeros."""
    return digest.ljust(SLOT, b"\0")


def step_root(program: bytes, ctx: bytes, geo: bytes, layout: bytes, n: int, tree_root: bytes) -> bytes:
    return step_rows.step_root(program, ctx, geo, layout, n, slot(tree_root[:DIGEST_BYTES]))

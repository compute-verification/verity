"""hm96-sha512/v1's hiding leaf as a public Boolean Definition: what a unit computes from a private row digest and salt, under
the pinned key (`hm96.SHA512.default_key`, a circuit constant: another key is another circuit).

    Hm96Sha512Leaf_v1   (x: Digest, y: Array<24, Word>) -> Array<3, Digest>       (b, c, leaf)
        Digest = Array<8, Word>, Word = Array<64, Value<1>>

With M the 512 x 1536 Hankel matrix of the key over GF(2) (``hm96/PROTOCOL.md``):

    c    = SHA-512(salt_prefix || y)                 two compressions from salt_prefix's constant midstate
    b    = x XOR (M * y)                             XORs only: a balanced tree per row, `Instance.xor_count` of them
    leaf = SHA-512(leaf_prefix(key) || b || c)       two compressions from leaf_prefix's constant midstate, the second over
                                                     a constant block (its schedule free)

``x`` is the unsalted 64-byte digest (a ``sha512/row/v2`` or ``sha512/row-seg/v1`` row digest), ``y`` the 192-byte salt, and
``leaf`` the 64 bytes a frame-v3-sha512 row leaf binds under ``hm96-sha512/row/v2`` or ``hm96-sha512/row-seg/v1``
(`gates.frame_v3`); ``b || c`` is the commit string. Bytes are big-endian in 64-bit words, LSB first (`sha512.words_be`), and bit
t of a byte string is bit t & 7 of byte t >> 3, as `hm96.Instance.mask` reads it.
"""
from __future__ import annotations

from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import const_bits
from verity.ml.circuits.types import Array

from .. import hm96
from .sha512 import commit, compress_gadget, midstate, pad_tail, words_be

__all__ = ["Hm96Sha512Leaf", "arguments", "leaf_gadget", "reference"]

INSTANCE = hm96.SHA512
WORD = Array(64, BL.BIT)
DIGEST = Array(8, WORD)
SALT = Array(INSTANCE.salt_bytes // 8, WORD)


def reference(x: bytes, y: bytes) -> tuple[bytes, bytes, bytes]:
    """``Hm96Sha512Leaf_v1``'s reference: (b, c, leaf) under the pinned key (`hm96.Instance.commit_string`, `tree_leaf`)."""
    key = INSTANCE.default_key
    bc = INSTANCE.commit_string(key, x, y)
    return bc[:64], bc[64:], INSTANCE.tree_leaf(key, bc)


def _bits(data: bytes) -> list[int]:
    return [(w >> j) & 1 for w in words_be(data) for j in range(64)]


def arguments(x: bytes, y: bytes) -> tuple[list[int], list[int]]:
    """The two arguments, each a flat list of bits in leaf order."""
    if len(x) != INSTANCE.digest_bytes or len(y) != INSTANCE.salt_bytes:
        raise ValueError(f"x is {INSTANCE.digest_bytes} bytes and y {INSTANCE.salt_bytes}")
    return _bits(x), _bits(y)


def _bit(words: Sequence[Sequence], t: int):
    """Bit t of the byte string held in ``words`` (bit t & 7 of byte t >> 3)."""
    byte = t >> 3
    return words[byte // 8][8 * (7 - byte % 8) + (t & 7)]


def _xor_tree(terms: list):
    while len(terms) > 1:
        terms = [terms[k] ^ terms[k + 1] for k in range(0, len(terms) - 1, 2)] + terms[len(terms) & ~1:]
    return terms[0]


def _sha512(C, cv: list[int], blocks: Sequence[list]) -> list[list]:
    st = [const_bits(v, 64) for v in cv]
    for block in blocks:
        st = compress_gadget(C, st, block)
    return st


def _const_words(data: bytes) -> list[list]:
    return [const_bits(v, 64) for v in words_be(data)]


def leaf_gadget(C, x: Sequence[Sequence], y: Sequence[Sequence]) -> list[list]:
    """b, c and leaf (24 words of forms) of the digest ``x`` (8 words) and salt ``y`` (24 words)."""
    key = INSTANCE.default_key
    rows = INSTANCE.rows_of(key)
    ys = [_bit(y, j) for j in range(INSTANCE.cols)]
    bits = [_xor_tree([_bit(x, i)] + [ys[j] for j in range(INSTANCE.cols) if rows[i] >> j & 1]) for i in range(INSTANCE.rows)]
    # a committed b keeps the 769-term rows out of every form the hashes build on it
    b = [commit(C, [bits[8 * (8 * k + 7 - p // 8) + p % 8] for p in range(64)]) for k in range(8)]
    total = len(INSTANCE.salt_prefix) + INSTANCE.salt_bytes
    c = _sha512(C, midstate(INSTANCE.salt_prefix), [list(y[:16]), list(y[16:]) + _const_words(pad_tail(total))])
    prefix = INSTANCE.leaf_prefix(key)
    leaf = _sha512(C, midstate(prefix), [b + c, _const_words(pad_tail(len(prefix) + INSTANCE.proven_bytes))])
    return b + c + leaf


Hm96Sha512Leaf = TR.definition(
    "Hm96Sha512Leaf", 1, lambda C, *e: leaf_gadget(C, e[:8], e[8:]), (("x", DIGEST), ("y", SALT)), Array(3, DIGEST),
    doc="hm96-sha512/v1 under its pinned key: b = x XOR M_key y, c = SHA-512(salt_prefix || y) and the tree leaf "
        "SHA-512(leaf_prefix(key) || b || c), of the 64-byte digest x and the 192-byte salt y")

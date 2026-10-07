"""SHA-512 in Flock's circuit model, for ``verity/flock-circuit``'s serving rows on ``frame-v3-sha512`` with ``hm96-sha512/v1``
leaves (``verity.ml.commitments.hm96`` §5, ``frame_v3`` PROTOCOL.md §6).

* :func:`compression`: one SHA-512 compression, ``out = F(cv, msg)`` with the feed-forward: core's
  ``verity.ml.commitments.gates.sha512.compress_gadget`` (58,120 ANDs), whose reference, constants and padding this module re-exports.
* :func:`compression3`: three independent compressions (a 2^18 slot, 99% full).
* :func:`hm96_row`: a row's hiding commit string from its chaining value after the last row block and its salt ``y``: the
  padding block's compression gives the inner digest ``x`` (``sha512/row/v1``), ``c = SHA-512(salt_prefix || y)`` (two
  compressions from the prefix's constant midstate) and ``b = x XOR M(key) y`` (XOR only); output ``b || c``.

Words are 64-bit values as little-endian bit lists (bit i of a word is its value's bit i); a digest's bytes are its words
big-endian. :func:`word_bit` maps a byte-string bit (bit ``t & 7`` of byte ``t >> 3``, hm96's order) to its word bit.
"""
from __future__ import annotations

import hashlib

from verity.ml.commitments.gates.sha512 import IV, compress_gadget, compress_ref, midstate, pad_tail, words_be  # noqa: F401

from .gf2 import Circuit, const_bits


def _words(bits: list, n: int) -> list[list]:
    return [bits[64 * i:64 * i + 64] for i in range(n)]


def compression(n: int = 1) -> Circuit:
    """``n`` independent compressions: inputs ``cv{i}`` (8 words) and ``msg{i}`` (16 words), outputs ``out{i}`` (8 words), words
    concatenated; ``n == 1`` names them ``cv``, ``msg``, ``out``."""
    C = Circuit()
    ins = [(C.input(f"cv{i}" if n > 1 else "cv", 512), C.input(f"msg{i}" if n > 1 else "msg", 1024)) for i in range(n)]
    for i, (cv, msg) in enumerate(ins):
        C.output(f"out{i}" if n > 1 else "out", [b for w in compress_gadget(C, _words(cv, 8), _words(msg, 16)) for b in w])
    return C


def compression3() -> Circuit:
    return compression(3)


def word_bit(t: int) -> int:
    """The word-order position of byte-string bit ``t`` (bit ``t & 7`` of byte ``t >> 3``; words big-endian)."""
    byte, b = t >> 3, t & 7
    return 64 * (byte >> 3) + 56 - 8 * (byte & 7) + b


def hm96_row(key: bytes | None = None) -> Circuit:
    """Inputs ``cv`` (the chaining value after the row's last block, 8 words), ``pad`` (the padding block, 16 message words)
    and ``y`` (the salt, 1536 bits in byte-string order); output ``bc`` (``b || c``, 1024 bits in byte-string order)."""
    from verity.ml.commitments import hm96

    inst = hm96.SHA512
    key = inst.default_key if key is None else key
    C = Circuit()
    cv, pad, y = C.input("cv", 512), C.input("pad", 1024), C.input("y", 1536)
    x = [b for w in compress_gadget(C, _words(cv, 8), _words(pad, 16)) for b in w]
    # the salt's message words: byte j of the salt is bits [8j, 8j + 8) of y; word i is bytes 8i..8i+7, big-endian
    yw = [[None] * 64 for _ in range(24)]
    for t in range(1536):
        p = word_bit(t)
        yw[p >> 6][p & 63] = y[t]
    mid = midstate(inst.salt_prefix)
    st = compress_gadget(C, [const_bits(v, 64) for v in mid], yw[:16])
    tail = words_be(bytes(64) + pad_tail(128 + 192)[:64])[8:]
    st = compress_gadget(C, st, yw[16:] + [const_bits(v, 64) for v in tail])
    cbits = [b for w in st for b in w]
    xs = [x[word_bit(t)] for t in range(512)]
    rows = inst.rows_of(key)
    bits = []
    for i in range(512):
        acc = xs[i]
        r = rows[i]
        while r:
            j = (r & -r).bit_length() - 1
            acc = acc ^ y[j]
            r &= r - 1
        bits.append(acc)
    bits += [cbits[word_bit(t)] for t in range(512)]
    C.output("bc", bits)
    return C


def sha512_row_prefix(n_words: int) -> bytes:
    """``sha512/row/v1``'s first block for an x row of ``n_words`` u16 words (``verity.ml.commitments.rowleaf``)."""
    from verity.ml.commitments.rowleaf import ROLE_X
    from verity.ml.commitments.rowleaf import sha512_row_prefix as core

    return core(ROLE_X, 16, n_words)


def row_digest_ref(prefix: bytes, row: bytes) -> bytes:
    return hashlib.sha512(prefix + row).digest()

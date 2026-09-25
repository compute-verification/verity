"""Row leaves of frame-v3: one tensor row per leaf, committed as a row digest under the SHA-256 indexed trees of :mod:`merkle`.

A leaf is one *row* of a tensor in the orientation the GEMM consumes it -- a row of the activation matrix ``x`` (K
words), a column of the weight matrix ``W`` (K words).  The row's digest is the leaf *value* under
:meth:`CommitmentDomain.leaf` with the schema's string; ranks are row indices and the tree above is the ordinary
SHA-256 ``node`` / pad framing, so a verifier that has the digest checks an opening with SHA-256 alone.  A backend
proves only the row digest (its ``leaf_layout``) and publishes it.

Three schemas:

``sha256/row/v1``            ``SHA-256(sha256_row_prefix(role, word_bits, n_words) || row_bytes)``: a 64-byte constant
                             first block (tag, role, word width, row length), so the gadget starts from a fixed
                             midstate.
``blake3-keyed/row/v2``      ``BLAKE3(row_bytes, key=blake3_row_key(role))``: B-Ligero's ``leaf = blake3`` framing
                             (``backends/direct/ligero/leaf/blake3.py``), promoted byte for byte; its schema string and
                             keys are pinned by ``ligero-verify``.
``poseidon2-babybear-w24/row/v2h``  the Poseidon2 row digest (algebraic: kept so recorded results keep verifying, not
                             admissible in a core scheme's Table 2 cell).

``row_bytes`` are the row's words little-endian, ``word_bits`` 8, 16 or 32 bits each, in row order.  Roles are
:data:`ROLE_X` (1, an x row) and :data:`ROLE_W` (2, a W column).  Unsalted: a low-entropy row is preimage-searchable.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from . import blake3 as _blake3
from . import poseidon2_babybear as h2
from .merkle import CommitmentDomain
from .scheme import LeafLayout

__all__ = [
    "ROLE_W",
    "ROLE_X",
    "SCHEMA_BLAKE3_ROW",
    "SCHEMA_ROW",
    "SCHEMA_SHA256_ROW",
    "blake3_row_digest",
    "blake3_row_key",
    "blake3_row_layout",
    "row_bytes",
    "row_digest",
    "row_leaf",
    "row_leaf_value",
    "sha256_row_digest",
    "sha256_row_layout",
    "sha256_row_prefix",
]

SCHEMA_ROW = "poseidon2-babybear-w24/row/v2h"
SCHEMA_SHA256_ROW = "sha256/row/v1"
SCHEMA_BLAKE3_ROW = "blake3-keyed/row/v2"

ROLE_X, ROLE_W = h2.ROLE_X, h2.ROLE_W
_ROLE_KEY_NAME = {ROLE_X: b"x", ROLE_W: b"w"}
SHA256_ROW_TAG = b"verity/sha256-row/v1\0"
BLAKE3_KEY_PREFIX = b"verity/blake3-leaf/v1/"


def _role(role: int) -> int:
    if role not in _ROLE_KEY_NAME:
        raise ValueError(f"row role must be {ROLE_X} (x row) or {ROLE_W} (W column), not {role!r}")
    return role


def _width(word_bits: int) -> int:
    if word_bits not in (8, 16, 32):
        raise ValueError(f"row words are 8, 16 or 32 bits, not {word_bits!r}")
    return word_bits


def row_bytes(words: Sequence[int], word_bits: int) -> bytes:
    """The row's words as little-endian ``word_bits``-bit integers, in order: the bytes both row digests hash."""

    n = _width(word_bits) // 8
    bound = 1 << word_bits
    out = bytearray()
    for w in words:
        w = int(w)
        if not 0 <= w < bound:
            raise ValueError(f"row word {w} does not fit {word_bits} bits")
        out += w.to_bytes(n, "little")
    return bytes(out)


# --- sha256/row/v1 -----------------------------------------------------------------------------------------------


def sha256_row_prefix(role: int, word_bits: int, n_words: int) -> bytes:
    """The 64-byte first block: ``"verity/sha256-row/v1\\0" || u8 role || u8 word_bits || u32be n_words``, zero padded."""

    if not 0 < n_words < 1 << 32:
        raise ValueError("n_words must be a positive u32")
    head = SHA256_ROW_TAG + bytes([_role(role), _width(word_bits)]) + n_words.to_bytes(4, "big")
    return head.ljust(64, b"\0")


def sha256_row_layout(role: int, word_bits: int, n_words: int) -> LeafLayout:
    return LeafLayout("sha256", sha256_row_prefix(role, word_bits, n_words), n_words * word_bits // 8)


def sha256_row_digest(words: Sequence[int], word_bits: int, role: int) -> bytes:
    return sha256_row_layout(role, word_bits, len(words)).digest(row_bytes(words, word_bits))


# --- blake3-keyed/row/v2 -----------------------------------------------------------------------------------------


def blake3_row_key(role: int) -> bytes:
    """``"verity/blake3-leaf/v1/x"`` (x rows) or ``".../w"`` (W columns), zero padded to 32 bytes."""

    return (BLAKE3_KEY_PREFIX + _ROLE_KEY_NAME[_role(role)]).ljust(_blake3.KEY_LEN, b"\0")


def blake3_row_layout(role: int, word_bits: int, n_words: int) -> LeafLayout:
    if not 0 < n_words:
        raise ValueError("n_words must be positive")
    return LeafLayout("blake3", b"", n_words * _width(word_bits) // 8, key=blake3_row_key(role))


def blake3_row_digest(words: Sequence[int], word_bits: int, role: int) -> bytes:
    return blake3_row_layout(role, word_bits, len(words)).digest(row_bytes(words, word_bits))


# --- poseidon2-babybear-w24/row/v2h ------------------------------------------------------------------------------


def row_leaf_value(digest) -> bytes:
    """The 8 Poseidon2 digest lanes as 32 bytes (each lane a big-endian u32)."""
    d = [int(x) for x in digest]
    if len(d) != h2.OUT or any(not 0 <= x < h2.P for x in d):
        raise ValueError("a row digest is 8 BabyBear lanes")
    return b"".join(x.to_bytes(4, "big") for x in d)


def row_digest(words, word_bits: int, role: int) -> list[int]:
    """Poseidon2 digest (8 lanes) of one row of ``word_bits``-bit words (``role``: ``h2.ROLE_X`` or ``h2.ROLE_W``)."""
    return h2.hash_row(words, word_bits, role)


def row_leaf(domain: CommitmentDomain, rank: int, digest, schema: str = SCHEMA_ROW) -> bytes:
    """The frame-v3 leaf of the row at ``rank`` (rank == position in a range domain).

    ``digest`` is 8 Poseidon2 lanes for ``leaf/v2h`` (the default) or the 32-byte SHA-256 / BLAKE3 row digest.
    """
    value = row_leaf_value(digest) if schema == SCHEMA_ROW else bytes(digest)
    if schema != SCHEMA_ROW and (schema not in (SCHEMA_SHA256_ROW, SCHEMA_BLAKE3_ROW) or len(value) != 32):
        raise ValueError(f"unknown row schema {schema!r} or a digest that is not 32 bytes")
    return domain.leaf(rank, rank, schema, value)

"""``leaf/v2h``: committing tensor rows as Poseidon2 digests under the SHA-256 indexed Merkle trees of :mod:`merkle`.

A leaf is one *row* of a tensor in the orientation the GEMM consumes it -- a row of the activation matrix ``x`` (K
words), a column of the weight matrix ``W`` (K words) -- hashed with the padding-free Poseidon2 sponge of
:mod:`poseidon2_babybear` (width 24, rate 16, 16 bits of operand words per lane, the domain tag ``IV(role, word_bits,
n_words)`` in the capacity).  The 8-lane digest is the leaf *value* (32 bytes: each lane as a big-endian u32) under
:meth:`CommitmentDomain.leaf` with schema :data:`SCHEMA_ROW`; ranks are row indices, the tree above is the ordinary
SHA-256 ``node`` / pad framing, so a verifier that has the digest checks a ``leaf/v2h`` opening with SHA-256 alone.

Why a hash and not the words: the B-Ligero relation (``backends/direct/ligero/hashchain.py``) evaluates exactly this
sponge on its private operand wires and publishes the digest, so a proof binds its operands to a committed row without
the verifier seeing a word.  The gadget and this committer are the same function (:func:`poseidon2_babybear.hash_row`; the vectorised
committer over many rows is ``backends.direct.ligero.poseidon2.row_digests``, this package stays standard-library only).
"""
from __future__ import annotations

from . import poseidon2_babybear as h2
from .merkle import CommitmentDomain

__all__ = ["SCHEMA_ROW", "row_digest", "row_leaf", "row_leaf_value"]

SCHEMA_ROW = "poseidon2-babybear-w24/row/v2h"


def row_leaf_value(digest) -> bytes:
    """The 8 digest lanes as 32 bytes (each lane a big-endian u32)."""
    d = [int(x) for x in digest]
    if len(d) != h2.OUT or any(not 0 <= x < h2.P for x in d):
        raise ValueError("a row digest is 8 BabyBear lanes")
    return b"".join(x.to_bytes(4, "big") for x in d)


def row_digest(words, word_bits: int, role: int) -> list[int]:
    """Poseidon2 digest (8 lanes) of one row of ``word_bits``-bit words (``role``: ``h2.ROLE_X`` or ``h2.ROLE_W``)."""
    return h2.hash_row(words, word_bits, role)


def row_leaf(domain: CommitmentDomain, rank: int, digest) -> bytes:
    """The leaf hash of the row at ``rank`` (rank == position in a range domain)."""
    return domain.leaf(rank, rank, SCHEMA_ROW, row_leaf_value(digest))

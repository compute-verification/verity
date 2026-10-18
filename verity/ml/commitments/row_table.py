"""Row tables: one ``frame-v3-sha512`` tree of ``hm96-sha512/row-seg/v1`` hiding leaves over a matrix's rows, the CPU reference.

A row table commits a matrix one row per leaf, in row order: the token rows of an int8 quantize output (``Int8RowQuant_v1``,
each row its K s8 codes then its f32 scale, :func:`s8_row`), which are PoUW's A rows and the ``x`` rows of the serving rows'
``ScaledMmS8Coordinate_v1`` units.  With ``row`` a bit row's bytes (``rowleaf.bit_row``) of ``n_bits`` bits, ``y`` its
192-byte salt and ``key`` hm96's pinned key::

  x       = sha512_row_seg_digest(row, n_bits)                          private (``sha512/row-seg/v1``)
  string  = b || c = hm96.SHA512.commit_string(key, x, y)                public, 128 bytes
  leaf    = row_leaf(domain, rank, hm96.SHA512.tree_leaf(key, string), "hm96-sha512/row-seg/v1")

and the root is the ``frame-v3-sha512`` tree over the leaves under the table's domain (a ``hash="sha512"``
:class:`~verity.ml.commitments.merkle.CommitmentDomain` over ``[0, n)``; the serving rows' tables take core's served-domain
rule).  Two tables' leaves agree on a row exactly when their strings do, and a string binds its row and salt (hm96 binding).

An opening of a set of ranks is their strings and the multiproof over their leaves (:mod:`multiproof`).  :func:`verify` checks
one from public data alone (the root, the domain id, the leaf count); a proof that holds the rows and salts recomputes the
strings from them inside it, and :func:`verify_rows` is that check in the clear (tests, the prover).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.errors import InvalidArtifact, ProtocolError

from . import hm96
from .merkle import _LEAF, CommitmentDomain, MerkleTree, _hash, _uint
from .multiproof import multiproof, verify_multiproof
from .rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, sha512_row_seg_digest

__all__ = ["SCHEMA", "RowOpening", "RowTable", "commit", "leaf", "s8_row", "string", "verify", "verify_rows"]

SCHEMA = SCHEMA_HM96_SHA512_ROW_SEG
INST = hm96.SHA512
KEY = INST.default_key
STRING_BYTES = INST.proven_bytes
DIGEST_BYTES = INST.digest_bytes


def s8_row(codes: Iterable[int] | bytes, scale_bits: int) -> tuple[bytes, int]:
    """An int8 quantize output's token row as a bit row ``(row, n_bits)``: its K s8 codes (``-128 .. 127``, or their bytes),
    then its f32 scale's four bytes little-endian (``scale_bits`` the u32 bit pattern), ``8 K + 32`` bits; the serving rows'
    ``x`` row, its u16 words little-endian."""

    if isinstance(codes, (bytes, bytearray, memoryview)):
        raw = bytes(codes)
    else:
        vals = [int(c) for c in codes]
        if any(not -128 <= c < 128 for c in vals):
            raise ValueError("an s8 code is in [-128, 128)")
        raw = bytes(c & 0xFF for c in vals)
    if isinstance(scale_bits, bool) or not isinstance(scale_bits, int) or not 0 <= scale_bits < 1 << 32:
        raise ValueError(f"a scale is its f32's u32 bit pattern, not {scale_bits!r}")
    row = raw + scale_bits.to_bytes(4, "little")
    return row, 8 * len(row)


def string(row: bytes, n_bits: int, salt: bytes) -> bytes:
    """A row's commit string ``b || c`` (128 bytes) under its salt."""

    return INST.commit_string(KEY, sha512_row_seg_digest(row, n_bits), salt)


def leaf(domain_id: bytes, rank: int, bc: bytes) -> bytes:
    """The ``frame-v3-sha512`` leaf of the row at ``rank`` whose commit string is ``bc``, from the domain id alone: ``row_leaf``
    over ``hm96.SHA512.tree_leaf(key, bc)`` under ``hm96-sha512/row-seg/v1``."""

    return _hash(_LEAF, domain_id, _uint(rank), _uint(rank), SCHEMA.encode(), INST.tree_leaf(KEY, bc), hash="sha512")


def _string(s: Any) -> bytes:
    if type(s) is not bytes or len(s) != STRING_BYTES:
        raise ProtocolError(f"a row's commit string is {STRING_BYTES} bytes")
    return s


@dataclass(frozen=True, slots=True)
class RowOpening:
    """The strings of a table's rows at ``ranks`` (ascending, distinct) and the multiproof over their leaves."""

    ranks: tuple[int, ...]
    strings: tuple[bytes, ...]
    proof: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not self.ranks or list(self.ranks) != sorted(set(self.ranks)) or len(self.strings) != len(self.ranks):
            raise ProtocolError("an opening names its ranks ascending and distinct, one string each")
        for s in self.strings:
            _string(s)
        if any(type(p) is not bytes or len(p) != DIGEST_BYTES for p in self.proof):
            raise ProtocolError(f"a multiproof's nodes are {DIGEST_BYTES}-byte digests")

    @property
    def nbytes(self) -> int:
        """What it carries: the strings and the proof's nodes."""
        return STRING_BYTES * len(self.strings) + DIGEST_BYTES * len(self.proof)

    def to_json(self) -> dict[str, Any]:
        return {"ranks": list(self.ranks), "strings": [s.hex() for s in self.strings], "proof": [p.hex() for p in self.proof]}

    @classmethod
    def of_json(cls, doc: Mapping[str, Any]) -> RowOpening:
        return cls(tuple(int(r) for r in doc["ranks"]), tuple(bytes.fromhex(s) for s in doc["strings"]),
                   tuple(bytes.fromhex(p) for p in doc["proof"]))


class RowTable:
    """A committed table: its domain, every row's commit string in rank order, and the tree over their leaves (the prover's)."""

    __slots__ = ("_tree", "domain", "strings")

    def __init__(self, domain: CommitmentDomain, strings: Sequence[bytes]) -> None:
        if domain.hash != "sha512":
            raise ProtocolError("a row table is a frame-v3-sha512 tree")
        self.domain, self.strings = domain, tuple(_string(s) for s in strings)
        self._tree = MerkleTree(domain, {r: INST.tree_leaf(KEY, s) for r, s in enumerate(self.strings)}, lambda _r: SCHEMA)

    @property
    def root(self) -> bytes:
        return self._tree.commitment.root

    @property
    def count(self) -> int:
        return self.domain.count

    def open(self, ranks: Iterable[int]) -> RowOpening:
        """The opening of ``ranks`` (in any order, repeats dropped)."""
        ordered = tuple(sorted(set(ranks)))
        proof = tuple(multiproof(self._tree, ordered))
        return RowOpening(ordered, tuple(self.strings[r] for r in ordered), proof)


def commit(domain: CommitmentDomain, rows: Sequence[tuple[bytes, int]], salts: Sequence[bytes]) -> RowTable:
    """The table of ``rows`` (each ``(row, n_bits)``, in rank order) under their ``salts`` (192 bytes each, never reused)."""

    if len(rows) != len(salts):
        raise ProtocolError(f"{len(rows)} rows, {len(salts)} salts")
    return RowTable(domain, [string(r, b, y) for (r, b), y in zip(rows, salts, strict=True)])


def verify(root: bytes, domain_id: bytes, count: int, opening: RowOpening) -> bool:
    """Whether ``opening``'s strings are the rows at its ranks of the table of ``count`` rows under ``root`` and ``domain_id`` (the
    domain id binds the count; ``count`` fixes the tree's depth)."""

    leaves = {r: leaf(domain_id, r, s) for r, s in zip(opening.ranks, opening.strings, strict=True)}
    return verify_multiproof(root, count, domain_id, leaves, opening.proof, hash="sha512")


def verify_rows(root: bytes, domain_id: bytes, count: int, opening: RowOpening, rows: Mapping[int, tuple[bytes, int, bytes]]) -> bool:
    """:func:`verify`, and ``rows[rank] = (row, n_bits, salt)`` are the rows and salts of exactly its ranks' strings."""

    if sorted(rows) != list(opening.ranks):
        return False
    try:
        if any(string(*rows[r]) != s for r, s in zip(opening.ranks, opening.strings, strict=True)):
            return False
    except (ValueError, ProtocolError, InvalidArtifact):
        return False
    return verify(root, domain_id, count, opening)

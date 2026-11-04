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
strings from them inside it, and :func:`verify_rows` is that check in the clear (tests, the prover).  :meth:`RowTable.path` is one
row's own path, the form a statement's registered read of the row opens (``one_stage.registered``).

Serving commits its tables step by step, each step's before that step's live coin.  A step's tables (one per int8 quantize
output, in the step's order) take serve-time domains, :func:`table_domain` over ``context`` (64 bytes serving knows before
its first step: the live coin commitment's root), the step, the table's index and its row count; the step's root is the
``frame-v3-sha512`` tree over their roots (:func:`step_tree`, leaf schema ``ROOTS_SCHEMA``, domain :func:`step_domain`), the
one 64-byte root a step records before its coin.  A window's table is the tree over its steps' roots (:class:`RootTree` under
the caller's domain), so no row is committed twice.  :class:`StepOpening` opens rows of one table and that table's root in
its step's tree; :func:`verify_step` checks one against the step's root, and :func:`verify_roots` a step's root against the
window's.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.errors import InvalidArtifact, ProtocolError

from . import hm96
from .identity import identity_digest_sha512
from .indexed import RangeIndexedDomain
from .merkle import _LEAF, CommitmentDomain, MerkleTree, frame_hash, _uint
from .multiproof import multiproof, verify_multiproof
from .rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, sha512_row_seg_digest

__all__ = ["ROOTS_SCHEMA", "SCHEMA", "RootOpening", "RootTree", "RowOpening", "RowTable", "Step", "StepOpening", "commit",
           "commit_step", "leaf", "root_leaf", "s8_row", "step_domain", "step_tree", "string", "table_domain", "verify",
           "verify_roots", "verify_rows", "verify_step"]

SCHEMA = SCHEMA_HM96_SHA512_ROW_SEG
#: the leaf schema of a tree over roots: a step's over its tables', a window's over its steps'
ROOTS_SCHEMA = "frame-v3-sha512/root/v1"
TABLE_DOMAIN, STEP_DOMAIN = "verity/row-table/table/v1", "verity/row-table/step/v1"
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

    return frame_hash(_LEAF, domain_id, _uint(rank), _uint(rank), SCHEMA.encode(), INST.tree_leaf(KEY, bc), hash="sha512")


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

    def path(self, rank: int) -> bytes:
        """The row at ``rank``'s path to the root: its siblings, leaf level first, 64 bytes each (a registered read's path,
        ``one_stage.registered.verify_read``)."""
        return b"".join(self._tree.open(rank).path)


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


# --- steps and windows: trees over roots ---------------------------------------------------------------------------------

def _digest(d: Any, what: str) -> bytes:
    if type(d) is not bytes or len(d) != DIGEST_BYTES:
        raise ProtocolError(f"{what} is {DIGEST_BYTES} bytes")
    return d


def _count(v: Any, what: str) -> int:
    if type(v) is not int or v < 0:
        raise ProtocolError(f"{what} is a nonnegative integer, not {v!r}")
    return v


def _domain(tag: str, manifest: dict[str, Any], leaves: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(tag, manifest), -1, RangeIndexedDomain(0, leaves), hash="sha512")


def table_domain(context: bytes, step: int, index: int, leaves: int) -> CommitmentDomain:
    """The domain of table ``index`` (in its step's order) of serving step ``step``, of ``leaves`` rows, under ``context``."""

    return _domain(TABLE_DOMAIN, {"context": _digest(context, "a context").hex(), "step": _count(step, "a step"),
                                  "table": _count(index, "a table index"), "leaves": _count(leaves, "a row count"),
                                  "schema": SCHEMA}, leaves)


def step_domain(context: bytes, step: int, tables: int) -> CommitmentDomain:
    """The domain of serving step ``step``'s tree over its ``tables`` tables' roots, under ``context``."""

    return _domain(STEP_DOMAIN, {"context": _digest(context, "a context").hex(), "step": _count(step, "a step"),
                                 "tables": _count(tables, "a table count"), "schema": ROOTS_SCHEMA}, tables)


def root_leaf(domain_id: bytes, index: int, root: bytes) -> bytes:
    """The ``frame-v3-sha512`` leaf at ``index`` of a tree over roots, whose value is ``root``, from the domain id alone."""

    return frame_hash(_LEAF, domain_id, _uint(index), _uint(index), ROOTS_SCHEMA.encode(), _digest(root, "a root"), hash="sha512")


@dataclass(frozen=True, slots=True)
class RootOpening:
    """The roots at ``indices`` (ascending, distinct) of a tree over roots and the multiproof over their leaves."""

    indices: tuple[int, ...]
    roots: tuple[bytes, ...]
    proof: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if not self.indices or list(self.indices) != sorted(set(self.indices)) or len(self.roots) != len(self.indices):
            raise ProtocolError("an opening names its indices ascending and distinct, one root each")
        for d in (*self.roots, *self.proof):
            _digest(d, "a root or a multiproof node")

    @property
    def nbytes(self) -> int:
        return DIGEST_BYTES * (len(self.roots) + len(self.proof))

    def to_json(self) -> dict[str, Any]:
        return {"indices": list(self.indices), "roots": [r.hex() for r in self.roots], "proof": [p.hex() for p in self.proof]}

    @classmethod
    def of_json(cls, doc: Mapping[str, Any]) -> RootOpening:
        return cls(tuple(int(i) for i in doc["indices"]), tuple(bytes.fromhex(r) for r in doc["roots"]),
                   tuple(bytes.fromhex(p) for p in doc["proof"]))


class RootTree:
    """A ``frame-v3-sha512`` tree over roots in index order, one ``ROOTS_SCHEMA`` leaf each: a step's over its tables' roots
    (:func:`step_tree`), or a window's over its steps' roots."""

    __slots__ = ("_tree", "domain", "roots")

    def __init__(self, domain: CommitmentDomain, roots: Sequence[bytes]) -> None:
        if domain.hash != "sha512":
            raise ProtocolError("a tree over roots is a frame-v3-sha512 tree")
        self.domain, self.roots = domain, tuple(_digest(r, "a root") for r in roots)
        self._tree = MerkleTree(domain, dict(enumerate(self.roots)), lambda _i: ROOTS_SCHEMA)

    @property
    def root(self) -> bytes:
        return self._tree.commitment.root

    @property
    def count(self) -> int:
        return self.domain.count

    def open(self, indices: Iterable[int]) -> RootOpening:
        """The opening of ``indices`` (in any order, repeats dropped)."""
        ordered = tuple(sorted(set(indices)))
        proof = tuple(multiproof(self._tree, ordered))
        return RootOpening(ordered, tuple(self.roots[i] for i in ordered), proof)


def verify_roots(root: bytes, domain_id: bytes, count: int, opening: RootOpening) -> bool:
    """Whether ``opening``'s roots are the leaves at its indices of the tree over ``count`` roots under ``root`` and
    ``domain_id``."""

    leaves = {i: root_leaf(domain_id, i, r) for i, r in zip(opening.indices, opening.roots, strict=True)}
    return verify_multiproof(root, count, domain_id, leaves, opening.proof, hash="sha512")


def step_tree(context: bytes, step: int, table_roots: Sequence[bytes]) -> RootTree:
    """Serving step ``step``'s tree over its tables' roots (in the step's order): its root is the step's root."""

    return RootTree(step_domain(context, step, len(table_roots)), table_roots)


@dataclass(frozen=True, slots=True)
class StepOpening:
    """Rows of one of a step's tables: that table's root opened in the step's tree (``table``, one index), and the rows
    opened against it (``rows``)."""

    table: RootOpening
    rows: RowOpening

    def __post_init__(self) -> None:
        if len(self.table.indices) != 1:
            raise ProtocolError("a step opening opens one table's root")

    @property
    def index(self) -> int:
        return self.table.indices[0]

    @property
    def root(self) -> bytes:
        return self.table.roots[0]

    @property
    def nbytes(self) -> int:
        return self.table.nbytes + self.rows.nbytes

    def to_json(self) -> dict[str, Any]:
        return {"table": self.table.to_json(), "rows": self.rows.to_json()}

    @classmethod
    def of_json(cls, doc: Mapping[str, Any]) -> StepOpening:
        return cls(RootOpening.of_json(doc["table"]), RowOpening.of_json(doc["rows"]))


class Step:
    """One serving step's committed tables, in the step's order, each under its :func:`table_domain`, and the tree over their
    roots (the prover's); ``root`` is what the step records before its coin."""

    __slots__ = ("context", "step", "tables", "tree")

    def __init__(self, context: bytes, step: int, tables: Sequence[RowTable]) -> None:
        for i, t in enumerate(tables):
            if t.domain.domain_id != table_domain(context, step, i, t.count).domain_id:
                raise ProtocolError(f"step {step}'s table {i} is not under its table domain")
        self.context, self.step, self.tables = context, step, tuple(tables)
        self.tree = step_tree(context, step, [t.root for t in self.tables])

    @property
    def root(self) -> bytes:
        return self.tree.root

    def open(self, index: int, ranks: Iterable[int]) -> StepOpening:
        """The opening of table ``index``'s rows at ``ranks``."""
        return StepOpening(self.tree.open([index]), self.tables[index].open(ranks))


def commit_step(context: bytes, step: int, tables: Sequence[tuple[Sequence[tuple[bytes, int]], Sequence[bytes]]]) -> Step:
    """Serving step ``step``'s tables, each ``(rows, salts)`` as :func:`commit` takes them, in the step's order."""

    return Step(context, step, [commit(table_domain(context, step, i, len(rows)), rows, salts) for i, (rows, salts) in enumerate(tables)])


def verify_step(step_root: bytes, context: bytes, step: int, tables: int, leaves: int, opening: StepOpening,
                rows: Mapping[int, tuple[bytes, int, bytes]] | None = None) -> bool:
    """Whether ``opening`` opens rows of table ``opening.index`` (of ``leaves`` rows) of serving step ``step``, whose root is
    ``step_root`` over its ``tables`` tables under ``context``; with ``rows``, they are its rows and salts (:func:`verify_rows`)."""

    if not verify_roots(step_root, step_domain(context, step, tables).domain_id, tables, opening.table):
        return False
    did = table_domain(context, step, opening.index, leaves).domain_id
    if rows is None:
        return verify(opening.root, did, leaves, opening.rows)
    return verify_rows(opening.root, did, leaves, opening.rows, rows)

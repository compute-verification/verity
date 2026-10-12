"""``frame-v3``: SHA-256 indexed trees over verifier-derived domains, as a configured :class:`CommitmentScheme`; and
``frame-v3-sha512``, the same framing through SHA-512 (ports whose domains have ``hash="sha512"``; PROTOCOL.md §6).

Spec: ``PROTOCOL.md`` beside this file (the leaf layouts and the row-leaf schemas; the framing itself is
:mod:`verity.ml.commitments.merkle`).  Vectors: ``vectors.json``, written by ``python -m verity.ml.commitments.frame_v3.vectors``.

A configuration maps each committed port to its :class:`~verity.ml.commitments.merkle.CommitmentDomain` and leaf kind:
:class:`WordLeaf` (the value is the leaf's own bytes, hashed by the frame-v3 leaf rule itself), :class:`RowLeaf` (the
value is a row, the proven digest is its SHA-256 or keyed-BLAKE3 row digest, and the frame-v3 leaf over that digest is
computed natively by the verifier) or :class:`BitRowLeaf` (the same for a row of bits under ``sha512/row/v2``, in
``frame-v3-sha512``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..merkle import CommitmentDomain, merkle_depth, tree_levels
from ..rowleaf import (SCHEMA_BLAKE3_ROW, SCHEMA_HM96_SHA512_ROW_V2, SCHEMA_SHA256_ROW, SCHEMA_SHA512_ROW, SCHEMA_SHA512_ROW_V2,
                       blake3_row_layout, sha256_row_layout, sha512_row_layout, sha512_row_v2_layout)
from ..scheme import DIGEST_BYTES, LeafLayout

__all__ = ["NAME", "NAME_SHA512", "BitRowLeaf", "FrameV3", "Port", "RowLeaf", "WordLeaf"]

NAME = "frame-v3"
NAME_SHA512 = "frame-v3-sha512"

_ROW_SCHEMAS = {"sha256": SCHEMA_SHA256_ROW, "blake3": SCHEMA_BLAKE3_ROW, "sha512": SCHEMA_SHA512_ROW}
_ROW_LAYOUTS = {"sha256": sha256_row_layout, "blake3": blake3_row_layout, "sha512": sha512_row_layout}
#: The row digests each frame hash admits: a SHA-512 frame binds SHA-512 rows only.
_FRAME_ROWS = {"sha256": ("sha256", "blake3"), "sha512": ("sha512",)}


@dataclass(frozen=True, slots=True)
class WordLeaf:
    """A leaf whose value is ``nbytes`` bytes under ``schema`` (``u16``, ``u32``, ...): the leaf digest is the tree leaf."""

    schema: str
    nbytes: int


@dataclass(frozen=True, slots=True)
class RowLeaf:
    """A leaf whose value is one row of ``n_words`` ``word_bits``-bit words; ``hash`` is ``sha256``, ``blake3`` or ``sha512``."""

    hash: str
    role: int
    word_bits: int
    n_words: int

    def __post_init__(self) -> None:
        if self.hash not in _ROW_SCHEMAS:
            raise ValueError(f"row leaf hash must be one of {sorted(_ROW_SCHEMAS)}, not {self.hash!r}")
        self.layout()

    @property
    def schema(self) -> str:
        return _ROW_SCHEMAS[self.hash]

    def layout(self) -> LeafLayout:
        return _ROW_LAYOUTS[self.hash](self.role, self.word_bits, self.n_words)


@dataclass(frozen=True, slots=True)
class BitRowLeaf:
    """A leaf whose value is one row of ``n_bits`` bits under ``sha512/row/v2`` (``rowleaf.bit_row``: each element least
    significant bit first, elements in order); a hiding row's leaf binds ``hiding_schema`` (``hm96-sha512/row/v2``)."""

    n_bits: int
    hash = "sha512"
    schema = SCHEMA_SHA512_ROW_V2
    hiding_schema = SCHEMA_HM96_SHA512_ROW_V2

    def __post_init__(self) -> None:
        self.layout()

    def layout(self) -> LeafLayout:
        return sha512_row_v2_layout(self.n_bits)


_ROWS = (RowLeaf, BitRowLeaf)


@dataclass(frozen=True, slots=True)
class Port:
    domain: CommitmentDomain
    leaf: WordLeaf | RowLeaf | BitRowLeaf


class FrameV3:
    """frame-v3 over a statement's ports; nodes are bound to the port's domain, level and index.  Every port's domain has one
    hash: ``sha256`` configures ``frame-v3``, ``sha512`` configures ``frame-v3-sha512`` (whose row leaves are ``sha512/row/v1``
    and ``sha512/row/v2``)."""

    def __init__(self, ports: Mapping[str, Port]) -> None:
        self.ports = dict(ports)
        frames = {p.domain.hash for p in self.ports.values()} or {"sha256"}
        if len(frames) != 1:
            raise ValueError(f"one configuration has one frame hash, not {sorted(frames)}")
        self.frame_hash = frames.pop()
        bad = [n for n, p in self.ports.items() if isinstance(p.leaf, _ROWS) and p.leaf.hash not in _FRAME_ROWS[self.frame_hash]]
        if bad:
            raise ValueError(f"ports {bad}: a {self.frame_hash} frame binds {_FRAME_ROWS[self.frame_hash]} row digests only")
        self.name = NAME if self.frame_hash == "sha256" else NAME_SHA512
        uses_blake3 = any(isinstance(p.leaf, RowLeaf) and p.leaf.hash == "blake3" for p in self.ports.values())
        self.hashes = (self.frame_hash, "blake3") if uses_blake3 else (self.frame_hash,)

    def _port(self, port: str) -> Port:
        try:
            return self.ports[port]
        except KeyError as error:
            raise KeyError(f"{port!r} is not a port of this configuration") from error

    def leaf_layout(self, port: str, position: int) -> LeafLayout:
        p = self._port(port)
        if isinstance(p.leaf, _ROWS):
            p.domain.positions.rank(position)
            return p.leaf.layout()
        rank = p.domain.positions.rank(position)
        return LeafLayout(p.domain.hash, p.domain.leaf_prefix(rank, position, p.leaf.schema, p.leaf.nbytes), p.leaf.nbytes)

    def tree_leaf(self, port: str, position: int, digest: bytes) -> bytes:
        p = self._port(port)
        want = DIGEST_BYTES[p.leaf.hash] if isinstance(p.leaf, _ROWS) else p.domain.digest_bytes
        if len(digest) != want:
            raise ValueError(f"this port's leaf digest is {want} bytes")
        if isinstance(p.leaf, _ROWS):
            return p.domain.leaf(p.domain.positions.rank(position), position, p.leaf.schema, bytes(digest))
        return bytes(digest)

    def root(self, port: str, leaves: Sequence[bytes]) -> bytes:
        domain = self._port(port).domain
        levels = tree_levels(domain, leaves)
        return levels[-1][0] if levels else domain.empty_root()

    def path(self, port: str, leaves: Sequence[bytes], index: int) -> tuple[bytes, ...]:
        if not 0 <= index < len(leaves):
            raise ValueError(f"index {index} out of range for {len(leaves)} leaves")
        out = []
        for level in tree_levels(self._port(port).domain, leaves)[:-1]:
            out.append(level[index ^ 1])
            index >>= 1
        return tuple(out)

    def verify_path(self, port: str, root: bytes, leaf_count: int, index: int, leaf: bytes, path: Sequence[bytes | None]) -> bool:
        try:
            domain = self._port(port).domain
        except KeyError:
            return False
        if leaf_count != domain.count or not 0 <= index < leaf_count or len(path) != merkle_depth(leaf_count):
            return False
        n = domain.digest_bytes
        if type(leaf) is not bytes or len(leaf) != n or any(type(s) is not bytes or len(s) != n for s in path):
            return False
        digest, cursor = leaf, index
        for level, sibling in enumerate(path):
            pair = (digest, sibling) if cursor % 2 == 0 else (sibling, digest)
            digest = domain.node(level, cursor >> 1, *pair)
            cursor >>= 1
        return digest == root

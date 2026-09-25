"""``frame-v3``: SHA-256 indexed trees over verifier-derived domains, as a configured :class:`CommitmentScheme`.

Spec: ``PROTOCOL.md`` beside this file (the leaf layouts and the row-leaf schemas; the framing itself is
:mod:`verity.commitments.merkle`).  Vectors: ``vectors.json``, written by ``python -m verity.commitments.frame_v3.vectors``.

A configuration maps each committed port to its :class:`~verity.commitments.merkle.CommitmentDomain` and leaf kind:
:class:`WordLeaf` (the value is the leaf's own bytes, hashed by the frame-v3 leaf rule itself) or :class:`RowLeaf` (the
value is a row, the proven digest is its SHA-256 or keyed-BLAKE3 row digest, and the frame-v3 leaf over that digest is
computed natively by the verifier).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..merkle import CommitmentDomain, merkle_depth, tree_levels
from ..rowleaf import SCHEMA_BLAKE3_ROW, SCHEMA_SHA256_ROW, blake3_row_layout, sha256_row_layout
from ..scheme import LeafLayout

__all__ = ["NAME", "FrameV3", "Port", "RowLeaf", "WordLeaf"]

NAME = "frame-v3"

_ROW_SCHEMAS = {"sha256": SCHEMA_SHA256_ROW, "blake3": SCHEMA_BLAKE3_ROW}


@dataclass(frozen=True, slots=True)
class WordLeaf:
    """A leaf whose value is ``nbytes`` bytes under ``schema`` (``u16``, ``u32``, ...): the leaf digest is the tree leaf."""

    schema: str
    nbytes: int


@dataclass(frozen=True, slots=True)
class RowLeaf:
    """A leaf whose value is one row of ``n_words`` ``word_bits``-bit words; ``hash`` is ``sha256`` or ``blake3``."""

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
        make = sha256_row_layout if self.hash == "sha256" else blake3_row_layout
        return make(self.role, self.word_bits, self.n_words)


@dataclass(frozen=True, slots=True)
class Port:
    domain: CommitmentDomain
    leaf: WordLeaf | RowLeaf


class FrameV3:
    """frame-v3 over a statement's ports; nodes are bound to the port's domain, level and index."""

    name = NAME

    def __init__(self, ports: Mapping[str, Port]) -> None:
        self.ports = dict(ports)
        uses_blake3 = any(isinstance(p.leaf, RowLeaf) and p.leaf.hash == "blake3" for p in self.ports.values())
        self.hashes = ("sha256", "blake3") if uses_blake3 else ("sha256",)

    def _port(self, port: str) -> Port:
        try:
            return self.ports[port]
        except KeyError as error:
            raise KeyError(f"{port!r} is not a port of this configuration") from error

    def leaf_layout(self, port: str, position: int) -> LeafLayout:
        p = self._port(port)
        if isinstance(p.leaf, RowLeaf):
            p.domain.positions.rank(position)
            return p.leaf.layout()
        rank = p.domain.positions.rank(position)
        return LeafLayout("sha256", p.domain.leaf_prefix(rank, position, p.leaf.schema, p.leaf.nbytes), p.leaf.nbytes)

    def tree_leaf(self, port: str, position: int, digest: bytes) -> bytes:
        p = self._port(port)
        if len(digest) != 32:
            raise ValueError("a leaf digest is 32 bytes")
        if isinstance(p.leaf, RowLeaf):
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
        if type(leaf) is not bytes or len(leaf) != 32 or any(type(s) is not bytes or len(s) != 32 for s in path):
            return False
        digest, cursor = leaf, index
        for level, sibling in enumerate(path):
            pair = (digest, sibling) if cursor % 2 == 0 else (sibling, digest)
            digest = domain.node(level, cursor >> 1, *pair)
            cursor >>= 1
        return digest == root

"""A committed table: a matrix over F whose rows are Reed–Solomon encoded, committed column by column in a frame-v3-sha512
tree (`verity.ml.commitments.merkle`). Column j's leaf value is a fresh 64-byte salt, then the column's r elements
(8 bytes each, little-endian), under the schema `SCHEMA`; the salt hides a column until it is opened.

A table's domain binds its shape (`Shape.binding`), so a root names one role, size and code.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from verity.ml.commitments.indexed import RangeU64Domain
from verity.ml.commitments.limits import VerificationLimits
from verity.ml.commitments.merkle import Commitment, CommitmentDomain, MerkleTree, Opening, verify_opening

from . import code
from .field import P, from_bytes, to_bytes

SCHEMA = "verity/lin/column/v0"
SALT_BYTES = 64
_LIMITS = VerificationLimits()


@dataclass(frozen=True)
class Shape:
    """`rows` rows of `width` elements (a power of two), encoded at `blowup` (a power of two at least 2)."""

    role: str
    rows: int
    width: int
    blowup: int

    def __post_init__(self) -> None:
        for name in ("width", "blowup"):
            v = getattr(self, name)
            if v < 1 or v & (v - 1):
                raise ValueError(f"a table's {name} must be a power of two, not {v}")
        if self.blowup < 2 or self.rows < 1:
            raise ValueError("a table has at least one row and a blowup of at least 2")

    @property
    def n(self) -> int:
        return self.width * self.blowup

    @property
    def binding(self) -> bytes:
        doc = json.dumps({"format": "verity/lin/table/v0", "role": self.role, "rows": self.rows, "width": self.width,
                          "blowup": self.blowup}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha512(doc.encode()).digest()

    @property
    def domain(self) -> CommitmentDomain:
        return CommitmentDomain(self.binding, -1, RangeU64Domain(0, self.n), hash="sha512")


class Table:
    """The prover's side: the rows, their codewords and the tree over the columns."""

    def __init__(self, shape: Shape, rows: Sequence[Sequence[int]], salt: Callable[[int], bytes] = os.urandom) -> None:
        if len(rows) != shape.rows or any(len(r) != shape.width for r in rows):
            raise ValueError(f"expected {shape.rows} rows of {shape.width}")
        self.shape = shape
        self.rows = [[x % P for x in r] for r in rows]
        self.encoded = [code.encode(r, shape.n) for r in self.rows]
        self._values = {j: salt(SALT_BYTES) + b"".join(to_bytes(u[j]) for u in self.encoded) for j in range(shape.n)}
        self._tree = MerkleTree(shape.domain, self._values, lambda _: SCHEMA)

    @property
    def root(self) -> bytes:
        return self._tree.commitment.root

    def open(self, j: int) -> Opening:
        return self._tree.open(j)


def column(shape: Shape, root: bytes, opening: Opening) -> list[int] | None:
    """The column an opening authenticates under `root`, or None."""
    if not verify_opening(shape.domain, Commitment(root, shape.n), opening, SCHEMA, _LIMITS):
        return None
    if len(opening.value) != SALT_BYTES + 8 * shape.rows:
        return None
    return from_bytes(opening.value[SALT_BYTES:])


def opening_bytes(o: Opening) -> bytes:
    return (o.position.to_bytes(8, "big") + len(o.value).to_bytes(8, "big") + o.value
            + len(o.path).to_bytes(2, "big") + b"".join(o.path))


def parse_opening(data: bytes) -> tuple[Opening, bytes] | None:
    """One opening from the head of `data`, and the rest; None if it doesn't parse."""
    try:
        pos = int.from_bytes(data[:8], "big")
        k = int.from_bytes(data[8:16], "big")
        value = data[16:16 + k]
        rest = data[16 + k:]
        depth = int.from_bytes(rest[:2], "big")
        path = tuple(rest[2 + 64 * i:2 + 64 * (i + 1)] for i in range(depth))
        if len(value) != k or any(len(p) != 64 for p in path):
            return None
        return Opening(pos, value, path), rest[2 + 64 * depth:]
    except (ValueError, IndexError):
        return None

"""A committed table: a matrix over F whose rows are Reed–Solomon encoded, committed column by column in a frame-v3-sha512
tree (`verity.ml.commitments.merkle`). Column j's leaf value is a fresh 64-byte salt, then the column's r elements
(8 bytes each, little-endian), under the schema `SCHEMA`; the salt hides a column until it is opened. Columns open
together, under one multiproof (`verity.ml.commitments.multiproof`).

A table's domain binds its shape (`Shape.binding`), so a root names one role, size and code.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from verity.ml.commitments.indexed import RangeU64Domain
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree
from verity.ml.commitments.multiproof import multiproof, verify_multiproof

from . import code
from .field import P, from_bytes, to_bytes

SCHEMA = "verity/lin/column/v0"
SALT_BYTES = 64


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

    def open(self, js: Iterable[int]) -> bytes:
        """The columns at `js` and one multiproof for them (`verity.ml.commitments.multiproof`), as `columns` reads them."""
        ranks = sorted(set(js))
        proof = multiproof(self._tree, ranks)
        return (len(ranks).to_bytes(4, "big") + b"".join(j.to_bytes(8, "big") + self._values[j] for j in ranks)
                + len(proof).to_bytes(4, "big") + b"".join(proof))


def columns(shape: Shape, root: bytes, data: bytes) -> tuple[dict[int, list[int]] | None, str]:
    """The columns `data` opens under `root`, by index, or None and why: `encoding` when it doesn't parse, `column-path`
    when the multiproof doesn't reconstruct the root."""
    value_len = SALT_BYTES + 8 * shape.rows
    try:
        count, at = int.from_bytes(data[:4], "big"), 4
        values: dict[int, bytes] = {}
        for _ in range(count):
            j = int.from_bytes(data[at:at + 8], "big")
            values[j] = data[at + 8:at + 8 + value_len]
            at += 8 + value_len
        k, at = int.from_bytes(data[at:at + 4], "big"), at + 4
        proof = [data[at + 64 * i:at + 64 * (i + 1)] for i in range(k)]
    except IndexError:
        return None, "encoding"
    if (len(values) != count or at + 64 * k != len(data) or any(len(v) != value_len for v in values.values())
            or any(len(p) != 64 for p in proof) or any(not 0 <= j < shape.n for j in values)):
        return None, "encoding"
    domain = shape.domain
    leaves = {j: domain.leaf(j, j, SCHEMA, v) for j, v in values.items()}
    if not verify_multiproof(root, shape.n, domain.domain_id, leaves, proof, hash="sha512"):
        return None, "column-path"
    cols = {j: from_bytes(v[SALT_BYTES:]) for j, v in values.items()}
    if any(c is None for c in cols.values()):
        return None, "encoding"
    return cols, ""

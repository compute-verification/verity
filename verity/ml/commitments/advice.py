"""Private advice committed as hiding rows: fixed-width bit rows under ``hm96-sha512/row/v2`` leaves in one ``frame-v3-sha512``
tree, so a root fixes the rows before any coin and reveals nothing of them.

Row ``i`` (``n_bits`` bits, ``rowleaf.bit_row``'s bytes) with salt ``y_i`` has, under ``hm96.SHA512`` with key ``key``::

    x_i    = sha512/row/v2 digest of the row         (frame_v3.BitRowLeaf(n_bits).layout())
    bc_i   = commit_string(key, x_i, y_i)            the commit string b || c
    leaf_i = row_leaf(D, i, tree_leaf(key, bc_i), "hm96-sha512/row/v2")

and the root is the ``frame-v3-sha512`` root of the leaves over ``D = CommitmentDomain(binding, -1, RangeIndexedDomain(R),
"sha512")``: the binding is what the commitment may not outlive (the consumer's, such as `partition_v2.BINDING`), ``-1`` the
boundary owner, R the row count, which the consumer's statement fixes.  An opening of row i is the row, its salt and the path;
`verify` recomputes the leaf and walks the path.  A proof that holds the rows privately proves ``bc_i`` from the row and the salt
(the backend's hm96 gadget) and the verifier finishes natively, as for every hm96 port.  Salts are fresh from the OS unless given
(vectors pass derived ones); hiding is hm96's, statistical given uniform salts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import hm96
from .frame_v3 import BitRowLeaf, FrameV3, Port
from .indexed import RangeIndexedDomain
from .merkle import CommitmentDomain
from .rowleaf import row_leaf

__all__ = ["OWNER", "PORT", "Committed", "Opening", "commit", "domain", "leaf", "verify"]

INST = hm96.SHA512
OWNER = -1
PORT = "advice"


def domain(binding: bytes, count: int) -> CommitmentDomain:
    return CommitmentDomain(bytes(binding), OWNER, RangeIndexedDomain(count), hash="sha512")


def _scheme(binding: bytes, count: int, n_bits: int) -> FrameV3:
    return FrameV3({PORT: Port(domain(binding, count), BitRowLeaf(n_bits))})


def leaf(d: CommitmentDomain, n_bits: int, i: int, row: bytes, salt: bytes, key: bytes = INST.default_key) -> bytes:
    """Row i's tree leaf: its ``hm96-sha512/row/v2`` leaf at rank i of `d`."""
    kind = BitRowLeaf(n_bits)
    bc = INST.commit_string(key, kind.layout().digest(bytes(row)), salt)
    return row_leaf(d, i, INST.tree_leaf(key, bc), kind.hiding_schema)


@dataclass(frozen=True)
class Opening:
    index: int
    row: bytes
    salt: bytes
    path: tuple[bytes, ...]


@dataclass(frozen=True)
class Committed:
    """The prover's side: the rows, their salts and the tree.  `root` is what goes public."""

    binding: bytes
    n_bits: int
    rows: tuple[bytes, ...]
    salts: tuple[bytes, ...]
    key: bytes = INST.default_key
    leaves: tuple[bytes, ...] = field(init=False, repr=False)
    root: bytes = field(init=False)

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.salts) or not self.rows:
            raise ValueError("one salt per row, and at least one row")
        width = (self.n_bits + 7) // 8
        if any(len(r) != width for r in self.rows):
            raise ValueError(f"a row of {self.n_bits} bits is {width} bytes")
        d = domain(self.binding, len(self.rows))
        leaves = tuple(leaf(d, self.n_bits, i, r, s, self.key) for i, (r, s) in enumerate(zip(self.rows, self.salts)))
        object.__setattr__(self, "leaves", leaves)
        object.__setattr__(self, "root", _scheme(self.binding, len(self.rows), self.n_bits).root(PORT, leaves))

    def open(self, i: int) -> Opening:
        path = _scheme(self.binding, len(self.rows), self.n_bits).path(PORT, self.leaves, i)
        return Opening(i, self.rows[i], self.salts[i], path)


def commit(rows: Sequence[bytes], n_bits: int, binding: bytes, salts: Sequence[bytes] | None = None,
           key: bytes = INST.default_key) -> Committed:
    """Commit `rows` (each ``ceil(n_bits / 8)`` bytes) under `binding`; fresh salts unless `salts` is given."""
    rows = tuple(bytes(r) for r in rows)
    salts = tuple(bytes(s) for s in salts) if salts is not None else tuple(INST.fresh_salt() for _ in rows)
    return Committed(bytes(binding), n_bits, rows, salts, key)


def verify(root: bytes, count: int, n_bits: int, binding: bytes, opening: Opening, key: bytes = INST.default_key) -> bool:
    """Whether `opening` is row ``opening.index`` of the `count` rows committed under `binding` with root `root`."""
    try:
        d = domain(binding, count)
        lf = leaf(d, n_bits, opening.index, opening.row, opening.salt, key)
    except (ValueError, TypeError):
        return False
    return _scheme(binding, count, n_bits).verify_path(PORT, root, count, opening.index, lf, opening.path)

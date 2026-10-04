"""Commitment schemes: a commitment binds its maker to a vector of leaves before any coins are drawn.

Everyone who opens a commitment in a deployment must use the same scheme, so the configuration names it. A toy scheme
(the hash of the leaves' hashes, opened by revealing them) stands in for today's `verity.commitments`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Commitment:
    scheme: str
    root: bytes


class CommitmentScheme(Protocol):
    name: str

    def commit(self, leaves: Sequence[bytes]) -> Commitment: ...

    def open(self, leaves: Sequence[bytes], index: int) -> object:
        """A proof that `leaves[index]` is the leaf at `index`."""

    def verify(self, commitment: Commitment, index: int, leaf: bytes, proof: object) -> bool: ...


class Sha256Leaves:
    name = "sha256-leaves"

    def commit(self, leaves: Sequence[bytes]) -> Commitment:
        return Commitment(self.name, hashlib.sha256(b"".join(hashlib.sha256(x).digest() for x in leaves)).digest())

    def open(self, leaves: Sequence[bytes], index: int) -> object:
        return list(leaves)

    def verify(self, commitment: Commitment, index: int, leaf: bytes, proof: object) -> bool:
        return proof[index] == leaf and self.commit(proof) == commitment


SCHEMES: dict[str, type[CommitmentScheme]] = {Sha256Leaves.name: Sha256Leaves}

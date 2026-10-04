"""What a commitment must be, not how to build one.

A commitment scheme binds the developer to a vector of leaves before the auditor draws any coins; some constructions also
hide them. The constructions are today's `verity.commitments` (`frame_v3`, `vllm_v1`, `hm96`, their SHA-512 variants),
selected by the configuration, since everyone who opens a commitment in a deployment must use the same one.
"""

from __future__ import annotations

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

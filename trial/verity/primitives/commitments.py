"""Commitment schemes: a commitment binds its maker to a vector of leaves before any coins are drawn, and opens one leaf
at a time.

Everyone who opens a commitment in a deployment must use the same scheme, so the configuration names it. `MerkleSha256`
stands in for today's `verity.commitments` (`frame_v3`, `hm96`, `vllm_v1` and their SHA-512 variants); unlike them, its
leaves do not hide.
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
    leaves: int


class CommitmentScheme(Protocol):
    name: str

    def commit(self, leaves: Sequence[bytes]) -> Commitment: ...

    def open(self, leaves: Sequence[bytes], index: int) -> object:
        """A proof that `leaves[index]` is the leaf at `index`."""

    def verify(self, commitment: Commitment, index: int, leaf: bytes, proof: object) -> bool: ...


def _h(*parts: bytes) -> bytes:
    return hashlib.sha256(b"".join(parts)).digest()


class MerkleSha256:
    name = "merkle-sha256"

    def _levels(self, leaves: Sequence[bytes]) -> list[list[bytes]]:
        level = [_h(b"\x00", x) for x in leaves] or [_h(b"\x00")]
        levels = [level]
        while len(level) > 1:
            if len(level) % 2:
                level = level + [level[-1]]
            level = [_h(b"\x01", level[i], level[i + 1]) for i in range(0, len(level), 2)]
            levels.append(level)
        return levels

    def commit(self, leaves: Sequence[bytes]) -> Commitment:
        return Commitment(self.name, self._levels(leaves)[-1][0], len(leaves))

    def open(self, leaves: Sequence[bytes], index: int) -> list[bytes]:
        path = []
        for level in self._levels(leaves)[:-1]:
            sibling = index ^ 1
            path.append(level[sibling] if sibling < len(level) else level[index])
            index //= 2
        return path

    def verify(self, commitment: Commitment, index: int, leaf: bytes, proof: object) -> bool:
        if commitment.scheme != self.name or not 0 <= index < commitment.leaves:
            return False
        node = _h(b"\x00", leaf)
        for sibling in proof:
            node = _h(b"\x01", node, sibling) if index % 2 == 0 else _h(b"\x01", sibling, node)
            index //= 2
        return node == commitment.root


SCHEMES: dict[str, type[CommitmentScheme]] = {MerkleSha256.name: MerkleSha256}

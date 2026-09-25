"""The ``CommitmentScheme`` interface: what every named commitment format exposes to a verifier.

A scheme is a leaf rule plus a tree rule plus root bindings.  Leaf encodings differ per scheme and are plain
functions of that scheme's module; this interface is the part a verifier drives generically: fold leaf digests
(in tree order) to a tree root, produce the authentication path of one leaf, and check a path whose shape the
verifier derives from ``(leaf_count, index)`` itself, never from the prover.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

Path = tuple[bytes | None, ...]


@runtime_checkable
class CommitmentScheme(Protocol):
    #: The scheme's registered name; its spec and conformance vectors carry the same string.
    name: str

    def root(self, leaves: Sequence[bytes]) -> bytes:
        """The tree root over 32-byte leaf digests given in tree order."""
        ...

    def path(self, leaves: Sequence[bytes], index: int) -> Path:
        """The authentication path of leaf ``index``, one entry per level from the leaves up."""
        ...

    def verify_path(self, root: bytes, leaf_count: int, index: int, leaf: bytes, path: Sequence[bytes | None]) -> bool:
        """Does ``leaf`` at ``index`` of a ``leaf_count``-leaf tree fold to ``root`` along ``path``?  Never raises."""
        ...

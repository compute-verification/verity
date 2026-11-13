"""A registered value's ``b || c`` strings or one level of its tree as `registered.Registered` holds them, read out of the
array the hash wrote: item i is node i's bytes, made on read, so a value whose nodes are never opened makes none. It is
equal to the tuple of the same byte strings, and hashes as that tuple does."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

__all__ = ["Nodes"]


class Nodes(Sequence):
    """``raw``'s bytes as nodes of ``width`` bytes each, in order."""

    __slots__ = ("raw", "width")

    def __init__(self, raw: np.ndarray, width: int) -> None:
        if raw.dtype != np.uint8 or raw.ndim != 1 or width < 1 or raw.size % width:
            raise ValueError(f"{raw.size} {raw.dtype} bytes as nodes of {width}")
        self.raw, self.width = raw, int(width)

    def __len__(self) -> int:
        return self.raw.size // self.width

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return tuple(self[j] for j in range(*i.indices(len(self))))
        n = len(self)
        j = int(i) + n if i < 0 else int(i)
        if not 0 <= j < n:
            raise IndexError(f"node {i} of {n}")
        return self.raw[j * self.width:(j + 1) * self.width].tobytes()

    def __iter__(self) -> Iterator[bytes]:
        w = self.width
        for j in range(len(self)):
            yield self.raw[j * w:(j + 1) * w].tobytes()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Nodes):
            return np.array_equal(self.raw, other.raw) if self.width == other.width else tuple(self) == tuple(other)
        if isinstance(other, tuple):
            return len(self) == len(other) and all(a == b for a, b in zip(self, other))
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __reduce__(self) -> tuple[Any, ...]:
        return Nodes, (self.raw, self.width)

    def __repr__(self) -> str:
        return f"Nodes({len(self)} of {self.width} bytes)"

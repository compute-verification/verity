"""The anchored inputs' anchors: what each must equal, built by the verifier (the red team's X-SPC-72).

`ncp-v2`'s index binding is Program inputs, so each needs an anchor the verifier owns:

    field       anchor                                                                       input
    salt        the run's salt                                                               `salt`
    forward     the verifier's own enumeration of the window's forwards                      `forward`
    position    (layer, site) from the registered layout: layer << 32 | site                 `position`
    weight id   the registered id of the matrix whose rows B opens under the weights root    `weight`
    row ids     the call's global row indices, distinct                                      `rowids`
    column ids  the weight's global column indices, shards included (X-SPC-40), distinct     `colids`

The verifier builds these inputs itself (`Anchors.call_inputs`, `Anchors.weight_inputs`), or compares a prover's
(`Anchors.check_call`, `Anchors.check_weight`); it never takes them as stated.  A window's `Ledger` refuses a repeated
call index, (forward, position): the Lean statement's A11, `UniqueCallIndices`.

H-1T's (`H1tAnchors`) are `ncp-v2`'s fields, with its strip ids and tag tuples derived from them by the verifier:

    field        value                                                                     input
    strip ids    each strip's global index, its rows' // 16 (rows come in aligned strips)  `stripids`
    tag tuples   `h1t.tag_tuple(j)` of each of the weight's global columns j               `tags`
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..protocol import DomainError
from .ncp2 import Shape

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1


def _lane(v: int) -> list[int]:
    return [v & M32, (v >> 32) & M32]


@dataclass(frozen=True)
class Anchors:
    """One call's anchors, as the verifier derives them."""

    salt: bytes
    forward: int
    layer: int
    site: int
    weight: int
    rows: tuple[int, ...]
    cols: tuple[int, ...]

    @property
    def position(self) -> int:
        return (self.layer << 32) | self.site

    @property
    def call_index(self) -> tuple[int, int]:
        return self.forward, self.position

    def check(self, shape: Shape) -> None:
        """Refuse anchors that cannot be a call's: the wrong counts, repeats, or values outside their fields."""
        if len(self.salt) != 32:
            raise DomainError("the salt is 32 bytes")
        if not all(0 <= v <= M64 for v in (self.forward, self.weight)) or not all(0 <= v <= M32 for v in (self.layer, self.site)):
            raise DomainError("the forward index and the weight id are 64-bit, the layer and the site 32-bit")
        for name, ids, count in (("row", self.rows, shape.m), ("column", self.cols, shape.n)):
            if len(ids) != count or len(set(ids)) != count or not all(0 <= v <= M32 for v in ids):
                raise DomainError(f"the {name} ids are {count} distinct 32-bit global indices")

    def _salt_words(self) -> list[int]:
        return [int.from_bytes(self.salt[4 * i:4 * i + 4], "little") for i in range(8)]

    def call_inputs(self) -> dict[str, list[int]]:
        """`NcpLinear`'s anchored inputs, by parameter."""
        return {"salt": self._salt_words(), "forward": _lane(self.forward), "position": _lane(self.position),
                "weight": _lane(self.weight), "rowids": list(self.rows)}

    def weight_inputs(self) -> dict[str, list[int]]:
        """`NcpWeight`'s anchored inputs, by parameter."""
        return {"salt": self._salt_words(), "weight": _lane(self.weight), "colids": list(self.cols)}

    def check_call(self, fn, args: Sequence[Sequence[int]]) -> None:
        """Refuse a call's flat input leaves whose anchored inputs are not these anchors."""
        _check(fn, args, self.call_inputs())

    def check_weight(self, fn, args: Sequence[Sequence[int]]) -> None:
        _check(fn, args, self.weight_inputs())


@dataclass(frozen=True)
class H1tAnchors:
    """One H-1T call's anchors: `ncp-v2`'s fields, from which the verifier also builds the strip ids and the tag tuples."""

    salt: bytes
    forward: int
    layer: int
    site: int
    weight: int
    rows: tuple[int, ...]
    cols: tuple[int, ...]

    @property
    def position(self) -> int:
        return (self.layer << 32) | self.site

    @property
    def call_index(self) -> tuple[int, int]:
        return self.forward, self.position

    def check(self, shape) -> None:
        """Refuse anchors that cannot be a call's: `Anchors.check`'s conditions, rows in aligned strips of 16, and columns
        inside the tag tuples' injective range."""
        from .h1t import TAG_COLUMNS

        Anchors(self.salt, self.forward, self.layer, self.site, self.weight, self.rows, self.cols).check(shape)
        for s in range(shape.m // 16):
            first = self.rows[16 * s]
            if first % 16 or self.rows[16 * s:16 * s + 16] != tuple(range(first, first + 16)):
                raise DomainError("H-1T's rows come in aligned strips of 16: a strip's id is its rows' // 16")
        if max(self.cols) >= TAG_COLUMNS:
            raise DomainError(f"the tag tuples are injective for global columns below {TAG_COLUMNS}")

    def call_inputs(self) -> dict[str, list[int]]:
        """`H1tLinear`'s anchored inputs, by parameter."""
        from .h1t import tag_tuple

        return {"salt": [int.from_bytes(self.salt[4 * i:4 * i + 4], "little") for i in range(8)],
                "forward": _lane(self.forward), "position": _lane(self.position), "weight": _lane(self.weight),
                "stripids": [self.rows[16 * s] // 16 for s in range(len(self.rows) // 16)], "rowids": list(self.rows),
                "tags": [c for j in self.cols for c in tag_tuple(j)]}

    def check_call(self, fn, args: Sequence[Sequence[int]]) -> None:
        """Refuse a call's flat input leaves whose anchored inputs are not these anchors'."""
        _check(fn, args, self.call_inputs())


def _check(fn, args, want: dict[str, list[int]]) -> None:
    params, _ = fn.signature()
    names = [name for name, _t in params]
    if len(args) != len(names):
        raise DomainError(f"{fn.id} takes {len(names)} inputs, not {len(args)}")
    for name, value in want.items():
        if list(args[names.index(name)]) != value:
            raise DomainError(f"{fn.id}: the anchored input {name!r} is not the verifier's")


@dataclass
class Ledger:
    """A window's calls, as the verifier admits them: a repeated call index is refused."""

    seen: set = field(default_factory=set)

    def admit(self, anchors: Anchors | H1tAnchors, shape) -> None:
        anchors.check(shape)
        if anchors.call_index in self.seen:
            raise DomainError(f"call index (forward, position) = {anchors.call_index} is repeated in the window")
        self.seen.add(anchors.call_index)


__all__ = ["Anchors", "H1tAnchors", "Ledger"]

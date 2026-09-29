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

NCP-FP8's (`Fp8Anchors`, X-SPC-89) are its declaration, (salt, index, weight id), and ids that are functions of the shape
alone, which the verifier builds the same way:

    field        value                                                                     input
    salt, weight the run's salt and the registered id of the matrix B is: the Program's    `prefix`
      id         gates over these alone, evaluated natively (`fp8.prefix`, X-SPC-90)
    index        the matmul's index, the call index                                        `index`
    row ranks    0 … m−1, root_A's leaves (`audit.commit_rows`)                            `rowids`
    row lines    the row ranks: E₁'s row i is row i's                                      `lines`
    column lines 0 … n−1, F₁'s column j is B's column j under the weight id (#295)         `colids`
    pad ranks    m … width−1, the tree's pad leaves                                        `padids`
    upper nodes  (level, index) of each node above the strips (`fp8.upper_nodes`)          `nodeids`

So a row's rank and its line, two inputs, are equal by construction.
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
class Fp8Anchors:
    """One NCP-FP8 call's anchors: its shape and declaration, from which the verifier builds every id input."""

    shape: object                         # `fp8.Shape`
    salt: bytes
    index: int
    weight: int
    sums: str = "f16"                     # the scheme's D-3s sums: the streams' tag, which the prefix reads

    @property
    def call_index(self) -> int:
        return self.index

    @property
    def lengths(self):
        """The Definition's byte lengths (`fp8.Lengths`) these anchors' index and weight id take."""
        from .fp8 import Lengths

        return Lengths.of(self.index, self.weight)

    def check(self, shape) -> None:
        if shape != self.shape:
            raise DomainError(f"the anchors are a {self.shape} call's, not a {shape} call's")
        shape.check()
        if len(self.salt) != 32:
            raise DomainError("the salt is 32 bytes")
        if not all(0 <= v <= M64 for v in (self.index, self.weight)):
            raise DomainError("the index and the weight id are 64-bit")

    def call_inputs(self) -> dict[str, list[int]]:
        """`NcpFp8Linear`'s anchored inputs, by parameter."""
        from .fp8 import prefix, upper_nodes

        m, n, width = self.shape.m, self.shape.n, self.shape.width
        out = {"prefix": prefix(self.shape, self.sums, self.lengths, self.salt, self.weight), "index": _lane(self.index)[::-1],
               "rowids": list(range(m)), "lines": list(range(m)), "colids": list(range(n))}
        if width > m:
            out["padids"] = list(range(m, width))
        if upper_nodes(self.shape):
            out["nodeids"] = [v for node in upper_nodes(self.shape) for v in node]
        return out

    def weight_inputs(self) -> dict[str, list[int]]:
        """`NcpFp8Weight`'s anchored inputs: none, since h and the amplitudes read B alone."""
        return {}

    def check_call(self, fn, args: Sequence[Sequence[int]]) -> None:
        """Refuse a call's flat input leaves whose anchored inputs are not these anchors'."""
        _check(fn, args, self.call_inputs())


def fp8_window_inputs(anchors: Sequence[Fp8Anchors]) -> dict[str, list[int]]:
    """`NcpFp8Window`'s per-call anchored inputs (X-SPC-102), each call's own (`Fp8Anchors.call_inputs`) in call order,
    after a `Ledger` refuses a repeated index; every call is of one shape, sums and set of byte lengths, and of the
    window's one salt and weight id (X-SPC-108): the window reads its weight side once."""
    if not anchors:
        raise DomainError("a window has at least one call")
    first, ledger = anchors[0], Ledger()
    for an in anchors:
        if (an.shape, an.sums, an.lengths) != (first.shape, first.sums, first.lengths):
            raise DomainError("a window's calls share one shape, sums and byte lengths")
        if (an.salt, an.weight) != (first.salt, first.weight):
            raise DomainError("a window's calls share one salt and weight id")
        ledger.admit(an, an.shape)
    per = [an.call_inputs() for an in anchors]
    return {p: [v for d in per for v in d[p]] for p in per[0]}


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

    def admit(self, anchors: Anchors | Fp8Anchors, shape) -> None:
        anchors.check(shape)
        if anchors.call_index in self.seen:
            raise DomainError(f"call index {anchors.call_index} is repeated in the window")
        self.seen.add(anchors.call_index)


__all__ = ["Anchors", "Fp8Anchors", "Ledger", "fp8_window_inputs"]

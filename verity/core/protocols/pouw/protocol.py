"""The ``PoUWScheme`` interface: what every named proof-of-useful-work scheme exposes to the protocol (`audit`).

A scheme turns one declared matmul's committed operand rows into the checked values of each tile, the tile's
transcript leaf and the useful output.  Everything else (commitments, the salt, the sample, the openings, the profile) is
the protocol's, in `audit`, and is the same for every scheme.  Like `verity.ml.commitments.scheme.CommitmentScheme`, a
scheme is a named, versioned object: its ``name`` is what `verity.core.protocols.pouw.schemes.get` looks up, and its spec section in
``PROTOCOL.md`` and its pinned vectors under ``tests/vectors`` carry the same string.

Operands are committed as rows along k: ``A`` is ``m x k`` and the weight ``B`` is stored as its ``n`` rows, so a matmul
computes ``A·Bᵀ`` and a tile (rows of A, rows of B) reads only its own strips.  Each scheme fixes its row encoding.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol, runtime_checkable

from verity.ml.randomness import Part, frame

U64 = 1 << 64
CALL_RANGE = 1 << 32
"""Call indices of one epoch (one salt) lie below this: -h3's row seed (`pearl_c.row_seed`) packs the index into the high
half of a u64."""


class DomainError(ValueError):
    """A declaration outside the scheme's domain; the verifier refuses it before any work."""


def tagged(tag: str, *parts: Part) -> bytes:
    """SHA-512 over ``tag \\0 || frame(part)...`` (`verity.ml.randomness.frame`), all 64 bytes."""
    return hashlib.sha512(tag.encode() + b"\0" + b"".join(frame(p) for p in parts)).digest()


@dataclass(frozen=True, slots=True)
class Shape:
    m: int
    k: int
    n: int

    def __post_init__(self) -> None:
        if not all(type(x) is int and x >= 1 for x in (self.m, self.k, self.n)):
            raise DomainError("a shape has m, k, n >= 1")

    @property
    def wmm(self) -> int:
        return self.m * self.k * self.n


@dataclass(frozen=True, slots=True)
class Matmul:
    """One declared GEMM of the workload: ``A (m x k) · Bᵀ`` with ``B`` the weight's ``n x k`` rows."""

    index: int
    shape: Shape
    weight: int

    def __post_init__(self) -> None:
        if not (type(self.index) is int and 0 <= self.index < U64 and type(self.weight) is int and 0 <= self.weight < U64):
            raise DomainError("a matmul's index and weight id are 64-bit")


@dataclass(frozen=True, slots=True)
class Tile:
    """The checked unit: rows of A and rows of B (columns of the output), at full k."""

    rows: tuple[int, ...]
    cols: tuple[int, ...]

    @property
    def cells(self) -> int:
        return len(self.rows) * len(self.cols)


def grid_tiles(shape: Shape, h: int, w: int) -> list[Tile]:
    """Consecutive ``h x w`` tiles in row-major order; the last row or column block may be narrower."""
    return [Tile(tuple(range(i, min(i + h, shape.m))), tuple(range(j, min(j + w, shape.n))))
            for i in range(0, shape.m, h) for j in range(0, shape.n, w)]


@dataclass(frozen=True, slots=True)
class Context:
    """What a scheme's noise may depend on for one matmul: the epoch's salt, the declaration and the committed roots."""

    salt: bytes
    matmul: Matmul
    root_a: bytes
    root_b: bytes


@dataclass(frozen=True, slots=True)
class Certificate:
    """What a scheme's guarantee rests on.

    ``omega`` is Ω*, the largest W_ref over the work the conjecture credits, across the domain (m·k·n for TT, 3·m·k·n for
    TT_NCP); ``gamma = 1 − (1 − gamma0)/omega`` is the conservative reduction (Lean `gammaFromTT`, `gammaFromTTNCP`).  A
    scheme whose conjecture names no target has ``gamma0 = None`` and no gamma.  ``claims`` are `verity.core.claims` ids, or ids
    in the same format that no render cites yet; ``theorems`` are pinned Lean declarations; ``allowances`` are gaps the
    scheme admits by design; ``parameters`` are the values the theorems are instantiated at, including the domain bounds
    their γ holds on."""

    conjecture: str
    gamma0: Fraction | None
    omega: Fraction
    claims: tuple[str, ...]
    theorems: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    allowances: tuple[str, ...] = ()
    parameters: tuple[tuple[str, int], ...] = ()

    @property
    def gamma(self) -> Fraction | None:
        return None if self.gamma0 is None else 1 - (1 - self.gamma0) / self.omega


@runtime_checkable
class PoUWScheme(Protocol):
    #: The scheme's registered name; its spec section and pinned vectors carry the same string.
    name: str
    #: What an accepted audit's guarantee rests on.
    certificate: Certificate

    def check(self, matmul: Matmul) -> None:
        """Raise `DomainError` when the declaration is outside the domain."""
        ...

    def tiles(self, shape: Shape) -> Sequence[Tile]:
        """The checked units of one matmul; they partition its ``m x n`` output cells."""
        ...

    def wref(self, shape: Shape) -> int:
        """The honest reference's cost in W1 units, read from the layout only, never from values."""
        ...

    def noise(self, ctx: Context) -> Any:
        """Every hash-derived value of one matmul, random-access by row (per-weight parts may be memoized)."""
        ...

    def encode(self, ctx: Context, noise: Any, a_rows: Sequence[bytes], b_rows: Sequence[bytes], tile: Tile) -> Any:
        """The checked product's operands for one tile from its strips (``a_rows[i]`` is row ``tile.rows[i]``), or
        None when a row is not a well-formed operand row."""
        ...

    def admissible(self, ctx: Context, ops: Any) -> bool:
        """The scheme's input policy on one tile; a worst-case-input scheme accepts every well-formed tile."""
        ...

    def checked(self, ctx: Context, ops: Any) -> Any:
        """The tile's checked values (the reference computation)."""
        ...

    def digest(self, ctx: Context, noise: Any, checked: Any) -> bytes:
        """The tile's transcript leaf."""
        ...

    def useful(self, ctx: Context, noise: Any, ops: Any, checked: Any) -> list[list]:
        """The tile's useful output, ``len(rows) x len(cols)``."""
        ...


__all__ = ["U64", "Certificate", "Context", "DomainError", "Matmul", "PoUWScheme", "Shape", "Tile", "grid_tiles", "tagged"]

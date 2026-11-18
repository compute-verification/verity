"""PoUB's refresh rota: the window each credited chunk's re-key joins and the key a plan may hold it under, from public
parameters alone."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

REGISTRATION = 0
"""The window a registered block's first key counts as joined: the registration's, before window 0's first root."""
REKEY_MAX = 512
"""The most blocks a re-key append holds, so its opening unit stays one proof block."""

Origin = tuple[str, int] | tuple[str, int, int]
"""A re-key's origin as `audit.AuditRun.append` takes it: `("registered", a)` or `("append", k, o)`."""

Fresh = tuple[int, int, int]
"""A fresh append as the rota reads it: (its index k among the session's appends, re-keys counted; the window it
joined; its count)."""


@dataclass(frozen=True, slots=True)
class Rota:
    """The rota of a session over `blocks` registered blocks of `windows` windows, the refresh period `lifetime` = L
    windows (`audit.Params.lifetime`), read from no coin and no payload.

    A chunk is a registered block (base window 0, class `j·L // N`, so a class is one run) or a fresh append's block
    (base the window the append joined, class its fresh ordinal mod L). Its re-keys join the windows ≡ its class mod L
    from base + 1 on, one in each period of L windows after its base, each of its origin, never of an earlier re-key.
    A plan of window t holds it under its newest key joined at or before t: the registration's while t < L for a
    registered block, else its newest re-key's, joined fewer than L windows before t; a fresh block's only from
    base + L on (probation), never under its own append's key."""

    blocks: int
    lifetime: int
    windows: int

    def __post_init__(self) -> None:
        for name in ("blocks", "lifetime", "windows"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"a rota's {name} is a positive integer, not {getattr(self, name)!r}")

    def registered_class(self, j: int) -> int:
        if type(j) is not int or not 0 <= j < self.blocks:
            raise ValueError(f"a registered block is below {self.blocks}, not {j!r}")
        return j * self.lifetime // self.blocks

    def append_class(self, f: int) -> int:
        """The class of the session's f-th fresh append, from 0, re-keys not counted."""
        if type(f) is not int or f < 0:
            raise ValueError(f"a fresh append's ordinal is a nonnegative integer, not {f!r}")
        return f % self.lifetime

    def class_runs(self, c: int) -> tuple[int, int]:
        """The registered blocks of class c, (first, count): j with c·N ≤ j·L < (c + 1)·N; count 0 where L > N leaves
        the class empty."""
        n, life = self.blocks, self.lifetime
        first, end = -(-c * n // life), -(-(c + 1) * n // life)
        return first, end - first

    def first(self, base: int, c: int) -> int:
        """The window of the first re-key of a chunk of base window `base` and class `c`: the one in
        [base + 1, base + L] congruent to c mod L."""
        return base + 1 + (c - base - 1) % self.lifetime

    def rekeys(self, base: int, c: int) -> range:
        """Every window a chunk of base `base` and class `c` is re-keyed to join, below the session's last."""
        return range(self.first(base, c), self.windows, self.lifetime)

    def key(self, base: int, c: int, t: int, *, registered: bool) -> int | None:
        """The window the key a plan of window t may hold the chunk under joined (`REGISTRATION` for the
        registration's), or None where no plan of t may hold it: before its probation ends, or past the session."""
        if type(t) is not int or not 0 <= t < self.windows:
            return None
        if not registered and t < base + self.lifetime:
            return None
        first = self.first(base, c)
        if t < first:
            return REGISTRATION if registered else None
        return t - (t - first) % self.lifetime

    def registered_due(self, r: int) -> tuple[tuple[int, int], ...]:
        """The registered runs (first, count) whose re-keys join window r: its class's run in pieces of at most
        `REKEY_MAX` blocks, in index order; none outside windows 1 … W − 1."""
        if not 1 <= r < self.windows:
            return ()
        first, count = self.class_runs(r % self.lifetime)
        return tuple((a, min(REKEY_MAX, first + count - a)) for a in range(first, first + count, REKEY_MAX))

    def appends_due(self, r: int, fresh: Sequence[Fresh]) -> tuple[tuple[int, int, int], ...]:
        """The pieces (k, o, count) of the fresh appends whose re-keys join window r, in order: `fresh` is every fresh
        append the session received, in order, so a fresh append's ordinal is its place in it; each due append is
        re-keyed whole, from offset 0 in pieces of at most `REKEY_MAX` blocks."""
        return tuple((k, o, min(REKEY_MAX, count - o))
                     for f, (k, b, count) in enumerate(fresh)
                     if b < r < self.windows and (r - self.first(b, self.append_class(f))) % self.lifetime == 0
                     for o in range(0, count, REKEY_MAX))

    def due(self, r: int, fresh: Sequence[Fresh]) -> tuple[tuple[Origin, int], ...]:
        """Every re-key that joins window r, (origin, count), in the order the prover sends them: the registered runs',
        then the fresh appends' pieces."""
        return (*((("registered", a), n) for a, n in self.registered_due(r)),
                *((("append", k, o), n) for k, o, n in self.appends_due(r, fresh)))


__all__ = ["REGISTRATION", "REKEY_MAX", "Fresh", "Origin", "Rota"]

"""PoUB's refresh rota: the window each credited chunk's re-key joins and the key a plan may hold it under, from public
parameters alone."""

from __future__ import annotations

from dataclasses import dataclass

REGISTRATION = 0
"""The window a registered block's first key counts as joined: the registration's, before window 0's first root."""


@dataclass(frozen=True, slots=True)
class Rota:
    """The rota of a session over `blocks` registered blocks of `windows` windows, the refresh period `lifetime` = L
    windows (`audit.Params.lifetime`), read from no coin and no payload.

    A chunk is a registered block (base window 0, class `j·L // N`, so a class is one run) or a fresh append's block
    (base the window the append joined, class its ordinal mod L). Its re-keys join the windows ≡ its class mod L from
    base + 1 on, one in each period of L windows after its base. A plan of window t holds it under its newest key joined
    at or before t: the registration's while t < L for a registered block, else its newest re-key's, joined fewer than
    L windows before t; a fresh block's only from base + L on (probation), never under its own append's key."""

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

    def append_class(self, k: int) -> int:
        if type(k) is not int or k < 0:
            raise ValueError(f"a fresh append's ordinal is a nonnegative integer, not {k!r}")
        return k % self.lifetime

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

    def registered_due(self, r: int) -> tuple[int, int] | None:
        """The registered run (first, count) whose re-key joins window r, None where none does."""
        if not 1 <= r < self.windows:
            return None
        first, count = self.class_runs(r % self.lifetime)
        return (first, count) if count else None

    def appends_due(self, r: int, born: list[int] | tuple[int, ...]) -> tuple[int, ...]:
        """The ordinals of the fresh appends whose re-key joins window r, `born[k]` the window fresh append k joined."""
        return tuple(k for k, b in enumerate(born)
                     if b < r < self.windows and (r - self.first(b, self.append_class(k))) % self.lifetime == 0)


__all__ = ["REGISTRATION", "Rota"]

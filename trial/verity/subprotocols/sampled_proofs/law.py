"""The draw's law and what it establishes. Today's references: `verity_sampled_proofs.one_stage.draw` (the laws, and
`subset`/`stratified`/`work`) and `verity.proofs.profile` (`IntegrityProfile.accept` and `bound`).

`Subset(k)`: k distinct units, uniform over the n registered. A transcript with m wrong units passes every drawn check
with probability C(n - m, k) / C(n, k); so, except with probability δ, an accepted transcript has at most `bound` wrong.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from math import comb


@dataclass(frozen=True)
class Subset:
    k: int

    def accept(self, n: int, m: int) -> Fraction:
        """The exact probability that a transcript with m wrong of n units passes."""
        return Fraction(comb(n - m, self.k), comb(n, self.k))

    def bound(self, n: int, delta: float) -> int:
        """The most wrong units an accepted transcript has, except with probability δ: the largest m that passes with
        probability above δ."""
        m = 0
        while m < n and self.accept(n, m + 1) > delta:
            m += 1
        return m

    def draw(self, coins: bytes, n: int) -> tuple[int, ...]:
        """min(k, n) distinct units, exactly uniform: counter-mode SHA-256 with rejection (`verity.randomness.subset`)."""
        chosen: list[int] = []
        limit = (2**256 // n) * n if n else 0
        counter = 0
        while len(chosen) < min(self.k, n):
            x = int.from_bytes(hashlib.sha256(coins + counter.to_bytes(8, "big")).digest(), "big")
            counter += 1
            if x < limit and x % n not in chosen:
                chosen.append(x % n)
        return tuple(sorted(chosen))

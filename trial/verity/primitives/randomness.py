"""Coins, and the laws that sample with them.

A verifier draws from a secret only it holds, after the commitments the draw must follow are in its view, and in a
domain of its own, so no two draws share coins (`derive`, standing in for `verity.randomness.derive`).

`Weighted(gates, outputs)` samples units independently, each with probability proportional to its gates or to its output
bits, whichever asks more: Poisson sampling, the continuous form of the main repo's `work` law
(`verity_sampled_proofs.one_stage.draw`), which sizes each stratum's draw by its work. With a = ln(1/δ) / (`gates` ×
all units' gates) and b likewise for output bits, unit u is drawn with probability p_u = min(1, max(a g_u, b o_u)).
Bad units S escape every draw with probability Π (1 - p_u) <= exp(-Σ p_u), which is below δ once S holds more than
`gates` of the gates or `outputs` of the output bits. So the bound is in what the account reads, however unequal the
units (a batch of eight requests beside a batch of one, an adder beside a comparison), and the draw is about
ln(1/δ) (1/gates + 1/outputs) units, however many there are.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass


def derive(secret: bytes, domain: str, context: str, size: int = 32) -> bytes:
    """`size` bytes nobody without `secret` can predict, bound to `domain` and `context`."""
    return hashlib.shake_256(b"\0".join((secret, domain.encode(), context.encode()))).digest(size)


@dataclass(frozen=True)
class Weighted:
    gates: float  # the share of all units' gates the bound allows in bad units, in (0, 1]
    outputs: float  # and of their output bits

    def rates(self, gates: Sequence[int], outputs: Sequence[int], delta: float) -> list[float]:
        log = math.log(1 / delta)
        a = log / (self.gates * sum(gates)) if sum(gates) else 0.0
        b = log / (self.outputs * sum(outputs)) if sum(outputs) else 0.0
        return [min(1.0, max(a * g, b * o)) for g, o in zip(gates, outputs)]

    def draw(self, coins: bytes, gates: Sequence[int], outputs: Sequence[int], delta: float) -> tuple[int, ...]:
        rates = self.rates(gates, outputs, delta)
        return tuple(u for u, p in enumerate(rates) if _coin(coins, u) < p * 2**256)

    def bound(self, gates: Sequence[int], outputs: Sequence[int], delta: float) -> tuple[int, int]:
        """The most gates and output bits in bad units when every drawn unit was good, except with probability δ: the
        law's shares, and never more than the units it might not have drawn hold, since a unit drawn surely was
        checked."""
        unsure = [u for u, p in enumerate(self.rates(gates, outputs, delta)) if p < 1]
        return (
            min(int(self.gates * sum(gates)), sum(gates[u] for u in unsure)),
            min(int(self.outputs * sum(outputs)), sum(outputs[u] for u in unsure)),
        )


def _coin(coins: bytes, unit: int) -> int:
    """The unit's coin, uniform over [0, 2**256): SHA-256 over the coins and the unit's index, as the main repo's
    `verity.randomness` samplers draw."""
    return int.from_bytes(hashlib.sha256(coins + unit.to_bytes(8, "big")).digest(), "big")

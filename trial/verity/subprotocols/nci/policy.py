"""`Nci.Complies Γ α`: the declared computation splits into isolation units of at most Γ gates, each reading at most
K = α·Γ bits beyond the baseline θ, counting every route: its declared novel reads, the covert ingress the warden could
not rule out since its last wipe, and memory on its devices that the evidence leaves unaccounted, which could have
survived the wipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class NciBound:
    gates: int  # Γ
    alpha: Fraction  # α, bits read per gate

    @property
    def budget_bits(self) -> int:
        return int(self.alpha * self.gates)


@dataclass(frozen=True)
class Charge:
    """What one isolation unit is charged."""

    gates: int
    novel_bits: int
    covert_bits: int | None  # None when nothing bounds its channel
    surviving_bits: int


def complies(charges: Mapping[str, Charge], bound: NciBound) -> bool:
    return all(
        c.covert_bits is not None and c.gates <= bound.gates
        and c.novel_bits + c.covert_bits + c.surviving_bits <= bound.budget_bits
        for c in charges.values()
    )

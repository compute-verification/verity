"""The policy: the rules compliance checks, with their thresholds, set in the configuration.

A rule here is its name and parameters; its meaning is the compliance spec's (`protocols/nci/lean` today), and a
compliance proof shows the committed account satisfies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar, Protocol


class Rule(Protocol):
    name: str


@dataclass(frozen=True)
class NciBound:
    """Bounded novel communication intensity (`Nci.Complies Γ α`).

    The declared computation splits into isolation units of at most Γ gates, each reading at most α·size bits beyond the
    baseline θ committed up front. Each isolation unit is also charged, against K = α·Γ, what accounting could not rule
    out reaching it another way: the warden's covert ingress capacity since its last wipe, and memory on its spatial unit
    that the evidence leaves unaccounted, which could have survived the wipe.
    """

    gates: int  # Γ
    alpha: Fraction  # α, bits read per gate
    name: ClassVar[str] = "nci"

    @property
    def budget_bits(self) -> int:
        """K = α·Γ: what one isolation unit may read beyond the baseline, by any route."""
        return int(self.alpha * self.gates)


Policy = tuple[Rule, ...]

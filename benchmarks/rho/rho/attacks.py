"""Attacks: constructions that finish counted products in a block, so ρ's upper end.

An attack is an `Argument` on the `attack` side: `allowance(profile)` is what the construction finishes in a block with
that budget (never more than the block's compute, which `theorem.share` caps). Its `cites` are what it rests on: an
analysis (`draft:`), a measured run (`art:`), a finding (`note:`). The best attack for an algorithm sets its `hi`; a
bound above an attack's achieved count (`lo > hi`) means a premise is false for that algorithm at that budget, and
`scenario.evaluate` reports it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from rho.classes import Argument
from rho.profile import Profile
from rho.theorem import Allowance, Budget


@dataclass(frozen=True)
class Comply:
    """Run the declared work as declared and train only on free compute: always available."""

    name: str = "comply"
    side: Literal["bound", "attack"] = "attack"
    cites: tuple[str, ...] = ()

    def allowance(self, p: Profile) -> Allowance:
        return lambda budget: 0.0


@dataclass(frozen=True)
class StateFits:
    """A block whose novel input holds the whole trained state trains unthrottled (the small-state exception, N1): it
    reads the state in, runs the reference algorithm on its compute, and leaves the state behind for the next."""

    name: str = "state-fits"
    side: Literal["bound", "attack"] = "attack"
    cites: tuple[str, ...] = ("draft:N1",)

    def allowance(self, p: Profile) -> Allowance:
        def a(budget: Budget) -> float:
            return p.kappa * budget.compute / 2 if p.state_bits <= budget.inputs else 0.0
        return a


ATTACKS: Mapping[str, Argument] = {a.name: a for a in (Comply(), StateFits())}

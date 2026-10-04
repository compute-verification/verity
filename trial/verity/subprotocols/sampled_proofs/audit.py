"""The audit: the units are fixed (committed) before the challenger draws which to verify.

The prover answers `executions()`: per unit, in order, its circuit, inputs and witness (a trace for replay, a proof for a
proof construction).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.primitives import registry
from verity.primitives.randomness import Challenger
from verity.subprotocols.sampled_proofs.replay import Replay

SHIPPED = {Replay.name: Replay}


@dataclass(frozen=True)
class IntegrityProfile:
    accepted: bool
    delta: float  # the probability of accepting while more than `wrong_units` units are wrong
    covered: frozenset[str]  # the circuits whose units the draw was over
    units: int
    wrong_units: int
    record: Mapping[str, object]


def draw(coins: bytes, units: int, k: int) -> list[int]:
    """k distinct unit indices, uniform, from the coins."""
    chosen: list[int] = []
    counter = 0
    while len(chosen) < min(k, units):
        x = int.from_bytes(hashlib.sha256(coins + counter.to_bytes(8, "big")).digest(), "big") % units
        counter += 1
        if x not in chosen:
            chosen.append(x)
    return chosen


class Audit:
    def __init__(self, construction: str, parameters: Mapping[str, object]):
        self.construction = registry.load("sampled_proofs", construction, SHIPPED)
        self.k = int(parameters.get("k", 8))
        self.fraction = float(parameters.get("fraction", 0.25))  # the wrong fraction the bound is stated at

    def open(self, challenger: Challenger) -> None:
        self.challenger = challenger

    def close(self, units: Sequence[str], units_digest: str, prover: object) -> IntegrityProfile:
        """`units`: each unit's circuit id, as committed; `units_digest` binds the draw to them."""
        executions = prover.executions()
        n = len(units)
        drawn = draw(self.challenger.draw("units", units=units_digest), n, self.k)
        failed = [i for i in drawn if executions[i][0].id() != units[i] or not self.construction.verify_unit(*executions[i])]
        exhaustive = len(drawn) == n
        return IntegrityProfile(
            accepted=not failed,
            delta=0.0 if exhaustive else (1 - self.fraction) ** len(drawn),
            covered=frozenset(units),
            units=n,
            wrong_units=0 if exhaustive else math.floor(self.fraction * n),
            record={"drawn": drawn, "failed": failed},
        )

"""The audit, a toy of `verity_pouw`'s shape: the beacon salt is drawn when it opens, and at close the prover answers
`certified(salt)` per device and is believed; the real scheme checks sampled tiles against the salt."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.primitives import registry
from verity.primitives.randomness import Challenger
from verity.subprotocols.pouw.pearl_c import PearlC

SHIPPED = {PearlC.name: PearlC}


@dataclass(frozen=True)
class WorkProfile:
    accepted: bool
    delta: float
    certified: Mapping[str, int]  # per device: work units shown performed, (1 - γ)(1 - ε_s)·W
    record: Mapping[str, object]


class Audit:
    def __init__(self, construction: str, parameters: Mapping[str, object]):
        self.construction = registry.load("pouw", construction, SHIPPED)

    def open(self, challenger: Challenger) -> None:
        self.salt = challenger.draw("salt")

    def close(self, prover: object) -> WorkProfile:
        return WorkProfile(True, self.construction.delta, dict(prover.certified(self.salt)), {"salt": self.salt.hex()})

"""The audit, a toy of `verity_pous`'s shape: the salt is drawn when it opens (after the weights commitment exists), the
prover encodes under it, and at close the audit challenges and reads back what each device holds. Here the prover
answers `occupied(salt)` and is believed; the real audit times the responses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.primitives import registry
from verity.primitives.commitments import Commitment
from verity.primitives.randomness import Challenger
from verity.subprotocols.pous.band_chain import BandChain

SHIPPED = {BandChain.name: BandChain}


@dataclass(frozen=True)
class StorageProfile:
    accepted: bool
    delta: float
    occupied_bits: Mapping[str, int]  # per device: memory shown to hold the encoding
    record: Mapping[str, object]


class Audit:
    def __init__(self, construction: str, parameters: Mapping[str, object]):
        self.construction = registry.load("pous", construction, SHIPPED)

    def open(self, weights: Commitment, challenger: Challenger) -> None:
        self.weights = weights
        self.salt = challenger.draw("salt")

    def close(self, prover: object) -> StorageProfile:
        occupied = dict(prover.occupied(self.salt))
        return StorageProfile(True, self.construction.delta, occupied, {"salt": self.salt.hex()})

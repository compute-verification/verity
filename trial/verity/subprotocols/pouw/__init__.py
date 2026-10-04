"""Proofs of useful work: the declared work was performed, at a rate only the hardware of record reaches. Meets `work`.

A toy of `verity_pouw`'s shape: the beacon salt is drawn when the session opens, and at close the developer's side
answers `certified(salt)` per device and is believed; the real scheme checks sampled tiles against the salt.
"""

from __future__ import annotations

from collections.abc import Mapping

from verity.protocol.accounting.account import Account
from verity.protocol.accounting.interfaces import EpochStart, WorkFinding
from verity.protocol.challenges import Challenger
from verity.protocol.configuration import Configuration


class PearlCSession:
    def __init__(self, parameters: Mapping[str, object]):
        self.parameters = parameters

    def open(self, start: EpochStart, challenger: Challenger) -> None:
        self.salt = challenger.draw("salt")

    def close(self, account: Account, developer: object) -> WorkFinding:
        return WorkFinding(accepted=True, delta=0.01, record={"salt": self.salt.hex()}, certified=dict(developer.certified(self.salt)))


class PearlC:
    interface = "work"
    subprotocol = "pouw"
    name = "pearl-c"

    def refusals(self, configuration: Configuration) -> tuple[str, ...]:
        return ()

    def session(self, parameters: Mapping[str, object]) -> PearlCSession:
        return PearlCSession(parameters)

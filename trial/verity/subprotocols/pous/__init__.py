"""Proofs of useful space: memory holds the encoded weights of record, salted after they are committed. Meets `space`.

A toy of `verity_pous`'s shape: the session draws the salt when it opens (after the weights commitment exists), the
developer encodes under it while serving, and at close the session challenges and reads back what each device holds.
Here the developer's side answers `occupied(salt)` and is believed; the real audit times the responses.
"""

from __future__ import annotations

from collections.abc import Mapping

from verity.protocol.accounting.account import Account
from verity.protocol.accounting.interfaces import EpochStart, SpaceFinding
from verity.protocol.challenges import Challenger
from verity.protocol.configuration import Configuration


class BandChainSession:
    def __init__(self, parameters: Mapping[str, object]):
        self.parameters = parameters

    def open(self, start: EpochStart, challenger: Challenger) -> None:
        self.salt = challenger.draw("salt")

    def close(self, account: Account, developer: object) -> SpaceFinding:
        occupied = developer.occupied(self.salt)
        return SpaceFinding(accepted=True, delta=0.01, record={"salt": self.salt.hex()}, occupied_bits=dict(occupied))


class BandChain:
    interface = "space"
    subprotocol = "pous"
    name = "band-chain"

    def refusals(self, configuration: Configuration) -> tuple[str, ...]:
        return ()

    def session(self, parameters: Mapping[str, object]) -> BandChainSession:
        return BandChainSession(parameters)

"""The network warden: what crossed each spatial unit's boundary, and the covert timing capacity left. Meets `channel`.

Exact and coin-free. The warden's taps are the auditor's, not the developer's; here a `taps` callable stands in for them,
passed as a parameter, and returns covert ingress bits per isolation unit since its last wipe.
"""

from __future__ import annotations

from collections.abc import Mapping

from verity.protocol.accounting.account import Account
from verity.protocol.accounting.interfaces import ChannelFinding, EpochStart
from verity.protocol.challenges import Challenger
from verity.protocol.configuration import Configuration


class ConstantRateSession:
    def __init__(self, parameters: Mapping[str, object]):
        self.bits_per_unit = int(parameters.get("bits_per_unit", 0))
        self.violations = tuple(parameters.get("violations", ()))

    def open(self, start: EpochStart, challenger: Challenger) -> None:
        pass

    def close(self, account: Account, developer: object) -> ChannelFinding:
        covert = {u.id: self.bits_per_unit for u in account.units}
        return ChannelFinding(accepted=not self.violations, delta=0.0, record={"violations": list(self.violations)}, covert_bits=covert)


class ConstantRate:
    interface = "channel"
    subprotocol = "network_warden"
    name = "constant-rate"

    def refusals(self, configuration: Configuration) -> tuple[str, ...]:
        return ()

    def session(self, parameters: Mapping[str, object]) -> ConstantRateSession:
        return ConstantRateSession(parameters)

"""The audit. The taps are the auditor's, not the prover's; here the parameters stand in for them: `bits_per_unit` of
covert ingress capacity per isolation unit since its last wipe, and any `violations` the taps saw."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.primitives.registry import NotInstalled


@dataclass(frozen=True)
class ChannelProfile:
    accepted: bool
    covert_bits: Mapping[str, int]  # per isolation unit: ingress timing capacity since its last wipe; egress is not charged
    record: Mapping[str, object]
    delta: float = 0.0  # exact


class Audit:
    def __init__(self, construction: str, parameters: Mapping[str, object]):
        if construction != "constant-rate":
            raise NotInstalled(f"network_warden/{construction} is not installed (available: constant-rate)")
        self.bits_per_unit = int(parameters.get("bits_per_unit", 0))
        self.violations = tuple(parameters.get("violations", ()))

    def close(self, isolation_units: Sequence[str]) -> ChannelProfile:
        covert = {u: self.bits_per_unit for u in isolation_units}
        return ChannelProfile(not self.violations, covert, {"violations": list(self.violations)})

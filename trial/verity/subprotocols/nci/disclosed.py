"""The toy proof: the prover discloses the charges in the clear and the verifier checks the rule. The construction of
record proves `complies` in zero knowledge over a committed account, and the verifier never sees the charges."""

from __future__ import annotations

from collections.abc import Mapping

from verity.subprotocols.nci.policy import Charge, NciBound, complies


class Disclosed:
    name = "disclosed"

    def verify(self, charges: Mapping[str, Charge], bound: NciBound) -> bool:
        return complies(charges, bound)

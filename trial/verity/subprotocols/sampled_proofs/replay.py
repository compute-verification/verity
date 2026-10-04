"""The replay construction: the auditor recomputes each drawn execution's trace. Light; ships with `verity`."""

from __future__ import annotations

from collections.abc import Mapping

from verity.circuits import Circuit, trace
from verity.protocol.configuration import Configuration
from verity.subprotocols.sampled_proofs import SampledSession


def verify_unit(circuit: Circuit, inputs: tuple[int, ...], claimed: object) -> bool:
    return trace(circuit, inputs) == claimed


class Replay:
    interface = "integrity"
    subprotocol = "sampled_proofs"
    name = "replay"

    def refusals(self, configuration: Configuration) -> tuple[str, ...]:
        return ()

    def session(self, parameters: Mapping[str, object]) -> SampledSession:
        return SampledSession(parameters, verify_unit)

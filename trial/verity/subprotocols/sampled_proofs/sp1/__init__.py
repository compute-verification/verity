"""The SP1 construction: each drawn execution comes with a proof the auditor verifies. Heavy: its dependencies (here
`tomli_w`, standing in for the SP1 toolchain) come only with the `verity-sp1` distribution, whose `pyproject.toml` beside
this file declares them and registers the construction. Its source ships with `verity` and is never imported unless a
configuration names `sampled_proofs/sp1`.
"""

from __future__ import annotations

from collections.abc import Mapping

import tomli_w  # the heavy dependency

from verity.circuits import Circuit, trace
from verity.protocol.configuration import Configuration
from verity.subprotocols.sampled_proofs import SampledSession


def verify_unit(circuit: Circuit, inputs: tuple[int, ...], proof: object) -> bool:
    # A stand-in for SP1's verifier: the "proof" is the trace, serialized by the heavy dependency.
    return proof == tomli_w.dumps({"trace": list(trace(circuit, inputs))}) or proof == trace(circuit, inputs)


class Sp1:
    interface = "integrity"
    subprotocol = "sampled_proofs"
    name = "sp1"

    def refusals(self, configuration: Configuration) -> tuple[str, ...]:
        return ()

    def session(self, parameters: Mapping[str, object]) -> SampledSession:
        return SampledSession(parameters, verify_unit)

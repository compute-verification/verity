"""The SP1 construction: each drawn unit comes with a proof the auditor verifies. Heavy: its dependencies (here
`tomli_w`, standing in for the SP1 toolchain) come only with the `verity-sp1` distribution, whose `pyproject.toml` beside
this file declares them and registers the construction. Never imported unless a configuration names `sp1`.
"""

from __future__ import annotations

import tomli_w  # the heavy dependency

from verity.circuits.circuit import Circuit, trace


class Sp1:
    name = "sp1"

    def verify_unit(self, circuit: Circuit, inputs: tuple[int, ...], proof: object) -> bool:
        # A stand-in for SP1's verifier: the "proof" is the trace, serialized by the heavy dependency.
        return proof == tomli_w.dumps({"trace": list(trace(circuit, inputs))}) or proof == trace(circuit, inputs)

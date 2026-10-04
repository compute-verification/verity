"""The SP1 construction: the prover proves each drawn unit and the verifier checks the proof. Heavy: its dependencies
(here `tomli_w`, standing in for the SP1 toolchain) come only with the `verity-sp1` distribution, whose `pyproject.toml`
beside this file declares them and registers the construction. Never imported unless a configuration names `sp1`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import tomli_w  # the heavy dependency

from verity.circuits import partition as P
from verity.circuits.circuit import Circuit


def prove(circuit: Circuit, region: Sequence[int], inputs: Mapping[int, int]) -> str:
    """A stand-in proof: the region's outputs, serialized by the heavy dependency."""
    produced = P.evaluate_region(circuit, region, inputs)
    return tomli_w.dumps({"outputs": {str(w): v for w, v in sorted(produced.items())}})


class Sp1:
    name = "sp1"

    def verify_unit(self, circuit: Circuit, region: Sequence[int], inputs: Mapping[int, int], outputs: Mapping[int, int], witness: object) -> bool:
        produced = P.evaluate_region(circuit, region, inputs)
        return witness == prove(circuit, region, inputs) and all(produced[w] == v for w, v in outputs.items())

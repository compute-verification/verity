"""The SP1 construction: the prover proves each drawn unit and the verifier checks the proof. Heavy: its dependencies
(here `tomli_w`, standing in for the SP1 toolchain) come only with the `verity-sp1` distribution, whose `pyproject.toml`
beside this file declares them. Never imported unless someone asks for `Sp1`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import tomli_w  # the heavy dependency

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit


class Sp1:
    def prove(self, circuit: Circuit, region: Sequence[int], inputs: Mapping[int, int]) -> str:
        """A stand-in proof: the region's outputs, serialized by the heavy dependency."""
        produced = P.evaluate_region(circuit, region, inputs)
        return tomli_w.dumps({"outputs": {str(w): v for w, v in sorted(produced.items())}})

    def verify(self, circuit: Circuit, region: Sequence[int], inputs: Mapping[int, int], outputs: Mapping[int, int], witness: object) -> bool:
        produced = P.evaluate_region(circuit, region, inputs)
        return witness == self.prove(circuit, region, inputs) and all(produced[w] == v for w, v in outputs.items())

"""The replay construction: the verifier recomputes the drawn unit's region from its opened inputs and compares its
opened outputs. Transparent (the openings are the proof), so it reveals the unit's boundary values. Light; ships with
`verity`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from verity.circuits import partition as P
from verity.circuits.circuit import Circuit


class Replay:
    name = "replay"

    def verify_unit(self, circuit: Circuit, region: Sequence[int], inputs: Mapping[int, int], outputs: Mapping[int, int], witness: object) -> bool:
        produced = P.evaluate_region(circuit, region, inputs)
        return all(produced[w] == v for w, v in outputs.items())

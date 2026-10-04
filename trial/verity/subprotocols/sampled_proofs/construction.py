"""How a drawn unit is proved and checked. `Replay` ships here: the openings are the proof, and the verifier recomputes
the unit's region from its opened inputs and compares its opened outputs; transparent, so it reveals the unit's boundary
values. A heavy one (`sp1`) ships in its own distribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit


class Construction(Protocol):
    def prove(self, circuit: Circuit, region: Sequence[int], inputs: Mapping[int, int]) -> object: ...

    def verify(self, circuit: Circuit, region: Sequence[int], inputs: Mapping[int, int], outputs: Mapping[int, int], witness: object) -> bool: ...


class Replay:
    def prove(self, circuit: Circuit, region: Sequence[int], inputs: Mapping[int, int]) -> None:
        return None

    def verify(self, circuit: Circuit, region: Sequence[int], inputs: Mapping[int, int], outputs: Mapping[int, int], witness: object) -> bool:
        produced = P.evaluate_region(circuit, region, inputs)
        return all(produced[w] == v for w, v in outputs.items())

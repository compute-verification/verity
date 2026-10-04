"""What a proof system proves, and its interface. A unit is one region of one streamed execution, whose circuit and
partition the builder built from its advice; it is right when the outputs committed for it are what its region computes
from the inputs committed for it. A proof system proves a batch of units right against their roots, and checks such a
proof, in the verifier's view: a zero-knowledge one reveals nothing, a transparent one reveals the units' boundary
values."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit
from verity.primitives.commitments import Commitment, CommitmentScheme


@dataclass(frozen=True)
class Unit:
    circuit: Circuit
    partition: P.Partition
    root: Commitment
    gates: P.Region  # one of the partition's regions

    @property
    def outputs(self) -> tuple[int, ...]:
        """The circuit's outputs this unit computes."""
        return tuple(w for w in P.boundary(self.circuit, self.gates)[1] if w in _outputs(self.circuit))


@functools.cache
def _outputs(circuit: Circuit) -> frozenset[int]:
    return frozenset(circuit.outputs)


class ProofSystem(Protocol):
    def prove(self, scheme: CommitmentScheme, units: Sequence[Unit], values: Sequence[Sequence[int]]) -> object:
        """One proof of every unit in the batch; `values` is every wire of each unit's execution, rematerialized."""

    def verify(self, scheme: CommitmentScheme, units: Sequence[Unit], proof: object) -> bool: ...

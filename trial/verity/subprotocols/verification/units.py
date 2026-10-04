"""What a unit is and how its values are committed, and the interface of anything that proves one.

A unit is one region of one execution of a circuit, under the partition the prover chose. The prover commits to every
boundary wire of an execution in one root: one leaf per wire, in wire order. A unit is right when the outputs committed
for it are what its region computes from the inputs committed for it. A `ProofSystem` proves that against the root,
without the verifier holding the execution: a replay by opening the unit's inputs and outputs, a zero-knowledge proof
system without revealing them.
"""

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
    region: int
    root: Commitment

    @property
    def gates(self) -> tuple[int, ...]:
        return self.partition[self.region]


@functools.cache
def wires(circuit: Circuit, partition: P.Partition) -> tuple[int, ...]:
    """Every boundary wire of the partition, in the order the root commits them."""
    return tuple(sorted({w for region in partition for side in P.boundary(circuit, region) for w in side}))


def leaf(wire: int, value: int) -> bytes:
    return wire.to_bytes(4, "big") + value.to_bytes(4, "big")


def leaves(circuit: Circuit, partition: P.Partition, values: Sequence[int]) -> list[bytes]:
    """The leaves of an execution's root, from every wire's value."""
    return [leaf(w, values[w]) for w in wires(circuit, partition)]


class ProofSystem(Protocol):
    scheme: CommitmentScheme  # the roots it proves against

    def prove(self, unit: Unit, values: Sequence[int]) -> object:
        """A proof that `unit` is right; `values` is every wire of its execution, which the prover rematerializes."""

    def verify(self, unit: Unit, proof: object) -> bool: ...

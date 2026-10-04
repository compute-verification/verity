"""Remote Verity: what the verifier establishes with no hardware of its own at the prover's site. Integrity by sampled
proofs, PoUW and the NCI policy, all over the prover's registration; and compute accounting against the prover's
declared capacity, an assumption onsite Verity replaces with a measurement.

    verifier    integrity, PoUW and NCI read the registration; compute reads integrity's profile and PoUW's bound
    prover      registers its work; proves it, salts it, and declares its capacity
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit
from verity.primitives.events import Actor, Send, tell
from verity.primitives.registration import register
from verity.protocols.compute import Compute
from verity.protocols.integrity.proofs.construction import Proofs
from verity.protocols.nci import Nci
from verity.protocols.pouw import Pouw


@dataclass(frozen=True)
class Remote:
    integrity: Proofs
    pouw: Pouw
    nci: Nci
    compute: ClassVar[Compute] = Compute(capacity_from="prover")

    def verifier(self) -> list[Actor]:
        return [p.verifier() for p in (self.integrity, self.pouw, self.nci, self.compute)]

    def prover(
        self, circuit: Circuit, partition: P.Partition, inputs: Sequence[tuple[int, ...]], capacity: int
    ) -> list[Actor]:
        registration = register(circuit, partition, inputs, self.integrity.scheme)
        return [
            tell(Send(registration, "verifier")),
            self.integrity.prover(inputs, registration),
            self.pouw.prover(registration),
            self.compute.capacity(capacity),
        ]

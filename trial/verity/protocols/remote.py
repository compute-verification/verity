"""Remote Verity: what the verifier establishes with no hardware of its own at the prover's site. Integrity by sampled
proofs; PoUW over the registration integrity accepted; compute accounting against the prover's declared capacity, an
assumption onsite Verity replaces with a measurement; and the NCI policy over the same registration.

    verifier    integrity -> pouw, nci, compute;  pouw -> compute
    prover      registers its work once; proves it, salts it, and declares its capacity
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit
from verity.primitives.events import Actor
from verity.protocols.compute import Compute
from verity.protocols.integrity.proofs.construction import Proofs
from verity.protocols.integrity.registration import register
from verity.protocols.nci import Nci
from verity.protocols.pouw import Pouw


@dataclass(frozen=True)
class Remote:
    integrity: Proofs
    pouw: Pouw
    nci: Nci
    compute: Compute = Compute()

    def verifier(self) -> dict[str, Actor]:
        pouw, nci, compute = (f"verifier/{p.name}" for p in (self.pouw, self.nci, self.compute))
        return {
            **self.integrity.verifier(to=(pouw, nci, compute)),
            **self.pouw.verifier(to=(compute,)),
            **self.nci.verifier(),
            **self.compute.verifier(capacity_from="prover"),
        }

    def prover(
        self, circuit: Circuit, partition: P.Partition, inputs: Sequence[tuple[int, ...]], capacity: int
    ) -> dict[str, Actor]:
        registration = register(circuit, partition, inputs, self.integrity.scheme)
        return {
            **self.integrity.prover(circuit, inputs, registration),
            **self.pouw.prover(circuit, inputs, registration),
            **self.compute.prover(capacity),
        }

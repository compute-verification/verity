"""Remote Verity: what the verifier establishes with no hardware of its own at the prover's site. The prover registers
its builder, takes PoUW's salt, and streams its executions; integrity by sampled proofs, PoUW and the NCI policy are all
over that stream, and compute accounting is against the prover's declared capacity, an assumption onsite Verity
replaces with a measurement.

    verifier    integrity, PoUW and NCI read the registration and the stream; compute reads integrity's profile and
                PoUW's bound
    prover      registers its builder, streams its executions once it has the salt, proves them, finds their shares,
                and declares its capacity
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from verity.primitives.events import Actor
from verity.primitives.registration import Builder, Registration, execute, serve
from verity.protocols.compute import Compute
from verity.protocols.integrity.proofs.construction import Proofs
from verity.protocols.nci import Nci
from verity.protocols.pouw import Pouw, Salt


@dataclass(frozen=True)
class Remote:
    integrity: Proofs
    pouw: Pouw
    nci: Nci
    compute: ClassVar[Compute] = Compute(capacity_from="prover")

    def verifier(self) -> list[Actor]:
        return [p.verifier() for p in (self.integrity, self.pouw, self.nci, self.compute)]

    def prover(self, builder: Builder, steps: Sequence[tuple[str, tuple[int, ...]]], capacity: int) -> list[Actor]:
        """`steps` are the executions as the prover runs them: each one's advice and inputs."""
        registration = Registration(builder)
        executions = [execute(builder, advice, inputs, self.integrity.scheme) for advice, inputs in steps]
        inputs = [inputs for _, inputs in steps]
        return [
            serve(registration, executions, after=Salt),
            self.integrity.prover(registration, executions, inputs),
            self.pouw.prover(registration, executions),
            self.compute.capacity(capacity),
        ]

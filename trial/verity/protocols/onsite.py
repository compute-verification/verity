"""Onsite Verity: remote Verity's protocols, with the verifier's hardware at the prover's site. A replayer there runs
drawn executions again without their inputs leaving the site; a pinger measures capacity instead of taking the
prover's word, and times PoUS's audit; a warden on the link sees the traffic.

    verifier    integrity, PoUW and NCI read the registration; compute reads integrity's profile and PoUW's bound;
                memory reads PoUS's bound; network reads the traffic
    prover      registers its work; opens it for the replayer, salts it, and holds PoUS's encoding
    replayer    integrity's
    pinger      measures compute and memory capacity, and times PoUS's challenges
    warden      reads the link
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
from verity.protocols.integrity.replay import Replay
from verity.protocols.memory import Memory
from verity.protocols.nci import Nci
from verity.protocols.network import Network
from verity.protocols.pous import Pous
from verity.protocols.pouw import Pouw
from verity.protocols.transparency import Transparency


@dataclass(frozen=True)
class Onsite:
    integrity: Replay
    pouw: Pouw
    nci: Nci
    pous: Pous
    transparency: Transparency
    network: Network
    compute: ClassVar[Compute] = Compute(capacity_from="pinger")
    memory: ClassVar[Memory] = Memory(capacity_from="pinger")

    def verifier(self) -> list[Actor]:
        protocols = (
            self.integrity,
            self.pouw,
            self.nci,
            self.compute,
            self.pous,
            self.memory,
            self.transparency,
            self.network,
        )
        return [p.verifier() for p in protocols]

    def prover(
        self, circuit: Circuit, partition: P.Partition, inputs: Sequence[tuple[int, ...]], encoding: Sequence[bytes]
    ) -> list[Actor]:
        registration = register(circuit, partition, inputs, self.integrity.scheme)
        return [
            tell(Send(registration, "verifier")),
            self.integrity.prover(inputs, registration),
            self.pouw.prover(registration),
            self.pous.prover(encoding),
        ]

    def replayer(self) -> list[Actor]:
        return [self.integrity.replayer()]

    def pinger(self, compute: int, memory: int, elapsed: float) -> list[Actor]:
        return [self.compute.capacity(compute), self.memory.capacity(memory), self.pous.pinger(elapsed)]

    def warden(self, bytes_out: int, timing_bits: int) -> list[Actor]:
        return [self.transparency.warden(bytes_out, timing_bits)]

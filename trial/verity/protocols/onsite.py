"""Onsite Verity: remote Verity's protocols, with the verifier's hardware at the prover's site. A replayer there can run
executions again without their inputs leaving the site; pingers measure capacity instead of taking the prover's word,
and time PoUS's audit; a warden on the link sees the traffic.

    verifier    integrity -> pouw, nci, compute;  pouw -> compute;  pous -> memory;  transparency -> network
    prover      registers its work once; answers integrity and PoUW as remotely, and holds PoUS's encoding
    replayer    integrity's, when it replays
    pinger      measures compute and memory capacity, and times PoUS's challenges
    warden      reads the link
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
from verity.protocols.integrity.replay import Replay
from verity.protocols.memory import Memory
from verity.protocols.nci import Nci
from verity.protocols.network import Network
from verity.protocols.pous import Pous
from verity.protocols.pouw import Pouw
from verity.protocols.transparency import Transparency


@dataclass(frozen=True)
class Onsite:
    integrity: Replay | Proofs
    pouw: Pouw
    nci: Nci
    pous: Pous
    transparency: Transparency
    network: Network
    compute: Compute = Compute()
    memory: Memory = Memory()

    def verifier(self) -> dict[str, Actor]:
        pouw, nci, compute, memory, network = (
            f"verifier/{p.name}" for p in (self.pouw, self.nci, self.compute, self.memory, self.network)
        )
        return {
            **self.integrity.verifier(to=(pouw, nci, compute)),
            **self.pouw.verifier(to=(compute,)),
            **self.nci.verifier(),
            **self.compute.verifier(capacity_from="pinger"),
            **self.pous.verifier(to=(memory,)),
            **self.memory.verifier(capacity_from="pinger"),
            **self.transparency.verifier(to=(network,)),
            **self.network.verifier(),
        }

    def prover(
        self, circuit: Circuit, partition: P.Partition, inputs: Sequence[tuple[int, ...]], encoding: Sequence[bytes]
    ) -> dict[str, Actor]:
        registration = register(circuit, partition, inputs, self.integrity.scheme)
        return {
            **self.integrity.prover(circuit, inputs, registration),
            **self.pouw.prover(circuit, inputs, registration),
            **self.pous.prover(encoding),
        }

    def replayer(self) -> dict[str, Actor]:
        return self.integrity.replayer() if isinstance(self.integrity, Replay) else {}

    def pinger(self, compute: int, memory: int, elapsed: float) -> dict[str, Actor]:
        return {**self.compute.pinger(compute), **self.memory.pinger(memory), **self.pous.pinger(elapsed)}

    def warden(self, bytes_out: int, timing_bits: int) -> dict[str, Actor]:
        return self.transparency.warden(bytes_out, timing_bits)

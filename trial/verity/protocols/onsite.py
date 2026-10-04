"""Onsite Verity: remote Verity's protocols, with the verifier's hardware at the prover's site. A replayer there runs
drawn executions again without their inputs leaving the site; a pinger measures capacity instead of taking the
prover's word, and times PoUS's audit; a warden on the link sees the traffic.

    verifier    integrity, PoUW and NCI read the registration and the stream; compute reads integrity's profile and
                PoUW's bound; memory reads PoUS's bound; network reads the traffic
    prover      registers its builder and streams its executions as remotely; opens them for the replayer, finds their
                shares, and holds PoUS's encoding
    replayer    integrity's
    pinger      measures compute and memory capacity, and times PoUS's challenges
    warden      reads the link
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from verity.primitives.events import Actor
from verity.primitives.registration import Builder, Registration, execute, serve
from verity.protocols.compute import Compute
from verity.protocols.integrity.replay import Replay
from verity.protocols.memory import Memory
from verity.protocols.nci import Nci
from verity.protocols.network import Network
from verity.protocols.pous import Pous
from verity.protocols.pouw import Pouw, Salt
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
        self, builder: Builder, steps: Sequence[tuple[str, tuple[int, ...]]], encoding: Sequence[bytes]
    ) -> list[Actor]:
        """`steps` are the executions as the prover runs them: each one's advice and inputs."""
        registration = Registration(builder)
        executions = [execute(builder, advice, inputs, self.integrity.scheme) for advice, inputs in steps]
        inputs = [inputs for _, inputs in steps]
        return [
            serve(registration, executions, after=Salt),
            self.integrity.prover(registration, executions, inputs),
            self.pouw.prover(registration, executions),
            self.pous.prover(encoding),
        ]

    def replayer(self) -> list[Actor]:
        return [self.integrity.replayer()]

    def pinger(self, compute: int, memory: int, elapsed: float) -> list[Actor]:
        return [self.compute.capacity(compute), self.memory.capacity(memory), self.pous.pinger(elapsed)]

    def warden(self, bytes_out: int, timing_bits: int) -> list[Actor]:
        return [self.transparency.warden(bytes_out, timing_bits)]

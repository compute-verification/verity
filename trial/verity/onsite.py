"""Onsite Verity: remote Verity's accounting, with the verifier's hardware at the site in place of the prover's word.
A replayer runs drawn executions again without their inputs leaving the site; a pinger counts the fleet, times the
work, and challenges memory where a deadline can be tight; a warden on the links sees what leaves. So onsite also
bounds memory and network, and its inventory is a count rather than an assumption.

    verifier    integrity, work and NCI read the registration and the stream; compute coverage reads integrity's
                profile, the work bound and the counted inventory; memory coverage the space bound and the inventory;
                network coverage integrity's profile and the warden's report
    prover      registers its builder and streams its executions as remotely; opens them for the replayer, finds their
                shares, and answers space's challenges from the encoding it holds
    replayer    integrity's
    pinger      counts the fleet, times the work, and challenges and times space
    warden      reads the links
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.accounting.coverage.compute import Compute
from verity.accounting.coverage.inventory import inventory
from verity.accounting.coverage.memory import Memory
from verity.accounting.coverage.network import Network
from verity.accounting.integrity.replay import Replay
from verity.accounting.utilization.space import Space
from verity.accounting.utilization.work import Salt, Work
from verity.compliance.nci import Nci
from verity.primitives.events import Actor
from verity.primitives.registration import Execution, Registration, serve


@dataclass(frozen=True)
class Onsite:
    integrity: Replay
    work: Work
    space: Space
    compute: Compute
    memory: Memory
    network: Network
    nci: Nci

    def verifier(self) -> list[Actor]:
        return [
            self.integrity.verifier(),
            self.work.verifier(clock_from="pinger"),
            self.space.verifier(),
            self.compute.verifier(inventory_from="pinger"),
            self.memory.verifier(inventory_from="pinger"),
            self.network.verifier(),
            self.nci.verifier(),
        ]

    def prover(
        self,
        registration: Registration,
        executions: Sequence[Execution],
        inputs: Sequence[tuple[int, ...]],
        encoding: Sequence[bytes],
    ) -> list[Actor]:
        return [
            serve(registration, executions, after=Salt),
            self.integrity.prover(registration, executions, inputs),
            self.work.prover(registration, executions),
            self.space.prover(encoding),
        ]

    def replayer(self) -> list[Actor]:
        return [self.integrity.replayer()]

    def pinger(self, fleet: Mapping[str, int], work_seconds: float, space_seconds: float) -> list[Actor]:
        """Its readings: the devices it counted, and its clock's for the work's window and space's answer."""
        return [inventory(fleet), self.work.clock(work_seconds), self.space.pinger(space_seconds)]

    def warden(self, bytes_out: int, timing_bits: int) -> list[Actor]:
        return [self.network.warden(bytes_out, timing_bits)]

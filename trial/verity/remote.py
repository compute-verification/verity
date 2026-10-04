"""Remote Verity: what the verifier establishes with no hardware of its own at the site. The prover registers its
builder, takes work's salt, streams its executions and declares its fleet; integrity by sampled proofs, work and
compute coverage account for the epoch, and NCI judges it. Every reading is the verifier's own (its clock) or the
prover's (its inventory), so the account holds only if the prover declared every device: remote shows that the
declared fleet was busy with the declared work, never that there was no other.

    verifier    integrity, work and NCI read the registration and the stream; compute coverage reads integrity's
                profile, the work bound and the declared inventory; the verifier's clock times the work
    prover      registers its builder, streams its executions once it has the salt, proves them, finds their shares,
                and declares its fleet
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.accounting.coverage.compute import Compute
from verity.accounting.coverage.inventory import inventory
from verity.accounting.integrity.proofs.construction import Proofs
from verity.accounting.utilization.work import Salt, Window, Work
from verity.compliance.nci import Nci
from verity.primitives.events import Actor
from verity.primitives.registration import Stream, serve


@dataclass(frozen=True)
class Remote:
    integrity: Proofs
    work: Work
    compute: Compute
    nci: Nci

    def verifier(self, window: Window) -> list[Actor]:
        """`window`: the verifier's clock's readings when it sent the salt and when the shares came in."""
        return [
            self.integrity.verifier(),
            self.work.verifier(clock_from="verifier"),
            self.work.clock(window),
            self.compute.verifier(inventory_from="prover"),
            self.nci.verifier(),
        ]

    def prover(self, stream: Stream, transcripts: Sequence[Sequence[int]], fleet: Mapping[str, int]) -> list[Actor]:
        return [
            serve(stream, after=Salt),
            self.integrity.prover(stream, transcripts),
            self.work.prover(stream),
            inventory(fleet),
        ]

"""Remote Verity: what the verifier establishes with no hardware of its own at the site. The prover registers its
builder, takes work's salt, streams its executions and surveys its site; integrity by sampled proofs and work explain
the epoch's compute, coverage accounts for it, and NCI judges it. Every reading is the verifier's own (its clock) or the
prover's (its site), so the account holds only if the prover reported every device: remote shows that the declared
site was busy with the declared work, never that there was no other. Nothing at the site observes anything.

    verifier    integrity, work and NCI read the registration and the stream; coverage reads integrity's profile, the
                work bound and the prover's site; the verifier's clock times the work
    prover      registers its builder, streams its executions once it has the salt, proves them, finds their shares,
                and surveys its site
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.accounting.capacity import survey
from verity.accounting.coverage import Coverage
from verity.accounting.explained.integrity.proofs.construction import Proofs
from verity.accounting.explained.work import Salt, Window, Work
from verity.compliance.nci import Nci
from verity.primitives.events import Actor
from verity.primitives.registration import Stream, serve


@dataclass(frozen=True)
class Remote:
    integrity: Proofs
    work: Work
    coverage: Coverage
    nci: Nci

    def verifier(self, window: Window) -> list[Actor]:
        """`window`: the verifier's clock's readings when it sent the salt and when the shares came in."""
        return [
            self.integrity.verifier(),
            self.work.verifier(clock_from="verifier"),
            self.work.clock(window),
            self.coverage.verifier(site_from="prover"),
            self.nci.verifier(),
        ]

    def prover(self, stream: Stream, transcripts: Sequence[Sequence[int]], site: Mapping[str, int]) -> list[Actor]:
        return [
            serve(stream, after=Salt),
            self.integrity.prover(stream, transcripts),
            self.work.prover(stream),
            survey(site),
        ]

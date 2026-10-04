"""Remote Verity: what the verifier establishes with no hardware of its own at the site. The prover registers its
builder, takes work's salt, streams its executions and reports its site; integrity by sampled proofs says what ran,
work bounds the tries made, reconciling sets them against the site, and NCI judges it. Every reading is the verifier's
own (its clock) or the prover's (its site), so the balance holds only if the prover reported every device: remote shows
that the declared site was busy with the declared work, never that there was no other. Nothing at the site observes
anything.

    verifier    integrity and its proofs, work, reconciling and NCI; its clock's readings time the work
    prover      registers its builder, streams its executions once it has the salt, proves the drawn regions, finds
                their shares, and reports its site
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, tell
from verity.primitives.physical import Site, Window
from verity.protocols.accounting.integrity import Integrity, Stream, serve
from verity.protocols.accounting.reconcile import Reconcile
from verity.protocols.accounting.work.pouw import Pouw, Salt
from verity.protocols.compliance.nci import Nci


@dataclass(frozen=True)
class Remote:
    integrity: Integrity  # by proofs
    work: Pouw
    reconcile: Reconcile
    nci: Nci

    def verifier(self, window: Window) -> list[Actor]:
        """`window`: the verifier's clock's readings when it sent the salt and when the shares came in."""
        return [
            self.integrity.verifier(),
            self.integrity.verification.verifier(),
            self.work.verifier(),
            tell(Send(window)),
            self.reconcile.verifier(clock_from="verifier", site_from="prover"),
            self.nci.verifier(),
        ]

    def prover(self, stream: Stream, transcripts: Sequence[Sequence[int]], site: Mapping[str, int]) -> list[Actor]:
        return [
            serve(stream, after=Salt),
            self.integrity.prover(stream, transcripts),
            self.work.prover(stream),
            tell(Send(Site(site), "verifier")),
        ]

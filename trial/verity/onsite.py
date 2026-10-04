"""Onsite Verity: remote Verity's accounting, with the verifier's hardware at the site in place of the prover's word.
A replayer runs drawn executions again without their inputs leaving the site; a pinger surveys the site, times the
work, and challenges memory where a deadline can be tight; a warden on the links observes what leaves. So onsite also
accounts for memory and network, and its site is a count rather than an assumption.

    verifier    integrity, work and NCI read the registration and the stream; space and the warden's audit read their
                readings; coverage reads all of them and the pinger's site
    prover      registers its builder and streams its executions as remotely; opens them for the replayer, finds their
                shares, and answers space's challenges from the encoding it holds
    replayer    integrity's
    pinger      surveys the site, times the work, and challenges and times space
    warden      observes the links
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.accounting.capacity import survey
from verity.accounting.coverage import Coverage
from verity.accounting.explained.integrity.replay import Replay
from verity.accounting.explained.space import Space
from verity.accounting.explained.work import Salt, Window, Work
from verity.accounting.observed import Network
from verity.compliance.nci import Nci
from verity.primitives.events import Actor
from verity.primitives.registration import Stream, serve


@dataclass(frozen=True)
class Onsite:
    integrity: Replay
    work: Work
    space: Space
    network: Network
    coverage: Coverage
    nci: Nci

    def verifier(self) -> list[Actor]:
        return [
            self.integrity.verifier(),
            self.work.verifier(clock_from="pinger"),
            self.space.verifier(),
            self.network.verifier(),
            self.coverage.verifier(site_from="pinger"),
            self.nci.verifier(),
        ]

    def prover(self, stream: Stream, transcripts: Sequence[Sequence[int]], encoding: Sequence[bytes]) -> list[Actor]:
        return [
            serve(stream, after=Salt),
            self.integrity.prover(stream, transcripts),
            self.work.prover(stream),
            self.space.prover(encoding),
        ]

    def replayer(self) -> list[Actor]:
        return [self.integrity.replayer()]

    def pinger(self, site: Mapping[str, int], work_window: Window, space_seconds: float) -> list[Actor]:
        """Its readings: the most devices it counted at any time in the work's window, and its clock's for that window
        and for space's answer."""
        return [survey(site), self.work.clock(work_window), self.space.pinger(space_seconds)]

    def warden(self, bytes_out: int, timing_bits: int) -> list[Actor]:
        return [self.network.warden(bytes_out, timing_bits)]

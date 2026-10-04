"""Onsite Verity: remote Verity's accounting, with the verifier's hardware at the site in place of the prover's word.
A replayer runs drawn executions again without their values leaving the site; a pinger counts the site, times the
work, and challenges memory where a deadline can be tight; a warden on the links observes what leaves. So onsite also
accounts for space and communication, and its site is a count rather than an assumption.

    verifier    integrity and its replay, work, space, communication, the account and NCI
    prover      registers its builder and streams its executions as remotely; opens the drawn ones for the replayer,
                finds their shares, and answers space's challenges from the encoding it holds
    replayer    replay's
    pinger      counts the site, times the work, and challenges and times space
    warden      observes the links
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.primitives.events import Actor, Send, tell
from verity.primitives.physical import Site, Window
from verity.protocols.accounting.account import Accounting
from verity.protocols.accounting.communication import Communication
from verity.protocols.accounting.integrity import Integrity, Stream, serve
from verity.protocols.accounting.space import Space
from verity.protocols.accounting.work import Salt, Work
from verity.protocols.compliance.nci import Nci


@dataclass(frozen=True)
class Onsite:
    integrity: Integrity  # by replay
    work: Work
    space: Space
    communication: Communication
    accounting: Accounting
    nci: Nci

    def verifier(self) -> list[Actor]:
        return [
            self.integrity.verifier(),
            self.integrity.verification.verifier(),
            self.work.verifier(clock_from="pinger", site_from="pinger"),
            self.space.verifier(site_from="pinger"),
            self.communication.verifier(site_from="pinger"),
            self.accounting.verifier(),
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
        return [self.integrity.verification.replayer()]

    def pinger(self, site: Mapping[str, int], work_window: Window, space_window: Window) -> list[Actor]:
        """Its readings: the most devices it counted at any time in the work's window, and its clock's for that window
        and for space's challenge."""
        return [tell(Send(Site(site), "verifier"), Send(work_window, "verifier")), self.space.pinger(space_window)]

    def warden(self, bytes_out: int, timing_bits: int, window: Window) -> list[Actor]:
        return [self.communication.warden(bytes_out, timing_bits, window)]

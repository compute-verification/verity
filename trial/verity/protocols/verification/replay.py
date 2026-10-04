"""Replay: trusted hardware where the values are runs claimed computations again.

    prover      sends the replayer a batch of claims and their openings
    replayer    checks each claim from its opening, and tells the verifier whether each holds; it is trusted and sealed:
                it sees the values, and the verifier and the auditor do not
    verifier    records what the replayer tells it

Replay needs the computation to be reproducible bit for bit on the replayer, which the circuit's exact semantics give.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.primitives.commitments import CommitmentScheme, MerkleSha256
from verity.primitives.events import Actor, Send, receive, tell
from verity.protocols.verification.claim import Claim, Opening, Verified


@dataclass(frozen=True)
class Opened:
    """Prover to replayer: a batch of claims, and their openings."""

    claims: tuple[Claim, ...]
    openings: tuple[Opening, ...]


@dataclass(frozen=True)
class Replay:
    scheme: CommitmentScheme = MerkleSha256()

    def verifier(self) -> Actor:
        recorded = ()
        while True:
            verified = yield from receive(Verified, "replayer", *recorded)
            recorded = (Send(verified),)

    def replayer(self) -> Actor:
        replayed = ()
        while True:
            opened = yield from receive(Opened, "prover", *replayed)
            pairs = list(zip(opened.claims, opened.openings))
            verified = Verified(tuple(c for c, _ in pairs), tuple(c.check(self.scheme, o) for c, o in pairs))
            replayed = (Send(verified, "verifier"),)

    def prover(self, claims: tuple[Claim, ...], openings: Sequence[Opening]) -> Actor:
        return tell(Send(Opened(claims, tuple(openings)), "replayer"))

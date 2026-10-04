"""Proofs: the prover proves claims, and the verifier checks each proof with a proof system.

    prover      sends a proof of a batch of claims, from their openings (the witness)
    verifier    checks each proof it receives against the claims alone, and records which it verified

Everything happens in the verifier's view, so the proof system must reveal no more than the auditor may see: a
zero-knowledge one reveals nothing, a transparent one the claims' boundary values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from verity.primitives.commitments import CommitmentScheme, MerkleSha256
from verity.primitives.events import Actor, Send, receive, tell
from verity.protocols.verification.claim import Claim, Opening, Verified


class ProofSystem(Protocol):
    def prove(self, claims: Sequence[Claim], openings: Sequence[Opening]) -> object:
        """One proof of every claim in the batch."""

    def verify(self, scheme: CommitmentScheme, claims: Sequence[Claim], proof: object) -> bool: ...


@dataclass(frozen=True)
class Proved:
    """Prover to verifier: one proof of `claims`."""

    claims: tuple[Claim, ...]
    proof: object


@dataclass(frozen=True)
class Proofs:
    system: ProofSystem
    scheme: CommitmentScheme = MerkleSha256()

    def verifier(self) -> Actor:
        verified = ()
        while True:
            proved = yield from receive(Proved, "prover", *verified)
            holds = self.system.verify(self.scheme, proved.claims, proved.proof)
            verified = (Send(Verified(proved.claims, (holds,) * len(proved.claims))),)

    def prover(self, claims: tuple[Claim, ...], openings: Sequence[Opening]) -> Actor:
        return tell(Send(Proved(claims, self.system.prove(claims, openings)), "verifier"))

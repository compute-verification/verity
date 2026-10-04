"""The transparent proof system: the witness is the proof. The verifier checks each claim from its opening. Light, and
reveals the claims' boundary values."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.primitives.commitments import CommitmentScheme
from verity.protocols.verification.claim import Claim, Opening
from verity.protocols.verification.proofs.protocol import ProofSystem


@dataclass(frozen=True)
class Transparent(ProofSystem):
    def prove(self, claims: Sequence[Claim], openings: Sequence[Opening]) -> tuple[Opening, ...]:
        return tuple(openings)

    def verify(self, scheme: CommitmentScheme, claims: Sequence[Claim], proof: object) -> bool:
        return (
            isinstance(proof, tuple)
            and len(proof) == len(claims)
            and all(isinstance(o, Opening) and c.check(scheme, o) for c, o in zip(claims, proof))
        )

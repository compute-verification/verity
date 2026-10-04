"""SP1, a proof system. A stand-in for a zero-knowledge proof of a batch of claims: their openings, serialized by the
heavy dependency (`tomli_w`, standing in for the SP1 toolchain), which only the `verity-sp1` distribution beside this
file installs; unlike a real proof, it reveals the claims' boundary values."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass

import tomli_w  # the heavy dependency

from verity.primitives.commitments import CommitmentScheme
from verity.protocols.verification.claim import Claim, Opening
from verity.protocols.verification.proofs.protocol import ProofSystem


@dataclass(frozen=True)
class Sp1(ProofSystem):
    def prove(self, claims: Sequence[Claim], openings: Sequence[Opening]) -> str:
        return tomli_w.dumps(
            {
                "claims": [
                    {str(w): {"value": v, "path": [p.hex() for p in o.paths[w]]} for w, v in o.values.items()}
                    for o in openings
                ]
            }
        )

    def verify(self, scheme: CommitmentScheme, claims: Sequence[Claim], proof: object) -> bool:
        if not isinstance(proof, str):
            return False
        openings = [
            Opening(
                {int(w): x["value"] for w, x in o.items()},
                {int(w): [bytes.fromhex(p) for p in x["path"]] for w, x in o.items()},
            )
            for o in tomllib.loads(proof)["claims"]
        ]
        return len(openings) == len(claims) and all(c.check(scheme, o) for c, o in zip(claims, openings))

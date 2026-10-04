"""SP1, a proof system. A stand-in for a zero-knowledge proof of a batch against its roots: the transparent openings,
serialized by the heavy dependency (`tomli_w`, standing in for the SP1 toolchain), which only the `verity-sp1`
distribution beside this file installs; unlike a real proof, it reveals the units' values."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass

import tomli_w  # the heavy dependency

from verity.accounting.integrity.proofs import transparent
from verity.accounting.integrity.proofs.system import ProofSystem, Unit
from verity.primitives.commitments import CommitmentScheme
from verity.primitives.registration import Opening


@dataclass(frozen=True)
class Sp1(ProofSystem):
    def prove(self, scheme: CommitmentScheme, units: Sequence[Unit], transcripts: Sequence[Sequence[int]]) -> str:
        openings = transparent.Transparent().prove(scheme, units, transcripts)
        return tomli_w.dumps(
            {
                "units": [
                    {
                        f"{i}:{w}": {"value": v, "path": [p.hex() for p in o.paths[i, w]]}
                        for (i, w), v in o.values.items()
                    }
                    for o in openings
                ]
            }
        )

    def verify(self, scheme: CommitmentScheme, units: Sequence[Unit], proof: object) -> bool:
        if not isinstance(proof, str):
            return False
        parsed = [{tuple(map(int, k.split(":"))): x for k, x in o.items()} for o in tomllib.loads(proof)["units"]]
        openings = [
            Opening(
                {k: x["value"] for k, x in o.items()}, {k: [bytes.fromhex(p) for p in x["path"]] for k, x in o.items()}
            )
            for o in parsed
        ]
        return len(openings) == len(units) and all(transparent.check(scheme, u, o) for u, o in zip(units, openings))

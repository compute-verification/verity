"""SP1, a proof system. A stand-in for a zero-knowledge proof of a batch against its roots: the transparent openings,
serialized by the heavy dependency (`tomli_w`, standing in for the SP1 toolchain), which only the `verity-sp1`
distribution beside this file installs; unlike a real proof, it reveals the units' values."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass

import tomli_w  # the heavy dependency

from verity.primitives.commitments import CommitmentScheme
from verity.subprotocols.integrity.proofs import transparent
from verity.subprotocols.integrity.proofs.system import ProofSystem, Unit
from verity.subprotocols.integrity.registration import Opening


@dataclass(frozen=True)
class Sp1(ProofSystem):
    def prove(self, scheme: CommitmentScheme, units: Sequence[Unit], values: Sequence[Sequence[int]]) -> str:
        openings = transparent.Transparent().prove(scheme, units, values)
        return tomli_w.dumps({"units": [{str(w): {"value": v, "path": [p.hex() for p in o.paths[w]]} for w, v in o.values.items()} for o in openings]})

    def verify(self, scheme: CommitmentScheme, units: Sequence[Unit], proof: object) -> bool:
        if not isinstance(proof, str):
            return False
        parsed = tomllib.loads(proof)["units"]
        openings = [Opening({int(w): x["value"] for w, x in o.items()}, {int(w): [bytes.fromhex(p) for p in x["path"]] for w, x in o.items()}) for o in parsed]
        return len(openings) == len(units) and all(transparent.check(scheme, u, o) for u, o in zip(units, openings))

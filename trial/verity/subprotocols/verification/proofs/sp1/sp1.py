"""SP1: a stand-in for a zero-knowledge proof of a unit against its root. The stand-in is a replay's openings,
serialized by the heavy dependency (`tomli_w`, standing in for the SP1 toolchain), which only the `verity-sp1`
distribution beside this file installs; unlike a real proof, it reveals the unit's values."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence

import tomli_w  # the heavy dependency

from verity.primitives.commitments import CommitmentScheme
from verity.subprotocols.verification import units as U
from verity.subprotocols.verification.replay import Opened, Replay


class Sp1(U.ProofSystem):
    def __init__(self, scheme: CommitmentScheme):
        self.scheme = scheme
        self._replay = Replay(scheme)

    def prove(self, unit: U.Unit, values: Sequence[int]) -> str:
        opened = self._replay.prove(unit, values)
        return tomli_w.dumps({str(w): {"value": v, "path": [p.hex() for p in opened.paths[w]]} for w, v in opened.values.items()})

    def verify(self, unit: U.Unit, proof: object) -> bool:
        if not isinstance(proof, str):
            return False
        parsed = tomllib.loads(proof)
        opened = Opened({int(w): x["value"] for w, x in parsed.items()}, {int(w): [bytes.fromhex(p) for p in x["path"]] for w, x in parsed.items()})
        return self._replay.verify(unit, opened)

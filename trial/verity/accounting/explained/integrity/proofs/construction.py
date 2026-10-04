"""Proofs: sample regions of the streamed executions, and have the prover prove each drawn one.

    verifier    takes the registration and the stream, rebuilds each execution's circuit and partition from its advice,
                checks every partition against its width, draws regions from its secret once the stream closes, each
                as likely as its gates and output bits ask (`sampling.Weighted`), and checks each proof it receives
                with the proof system, until every drawn region is decided
    prover      proves the drawn regions from the executions' transcripts, batched as it likes

Everything happens in the verifier's view, so the proof system must reveal no more than the auditor may see.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from dataclasses import dataclass

from verity.accounting.explained.integrity.profile import profile
from verity.accounting.explained.integrity.proofs.system import ProofSystem, Unit
from verity.accounting.explained.integrity.sampling import Draw, Weighted
from verity.primitives import partition as P
from verity.primitives.commitments import CommitmentScheme, MerkleSha256
from verity.primitives.events import Actor, Send, collect, receive
from verity.primitives.randomness import derive
from verity.primitives.registration import Closed, Execution, Registration, Stream, check


@dataclass(frozen=True)
class Proved:
    """Prover to verifier: one proof of the drawn regions in `units`."""

    units: tuple[int, ...]
    proof: object


@dataclass(frozen=True)
class Proofs:
    law: Weighted
    max_width: int  # boundary wires per region
    delta: float
    system: ProofSystem
    scheme: CommitmentScheme = MerkleSha256()

    def verifier(self) -> Actor:
        secret = secrets.token_bytes(32)
        registration = yield from receive(Registration, "prover")
        stream = Stream(registration, (yield from collect(Execution, Closed, "prover")))
        refused = check(stream, self.scheme) or tuple(
            f"execution {i}: {why}"
            for i in range(len(stream.executions))
            for why in P.validate(*stream.built(i), self.max_width)
        )
        if refused:
            return (Send(profile(stream.digest(), "region", [], [], self.law, self.delta, refused)),)

        units = _units(stream)
        gates, outputs = [len(u.gates) for u in units], [len(u.outputs) for u in units]
        coins = derive(secret, "verity/integrity/proofs/v2", stream.digest())
        drawn = self.law.draw(coins, gates, outputs, self.delta)
        results: dict[int, bool] = {}
        proved = yield from receive(Proved, "prover", Send(Draw(drawn), "prover"))
        while True:
            ok = set(proved.units) <= set(drawn) and self.system.verify(
                self.scheme, [units[u] for u in proved.units], proved.proof
            )
            results.update((u, ok) for u in proved.units)
            if set(drawn) <= results.keys():
                break
            proved = yield from receive(Proved, "prover")

        reasons = tuple(f"region {u}: does not verify" for u in drawn if not results[u])
        return (Send(profile(stream.digest(), "region", gates, outputs, self.law, self.delta, reasons)),)

    def prover(self, stream: Stream, transcripts: Sequence[Sequence[int]]) -> Actor:
        draw = yield from receive(Draw, "verifier")
        units = _units(stream)
        proof = self.system.prove(self.scheme, [units[u] for u in draw.units], transcripts)
        return (Send(Proved(draw.units, proof), "verifier"),)


def _units(stream: Stream) -> list[Unit]:
    """Every region of every execution: execution 0's regions, then execution 1's, and so on."""
    return [Unit(stream, i, region) for i in range(len(stream.executions)) for region in P.regions(stream.built(i)[1])]

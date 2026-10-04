"""Proofs: sample regions of the streamed executions, and have the prover prove each drawn one.

    verifier    takes the registration and the stream, rebuilds each execution's circuit and partition from its advice,
                checks every partition against its width, draws regions from its secret once the stream closes, and
                checks each proof it receives with the proof system, until every drawn region is decided
    prover      rematerializes each drawn region's execution by replay and proves the drawn regions, batched as it likes

Everything happens in the verifier's view, so the proof system must reveal no more than the auditor may see.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from dataclasses import dataclass

from verity.accounting.integrity.profile import profile
from verity.accounting.integrity.proofs.system import ProofSystem, Unit
from verity.accounting.integrity.sampling import Draw, Subset
from verity.primitives import partition as P
from verity.primitives.circuit import transcript
from verity.primitives.commitments import CommitmentScheme, MerkleSha256
from verity.primitives.events import Actor, Send, collect, receive
from verity.primitives.randomness import derive
from verity.primitives.registration import (
    Closed,
    Execution,
    Registration,
    built,
    check,
    digest,
)


@dataclass(frozen=True)
class Proved:
    """Prover to verifier: one proof of the drawn regions in `units`."""

    units: tuple[int, ...]
    proof: object


@dataclass(frozen=True)
class Proofs:
    k: int  # regions drawn
    max_width: int  # boundary wires per region
    delta: float
    system: ProofSystem
    scheme: CommitmentScheme = MerkleSha256()

    def verifier(self) -> Actor:
        secret = secrets.token_bytes(32)
        registration = yield from receive(Registration, "prover")
        executions = yield from collect(Execution, Closed, "prover")
        stream, refused = digest(registration, executions), check(registration, executions, self.scheme)
        units = [] if refused else [u for _, u in _units(registration, executions)]
        refused = refused or tuple(
            f"execution {i}: {why}"
            for i, e in enumerate(executions)
            for why in P.validate(*built(registration.builder, e.advice), self.max_width)
        )
        n = len(units)
        if refused:
            return (Send(profile(stream, "region", [0] * n, [0] * n, n, self.delta, refused)),)

        law = Subset(self.k)
        drawn = law.draw(derive(secret, "verity/integrity/proofs/v1", stream), n)
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
        gates, outputs = [len(u.gates) for u in units], [len(u.outputs) for u in units]
        return (Send(profile(stream, "region", gates, outputs, law.bound(n, self.delta), self.delta, reasons)),)

    def prover(
        self, registration: Registration, executions: Sequence[Execution], inputs: Sequence[tuple[int, ...]]
    ) -> Actor:
        draw = yield from receive(Draw, "verifier")
        units = _units(registration, executions)
        drawn = [units[u] for u in draw.units]
        values = [transcript(unit.circuit, inputs[i]) for i, unit in drawn]
        return (Send(Proved(draw.units, self.system.prove(self.scheme, [u for _, u in drawn], values)), "verifier"),)


def _units(registration: Registration, executions: Sequence[Execution]) -> list[tuple[int, Unit]]:
    """Every region of every execution, with its execution's index: execution 0's regions, then execution 1's, and so
    on."""
    units = []
    for i, e in enumerate(executions):
        circuit, partition = built(registration.builder, e.advice)
        units.extend((i, Unit(circuit, partition, e.root, region)) for region in P.regions(partition))
    return units

"""Proofs: sample regions of the registered executions, and have the prover prove each drawn one.

    verifier    receives the registration, checks every partition against its width, draws regions from its secret,
                and checks each proof it receives with the proof system, until every drawn region is decided
    prover      rematerializes each drawn region's execution by replay and proves the drawn regions, batched as it likes

Everything happens in the verifier's view, so the proof system must reveal no more than the auditor may see.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from dataclasses import dataclass

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, trace
from verity.primitives.commitments import CommitmentScheme
from verity.primitives.events import Actor, Send, receive
from verity.primitives.randomness import derive
from verity.protocols.integrity.profile import IntegrityProfile
from verity.protocols.integrity.proofs.system import ProofSystem, Unit
from verity.protocols.integrity.registration import Registered, Registration, check, register
from verity.protocols.integrity.sampling import Draw, Subset


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
    name: str = "integrity"

    def verifier(self, scheme: CommitmentScheme) -> dict[str, Actor]:
        return {f"verifier/{self.name}": self._verifier(scheme)}

    def prover(self, scheme: CommitmentScheme, circuit: Circuit, partition: P.Partition, inputs: Sequence[tuple[int, ...]]) -> dict[str, Actor]:
        return {f"prover/{self.name}": self._prover(scheme, circuit, partition, inputs)}

    def _verifier(self, scheme: CommitmentScheme) -> Actor:
        secret = secrets.token_bytes(32)
        registered = (yield from receive(Registered)).body
        registration, circuits = registered.registration, registered.circuits
        digest, n = registration.digest(), sum(len(e.partition) for e in registration.executions)
        refused = check(registered, scheme) or tuple(
            f"execution {i}: {why}" for i, e in enumerate(registration.executions)
            for why in P.validate(circuits[e.circuit], e.partition, self.max_width)
        )
        if refused:
            return (Send(IntegrityProfile(digest, "region", False, n, n, self.delta, refused)),)

        law = Subset(self.k)
        drawn = law.draw(derive(secret, "verity/integrity/proofs/v1", digest), n)
        results: dict[int, bool] = {}
        proved = (yield from receive(Proved, Send(Draw(drawn), to=(f"prover/{self.name}",)))).body
        while True:
            ok = set(proved.units) <= set(drawn) and self.system.verify(scheme, [_unit(registration, circuits, u) for u in proved.units], proved.proof)
            results.update((u, ok) for u in proved.units)
            if set(drawn) <= results.keys():
                break
            proved = (yield from receive(Proved)).body

        reasons = tuple(f"region {u}: does not verify" for u in drawn if not results[u])
        return (Send(IntegrityProfile(digest, "region", not reasons, n, law.bound(n, self.delta), self.delta, reasons)),)

    def _prover(self, scheme: CommitmentScheme, circuit: Circuit, partition: P.Partition, inputs: Sequence[tuple[int, ...]]) -> Actor:
        registration = register(circuit, partition, inputs, scheme)
        circuits = {circuit.id(): circuit}
        draw = (yield from receive(Draw, Send(Registered(registration, circuits), to=(f"verifier/{self.name}",)))).body
        units = [_unit(registration, circuits, u) for u in draw.units]
        values = [trace(circuit, inputs[_locate(registration, u)[0]]) for u in draw.units]
        return (Send(Proved(draw.units, self.system.prove(scheme, units, values)), to=(f"verifier/{self.name}",)),)


def _locate(registration: Registration, u: int) -> tuple[int, int]:
    """(execution, region) of region u, numbering execution 0's regions, then execution 1's, and so on."""
    for i, e in enumerate(registration.executions):
        if u < len(e.partition):
            return i, u
        u -= len(e.partition)
    raise IndexError(u)


def _unit(registration: Registration, circuits: dict[str, Circuit], u: int) -> Unit:
    i, r = _locate(registration, u)
    e = registration.executions[i]
    return Unit(circuits[e.circuit], e, r)

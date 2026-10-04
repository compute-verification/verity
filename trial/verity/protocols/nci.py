"""Novel communication intensity: the policy that throttles frontier training and leaves inference alone. A condition on
the declaration, with no rounds of its own (`Nci.Protocol` in the main repo): every isolation unit, here a region of the
registered partition, has at most `gamma` gates and reads at most `alpha` bits of novel information per gate. A bit is
novel unless it is on its circuit's baseline.

    verifier    takes the registration, and records whether it complies
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.primitives import partition as P
from verity.primitives.events import Actor, Send, receive
from verity.primitives.registration import Registration


@dataclass(frozen=True)
class Complies:
    registration: str
    complies: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Nci:
    gamma: int  # gates per isolation unit
    alpha: float  # novel bits read per gate

    def verifier(self) -> Actor:
        registration = yield from receive(Registration, "prover")
        reasons = []
        for circuit_id, partition in {(e.circuit, e.partition) for e in registration.executions}:
            circuit = registration.circuits.get(circuit_id)
            if circuit is None:
                reasons.append(f"circuit {circuit_id} not given")
                continue
            for r, region in enumerate(partition):
                novel = len(set(P.boundary(circuit, region)[0]) - set(circuit.baseline))
                if len(region) > self.gamma or novel > self.alpha * len(region):
                    reasons.append(f"circuit {circuit_id} region {r}: {len(region)} gates reading {novel} novel bits")
        return (Send(Complies(registration.digest(), not reasons, tuple(sorted(reasons)))),)

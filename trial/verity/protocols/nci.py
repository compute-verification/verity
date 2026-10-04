"""Novel communication intensity: the policy that throttles frontier training and leaves inference alone. A condition on
the declaration, with no rounds of its own (`Nci.Protocol` in the main repo): every isolation unit, here a region of the
registered partition, has at most `gamma` gates and reads at most `alpha` bits of novel information per gate. A bit is
novel unless it is on a baseline wire, committed beforehand (weights, constants).

    verifier    reads the registration integrity accepted, and records whether it complies
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.primitives import partition as P
from verity.primitives.events import Actor, Send, receive
from verity.protocols.integrity.registration import Registered


@dataclass(frozen=True)
class Complies:
    registration: str
    complies: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Nci:
    gamma: int  # gates per isolation unit
    alpha: float  # novel bits read per gate
    baseline: frozenset[int] = frozenset()  # the baseline wires
    name: str = "nci"

    def verifier(self, to: tuple[str, ...] = ()) -> dict[str, Actor]:
        return {f"verifier/{self.name}": self._verifier(to)}

    def _verifier(self, to: tuple[str, ...]) -> Actor:
        registered = yield from receive(Registered, "verifier")
        reasons = []
        for circuit_id, partition in {(e.circuit, e.partition) for e in registered.registration.executions}:
            circuit = registered.circuits[circuit_id]
            for r, region in enumerate(partition):
                novel = len(set(P.boundary(circuit, region)[0]) - self.baseline)
                if len(region) > self.gamma or novel > self.alpha * len(region):
                    reasons.append(f"circuit {circuit_id} region {r}: {len(region)} gates reading {novel} novel bits")
        return (Send(Complies(registered.registration.digest(), not reasons, tuple(sorted(reasons))), to=to),)

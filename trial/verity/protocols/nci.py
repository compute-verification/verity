"""Novel communication intensity: the policy that throttles frontier training and leaves inference alone. A condition on
what the prover ran, with no rounds of its own (`Nci.Protocol` in the main repo): every isolation unit, here a region of
an execution's partition, has at most `gamma` gates and reads at most `alpha` bits of novel information per gate. A bit
is novel unless it is on the baseline, which is on no region's boundary. Advice is novel too, and reaches a unit
through the structure the builder gives it rather than through its boundary, so it has a budget of its own: at most
`advice` bits across the epoch's executions.

    verifier    takes the registration and the stream (one epoch), rebuilds each distinct advice's circuit, and records
                whether every execution complies
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.primitives import partition as P
from verity.primitives.events import Actor, Send, collect, receive
from verity.primitives.registration import (
    Closed,
    Execution,
    Registration,
    built,
    digest,
)


@dataclass(frozen=True)
class Complies:
    registration: str  # the stream's digest
    complies: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Nci:
    gamma: int  # gates per isolation unit
    alpha: float  # novel bits read per gate
    advice: int  # advice bits per epoch

    def verifier(self) -> Actor:
        registration = yield from receive(Registration, "prover")
        executions = yield from collect(Execution, Closed, "prover")
        spent = sum(len(e.advice) for e in executions)
        reasons = [f"{spent} bits of advice, over the epoch's {self.advice}"] if spent > self.advice else []
        for advice in dict.fromkeys(e.advice for e in executions):
            try:
                circuit, partition = built(registration.builder, advice)
            except ValueError as error:
                reasons.append(str(error))
                continue
            for r, region in enumerate(partition):
                novel = len(P.boundary(circuit, region)[0])
                if len(region) > self.gamma or novel > self.alpha * len(region):
                    reasons.append(f"advice {advice!r} region {r}: {len(region)} gates reading {novel} novel bits")
        return (Send(Complies(digest(registration, executions), not reasons, tuple(reasons))),)

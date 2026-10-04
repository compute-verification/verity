"""Novel communication intensity: the policy that throttles frontier training and leaves inference alone
(`Nci.Protocol` in the main repo). Every isolation unit of an execution's partition has at most `gamma` gates and
reads at most `alpha` bits of novel information per gate. A bit is novel unless it is on the baseline, which is on no
isolation unit's boundary; so the baseline must have been declared up front, before any epoch whose outputs could have
shaped it, and the registration must name one of the `builders` declared then. Advice is novel too, and reaches a unit
through the structure the builder gives it rather than through its boundary, so it has a budget of its own: at most
`advice` bits across the epoch's executions. And the rule binds only what is in the account: whatever ran in the
compute that coverage leaves uncovered went unchecked, so at most `headroom` of the capacity may be left.

    verifier    takes the registration, the stream (one epoch) and compute coverage over it; rebuilds each distinct
                advice's circuit, and records whether the epoch complies

Coverage reads integrity, so the policy reads it through coverage, and holds except with coverage's probability.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.accounting.coverage.compute import ComputeCoverage
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
    """Whether the stream kept to the policy; except with probability `delta`, it did if `complies`."""

    stream: str
    complies: bool
    delta: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Nci:
    gamma: int  # gates per isolation unit
    alpha: float  # novel bits read per gate
    advice: int  # advice bits per epoch
    headroom: float  # the most of the compute capacity coverage may leave uncovered
    builders: frozenset[str]  # the ids of the builders declared up front

    def verifier(self) -> Actor:
        registration = yield from receive(Registration, "prover")
        executions = yield from collect(Execution, Closed, "prover")
        coverage = yield from receive(ComputeCoverage, "verifier")
        stream, reasons = digest(registration, executions), []
        if registration.builder.id() not in self.builders:
            reasons.append(f"builder {registration.builder.id()} was not declared up front")
        if coverage.reasons or coverage.stream != stream:
            reasons.append("compute coverage not established")
        elif coverage.headroom > self.headroom * coverage.capacity:
            reasons.append(
                f"{coverage.headroom / coverage.capacity:.0%} of compute uncovered, over {self.headroom:.0%}"
            )
        spent = sum(len(e.advice) for e in executions)
        if spent > self.advice:
            reasons.append(f"{spent} bits of advice, over the epoch's {self.advice}")
        for advice in dict.fromkeys(e.advice for e in executions):
            try:
                circuit, partition = built(registration.builder, advice)
            except ValueError as error:
                reasons.append(str(error))
                continue
            for u, unit in enumerate(partition):
                gates = P.gates(unit)
                novel = len(P.boundary(circuit, gates)[0])
                if len(gates) > self.gamma or novel > self.alpha * len(gates):
                    reasons.append(
                        f"advice {advice!r} isolation unit {u}: {len(gates)} gates reading {novel} novel bits"
                    )
        return (Send(Complies(stream, not reasons, coverage.delta, tuple(reasons))),)

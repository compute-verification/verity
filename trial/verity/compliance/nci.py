"""Novel communication intensity: the policy that throttles frontier training and leaves inference alone
(`Nci.Protocol` in the main repo). Every isolation unit of an execution's partition has at most `gamma` gates and
reads at most `alpha` bits of novel information per gate. A bit is novel unless it is on the baseline, which is on no
isolation unit's boundary; so the baseline must have been declared up front, before any epoch whose outputs could have
shaped it, and the registration must name one of the `builders` declared then. State an execution reads from an
earlier one (its KV cache) is novel to it, like anything another unit computed: no unit spans executions, since one
joining a session's steps would grow with the session, past any `gamma` small enough to throttle training. Advice is
novel too, and reaches a unit through the structure the builder gives it rather than through its boundary: the builder
reads all of it, so any unit of the execution may depend on all of it, and each is charged it beside its boundary. And
the rule binds only what is in the account: whatever ran in the compute that coverage leaves uncovered went unchecked,
so at most `headroom` of the capacity may be left.

    verifier    takes the registration, the stream (one epoch) and compute coverage over it; rebuilds each distinct
                advice's circuit, and records whether the epoch complies

Coverage reads integrity, so the policy reads it through coverage, and holds except with coverage's probability.

An epoch's verdict says nothing outside its window. A period is epochs back to back (`period`): each complies, and each
window opens no later than the one before it closed, so no time between them goes unaccounted. An epoch with no
verdict leaves a gap, so the period fails closed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.accounting.coverage.compute import ComputeCoverage
from verity.accounting.utilization.work import Window
from verity.primitives import partition as P
from verity.primitives.events import Actor, Send, collect, receive
from verity.primitives.registration import Closed, Execution, Registration, Stream, built


@dataclass(frozen=True)
class Complies:
    """Whether the stream kept to the policy over the work's `window`; except with probability `delta`, it did if
    `complies`."""

    stream: str
    window: Window
    complies: bool
    delta: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Period:
    """Whether the epochs over `streams`, back to back over `window`, kept to the policy; except with probability
    `delta`, they did if `complies`."""

    streams: tuple[str, ...]
    window: Window
    complies: bool
    delta: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Nci:
    gamma: int  # gates per isolation unit
    alpha: float  # novel bits read per gate
    headroom: float  # the most of the compute capacity coverage may leave uncovered
    builders: frozenset[str]  # the ids of the builders declared up front

    def verifier(self) -> Actor:
        registration = yield from receive(Registration, "prover")
        stream = Stream(registration, (yield from collect(Execution, Closed, "prover")))
        coverage = yield from receive(ComputeCoverage, "verifier")
        reasons = []
        if registration.builder.id() not in self.builders:
            reasons.append(f"builder {registration.builder.id()} was not declared up front")
        if coverage.reasons or coverage.stream != stream.digest():
            reasons.append("compute coverage not established")
        elif coverage.headroom > self.headroom * coverage.capacity:
            reasons.append(
                f"{coverage.headroom / coverage.capacity:.0%} of compute uncovered, over {self.headroom:.0%}"
            )
        for advice in dict.fromkeys(e.advice for e in stream.executions):
            try:
                circuit, partition = built(registration.builder, advice)
            except ValueError as error:
                reasons.append(str(error))
                continue
            for u, unit in enumerate(partition):
                gates = P.gates(unit)
                novel = len(P.boundary(circuit, gates)[0]) + len(advice)
                if len(gates) > self.gamma or novel > self.alpha * len(gates):
                    reasons.append(
                        f"advice {advice!r} isolation unit {u}: {len(gates)} gates reading {novel} novel bits"
                    )
        return (Send(Complies(stream.digest(), coverage.window, not reasons, coverage.delta, tuple(reasons))),)


def period(epochs: Sequence[Complies]) -> Period:
    """The verdicts of a period's epochs, in order, by one clock. Windows that overlap only count capacity twice; a gap
    between two is time nothing accounts for. The deltas add, by a union bound."""
    reasons = [f"epoch {i}: {why}" for i, epoch in enumerate(epochs) for why in epoch.reasons]
    reasons += [
        f"{b.window.opened - a.window.closed:.2f}s unaccounted between epochs {i} and {i + 1}"
        for i, (a, b) in enumerate(zip(epochs, epochs[1:]))
        if b.window.opened > a.window.closed
    ]
    window = Window(epochs[0].window.opened, epochs[-1].window.closed)
    delta = sum(epoch.delta for epoch in epochs)
    return Period(tuple(epoch.stream for epoch in epochs), window, not reasons, delta, tuple(reasons))

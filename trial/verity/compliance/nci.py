"""Novel communication intensity: the policy that throttles frontier training and leaves inference alone
(`Nci.Protocol` in the main repo). Every isolation unit of the stream has at most `gamma` gates and reads at most
`alpha` bits of novel information per gate. The stream's isolation units are its executions', joined where one reads
state another wrote (`registration.isolation_units`), so a session is one unit however many steps it took, and state
it wrote itself is not novel to it: a session reads only its prompt, where a step on its own would read its whole KV
cache. Any other bit is novel unless it is on the baseline, which is on no isolation
unit's boundary; so the baseline must have been declared up front, before any epoch whose outputs could have shaped
it, and the registration must name one of the `builders` declared then. Advice is novel too, and reaches a unit
through the structure the builder gives it rather than through its boundary: the builder reads all of it, so any unit
of the execution may depend on all of it, and each is charged it beside its boundary. And the rule binds only what is
in the account: whatever ran in the compute that coverage leaves uncovered went unchecked, so at most `headroom` of
the capacity may be left.

    verifier    takes the registration, the stream (one epoch) and compute coverage over it; rebuilds each execution's
                circuit, and records whether the epoch complies

Coverage reads integrity, so the policy reads it through coverage, and holds except with coverage's probability.

An epoch's verdict says nothing outside its window. A period is epochs back to back (`period`): each complies, and each
window opens no later than the one before it closed, so no time between them goes unaccounted. `period` takes every
epoch's verdict; an epoch with none leaves the period with none, which fails closed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.accounting.coverage.compute import ComputeCoverage
from verity.accounting.utilization.work import Window
from verity.primitives import partition as P
from verity.primitives.events import Actor, Send, collect, receive
from verity.primitives.registration import Closed, Execution, Registration, Stream, isolation_units


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
        try:
            units = isolation_units(stream)
        except ValueError as error:
            units = []
            reasons.append(str(error))
        for parts in units:
            gates, novel = _reads(stream, parts)
            if gates > self.gamma or novel > self.alpha * gates:
                (i, u), steps = parts[0], len({i for i, _ in parts})
                reasons.append(
                    f"isolation unit {u} of execution {i}, over {steps} executions: {gates} gates reading {novel} novel"
                    " bits"
                )
        return (Send(Complies(stream.digest(), coverage.window, not reasons, coverage.delta, tuple(reasons))),)


def _reads(stream: Stream, parts: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """An isolation unit of the stream's gates, and the novel bits it reads: each part's boundary but the state the
    unit wrote itself, and the advice of each execution it is part of."""
    gates = novel = 0
    for i, u in parts:
        circuit, partition = stream.built(i)
        unit = P.gates(partition[u])
        gates += len(unit)
        novel += sum(circuit.carried(w) is None for w in P.boundary(circuit, unit)[0])
    return gates, novel + sum(len(stream.executions[i].advice) for i in {i for i, _ in parts})


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

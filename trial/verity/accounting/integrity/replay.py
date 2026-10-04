"""Replay: sample whole executions, and have a sealed replayer run each drawn one again.

    verifier    takes the registration and the stream, draws executions from its secret once the stream closes, and
                sends the draw to the prover and the registration with the drawn executions to the replayer
    prover      opens each drawn execution's inputs and outputs against its root, and sends them to the replayer
    replayer    rebuilds each drawn execution's circuit from its advice, runs it again from the opened inputs, and tells
                the verifier whether it gave the opened outputs; it is trusted and sealed: it sees the values, and the
                verifier and the auditor do not

Replay needs the execution to be reproducible bit for bit on the replayer, which the circuit's exact semantics give.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.accounting.integrity.profile import profile
from verity.accounting.integrity.sampling import Draw, Subset
from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, evaluate, transcript
from verity.primitives.commitments import CommitmentScheme, MerkleSha256
from verity.primitives.events import Actor, Send, collect, gather, receive
from verity.primitives.randomness import derive
from verity.primitives.registration import (
    Closed,
    Execution,
    Opening,
    Registration,
    built,
    check,
    digest,
    open_wires,
    verify_opening,
    wires,
)


@dataclass(frozen=True)
class ToReplay:
    """Verifier to replayer: the registration, and the drawn executions as streamed."""

    registration: Registration
    executions: Mapping[int, Execution]


@dataclass(frozen=True)
class Inputs:
    """Prover to replayer: each drawn execution's inputs and outputs, opened against its root."""

    openings: Mapping[int, Opening]


@dataclass(frozen=True)
class Replayed:
    """Replayer to verifier: whether each drawn execution, run again, gave the outputs committed for it."""

    results: Mapping[int, bool]


@dataclass(frozen=True)
class Replay:
    k: int  # executions drawn
    delta: float
    scheme: CommitmentScheme = MerkleSha256()

    def verifier(self) -> Actor:
        secret = secrets.token_bytes(32)
        registration = yield from receive(Registration, "prover")
        executions = yield from collect(Execution, Closed, "prover")
        stream, n = digest(registration, executions), len(executions)
        refused = check(registration, executions, self.scheme)
        if refused:
            return (Send(profile(stream, "execution", [0] * n, [0] * n, n, self.delta, refused)),)

        law = Subset(self.k)
        drawn = law.draw(derive(secret, "verity/integrity/replay/v1", stream), n)
        results = (
            yield from receive(
                Replayed,
                "replayer",
                Send(Draw(drawn), "prover"),
                Send(ToReplay(registration, {i: executions[i] for i in drawn}), "replayer"),
            )
        ).results
        reasons = tuple(
            f"execution {i}: " + ("not replayed" if i not in results else "replays differently")
            for i in drawn
            if not results.get(i)
        )
        circuits = [built(registration.builder, e.advice)[0] for e in executions]
        gates, outputs = [len(c.gates) for c in circuits], [len(c.outputs) for c in circuits]
        return (Send(profile(stream, "execution", gates, outputs, law.bound(n, self.delta), self.delta, reasons)),)

    def replayer(self) -> Actor:
        got = yield from gather({ToReplay: "verifier", Inputs: "prover"})
        expected, openings = got[ToReplay], got[Inputs].openings
        builder = expected.registration.builder
        results = {
            i: i in openings and _replays(self.scheme, *built(builder, e.advice), e, openings[i])
            for i, e in expected.executions.items()
        }
        return (Send(Replayed(results), "verifier"),)

    def prover(
        self, registration: Registration, executions: Sequence[Execution], inputs: Sequence[tuple[int, ...]]
    ) -> Actor:
        draw = yield from receive(Draw, "verifier")
        opened = {}
        for i in draw.units:
            circuit, partition = built(registration.builder, executions[i].advice)
            values = transcript(circuit, inputs[i])
            opened[i] = open_wires(self.scheme, circuit, partition, values, _io(circuit, partition))
        return (Send(Inputs(opened), "replayer"),)


def _io(circuit: Circuit, partition: P.Partition) -> tuple[int, ...]:
    """The committed wires that are the execution's inputs, then its outputs."""
    return tuple(dict.fromkeys([*(w for w in wires(circuit, partition) if w < circuit.inputs), *circuit.outputs]))


def _replays(
    scheme: CommitmentScheme, circuit: Circuit, partition: P.Partition, execution: Execution, opening: Opening
) -> bool:
    if set(opening.values) != set(_io(circuit, partition)):
        return False
    if not verify_opening(scheme, circuit, partition, execution.root, opening):
        return False
    # an input no gate reads is not committed
    inputs = tuple(opening.values.get(w, 0) for w in range(len(circuit.baseline), circuit.inputs))
    return evaluate(circuit, inputs) == tuple(opening.values[o] for o in circuit.outputs)

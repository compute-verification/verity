"""Replay: sample whole executions, and have a sealed replayer run each drawn one again.

    verifier    takes the registration, draws executions from its secret, and sends the draw to the prover and the
                drawn executions, as registered, to the replayer
    prover      opens each drawn execution's inputs and outputs against its root, and sends them to the replayer
    replayer    runs each execution again from its opened inputs and tells the verifier whether it gave the opened
                outputs; it is trusted and sealed: it sees the values, and the verifier and the auditor do not

Replay needs no fine partition (the prover may register each execution as one region, `partition.whole`), and the
execution to be reproducible bit for bit on the replayer, which the circuit's exact semantics give.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, evaluate, trace
from verity.primitives.commitments import CommitmentScheme, MerkleSha256
from verity.primitives.events import Actor, Send, gather, receive
from verity.primitives.randomness import derive
from verity.primitives.registration import (
    Execution,
    Opening,
    Registration,
    check,
    open_wires,
    verify_opening,
    wires,
)
from verity.protocols.integrity.profile import IntegrityProfile
from verity.protocols.integrity.sampling import Draw, Subset


@dataclass(frozen=True)
class ToReplay:
    """Verifier to replayer: the drawn executions, as registered."""

    executions: Mapping[int, Execution]


@dataclass(frozen=True)
class Inputs:
    """Prover to replayer: each drawn execution's circuit, and its inputs and outputs opened against its root."""

    executions: Mapping[int, tuple[Circuit, Opening]]


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
        digest, n = registration.digest(), len(registration.executions)
        refused = check(registration, self.scheme)
        if refused:
            return (Send(IntegrityProfile(digest, "execution", False, n, n, self.delta, refused)),)

        law = Subset(self.k)
        drawn = law.draw(derive(secret, "verity/integrity/replay/v1", digest), n)
        results = (
            yield from receive(
                Replayed,
                "replayer",
                Send(Draw(drawn), "prover"),
                Send(ToReplay({i: registration.executions[i] for i in drawn}), "replayer"),
            )
        ).results
        reasons = tuple(
            f"execution {i}: " + ("not replayed" if i not in results else "replays differently")
            for i in drawn
            if not results.get(i)
        )
        return (
            Send(IntegrityProfile(digest, "execution", not reasons, n, law.bound(n, self.delta), self.delta, reasons)),
        )

    def replayer(self) -> Actor:
        got = yield from gather({ToReplay: "verifier", Inputs: "prover"})
        expected, provided = got[ToReplay].executions, got[Inputs].executions
        results = {i: i in provided and _replays(self.scheme, e, *provided[i]) for i, e in expected.items()}
        return (Send(Replayed(results), "verifier"),)

    def prover(self, inputs: Sequence[tuple[int, ...]], registration: Registration) -> Actor:
        draw = yield from receive(Draw, "verifier")
        opened = {}
        for i in draw.units:
            e = registration.executions[i]
            circuit = registration.circuits[e.circuit]
            opened[i] = (
                circuit,
                open_wires(self.scheme, circuit, e.partition, trace(circuit, inputs[i]), _io(circuit, e.partition)),
            )
        return (Send(Inputs(opened), "replayer"),)


def _io(circuit: Circuit, partition: P.Partition) -> tuple[int, ...]:
    """The committed wires that are the execution's inputs, then its outputs."""
    return tuple(dict.fromkeys([*(w for w in wires(circuit, partition) if w < circuit.inputs), *circuit.outputs]))


def _replays(scheme: CommitmentScheme, execution: Execution, circuit: Circuit, opening: Opening) -> bool:
    if circuit.id() != execution.circuit or set(opening.values) != set(_io(circuit, execution.partition)):
        return False
    if not verify_opening(scheme, circuit, execution, opening):
        return False
    inputs = tuple(opening.values.get(w, 0) for w in range(circuit.inputs))  # an input no gate reads is not committed
    return evaluate(circuit, inputs) == tuple(opening.values[o] for o in circuit.outputs)

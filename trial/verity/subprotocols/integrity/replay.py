"""Replay: sample whole executions, and have a sealed replayer run each drawn one again.

    verifier    receives the registration, draws executions from its secret, and sends the draw to the prover and the
                drawn executions, as registered, to the replayer
    prover      opens each drawn execution's inputs and outputs against its root, and sends them to the replayer
    replayer    runs each execution again from its opened inputs and tells the verifier whether it gave the opened
                outputs; it is trusted and sealed: it sees the values, and the verifier and the auditor do not

Replay needs no fine partition (the prover may register each execution as one region), and the execution to be
reproducible bit for bit on the replayer, which the circuit's exact semantics give.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, evaluate, trace
from verity.primitives.commitments import CommitmentScheme
from verity.primitives.events import Party, Send, receive
from verity.primitives.randomness import derive
from verity.subprotocols.integrity.profile import IntegrityProfile
from verity.subprotocols.integrity.registration import (
    Execution, Opening, Registered, check, open_wires, register, verify_opening, wires,
)
from verity.subprotocols.integrity.sampling import Draw, Subset


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
    name: str = "integrity"

    def verifier(self, scheme: CommitmentScheme) -> dict[str, Party]:
        return {f"verifier/{self.name}": self._verifier(scheme)}

    def replayer(self, scheme: CommitmentScheme) -> dict[str, Party]:
        return {f"replayer/{self.name}": self._replayer(scheme)}

    def prover(
        self, scheme: CommitmentScheme, circuit: Circuit, inputs: Sequence[tuple[int, ...]], partition: P.Partition | None = None
    ) -> dict[str, Party]:
        whole = (tuple(range(len(circuit.gates))),)
        return {f"prover/{self.name}": self._prover(scheme, circuit, partition or whole, inputs)}

    def _verifier(self, scheme: CommitmentScheme) -> Party:
        secret = secrets.token_bytes(32)
        registered = (yield from receive(Registered)).body
        registration = registered.registration
        digest, n = registration.digest(), len(registration.executions)
        refused = check(registered, scheme)
        if refused:
            return (Send(IntegrityProfile(digest, "execution", False, n, n, self.delta, refused)),)

        law = Subset(self.k)
        drawn = law.draw(derive(secret, "verity/integrity/replay/v1", digest), n)
        results = (yield from receive(
            Replayed,
            Send(Draw(drawn), to=(f"prover/{self.name}",)),
            Send(ToReplay({i: registration.executions[i] for i in drawn}), to=(f"replayer/{self.name}",)),
        )).body.results
        reasons = tuple(f"execution {i}: " + ("not replayed" if i not in results else "replays differently") for i in drawn if not results.get(i))
        return (Send(IntegrityProfile(digest, "execution", not reasons, n, law.bound(n, self.delta), self.delta, reasons)),)

    def _replayer(self, scheme: CommitmentScheme) -> Party:
        expected: Mapping[int, Execution] | None = None
        provided: Mapping[int, tuple[Circuit, Opening]] | None = None
        while expected is None or provided is None:
            event = yield ()
            match event.body:
                case ToReplay(executions):
                    expected = executions
                case Inputs(executions):
                    provided = executions
        results = {i: i in provided and _replays(scheme, e, *provided[i]) for i, e in expected.items()}
        return (Send(Replayed(results), to=(f"verifier/{self.name}",)),)

    def _prover(self, scheme: CommitmentScheme, circuit: Circuit, partition: P.Partition, inputs: Sequence[tuple[int, ...]]) -> Party:
        registration = register(circuit, partition, inputs, scheme)
        draw = (yield from receive(Draw, Send(Registered(registration, {circuit.id(): circuit}), to=(f"verifier/{self.name}",)))).body
        io = _io(circuit, partition)
        opened = {i: (circuit, open_wires(scheme, circuit, partition, trace(circuit, inputs[i]), io)) for i in draw.units}
        return (Send(Inputs(opened), to=(f"replayer/{self.name}",)),)


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

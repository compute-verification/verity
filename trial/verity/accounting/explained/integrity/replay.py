"""Replay: sample whole executions, and have a sealed replayer run each drawn one again.

    verifier    takes the registration and the stream, draws executions from its secret once the stream closes, each
                as likely as its gates and output bits ask (`sampling.Weighted`), and sends the draw to the prover and
                the stream with the draw to the replayer
    prover      opens each drawn execution's inputs and outputs from the transcripts where they are committed (its
                state where its writer committed it), and sends them to the replayer
    replayer    rebuilds each drawn execution's circuit from its advice, runs it again from the opened inputs, and tells
                the verifier whether it gave the opened outputs; it is trusted and sealed: it sees the values, and the
                verifier and the auditor do not

Replay needs the execution to be reproducible bit for bit on the replayer, which the circuit's exact semantics give.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.accounting.explained.integrity.profile import profile
from verity.accounting.explained.integrity.sampling import Draw, Weighted
from verity.primitives import partition as P
from verity.primitives.circuit import evaluate
from verity.primitives.commitments import CommitmentScheme, MerkleSha256
from verity.primitives.events import Actor, Send, collect, gather, receive
from verity.primitives.randomness import derive
from verity.primitives.registration import (
    Closed,
    Execution,
    Opening,
    Registration,
    Stream,
    check,
    open_wires,
    verify_opening,
)


@dataclass(frozen=True)
class ToReplay:
    """Verifier to replayer: the stream, and which of its executions were drawn."""

    stream: Stream
    drawn: tuple[int, ...]


@dataclass(frozen=True)
class Inputs:
    """Prover to replayer: each drawn execution's inputs and outputs, opened where they are committed."""

    openings: Mapping[int, Opening]


@dataclass(frozen=True)
class Replayed:
    """Replayer to verifier: whether each drawn execution, run again, gave the outputs committed for it."""

    results: Mapping[int, bool]


@dataclass(frozen=True)
class Replay:
    law: Weighted
    delta: float
    scheme: CommitmentScheme = MerkleSha256()

    def verifier(self) -> Actor:
        secret = secrets.token_bytes(32)
        registration = yield from receive(Registration, "prover")
        stream = Stream(registration, (yield from collect(Execution, Closed, "prover")))
        refused = check(stream, self.scheme)
        if refused:
            return (Send(profile(stream.digest(), "execution", [], [], self.law, self.delta, refused)),)

        circuits = [stream.built(i)[0] for i in range(len(stream.executions))]
        gates = [len(c.gates) for c in circuits]
        outputs = [len({o for o in c.outputs if o >= c.inputs}) for c in circuits]
        coins = derive(secret, "verity/integrity/replay/v2", stream.digest())
        drawn = self.law.draw(coins, gates, outputs, self.delta)
        results = (
            yield from receive(
                Replayed, "replayer", Send(Draw(drawn), "prover"), Send(ToReplay(stream, drawn), "replayer")
            )
        ).results
        reasons = tuple(
            f"execution {i}: " + ("not replayed" if i not in results else "replays differently")
            for i in drawn
            if not results.get(i)
        )
        return (Send(profile(stream.digest(), "execution", gates, outputs, self.law, self.delta, reasons)),)

    def replayer(self) -> Actor:
        got = yield from gather({ToReplay: "verifier", Inputs: "prover"})
        expected, openings = got[ToReplay], got[Inputs].openings
        results = {i: i in openings and _replays(self.scheme, expected.stream, i, openings[i]) for i in expected.drawn}
        return (Send(Replayed(results), "verifier"),)

    def prover(self, stream: Stream, transcripts: Sequence[Sequence[int]]) -> Actor:
        draw = yield from receive(Draw, "verifier")
        opened = {i: open_wires(self.scheme, stream, transcripts, _io(stream, i)) for i in draw.units}
        return (Send(Inputs(opened), "replayer"),)


def _io(stream: Stream, i: int) -> tuple[tuple[int, int], ...]:
    """Where execution i's inputs that a gate reads, then its outputs, are committed."""
    circuit = stream.built(i)[0]
    read = P.boundary(circuit, range(len(circuit.gates)))[0]
    return tuple(dict.fromkeys([*(stream.locate(i, w) for w in read), *((i, o) for o in circuit.outputs)]))


def _replays(scheme: CommitmentScheme, stream: Stream, i: int, opening: Opening) -> bool:
    if set(opening.values) != set(_io(stream, i)) or not verify_opening(scheme, stream, opening):
        return False
    circuit = stream.built(i)[0]
    # an input no gate reads is not committed
    inputs = tuple(opening.values.get(stream.locate(i, w), 0) for w in range(len(circuit.baseline), circuit.inputs))
    return evaluate(circuit, inputs) == tuple(opening.values[i, o] for o in circuit.outputs)

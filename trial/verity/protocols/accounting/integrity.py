"""Integrity: what ran. The stream of the prover's executions, and checks of units drawn from it.

Before any coins, the prover registers its builder (`circuits.Builder`), and with it the baseline every circuit it
builds holds as its first inputs. Then it streams its executions as it runs them, and closes the stream: one epoch.
Per execution: its advice, and one root over the value of every boundary wire of the regions of the partition the
builder gave it (one leaf per wire, in wire order), but its state inputs. An execution that reads what an earlier one
wrote (a decode step, its KV cache) reads it from the writer's root (`Stream.locate`), so a wire is committed once, by
the execution that computed it. State links only within the stream: a session that outlives its epoch carries its
state into the next one as fresh inputs. The verifier rebuilds each execution's circuit and partition from its advice
and records the `Stream`, which work, the account and NCI read; every statement over the stream names its digest.
Today's reference: `verity_sampled_proofs.one_stage.registration`.

Then it checks units of the stream: whole executions, or with `max_width` every region of every execution, each of at
most that many boundary wires. It draws units from its secret once the stream closes, each as likely as its gates and
output bits ask (`randomness.Weighted`), and `verification` decides each drawn unit's claim: that its gates compute the
outputs committed for it from the inputs committed for it, each where `Stream.locate` says. Passing establishes the
`IntegrityProfile`.

    verifier    takes the registration and the stream, records the stream, draws units, and reads verification's
                `Verified` records until every drawn unit's claim is decided; records the profile
    prover      registers and streams (`serve`); opens each drawn unit's claim from the transcripts it keeps (which it
                rematerializes) and runs verification's prover on them
"""

from __future__ import annotations

import functools
import hashlib
import json
import secrets
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from verity.primitives import circuits as C
from verity.primitives.circuits import Builder, Circuit, built
from verity.primitives.commitments import Commitment, CommitmentScheme
from verity.primitives.events import Actor, Send, collect, receive
from verity.primitives.randomness import Weighted, derive
from verity.protocols.verification.claim import Claim, Opening, Verified, leaf
from verity.protocols.verification.proofs.protocol import Proofs
from verity.protocols.verification.replay import Replay


@dataclass(frozen=True)
class Registration:
    builder: Builder


@dataclass(frozen=True)
class Execution:
    advice: str  # a bitstring
    root: Commitment  # over its committed wires' values


@dataclass(frozen=True)
class Closed:
    """The stream's last message: the executions before it are all there are."""


@dataclass(frozen=True)
class Draw:
    """Verifier to prover: the drawn units, numbered at the construction's scale."""

    units: tuple[int, ...]


@dataclass(frozen=True)
class Stream:
    """The registration and the executions streamed after it."""

    registration: Registration
    executions: tuple[Execution, ...] = ()

    def digest(self) -> str:
        """What every statement over the stream names."""
        body = json.dumps(
            [self.registration.builder.id(), [[e.advice, asdict(e.root)] for e in self.executions]],
            separators=(",", ":"),
            default=lambda b: b.hex(),
        )
        return hashlib.sha256(b"verity/stream/v0\0" + body.encode()).hexdigest()

    def built(self, i: int) -> tuple[Circuit, C.Partition]:
        return built(self.registration.builder, self.executions[i].advice)

    def locate(self, i: int, wire: int) -> tuple[int, int]:
        """Where execution i's `wire` is committed: (the execution whose root holds it, its wire there). A state input
        is held where its writer computed it; raises ValueError if that is not an earlier execution's gate."""
        if (link := self.built(i)[0].carried(wire)) is None:
            return i, wire
        back, output = link
        if not 0 < back <= i:
            raise ValueError(f"state input {wire} reads {back} executions back, from execution {i}")
        writer = self.built(i - back)[0]
        if not (output < len(writer.outputs) and writer.outputs[output] >= writer.inputs):
            raise ValueError(f"state input {wire} reads no output a gate of execution {i - back} computed")
        return i - back, writer.outputs[output]


@functools.cache
def wires(circuit: Circuit, partition: C.Partition) -> tuple[int, ...]:
    """The wires an execution's root commits, in order: every boundary wire of its partition's regions but its state
    inputs, which their writers commit."""
    boundaries = (w for region in C.regions(partition) for side in C.boundary(circuit, region) for w in side)
    return tuple(sorted({w for w in boundaries if circuit.carried(w) is None}))


@functools.cache
def _index(circuit: Circuit, partition: C.Partition) -> dict[int, int]:
    return {w: k for k, w in enumerate(wires(circuit, partition))}


def commit(builder: Builder, advice: str, values: Sequence[int], scheme: CommitmentScheme) -> Execution:
    """Commit to an execution's boundary values, from its transcript (every wire's value, as the prover's runtime ran
    it: `circuits.transcript` if it ran honestly)."""
    return Execution(advice, scheme.commit([leaf(values[w]) for w in wires(*built(builder, advice))]))


def serve(stream: Stream, after: type | None = None) -> Actor:
    """The prover's stream: the registration, then each execution and the close, once `after` comes from the verifier
    (a coin the executions must follow, such as work's salt)."""
    first = (Send(stream.registration, "verifier"),)
    if after is not None:
        yield from receive(after, "verifier", *first)
        first = ()
    return (*first, *(Send(e, "verifier") for e in stream.executions), Send(Closed(), "verifier"))


def well_formed(stream: Stream, scheme: CommitmentScheme) -> tuple[str, ...]:
    """What is wrong with the stream's form: advice that builds nothing, state that reads no earlier execution's gate,
    a partition that leaves a gate out or holds one twice (so a wire on the execution's own boundary could go
    uncommitted), a root that is not one leaf per committed wire under the verifier's scheme."""
    why = []
    for i, e in enumerate(stream.executions):
        try:
            circuit, partition = stream.built(i)
            for w in range(len(circuit.baseline), len(circuit.baseline) + len(circuit.state)):
                stream.locate(i, w)
        except ValueError as error:
            why.append(f"execution {i}: {error}")
            continue
        if not C.covers(circuit, partition):
            why.append(f"execution {i}: the regions do not cover every gate exactly once")
        elif e.root.scheme != scheme.name or e.root.leaves != len(wires(circuit, partition)):
            why.append(f"execution {i}: its root is not one {scheme.name} leaf per committed wire")
    return tuple(why)


@dataclass(frozen=True)
class Count:
    total: int
    wrong: int  # the most of `total` in bad units


@dataclass(frozen=True)
class IntegrityProfile:
    """If `accepted`: of the stream's gates, at most `gates.wrong` are in bad units at `scale`, and of its output bits
    at most `outputs.wrong`, except with probability `delta`. A unit is bad when its claim does not hold. An output bit
    is a distinct circuit output a gate computes: an input passed through is no output, however often it is listed."""

    stream: str  # its digest
    scale: str  # what a unit is: "execution", "region"
    accepted: bool
    units: int
    gates: Count
    outputs: Count
    delta: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Integrity:
    verification: Proofs | Replay  # how a drawn unit's claim is decided
    law: Weighted
    delta: float
    max_width: int | None = None  # boundary wires per region; None: units are whole executions

    @property
    def scheme(self) -> CommitmentScheme:
        return self.verification.scheme

    @property
    def scale(self) -> str:
        return "execution" if self.max_width is None else "region"

    def verifier(self) -> Actor:
        secret = secrets.token_bytes(32)
        registration = yield from receive(Registration, "prover")
        stream = Stream(registration, (yield from collect(Execution, Closed, "prover")))
        refused = well_formed(stream, self.scheme)
        if not refused and self.max_width is not None:
            refused = tuple(
                f"execution {i}: {why}"
                for i in range(len(stream.executions))
                for why in C.validate(*stream.built(i), self.max_width)
            )
        if refused:
            return (Send(stream), Send(self._profile(stream, [], [], refused)))

        units = self.units(stream)
        gates, outputs = [len(g) for _, g in units], [_outputs(stream, i, g) for i, g in units]
        coins = derive(secret, f"verity/integrity/{self.scale}/v3", stream.digest())
        drawn = self.law.draw(coins, gates, outputs, self.delta)
        claims = {u: claim(stream, *units[u]) for u in drawn}
        pending, holds = set(claims.values()), {}
        sends = (Send(stream), Send(Draw(drawn), "prover"))
        while pending - holds.keys():
            verified = yield from receive(Verified, "verifier", *sends)
            sends = ()
            for c, ok in zip(verified.claims, verified.holds):
                if c in pending:
                    holds.setdefault(c, ok)
        reasons = tuple(f"{self.scale} {u}: does not verify" for u in drawn if not holds[claims[u]])
        return (*sends, Send(self._profile(stream, gates, outputs, reasons)))

    def prover(self, stream: Stream, transcripts: Sequence[Sequence[int]]) -> Actor:
        draw = yield from receive(Draw, "verifier")
        units = self.units(stream)
        drawn = [units[u] for u in draw.units]
        claims = tuple(claim(stream, i, g) for i, g in drawn)
        openings = tuple(opening(self.scheme, stream, transcripts, i, g) for i, g in drawn)
        return (yield from self.verification.prover(claims, openings))

    def units(self, stream: Stream) -> list[tuple[int, C.Region]]:
        """Every unit at the construction's scale, in order: (execution, its gates)."""
        if self.max_width is None:
            return [(i, tuple(range(len(stream.built(i)[0].gates)))) for i in range(len(stream.executions))]
        return [(i, region) for i in range(len(stream.executions)) for region in C.regions(stream.built(i)[1])]

    def _profile(
        self, stream: Stream, gates: list[int], outputs: list[int], reasons: tuple[str, ...]
    ) -> IntegrityProfile:
        """Accepted if no check failed (no `reasons`), with the law's bound; and if one did, everything could be
        wrong."""
        wrong = self.law.bound(gates, outputs, self.delta) if not reasons else (sum(gates), sum(outputs))
        return IntegrityProfile(
            stream.digest(),
            self.scale,
            not reasons,
            len(gates),
            Count(sum(gates), wrong[0]),
            Count(sum(outputs), wrong[1]),
            self.delta,
            reasons,
        )


def claim(stream: Stream, i: int, gates: C.Region) -> Claim:
    """The claim that `gates` of execution i compute what is committed for their boundary."""
    located = _located(stream, i, gates).items()
    at = tuple((w, stream.executions[j].root, _index(*stream.built(j))[written]) for w, (j, written) in located)
    return Claim(stream.built(i)[0], gates, at)


def opening(
    scheme: CommitmentScheme, stream: Stream, transcripts: Sequence[Sequence[int]], i: int, gates: C.Region
) -> Opening:
    """The opening of that claim, from the executions' transcripts."""
    values, paths, leaves = {}, {}, {}
    for w, (j, written) in _located(stream, i, gates).items():
        if j not in leaves:
            leaves[j] = [leaf(transcripts[j][x]) for x in wires(*stream.built(j))]
        values[w] = transcripts[j][written]
        paths[w] = scheme.open(leaves[j], _index(*stream.built(j))[written])
    return Opening(values, paths)


def _located(stream: Stream, i: int, gates: C.Region) -> dict[int, tuple[int, int]]:
    """Each boundary wire of `gates` of execution i, by wire, and where it is committed."""
    return {w: stream.locate(i, w) for w in sorted({*sum(C.boundary(stream.built(i)[0], gates), ())})}


def _outputs(stream: Stream, i: int, gates: C.Region) -> int:
    """The output bits `gates` of execution i compute."""
    circuit = stream.built(i)[0]
    return len(_circuit_outputs(circuit).intersection(C.boundary(circuit, gates)[1]))


@functools.cache
def _circuit_outputs(circuit: Circuit) -> frozenset[int]:
    return frozenset(circuit.outputs)

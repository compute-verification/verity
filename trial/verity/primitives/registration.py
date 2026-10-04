"""The registration and the stream: what everything over the prover's work reads.

Before any coins, the prover registers its builder: the program that builds each execution's circuit and partition from
that execution's advice. Advice is a bitstring of at most `advice_bits`, the prover's record of the runtime choices no
one could fix ahead of time (how the scheduler batched and chunked requests, which step a session continues, a
preemption, an autotune pick), which the builder knows how to read. Every circuit it builds holds the baseline
(weights, constants) as its first inputs, so registering the builder registers the baseline.

Then the prover streams its executions as it runs them, and closes the stream: one epoch. Per execution: its advice,
and one root over the value of every boundary wire of the regions of the partition the builder gave it (one leaf per
wire, in wire order), but its state inputs. An execution that reads what an earlier one wrote (a decode step, its KV
cache) reads it from the writer's root (`Stream.locate`), so a wire is committed once, by the execution that computed
it, and the region that reads a wire reads what the region that wrote it committed. The verifier rebuilds each
execution's circuit from its advice (`Stream`): integrity opens and proves against the roots, work keys each
execution's work from its salt and root, and NCI checks the stream's inference units. Every statement over the stream
names its digest. Today's reference for the commitment: `verity_sampled_proofs.one_stage.registration`.

The stream is one circuit, cut into executions as the prover ran them. Its inference units are the executions', joined
where one reads state another wrote (`inference_units`), so a session's steps are one inference unit, as NCI's
inference guarantee has it. State links only within the stream: a session that outlives its epoch carries its state
into the next one as fresh inputs.

The builder here is Python, which prover and verifier both run, standing in for a provable program (Rust under SP1, say)
whose id is its program's; the verifier's rerun stands in for checking a proof that the circuit is its output.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit
from verity.primitives.commitments import Commitment, CommitmentScheme
from verity.primitives.events import Actor, Send, receive


class Builder(Protocol):
    advice_bits: int  # the most advice one execution may take

    def id(self) -> str: ...

    def build(self, advice: str) -> tuple[Circuit, P.Partition]:
        """The execution's circuit and partition; raises ValueError on advice it cannot read."""
        ...


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


@functools.cache
def built(builder: Builder, advice: str) -> tuple[Circuit, P.Partition]:
    if len(advice) > builder.advice_bits or set(advice) - {"0", "1"}:
        raise ValueError(f"advice {advice!r} is not a bitstring of at most {builder.advice_bits} bits")
    return builder.build(advice)


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
        return hashlib.sha256(b"verity/registration/v0\0" + body.encode()).hexdigest()

    def built(self, i: int) -> tuple[Circuit, P.Partition]:
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
def wires(circuit: Circuit, partition: P.Partition) -> tuple[int, ...]:
    """The wires an execution's root commits, in order: every boundary wire of its partition's regions but its state
    inputs, which their writers commit."""
    boundaries = (w for region in P.regions(partition) for side in P.boundary(circuit, region) for w in side)
    return tuple(sorted({w for w in boundaries if circuit.carried(w) is None}))


def leaf(wire: int, value: int) -> bytes:
    return wire.to_bytes(4, "big") + value.to_bytes(4, "big")


def commit(builder: Builder, advice: str, values: Sequence[int], scheme: CommitmentScheme) -> Execution:
    """Commit to an execution's boundary values, from its transcript (every wire's value, as the prover's runtime ran
    it: `circuit.transcript` if it ran honestly)."""
    circuit, partition = built(builder, advice)
    return Execution(advice, scheme.commit(_leaves(circuit, partition, values)))


def serve(stream: Stream, after: type | None = None) -> Actor:
    """The prover's stream: the registration, then each execution and the close, once `after` comes from the verifier
    (a coin the executions must follow, such as work's salt)."""
    first = (Send(stream.registration, "verifier"),)
    if after is not None:
        yield from receive(after, "verifier", *first)
        first = ()
    return (*first, *(Send(e, "verifier") for e in stream.executions), Send(Closed(), "verifier"))


def check(stream: Stream, scheme: CommitmentScheme) -> tuple[str, ...]:
    """Every execution's advice builds, its state inputs read what earlier executions computed, and its root is one
    leaf per committed wire under the verifier's scheme."""
    why = []
    for i, e in enumerate(stream.executions):
        try:
            circuit, partition = stream.built(i)
            for w in range(len(circuit.baseline), len(circuit.baseline) + len(circuit.state)):
                stream.locate(i, w)
        except ValueError as error:
            why.append(f"execution {i}: {error}")
            continue
        if e.root.scheme != scheme.name or e.root.leaves != len(wires(circuit, partition)):
            why.append(f"execution {i}: its root is not one {scheme.name} leaf per committed wire")
    return tuple(why)


def inference_units(stream: Stream) -> list[list[tuple[int, int]]]:
    """The stream's inference units, each its parts (execution, inference unit of that execution's partition): the
    executions' own, joined where one reads state another wrote. Raises ValueError where `Stream.locate` does, and
    where a partition does not hold every gate exactly once."""
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for i in range(len(stream.executions)):
        circuit, partition = stream.built(i)
        if not P.covers(circuit, partition):
            raise ValueError(f"execution {i}: the regions do not cover every gate exactly once")
        for u, unit in enumerate(partition):
            find((i, u))
            for w in P.boundary(circuit, P.gates(unit))[0]:
                if circuit.carried(w) is not None:
                    j, written = stream.locate(i, w)
                    writer, cut = stream.built(j)
                    parent[find((i, u))] = find((j, P.unit_of(cut)[written - writer.inputs]))
    units: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for part in parent:
        units.setdefault(find(part), []).append(part)
    return sorted(units.values())


@dataclass(frozen=True)
class Opening:
    values: Mapping[tuple[int, int], int]  # (execution, wire), as `Stream.locate` gives it -> its committed value
    paths: Mapping[tuple[int, int], object]  # -> its opening against that execution's root


def open_wires(
    scheme: CommitmentScheme, stream: Stream, transcripts: Sequence[Sequence[int]], which: Iterable[tuple[int, int]]
) -> Opening:
    """Open committed wires, each (execution, wire) as `Stream.locate` gives it, from the executions' transcripts
    (which the prover rematerializes)."""
    by: dict[int, list[int]] = {}
    for i, w in which:
        by.setdefault(i, []).append(w)
    values, paths = {}, {}
    for i, ws in by.items():
        circuit, partition = stream.built(i)
        leaves, index = _leaves(circuit, partition, transcripts[i]), _index(circuit, partition)
        for w in ws:
            values[i, w], paths[i, w] = transcripts[i][w], scheme.open(leaves, index[w])
    return Opening(values, paths)


def verify_opening(scheme: CommitmentScheme, stream: Stream, opening: Opening) -> bool:
    for (i, w), v in opening.values.items():
        if not 0 <= i < len(stream.executions):
            return False
        root, index = stream.executions[i].root, _index(*stream.built(i))
        if w not in index or not scheme.verify(root, index[w], leaf(w, v), opening.paths.get((i, w))):
            return False
    return True


@functools.cache
def _index(circuit: Circuit, partition: P.Partition) -> dict[int, int]:
    return {w: k for k, w in enumerate(wires(circuit, partition))}


def _leaves(circuit: Circuit, partition: P.Partition, values: Sequence[int]) -> list[bytes]:
    return [leaf(w, values[w]) for w in wires(circuit, partition)]

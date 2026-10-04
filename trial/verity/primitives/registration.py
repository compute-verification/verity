"""The registration and the stream: what everything over the prover's work reads.

Before any coins, the prover registers its builder: the program that builds each execution's circuit and partition from
that execution's advice. Advice is a bitstring of at most `advice_bits`, the prover's record of the runtime choices no
one could fix ahead of time (how the scheduler batched and chunked requests, a preemption, an autotune pick), which the
builder knows how to read. Every circuit it builds holds the baseline (weights, constants) as its first inputs, so
registering the builder registers the baseline.

Then the prover streams its executions as it runs them, and closes the stream: one epoch. Per execution: its advice,
and one root over the value of every boundary wire of the regions of the partition the builder gave it (one leaf per
wire, in wire order). The verifier rebuilds each execution's circuit from its advice: integrity opens and proves against
the roots, work keys each execution's work from its salt and root, and NCI checks the partitions' isolation units.
Every statement over the stream names its digest. Today's reference for the commitment:
`verity_sampled_proofs.one_stage.registration`.

The builder here is Python, which prover and verifier both run, standing in for a provable program (Rust under SP1, say)
whose id is its program's; the verifier's rerun stands in for checking a proof that the circuit is its output.

Since a wire is committed once, the region that reads a wire reads what the region that wrote it committed.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, transcript
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
    root: Commitment  # over its boundary wires' values


@dataclass(frozen=True)
class Closed:
    """The stream's last message: the executions before it are all there are."""


@functools.cache
def built(builder: Builder, advice: str) -> tuple[Circuit, P.Partition]:
    if len(advice) > builder.advice_bits or set(advice) - {"0", "1"}:
        raise ValueError(f"advice {advice!r} is not a bitstring of at most {builder.advice_bits} bits")
    return builder.build(advice)


def digest(registration: Registration, executions: Sequence[Execution] = ()) -> str:
    """What every statement over the stream names."""
    body = json.dumps(
        [registration.builder.id(), [[e.advice, asdict(e.root)] for e in executions]],
        separators=(",", ":"),
        default=lambda b: b.hex(),
    )
    return hashlib.sha256(b"verity/registration/v0\0" + body.encode()).hexdigest()


@functools.cache
def wires(circuit: Circuit, partition: P.Partition) -> tuple[int, ...]:
    """Every boundary wire of the partition's regions, in the order the root commits them."""
    return tuple(sorted({w for region in P.regions(partition) for side in P.boundary(circuit, region) for w in side}))


def leaf(wire: int, value: int) -> bytes:
    return wire.to_bytes(4, "big") + value.to_bytes(4, "big")


def execute(builder: Builder, advice: str, inputs: tuple[int, ...], scheme: CommitmentScheme) -> Execution:
    """Run one execution and commit to its boundary values; the transcript is not kept."""
    circuit, partition = built(builder, advice)
    return Execution(advice, scheme.commit(_leaves(circuit, partition, transcript(circuit, inputs))))


def serve(registration: Registration, executions: Sequence[Execution], after: type | None = None) -> Actor:
    """The prover's stream: the registration, then each execution and the close, once `after` comes from the verifier
    (a coin the executions must follow, such as work's salt)."""
    first = (Send(registration, "verifier"),)
    if after is not None:
        yield from receive(after, "verifier", *first)
        first = ()
    return (*first, *(Send(e, "verifier") for e in executions), Send(Closed(), "verifier"))


def check(registration: Registration, executions: Sequence[Execution], scheme: CommitmentScheme) -> tuple[str, ...]:
    """Every execution's advice builds, and its root is one leaf per boundary wire under the verifier's scheme."""
    why = []
    for i, e in enumerate(executions):
        try:
            circuit, partition = built(registration.builder, e.advice)
        except ValueError as error:
            why.append(f"execution {i}: {error}")
            continue
        if e.root.scheme != scheme.name or e.root.leaves != len(wires(circuit, partition)):
            why.append(f"execution {i}: its root is not one {scheme.name} leaf per boundary wire")
    return tuple(why)


@dataclass(frozen=True)
class Opening:
    values: Mapping[int, int]  # wire -> its committed value
    paths: Mapping[int, object]  # wire -> its opening against the root


def open_wires(
    scheme: CommitmentScheme, circuit: Circuit, partition: P.Partition, values: Sequence[int], which: Iterable[int]
) -> Opening:
    """Open `which` of an execution's committed wires, from every wire's value (which the prover rematerializes)."""
    leaves = _leaves(circuit, partition, values)
    index = wires(circuit, partition).index
    which = tuple(which)
    return Opening({w: values[w] for w in which}, {w: scheme.open(leaves, index(w)) for w in which})


def verify_opening(
    scheme: CommitmentScheme, circuit: Circuit, partition: P.Partition, root: Commitment, opening: Opening
) -> bool:
    index = {w: i for i, w in enumerate(wires(circuit, partition))}
    return all(
        w in index and scheme.verify(root, index[w], leaf(w, v), opening.paths.get(w))
        for w, v in opening.values.items()
    )


def _leaves(circuit: Circuit, partition: P.Partition, values: Sequence[int]) -> list[bytes]:
    return [leaf(w, values[w]) for w in wires(circuit, partition)]

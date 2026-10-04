"""The registration: what the prover sends the verifier before any coins are drawn, and the one object every protocol
over its work reads. It is the declaration (every circuit it ran, with its baseline) and a commitment to each run: per
execution, in the order the prover ran them, the circuit's id, the partition it chose, and one root over the value of
every boundary wire of that partition (one leaf per wire, in wire order). Integrity opens and proves against the roots,
PoUW binds its salt to the digest, and NCI checks the partitions as isolation units. Today's reference:
`verity_sampled_proofs.one_stage.registration` (`verity/registration/v1`, checks R1-R7, the receipt).

Since a wire is committed once, the region that reads a wire reads what the region that wrote it committed.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit, trace
from verity.primitives.commitments import Commitment, CommitmentScheme


@dataclass(frozen=True)
class Execution:
    circuit: str  # its id
    partition: P.Partition
    root: Commitment  # over its boundary wires' values


@dataclass(frozen=True)
class Registration:
    circuits: Mapping[str, Circuit]  # by id
    executions: tuple[Execution, ...]

    def digest(self) -> str:
        """Over the executions; their circuit ids bind the circuits."""
        body = json.dumps(
            [asdict(e) for e in self.executions], sort_keys=True, separators=(",", ":"), default=lambda b: b.hex()
        )
        return hashlib.sha256(b"verity/registration/v0\0" + body.encode()).hexdigest()


@functools.cache
def wires(circuit: Circuit, partition: P.Partition) -> tuple[int, ...]:
    """Every boundary wire of the partition, in the order the root commits them."""
    return tuple(sorted({w for region in partition for side in P.boundary(circuit, region) for w in side}))


def leaf(wire: int, value: int) -> bytes:
    return wire.to_bytes(4, "big") + value.to_bytes(4, "big")


def register(
    circuit: Circuit, partition: P.Partition, inputs: Sequence[tuple[int, ...]], scheme: CommitmentScheme
) -> Registration:
    """Run each execution once and commit to its boundary values; the traces are not kept."""
    executions = tuple(
        Execution(circuit.id(), partition, scheme.commit(_leaves(circuit, partition, trace(circuit, x))))
        for x in inputs
    )
    return Registration({circuit.id(): circuit}, executions)


def check(registration: Registration, scheme: CommitmentScheme) -> tuple[str, ...]:
    """Every circuit is the one its id names, and every root is one leaf per boundary wire under the verifier's
    scheme."""
    why = []
    for i, e in enumerate(registration.executions):
        c = registration.circuits.get(e.circuit)
        if c is None or c.id() != e.circuit:
            why.append(f"execution {i}: circuit {e.circuit} not given")
        elif e.root.scheme != scheme.name or e.root.leaves != len(wires(c, e.partition)):
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


def verify_opening(scheme: CommitmentScheme, circuit: Circuit, execution: Execution, opening: Opening) -> bool:
    index = {w: i for i, w in enumerate(wires(circuit, execution.partition))}
    return all(
        w in index and scheme.verify(execution.root, index[w], leaf(w, v), opening.paths.get(w))
        for w, v in opening.values.items()
    )


def _leaves(circuit: Circuit, partition: P.Partition, values: Sequence[int]) -> list[bytes]:
    return [leaf(w, values[w]) for w in wires(circuit, partition)]

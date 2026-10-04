"""The registration: everything a drawn unit's statement depends on, fixed before the verifier draws. Today's reference:
`verity_sampled_proofs.one_stage.registration` (`verity/registration/v1`, checks R1-R7, the receipt).

Per execution, in the order the prover committed them: its circuit, the partition it chose, and the root of a commitment
to the value of every boundary wire (one leaf per wire, in wire order). Units are numbered across executions: execution
0's regions, then execution 1's, and so on. A unit is wrong when its committed outputs are not what its region computes
from its committed inputs; since a wire is committed once, the unit that reads a wire reads what the unit that wrote it
committed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from verity.circuits import partition as P
from verity.circuits.circuit import Circuit
from verity.primitives.commitments import Commitment
from verity.subprotocols.sampled_proofs.law import Subset


@dataclass(frozen=True)
class Execution:
    circuit: str  # its id
    partition: P.Partition
    root: Commitment  # over its boundary wires' values

    @property
    def units(self) -> int:
        return len(self.partition)


@dataclass(frozen=True)
class Registration:
    executions: tuple[Execution, ...]
    law: Subset
    previous: str | None = None  # the prover's previous registration, chaining its windows

    @property
    def population(self) -> int:
        return sum(e.units for e in self.executions)

    def unit(self, u: int) -> tuple[int, int]:
        """(execution, region) of unit u."""
        for i, e in enumerate(self.executions):
            if u < e.units:
                return i, u
            u -= e.units
        raise IndexError(u)

    def digest(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=lambda b: b.hex())
        return hashlib.sha256(b"verity/registration/v0\0" + body.encode()).hexdigest()


def boundary_wires(circuit: Circuit, partition: P.Partition) -> tuple[int, ...]:
    return tuple(sorted({w for region in partition for side in P.boundary(circuit, region) for w in side}))


def leaf(wire: int, value: int) -> bytes:
    return wire.to_bytes(4, "big") + value.to_bytes(4, "big")


def check(
    registration: Registration, circuits: Mapping[str, Circuit], declared: Sequence[str], law: Subset, max_width: int, scheme: str
) -> tuple[str, ...]:
    """The verifier's checks, against its own copies: the executions are the declared ones, every circuit is the one its
    id names, every partition passes `partition.validate`, every root commits one leaf per boundary wire under the
    configured scheme, and the law is the verifier's."""
    why = []
    if [e.circuit for e in registration.executions] != list(declared):
        why.append("registration: its executions are not the declared ones")
    if registration.law != law:
        why.append(f"registration: law {registration.law}, the verifier's is {law}")
    for i, e in enumerate(registration.executions):
        c = circuits.get(e.circuit)
        if c is None or c.id() != e.circuit:
            why.append(f"execution {i}: circuit {e.circuit} not given")
            continue
        why += [f"execution {i}: {r}" for r in P.validate(c, e.partition, max_width)]
        if e.root.scheme != scheme or e.root.leaves != len(boundary_wires(c, e.partition)):
            why.append(f"execution {i}: its root is not one {scheme} leaf per boundary wire")
    return tuple(why)

"""The registration: everything a drawn unit's statement depends on, fixed before the verifier draws. Today's reference:
`verity_sampled_proofs.one_stage.registration` (`verity/registration/v1`, checks R1-R7, the receipt).

Per execution, in the order the prover committed them: its circuit, the partition it chose, and its root (`units`).
Units are numbered across executions: execution 0's regions, then execution 1's, and so on. Since a wire is committed
once, the unit that reads a wire reads what the unit that wrote it committed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from verity.primitives import partition as P
from verity.primitives.circuit import Circuit
from verity.primitives.commitments import Commitment
from verity.subprotocols.verification.units import Unit, wires


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

    @property
    def population(self) -> int:
        return sum(e.units for e in self.executions)

    def locate(self, u: int) -> tuple[int, int]:
        """(execution, region) of unit u."""
        for i, e in enumerate(self.executions):
            if u < e.units:
                return i, u
            u -= e.units
        raise IndexError(u)

    def unit(self, u: int, circuits: Mapping[str, Circuit]) -> Unit:
        i, r = self.locate(u)
        e = self.executions[i]
        return Unit(circuits[e.circuit], e.partition, r, e.root)

    def digest(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=lambda b: b.hex())
        return hashlib.sha256(b"verity/registration/v0\0" + body.encode()).hexdigest()


def check(registration: Registration, circuits: Mapping[str, Circuit], max_width: int, scheme: str) -> tuple[str, ...]:
    """The verifier's checks, against its own copies of the circuits: every execution runs one of them, every partition
    passes `partition.validate`, and every root commits one leaf per boundary wire under the verifier's scheme."""
    why = []
    for i, e in enumerate(registration.executions):
        c = circuits.get(e.circuit)
        if c is None:
            why.append(f"execution {i}: circuit {e.circuit} is not one the verifier holds")
            continue
        why += [f"execution {i}: {r}" for r in P.validate(c, e.partition, max_width)]
        if e.root.scheme != scheme or e.root.leaves != len(wires(c, e.partition)):
            why.append(f"execution {i}: its root is not one {scheme} leaf per boundary wire")
    return tuple(why)

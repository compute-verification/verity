"""What verification establishes: that a computation gave some outputs.

A claim is `circuits.consistent` on values the prover committed to before any coins were drawn: a set of gates of a
circuit, and where each wire on its boundary is committed (the commitment, and the leaf's index in it; each wire's
value is one leaf, `leaf`). An opening is those values with their paths, and `check` decides the claim from one: the
values open against the commitments, and the gates compute them.

Every construction records `Verified`: the claims it was given, and whether each holds. They differ in who sees the
opening: the prover proves the claims to the verifier (`proofs`), or trusted hardware checks them where the values
are (`replay`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.primitives.circuits import Circuit, Region, consistent
from verity.primitives.commitments import Commitment, CommitmentScheme


def leaf(bit: int) -> bytes:
    """A committed wire's leaf: its value, a bit."""
    return bytes([bit])


@dataclass(frozen=True)
class Opening:
    values: Mapping[int, int]  # boundary wire -> its committed value
    paths: Mapping[int, object]  # -> its opening against its commitment


@dataclass(frozen=True)
class Claim:
    circuit: Circuit
    gates: Region
    at: tuple[tuple[int, Commitment, int], ...]  # per boundary wire, by wire: (wire, its commitment, its leaf's index)

    def check(self, scheme: CommitmentScheme, opening: Opening) -> bool:
        values = opening.values
        return (
            set(values) == {w for w, _, _ in self.at}
            and consistent(self.circuit, self.gates, values)
            and all(scheme.verify(c, k, leaf(values[w]), opening.paths.get(w)) for w, c, k in self.at)
        )


@dataclass(frozen=True)
class Verified:
    """Each of `claims` holds if its entry in `holds` says so."""

    claims: tuple[Claim, ...]
    holds: tuple[bool, ...]

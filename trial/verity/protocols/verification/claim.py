"""What verification establishes: that a computation gave some outputs.

A claim is that a set of gates of a circuit computes, from the values committed for the wires it reads from outside,
the values committed for the wires on its boundary that it produces (`circuits.boundary`). Each of those wires' value
is committed as one leaf (`leaf`) of a commitment made before any coins were drawn, so a claim names where: the
commitment, and the leaf's index in it. An opening of a claim is its boundary values with their paths, and `check`
decides the claim from one: it opens against the commitments, and runs the gates again from the opened inputs.

Every construction records `Verified`: the claims it was given, and whether each holds. They differ in who sees the
opening: the prover proves the claims to the verifier (`proofs`), or trusted hardware checks them where the values
are (`replay`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.primitives.circuits import Circuit, Region, boundary, evaluate_region
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
        ins, outs = boundary(self.circuit, self.gates)
        if [w for w, _, _ in self.at] != sorted({*ins, *outs}) or set(opening.values) != {*ins, *outs}:
            return False
        for w, commitment, index in self.at:
            value = opening.values[w]
            if value not in (0, 1) or not scheme.verify(commitment, index, leaf(value), opening.paths.get(w)):
                return False
        produced = evaluate_region(self.circuit, self.gates, {w: opening.values[w] for w in ins})
        return all(produced[w] == opening.values[w] for w in outs)


@dataclass(frozen=True)
class Verified:
    """Each of `claims` holds if its entry in `holds` says so."""

    claims: tuple[Claim, ...]
    holds: tuple[bool, ...]

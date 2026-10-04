"""Replay: to prove a unit, open its inputs and outputs against the root; to verify, check the openings and recompute
the region from the inputs. Transparent: the verifier learns the unit's boundary values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.primitives import partition as P
from verity.primitives.commitments import CommitmentScheme
from verity.subprotocols.verification import units as U


@dataclass(frozen=True)
class Opened:
    values: Mapping[int, int]  # every boundary wire of the unit
    paths: Mapping[int, object]  # each value's opening against the root


class Replay(U.ProofSystem):
    def __init__(self, scheme: CommitmentScheme):
        self.scheme = scheme

    def prove(self, unit: U.Unit, values: Sequence[int]) -> Opened:
        ins, outs = P.boundary(unit.circuit, unit.gates)
        leaves = U.leaves(unit.circuit, unit.partition, values)
        index = U.wires(unit.circuit, unit.partition).index
        return Opened({w: values[w] for w in (*ins, *outs)}, {w: self.scheme.open(leaves, index(w)) for w in (*ins, *outs)})

    def verify(self, unit: U.Unit, proof: object) -> bool:
        if not isinstance(proof, Opened):
            return False
        ins, outs = P.boundary(unit.circuit, unit.gates)
        if set(proof.values) != {*ins, *outs}:
            return False
        index = U.wires(unit.circuit, unit.partition).index
        if not all(self.scheme.verify(unit.root, index(w), U.leaf(w, v), proof.paths[w]) for w, v in proof.values.items()):
            return False
        produced = P.evaluate_region(unit.circuit, unit.gates, {w: proof.values[w] for w in ins})
        return all(produced[w] == proof.values[w] for w in outs)

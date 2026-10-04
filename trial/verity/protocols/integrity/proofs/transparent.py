"""The transparent proof system: the witness is the proof. It opens each unit's boundary against its root, and the
verifier recomputes the region from the opened inputs. Light, and reveals the units' boundary values."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.primitives import partition as P
from verity.primitives.commitments import CommitmentScheme
from verity.primitives.registration import Opening, open_wires, verify_opening
from verity.protocols.integrity.proofs.system import ProofSystem, Unit


@dataclass(frozen=True)
class Transparent(ProofSystem):
    def prove(
        self, scheme: CommitmentScheme, units: Sequence[Unit], values: Sequence[Sequence[int]]
    ) -> tuple[Opening, ...]:
        return tuple(open_wires(scheme, u.circuit, u.partition, v, _boundary(u)) for u, v in zip(units, values))

    def verify(self, scheme: CommitmentScheme, units: Sequence[Unit], proof: object) -> bool:
        return (
            isinstance(proof, tuple)
            and len(proof) == len(units)
            and all(check(scheme, u, o) for u, o in zip(units, proof))
        )


def check(scheme: CommitmentScheme, unit: Unit, opening: Opening) -> bool:
    ins, outs = P.boundary(unit.circuit, unit.gates)
    if set(opening.values) != {*ins, *outs} or not verify_opening(
        scheme, unit.circuit, unit.partition, unit.root, opening
    ):
        return False
    produced = P.evaluate_region(unit.circuit, unit.gates, {w: opening.values[w] for w in ins})
    return all(produced[w] == opening.values[w] for w in outs)


def _boundary(unit: Unit) -> tuple[int, ...]:
    ins, outs = P.boundary(unit.circuit, unit.gates)
    return (*ins, *outs)

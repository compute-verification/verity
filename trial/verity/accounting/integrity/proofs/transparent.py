"""The transparent proof system: the witness is the proof. It opens each unit's boundary where it is committed, and the
verifier recomputes the region from the opened inputs. Light, and reveals the units' boundary values."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.accounting.integrity.proofs.system import ProofSystem, Unit
from verity.primitives import partition as P
from verity.primitives.commitments import CommitmentScheme
from verity.primitives.registration import Opening, open_wires, verify_opening


@dataclass(frozen=True)
class Transparent(ProofSystem):
    def prove(
        self, scheme: CommitmentScheme, units: Sequence[Unit], transcripts: Sequence[Sequence[int]]
    ) -> tuple[Opening, ...]:
        return tuple(open_wires(scheme, u.stream, transcripts, _located(u).values()) for u in units)

    def verify(self, scheme: CommitmentScheme, units: Sequence[Unit], proof: object) -> bool:
        return (
            isinstance(proof, tuple)
            and len(proof) == len(units)
            and all(check(scheme, u, o) for u, o in zip(units, proof))
        )


def check(scheme: CommitmentScheme, unit: Unit, opening: Opening) -> bool:
    at = _located(unit)
    if set(opening.values) != set(at.values()) or not verify_opening(scheme, unit.stream, opening):
        return False
    ins, outs = P.boundary(unit.circuit, unit.gates)
    produced = P.evaluate_region(unit.circuit, unit.gates, {w: opening.values[at[w]] for w in ins})
    return all(produced[w] == opening.values[at[w]] for w in outs)


def _located(unit: Unit) -> dict[int, tuple[int, int]]:
    """Each boundary wire of the unit, and where it is committed."""
    ins, outs = P.boundary(unit.circuit, unit.gates)
    return {w: unit.stream.locate(unit.execution, w) for w in (*ins, *outs)}

"""What the prover and the verifier send each other, and the profile the verifier records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.subprotocols.sampled_proofs.registration import Registration


@dataclass(frozen=True)
class Registered:
    """Prover to verifier: its computation as units, and its commitment to every unit's inputs and outputs."""

    registration: Registration


@dataclass(frozen=True)
class Draw:
    """Verifier to prover: the units it must prove."""

    units: tuple[int, ...]


@dataclass(frozen=True)
class Opening:
    values: Mapping[int, int]  # every boundary wire of the unit: inputs and outputs
    proofs: Mapping[int, object]  # each value's opening against the execution's root
    witness: object  # the construction's: nothing for replay, a proof for a proof system


@dataclass(frozen=True)
class Opened:
    """Prover to verifier: each drawn unit, opened and proved."""

    openings: Mapping[int, Opening]


@dataclass(frozen=True)
class IntegrityProfile:
    """The verifier's record: if `accepted`, at most `wrong_units` of the registration's `units` are wrong, except with
    probability `delta`."""

    registration: str  # its digest
    accepted: bool
    units: int
    wrong_units: int
    delta: float
    reasons: tuple[str, ...]

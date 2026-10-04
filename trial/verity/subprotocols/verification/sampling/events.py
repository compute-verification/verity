"""What the prover and the verifier send each other, and the profile the verifier records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from verity.subprotocols.verification.sampling.registration import Registration


@dataclass(frozen=True)
class Registered:
    """Prover to verifier: its computation as units, and its commitment to every unit's inputs and outputs."""

    registration: Registration


@dataclass(frozen=True)
class Draw:
    """Verifier to prover: the units it must prove."""

    units: tuple[int, ...]


@dataclass(frozen=True)
class Proved:
    """Prover to verifier: a proof of each drawn unit."""

    proofs: Mapping[int, object]


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

"""What every integrity construction establishes, whatever scale it samples at, and what accounting reads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrityProfile:
    """If `accepted`: of the registration's `units` at `scale`, at most `wrong_units` are wrong, except with probability
    `delta`."""

    registration: str  # its digest
    scale: str  # what a unit is: "execution", "region"
    accepted: bool
    units: int
    wrong_units: int
    delta: float
    reasons: tuple[str, ...]

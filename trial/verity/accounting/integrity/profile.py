"""What every integrity construction establishes, whatever scale it samples at, and what coverage reads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrityProfile:
    """If `accepted`: of the stream's `units` at `scale`, at most `wrong_units` are wrong, and so at most `wrong_gates`
    of its `gates` (the most the heaviest `wrong_units` units hold), except with probability `delta`."""

    stream: str  # its digest
    scale: str  # what a unit is: "execution", "region"
    accepted: bool
    units: int
    wrong_units: int
    gates: int
    wrong_gates: int
    delta: float
    reasons: tuple[str, ...]


def profile(
    stream: str, scale: str, sizes: list[int], wrong_units: int, delta: float, reasons: tuple[str, ...]
) -> IntegrityProfile:
    """The profile of a stream whose units hold `sizes` gates each: accepted if nothing is wrong with it."""
    wrong_gates = sum(sorted(sizes, reverse=True)[:wrong_units])
    return IntegrityProfile(
        stream, scale, not reasons, len(sizes), wrong_units, sum(sizes), wrong_gates, delta, reasons
    )

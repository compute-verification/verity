"""What every integrity construction establishes, whatever scale it samples at, and what coverage reads: a bound on
how many units are wrong, and so on how much of the stream's work and output they could hold."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Count:
    total: int
    wrong: int  # the most of `total` in wrong units


@dataclass(frozen=True)
class IntegrityProfile:
    """If `accepted`: of the stream's units at `scale`, at most `units.wrong` are wrong, and they hold at most
    `gates.wrong` of its gates and `outputs.wrong` of its output bits (the most the heaviest wrong units could hold
    of each), except with probability `delta`."""

    stream: str  # its digest
    scale: str  # what a unit is: "execution", "region"
    accepted: bool
    units: Count
    gates: Count
    outputs: Count
    delta: float
    reasons: tuple[str, ...]


def profile(
    stream: str,
    scale: str,
    gates: Sequence[int],
    outputs: Sequence[int],
    wrong: int,
    delta: float,
    reasons: tuple[str, ...],
) -> IntegrityProfile:
    """The profile of a stream whose units hold `gates` gates and `outputs` output bits each, at most `wrong` of them
    wrong: accepted if there are no `reasons`."""

    def count(sizes: Sequence[int]) -> Count:
        return Count(sum(sizes), sum(sorted(sizes, reverse=True)[:wrong]))

    return IntegrityProfile(
        stream, scale, not reasons, Count(len(gates), wrong), count(gates), count(outputs), delta, reasons
    )

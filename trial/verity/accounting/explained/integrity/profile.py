"""What every integrity construction establishes, whatever scale it samples at, and what coverage reads: how much of
the stream's work and output could be in wrong units."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.accounting.explained.integrity.sampling import Weighted


@dataclass(frozen=True)
class Count:
    total: int
    wrong: int  # the most of `total` in wrong units


@dataclass(frozen=True)
class IntegrityProfile:
    """If `accepted`: of the stream's gates, at most `gates.wrong` are in wrong units at `scale`, and of its output bits
    at most `outputs.wrong`, except with probability `delta`. A unit is wrong when the outputs committed for it are not
    what its gates compute from the inputs committed for it."""

    stream: str  # its digest
    scale: str  # what a unit is: "execution", "region"
    accepted: bool
    units: int
    gates: Count
    outputs: Count
    delta: float
    reasons: tuple[str, ...]


def profile(
    stream: str,
    scale: str,
    gates: Sequence[int],
    outputs: Sequence[int],
    law: Weighted,
    delta: float,
    reasons: tuple[str, ...],
) -> IntegrityProfile:
    """The profile of units holding `gates` gates and `outputs` output bits each, drawn under `law`: accepted if no
    check failed (no `reasons`), with the law's bound; and if one did, everything could be wrong."""
    wrong = law.bound(gates, outputs, delta) if not reasons else (sum(gates), sum(outputs))
    return IntegrityProfile(
        stream,
        scale,
        not reasons,
        len(gates),
        Count(sum(gates), wrong[0]),
        Count(sum(outputs), wrong[1]),
        delta,
        reasons,
    )

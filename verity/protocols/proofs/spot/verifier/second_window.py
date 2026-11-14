"""Round 2's reader over the joined record and its ranks' check, as the verifier's `Spot.SecondWindow` and `Spot.Ranks`
define them: a two-stage run's second-stage sessions held to the joined circuit's record (its ``ranks``, ``wires`` and
``boundary_reads``, `verity.commitments.windows.interface`) and to each rank's registration, with no interior
registration. The rule, the codes and why the interior registration's checks go are the Lean's (its module docstrings).

Nothing is built yet: each function raises naming the check it is to be, and nothing calls one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

#: the ranks' checks' codes, in the order they run (`Spot.Ranks.CODES`)
RANK_CODES = ("ranks-shape", "ranks-window", "ranks-run", "ranks-members", "ranks-units", "ranks-wires", "ranks-reads")
#: `Spot.SecondStage`'s per-session codes this reader reuses, read against a rank member (`Spot.SecondWindow.SESSION_CODES`)
SESSION_CODES = ("second-header", "second-population", "second-roots", "second-sample", "second-read", "second-values",
                 "second-refs", "second-tables", "second-verified")


def member_of(ranks: Mapping[str, Any], registrations: Sequence[Mapping[str, Any]], template: str,
              unit: int) -> tuple[int, str, int, int, int] | None:
    """`Spot.Ranks.memberOf`: the rank member ``(rank, template, j, base, units)`` of ``registrations`` that holds unit
    ``unit`` of ``template`` in the joined circuit."""
    raise NotImplementedError("Spot.Ranks.memberOf: the unit-to-member rule over the ranks' registrations")


def rank_failures(record: Mapping[str, Any], records: Sequence[Mapping[str, Any]], served: Sequence[tuple[str, str]],
                  registrations: Sequence[Mapping[str, Any]], root: str,
                  logged: Callable[[str], bool]) -> list[tuple[str, Any]]:
    """`Spot.Ranks.failures`: ``ranks-shape``, ``ranks-window``, ``ranks-run``, ``ranks-members``, ``ranks-units``,
    ``ranks-wires`` and ``ranks-reads`` of the joined ``record`` against its ranks' records, served records and
    registrations at the serve's ``root``."""
    raise NotImplementedError("Spot.Ranks.failures: the ranks' check")


def failures(allow: bool, record: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
             served: Sequence[tuple[str, str]], registrations: Sequence[Mapping[str, Any]], root: str,
             logged: Callable[[str], bool], sample: Sequence[int],
             statements: Sequence[Mapping[str, Any]]) -> list[tuple[str, Any]]:
    """`Spot.SecondWindow.failures`: the ranks' failures, ``ranks-wire-tree`` and the per-session codes against rank
    members, then ``two-stage-form`` while the form is off."""
    raise NotImplementedError("Spot.SecondWindow.failures: round 2's check over the joined record")

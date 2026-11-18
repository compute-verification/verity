"""The one circuit at the served verifier's door: C_S, which the door derives from C (the registered Program), the
partition and the block's units (`restrict`), and the holes it does not yet derive, each refusing by name
(`NotDerived`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity.circuits import restrict as R

PREFIX = "one-circuit: not derived yet: "

RESTRICT = "restrict : C -> Partition -> Block -> Circuit (C_S, the lowering of the block's Definition in C)"

LOWER = "lower : C_S -> the base argument's circuit (Flock's verifier takes a Boolean block circuit and lowers it to rows)"

Refused = R.Refused


class NotDerived(ValueError):
    """A derivation from C and the partition that is not written yet (`Flock.OneCircuit.notDerived`)."""

    def __init__(self, what: str) -> None:
        super().__init__(PREFIX + what)


def restrict(c: bytes, partition: Mapping[str, Any], units: Sequence[tuple[int, int]]) -> bytes:
    """C_S (`OneStageZk.restrict`): the descriptor bytes of C compacted to the block of ``units`` (ascending ``[lo, hi)``
    ranges of the partition's units), Core's `Circuit.restrict` (`restrict.restrict`); `Refused` where C, the partition
    or the units give no block."""
    return R.restrict(c, partition, [tuple(u) for u in units]).descriptor


def lower(c_s: bytes) -> bytes:
    """The base argument's circuit of C_S, the Boolean block circuit (`restrict`): the lowering to rows is the base
    argument's, and Flock's verifier takes no Boolean block circuit yet."""
    raise NotDerived(LOWER)

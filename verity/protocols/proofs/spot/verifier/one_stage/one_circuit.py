"""The holes of the one circuit: what the served verifier is to derive from C (the registered Program) and the partition,
and does not yet. Each refuses by name (`NotDerived`), with the text its Lean twin refuses with."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PREFIX = "one-circuit: not derived yet: "

RESTRICT = "restrict : C -> Partition -> Block -> Circuit (C_S, the lowering of the block's Definition in C)"


class NotDerived(ValueError):
    """A derivation from C and the partition that is not written yet (`Flock.OneCircuit.notDerived`)."""

    def __init__(self, what: str) -> None:
        super().__init__(PREFIX + what)


def restrict(c: bytes, partition: Mapping[str, Any], block: int) -> bytes:
    """C_S (`OneStageZk.restrict`): the circuit file of the proof block whose first unit is `block`, the Boolean
    lowering of its Definition in C, its ports in(S) and out(S): Core's `Circuit.restrict`."""
    raise NotDerived(RESTRICT)


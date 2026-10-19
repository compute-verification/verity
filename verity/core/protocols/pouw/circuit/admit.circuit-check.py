"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.admit`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwQueueAdmit_v2`, a call's 16 slots admitted from the committed request queue on bits (49 ANDs a slot): static-free,
    so a catalog root anyway, and here so it is checked whatever the catalog holds.  Its word, `PouwQueueAdmit_v1`, is a
    primitive, a catalog root."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import admit as A

    return [bind(A.PouwQueueAdmit)]

"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.admit`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwQueueAdmit`, a call's 16 slots admitted from the committed request queue, on words and on bits (49 ANDs a slot):
    static-free, so catalog roots anyway, and here so each is checked whatever the catalog holds."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import admit as A

    return [bind(A.PouwQueueAdmitWord), bind(A.PouwQueueAdmit)]

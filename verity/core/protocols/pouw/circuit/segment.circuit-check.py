"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.segment`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The checkpointed spot check under duo at every spacing the slice measures (S = 1, 4, 8, 16, 32), under XOR-in at S = 1,
    8 and 32, and with decision 29's dither on the served side at S = 8."""
    from verity.core.protocols.pouw.circuit import segment as SG
    from verity.ml.circuits.defs import bind

    return ([bind(SG.PouwSegCheck, S=s, FOLD="duo") for s in (1, 4, 8, 16, 32)]
            + [bind(SG.PouwSegCheck, S=s, FOLD="xin") for s in (1, 8, 32)] + [bind(SG.PouwSegCheckDither, S=8, FOLD="duo")])

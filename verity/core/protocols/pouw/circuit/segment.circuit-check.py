"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.segment`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The checkpointed spot check under duo at the spacings in use (S = 1, 4, 8, 16; S = 32 isn't adopted, and its lowering
    alone takes ~650 GB), under XOR-in at S = 1 and 8, and with decision 29's dither on the served side at S = 8."""
    from verity.core.protocols.pouw.circuit import segment as SG
    from verity.ml.circuits.defs import bind

    return ([bind(SG.PouwSegCheck, S=s, FOLD="duo") for s in (1, 4, 8, 16)]
            + [bind(SG.PouwSegCheck, S=s, FOLD="xin") for s in (1, 8)] + [bind(SG.PouwSegCheckDither, S=8, FOLD="duo")])

"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.segment`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The per-segment spot check at the e2e slice's k = 4,096 (128 segments) and at 4 segments, the smallest receipt row
    with a segment that is neither first nor last."""
    from verity.core.protocols.pouw.circuit import segment as SG
    from verity.ml.circuits.defs import bind

    return [bind(SG.PouwSegCheck, KS=4), bind(SG.PouwSegCheck, KS=128)]

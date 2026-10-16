"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.segment`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The per-segment spot check, with F ±1 and with decision 29's dither on the served side, at the e2e slice's k = 4,096
    (128 segments) and at 4 segments, the smallest receipt row with a segment that is neither first nor last."""
    from verity.core.protocols.pouw.circuit import segment as SG
    from verity.ml.circuits.defs import bind

    return [bind(D, KS=ks) for D in (SG.PouwSegCheck, SG.PouwSegCheckDither) for ks in (4, 128)]

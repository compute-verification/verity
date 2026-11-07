"""circuit-check's bindings of `verity.ml.protocols.pouw.circuit.segment`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """The spot checks at their smallest shapes, every primitive they call with them: `PouwSegCheck_v2` at S = 1 (duo, and
    xin dithered); `PouwSegCheck_v3` at two segments a window each (duo) and one window of two (xin, dithered);
    `PouwCallCheck_v1` at KS = 2 (one digit word) on duo's chain words, on fx1's full state and on fx1's chain words (the
    EPS1 sm_120 scheme's spot check, `circuit.eps1`); and `PouwWarpCheck_v1` (li) and `PouwWarpCheck_v2` (duo, fx1) at
    KS = 2 over two lanes. Each grows linearly in KS (and NL), which the served shapes are priced by."""
    from verity.ml.circuits.defs import bind
    from verity.ml.protocols.pouw.circuit import segment as SG

    return [bind(SG.PouwSegCheck, S=1, FOLD="duo"),
            bind(SG.PouwSegCheckDither, S=1, FOLD="xin"),
            bind(SG.PouwSegCheckV3, KS=2, S=1, FOLD="duo"),
            bind(SG.PouwSegCheckDitherV3, KS=2, S=2, FOLD="xin"),
            bind(SG.PouwCallCheck, KS=2, FOLD="duo", OUT="chains"),
            bind(SG.PouwCallCheck, KS=2, FOLD="fx1", OUT="state"),
            bind(SG.PouwCallCheck, KS=2, FOLD="fx1", OUT="chains"),
            bind(SG.PouwWarpCheck, KS=2, NL=2),
            bind(SG.PouwWarpCheckV2, KS=2, NL=2, FOLD="duo"),
            bind(SG.PouwWarpCheckV2, KS=2, NL=2, FOLD="fx1")]

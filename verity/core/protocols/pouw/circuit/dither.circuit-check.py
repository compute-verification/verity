"""circuit-check's bindings of `verity.core.protocols.pouw.circuit.dither`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _pouw_roots() -> list[Any]:
    """`PouwDitherBlock_v1` at L = 32 (one `PouwSegDot0_v2`), 64 (then one `PouwSegDot_v2`) and the served default 1,024,
    each first (`PouwDitherFirst_v2`, no t0) and later (`PouwDitherStep_v2`), each operand formed by `PouwI8Add_v2`; and
    `PouwDitherLast_v1` at K = 64 as one segment (`PouwDitherOnlyOk_v2`, O alone) and at K = 96 over L = 64, two
    segments, the last short (`PouwDitherLastOk_v2`, X W′ᵀ over the first segment, T_S being O's), O's sum through
    `PouwDitherOutDot0_v2` and `PouwDitherOutDot_v2`. Both grow linearly in their lengths / 32, which larger L and K are
    priced by."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import dither as D

    return ([bind(D.PouwDitherBlock, L=L, FIRST=first) for L in (32, 64, 1024) for first in (True, False)]
            + [bind(D.PouwDitherLast, K=64, L=64), bind(D.PouwDitherLast, K=96, L=64)])

"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.quant`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The bindings `ops.quant`'s sidecar checks at H = 128, on bits. The trace holds 4-7 KB a gate
    (``QuantRowS8_v2{128}`` peaks at 4.9 GB and 210 s in circuit-check, ``{512}`` at 18 GB and 1,640 s), so the wider rows
    are checked as words; and a chunk's two steps, the scale and a code."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import quant as BQ

    return ([bind(BQ.QuantRowS8, N=128, RELU=relu) for relu in (True, False)]
            + [bind(BQ.RowAbsMax, N=128, RELU=True), bind(BQ.RowAbsMax, N=4, RELU=False),
               bind(BQ.QuantChunkS8, N=128, RELU=True), bind(BQ.QuantScaleS8), bind(BQ.QuantCodeS8)])

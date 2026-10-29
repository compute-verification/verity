"""circuit-check's bindings of `verity.ml.circuits.ops.quant`'s Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    """Phase 1: the row proof block at H = 128, with and without the ReLU, and at 512; the amax of a row of 128
    and over four partial maxima; a chunk of 128. Phase 2, each in a form whose proof block fits a Flock block of 2^27:

    * a 16,384-wide row without the ReLU (an RMSNorm's or attention's output): its amax, and the two chunks of 8,192;
    * the MLP's 65,536-wide hidden row, relu'd: in four 32 KiB rows, its amax over them and the four chunks of 16,384; in 256-byte
      rows, three chunks' amaxes, the row's from the last chunk and those three (16,387 values), and the eight chunks of 8,192."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import quant as Q

    return ([bind(Q.QuantRowS8, N=128, RELU=relu) for relu in (True, False)] + [bind(Q.QuantRowS8, N=512, RELU=True)]
            + [bind(Q.RowAbsMax, N=128, RELU=True), bind(Q.RowAbsMax, N=4, RELU=False), bind(Q.QuantChunkS8, N=128, RELU=True)]
            + [bind(Q.RowAbsMax, N=16384, RELU=False), bind(Q.QuantChunkS8, N=8192, RELU=False)]
            + [bind(Q.RowAbsMax, N=65536, RELU=True), bind(Q.QuantChunkS8, N=16384, RELU=True)]
            + [bind(Q.RowAbsMax, N=n, RELU=True) for n in (16384, 16387)] + [bind(Q.QuantChunkS8, N=8192, RELU=True)])

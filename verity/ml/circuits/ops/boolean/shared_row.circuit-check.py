"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.shared_row`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The writers table's shapes (FAMILIES' 1 KiB layout, in words): the check and the select at each segment shape of a 1 KiB
    row (512 of 1 word, 16 of 32, 8 of 64, 4 of 128, the sums' 4 of 64 at words 0 and 256), of a 512 B row (256 of 1, 4 of 64)
    and of K and V's 128 B row (64 of 1), and in the amax rows of 2 and 5 words (one word at 0, 1 or 4, 3 at 1); the writer on
    each family, at its row, offset and G, on the served statics where the family's member is under `targets.MAX_GATES` (the s8
    coordinate on two rows at K = 64 in its three rows, the embedding's coordinate at V = 256, the argmax at V = 256) and on a
    smaller member of it otherwise; and the readers: the norm's sums at words 0 and 256 of the pooled sums row and its x from
    a residual row, the residual's a and b, PV's p from the softmax row and the softmax's q from the q row, each inside its
    writer, and a quant chunk's amax word from its pooled amax rows (its codes and scale are its row's every word, the
    block's own output row, so it has no writer)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.ops.boolean import quant as BQ
    from verity.ml.circuits.ops.boolean import rmsnorm as RN
    from verity.ml.circuits.ops.boolean import rows as BRW
    from verity.ml.circuits.ops.boolean import rowwise as BR
    from verity.ml.circuits.ops.boolean import shared_row as SR
    from verity.ml.circuits.ops.boolean import softmax_row as SM

    def window(fn, ROW, OFF, G):
        return bind(SR.SharedRowWindow, FN=fn, ROW=ROW, OFF=OFF, G=G)

    def read(fn, port, ROW, OFF, G):
        return bind(SR.SharedRowRead, FN=fn, PORT=port, ROW=ROW, OFF=OFF, G=G)

    s8 = bind(BG.ScaledMmS8CoordinateRows, K=64, R=2, **BG.GEMM_SM120_S8)
    norm = bind(RN.RmsNormOutSums, C=8, S=8, G=2, N=64, EPS=1e-5)
    add = bind(BRW.Bf16AddRow, N=16)
    quant = bind(BQ.QuantChunkS8, N=8, RELU=False)
    return [
        *(bind(SR.SharedRowCheck, W=W, ROW=ROW, OFF=OFF, G=G)
          for W, ROW, OFF, G in ((1, 512, 0, 512), (32, 512, 0, 16), (64, 512, 0, 8), (128, 512, 0, 4), (64, 512, 0, 4),
                                 (64, 512, 256, 4), (1, 256, 0, 256), (64, 256, 0, 4), (1, 64, 0, 64), (1, 2, 0, 1),
                                 (1, 2, 1, 1), (1, 5, 0, 1), (1, 5, 1, 3), (1, 5, 4, 1))),
        *(bind(SR.SharedRowSelect, W=W, ROW=512, OFF=0, G=G) for W, G in ((128, 4), (32, 16), (64, 8))),
        window(s8, 512, 0, 512),
        window(s8, 64, 0, 64),
        window(s8, 256, 0, 256),
        window(bind(BR.EmbeddingCoordinate, V=256), 512, 0, 512),
        window(bind(BRW.Argmax, V=256), 2, 1, 1),
        window(bind(RN.RmsSumSqChunks, C=256, K=128), 512, 256, 4),
        window(bind(BQ.RowAbsMax, N=8, RELU=True), 5, 1, 3),
        window(read(read(norm, "s", 512, 256, 1), "x", 512, 0, 4), 512, 0, 4),
        window(read(read(norm, "s", 512, 0, 1), "x", 512, 0, 4), 512, 0, 4),
        window(read(read(add, "a", 512, 0, 4), "b", 512, 0, 4), 512, 0, 4),
        window(read(bind(BG.PvRow, T=2, K=16, D=4, **BG.GEMM_HOPPER), "p", 512, 0, 16), 512, 0, 8),
        window(read(bind(SM.QkSoftmaxRowPad, T=2, D=16, **BG.GEMM_HOPPER), "q", 512, 0, 8), 512, 0, 16),
        read(quant, "amax", 2, 1, 1),
        read(quant, "amax", 5, 4, 1),
    ]

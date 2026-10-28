"""cuBLAS's bf16 GEMM of one token row on cc 12.0 with 188 SMs: `torch.mm(x, W.T)` at M = 1 launches cuBLAS 12.9.1's
`internal::gemvx::kernel` (bf16 operands, f32 compute), a block of LANES x 4 threads per four W rows, LANES 16 at its template's
log-block 6 and 32 at 7, which cuBLAS picks by (K, N):

    CublasGemvDot_v1{K, LANES}       (x, w: Array<K, BF16>) -> BF16
        lane t of LANES runs one f32 chain from +0 over k = t, t + LANES, ... (ascending), each step one FFMA of the widened words;
        the lanes meet in a butterfly (offsets LANES/2, ..., 1: lane i adds lane i + o), and lane 0's sum is added to +0 (so a -0
        sum leaves as +0) and rounded to bf16 (`F2fpBf16`)
    CublasGemv_v1{K, N, LANES}       (x: Array<K, BF16>, w: Array<N, Array<K, BF16>>) -> Array<N, BF16>
        the dot on each W row

on words (`Bf16ToF32_v1`, `F32Fma_v1`, `F32Add_v1`, `F2fpBf16_v1`), and each one's v2 the same function on bits (its ``word`` the v1,
so `boolean.boolean_version` finds it): the chain's steps are `ll_bf16.LL_FMA_STEP`, the rows' taking each step of every W row at
once, x's own gates once, then `F32Add_v3` for the butterfly and the +0 and `F2fpBf16_v2`.  `bf16_gemm_check.cublas_gemv_rows` is
the numpy twin.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import cublas_bf16 as CB
from verity.ml.definitions.registry import ll_bf16 as LL

__all__ = ["CublasGemv", "CublasGemvDot", "CublasGemvDotWord", "CublasGemvWord"]

BF16 = Array(16, BL.BIT)
CONFORMANCE = CB.conformance_cc12("`internal::gemvx::kernel` (M 1: LANES 16 at K/N 2048/{2048,3072,16384,128256} and "
                                  "4096/{4096,6144,151936}, 32 at 4096/24576 and 12288/4096)")


def _check(S) -> None:
    bad = [k for k, v in {"LANES_power_of_two_to_32": S.LANES in (1, 2, 4, 8, 16, 32),
                          "K_positive_multiple_of_LANES": S.K > 0 and S.K % S.LANES == 0}.items() if not v]
    if bad:
        raise ValueError(f"CublasGemv: unsupported statics ({', '.join(bad)})")


def _butterfly(add, lanes):
    o = len(lanes) // 2
    while o:
        lanes = [add(lanes[i], lanes[i + o]) for i in range(o)]
        o //= 2
    return lanes[0]


def _dot_word(B, S, x, w):
    _check(S)
    xf, wf = (B.batch(P.Bf16ToF32, a, axes=(0,)) for a in (x, w))
    lanes = []
    for t in range(S.LANES):
        acc = B.call(P.ZERO32)
        for k in range(t, S.K, S.LANES):
            acc = B.call(W.F32Fma, xf[k], wf[k], acc)
        lanes.append(acc)
    total = _butterfly(lambda a, b: B.call(W.F32Add, a, b), lanes)
    return B.call(P.F2fpBf16, B.call(W.F32Add, B.call(P.ZERO32), total))


def _bits(rows: bool):
    def body(B, S, x, w):
        _check(S)
        lanes = []
        for t in range(S.LANES):
            acc = None
            for k in range(t, S.K, S.LANES):
                acc = LL.LL_FMA_STEP(B, acc, x[k], w.take(1, k) if rows else w[k], rows)
            lanes.append(acc)
        add = LL._adder(B, rows)
        out = add(LL._bits_zero(B, S.N if rows else None), _butterfly(add, lanes))
        return B.batch(BG.F2fpBf16, out, axes=(0,)) if rows else B.call(BG.F2fpBf16, out)
    return body


def _rows(B, S, x, w):
    return B.batch(bind(CublasGemvDotWord, K=S.K, LANES=S.LANES), x, w, axes=(None, 0))


def _dot_type(X):
    return lambda S: ((("x", Array(S.K, X)), ("w", Array(S.K, X))), X)


def _gemm_type(X):
    return lambda S: ((("x", Array(S.K, X)), ("w", Array(S.N, Array(S.K, X)))), Array(S.N, X))


DOC = ("cuBLAS's M = 1 gemvx on cc 12.0: LANES f32 FFMA chains from +0 over k = t, t + LANES, ..., the lanes' butterfly added to "
       "+0, one bf16 rounding")
ROWS_DOC = "torch.mm of one bf16 token row on cc 12.0: cuBLAS's gemvx dot on each W row"

CublasGemvDotWord = CompositeDefinition("CublasGemvDot", 1, ("K", "LANES"), _dot_type(P.BF16), _dot_word, doc=DOC,
                                        conformance=CONFORMANCE)
CublasGemvWord = CompositeDefinition("CublasGemv", 1, ("K", "N", "LANES"), _gemm_type(P.BF16), _rows, doc=ROWS_DOC,
                                     conformance=CONFORMANCE)
CublasGemvDot = CompositeDefinition("CublasGemvDot", 2, ("K", "LANES"), _dot_type(BF16), _bits(False), doc=f"{DOC} (on bits)",
                                    conformance=CONFORMANCE)
CublasGemv = CompositeDefinition("CublasGemv", 2, ("K", "N", "LANES"), _gemm_type(BF16), _bits(True),
                                 doc=f"{ROWS_DOC} (on bits: each step over every W row at once, x's own gates once)",
                                 conformance=CONFORMANCE)
CublasGemvDot.word = lambda b: bind(CublasGemvDotWord, K=b["K"], LANES=b["LANES"])  # type: ignore[attr-defined]
CublasGemv.word = lambda b: bind(CublasGemvWord, K=b["K"], N=b["N"], LANES=b["LANES"])  # type: ignore[attr-defined]

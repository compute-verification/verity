"""The served RMSNorms on bits: `RMSNormFusedCuda_v2` and `RMSNormTriton_v1` as Boolean Definitions, cut into Calls a root can hold.

    RMSNormFusedCuda_v3{N, EPS, RSQRT}   (x, res, w: Array<N, Array<16, Value<1>>>) -> (normed, res')    word view: RMSNormFusedCuda_v2{N, EPS}
    RMSNormTriton_v2{N, EPS, SQRT, SCALEA, RCP}
                                         (x, w: Array<N, Array<16, Value<1>>>) -> Array<N, Array<16, ...>>  word view: RMSNormTriton_v1{N, EPS}

Each is the word body's dataflow, in the word body's operand order, over these parts (`rmsnorm_cuda_calls`, `rmsnorm_triton_calls`):

    RMSNormLanesCuda_v1{N, BLOCK}  per thread t < T = min(N, BLOCK): r = Bf16AddF2fp_v2(x, res) for its elements t, t + T, ...
                                   (`RMSNormLaneCuda_v1{J}`), partial = square of the first, then FFMA of each next square
    RMSNormWarpsCuda_v1{T}         per warp, cub's shfl_down tree 1, 2, 4, 8, 16 over the valid lanes (`RMSNormWarpCuda_v1{L}`)
    RMSNormArgCuda_v1{N, EPS, NW}  the warp sums added from warp 0, `/ N` (div.rn), `+ EPS`
    RSQRT                          `rsqrtf` (`RsqrtApprox_v1`'s function)
    RMSNormOutRowCuda_v1{N}        bf16_rn((f32(r_j) * s) * f32(w_j))   (`RMSNormOutCuda_v2`)

    RMSNormThreadsTriton_v1{N}     per thread of 8 columns: x1 * x1, then FFMA x0, x2, ..., x7  (`RMSNormThreadTriton_v2`)
    RMSNormWarpsTriton_v1{N}       per warp, the xor butterfly 16, 8, 4, 2, 1, lane 0's cone (`RMSNormWarpTriton_v1{L}`)
    RMSNormMeanEpsTriton_v1{N, EPS}  per chunk of 1024 columns (w0 + w2) + (w1 + w3), chunks added in order, FFMA(rcp(N), sum, EPS)
    SQRT, SCALEA(1.0, .), RCP      `sqrt.approx.ftz` and `div.full`'s two MUFU pieces; `F32Mul_v3` of the last two is `inv`
    RMSNormOutRowTriton_v1{N}      bf16_rn((inv * f32(x_j)) * f32(w_j))   (`RMSNormOutTriton_v2`)

The leaf pieces (`RMSNormSquare_v2`, `RMSNormSquareFma_v2`, `RMSNormThreadTriton_v2`, `RMSNormOutCuda_v2`, `RMSNormOutTriton_v2`) are
traced from `fp`'s builders (`trace.definition`), so the bf16 operands' 16 constant-zero bits specialize them; each names a word
composite of the `_v1` primitives (`RMSNormSquare_v1`, ...) as its word Definition. The f32 adds, divide, multiply and FMA between
them are `verity.ml.boolean.elementwise`'s Calls. The lanes, warps and output rows are separable (each node reads only parameters
and every leaf it makes is returned), so `Q_word` cuts them member by member; the norm itself is not, so a served Program calls the
parts at its root (the two emitters above), never the whole norm, whose flat circuit is millions of gates at a served width.

MUFU: ``RSQRT``, ``SQRT``, ``SCALEA`` and ``RCP`` are statics, Definitions on bits whose word Definitions are `RsqrtApprox_v1`,
`MufuSqrtFtz_v1`, `DivFullScaleA_v1` (or `_v2`, equal on the numerator 1.0) and `DivFullRcp_v1`. Until proofs' Boolean MUFU
Definitions exist they are bound to the word ids through `WordOnBits_v1{WORD}` (`MUFU_WORD`), which packs the bits into a word, calls WORD
and unpacks it: those Calls are the only word gates left in a norm. `rcp(N)` is a constant of the specialization, folded when it
is built (`DivFullRcp_v1` of the word N), as is every other constant.

Triton's idle lanes and warps (columns past N) hold +0, and the word body adds them. A lane's partial is never -0 and never a
signalling NaN (a square's sign is clear, an FFMA's NaN is 0x7FC00000, an FADD quiets its NaN), so adding such a +0 returns the
other operand unchanged under `F32Add_v1`: those adds are not emitted.
"""
from __future__ import annotations

import numpy as np

from verity.evaluation import evaluate
from verity.ir import boolean as BL
from verity.ir.defs import CompositeDefinition, bind, composite
from verity.ir.refs import Coll, Explicit, array_of, coll_concat, tuple_of
from verity.ir.types import Array, Tuple, Value
from verity.ml import fp32 as W
from verity.ml.boolean import elementwise as EW
from verity.ml.boolean import fp
from verity.ml.boolean.trace import definition, word_of
from verity.ml.prims import const
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.sampling import bit_at

__all__ = ["CUDA_WORD", "MUFU_WORD", "RMSNormFusedCuda", "RMSNormTriton", "TRITON_WORD", "WordOnBits", "rmsnorm_cuda_calls",
           "rmsnorm_triton_calls"]

BIT = BL.BIT
F32 = Array(32, BIT)
BF16 = Array(16, BIT)
F32W, BF16W = Value(32), Value(16)

#: lanes per warp, columns per Triton thread, Triton threads per chunk
WARP, TRITON_COLS, TRITON_THREADS = 32, 8, 128


def _f32(x: float) -> int:
    return P.bits_of(np.float32(x))


class _Consts:
    """Constant words in one body, from the two constant bits (each emitted once, when first read)."""

    def __init__(self, B) -> None:
        self.B, self.bit = B, {}

    def __call__(self, value: int, width: int = 32) -> Coll:
        refs = []
        for j in range(width):
            b = (value >> j) & 1
            if b not in self.bit:
                self.bit[b] = self.B.call(BL.constant(b)).refs[0]
            refs.append(self.bit[b])
        return Coll(Array(width, BIT), Explicit(tuple(refs)))


# --- the MUFU seam ------------------------------------------------------------------------------------------------------------

def _word_on_bits_sig(S):
    params, ret = S.WORD.signature()
    if ret != F32W or any(t != F32W for _, t in params):
        raise TypeError(f"WordOnBits: {S.WORD.id} must take and return 32-bit words")
    return tuple((name, F32) for name, _ in params), F32


@composite("WordOnBits", 1, ["WORD"], _word_on_bits_sig, conformance="exact-model-tested")
def WordOnBits(B, S, *args):
    """The 32-bit word Definition ``WORD`` on bits: each argument packed into a word (`SelectI32_v1` of each bit, `I32Add_v1`), one
    Call of ``WORD``, its result unpacked (`BitAtx32_v1`). A bridge for a word id that has no Boolean Definition yet."""
    words = []
    for a in args:
        acc = None
        for j in range(32):
            v = B.call(P.SelectI32, a[j], B.call(const(32, 1 << j)), B.call(const(32, 0)))
            acc = v if acc is None else B.call(P.I32Add, acc, v)
        words.append(acc)
    out = B.call(S.WORD, *words)
    return array_of([B.call(bit_at(32), out, B.call(const(32, j))) for j in range(32)])


WordOnBits.word = lambda b: b["WORD"]  # type: ignore[attr-defined]

#: the norms' MUFU statics bound to the word ids, until proofs' Boolean MUFU Definitions replace them
MUFU_WORD: dict = {"RSQRT": bind(WordOnBits, WORD=P.RsqrtApprox), "SQRT": bind(WordOnBits, WORD=P.MufuSqrtFtz),
                   "SCALEA": bind(WordOnBits, WORD=P.DivFullScaleA), "RCP": bind(WordOnBits, WORD=P.DivFullRcp)}
CUDA_WORD: dict = {"RSQRT": MUFU_WORD["RSQRT"]}
TRITON_WORD: dict = {k: MUFU_WORD[k] for k in ("SQRT", "SCALEA", "RCP")}

#: the word Definition each MUFU static must compute (DivFullScaleA's _v1 and _v2 differ only on a NaN numerator; here it is 1.0)
_MUFU_WORDS = {"RSQRT": {P.RsqrtApprox.id}, "SQRT": {P.MufuSqrtFtz.id}, "RCP": {P.DivFullRcp.id},
               "SCALEA": {P.DivFullScaleA.id, P.DivFullScaleAV2.id}}


def _check_mufu(b: dict, keys) -> None:
    for k in keys:
        got = word_of(b[k]).id
        if got not in _MUFU_WORDS[k]:
            raise ValueError(f"{k} computes {got}, not {' or '.join(sorted(_MUFU_WORDS[k]))}")


# --- leaf pieces: word composites of the _v1 primitives, and the same functions traced on bits ---------------------------------

def _square_word(B, S, z):
    zf = B.call(P.Bf16ToF32, z)
    return B.call(P.F32Mul, zf, zf)


def _square_fma_word(B, S, z, acc):
    zf = B.call(P.Bf16ToF32, z)
    return B.call(P.F32Fma, zf, zf, acc)


def _thread_word(B, S, x):
    xf = [B.call(P.Bf16ToF32, x[j]) for j in range(TRITON_COLS)]
    acc = B.call(P.F32Mul, xf[1], xf[1])
    acc = B.call(P.F32Fma, xf[0], xf[0], acc)
    for j in range(2, TRITON_COLS):
        acc = B.call(P.F32Fma, xf[j], xf[j], acc)
    return acc


def _out_cuda_word(B, S, z, s, w):
    return B.call(P.F2fpBf16, B.call(P.F32Mul, B.call(P.F32Mul, B.call(P.Bf16ToF32, z), s), B.call(P.Bf16ToF32, w)))


def _out_triton_word(B, S, inv, x, w):
    return B.call(P.F2fpBf16, B.call(P.F32Mul, B.call(P.F32Mul, inv, B.call(P.Bf16ToF32, x)), B.call(P.Bf16ToF32, w)))


def _word(name: str, body, params, ret, doc: str):
    return bind(CompositeDefinition(name, 1, (), lambda S: (params, ret), body, doc=doc, conformance="exact-model-tested"))


RMSNormSquareWord = _word("RMSNormSquare", _square_word, (("z", BF16W),), F32W,
                          "F32Mul_v1(f32(z), f32(z)): the fused norm's first square (`variance += x * x` from 0)")
RMSNormSquareFmaWord = _word("RMSNormSquareFma", _square_fma_word, (("z", BF16W), ("acc", F32W)), F32W,
                             "F32Fma_v1(f32(z), f32(z), acc): the fused norm's next square")
RMSNormThreadTritonWord = _word("RMSNormThreadTriton", _thread_word, (("x", Array(TRITON_COLS, BF16W)),), F32W,
                                "one Triton thread's 8 columns: acc = x1 * x1, then FFMA x0, x2, ..., x7 (F32Mul_v1, F32Fma_v1)")
RMSNormOutCudaWord = _word("RMSNormOutCuda", _out_cuda_word, (("z", BF16W), ("s", F32W), ("w", BF16W)), BF16W,
                           "F2fpBf16_v1((f32(z) * s) * f32(w)): the fused norm's output word")
RMSNormOutTritonWord = _word("RMSNormOutTriton", _out_triton_word, (("inv", F32W), ("x", BF16W), ("w", BF16W)), BF16W,
                             "F2fpBf16_v1((inv * f32(x)) * f32(w)): the Triton norm's output word")


def _square(C, z):
    zf = fp.bf16_to_f32(z)
    return fp.f32_mul(C, zf, zf, ir_nan=True)


def _square_fma(C, z, acc):
    zf = fp.bf16_to_f32(z)
    return fp.f32_fma(C, zf, zf, acc)


def _thread(C, *x):
    xf = [fp.bf16_to_f32(v) for v in x]
    acc = fp.f32_mul(C, xf[1], xf[1], ir_nan=True)
    acc = fp.f32_fma(C, xf[0], xf[0], acc)
    for j in range(2, TRITON_COLS):
        acc = fp.f32_fma(C, xf[j], xf[j], acc)
    return acc


def _out_cuda(C, z, s, w):
    y = fp.f32_mul(C, fp.f32_mul(C, fp.bf16_to_f32(z), s, ir_nan=True), fp.bf16_to_f32(w), ir_nan=True)
    return fp.f32_to_bf16(C, y, W.BF16_NAN_F2FP)


def _out_triton(C, inv, x, w):
    y = fp.f32_mul(C, fp.f32_mul(C, inv, fp.bf16_to_f32(x), ir_nan=True), fp.bf16_to_f32(w), ir_nan=True)
    return fp.f32_to_bf16(C, y, W.BF16_NAN_F2FP)


RMSNormSquare = definition("RMSNormSquare", 2, _square, (("z", BF16),), F32, word=RMSNormSquareWord,
                           doc="the fused norm's first square on bits: RMSNormSquare_v1, F32Mul_v1(f32(z), f32(z))")
RMSNormSquareFma = definition("RMSNormSquareFma", 2, _square_fma, (("z", BF16), ("acc", F32)), F32, word=RMSNormSquareFmaWord,
                              doc="the fused norm's next square on bits: RMSNormSquareFma_v1, F32Fma_v1(f32(z), f32(z), acc)")
RMSNormThreadTriton = definition("RMSNormThreadTriton", 2, _thread, (("x", Array(TRITON_COLS, BF16)),), F32,
                                 word=RMSNormThreadTritonWord,
                                 doc="one Triton thread's sum of squares on bits: RMSNormThreadTriton_v1")
RMSNormOutCuda = definition("RMSNormOutCuda", 2, _out_cuda, (("z", BF16), ("s", F32), ("w", BF16)), BF16, word=RMSNormOutCudaWord,
                            doc="the fused norm's output word on bits: RMSNormOutCuda_v1, bf16_rn((f32(z) * s) * f32(w))")
RMSNormOutTriton = definition("RMSNormOutTriton", 2, _out_triton, (("inv", F32), ("x", BF16), ("w", BF16)), BF16,
                              word=RMSNormOutTritonWord,
                              doc="the Triton norm's output word on bits: RMSNormOutTriton_v1, bf16_rn((inv * f32(x)) * f32(w))")


# --- the fused CUDA norm (vLLM's generic fused_add_rms_norm_kernel) -------------------------------------------------------------

def _threads_cuda(N: int, BLOCK: int) -> list[tuple[int, int, int]]:
    """(first thread, end, elements per thread) of each run of threads holding equally many elements."""
    T = min(N, BLOCK)
    runs: list[tuple[int, int, int]] = []
    for t in range(T):
        J = -(-(N - t) // T)
        if runs and runs[-1][2] == J:
            runs[-1] = (runs[-1][0], t + 1, J)
        else:
            runs.append((t, t + 1, J))
    return runs


@composite("RMSNormLaneCuda", 1, ["J"],
           lambda S: ((("x", Array(S.J, BF16)), ("res", Array(S.J, BF16))), Tuple(Array(S.J, BF16), F32)),
           conformance="exact-model-tested")
def RMSNormLaneCuda(B, S, x, res):
    """One thread's J elements: r = Bf16AddF2fp_v2(x, res) each, and its partial: the first r squared, then FFMA of each next."""
    r = B.batch(EW.Bf16AddF2fp, x, res, axes=(0, 0))
    acc = B.call(RMSNormSquare, r[0])
    for k in range(1, S.J):
        acc = B.call(RMSNormSquareFma, r[k], acc)
    return tuple_of(r, acc)


@composite("RMSNormLanesCuda", 1, ["N", "BLOCK"],
           lambda S: ((("x", Array(S.N, BF16)), ("res", Array(S.N, BF16))),
                      Tuple(Array(S.N, BF16), Array(min(S.N, S.BLOCK), F32))),
           conformance="exact-model-tested")
def RMSNormLanesCuda(B, S, x, res):
    """The residual sum r and every thread's partial of a block of min(N, BLOCK) threads, thread t owning t, t + BLOCK, ...:
    one batch of `RMSNormLaneCuda_v1{J}` per run of threads with J elements (separable)."""
    N, T = S.N, min(S.N, S.BLOCK)
    r: dict[int, Coll] = {}
    partials: list[Coll] = []
    for t0, t1, J in _threads_cuda(N, S.BLOCK):
        M = t1 - t0
        els = [t0 + m + k * T for m in range(M) for k in range(J)]
        out = B.batch(bind(RMSNormLaneCuda, J=J), coll_concat(Array(M, Array(J, BF16)), [x[e] for e in els]),
                      coll_concat(Array(M, Array(J, BF16)), [res[e] for e in els]), axes=(0, 0))
        for m in range(M):
            for k in range(J):
                r[t0 + m + k * T] = out[m][0][k]
            partials.append(out[m][1])
    return tuple_of(array_of([r[e] for e in range(N)]), array_of(partials))


@composite("RMSNormWarpCuda", 1, ["L"], lambda S: ((("p", Array(S.L, F32)),), F32), conformance="exact-model-tested")
def RMSNormWarpCuda(B, S, p):
    """cub `WarpReduceShfl` over L valid lanes: offsets 1, 2, 4, 8, 16, lane i adding lane i + offset when it exists; lane 0."""
    if not 1 <= S.L <= WARP:
        raise ValueError(f"a warp has 1 to {WARP} lanes, not {S.L}")
    lanes = {i: p[i] for i in range(S.L)}
    for off in (1, 2, 4, 8, 16):
        lanes = {i: (B.call(EW.F32Add, v, lanes[i + off]) if i + off in lanes else v)
                 for i, v in lanes.items() if i % (2 * off) == 0}
    return lanes[0]


def _warps(B, p: Coll, n: int, warp) -> Coll:
    """Each 32-lane warp's sum of the n partials ``p``, the last warp possibly partial: ``warp(L)`` is the warp Definition."""
    full, rest = divmod(n, WARP)
    sums: list[Coll] = []
    if full:
        out = B.batch(warp(WARP), p[:WARP * full].reshape(Array(full, Array(WARP, F32))), axes=(0,))
        sums += [out[i] for i in range(full)]
    if rest:
        sums.append(B.call(warp(rest), p[WARP * full:]))
    return array_of(sums)


@composite("RMSNormWarpsCuda", 1, ["T"], lambda S: ((("p", Array(S.T, F32)),), Array(-(-S.T // WARP), F32)),
           conformance="exact-model-tested")
def RMSNormWarpsCuda(B, S, p):
    """Every warp's aggregate of the T threads' partials (`RMSNormWarpCuda_v1`; separable)."""
    return _warps(B, p, S.T, lambda L: bind(RMSNormWarpCuda, L=L))


@composite("RMSNormArgCuda", 1, ["N", "EPS", "NW"], lambda S: ((("s", Array(S.NW, F32)),), F32), conformance="exact-model-tested")
def RMSNormArgCuda(B, S, s):
    """`variance / N + EPS`: cub's `ApplyWarpAggregates` (warp sums added in order from warp 0), F32Div_v3, F32Add_v3."""
    c = _Consts(B)
    var = s[0]
    for i in range(1, S.NW):
        var = B.call(EW.F32Add, var, s[i])
    return B.call(EW.F32Add, B.call(EW.F32Div, var, c(_f32(float(S.N)))), c(_f32(S.EPS)))


@composite("RMSNormOutRowCuda", 1, ["N"],
           lambda S: ((("r", Array(S.N, BF16)), ("s", F32), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def RMSNormOutRowCuda(B, S, r, s, w):
    """bf16_rn((f32(r_j) * s) * f32(w_j)) for every j (`RMSNormOutCuda_v2`; separable)."""
    return B.batch(RMSNormOutCuda, r, s, w, axes=(0, None, 0))


def rmsnorm_cuda_calls(B, N: int, EPS: float, RSQRT, x: Coll, res: Coll, w: Coll) -> Coll:
    """`RMSNormFusedCuda_v3`'s Calls, written into the body ``B`` (a composite's, or a Program root's): (normed, res')."""
    T = min(N, b1.FUSED_RMS_BLOCK)
    lanes = B.call(bind(RMSNormLanesCuda, N=N, BLOCK=b1.FUSED_RMS_BLOCK), x, res)
    sums = B.call(bind(RMSNormWarpsCuda, T=T), lanes[1])
    arg = B.call(bind(RMSNormArgCuda, N=N, EPS=EPS, NW=-(-T // WARP)), sums)
    return tuple_of(B.call(bind(RMSNormOutRowCuda, N=N), lanes[0], B.call(RSQRT, arg), w), lanes[0])


@composite("RMSNormFusedCuda", 3, ["N", "EPS", "RSQRT"],
           lambda S: ((("x", Array(S.N, BF16)), ("res", Array(S.N, BF16)), ("w", Array(S.N, BF16))),
                      Tuple(Array(S.N, BF16), Array(S.N, BF16))),
           conformance="exact-model-tested")
def RMSNormFusedCuda(B, S, x, res, w):
    """`RMSNormFusedCuda_v2` on bits (`rmsnorm_cuda_calls`): vLLM's generic `fused_add_rms_norm_kernel<bf16, 0, true>`,
    `s = rsqrtf(variance / N + eps)` with ``RSQRT`` computing `RsqrtApprox_v1`. Returns (normed, res')."""
    return rmsnorm_cuda_calls(B, S.N, S.EPS, S.RSQRT, x, res, w)


def _cuda_word(b: dict):
    _check_mufu(b, ("RSQRT",))
    return bind(b1.RMSNormFusedCuda, N=b["N"], EPS=b["EPS"])


RMSNormFusedCuda.word = _cuda_word  # type: ignore[attr-defined]


# --- the Triton norm (vLLM's batch-invariant _rms_norm_kernel) ------------------------------------------------------------------

def _add0(B, a: Coll | None, b: Coll | None) -> Coll | None:
    """``a + b`` (F32Add_v3), an idle +0 (None) dropped: neither operand is -0 or a signalling NaN (the module docstring)."""
    if a is None or b is None:
        return b if a is None else a
    return B.call(EW.F32Add, a, b)


@composite("RMSNormThreadsTriton", 1, ["N"], lambda S: ((("x", Array(S.N, BF16)),), Array(S.N // TRITON_COLS, F32)),
           conformance="exact-model-tested")
def RMSNormThreadsTriton(B, S, x):
    """Each thread's sum of squares of its 8 columns, threads in column order (`RMSNormThreadTriton_v2`; separable)."""
    if S.N % TRITON_COLS:
        raise ValueError(f"the Triton norm reads {TRITON_COLS} columns per thread; N = {S.N} is not a multiple")
    return B.batch(RMSNormThreadTriton, x.reshape(Array(S.N // TRITON_COLS, Array(TRITON_COLS, BF16))), axes=(0,))


@composite("RMSNormWarpTriton", 1, ["L"], lambda S: ((("p", Array(S.L, F32)),), F32), conformance="exact-model-tested")
def RMSNormWarpTriton(B, S, p):
    """Lane 0 after the xor butterfly 16, 8, 4, 2, 1 (lane i adds lane i ^ m, its own value first) over a warp whose lanes from L
    on are idle (+0): only lane 0's cone, without the idle adds."""
    if not 1 <= S.L <= WARP:
        raise ValueError(f"a warp has 1 to {WARP} lanes, not {S.L}")
    masks = (16, 8, 4, 2, 1)
    memo: dict[tuple[int, int], Coll | None] = {}

    def lane(k: int, i: int) -> Coll | None:
        if k == 0:
            return p[i] if i < S.L else None
        if (k, i) not in memo:
            memo[(k, i)] = _add0(B, lane(k - 1, i), lane(k - 1, i ^ masks[k - 1]))
        return memo[(k, i)]

    return lane(len(masks), 0)


@composite("RMSNormWarpsTriton", 1, ["N"], lambda S: ((("p", Array(S.N // TRITON_COLS, F32)),),
                                                        Array(-(-S.N // (TRITON_COLS * WARP)), F32)),
           conformance="exact-model-tested")
def RMSNormWarpsTriton(B, S, p):
    """Each warp's butterfly sum, for the warps holding at least one column (chunk-major, 4 per chunk; separable)."""
    return _warps(B, p, S.N // TRITON_COLS, lambda L: bind(RMSNormWarpTriton, L=L))


@composite("RMSNormMeanEpsTriton", 1, ["N", "EPS"],
           lambda S: ((("s", Array(-(-S.N // (TRITON_COLS * WARP)), F32)),), F32), conformance="exact-model-tested")
def RMSNormMeanEpsTriton(B, S, s):
    """Per chunk of 1024 columns `(w0 + w2) + (w1 + w3)` (idle warps +0), the chunks added in order from +0, then
    F32Fma_v3(rcp(N), sum, EPS) with rcp(N) = DivFullRcp_v1(f32(N)) folded to a constant."""
    c = _Consts(B)
    nw = len(s)
    warps = TRITON_THREADS // WARP
    total = None
    for c0 in range(0, nw, warps):
        w = [s[c0 + i] if c0 + i < nw else None for i in range(warps)]
        total = _add0(B, total, _add0(B, _add0(B, w[0], w[2]), _add0(B, w[1], w[3])))
    (rcp_n,) = evaluate(P.DivFullRcp, _f32(float(S.N)))
    return B.call(EW.F32Fma, c(rcp_n), total, c(_f32(S.EPS)))


@composite("RMSNormOutRowTriton", 1, ["N"],
           lambda S: ((("inv", F32), ("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def RMSNormOutRowTriton(B, S, inv, x, w):
    """bf16_rn((inv * f32(x_j)) * f32(w_j)) for every j (`RMSNormOutTriton_v2`; separable)."""
    return B.batch(RMSNormOutTriton, inv, x, w, axes=(None, 0, 0))


def rmsnorm_triton_calls(B, N: int, EPS: float, SQRT, SCALEA, RCP, x: Coll, w: Coll) -> Coll:
    """`RMSNormTriton_v2`'s Calls, written into the body ``B`` (a composite's, or a Program root's)."""
    p = B.call(bind(RMSNormThreadsTriton, N=N), x)
    s = B.call(bind(RMSNormWarpsTriton, N=N), p)
    rms = B.call(SQRT, B.call(bind(RMSNormMeanEpsTriton, N=N, EPS=EPS), s))
    inv = B.call(EW.F32Mul, B.call(SCALEA, _Consts(B)(_f32(1.0)), rms), B.call(RCP, rms))
    return B.call(bind(RMSNormOutRowTriton, N=N), inv, x, w)


@composite("RMSNormTriton", 2, ["N", "EPS", "SQRT", "SCALEA", "RCP"],
           lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def RMSNormTriton(B, S, x, w):
    """`RMSNormTriton_v1` on bits (`rmsnorm_triton_calls`): vLLM's batch-invariant Triton `_rms_norm_kernel`,
    `inv = div.full(1, sqrt.approx.ftz(fma(rcp(N), sum, eps)))` with ``SQRT``, ``SCALEA`` and ``RCP`` computing `MufuSqrtFtz_v1`,
    `DivFullScaleA_v1` and `DivFullRcp_v1`."""
    return rmsnorm_triton_calls(B, S.N, S.EPS, S.SQRT, S.SCALEA, S.RCP, x, w)


def _triton_word(b: dict):
    _check_mufu(b, ("SQRT", "SCALEA", "RCP"))
    return bind(b1.RMSNormTriton, N=b["N"], EPS=b["EPS"])


RMSNormTriton.word = _triton_word  # type: ignore[attr-defined]

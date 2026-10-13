"""vLLM's cuteDSL router GEMM `ll_bf16_gemm` (`GateLinear` tier 1: bf16 x bf16 -> f32 at M <= 16 on sm_90+), GLM-5.3's and
DeepSeek-V4-Pro's MoE gates on decode steps, as its two kernels reduce K (vLLM d9105ea80 `kernels/linear/cute_dsl/`, CUTLASS DSL
4.6.2):

    LlBf16DotprodDot_v1{K, BS, VEC, FMA}     (x, w: Array<K, BF16>) -> F32
        `LLBf16Dotprod` (M <= 4): thread t of the CTA's BS runs one f32 chain from +0 over k = VEC BS i + VEC t + v (i, then the
        VEC elements of its vector load, ascending) on K's whole tiles of VEC BS, then likewise at width VEC/2 on what is left
        (the kernel's 64-bit tail loop beside its 128-bit main loop, VEC 8), then rounds of BS, one k per thread, the last on the
        threads below what remains; each step `acc + f32(x_k) * f32(w_k)`: one FFMA when FMA = 1 (libNVVM's
        default contraction), else an FMUL then an FADD.  Each warp's 32 chains meet in `warp_reduction_sum`'s butterfly
        (offsets 16, 8, 4, 2, 1: lane i adds lane i + o), and the BS / 32 warp sums in `TensorSSA.reduce(ADD, init 0.0)`'s
        ordered sum, ((0 + s_0) + s_1) + ...
    LlBf16SplitKDot_v1{K, SPLIT_K, TILE_K, WARPS, DOT}     (x, w: Array<K, BF16>) -> F32
        `LLBf16SplitK` (5 <= M <= 16): cluster rank z of SPLIT_K takes the K tiles z, z + SPLIT_K, ... of TILE_K; within each,
        MMA warp j takes the k16 blocks j, j + WARPS, ..., one `DOT` chain from +0 per warp across its tiles; the warps' chains
        add as ((0 + p_0) + p_1) + ..., and the ranks' sums likewise in rank order.  The epilogue's `* scale` (1.0, the only value
        `ll_bf16_gemm` passes) is the identity on every word that reaches the output, which the rank sum adds to +0.
    LlBf16DotprodGemm_v1{K, N, BS, VEC, FMA}, LlBf16SplitKGemm_v1{K, N, SPLIT_K, TILE_K, WARPS, DOT}
                                             (x: Array<K, BF16>, w: Array<N, Array<K, BF16>>) -> Array<N, F32>
        the dot on each W row: one token row of the router logits

on words (`F32Fma_v1`, `F32Mul_v1`, `F32Add_v1`, `DotBf16`'s step), and each one's v2 the same function on bits (`F32Add_v3`,
the steps as `cublas_bf16.ChainStep`s; its ``word`` the v1, so `boolean.boolean_version` finds it).  The dotprod's step on bits
is traced from the bf16 words, as `boolean_dense`'s elements are: `LL_FMA_STEP` (`F32Fma_v1` on the widened words) when FMA = 1,
else `LL_MUL_ADD_STEP` (`F32Mul_v1`, then `F32Add_v1`), so the widened operands' 16 zero bits fold into its gates; each lane's
first step is its own Definition, from the +0 accumulator with its constant bits folded, and each step is cut at the gates that
read only x_k.  The rows on bits take each step of every W row at once, one batch over the rows sharing x_k's own gates, as the
split-K's `DOT` steps do (`cublas_bf16.dot_step`), so no row computes a value another computes.  vLLM's sm_90
dispatch at GLM-5.3's gate (K 6144, N 256): M <= 4 the dotprod at BS 128 (VEC 8, so 6 tiles of 1024); M 9 and 11 the split-K at
SPLIT_K 8, else SPLIT_K 6 (TILE_K 256, WARPS 4, `mma.sync.m16n8k16`, `HopperBF16WgmmaDot16_v1`).  At DeepSeek-V4-Pro's (K 7168,
N 384): M 1 and 2 the dotprod at BS 256 (3 tiles of 2048, then one of 1024 at width 4), M 3 and 4 at BS 128 (7 tiles); M 5 the
split-K at SPLIT_K 6, M 6, 8 and 12 at 8, M 7 to 16 otherwise at 7, over its 28 tiles of 256 (so ranks may take one tile fewer
than rank 0).  `bf16_gemm_check.dispatch_sm90` holds the tables, beside the numpy models.  The H200's kernels are pinned by
`tests/properties/glm53_bf16_gemm_capture_gpu.py`, which also decides FMA.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.refs import Coll, broadcast
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import cublas_bf16 as CB

__all__ = ["LL_FMA_STEP", "LL_MUL_ADD_STEP", "LlBf16DotprodDot", "LlBf16DotprodDotWord", "LlBf16DotprodGemm", "LlBf16DotprodGemmWord",
           "LlBf16FmaStep", "LlBf16FmaStepFirst", "LlBf16FmaStepFirstX", "LlBf16FmaStepX", "LlBf16MulAddStep",
           "LlBf16MulAddStepFirst", "LlBf16MulAddStepFirstX", "LlBf16MulAddStepX", "LlBf16SplitKDot", "LlBf16SplitKDotWord",
           "LlBf16SplitKGemm", "LlBf16SplitKGemmWord", "WARP"]

BF16, F32 = Array(16, BL.BIT), Array(32, BL.BIT)
WARP = 32
CONFORMANCE = "not-yet-established: the capture's served gates (tests/properties/glm53_bf16_gemm_capture_gpu.py)"


def _check(name: str, **ok: bool) -> None:
    bad = [k for k, v in ok.items() if not v]
    if bad:
        raise ValueError(f"{name}: unsupported statics ({', '.join(bad)})")


def _check_dotprod(S) -> None:
    _check("LlBf16DotprodDot", BS_whole_warps=S.BS > 0 and S.BS % WARP == 0, K_at_least_BS=S.K >= S.BS, VEC_positive=S.VEC > 0,
           FMA_0_or_1=S.FMA in (0, 1))


def _check_splitk(S) -> None:
    _check("LlBf16SplitKDot", K_multiple_of_TILE_K=S.K > 0 and S.K % S.TILE_K == 0, TILE_K_multiple_of_16=S.TILE_K % 16 == 0,
           blocks_per_warp_whole=S.WARPS > 0 and (S.TILE_K // 16) % S.WARPS == 0,
           SPLIT_K_at_most_tiles=0 < S.SPLIT_K <= S.K // S.TILE_K)


_STEP_PARAMS = (("acc", F32), ("x", BF16), ("w", BF16))
LL_FMA_STEP = CB.chain_step("LlBf16FmaStep", lambda C, acc, x, w: fp.f32_fma(C, fp.bf16_to_f32(x), fp.bf16_to_f32(w), acc),
                            _STEP_PARAMS, ("fma(f32(x), f32(w), acc), one FFMA (F32Fma_v1 on the widened words)", "x"))
LL_MUL_ADD_STEP = CB.chain_step(
    "LlBf16MulAddStep",
    lambda C, acc, x, w: fp.f32_add(C, acc, fp.f32_mul(C, fp.bf16_to_f32(x), fp.bf16_to_f32(w), ir_nan=True), ir_nan=True),
    _STEP_PARAMS, ("acc + f32(x) * f32(w), an FMUL then an FADD (F32Mul_v1's and F32Add_v1's NaN words)", "x"))
LlBf16FmaStepFirstX, LlBf16FmaStepFirst, LlBf16FmaStepX, LlBf16FmaStep = LL_FMA_STEP
LlBf16MulAddStepFirstX, LlBf16MulAddStepFirst, LlBf16MulAddStepX, LlBf16MulAddStep = LL_MUL_ADD_STEP
_LL_STEPS = {1: LL_FMA_STEP, 0: LL_MUL_ADD_STEP}


def _bits_zero(B, rows: int | None = None) -> Coll:
    """+0 on bits: one F32, or with ``rows`` one per row."""
    space, idx, leaf = B.call(BL.constant(0)).refs[0]
    return Coll(F32 if rows is None else Array(rows, F32), broadcast(space, idx, leaf, 32 * (rows or 1)))


def _adder(B, rows: bool):
    return (lambda a, b: B.batch(EW.F32Add, a, b, axes=(0, 0))) if rows else (lambda a, b: B.call(EW.F32Add, a, b))


def _ordered_sum(add, zero, parts):
    out = zero()
    for p in parts:
        out = add(out, p)
    return out


def _lanes(S) -> list[list[int]]:
    """Each thread's chain: thread t's k in its order, as `LLBf16Dotprod._init_k_tiles` cuts K.  The whole tiles of VEC BS, k =
    VEC BS i + VEC t + v (i, then v); on what is left, the whole tiles of VEC/2 BS likewise; then rounds of BS, one k per thread,
    the last on the threads below what remains."""
    lanes: list[list[int]] = [[] for _ in range(S.BS)]
    k0 = 0
    for width in (S.VEC, S.VEC // 2):
        tiles = (S.K - k0) // (width * S.BS) if width else 0
        for t, ks in enumerate(lanes):
            ks += [k0 + width * (S.BS * i + t) + v for i in range(tiles) for v in range(width)]
        k0 += tiles * width * S.BS
    for t, ks in enumerate(lanes):
        ks += range(k0 + t, S.K, S.BS)
    return lanes


def _reduce(add, zero, lanes):
    """Each warp's butterfly (offsets 16, ..., 1: lane i adds lane i + o), then the warp sums in order from +0."""
    sums = []
    for j in range(len(lanes) // WARP):
        vals = lanes[WARP * j:WARP * (j + 1)]
        o = WARP // 2
        while o:
            vals = [add(vals[i], vals[i + o]) for i in range(o)]
            o //= 2
        sums.append(vals[0])
    return _ordered_sum(add, zero, sums)


def _dotprod_word(B, S, x, w):
    _check_dotprod(S)
    xf, wf = (B.batch(P.Bf16ToF32, a, axes=(0,)) for a in (x, w))
    step = ((lambda k, acc: B.call(W.F32Fma, xf[k], wf[k], acc)) if S.FMA
            else (lambda k, acc: B.call(W.F32Add, acc, B.call(W.F32Mul, xf[k], wf[k]))))
    lanes = []
    for ks in _lanes(S):
        acc = B.call(P.ZERO32)
        for k in ks:
            acc = step(k, acc)
        lanes.append(acc)
    return _reduce(lambda a, b: B.call(W.F32Add, a, b), lambda: B.call(P.ZERO32), lanes)


def _dotprod_bits(rows: bool):
    def body(B, S, x, w):
        _check_dotprod(S)
        step = _LL_STEPS[S.FMA]
        lanes = []
        for ks in _lanes(S):
            acc = None
            for k in ks:
                acc = step(B, acc, x[k], w.take(1, k) if rows else w[k], rows)
            lanes.append(acc)
        return _reduce(_adder(B, rows), lambda: _bits_zero(B, S.N if rows else None), lanes)
    return body


def _splitk_word(B, S, x, w):
    _check_splitk(S)
    tiles, blocks = S.K // S.TILE_K, S.TILE_K // 16
    add, zero = (lambda a, b: B.call(W.F32Add, a, b)), (lambda: B.call(P.ZERO32))
    ranks = []
    for z in range(S.SPLIT_K):
        parts = []
        for j in range(S.WARPS):
            acc = zero()
            for t in range(z, tiles, S.SPLIT_K):
                for ki in range(blocks // S.WARPS):
                    k0 = S.TILE_K * t + 16 * (j + S.WARPS * ki)
                    acc = B.call(S.DOT, acc, x[k0:k0 + 16], w[k0:k0 + 16])
            parts.append(acc)
        ranks.append(_ordered_sum(add, zero, parts))
    return _ordered_sum(add, zero, ranks)


def _splitk_bits(rows: bool):
    def body(B, S, x, w):
        _check_splitk(S)
        step = CB.dot_step("LlBf16SplitKDot", S.DOT)
        tiles, blocks = S.K // S.TILE_K, S.TILE_K // 16
        wb = w.reshape(Array(S.N, Array(S.K // 16, Array(16, BF16)))) if rows else None
        add, zero = _adder(B, rows), (lambda: _bits_zero(B, S.N if rows else None))
        ranks = []
        for z in range(S.SPLIT_K):
            parts = []
            for j in range(S.WARPS):
                acc = None
                for t in range(z, tiles, S.SPLIT_K):
                    for ki in range(blocks // S.WARPS):
                        k0 = S.TILE_K * t + 16 * (j + S.WARPS * ki)
                        acc = step(B, acc, x[k0:k0 + 16], wb.take(1, k0 // 16) if rows else w[k0:k0 + 16], rows)
                parts.append(acc)
            ranks.append(_ordered_sum(add, zero, parts))
        return _ordered_sum(add, zero, ranks)
    return body


def _rows(dot, statics):
    def body(B, S, x, w):
        return B.batch(bind(dot, **{k: getattr(S, k) for k in statics}), x, w, axes=(None, 0))
    return body


def _dot_type(X, T):
    return lambda S: ((("x", Array(S.K, X)), ("w", Array(S.K, X))), T)


def _gemm_type(X, T):
    return lambda S: ((("x", Array(S.K, X)), ("w", Array(S.N, Array(S.K, X)))), Array(S.N, T))


DOTPROD = ("K", "BS", "VEC", "FMA")
SPLITK = ("K", "SPLIT_K", "TILE_K", "WARPS", "DOT")
DOTPROD_DOC = ("LLBf16Dotprod's coordinate: BS thread chains from +0 over k = VEC BS i + VEC t + v, then the tail at width "
               "VEC/2, then rounds of BS (FFMA when FMA, else FMUL then FADD), each warp's butterfly, then the warp sums in order "
               "from +0")
SPLITK_DOC = ("LLBf16SplitK's coordinate: per cluster rank and MMA warp one DOT chain from +0 over its k16 blocks, the warps' "
              "chains then the ranks' sums added in order from +0")
ROWS_DOC = "one token row of ll_bf16_gemm's f32 logits: the kernel's dot on each W row"
ROWS_BITS = "on bits: each step over every W row at once, x's own gates once"

LlBf16DotprodDotWord = CompositeDefinition("LlBf16DotprodDot", 1, DOTPROD, _dot_type(P.BF16, P.F32), _dotprod_word,
                                           doc=DOTPROD_DOC, conformance=CONFORMANCE)
LlBf16SplitKDotWord = CompositeDefinition("LlBf16SplitKDot", 1, SPLITK, _dot_type(P.BF16, P.F32), _splitk_word,
                                          doc=SPLITK_DOC, conformance=CONFORMANCE)
LlBf16DotprodGemmWord = CompositeDefinition("LlBf16DotprodGemm", 1, ("N",) + DOTPROD, _gemm_type(P.BF16, P.F32),
                                            _rows(LlBf16DotprodDotWord, DOTPROD), doc=ROWS_DOC, conformance=CONFORMANCE)
LlBf16SplitKGemmWord = CompositeDefinition("LlBf16SplitKGemm", 1, ("N",) + SPLITK, _gemm_type(P.BF16, P.F32),
                                           _rows(LlBf16SplitKDotWord, SPLITK), doc=ROWS_DOC, conformance=CONFORMANCE)

LlBf16DotprodDot = CompositeDefinition("LlBf16DotprodDot", 2, DOTPROD, _dot_type(BF16, F32), _dotprod_bits(False),
                                       doc=f"{DOTPROD_DOC} (on bits)", conformance=CONFORMANCE)
LlBf16SplitKDot = CompositeDefinition("LlBf16SplitKDot", 2, SPLITK, _dot_type(BF16, F32), _splitk_bits(False),
                                      doc=f"{SPLITK_DOC} (on bits)", conformance=CONFORMANCE)
LlBf16DotprodGemm = CompositeDefinition("LlBf16DotprodGemm", 2, ("N",) + DOTPROD, _gemm_type(BF16, F32), _dotprod_bits(True),
                                        doc=f"{ROWS_DOC} ({ROWS_BITS})", conformance=CONFORMANCE)
LlBf16SplitKGemm = CompositeDefinition("LlBf16SplitKGemm", 2, ("N",) + SPLITK, _gemm_type(BF16, F32), _splitk_bits(True),
                                       doc=f"{ROWS_DOC} ({ROWS_BITS})", conformance=CONFORMANCE)


def _word(defn, statics):
    return lambda b: bind(defn, **{k: word_of(b[k]) if k == "DOT" else b[k] for k in statics})


LlBf16DotprodDot.word = _word(LlBf16DotprodDotWord, DOTPROD)  # type: ignore[attr-defined]
LlBf16SplitKDot.word = _word(LlBf16SplitKDotWord, SPLITK)  # type: ignore[attr-defined]
LlBf16DotprodGemm.word = _word(LlBf16DotprodGemmWord, ("N",) + DOTPROD)  # type: ignore[attr-defined]
LlBf16SplitKGemm.word = _word(LlBf16SplitKGemmWord, ("N",) + SPLITK)  # type: ignore[attr-defined]

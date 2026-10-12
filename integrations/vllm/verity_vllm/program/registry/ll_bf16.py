"""vLLM's cuteDSL router GEMM `ll_bf16_gemm` (`GateLinear` tier 1: bf16 x bf16 -> f32 at M <= 16 on sm_90+), GLM-5.3's MoE gate
on decode steps, as its two kernels reduce K (vLLM d9105ea80 `kernels/linear/cute_dsl/`, CUTLASS DSL 4.6.2):

    LlBf16DotprodDot_v1{K, BS, VEC, FMA}     (x, w: Array<K, BF16>) -> F32
        `LLBf16Dotprod` (M <= 4): thread t of the CTA's BS runs one f32 chain from +0 over k = VEC BS i + VEC t + v (i, then the
        VEC elements of its vector load, ascending), each step `acc + f32(x_k) * f32(w_k)`: one FFMA when FMA = 1 (libNVVM's
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

on words (`F32Fma_v1`, `F32Mul_v1`, `F32Add_v1`, `DotBf16`'s step), and each one's v2 the same body on bits (`F32Add_v3`, the
step's Boolean version; its ``word`` the v1, so `boolean.boolean_version` finds it).  The dotprod's steps on bits are traced from
the bf16 words (`LlBf16FmaStep_v2`, `LlBf16MulAddStep_v2`, whose word views `LlBf16FmaStep_v1`, `LlBf16MulAddStep_v1` widen and
then call `F32Fma_v1`, or `F32Mul_v1` and `F32Add_v1`), so the widened operands' 16 zero bits fold into the gates: `F32Fma_v3`
and `F32Mul_v3` on widened words would recompute the same constant gates in every step.  vLLM's sm_90
dispatch at GLM-5.3's gate (K 6144, N 256): M <= 4 the dotprod at BS 128 (VEC 8, so 6 tiles of 1024); M 9 and 11 the split-K at
SPLIT_K 8, else SPLIT_K 6 (TILE_K 256, WARPS 4, `mma.sync.m16n8k16`, `HopperBF16WgmmaDot16_v1`); `bf16_gemm_check.dispatch_sm90`,
beside the numpy models.  The H200's kernels are pinned by `tests/properties/glm53_bf16_gemm_capture_gpu.py`, which also decides
FMA.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.trace import definition, word_of
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.refs import Coll, broadcast
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array

__all__ = ["LlBf16DotprodDot", "LlBf16DotprodDotWord", "LlBf16DotprodGemm", "LlBf16DotprodGemmWord", "LlBf16FmaStep",
           "LlBf16FmaStepWord", "LlBf16MulAddStep", "LlBf16MulAddStepWord", "LlBf16SplitKDot", "LlBf16SplitKDotWord",
           "LlBf16SplitKGemm", "LlBf16SplitKGemmWord", "WARP"]

BF16, F32 = Array(16, BL.BIT), Array(32, BL.BIT)
WARP = 32
CONFORMANCE = "not-yet-established: GLM-5.3's gate on the H200 (tests/properties/glm53_bf16_gemm_capture_gpu.py)"


def _check(name: str, **ok: bool) -> None:
    bad = [k for k, v in ok.items() if not v]
    if bad:
        raise ValueError(f"{name}: unsupported statics ({', '.join(bad)})")


def _word_zero(B) -> Coll:
    return B.call(P.ZERO32)


def _bits_zero(B) -> Coll:
    space, idx, leaf = B.call(BL.constant(0)).refs[0]
    return Coll(F32, broadcast(space, idx, leaf, 32))


def _step_type(X, T):
    return lambda S: ((("x", X), ("w", X), ("acc", T)), T)


def _fma_step(B, S, x, w, acc):
    """The dotprod's contracted step: ``fma(f32(x), f32(w), acc)``, one FFMA."""
    return B.call(W.F32Fma, B.call(P.Bf16ToF32, x), B.call(P.Bf16ToF32, w), acc)


def _mul_add_step(B, S, x, w, acc):
    """The dotprod's uncontracted step: ``acc + f32(x) * f32(w)``, an FMUL then an FADD."""
    return B.call(W.F32Add, acc, B.call(W.F32Mul, B.call(P.Bf16ToF32, x), B.call(P.Bf16ToF32, w)))


LlBf16FmaStepWord = CompositeDefinition("LlBf16FmaStep", 1, (), _step_type(P.BF16, P.F32), _fma_step, doc=_fma_step.__doc__,
                                        conformance=CONFORMANCE)
LlBf16MulAddStepWord = CompositeDefinition("LlBf16MulAddStep", 1, (), _step_type(P.BF16, P.F32), _mul_add_step,
                                           doc=_mul_add_step.__doc__, conformance=CONFORMANCE)
_STEP_PARAMS = (("x", BF16), ("w", BF16), ("acc", F32))
LlBf16FmaStep = definition("LlBf16FmaStep", 2, lambda C, x, w, acc: fp.f32_fma(C, fp.bf16_to_f32(x), fp.bf16_to_f32(w), acc),
                           _STEP_PARAMS, F32, word=bind(LlBf16FmaStepWord), doc=f"{_fma_step.__doc__} (on bits)",
                           conformance=CONFORMANCE)
LlBf16MulAddStep = definition(
    "LlBf16MulAddStep", 2,
    lambda C, x, w, acc: fp.f32_add(C, acc, fp.f32_mul(C, fp.bf16_to_f32(x), fp.bf16_to_f32(w), ir_nan=True), ir_nan=True),
    _STEP_PARAMS, F32, word=bind(LlBf16MulAddStepWord), doc=f"{_mul_add_step.__doc__} (on bits)", conformance=CONFORMANCE)

WORD = {"zero": _word_zero, "widen": P.Bf16ToF32, "fma": W.F32Fma, "mul": W.F32Mul, "add": W.F32Add}
#: on bits the dotprod calls one traced step per element on the bf16 words (`LlBf16FmaStep_v2`, `LlBf16MulAddStep_v2` by FMA)
BITS = {"zero": _bits_zero, "add": EW.F32Add, "step": {1: LlBf16FmaStep, 0: LlBf16MulAddStep}}


def _ordered_sum(B, ops, parts) -> Coll:
    out = ops["zero"](B)
    for p in parts:
        out = B.call(ops["add"], out, p)
    return out


def _dotprod(ops):
    def body(B, S, x, w):
        tile = S.VEC * S.BS
        _check("LlBf16DotprodDot", K_multiple_of_VEC_BS=S.K > 0 and S.K % tile == 0, BS_whole_warps=S.BS > 0 and S.BS % WARP == 0,
               FMA_0_or_1=S.FMA in (0, 1))
        if "step" in ops:
            step = lambda k, acc: B.call(ops["step"][S.FMA], x[k], w[k], acc)  # noqa: E731
        else:
            xf, wf = (B.batch(ops["widen"], a, axes=(0,)) for a in (x, w))
            step = lambda k, acc: (B.call(ops["fma"], xf[k], wf[k], acc) if S.FMA  # noqa: E731
                                   else B.call(ops["add"], acc, B.call(ops["mul"], xf[k], wf[k])))
        lanes = []
        for t in range(S.BS):
            acc = ops["zero"](B)
            for i in range(S.K // tile):
                for v in range(S.VEC):
                    acc = step(tile * i + S.VEC * t + v, acc)
            lanes.append(acc)
        sums = []
        for j in range(S.BS // WARP):
            vals = lanes[WARP * j:WARP * (j + 1)]
            o = WARP // 2
            while o:
                vals = [B.call(ops["add"], vals[i], vals[i + o]) for i in range(o)]
                o //= 2
            sums.append(vals[0])
        return _ordered_sum(B, ops, sums)
    return body


def _splitk(ops):
    def body(B, S, x, w):
        tiles, blocks = S.K // S.TILE_K, S.TILE_K // 16
        _check("LlBf16SplitKDot", K_multiple_of_TILE_K=S.K > 0 and S.K % S.TILE_K == 0, TILE_K_multiple_of_16=S.TILE_K % 16 == 0,
               blocks_per_warp_whole=S.WARPS > 0 and blocks % S.WARPS == 0, SPLIT_K_at_most_tiles=0 < S.SPLIT_K <= tiles)
        ranks = []
        for z in range(S.SPLIT_K):
            parts = []
            for j in range(S.WARPS):
                acc = ops["zero"](B)
                for t in range(z, tiles, S.SPLIT_K):
                    for ki in range(blocks // S.WARPS):
                        k0 = S.TILE_K * t + 16 * (j + S.WARPS * ki)
                        acc = B.call(S.DOT, acc, x[k0:k0 + 16], w[k0:k0 + 16])
                parts.append(acc)
            ranks.append(_ordered_sum(B, ops, parts))
        return _ordered_sum(B, ops, ranks)
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
DOTPROD_DOC = ("LLBf16Dotprod's coordinate: BS thread chains from +0 over k = VEC BS i + VEC t + v (FFMA when FMA, else FMUL "
               "then FADD), each warp's butterfly, then the warp sums in order from +0")
SPLITK_DOC = ("LLBf16SplitK's coordinate: per cluster rank and MMA warp one DOT chain from +0 over its k16 blocks, the warps' "
              "chains then the ranks' sums added in order from +0")
ROWS_DOC = "one token row of ll_bf16_gemm's f32 logits: the kernel's dot on each W row"

LlBf16DotprodDotWord = CompositeDefinition("LlBf16DotprodDot", 1, DOTPROD, _dot_type(P.BF16, P.F32), _dotprod(WORD),
                                           doc=DOTPROD_DOC, conformance=CONFORMANCE)
LlBf16SplitKDotWord = CompositeDefinition("LlBf16SplitKDot", 1, SPLITK, _dot_type(P.BF16, P.F32), _splitk(WORD),
                                          doc=SPLITK_DOC, conformance=CONFORMANCE)
LlBf16DotprodGemmWord = CompositeDefinition("LlBf16DotprodGemm", 1, ("N",) + DOTPROD, _gemm_type(P.BF16, P.F32),
                                            _rows(LlBf16DotprodDotWord, DOTPROD), doc=ROWS_DOC, conformance=CONFORMANCE)
LlBf16SplitKGemmWord = CompositeDefinition("LlBf16SplitKGemm", 1, ("N",) + SPLITK, _gemm_type(P.BF16, P.F32),
                                           _rows(LlBf16SplitKDotWord, SPLITK), doc=ROWS_DOC, conformance=CONFORMANCE)

LlBf16DotprodDot = CompositeDefinition("LlBf16DotprodDot", 2, DOTPROD, _dot_type(BF16, F32), _dotprod(BITS),
                                       doc=f"{DOTPROD_DOC} (on bits)", conformance=CONFORMANCE)
LlBf16SplitKDot = CompositeDefinition("LlBf16SplitKDot", 2, SPLITK, _dot_type(BF16, F32), _splitk(BITS),
                                      doc=f"{SPLITK_DOC} (on bits)", conformance=CONFORMANCE)
LlBf16DotprodGemm = CompositeDefinition("LlBf16DotprodGemm", 2, ("N",) + DOTPROD, _gemm_type(BF16, F32),
                                        _rows(LlBf16DotprodDot, DOTPROD), doc=f"{ROWS_DOC} (on bits)", conformance=CONFORMANCE)
LlBf16SplitKGemm = CompositeDefinition("LlBf16SplitKGemm", 2, ("N",) + SPLITK, _gemm_type(BF16, F32),
                                       _rows(LlBf16SplitKDot, SPLITK), doc=f"{ROWS_DOC} (on bits)", conformance=CONFORMANCE)


def _word(defn, statics):
    return lambda b: bind(defn, **{k: word_of(b[k]) if k == "DOT" else b[k] for k in statics})


LlBf16DotprodDot.word = _word(LlBf16DotprodDotWord, DOTPROD)  # type: ignore[attr-defined]
LlBf16SplitKDot.word = _word(LlBf16SplitKDotWord, SPLITK)  # type: ignore[attr-defined]
LlBf16DotprodGemm.word = _word(LlBf16DotprodGemmWord, ("N",) + DOTPROD)  # type: ignore[attr-defined]
LlBf16SplitKGemm.word = _word(LlBf16SplitKGemmWord, ("N",) + SPLITK)  # type: ignore[attr-defined]

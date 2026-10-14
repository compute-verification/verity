"""[GLM-5.3, DeepSeek-V4-Pro] The block-scaled FP8 linear as vLLM d9105ea80 runs it on sm_90 by default, and DeepSeek-V4's grouped
`wo_a` einsum: DeepGEMM 8b1392b9 (tools/install_deepgemm.sh:11), FlashInfer 0.6.18 (requirements/cuda.txt:17).  Line numbers are at
those commits.

On Hopper `is_deep_gemm_e8m0_used()` holds and `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=1` is the default, so a 128x128-block FP8 linear
is `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` (kernels/linear/scaled_mm/flashinfer.py:149-330).  Its weights are processed by
`deepgemm_post_process_fp8_weight_block` (fp8_utils.py:1111-1175) and are derived Inputs (fp8.md §2: the served words are bound,
`kernels.deepgemm_twins` re-derives them): DeepSeek-V4's e8m0 scales upcast as `byte << 23` (fp8_utils.py:1071-1080: byte 0 is +0,
0xFF is +inf; no requantisation), GLM-5.3's f32 scales requantised to powers of two by `requant_weight_ue8m0_inplace`
(fp8_utils.py:1011-1068).  `apply_block_scaled_mm` passes the bf16 input to `dynamic_flashinfer_deepgemm_blockscale_gemm`, which branches
on the token count M (`torch.cond`, flashinfer.py:301-316; `VLLM_BATCH_INVARIANT` takes the second branch at every M):

M < 32   FlashInfer's `fp8_blockscale_gemm` (TRT-LLM's swapAB kernel), which quantises its input itself: `scale_1x128_kernel`
         (fp8_blockscale_gemm_kernel.cuh:820-895, built with `-use_fast_math`): `FlashInferFp8Quant1x128_v1{K}`.  Neither the e4m3
         words nor the scales leave the op.
M >= 32  `per_token_group_quant_fp8(x, 128, column_major_scales=True, use_ue8m0=True)` (flashinfer.py:277-299) ->
         `_C.per_token_group_fp8_quant` (fp8_utils.py:619-635) -> `per_token_group_quant_8bit_kernel<.., SCALE_UE8M0=true>`
         (per_token_group_quant.cu:42-160, 193-286; nvcc without fast-math): `Fp8GroupQuantUe8m0_v1{K,G}`; then DeepGEMM `fp8_gemm_nt`.

The GEMM of both branches and of the einsum is `fp8.ScaledMmFp8BlockCoordinate_v1{K,128}` per output word: DeepGEMM's sm_90 1d2d
kernel (sm90_fp8_gemm_1d2d.cuh:283-347, 405-420; f32 scales pass `transform_sf_into_required_layout` unchanged on sm_90,
csrc/apis/layout.hpp:41-47) runs four `wgmma.m64nNk32.f32.e4m3.e4m3` steps per 128-wide k-block from `scale-d = 0`, then
`final += (sfa * sfb) * accum` (FMUL, then an FFMA: nvcc contracts, no fast-math in the JIT, csrc/jit/compiler.hpp:212-213) from +0 in
ascending k, then `__float22bfloat162_rn`; TRT-LLM's swapAB kernel and `sm90_fp8_bmm` (`GemmType::Batched` of the same template,
csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:291-330) are the same chain.  So the GEMM is `fp8_vocab.block_gemm(K, N, 128, scale)`,
`ScaledMmFp8BlockSharedScale_v1` by default.

FlashInfer's `scale_1x128_kernel`, one 128-element group per warp, as its sm_90 PTX has it:

    lane l holds x[2l], x[2l+1], x[64+2l], x[65+2l]; v_l = max.bf16 from 0 over abs.bf16 of them   exact, NaN dropped, subnormals kept
    amax  = lane 0 after `v = std::max(float(v), shfl_down(v, o))`, o = 16, 8, 4, 2, 1             setp.lt.ftz.f32 + selp
    scale = setp.neu.bf16(amax, 0) ? div.approx.ftz(448, f32(amax)) : 1.0                         MUFU.RCP, FMUL.FTZ
    stored scale = rcp.approx.ftz(scale);  q[i] = cvt.rn.satfinite.e4m3x2(mul.ftz(scale, f32(x[i])))   no clamp

The compare flushes subnormals, so an all-subnormal group's amax is lane 0's value (every compare false): 0 gives scale 1 and zero
words, a subnormal gives scale +inf, a stored scale of +0 and every word 0x7F.

Registered functions (all `Name_v1`):
  F32LtFtz                          primitive: setp.lt.ftz.f32 -> bit
  Fp8Ue8m0Ceil                      (y: f32) -> f32: the power of two at or above max(|y|, 1e-10)
  Fp8GroupScaleUe8m0{G}             (x: G bf16) -> f32
  Fp8GroupQuantUe8m0Group{G}        (x: G bf16) -> (q: G e4m3, s: f32)
  Fp8GroupQuantUe8m0{K,G}           (x: K bf16) -> (q: K e4m3, s: K/G f32)                    the M >= 32 branch's quantisation
  FlashInferFp8Quant1x128Group      (x: 128 bf16) -> (q: 128 e4m3, s: f32)
  FlashInferFp8Quant1x128{K}        (x: K bf16) -> (q: K e4m3, s: K/128 f32)                  the M < 32 branch's quantisation
  DeepGemmFp8Linear{K,N}            (x: K bf16, w: N x K e4m3, ws: cdiv(N,128) x K/128 f32) -> (q, s, y: N bf16)
  FlashInferFp8Linear{K,N}          (x, w, ws as above) -> y: N bf16
  ScaledMmFp8BlockGrouped{H,K,N,G}  (q: H x K e4m3, s: H x K/G f32, w: H x N x K e4m3, ws: H x cdiv(N,G) x K/G f32) -> H x N bf16

`ScaledMmFp8BlockGrouped` is `fp8_einsum("bhr,hdr->bhd", recipe=(1, 128, 128))` on one token (csrc/apis/einsum.hpp:232-239,
deepseek_v4/nvidia/ops/o_proj.py:28-73): DeepSeek-V4's `wo_a` at TP8 is H = 2 local groups, K = 4096 (8 heads x 512, the e4m3 words
and 32 scales `fused_inv_rope_fp8_quant` writes), N = 1024, the weight [g, 1024, 4096] with scales [g, 8, 32]
(`deepgemm_post_process_fp8_weight_block(is_bmm=True)`, fp8_utils.py:1142-1158).
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.fp32 import ftz_bits, f32_of
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.ops.scalar import BIT, F32Fminf, F32Neg
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry import fp8 as F8
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.fp8 import E4M3, FP8_BLOCK, weight_blocks
from verity_vllm.program.registry.prims import BF16, F32

F_ONE, F_448, F_NEG448 = 0x3F800000, 0x43E00000, 0xC3E00000
F_1EM10 = P.bits_of(1e-10)
#: `Fp8Ue8m0Ceil`'s rounding: adding the mantissa mask carries into the exponent unless the mantissa is zero, then the mask keeps it
UE8M0_ROUND, UE8M0_KEEP = 0x007FFFFF, 0xFF800000
#: the lanes and their elements of FlashInfer's warp-per-group quantiser
FI_LANES = 32
#: (bf16, e4m3, f32): the word types; `boolean_deepgemm` passes their bits
WORD_TYPES = (BF16, E4M3, F32)

_UE8M0_CONF = ("model-untested: read from vLLM d9105ea80 per_token_group_quant.cu:42-160 (nvcc without fast-math); the ue8m0 rounding "
               "written as the bit form of per_token_group_quant.cu:381-387, equal on every reachable scale (Fp8Ue8m0Ceil_v1's doc); "
               "tests/properties/deepgemm_sm90_capture_gpu.py is its hardware check")
_FI_CONF = ("model-untested: read from FlashInfer 0.6.18 fp8_blockscale_gemm_kernel.cuh:820-895 and its sm_90 PTX (-use_fast_math); "
            "MufuRcpFtz_v1's table is the RTX 4090's; tests/properties/deepgemm_sm90_capture_gpu.py is its hardware check")
_GEMM_CONF = ("model-untested as composed: the GEMM is ScaledMmFp8BlockSharedScale_v1 (exact-model-tested against CUTLASS's blockwise "
              "kernel on the H100), read from DeepGEMM 8b1392b9 sm90_fp8_gemm_1d2d.cuh:283-347 and 405-420 as the same chain; "
              "tests/properties/deepgemm_sm90_capture_gpu.py is its hardware check")


@primitive("F32LtFtz", 1, [("a", F32), ("b", F32)], BIT, conformance="exact-model-tested")
def F32LtFtz(a: int, b: int) -> int:
    """`setp.lt.ftz.f32`: 1 iff ftz(a) < ftz(b) (a subnormal compares as the zero of its sign, +0 == -0, a NaN compares false)."""
    x, y = f32_of(ftz_bits(a & 0xFFFFFFFF)), f32_of(ftz_bits(b & 0xFFFFFFFF))
    return 1 if x < y else 0


DEEPGEMM_PRIMITIVES = [F32LtFtz]


class Ops(D.Ops):
    """`dsa.Ops` with ``half(bits)``, a constant 16-bit word."""

    def __init__(self, B, defs: dict, word, neg, half):
        super().__init__(B, defs, word, neg)
        self.half = half


WORD = dict(D.WORD, fminf=F32Fminf, lt_ftz=F32LtFtz, mul_ftz=P.F32MulFtz, rcp_ftz=P.MufuRcpFtz, bf16_gt=P.Bf16GtStrict,
            f2bf16=P.F2fpBf16)


def word_ops(B) -> Ops:
    return Ops(B, WORD, lambda bits: B.call(const(32, bits)), lambda x: B.call(F32Neg, x), lambda bits: B.call(const(16, bits)))


def _row(n: int, words: list, t):
    return Coll(Array(n, t), concat([w.refs for w in words]))


def quant_sig(S, G: int, T=WORD_TYPES):
    """(x: K bf16) -> (q: K e4m3, s: K/G f32), over the types T."""
    bf16, e4m3, f32 = T
    return (("x", Array(S.K, bf16)),), Tuple(Array(S.K, e4m3), Array(S.K // G, f32))


def group_sig(G: int, T=WORD_TYPES):
    """(x: G bf16) -> (q: G e4m3, s: f32), over the types T."""
    bf16, e4m3, f32 = T
    return (("x", Array(G, bf16)),), Tuple(Array(G, e4m3), f32)


def fi_group_sig(S, T=WORD_TYPES):
    return group_sig(FP8_BLOCK, T)


def linear_sig(S, materialise: bool, T=WORD_TYPES):
    """(x: K bf16, w: N x K e4m3, ws: cdiv(N, 128) x K/128 f32) -> (q, s, y: N bf16) when ``materialise``, else y."""
    bf16, e4m3, f32 = T
    y = Array(S.N, bf16)
    return ((("x", Array(S.K, bf16)), ("w", Array(S.N, Array(S.K, e4m3))),
             ("ws", Array(weight_blocks(S.N, FP8_BLOCK), Array(S.K // FP8_BLOCK, f32)))),
            Tuple(Array(S.K, e4m3), Array(S.K // FP8_BLOCK, f32), y) if materialise else y)


def grouped_sig(S, T=WORD_TYPES):
    bf16, e4m3, f32 = T
    return ((("xq", Array(S.H, Array(S.K, e4m3))), ("sx", Array(S.H, Array(S.K // S.G, f32))),
             ("w", Array(S.H, Array(S.N, Array(S.K, e4m3)))), ("ws", Array(S.H, Array(weight_blocks(S.N, S.G), Array(S.K // S.G, f32))))),
            Array(S.H, Array(S.N, bf16)))


def ue8m0_ceil(o: Ops, y):
    """`Fp8Ue8m0Ceil_v1`'s body."""
    m = o.fmaxf(o.fabs(y), o.word(F_1EM10))
    return o.iand(o.iadd(m, o.word(UE8M0_ROUND)), o.word(UE8M0_KEEP))


@composite("Fp8Ue8m0Ceil", 1, [], lambda S: ((("y", F32),), F32), conformance=_UE8M0_CONF)
def Fp8Ue8m0Ceil(B, S, y):
    """The ue8m0 group scale from `y = absmax / 448` (per_token_group_quant.cu:68-71): `exp2f(ceilf(log2f(fmaxf(fabsf(y), 1e-10f))))`,
    written as its bit form `(bits(m) + 0x7FFFFF) & 0xFF800000`, m = `fmaxf(fabsf(y), 1e-10f)` (the register-resident kernel's,
    per_token_group_quant.cu:381-387, "bit-exact with exp2f(ceilf(log2f))" at :293): the power of two at or above m, +inf for
    +inf.  The two agree on every m the kernel reaches: m is 1e-10, +inf, or RN(a / 448) for a bf16 magnitude a, and such a quotient
    is a power of two (a's significand 224) or at least 1/224 relative from one, so log2f's error (at most an ulp, below 2^-16 for
    |log2 m| < 128) cannot carry ceilf across an integer."""
    return ue8m0_ceil(word_ops(B), y)


@composite("Fp8GroupScaleUe8m0", 1, ["G"], lambda S: ((("x", Array(S.G, BF16)),), F32), conformance=_UE8M0_CONF)
def Fp8GroupScaleUe8m0(B, S, x):
    """One group's ue8m0 scale: `Fp8Ue8m0Ceil_v1` of `fp8.Fp8GroupScale_v1{G}` (`fmaxf`-fold from eps = 1e-10 of |f32(x_e)|, then
    `div.rn` by 448)."""
    return B.call(Fp8Ue8m0Ceil, B.call(bind(F8.Fp8GroupScale, G=S.G), x))


def group_quant(o: Ops, G: int, x, scale, T=WORD_TYPES):
    """`Fp8GroupQuantUe8m0Group`'s body, its scale a Call of ``scale`` on the G words: (q, s)."""
    s = o.B.call(scale, x)
    q = [o.e4m3(o.fminf(o.fmaxf(o.div(o.cvt(x[e]), s), o.word(F_NEG448)), o.word(F_448))) for e in range(G)]
    return tuple_of(_row(G, q, T[1]), s)


@composite("Fp8GroupQuantUe8m0Group", 1, ["G"], lambda S: group_sig(S.G), conformance=_UE8M0_CONF)
def Fp8GroupQuantUe8m0Group(B, S, x):
    """One group of `per_token_group_quant_8bit_kernel`: `s = Fp8GroupScaleUe8m0_v1{G}` and `q[e] = cvt.rn.satfinite.e4m3x2(
    fminf(fmaxf(f32(x[e]) / s, -448), 448))` (per_token_group_quant.cu:85-86: `div.rn` by a power of two; a NaN quotient is dropped
    by fmaxf)."""
    return group_quant(word_ops(B), S.G, x, bind(Fp8GroupScaleUe8m0, G=S.G))


def group_row(B, name: str, K: int, G: int, x, group, T=WORD_TYPES):
    """A row quantiser's body: the K/G groups' Calls of ``group``, their words and scales concatenated."""
    if K % G:
        raise ValueError(f"{name}: K={K} is not a multiple of G={G}")
    parts = [B.call(group, x[g * G:(g + 1) * G]) for g in range(K // G)]
    return tuple_of(Coll(Array(K, T[1]), concat([p[0].refs for p in parts])),
                    Coll(Array(K // G, T[2]), concat([p[1].refs for p in parts])))


@composite("Fp8GroupQuantUe8m0", 1, ["K", "G"], lambda S: quant_sig(S, S.G), conformance=_UE8M0_CONF)
def Fp8GroupQuantUe8m0(B, S, x):
    """`_C.per_token_group_fp8_quant(.., use_ue8m0=True)` on one bf16 row: group g is `Fp8GroupQuantUe8m0Group_v1{G}` over
    x[Gg:Gg+G].  Both outputs are materialised (the M >= 32 branch's `q_input`, `input_scale`)."""
    return group_row(B, "Fp8GroupQuantUe8m0", S.K, S.G, x, bind(Fp8GroupQuantUe8m0Group, G=S.G))


def fi_lane_elements(l: int) -> tuple[int, ...]:
    return (2 * l, 2 * l + 1, 64 + 2 * l, 65 + 2 * l)


def fi_group(o: Ops, x, T=WORD_TYPES):
    """`FlashInferFp8Quant1x128Group_v1`'s body: (q words, stored scale)."""
    zero = o.word(0)
    xf = [o.cvt(x[i]) for i in range(FP8_BLOCK)]
    lanes = []
    for l in range(FI_LANES):
        v = zero
        for i in fi_lane_elements(l):
            v = o.fmaxf(v, o.fabs(xf[i]))
        lanes.append(v)
    amax = D.halving_tree(lanes, lambda a, b: o.isel(o.lt_ftz(a, b), b, a))
    nz = o.bf16_gt(o.f2bf16(amax), o.half(0))
    scale = o.isel(nz, o.mul_ftz(o.rcp_ftz(amax), o.word(F_448)), o.word(F_ONE))
    q = [o.e4m3(o.mul_ftz(scale, xf[i])) for i in range(FP8_BLOCK)]
    return tuple_of(_row(FP8_BLOCK, q, T[1]), o.rcp_ftz(scale))


@composite("FlashInferFp8Quant1x128Group", 1, [], fi_group_sig, conformance=_FI_CONF)
def FlashInferFp8Quant1x128Group(B, S, x):
    """One 128-element group of FlashInfer's `scale_1x128_kernel` (the module docstring's sequence): lane l's value an `fmaxf` fold from
    +0 of |f32(x)| over its four elements (= max.bf16 of abs.bf16: exact, NaN dropped); amax the lane-0 halving tree of
    `SelectI32(F32LtFtz_v1(a, b), b, a)` (a lane t, b lane t + o); `scale = Bf16GtStrict(bf16(amax), 0) ? F32MulFtz(MufuRcpFtz(amax),
    448) : 1.0` (amax is never negative or NaN, so `> 0` is setp.neu's `!= 0`, and bf16(amax) is exact); q[i] =
    `F32ToE4m3Sat(F32MulFtz(scale, f32(x[i])))`; the stored scale `MufuRcpFtz(scale)`."""
    return fi_group(word_ops(B), x)


@composite("FlashInferFp8Quant1x128", 1, ["K"], lambda S: quant_sig(S, FP8_BLOCK), conformance=_FI_CONF)
def FlashInferFp8Quant1x128(B, S, x):
    """`scale_1x128_kernel` on one bf16 row: group g is `FlashInferFp8Quant1x128Group_v1` over x[128g:128g+128]; the e4m3 words and
    the stored scales are what the TRT-LLM GEMM reads (its workspace)."""
    return group_row(B, "FlashInferFp8Quant1x128", S.K, FP8_BLOCK, x, FlashInferFp8Quant1x128Group)


def linear(B, quant, gemm, x, w, ws, materialise: bool):
    """A linear's body on one token row: ``gemm`` over ``quant``'s (q, s); (q, s, y) when ``materialise``, else y."""
    q = B.call(quant, x)
    y = B.call(gemm, q[0], q[1], w, ws)
    return tuple_of(q[0], q[1], y) if materialise else y


@composite("DeepGemmFp8Linear", 1, ["K", "N"], lambda S: linear_sig(S, True), conformance=_GEMM_CONF)
def DeepGemmFp8Linear(B, S, x, w, ws):
    """The linear's M >= 32 branch on one token row: `Fp8GroupQuantUe8m0_v1{K,128}`, then `ScaledMmFp8BlockSharedScale_v1{K,N,128}`
    (DeepGEMM `fp8_gemm_nt`); returns (q, s, y), the three values vLLM materialises."""
    return linear(B, bind(Fp8GroupQuantUe8m0, K=S.K, G=FP8_BLOCK), bind(F8.ScaledMmFp8BlockSharedScale, K=S.K, N=S.N, G=FP8_BLOCK),
                  x, w, ws, True)


@composite("FlashInferFp8Linear", 1, ["K", "N"], lambda S: linear_sig(S, False), conformance=_GEMM_CONF)
def FlashInferFp8Linear(B, S, x, w, ws):
    """The linear's M < 32 branch on one token row: `FlashInferFp8Quant1x128_v1{K}`, then `ScaledMmFp8BlockSharedScale_v1{K,N,128}`
    (TRT-LLM's swapAB kernel, the stored scales as the activation's); returns y, the op's one output."""
    return linear(B, bind(FlashInferFp8Quant1x128, K=S.K), bind(F8.ScaledMmFp8BlockSharedScale, K=S.K, N=S.N, G=FP8_BLOCK), x, w, ws,
                  False)


@composite("ScaledMmFp8BlockGrouped", 1, ["H", "K", "N", "G"], grouped_sig, conformance=_GEMM_CONF)
def ScaledMmFp8BlockGrouped(B, S, xq, sx, w, ws):
    """`fp8_einsum("bhr,hdr->bhd", recipe=(1, G, G))` on one token: group h's N words are `ScaledMmFp8BlockSharedScale_v1{K,N,G}` over
    its e4m3 row xq[h], scales sx[h], weight w[h] and block scales ws[h] (`sm90_fp8_bmm`: the 1d2d chain per group)."""
    return B.batch(bind(F8.ScaledMmFp8BlockSharedScale, K=S.K, N=S.N, G=S.G), xq, sx, w, ws, axes=(0, 0, 0, 0))


DEEPGEMM_COMPOSITES = [Fp8Ue8m0Ceil, Fp8GroupScaleUe8m0, Fp8GroupQuantUe8m0Group, Fp8GroupQuantUe8m0, FlashInferFp8Quant1x128Group,
                       FlashInferFp8Quant1x128, DeepGemmFp8Linear, FlashInferFp8Linear, ScaledMmFp8BlockGrouped]

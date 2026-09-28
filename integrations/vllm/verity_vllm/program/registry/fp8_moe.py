"""[GLM-5.3 fp8] The block-scaled FP8 expert GEMMs of vLLM's fused MoE on Hopper: `fused_moe_kernel` (Triton) with `block_shape`
[128, 128], as the H200 runs GLM-5.3-FP8 (`quant_method` fp8, `fmt` e4m3, `weight_block_size` [128, 128], `activation_scheme`
dynamic, no `scale_fmt`) at TP without EP, eager, `VLLM_USE_DEEP_GEMM=0`; GLM-5.3's engine refuses `VLLM_BATCH_INVARIANT` on sm_90
(no sparse-MLA backend supports it), so its experts launch the tuned configs.  Pinned vLLM 0.28.1rc1.dev472 (d9105ea80), Triton
3.7.1; line numbers are at that commit.

The path, read from the source:

1. `Fp8MoEMethod` -> `select_fp8_moe_backend`: on cc 9.0 with block FP8 (`kFp8Static128BlockSym` weights, `kFp8Dynamic128Sym`
   activations) and `ep_size == 1`, `_get_priority_backends` moves TRITON to the front (fused_moe/oracle/fp8.py:113-123), so the
   experts are `TritonExperts` (batch invariant: `_supports_batch_invariance`; fused_moe/experts/triton_moe.py).
2. The block's input is `per_token_group_quant_fp8(x, 128)` (`moe_kernel_quantize_input` -> `_fp8_quantize`, fused_moe/utils.py:239-243;
   `use_ue8m0` = `is_deep_gemm_e8m0_used()`, False with DeepGEMM off): the kernel `fp8.Fp8GroupQuant_v1` already states.
3. `invoke_fused_moe_triton_kernel` (fused_moe/fused_moe.py:763-910) launches `fused_moe_kernel` (fused_moe.py:299-610) twice, w13
   (`mul_routed_weight` False) then w2 (`not apply_router_weight_on_input`, top_k 1; triton_moe.py:505-527), with the config of
   `try_get_optimal_moe_config` (fused_moe.py:1422-1451), `BLOCK_SIZE_K = min(BLOCK_SIZE_K, 128)` (fused_moe.py:844-846) and
   `SWAP_AB = enable_swap_ab(BLOCK_SIZE_M, BLOCK_SIZE_N)` (cc 9.0, BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64; fused_moe/utils.py:628-633).
   Without batch invariance the config is the tuned H200 file's entry at the nearest batch size (for the TP8 shard, `E=256,N=256,
   device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`: BLOCK_SIZE_K 128, and BLOCK_SIZE_M 8 or 16 under SWAP_AB from
   M = 4 to 512), or `get_default_config`'s block-FP8 tile where no file exists (whole experts: BLOCK_SIZE_K 128, BLOCK_SIZE_M 16
   under SWAP_AB up to M = 64); the Definitions take KB and SWAP as statics.  Under `VLLM_BATCH_INVARIANT` it is `get_default_config`'s
   fixed tile BLOCK_SIZE_M 64, N 64, K 32 (fused_moe.py:1308-1315): KB 32, SWAP_AB False.  The tuned file's M <= 3 entries,
   BLOCK_SIZE_M 8 x BLOCK_SIZE_N 32, are no wgmma tile (SWAP_AB needs BLOCK_SIZE_N >= 64), and no Definition here models their dot.
4. Between them, `ops.silu_and_mul_per_block_quant(group_size=128)` (triton_moe.py:471-485), the fused branch every GLM-5.3 layer
   takes (SiLU, fp8_w8a8, block [128, 128], no LoRA, no e8m0): `SiluMulFp8BlockQuant_v1` below.  Its (q, scale) is the w2 launch's
   (A, A_scale); no bf16 intermediate exists.

`fused_moe_kernel`'s block-wise branch (fused_moe.py:542-552), per K tile k of BLOCK_SIZE_K = KB:

    a_scale = A_scale[token, k*KB // 128];  b_scale = B_scale[e, n // 128, k*KB // 128]
    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]          (SWAP_AB: tl.dot(b, a) * b_scale * a_scale)

then `accumulator *= topk_weight` when MUL_ROUTED_WEIGHT (fused_moe.py:593-599) and `accumulator.to(bfloat16)` (fused_moe.py:603).
Per output coordinate this is:

* `tl.dot(a, b)` with no accumulator is a fresh f32 chain from +0: on cc 9.0 an FP8 dot takes `max_num_imprecise_acc = 2**30`
  (triton/backends/nvidia/compiler.py:204, language/semantic.py:1571-1573), so the whole dot is one wgmma chain with no promotion
  inside it, KB/32 `HopperE4m3QgmmaDot32` steps (wgmma k32 e4m3) in ascending k, the first from +0.
* `(dot * a_scale) * b_scale` is two FMULs left to right, and Triton compiles with `enable_fp_fusion` on (compiler.py:117, 201-202,
  447), so LLVM contracts the add into the second: `acc = FFMA(FMUL(dot, a_scale), b_scale, acc)` from acc = +0, the tiles in ascending
  k.  With SWAP_AB the scales swap places: `FFMA(FMUL(dot, b_scale), a_scale, acc)`.  The dot itself is the same either way (each
  e4m3 x e4m3 product is exact and the step sums in k order).
* the routed weight: one FMUL `acc * g` (f32 topk weight), then `cvt.rn.bf16.f32`.

So the f32 promotion here is per BLOCK_SIZE_K tile (every 32 elements under batch invariance), where CUTLASS's blockwise GEMM
(`fp8.ScaledMmFp8Block_v1`) promotes per 128 with `FFMA(temp, FMUL(sfa, sfb), acc)`: the two GEMMs are different functions.

Registered functions (all `Name_v1`; statics K, N, G = 128 the scale block, KB = BLOCK_SIZE_K, SWAP = SWAP_AB):
  MoeExpertFp8BlockCoordinate{K,G,KB,SWAP}    (xq, sx, wrow, sw) -> bf16: one w13 output word
  MoeExpertFp8BlockCoordinateW{K,G,KB,SWAP}   (xq, sx, g, wrow, sw) -> bf16: one w2 output word (routed weight g)
  MoeExpertFp8BlockGemm{K,N,G,KB,SWAP}        the w13 launch on one (token, slot) pair: the N coordinates over the expert's rows
  MoeExpertFp8BlockGemmW{K,N,G,KB,SWAP}       the w2 launch on one pair
and `Name_v3{..., DOT}` of each, the same body with the e4m3 k32 step `DOT` (the target's `tl.dot` step: `BlackwellE4m3QmmaDot32_v1`
on cc 12.0, where SWAP_AB is never taken and the block-[64, 64] checkpoints launch at G = 64).
The expert's e4m3 weight rows `w` [N, K] and their f32 block scales `ws` [cdiv(N, G), K/G] (`w13_weight_scale_inv` /
`w2_weight_scale_inv`, the checkpoint's words) are operands; `xq` / `sx` are the pair's quantised activation and its group scales.

`silu_and_mul_per_block_quant_kernel` (csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu, built with
VLLM_GPU_FLAGS: no `--use_fast_math`, so `expf` is the math library's sequence and `/` is `div.rn`), one thread block per (row, group),
one element per thread, on the w13 output viewed [pairs, 2I] = [gate || up]:

    g = f32(gate[e]); u = f32(up[e]); result = (g * (1.0f / (1.0f + expf(-g)))) * u              (cu:54-60)
    absmax = shared-memory tree of fmaxf over fabsf(result), stride G/2 down to 1                  (cu:63-73)
    scale = fmaxf(absmax / 448.0f, 1.0f / (448.0f * 512.0f))     (`scale_ub` None at the call site; cu:76-94)
    q[e] = e4m3(fmaxf(-448, fminf(result / scale, 448)))         (`ScaledQuant<fp8, false>`, quant_conversions.cuh:40-45, 70-74)

The scale is `fp8.Fp8RowScale_v1` and the clamp `fp8.Fp8ActQuant_v1`'s (NaN -> 448); `act.SiluMul_v2` rounds to bf16 and is not this
function.  `scales` is [pairs, I/G] row-major (`is_scale_transposed` False, _custom_ops.py:493), the w2 launch's A_scale.

  SiluMulFp8BlockQuantResult                  (gate, up: bf16) -> f32: one thread's `result`
  SiluMulFp8BlockQuantGroup{G}                (gate, up: G bf16) -> (q: G e4m3, scale: f32): one thread block
  SiluMulFp8BlockQuant{I,G}                   (gu: 2I bf16) -> (q: I e4m3, scale: I/G f32): one (token, slot) row
"""
from __future__ import annotations

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.prims import F32ToE4m3Sat, HopperE4m3QgmmaDot32
from verity.ml.circuits.ops.scalar import F32Fabs, F32Fmaxf, F32Fminf, F32Neg
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.fp8 import C448, C_NEG448, E4M3, Fp8RowScale, weight_blocks
from verity_vllm.program.registry.moe import ONE32, NvExpf
from verity_vllm.program.registry.prims import BF16, F32


_CONF = ("model-untested: read from vLLM d9105ea80 fused_moe_kernel and Triton 3.7.1; tests/properties/glm53_block_fp8_capture_gpu.py "
         "is its hardware check")


def check_statics(K: int, G: int, KB: int) -> None:
    if not (KB % 32 == 0 and G % KB == 0 and K % G == 0):
        raise ValueError(f"fused_moe_kernel block FP8 needs KB % 32 == 0, G % KB == 0 and K % G == 0 (K={K}, G={G}, KB={KB})")


def triton_block_acc(B, K: int, G: int, KB: int, SWAP: bool, xq, sx, wrow, sw, zero, dot, fmul, ffma):
    """The f32 accumulator of one coordinate: per K tile k (ascending), `d` = KB/32 `dot` steps from +0, then
    `acc = ffma(fmul(d, first), second, acc)` from +0 with (first, second) = (sx[kb], sw[kb]), swapped under SWAP, kb = k*KB // G."""
    check_statics(K, G, KB)
    acc = zero()
    for k in range(K // KB):
        d = zero()
        for st in range(KB // 32):
            lo = k * KB + 32 * st
            d = B.call(dot, d, xq[lo:lo + 32], wrow[lo:lo + 32])
        kb = k * KB // G
        first, second = (sw[kb], sx[kb]) if SWAP else (sx[kb], sw[kb])
        acc = B.call(ffma, B.call(fmul, d, first), second, acc)
    return acc


def _acc(B, S, xq, sx, wrow, sw, dot=HopperE4m3QgmmaDot32):
    return triton_block_acc(B, S.K, S.G, S.KB, bool(S.SWAP), xq, sx, wrow, sw, lambda: B.call(b1.ZERO32), dot, P.F32Mul, P.F32Fma)


@composite("MoeExpertFp8BlockCoordinate", 1, ["K", "G", "KB", "SWAP"],
           lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("wrow", Array(S.K, E4M3)), ("sw", Array(S.K // S.G, F32))),
                      BF16),
           conformance=_CONF)
def MoeExpertFp8BlockCoordinate(B, S, xq, sx, wrow, sw):
    """One output word of `fused_moe_kernel`'s w13 launch (MUL_ROUTED_WEIGHT False): the K/KB tile promotions
    `acc = FFMA(FMUL(dot, sx[kb]), sw[kb], acc)` (SWAP: `FFMA(FMUL(dot, sw[kb]), sx[kb], acc)`), each `dot` KB/32
    `HopperE4m3QgmmaDot32` steps from +0, then `cvt.rn.bf16.f32(acc)`."""
    return B.call(P.F2fpBf16, _acc(B, S, xq, sx, wrow, sw))


@composite("MoeExpertFp8BlockCoordinateW", 1, ["K", "G", "KB", "SWAP"],
           lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("g", F32), ("wrow", Array(S.K, E4M3)),
                       ("sw", Array(S.K // S.G, F32))), BF16),
           conformance=_CONF)
def MoeExpertFp8BlockCoordinateW(B, S, xq, sx, g, wrow, sw):
    """One output word of the w2 launch (MUL_ROUTED_WEIGHT True): `MoeExpertFp8BlockCoordinate_v1`'s accumulator, then
    `cvt.rn.bf16.f32(FMUL(acc, g))`, g the pair's f32 topk weight (`accumulator *= moe_weight[:, None]`, fused_moe.py:593-599)."""
    return B.call(P.F2fpBf16, B.call(P.F32Mul, _acc(B, S, xq, sx, wrow, sw), g))


def rows(B, coord, N: int, G: int, w, ws, lead: tuple, word=BF16):
    """The N coordinates of one pair, weight block nb over rows `w[nb*G:(nb+1)*G]` (the last one partial when G does not divide N) and
    the scale row `ws[nb]`; `lead` the operands every coordinate reads whole."""
    axes = (None,) * len(lead) + (0, None)
    parts = [B.batch(coord, *lead, w[nb * G:min((nb + 1) * G, N)], ws[nb], axes=axes) for nb in range(weight_blocks(N, G))]
    return Coll(Array(N, word), concat([p.refs for p in parts]))


@composite("MoeExpertFp8BlockGemm", 1, ["K", "N", "G", "KB", "SWAP"],
           lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("w", Array(S.N, Array(S.K, E4M3))),
                       ("ws", Array(weight_blocks(S.N, S.G), Array(S.K // S.G, F32)))), Array(S.N, BF16)),
           conformance=_CONF)
def MoeExpertFp8BlockGemm(B, S, xq, sx, w, ws):
    """`fused_moe_kernel`'s w13 launch on one (token, slot) pair: y[n] = `MoeExpertFp8BlockCoordinate_v1`(xq, sx, w[n], ws[n // G])
    over the pair's expert's rows (B_scale indexed by `offs_bn // group_n`, fused_moe.py:490-493)."""
    coord = bind(MoeExpertFp8BlockCoordinate, K=S.K, G=S.G, KB=S.KB, SWAP=S.SWAP)
    return rows(B, coord, S.N, S.G, w, ws, (xq, sx))


@composite("MoeExpertFp8BlockGemmW", 1, ["K", "N", "G", "KB", "SWAP"],
           lambda S: ((("xq", Array(S.K, E4M3)), ("sx", Array(S.K // S.G, F32)), ("g", F32), ("w", Array(S.N, Array(S.K, E4M3))),
                       ("ws", Array(weight_blocks(S.N, S.G), Array(S.K // S.G, F32)))), Array(S.N, BF16)),
           conformance=_CONF)
def MoeExpertFp8BlockGemmW(B, S, xq, sx, g, w, ws):
    """`fused_moe_kernel`'s w2 launch on one pair: y[n] = `MoeExpertFp8BlockCoordinateW_v1`(xq, sx, g, w[n], ws[n // G])."""
    coord = bind(MoeExpertFp8BlockCoordinateW, K=S.K, G=S.G, KB=S.KB, SWAP=S.SWAP)
    return rows(B, coord, S.N, S.G, w, ws, (xq, sx, g))


# ---------------------------------------------------------------------------------------------------------
# the four with the e4m3 k32 step as a static: `tl.dot`'s step on the declared target (`frontend.targets.moe_expert_fp8_dot`), the
# `_v1` bodies with `DOT` for `HopperE4m3QgmmaDot32_v1`; `_v2` is each `_v1`'s Boolean version
# ---------------------------------------------------------------------------------------------------------

DOT_CONF = ("exact-model-tested at the one DOT a registered target binds (GemmTarget.fp8_moe_dot), BlackwellE4m3QmmaDot32_v1 at G = 64, "
             "KB = 32, SWAP = False (cc 12.0): property blackwell_fp8_capture on an RTX PRO 6000, run r20261003-220018-3f6a, record "
             "0d346c79c642a0a1 (#1000 @ 2fd4e257f), cases moe/decode, moe/small, moe/block and "
             "moe/prefill: fused_experts (E 128, top 8, hidden 4096, I 192) under TritonExperts, every block-[64, 64] fused_moe_kernel "
             "launch at 64x64x32, 11,791,360 w13 and w2 words, 0 mismatches; Hopper's step, the swapped and CUTLASS scale orders and an "
             "unfused mul + add each refuted (3,942, 577, 568 and 575 words); read from vLLM d9105ea80 fused_moe_kernel and Triton 3.7.1 "
             "(each 32-wide tile one QMMA.16832.F32.E4M3.E4M3 from RZ); at DOT = HopperE4m3QgmmaDot32_v1 the function of the `_v1` "
             "forms (tests/program/test_fp8_moe_dot.py), which are model-untested")


def coordinate_sig(routed: bool, e4m3=E4M3, f32=F32, bf16=BF16):
    g = (("g", f32),) if routed else ()
    return lambda S: ((("xq", Array(S.K, e4m3)), ("sx", Array(S.K // S.G, f32))) + g
                      + (("wrow", Array(S.K, e4m3)), ("sw", Array(S.K // S.G, f32))), bf16)


def gemm_sig(routed: bool, e4m3=E4M3, f32=F32, bf16=BF16):
    g = (("g", f32),) if routed else ()
    return lambda S: ((("xq", Array(S.K, e4m3)), ("sx", Array(S.K // S.G, f32))) + g
                      + (("w", Array(S.N, Array(S.K, e4m3))), ("ws", Array(weight_blocks(S.N, S.G), Array(S.K // S.G, f32)))),
                      Array(S.N, bf16))


@composite("MoeExpertFp8BlockCoordinate", 3, ["K", "G", "KB", "SWAP", "DOT"], coordinate_sig(False), conformance=DOT_CONF)
def MoeExpertFp8BlockCoordinateV3(B, S, xq, sx, wrow, sw):
    """`MoeExpertFp8BlockCoordinate_v1` with the e4m3 k32 step `DOT`: each tile's dot KB/32 `DOT` steps from +0."""
    return B.call(P.F2fpBf16, _acc(B, S, xq, sx, wrow, sw, S.DOT))


@composite("MoeExpertFp8BlockCoordinateW", 3, ["K", "G", "KB", "SWAP", "DOT"], coordinate_sig(True), conformance=DOT_CONF)
def MoeExpertFp8BlockCoordinateWV3(B, S, xq, sx, g, wrow, sw):
    """`MoeExpertFp8BlockCoordinateW_v1` with the e4m3 k32 step `DOT`."""
    return B.call(P.F2fpBf16, B.call(P.F32Mul, _acc(B, S, xq, sx, wrow, sw, S.DOT), g))


@composite("MoeExpertFp8BlockGemm", 3, ["K", "N", "G", "KB", "SWAP", "DOT"], gemm_sig(False), conformance=DOT_CONF)
def MoeExpertFp8BlockGemmV3(B, S, xq, sx, w, ws):
    """`MoeExpertFp8BlockGemm_v1` with the e4m3 k32 step `DOT`: y[n] = `MoeExpertFp8BlockCoordinate_v3`(xq, sx, w[n], ws[n // G])."""
    coord = bind(MoeExpertFp8BlockCoordinateV3, K=S.K, G=S.G, KB=S.KB, SWAP=S.SWAP, DOT=S.DOT)
    return rows(B, coord, S.N, S.G, w, ws, (xq, sx))


@composite("MoeExpertFp8BlockGemmW", 3, ["K", "N", "G", "KB", "SWAP", "DOT"], gemm_sig(True), conformance=DOT_CONF)
def MoeExpertFp8BlockGemmWV3(B, S, xq, sx, g, w, ws):
    """`MoeExpertFp8BlockGemmW_v1` with the e4m3 k32 step `DOT`: y[n] = `MoeExpertFp8BlockCoordinateW_v3`(xq, sx, g, w[n], ws[n // G])."""
    coord = bind(MoeExpertFp8BlockCoordinateWV3, K=S.K, G=S.G, KB=S.KB, SWAP=S.SWAP, DOT=S.DOT)
    return rows(B, coord, S.N, S.G, w, ws, (xq, sx, g))


_ACT_CONF = ("model-untested: read from vLLM d9105ea80 fused_silu_mul_block_quant.cu and its build flags; "
             "tests/properties/glm53_block_fp8_capture_gpu.py is its hardware check")


def check_act_statics(I: int, G: int) -> None:
    if not (G > 0 and G & (G - 1) == 0 and I > 0 and I % G == 0):
        raise ValueError(f"silu_and_mul_per_block_quant needs a power-of-two G dividing I (I={I}, G={G})")


def max_tree(B, vals: list, fmaxf):
    """The kernel's shared-memory reduction (cu:63-73): for stride G/2, G/4, ..., 1, slot t < stride becomes `fmaxf(v[t], v[t + stride])`."""
    while len(vals) > 1:
        h = len(vals) // 2
        vals = [B.call(fmaxf, vals[t], vals[t + h]) for t in range(h)]
    return vals[0]


def act_row(B, group, I: int, G: int, gu, e4m3=E4M3, f32=F32):
    """The I/G groups of one row [gate || up]: group g over gate[gG:(g+1)G] and up[gG:(g+1)G], its words and scale in order."""
    check_act_statics(I, G)
    parts = [B.call(group, gu[g * G:(g + 1) * G], gu[I + g * G:I + (g + 1) * G]) for g in range(I // G)]
    return tuple_of(Coll(Array(I, e4m3), concat([p[0].refs for p in parts])), Coll(Array(I // G, f32), concat([p[1].refs for p in parts])))


@composite("SiluMulFp8BlockQuantResult", 1, [], lambda S: ((("gate", BF16), ("up", BF16)), F32), conformance=_ACT_CONF)
def SiluMulFp8BlockQuantResult(B, S, gate, up):
    """One thread's `result` (cu:54-60): g = f32(gate), `(g * (1.0f / (1.0f + expf(-g)))) * f32(up)`, `expf` = `NvExpf_v1`, the
    reciprocal an IEEE `rcp.rn` (the value of `div.rn`), each step rounded to nearest with subnormals kept.  A contraction of
    `1.0f + expf(.)` into an FFMA changes no word: `expf` ends with an exact product by a power of two."""
    g = B.call(P.Bf16ToF32, gate)
    one = B.call(ONE32)
    sig = B.call(P.F32Div, one, B.call(P.F32Add, one, B.call(NvExpf, B.call(F32Neg, g))))
    return B.call(P.F32Mul, B.call(P.F32Mul, g, sig), B.call(P.Bf16ToF32, up))


@composite("SiluMulFp8BlockQuantGroup", 1, ["G"],
           lambda S: ((("gate", Array(S.G, BF16)), ("up", Array(S.G, BF16))), Tuple(Array(S.G, E4M3), F32)), conformance=_ACT_CONF)
def SiluMulFp8BlockQuantGroup(B, S, gate, up):
    """One thread block: the G words r of `SiluMulFp8BlockQuantResult_v1`, absmax = `max_tree` over `fabsf(r)`, scale =
    `Fp8RowScale_v1`(absmax), and `q[e] = e4m3(fmaxf(-448, fminf(r[e] / scale, 448)))` (a true division; inside the clamp the
    satfinite and c10's software conversion agree).  A NaN r is dropped by the fmaxf tree and quantises to 448."""
    check_act_statics(S.G, S.G)
    r = [B.call(SiluMulFp8BlockQuantResult, gate[e], up[e]) for e in range(S.G)]
    s = B.call(Fp8RowScale, max_tree(B, [B.call(F32Fabs, x) for x in r], F32Fmaxf))
    c448, cneg = B.call(C448), B.call(C_NEG448)
    q = [B.call(F32ToE4m3Sat, B.call(F32Fmaxf, cneg, B.call(F32Fminf, B.call(P.F32Div, x, s), c448))) for x in r]
    return tuple_of(Coll(Array(S.G, E4M3), concat([w.refs for w in q])), s)


@composite("SiluMulFp8BlockQuant", 1, ["I", "G"],
           lambda S: ((("gu", Array(2 * S.I, BF16)),), Tuple(Array(S.I, E4M3), Array(S.I // S.G, F32))), conformance=_ACT_CONF)
def SiluMulFp8BlockQuant(B, S, gu):
    """`ops.silu_and_mul_per_block_quant(intermediate_cache1.view(-1, 2I), group_size=G)` on one (token, slot) row gu = [gate || up]
    of the w13 launch's output: (q, scale), that row of `qintermediate_cache2` [pairs, I] e4m3 and of `a2q_scale` [pairs, I/G] f32,
    group g `SiluMulFp8BlockQuantGroup_v1{G}`."""
    return act_row(B, bind(SiluMulFp8BlockQuantGroup, G=S.G), S.I, S.G, gu)


FP8_MOE_COMPOSITES = [MoeExpertFp8BlockCoordinate, MoeExpertFp8BlockCoordinateW, MoeExpertFp8BlockGemm, MoeExpertFp8BlockGemmW,
                      MoeExpertFp8BlockCoordinateV3, MoeExpertFp8BlockCoordinateWV3, MoeExpertFp8BlockGemmV3, MoeExpertFp8BlockGemmWV3,
                      SiluMulFp8BlockQuantResult, SiluMulFp8BlockQuantGroup, SiluMulFp8BlockQuant]

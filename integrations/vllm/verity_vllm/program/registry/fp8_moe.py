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
4. Between them, `ops.silu_and_mul_per_block_quant(group_size=128)` (triton_moe.py:471-485; csrc fused_silu_mul_block_quant.cu): its
   own Definition, not this module's.

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
The expert's e4m3 weight rows `w` [N, K] and their f32 block scales `ws` [cdiv(N, G), K/G] (`w13_weight_scale_inv` /
`w2_weight_scale_inv`, the checkpoint's words) are operands; `xq` / `sx` are the pair's quantised activation and its group scales.
"""
from __future__ import annotations

from verity.ir.defs import bind, composite
from verity.ir.refs import Coll, concat
from verity.ir.types import Array
from verity.ml.prims import HopperE4m3QgmmaDot32
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.fp8 import E4M3, weight_blocks
from verity_vllm.program.registry.prims import BF16, F32

#: `get_default_config` under VLLM_BATCH_INVARIANT (fused_moe.py:1308-1315) clamped by the block (fused_moe.py:844-846)
BATCH_INVARIANT_KB = 32
BATCH_INVARIANT_SWAP = False

_CONF = ("model-untested: read from vLLM d9105ea80 fused_moe_kernel and Triton 3.7.1; the H200 capture "
         "(tests/properties/glm53_block_fp8_capture_gpu.py) is its hardware check")


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


def _acc(B, S, xq, sx, wrow, sw):
    return triton_block_acc(B, S.K, S.G, S.KB, bool(S.SWAP), xq, sx, wrow, sw, lambda: B.call(b1.ZERO32), HopperE4m3QgmmaDot32,
                            P.F32Mul, P.F32Fma)


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


FP8_MOE_COMPOSITES = [MoeExpertFp8BlockCoordinate, MoeExpertFp8BlockCoordinateW, MoeExpertFp8BlockGemm, MoeExpertFp8BlockGemmW]

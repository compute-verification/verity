"""Composites of the B1 (Qwen2.5-1.5B, vLLM eager, `VLLM_BATCH_INVARIANT=1`) typed serving program.

Every composite body is explicit dataflow over one-output primitives (`prims.py`); production kernel
granularity does not determine circuit granularity (IR §C.1): a GEMM is one `AmpereBF16TcDot16` gate per
16-wide accumulator-coordinate transition, an attention head is the FA2 online softmax written out gate
by gate, an RMSNorm is the Triton kernel's exact reduction tree.

Static configuration (`B1_CONFIG`) is the model/profile record the definitions are specialised with;
conformance status is a separate record (`conformance.py`). Dynamic quantities (position `T`, prompt
length `LP`, number of decode steps) are static parameters of specialised definitions, never values.

Naming: static parameters in CAPITALS; `Coll` arguments in lower case; composite names `Name_vN`.
"""

from __future__ import annotations

import math

import numpy as np

from verity.ml.circuits.defs import Coll, CompositeDefinition, PrimitiveDefinition, bind, composite
from verity.ml.circuits.refs import Affine, Coll as _Coll, Explicit, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Record, Tuple
from verity.ml.circuits.ops.gemm import DotBf16 as DotBf16V2, Gemm as GemmV2, GemmCoordinate as GemmCoordinateV2
from verity.ml.circuits.ops.prims import ZERO32, const
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import A16, BF16, BIT, F32, I32

# ---------------------------------------------------------------------------------------------------------
# static configuration of the B1 profile
# ---------------------------------------------------------------------------------------------------------

B1_CONFIG = {
    "model": "Qwen/Qwen2.5-1.5B",
    "revision": "8faed761d45a263340a0528343f099c05c9a4323",
    "H": 1536, "NL": 28, "NH": 12, "KVH": 2, "D": 128, "I": 8960, "V": 151936,
    "QKV": 2048, "GU": 17920, "EPS": 1e-6, "ROPE_THETA": 1_000_000.0, "MAX_POS": 131072, "BIAS": True,
    "tie_word_embeddings": True,
    "profile": {"attention": "FA2 flash_fwd_splitkv_kernel<128,64,64,4> num_splits=1, causal, --use_fast_math",
                "linear": "vllm batch_invariant matmul_kernel_persistent (Triton), bias as a separate aten.add",
                "rmsnorm": "layers.0.input_layernorm: batch_invariant Triton _rms_norm_kernel; the other 55 norms: CUDA _C.fused_add_rms_norm",
                "rope": "vllm _C.rotary_embedding (NeoX), bf16 cos/sin cache",
                "act": "vllm _C.silu_and_mul", "sampler": "greedy argmax over bf16 logits"},
}

# B0 (expansion oracle): HuggingFaceTB/SmolLM2-135M, LlamaForCausalLM — same registered functions, different
# static configuration (no QKV bias, head dim 64, 9/3 heads, 30 layers). The FA2 kernel trait for head dim 64
# is not established from a profiler record, so `Attention` for B0 is `model-untested` (see typed-b1.md).
B0_CONFIG = {
    "model": "HuggingFaceTB/SmolLM2-135M",
    "revision": None,
    "H": 576, "NL": 30, "NH": 9, "KVH": 3, "D": 64, "I": 1536, "V": 49152,
    "QKV": 960, "GU": 3072, "EPS": 1e-5, "ROPE_THETA": 100_000.0, "MAX_POS": 8192, "BIAS": False,
    "tie_word_embeddings": True,
    "profile": {"attention": "FA2 flash_fwd_splitkv_kernel, head dim 64 (block trait not established)",
                "linear": "vllm batch_invariant matmul_kernel_persistent (Triton), no bias",
                "rmsnorm": "layers.0.input_layernorm: batch_invariant Triton _rms_norm_kernel; the other norms: CUDA _C.fused_add_rms_norm",
                "rope": "vllm _C.rotary_embedding (NeoX), bf16 cos/sin cache",
                "act": "vllm _C.silu_and_mul", "sampler": "greedy argmax over bf16 logits"},
}

# [E6 NIGHT 2026-09-12, G3] Two more static configurations of the SAME registered functions (data only; the functions above are unchanged).
# Values are the checkpoints' config.json (HF revision named; verified by download 2026-09-12T05:20Z), the semantic profile named beside each.
# SMOL360 -- same architecture as B0 (LlamaForCausalLM), different checkpoint: only H/NL/NH/KVH/I change (K of the GEMM coordinates, the
# rmsnorm divisor literal H); V, D, EPS, ROPE_THETA, BIAS are B0's.  Profile: llama-eager-batch-invariant-v3.
SMOL360_CONFIG = {
    "model": "HuggingFaceTB/SmolLM2-360M",
    "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    "H": 960, "NL": 32, "NH": 15, "KVH": 5, "D": 64, "I": 2560, "V": 49152,
    "QKV": 1600, "GU": 5120, "EPS": 1e-5, "ROPE_THETA": 100_000.0, "MAX_POS": 8192, "BIAS": False,
    "tie_word_embeddings": True,
    "profile": {"attention": "FA2 flash_fwd_splitkv_kernel, head dim 64 (block trait not established)",
                "linear": "vllm batch_invariant matmul_kernel_persistent (Triton), no bias",
                "rmsnorm": "layers.0.input_layernorm: batch_invariant Triton _rms_norm_kernel; the other norms: CUDA _C.fused_add_rms_norm",
                "rope": "vllm _C.rotary_embedding (NeoX), bf16 cos/sin cache",
                "act": "vllm _C.silu_and_mul", "sampler": "greedy argmax over bf16 logits"},
}

# QWEN05 -- a different architecture (Qwen2ForCausalLM): fused QKV projection WITH bias (`BiasAdd` = one aten.add bf16 node per layer between the
# GEMM and rope/attention), eps 1e-6, rope theta 1e6, vocabulary 151,936 (the whole-scan width), GQA 14/2; head dim 64 (B0's FA2 trait, not B1's
# <128,...>).  Profile: qwen2-eager-batch-invariant-v2.
QWEN05_CONFIG = {
    "model": "Qwen/Qwen2.5-0.5B",
    "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "H": 896, "NL": 24, "NH": 14, "KVH": 2, "D": 64, "I": 4864, "V": 151936,
    "QKV": 1152, "GU": 9728, "EPS": 1e-6, "ROPE_THETA": 1_000_000.0, "MAX_POS": 32768, "BIAS": True,
    "tie_word_embeddings": True,
    "profile": {"attention": "FA2 flash_fwd_splitkv_kernel, head dim 64 (block trait not established)",
                "linear": "vllm batch_invariant matmul_kernel_persistent (Triton), bias as a separate aten.add",
                "rmsnorm": "layers.0.input_layernorm: batch_invariant Triton _rms_norm_kernel; the other norms: CUDA _C.fused_add_rms_norm",
                "rope": "vllm _C.rotary_embedding (NeoX), bf16 cos/sin cache",
                "act": "vllm _C.silu_and_mul", "sampler": "greedy argmax over bf16 logits"},
}

# ---------------------------------------------------------------------------------------------------------
# constants are zero-arity gates
# ---------------------------------------------------------------------------------------------------------

ZERO16 = const(16, 0)
NEG_INF32 = const(32, 0xFF800000)


def f32c(x: float) -> PrimitiveDefinition:
    return const(32, P.bits_of(np.float32(x)))


def _scale_log2_bits(D: int) -> int:
    """FA2 `set_params_fprop`: `scale_softmax_log2 = float(softmax_scale) * M_LOG2E` (double), narrowed to f32."""
    s = float(np.float32(D ** -0.5))
    return P.bits_of(np.float32(s * 1.4426950408889634))


# ---------------------------------------------------------------------------------------------------------
# GEMM (batch-invariant Triton matmul_kernel_persistent): one ascending k-chain per output coordinate
# ---------------------------------------------------------------------------------------------------------


@composite("DotBf16", 1, ["K"], lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.K, BF16))), F32),
           conformance="exact-model-tested")
def DotBf16(B, S, x, w):
    """f32 accumulator after K/16 `AmpereBF16TcDot16` steps in ascending k from a zero accumulator."""
    acc = B.call(ZERO32)
    for s in range(S.K // 16):
        acc = B.call(P.AmpereBF16TcDot16, acc, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
    return acc


@composite("GemmCoordinate", 1, ["K"], lambda S: ((("x", Array(S.K, BF16)), ("wrow", Array(S.K, BF16))), BF16),
           conformance="exact-model-tested")
def GemmCoordinate(B, S, x, wrow):
    """One output word of `x @ W^T`: bf16_rne(DotBf16(x, W[j, :])) (Triton `acc.to(bf16)`, `cvt.rn.bf16.f32`)."""
    return B.call(P.F2fpBf16, B.call(bind(DotBf16, K=S.K), x, wrow))


@composite("Gemm", 1, ["K", "N"], lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16)))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def Gemm(B, S, x, w):
    """`x[K] @ W[N,K]^T -> [N]`, no epilogue (the batch-invariant linear adds bias as a separate node)."""
    return B.batch(bind(GemmCoordinate, K=S.K), x, w, axes=(None, 0))


@composite("BiasAdd", 1, ["N"], lambda S: ((("y", Array(S.N, BF16)), ("b", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def BiasAdd(B, S, y, b):
    """aten.add(bf16, bf16) element-wise (one f32 add, RNE)."""
    return B.batch(P.Bf16Add, y, b, axes=(0, 0))


ResidualAdd = CompositeDefinition("ResidualAdd", 1, ("N",),
                                  lambda S: ((("x", Array(S.N, BF16)), ("r", Array(S.N, BF16))), Array(S.N, BF16)),
                                  lambda B, S, x, r: B.batch(P.Bf16Add, x, r, axes=(0, 0)),
                                  doc="aten.add(x, residual) in bf16 — the batch-invariant RMSNorm's residual path",
                                  conformance="exact-model-tested")


@composite("Bf16MulScalarTensor", 1, ["N"], lambda S: ((("x", Array(S.N, BF16)), ("s", Array(1, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested; G4 difftest vs aten.mul.Tensor(bf16 [1,N], bf16 0-d CUDA tensor) on the L40S (lane dense, "
                       "R13 X-08) and the wave-1 exhaustive probe: bf16(f32(x) * 48) on all 65,282 non-NaN bf16 words (Gemma-2 normalizer)")
def Bf16MulScalarTensor(B, S, x, s):
    """[R13 X-08, promoted from `registry.quarantine.dense`] `aten.mul.Tensor(x bf16 [.., N], s bf16 0-d TENSOR)`: ATen's CUDA
    `mul_kernel` for bf16 computes in opmath float — `f32(x_j) * f32(s)` (exact widenings, one RN multiply) — and stores with
    `c10::BFloat16(float)` = `cvt.rn.bf16.f32` (NaN -> 0x7FFF, `F2fpBf16`).  Generic: the scalar is a graph VALUE (a buffer /
    input read by the Program), never a constant folded into the specialization — `s` is shared by every row; nothing here
    names a model (Gemma-2's `embed * model.normalizer` is one instance).  `aten.mul.Tensor(bf16, python scalar)` is a
    different op (`aten.mul.Scalar`-shaped, f32 scalar operand) and is NOT this Definition."""
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    sf = B.call(P.Bf16ToF32, s[0])
    return B.batch(P.F2fpBf16, B.batch(P.F32Mul, xf, sf, axes=(0, None)), axes=(0,))

# ---------------------------------------------------------------------------------------------------------
# RMSNorm — vLLM batch-invariant Triton `_rms_norm_kernel` (rmsnorm-model.md §3.1)
# ---------------------------------------------------------------------------------------------------------


def _butterfly(B, vals: list[Coll], masks: tuple[int, ...], add) -> Coll:
    """Lane 0's value after an xor-butterfly reduction with the given masks (`v = v + shfl_xor(v, m)`)."""
    n = len(vals)
    cur = list(vals)
    for m in masks:
        cur = [B.call(add, cur[i], cur[i ^ m]) if (i ^ m) < n else cur[i] for i in range(n)]
    return cur[0]


@composite("RMSNormTriton", 1, ["N", "EPS"], lambda S: ((("x", Array(S.N, BF16)), ("w", Array(S.N, BF16))), Array(S.N, BF16)),
           conformance="exact-model-tested")
def RMSNormTriton(B, S, x, w, eps=None):
    """`out_j = bf16_rne((inv * x_j) * w_j)`, `inv = div.full(1, sqrt.approx.ftz(fma(rcp(n), sum, eps)))`, `sum` =
    chunks of 1024 columns, 128 threads x 8 columns (`acc = x1*x1; fma(x0,x0,acc); fma(x2..x7)`), warp
    butterfly xor 16,8,4,2,1, CTA `(w0 + w2) + (w1 + w3)`, chunks added in order from +0.  `eps`: `RMSNormTriton_v3`'s input."""
    N = S.N
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    chunk_sums = []
    for c0 in range(0, N, 1024):
        warps = []
        for wi in range(4):
            lanes = []
            for ln in range(32):
                t = wi * 32 + ln
                base = c0 + 8 * t
                if base >= N:
                    lanes.append(B.call(ZERO32))
                    continue
                xs = [xf[base + j] for j in range(8)]
                acc = B.call(P.F32Mul, xs[1], xs[1])
                acc = B.call(P.F32Fma, xs[0], xs[0], acc)
                for j in range(2, 8):
                    acc = B.call(P.F32Fma, xs[j], xs[j], acc)
                lanes.append(acc)
            warps.append(_butterfly(B, lanes, (16, 8, 4, 2, 1), P.F32Add))
        chunk_sums.append(B.call(P.F32Add, B.call(P.F32Add, warps[0], warps[2]), B.call(P.F32Add, warps[1], warps[3])))
    total = B.call(ZERO32)
    for cs in chunk_sums:
        total = B.call(P.F32Add, total, cs)
    rcp_n = B.call(P.DivFullRcp, B.call(f32c(float(N))))
    mean_eps = B.call(P.F32Fma, rcp_n, total, B.call(f32c(S.EPS)) if eps is None else eps)
    rms = B.call(P.MufuSqrtFtz, mean_eps)
    inv = B.call(P.F32Mul, B.call(P.DivFullScaleA, B.call(f32c(1.0)), rms), B.call(P.DivFullRcp, rms))
    wf = B.batch(P.Bf16ToF32, w, axes=(0,))
    sx = B.batch(P.F32Mul, inv, xf, axes=(None, 0))
    y = B.batch(P.F32Mul, sx, wf, axes=(0, 0))
    return B.batch(P.F2fpBf16, y, axes=(0,))


def _cub_block_reduce_sum(B, partials: list[Coll]) -> Coll:
    """`cub::BlockReduce<float, 1024>(...).Reduce(v, cuda::std::plus<>{}, blockDim.x)` (CCCL 2.8, algorithm
    BLOCK_REDUCE_WARP_REDUCTIONS) for `len(partials)` launched threads: per warp `WarpReduceShfl` — steps
    `offset = 1, 2, 4, 8, 16` (ascending), `shfl.sync.down` + `add.f32` guarded by the valid-peer predicate
    (lane i adds lane i+offset only when that lane exists, so a partial last warp adds only valid lanes);
    then thread 0 adds the warp aggregates sequentially from warp 0 (`ApplyWarpAggregates`, warps w with
    32·w < blockDim.x). Only the gates feeding lane 0 / thread 0 are emitted."""
    warps = []
    for wi in range((len(partials) + 31) // 32):
        lanes = dict(enumerate(partials[32 * wi:32 * wi + 32]))
        for off in (1, 2, 4, 8, 16):
            lanes = {i: (B.call(P.F32Add, v, lanes[i + off]) if i + off in lanes else v)
                     for i, v in lanes.items() if i % (2 * off) == 0}
        warps.append(lanes[0])
    var = warps[0]
    for wv in warps[1:]:
        var = B.call(P.F32Add, var, wv)
    return var


def _fused_rms_epilogue(B, S, r, rf, var, w, eps=None):
    """`s_variance = rsqrtf(variance / hidden_size + epsilon)` (IEEE div, IEEE add, MUFU.RSQ via
    `rsqrt.approx.f32` range scaling); `out_j = (bf16)((float)res'_j * s * (float)w_j)` — two f32 products
    left to right, `__float2bfloat16` (RNE, canonical NaN 0x7FFF).  `eps`: None for `EPS`."""
    N = S.N
    mean = B.call(P.F32Div, var, B.call(f32c(float(N))))
    arg = B.call(P.F32Add, mean, B.call(f32c(S.EPS)) if eps is None else eps)
    sc = B.call(P.RsqrtApprox, arg)
    wf = B.batch(P.Bf16ToF32, w, axes=(0,))
    y = B.batch(P.F32Mul, B.batch(P.F32Mul, rf, sc, axes=(0, None)), wf, axes=(0, 0))
    return tuple_of(B.batch(P.F2fpBf16, y, axes=(0,)), r)


@composite("RMSNormFusedCuda", 1, ["N", "EPS"],
           lambda S: ((("x", Array(S.N, BF16)), ("res", Array(S.N, BF16)), ("w", Array(S.N, BF16))),
                      Tuple(Array(S.N, BF16), Array(S.N, BF16))),
           conformance="superseded by RMSNormFusedCuda_v2: models fused_add_rms_norm_kernel<bf16, 8>, which vLLM d9105ea80 does not "
                       "launch under VLLM_BATCH_INVARIANT=1 (layernorm_kernels.cu: `!batch_invariant_launch` guards the width-8 "
                       "path); its variance tree differs from production's in 1218 of the 2340 fused-norm rows of the B0 store and "
                       "the bf16 output in 13 of them (B0 store sweep, `verity_vllm.program.registry.rmsnorm_fused_sweep`); the B1 divergence "
                       "(layer 27, t=1, word 629; typed-b1.md §5.1) is the same class")
def RMSNormFusedCudaV1(B, S, x, res, w):
    """Superseded (kept registered so that TA1's B0/B1 records `Serve_v1{...}` with this body remain decodable):
    vLLM CUDA `fused_add_rms_norm_kernel<bf16, 8>`: `res' = bf16(x + res)`; N/8 threads x one 8-vector of res':
    `p = (((z0^2 + z1^2) + (z2^2 + z3^2)) + ...)` accumulated sequentially over the 4 bf16x2 pairs from +0; cub
    BlockReduce<1024> over the N/8 threads; `mean = var / N` (IEEE), `+ eps`, `s = rsqrtf(.)`,
    `out_j = bf16((res'_j * s) * w_j)`. Returns (normed, res'). The width-8 kernel is what runs *without*
    `VLLM_BATCH_INVARIANT=1`; the eager batch-invariant profile launches the generic kernel (see v2)."""
    N = S.N
    r = B.batch(P.Bf16AddF2fp, x, res, axes=(0, 0))
    rf = B.batch(P.Bf16ToF32, r, axes=(0,))
    nthreads = N // 8
    partials = []
    for t in range(nthreads):
        z = [rf[8 * t + j] for j in range(8)]
        acc = None
        for i in range(4):
            pair = B.call(P.F32Add, B.call(P.F32Mul, z[2 * i], z[2 * i]), B.call(P.F32Mul, z[2 * i + 1], z[2 * i + 1]))
            acc = pair if acc is None else B.call(P.F32Add, acc, pair)
        partials.append(acc)
    var = _cub_block_reduce_sum(B, partials)
    return _fused_rms_epilogue(B, S, r, rf, var, w)


FUSED_RMS_BLOCK = 1024  # `max_block_size = batch_invariant_launch ? 1024 : ...`; `block(std::min(hidden_size, max_block_size))`


@composite("RMSNormFusedCuda", 2, ["N", "EPS"],
           lambda S: ((("x", Array(S.N, BF16)), ("res", Array(S.N, BF16)), ("w", Array(S.N, BF16))),
                      Tuple(Array(S.N, BF16), Array(S.N, BF16))),
           conformance="exact-model-tested; evidence: reproduces production on 2,340/2,340 fused-norm rows (1,347,840 words) of "
                       "the B0 store (SmolLM2-135M, N=576, incl. the 13 rows v1 misses; `verity_vllm.program.registry.rmsnorm_fused_sweep`) "
                       "and on the B1 (N=1536) L27/t=1 row that v1 missed (typed-b1.md §5.1, closed 2026-09-08: "
                       "`results/TA1-typed-b1-20260907/corr-B1-L27t1-rmsv2.json`, 14/14 exact)")
def RMSNormFusedCuda(B, S, x, res, w, eps=None):
    """vLLM `d9105ea80` `csrc/layernorm_kernels.cu` **generic** `fused_add_rms_norm_kernel<c10::BFloat16, 0, true>` —
    the kernel `fused_add_rms_norm` launches when `vllm_is_batch_invariant()` (the width-8 `_f16Vec` kernel is
    guarded by `!batch_invariant_launch`): `block = min(N, 1024)` threads; thread t owns elements
    `t, t+block, ...`; `z = bf16(x + res)` (c10 `BFloat16::operator+=`, RNE), `variance += (float)z * (float)z`
    (FFMA, no `--use_fast_math`, no FTZ; the square of a bf16 value is exact in f32, so contraction is
    immaterial: N=1536 -> threads 0..511 hold `fma(z[t+1024], z[t+1024], z[t]^2)`, threads 512..1023 hold
    `z[t]^2`; N=576 -> 576 threads x one square); `cub::BlockReduce<float, 1024>` over
    the launched threads (`_cub_block_reduce_sum`: per-warp shfl_down tree 1,2,4,8,16; warp aggregates added
    sequentially from warp 0 — 32 warps for N=1536, 18 for N=576); `s = rsqrtf(var / N + eps)`;
    `out_j = bf16((res'_j * s) * w_j)`. Returns (normed, res').  `eps` as `RMSNormTriton_v1`'s."""
    N = S.N
    r = B.batch(P.Bf16AddF2fp, x, res, axes=(0, 0))
    rf = B.batch(P.Bf16ToF32, r, axes=(0,))
    nthreads = min(N, FUSED_RMS_BLOCK)
    partials = []
    for t in range(nthreads):
        acc = B.call(P.F32Mul, rf[t], rf[t])  # `variance = 0.0f; variance += x*x` -> fma(x, x, 0) == x*x (exact for bf16 x)
        for j in range(t + nthreads, N, nthreads):
            acc = B.call(P.F32Fma, rf[j], rf[j], acc)
        partials.append(acc)
    var = _cub_block_reduce_sum(B, partials)
    return _fused_rms_epilogue(B, S, r, rf, var, w, eps)


# ---------------------------------------------------------------------------------------------------------
# RoPE (NeoX, vLLM `rotary_embedding_kernel`), SiLU-and-mul
# ---------------------------------------------------------------------------------------------------------


COS_SIN_QUALIFICATION = ("the cos/sin row is an Input of a derived table rebuilt from vllm._compute_cos_sin_cache (CUDA cosf/sinf), "
                         "GPU only: the CPU rebuild differs in 17,261 of 16,777,216 words (B1) and 6 of 524,288 (B0); typed-b1.md §5.2")


@composite("RoPEHead", 1, ["D"], lambda S: ((("x", Array(S.D, BF16)), ("cs", Array(S.D, BF16))), Array(S.D, BF16)),
           conformance="exact-model-tested; qualification: " + COS_SIN_QUALIFICATION)
def RoPEHead(B, S, x, cs):
    """NeoX rotation of one head with the bf16 cos/sin cache row `cs = [cos[0:D/2] | sin[0:D/2]]`:
    `out[i] = bf16(x[i]*cos[i] - x[i+R]*sin[i])`, `out[i+R] = bf16(x[i+R]*cos[i] + x[i]*sin[i])`, R = D/2."""
    R = S.D // 2
    cos, sin = cs[0:R], cs[R:S.D]
    lo = B.batch(P.RopeOut, x[0:R], x[R:S.D], cos, sin, axes=(0, 0, 0, 0))  # x[i]*cos - x[i+R]*sin
    hi = B.batch(P.RopeOutAdd, x[0:R], x[R:S.D], cos, sin, axes=(0, 0, 0, 0))  # x[i+R]*cos + x[i]*sin
    return _Coll(Array(S.D, BF16), concat([lo.refs, hi.refs]))


@composite("RoPE", 1, ["NHEADS", "D"],
           lambda S: ((("x", Array(S.NHEADS, Array(S.D, BF16))), ("cs", Array(S.D, BF16))), Array(S.NHEADS, Array(S.D, BF16))),
           conformance="exact-model-tested; qualification: " + COS_SIN_QUALIFICATION)
def RoPE(B, S, x, cs):
    return B.batch(bind(RoPEHead, D=S.D), x, cs, axes=(0, None))


@composite("SiluMul", 1, ["I"], lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)),
           conformance="exact-model-tested")
def SiluMul(B, S, gu):
    """vLLM `silu_and_mul`: `out[i] = bf16(bf16(silu(gate[i])) * up[i])`, gate = gu[:I], up = gu[I:]."""
    return B.batch(P.SiluMulBf16, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


# ---------------------------------------------------------------------------------------------------------
# Attention — FlashAttention-2 forward, one (query position, head) row (attention-model.md §3)
# ---------------------------------------------------------------------------------------------------------

BN = 64  # kBlockN of the FA2 kernel at HEAD DIM 128 (W11's measured trait, Qwen2.5-1.5B) -- the hd-128 v2 record `ATTENTION_FA2`


def fa2_kblock_n(D: int) -> int:
    """[E7 D54 2026-09-13] kBlockN of the FA2 kernel vLLM launches at head dim ``D``: vllm-flash-attn `506341a`
    `csrc/flash_attn/src/flash_fwd_launch_template.h:172-192` -- paged varlen KV forces `num_splits == 1` (`flash_api.cpp:763`),
    so `run_mha_fwd_splitkv_dispatch` takes `run_mha_fwd_splitkv_align`: `kBlockM = 64; kBlockN = Headdim <= 64 ? 128 : 64`
    (prefill causal and the `seqlen_q == 1` decode launch alike). 64 was measured bit-exact at D = 128 (W11); at D = 64
    (SmolLM2-135M/360M, Qwen2.5-0.5B) the 128×64 record disagreed from the first two-block row (position 64) under BN = 64.
    A DERIVED static of the attention bodies, never a constant: the same kernel geometry for every head dim it dispatches."""
    return 128 if int(D) <= 64 else 64


def _lane(c: int) -> int:
    return (c % 8) // 2


@composite("AttnBlock", 1, ["D", "NVIS", "FIRST"],
           lambda S: (((("q", Array(S.D, BF16)), ("k", Array(BN, Array(S.D, BF16))), ("v", Array(BN, Array(S.D, BF16))),
                        ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="exact-model-tested")
def AttnBlock(B, S, q, k, v, m_prev, l_prev, o_prev):
    """One 64-key block of the FA2 online softmax for one query row. `NVIS` = visible (unmasked) columns
    (64 except the last-visited block); masked columns contribute P = 0 exactly (skipped). Returns
    (m_new, lane sums l[4], o[D]); with `FIRST` the state arguments are ignored (kernel `Is_first`).

    [E7 D54 2026-09-13] `BN` here is the module constant 64 = the kernel's kBlockN at HEAD DIM 128 only.  This body is
    what every pinned record so far was produced under and is retained unchanged for their re-verification; the
    head-dim-derived geometry is `AttnBlock_v3` (`fa2_kblock_n(D)`), reached through `Serve_v4`."""
    D, NVIS, FIRST = S.D, S.NVIS, S.FIRST
    dot = bind(DotBf16, K=D)
    s = [B.call(dot, q, k[c]) for c in range(NVIS)]  # A: S = Q K^T, chained mma k-steps, acc 0 per block
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = B.call(P.F32Max, bmax, s[c])  # B1: row max
    m_new = bmax if FIRST else B.call(P.F32Max, m_prev, bmax)
    m_use = B.call(P.GuardNegInfZero, m_new)
    scale_log2 = B.call(const(32, _scale_log2_bits(D)))
    if not FIRST:  # B2: rescale factor
        scores_scale = B.call(P.MufuEx2Ftz, B.call(P.F32MulFtz, B.call(P.F32SubFtz, m_prev, m_use), scale_log2))
    max_scaled = B.call(P.F32MulFtz, m_use, scale_log2)
    p = [B.call(P.MufuEx2Ftz, B.call(P.F32FmaSubFtz, s[c], scale_log2, max_scaled)) for c in range(NVIS)]  # B3
    # B4: per-lane sequential sums; the block's first add is contracted with the rescale
    lanes: list[Coll] = []
    for ln in range(4):
        cols = [c for c in range(NVIS) if _lane(c) == ln]
        if FIRST:
            acc = p[cols[0]] if cols else B.call(ZERO32)
            rest = cols[1:]
        else:
            first = p[cols[0]] if cols else B.call(ZERO32)
            acc = B.call(P.F32FmaFtz, scores_scale, l_prev[ln], first)
            rest = cols[1:]
        for c in rest:
            acc = B.call(P.F32AddFtz, acc, p[c])
        lanes.append(acc)
    # B5 + C: P -> bf16, O = (O * scale) + P V in 4 ascending k-steps of 16 keys
    pb = [B.call(P.F2fpBf16, x) for x in p]
    zero16 = B.call(ZERO16) if NVIS < BN else None
    o = [B.call(ZERO32) for _ in range(D)] if FIRST else [B.call(P.F32MulFtz, o_prev[d], scores_scale) for d in range(D)]
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break  # all-masked k-step: P = 0 exactly, no accumulator change
        prow = _Coll(A16, concat([(pb[k0 + j] if k0 + j < NVIS else zero16).refs for j in range(16)]))
        vblk = v[k0:k0 + 16]  # Array<16, Array<D>>
        # V column d over the 16 keys: strided view [D][16]
        vt_refs = strided_view(vblk, 0, 16, 1, D, 16 * D) if isinstance(vblk.refs, Affine) else \
            Explicit(tuple(vblk.refs[j * D + d] for d in range(D) for j in range(16)))
        vt = _Coll(Array(D, A16), vt_refs)
        o = list(B.batch(P.AmpereBF16TcDot16, array_of(o), prow, vt, axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(o))


@composite("AttentionHead", 1, ["T", "D"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="exact-model-tested")
def AttentionHead(B, S, q, k, v):
    """Causal attention of one query at position T-1 over keys 0..T-1 (FA2, blocks visited last -> first);
    epilogue `sum = (l0 + l2) + (l1 + l3)`, `inv = (sum == 0 || NaN) ? 1 : rcp(sum)`, `out = bf16(o * inv)`."""
    T, D = S.T, S.D
    nb = (T + BN - 1) // BN
    m = l = o = None
    zero_row = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:  # pad the block's K/V rows with zeros (Clear_OOB_MN); masked columns are never read
            if zero_row is None:
                zero_row = _Coll(Array(D, BF16), concat([B.call(ZERO16).refs] * D))
            pad = [zero_row.refs] * (BN - nvis)
            kb = _Coll(Array(BN, Array(D, BF16)), concat([kb.refs] + pad))
            vb = _Coll(Array(BN, Array(D, BF16)), concat([vb.refs] + pad))
        first = b == nb - 1
        blk = bind(AttnBlock, D=D, NVIS=nvis, FIRST=first)
        if first:
            z = B.call(ZERO32)
            m0 = z
            l0 = _Coll(Array(4, F32), concat([z.refs] * 4))
            o0 = _Coll(Array(D, F32), concat([z.refs] * D))
            st = B.call(blk, q, kb, vb, m0, l0, o0)
        else:
            st = B.call(blk, q, kb, vb, m, l, o)
        m, l, o = st[0], st[1], st[2]
    total = B.call(P.F32AddFtz, B.call(P.F32AddFtz, l[0], l[2]), B.call(P.F32AddFtz, l[1], l[3]))
    inv = B.call(P.Fa2InvSum, total)
    y = B.batch(P.F32MulFtz, o, inv, axes=(0, None))
    return B.batch(P.F2fpBf16, y, axes=(0,))


@composite("Attention", 1, ["T", "NH", "KVH", "D"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="exact-model-tested")
def Attention(B, S, q, kc, vc):
    """GQA: head h reads KV head h // (NH/KVH); the caches are `[T][KVH][D]` (one row per position)."""
    T, NH, KVH, D = S.T, S.NH, S.KVH, S.D
    G = NH // KVH
    outs = []
    for g in range(KVH):
        kg = _Coll(Array(T, Array(D, BF16)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = _Coll(Array(T, Array(D, BF16)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(bind(AttentionHead, T=T, D=D), q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return _Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


# ---------------------------------------------------------------------------------------------------------
# embedding, LM head, token selection
# ---------------------------------------------------------------------------------------------------------


@composite("Embedding", 1, ["V", "H"], lambda S: ((("tok", I32), ("table", Array(S.V, Array(S.H, BF16)))), Array(S.H, BF16)),
           conformance="exact-model-tested")
def Embedding(B, S, tok, table):
    """`h[j] = table[tok][j]`: one wide-fan-in gather gate per hidden coordinate (column j of the table)."""
    cols = _Coll(Array(S.H, Array(S.V, BF16)), strided_view(table, 0, S.V, 1, S.H, S.V * S.H))
    return B.batch(P.gather_bf16(S.V), tok, cols, axes=(None, 0))


@composite("ArgmaxStep", 1, [], lambda S: ((("carry", Tuple(BF16, I32, I32)), ("x", BF16)), Tuple(Tuple(BF16, I32, I32), I32)),
           conformance="exact-model-tested")
def ArgmaxStep(B, S, carry, x):
    """Greedy argmax step over (best, best_index, i): take (x, i) iff x > best (strict: the first index wins
    ties, NaN never wins), then i += 1."""
    best, best_i, i = carry[0], carry[1], carry[2]
    gt = B.call(P.Bf16GtStrict, x, best)
    nb = B.call(P.SelectBf16, gt, x, best)
    ni = B.call(P.SelectI32, gt, i, best_i)
    nxt = B.call(P.I32Add, i, B.call(const(32, 1)))
    return tuple_of(tuple_of(nb, ni, nxt), ni)


@composite("TokenSelect", 1, ["V"], lambda S: ((("logits", Array(S.V, BF16)),), I32), conformance="model-untested")
def TokenSelect(B, S, logits):
    """Greedy sampling (temperature 0): argmax over the bf16 logits, lowest index on ties (torch.argmax)."""
    init = tuple_of(logits[0], B.call(const(32, 0)), B.call(const(32, 1)))
    step = B.scan(ArgmaxStep, init, xs=(logits[1:S.V],))
    return step[0][1]


# ---------------------------------------------------------------------------------------------------------
# layer and model
# ---------------------------------------------------------------------------------------------------------


def layer_weights_type(C: dict) -> Record:
    H, QKV, GU, I = C["H"], C["QKV"], C["GU"], C["I"]
    fields = [("ln1", Array(H, BF16)), ("qkv_w", Array(QKV, Array(H, BF16)))]
    if C.get("BIAS", True):
        fields.append(("qkv_b", Array(QKV, BF16)))
    fields += [("o_w", Array(H, Array(C["NH"] * C["D"], BF16))),
               ("ln2", Array(H, BF16)), ("gate_up_w", Array(GU, Array(H, BF16))), ("down_w", Array(H, Array(I, BF16)))]
    return Record(*fields)


def weights_type(C: dict) -> Record:
    fields = [("embed", Array(C["V"], Array(C["H"], BF16))),
              ("layers", Array(C["NL"], layer_weights_type(C))),
              ("norm", Array(C["H"], BF16)),
              ("cos_sin", Array(C["MAX_POS"], Array(C["D"], BF16)))]
    if not C.get("tie_word_embeddings", True):
        # [PROPOSED GEN dense2] an untied LM head (Phi-3-mini: `lm_head.weight` is its own checkpoint tensor) is one more
        # `[V][H]` field after the base ones; every existing path is unchanged and tied configs (no key, or True) see the
        # identical Record.  `ServeV4` binds it for the logits GEMM below.
        fields.append(("lm_head", Array(C["V"], Array(C["H"], BF16))))
    return Record(*fields)


def lm_head_of(C: dict, weights):
    """The logits GEMM weight: the embedding (tied, the default) or the `lm_head` field of an untied checkpoint."""
    return weights["embed"] if C.get("tie_word_embeddings", True) else weights["lm_head"]


def kv_type(C: dict, T: int) -> Array:
    return Array(T, Array(C["KVH"], Array(C["D"], BF16)))


@composite("LayerPre", 1, ["C", "L0"],
           lambda S: ((("h", Array(S.C["H"], BF16)), ("res", Array(S.C["H"], BF16)), ("w", layer_weights_type(S.C)), ("cs", Array(S.C["D"], BF16))),
                      Tuple(Array(S.C["H"], BF16), Array(S.C["NH"], Array(S.C["D"], BF16)),
                            Array(S.C["KVH"], Array(S.C["D"], BF16)), Array(S.C["KVH"], Array(S.C["D"], BF16)))),
           conformance="exact-model-tested")
def LayerPre(B, S, h, res, w, cs):
    """Layer front half at one position: input norm (layer 0: Triton RMSNorm of h; else the fused CUDA
    add+RMSNorm of (h, res)), QKV GEMM + bias, RoPE on q, k.
    Returns (residual_out, q[NH][D], k[KVH][D], v[KVH][D])."""
    C = S.C
    H, NH, KVH, D = C["H"], C["NH"], C["KVH"], C["D"]
    if S.L0:
        x = h
        n1 = B.call(bind(RMSNormTriton, N=H, EPS=C["EPS"]), x, w["ln1"])
    else:
        nr = B.call(bind(RMSNormFusedCuda, N=H, EPS=C["EPS"]), h, res, w["ln1"])
        n1, x = nr[0], nr[1]
    qkv = B.call(bind(Gemm, K=H, N=C["QKV"]), n1, w["qkv_w"])
    if C.get("BIAS", True):
        qkv = B.call(bind(BiasAdd, N=C["QKV"]), qkv, w["qkv_b"])
    q = _Coll(Array(NH, Array(D, BF16)), qkv.refs.slice(0, NH * D))
    k = _Coll(Array(KVH, Array(D, BF16)), qkv.refs.slice(NH * D, (NH + KVH) * D))
    v = _Coll(Array(KVH, Array(D, BF16)), qkv.refs.slice((NH + KVH) * D, (NH + 2 * KVH) * D))
    qr = B.call(bind(RoPE, NHEADS=NH, D=D), q, cs)
    kr = B.call(bind(RoPE, NHEADS=KVH, D=D), k, cs)
    return tuple_of(x, qr, kr, v)


@composite("LayerPost", 1, ["C"],
           lambda S: ((("attn", Array(S.C["NH"], Array(S.C["D"], BF16))), ("res", Array(S.C["H"], BF16)), ("w", layer_weights_type(S.C))),
                      Tuple(Array(S.C["H"], BF16), Array(S.C["H"], BF16))),
           conformance="exact-model-tested")
def LayerPost(B, S, attn, res, w):
    """Layer back half: o_proj GEMM, fused CUDA add+RMSNorm, gate_up GEMM, SiLU-mul, down GEMM.
    Returns (mlp_out, residual) — the next layer adds them."""
    C = S.C
    H = C["H"]
    a = _Coll(Array(C["NH"] * C["D"], BF16), attn.refs)
    o = B.call(bind(Gemm, K=C["NH"] * C["D"], N=H), a, w["o_w"])
    nr = B.call(bind(RMSNormFusedCuda, N=H, EPS=C["EPS"]), o, res, w["ln2"])
    n2, r2 = nr[0], nr[1]
    gu = B.call(bind(Gemm, K=H, N=C["GU"]), n2, w["gate_up_w"])
    act = B.call(bind(SiluMul, I=C["I"]), gu)
    d = B.call(bind(Gemm, K=C["I"], N=H), act, w["down_w"])
    return tuple_of(d, r2)


@composite("Final", 1, ["C"],
           lambda S: ((("h", Array(S.C["H"], BF16)), ("res", Array(S.C["H"], BF16)), ("norm_w", Array(S.C["H"], BF16)),
                       ("lm_head", Array(S.C["V"], Array(S.C["H"], BF16)))), I32),
           conformance="model-untested")
def Final(B, S, h, res, norm_w, lm_head):
    """Final fused add+norm, LM head GEMM (tied embedding), greedy token selection."""
    C = S.C
    n = B.call(bind(RMSNormFusedCuda, N=C["H"], EPS=C["EPS"]), h, res, norm_w)[0]
    logits = B.call(bind(Gemm, K=C["H"], N=C["V"]), n, lm_head)
    return B.call(bind(TokenSelect, V=C["V"]), logits)


def _forward_positions(B, C: dict, toks: Coll, weights: Coll, T0: int, n_pos: int, caches: list[tuple[Coll, Coll]] | None):
    """Run all layers for positions T0..T0+n_pos-1 (prefill when caches is None: each position attends to the
    positions before it within this call; decode: n_pos == 1 attending to `caches` + itself). Returns
    (per-layer (k rows, v rows) of these positions, final (h, res) per position)."""
    NL, H = C["NL"], C["H"]
    hs = [B.call(bind(Embedding, V=C["V"], H=H), toks[i], weights["embed"], name=f"t={T0 + i}/embed") for i in range(n_pos)]
    res = [None] * n_pos
    new_kv: list[tuple[Coll, Coll]] = []
    for L in range(NL):
        w = weights["layers"][L]
        pre = [B.call(bind(LayerPre, C=C, L0=(L == 0)), hs[i], hs[i] if L == 0 else res[i], w, weights["cos_sin"][T0 + i],
                      name=f"t={T0 + i}/layers.{L}/pre") for i in range(n_pos)]
        k_rows = _Coll(kv_type(C, n_pos), concat([p[2].refs for p in pre]))
        v_rows = _Coll(kv_type(C, n_pos), concat([p[3].refs for p in pre]))
        if caches is None:
            kc, vc = k_rows, v_rows
        else:
            kc = _Coll(kv_type(C, T0 + n_pos), concat([caches[L][0].refs, k_rows.refs]))
            vc = _Coll(kv_type(C, T0 + n_pos), concat([caches[L][1].refs, v_rows.refs]))
        new_kv.append((k_rows, v_rows))
        for i in range(n_pos):
            T = T0 + i + 1
            attn = B.call(bind(Attention, T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"]), pre[i][1], kc[0:T], vc[0:T],
                          name=f"t={T - 1}/layers.{L}/attn")
            post = B.call(bind(LayerPost, C=C), attn, pre[i][0], w, name=f"t={T - 1}/layers.{L}/post")
            hs[i], res[i] = post[0], post[1]
    return new_kv, list(zip(hs, res))


@composite("Serve", 1, ["C", "LP", "STEPS"],
           lambda S: ((("prompt", Array(S.LP, I32)), ("weights", weights_type(S.C))), Array(S.STEPS + 1, I32)),
           conformance="model-untested")
def Serve(B, S, prompt, weights):
    """Greedy serving of one request: prefill of LP prompt tokens then STEPS decode steps; returns the
    STEPS+1 sampled tokens (the first from the last prompt position). The decode loop is a lowered chain of
    dependent per-position specialisations (the KV cache grows), not a fixed-shape `scan`."""
    C, LP, STEPS = S.C, S.LP, S.STEPS
    caches, finals = _forward_positions(B, C, prompt, weights, 0, LP, None)
    tok = B.call(bind(Final, C=C), finals[-1][0], finals[-1][1], weights["norm"], weights["embed"], name=f"t={LP - 1}/final")
    toks = [tok]
    for s in range(STEPS):
        T0 = LP + s
        new_kv, fin = _forward_positions(B, C, _Coll(Array(1, I32), tok.refs), weights, T0, 1, caches)
        caches = [(_Coll(kv_type(C, T0 + 1), concat([caches[L][0].refs, new_kv[L][0].refs])),
                   _Coll(kv_type(C, T0 + 1), concat([caches[L][1].refs, new_kv[L][1].refs]))) for L in range(C["NL"])]
        tok = B.call(bind(Final, C=C), fin[0][0], fin[0][1], weights["norm"], weights["embed"], name=f"t={T0}/final")
        toks.append(tok)
    return _Coll(Array(STEPS + 1, I32), concat([t.refs for t in toks]))


# ---------------------------------------------------------------------------------------------------------
# [E7 D54 2026-09-13] the kernel-geometry-derived attention chain and the Serve that reaches it
# ---------------------------------------------------------------------------------------------------------
#
# Finding (D54): the 128×64 SmolLM2-135M record (head dim 64) disagrees with `Serve_v1` from the first query position
# whose keys span two 64-key blocks (t = 64, layer 0, `attn` port 0); positions 0..63 agree at every layer.  Cause: the
# v1 bodies read the module constant `BN = 64`, which is the kernel's kBlockN at HEAD DIM 128 (W11 measured it on
# Qwen2.5-1.5B); at head dim <= 64 vllm-flash-attn `506341a` launches `run_mha_fwd_splitkv_align` with kBlockN = 128
# (`flash_fwd_launch_template.h:172-192`; paged varlen KV forces `num_splits == 1`, `flash_api.cpp:763`; prefill and the
# `seqlen_q == 1` decode launch alike).  `fa2_kblock_n(D)` is that dispatch rule.
#
# Why NEW versions and not a body edit: every pinned record (n4/sc1/s4/q3, E8's 32×16, ...) was produced under `Serve_v1`
# and is re-verified against ITS pinned program (the user: "retain their original pinned programs"); the verifier builds
# the program from `program.identity`, so `Serve_v1` must keep building the v1 body word for word.  The corrected program
# is `Serve_v4{C, LP, STEPS}`: TA1's Serve body over TA1's layer functions, with the attention chain `Attention_v3` ->
# `AttentionHead_v3` -> `AttnBlock_v3{D, NVIS, FIRST}`, whose block size is the DERIVED static `fa2_kblock_n(D)` (the
# same `DotBf16_v1{K=D}` QK dot, the same primitives: the population's kind ids are unchanged, only the per-kind counts
# and the program digest move).  (v2 of the chain is CF1's statics-lifted `AttnBlock_v2{..., BN, DOT}`; `Serve_v3` is
# serve3's padded program; hence v3 / v4.)  `SERVE_FAMILIES` names the two serving programs the pipeline can pin.


@composite("AttnBlock", 3, ["D", "NVIS", "FIRST", "BN"],
           lambda S: (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                        ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="exact-model-tested")
def AttnBlockV3(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2=None):
    """`AttnBlock_v1`'s body with the key-block size as the DECLARED static ``BN`` (the kernel's kBlockN, carried by the
    program identity -- D57): one BN-key block of the FA2 online softmax for one query row.  `NVIS` = visible (unmasked)
    columns (BN except the last-visited block); masked columns contribute P = 0 exactly (skipped).  Returns (m_new, lane
    sums l[4], o[D]); with `FIRST` the state arguments are ignored (kernel `Is_first`).  `BN = 64` is `AttnBlock_v1` gate for
    gate.  Conformance: `test_derived_rows` gate-by-gate at D = 64 / BN = 128; the 128×64 SmolLM2-135M record's two-block rows
    (D54 acceptance, pending).  `scale_log2`: the scale input (`value_inputs.AttnBlockV10`), or None for the constant."""
    D, NVIS, FIRST, BN = S.D, S.NVIS, S.FIRST, S.BN
    if BN % 16 or not 1 <= NVIS <= BN:
        raise ValueError(f"AttnBlock_v3: BN={BN} must be a multiple of 16 and 1 <= NVIS={NVIS} <= BN")
    dot = bind(DotBf16, K=D)
    s = [B.call(dot, q, k[c]) for c in range(NVIS)]  # A: S = Q K^T, chained mma k-steps, acc 0 per block
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = B.call(P.F32Max, bmax, s[c])  # B1: row max
    m_new = bmax if FIRST else B.call(P.F32Max, m_prev, bmax)
    m_use = B.call(P.GuardNegInfZero, m_new)
    scale_log2 = B.call(const(32, _scale_log2_bits(D))) if scale_log2 is None else scale_log2
    if not FIRST:  # B2: rescale factor
        scores_scale = B.call(P.MufuEx2Ftz, B.call(P.F32MulFtz, B.call(P.F32SubFtz, m_prev, m_use), scale_log2))
    max_scaled = B.call(P.F32MulFtz, m_use, scale_log2)
    p = [B.call(P.MufuEx2Ftz, B.call(P.F32FmaSubFtz, s[c], scale_log2, max_scaled)) for c in range(NVIS)]  # B3
    # B4: per-lane sequential sums; the block's first add is contracted with the rescale
    lanes: list[Coll] = []
    for ln in range(4):
        cols = [c for c in range(NVIS) if _lane(c) == ln]
        if FIRST:
            acc = p[cols[0]] if cols else B.call(ZERO32)
            rest = cols[1:]
        else:
            first = p[cols[0]] if cols else B.call(ZERO32)
            acc = B.call(P.F32FmaFtz, scores_scale, l_prev[ln], first)
            rest = cols[1:]
        for c in rest:
            acc = B.call(P.F32AddFtz, acc, p[c])
        lanes.append(acc)
    # B5 + C: P -> bf16, O = (O * scale) + P V in BN/16 ascending k-steps of 16 keys
    pb = [B.call(P.F2fpBf16, x) for x in p]
    zero16 = B.call(ZERO16) if NVIS < BN else None
    o = [B.call(ZERO32) for _ in range(D)] if FIRST else [B.call(P.F32MulFtz, o_prev[d], scores_scale) for d in range(D)]
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break  # all-masked k-step: P = 0 exactly, no accumulator change
        prow = _Coll(A16, concat([(pb[k0 + j] if k0 + j < NVIS else zero16).refs for j in range(16)]))
        vblk = v[k0:k0 + 16]  # Array<16, Array<D>>
        vt_refs = strided_view(vblk, 0, 16, 1, D, 16 * D) if isinstance(vblk.refs, Affine) else \
            Explicit(tuple(vblk.refs[j * D + d] for d in range(D) for j in range(16)))
        vt = _Coll(Array(D, A16), vt_refs)
        o = list(B.batch(P.AmpereBF16TcDot16, array_of(o), prow, vt, axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(o))


@composite("AttentionHead", 3, ["T", "D", "BN"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="exact-model-tested")
def AttentionHeadV3(B, S, q, k, v, *extra, block=AttnBlockV3):
    """`AttentionHead_v1`'s body over `AttnBlock_v3{..., BN}`: causal attention of one query at position T-1 over keys 0..T-1,
    ceil(T / BN) blocks visited last -> first; epilogue `sum = (l0 + l2) + (l1 + l3)`, `inv = (sum == 0 || NaN) ? 1 : rcp(sum)`,
    `out = bf16(o * inv)`.  ``BN`` = the kernel's kBlockN, a declared static (D57).  `extra` / `block`: `AttentionHead_v10`'s."""
    T, D, BN = S.T, S.D, S.BN
    nb = (T + BN - 1) // BN
    m = l = o = None
    zero_row = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:  # pad the block's K/V rows with zeros (Clear_OOB_MN); masked columns are never read
            if zero_row is None:
                zero_row = _Coll(Array(D, BF16), concat([B.call(ZERO16).refs] * D))
            pad = [zero_row.refs] * (BN - nvis)
            kb = _Coll(Array(BN, Array(D, BF16)), concat([kb.refs] + pad))
            vb = _Coll(Array(BN, Array(D, BF16)), concat([vb.refs] + pad))
        first = b == nb - 1
        blk = bind(block, D=D, NVIS=nvis, FIRST=first, BN=BN)
        if first:
            z = B.call(ZERO32)
            m0 = z
            l0 = _Coll(Array(4, F32), concat([z.refs] * 4))
            o0 = _Coll(Array(D, F32), concat([z.refs] * D))
            st = B.call(blk, q, kb, vb, m0, l0, o0, *extra)
        else:
            st = B.call(blk, q, kb, vb, m, l, o, *extra)
        m, l, o = st[0], st[1], st[2]
    total = B.call(P.F32AddFtz, B.call(P.F32AddFtz, l[0], l[2]), B.call(P.F32AddFtz, l[1], l[3]))
    inv = B.call(P.Fa2InvSum, total)
    y = B.batch(P.F32MulFtz, o, inv, axes=(0, None))
    return B.batch(P.F2fpBf16, y, axes=(0,))


@composite("Attention", 3, ["T", "NH", "KVH", "D", "BN"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="exact-model-tested")
def AttentionV3(B, S, q, kc, vc, *extra, head=AttentionHeadV3):
    """`Attention_v1`'s body over `AttentionHead_v3{T, D, BN}` (GQA: head h reads KV head h // (NH/KVH); caches `[T][KVH][D]`)."""
    T, NH, KVH, D, BN = S.T, S.NH, S.KVH, S.D, S.BN
    G = NH // KVH
    outs = []
    for g in range(KVH):
        kg = _Coll(Array(T, Array(D, BF16)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = _Coll(Array(T, Array(D, BF16)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(bind(head, T=T, D=D, BN=BN), q[g * G:(g + 1) * G], kg, vg, *extra, axes=(0, None, None) + (None,) * len(extra)))
    return _Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


def _forward_positions_v4(B, C: dict, BN: int, toks: Coll, weights: Coll, T0: int, n_pos: int, caches: list[tuple[Coll, Coll]] | None):
    """`_forward_positions` with `Attention_v3{..., BN}` at the attention call; everything else TA1's (same names, same order)."""
    NL, H = C["NL"], C["H"]
    hs = [B.call(bind(Embedding, V=C["V"], H=H), toks[i], weights["embed"], name=f"t={T0 + i}/embed") for i in range(n_pos)]
    res = [None] * n_pos
    new_kv: list[tuple[Coll, Coll]] = []
    for L in range(NL):
        w = weights["layers"][L]
        pre = [B.call(bind(LayerPre, C=C, L0=(L == 0)), hs[i], hs[i] if L == 0 else res[i], w, weights["cos_sin"][T0 + i],
                      name=f"t={T0 + i}/layers.{L}/pre") for i in range(n_pos)]
        k_rows = _Coll(kv_type(C, n_pos), concat([p[2].refs for p in pre]))
        v_rows = _Coll(kv_type(C, n_pos), concat([p[3].refs for p in pre]))
        if caches is None:
            kc, vc = k_rows, v_rows
        else:
            kc = _Coll(kv_type(C, T0 + n_pos), concat([caches[L][0].refs, k_rows.refs]))
            vc = _Coll(kv_type(C, T0 + n_pos), concat([caches[L][1].refs, v_rows.refs]))
        new_kv.append((k_rows, v_rows))
        for i in range(n_pos):
            T = T0 + i + 1
            attn = B.call(bind(AttentionV3, T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"], BN=BN), pre[i][1], kc[0:T], vc[0:T],
                          name=f"t={T - 1}/layers.{L}/attn")
            post = B.call(bind(LayerPost, C=C), attn, pre[i][0], w, name=f"t={T - 1}/layers.{L}/post")
            hs[i], res[i] = post[0], post[1]
    return new_kv, list(zip(hs, res))


@composite("Serve", 4, ["C", "LP", "STEPS", "BN"],
           lambda S: ((("prompt", Array(S.LP, I32)), ("weights", weights_type(S.C))), Array(S.STEPS + 1, I32)),
           conformance="model-untested")
def ServeV4(B, S, prompt, weights):
    """`Serve_v1`'s body with the attention kernel's key-block size as the DECLARED static ``BN`` (`Attention_v3{..., BN}`):
    greedy serving of one request, prefill of LP prompt tokens then STEPS decode steps, STEPS+1 sampled tokens.  With
    `BN = 64` it is `Serve_v1` gate for gate (the legacy geometry); with `BN = 128` (the FA2 split-KV kernel at head dim
    <= 64) it differs from `Serve_v1` exactly at query positions t >= 64 (one 128-key block where v1 had two 64-key blocks).
    The instance pins BN in `program.identity.attention.kBlockN` (D57); E6 fills it from the profile via
    `fa2_kernel_geometry`; the verifier binds what the identity says.  [E7 D54/D57 2026-09-13]"""
    C, LP, STEPS, BN = S.C, S.LP, S.STEPS, S.BN
    caches, finals = _forward_positions_v4(B, C, BN, prompt, weights, 0, LP, None)
    head = lm_head_of(C, weights)      # [PROPOSED GEN dense2] the embedding when tied (unchanged), `lm_head` when untied
    tok = B.call(bind(Final, C=C), finals[-1][0], finals[-1][1], weights["norm"], head, name=f"t={LP - 1}/final")
    toks = [tok]
    for s in range(STEPS):
        T0 = LP + s
        new_kv, fin = _forward_positions_v4(B, C, BN, _Coll(Array(1, I32), tok.refs), weights, T0, 1, caches)
        caches = [(_Coll(kv_type(C, T0 + 1), concat([caches[L][0].refs, new_kv[L][0].refs])),
                   _Coll(kv_type(C, T0 + 1), concat([caches[L][1].refs, new_kv[L][1].refs]))) for L in range(C["NL"])]
        tok = B.call(bind(Final, C=C), fin[0][0], fin[0][1], weights["norm"], head, name=f"t={T0}/final")
        toks.append(tok)
    return _Coll(Array(STEPS + 1, I32), concat([t.refs for t in toks]))


ALL_COMPOSITES = [DotBf16, GemmCoordinate, Gemm, BiasAdd, ResidualAdd, RMSNormTriton, RMSNormFusedCudaV1, RMSNormFusedCuda, RoPEHead, RoPE, SiluMul,
                  AttnBlock, AttentionHead, Attention, Embedding, ArgmaxStep, TokenSelect, LayerPre, LayerPost, Final, Serve,
                  AttnBlockV3, AttentionHeadV3, AttentionV3, ServeV4]


# ---------------------------------------------------------------------------------------------------------
# CF1: the same bodies with the accidental constants and the hard-coded
# callees as explicit statics — one `Serve_v2`, two binding records (Ampere: SERVE_AMPERE below; Hopper:
# `hopper.SERVE_HOPPER`)
# ---------------------------------------------------------------------------------------------------------
#
# The definitions above name their callees (`P.AmpereBF16TcDot16`, `P.Fa2InvSum`, `Gemm`, `Attention`, ...) and
# `AttnBlock`/`AttentionHead` read the module constant `BN` (kBlockN = 64). Lane HP obtained the H100 program by
# re-instantiating the same code over a substituted module environment (`hopper.py`): correct, but the selection
# lived in the Python environment, not in the program. Below the selections are static parameters, so the two
# serving programs are two binding records of ONE definition. The bodies are TA1's line for line (the correspondence
# `verity_vllm.program.frontend.correspond.correspond_rebound` shows node-for-node identity against `Serve_v1` and against
# `hopper.ServeHopper_v1`); only the callee/constant names are read from `S`. New *versions* because the identity
# (the statics) changed; every v1 definition stays registered and untouched (registry version discipline).
#
# statics introduced
#   DOT        the k16 tensor-core accumulator step: AmpereBF16TcDot16_v2 (mma.sync m16n8k16) | HopperBF16WgmmaDot16_v1 (wgmma)
#   BN         kBlockN of the attention kernel: 64 (FA2 hdim128 sm_80) | 128 (FA3 tile_size_fwd_sm90 hdim128 causal)
#   INV        the softmax-epilogue inverse sum: Fa2InvSum_v1 ((sum==0||NaN) ? 1 : rcp) | Fa3InvSum_v1 (... ? 0 : rcp)
#   GEMM       {"DOT": <prim>}                                   the linear layers' implementation record
#   ATTENTION  {"BN": int, "DOT": <prim>, "INV": <prim>}         the attention kernel's implementation record
#   NORM       {"L0": RMSNormTriton_v1{N,EPS}, "FUSED": RMSNormFusedCuda_v2{N,EPS}}   complete specialisations (a
#              partially bound composite is not an encodable static value: descriptor-codec.md §5), built by `norm_bundle(C)`
#
# NOT lifted (named in composition.md §1, unchanged across the three profiles): F2fpBf16 (the bf16 store conversion),
# the FA2 lane ownership `_lane`, the four-lane epilogue order, `_scale_log2_bits`, GuardNegInfZero, the zero padding of
# partial blocks, the RMSNorm reduction trees (RMSNormTriton's 128x8 / 1024-chunk, RMSNormFusedCuda's cub BlockReduce),
# BiasAdd / RoPE / SiluMul / Embedding / TokenSelect.


@composite("AttnBlock", 2, ["D", "NVIS", "FIRST", "BN", "DOT"],
           lambda S: (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                        ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="exact-model-tested")
def AttnBlockV2(B, S, q, k, v, m_prev, l_prev, o_prev, scale_log2=None):
    """`AttnBlock_v1` with kBlockN (`BN`) and the tensor-core step (`DOT`) as statics: one BN-key block of the flash online softmax
    for one query row (FA2 with BN=64/mma.sync; FA3 with BN=128/wgmma: hopper-profile.md §4).  `scale_log2`: `AttnBlock_v12`'s input."""
    D, NVIS, FIRST, BN, DOT = S.D, S.NVIS, S.FIRST, S.BN, S.DOT
    dot = bind(DotBf16V2, K=D, DOT=DOT)
    s = [B.call(dot, q, k[c]) for c in range(NVIS)]  # A: S = Q K^T, chained k-steps, acc 0 per block
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = B.call(P.F32Max, bmax, s[c])  # B1: row max
    m_new = bmax if FIRST else B.call(P.F32Max, m_prev, bmax)
    m_use = B.call(P.GuardNegInfZero, m_new)
    scale_log2 = B.call(const(32, _scale_log2_bits(D))) if scale_log2 is None else scale_log2
    if not FIRST:  # B2: rescale factor
        scores_scale = B.call(P.MufuEx2Ftz, B.call(P.F32MulFtz, B.call(P.F32SubFtz, m_prev, m_use), scale_log2))
    max_scaled = B.call(P.F32MulFtz, m_use, scale_log2)
    p = [B.call(P.MufuEx2Ftz, B.call(P.F32FmaSubFtz, s[c], scale_log2, max_scaled)) for c in range(NVIS)]  # B3
    # B4: per-lane sequential sums; the block's first add is contracted with the rescale
    lanes: list[Coll] = []
    for ln in range(4):
        cols = [c for c in range(NVIS) if _lane(c) == ln]
        if FIRST:
            acc = p[cols[0]] if cols else B.call(ZERO32)
            rest = cols[1:]
        else:
            first = p[cols[0]] if cols else B.call(ZERO32)
            acc = B.call(P.F32FmaFtz, scores_scale, l_prev[ln], first)
            rest = cols[1:]
        for c in rest:
            acc = B.call(P.F32AddFtz, acc, p[c])
        lanes.append(acc)
    # B5 + C: P -> bf16, O = (O * scale) + P V in BN/16 ascending k-steps of 16 keys
    pb = [B.call(P.F2fpBf16, x) for x in p]
    zero16 = B.call(ZERO16) if NVIS < BN else None
    o = [B.call(ZERO32) for _ in range(D)] if FIRST else [B.call(P.F32MulFtz, o_prev[d], scores_scale) for d in range(D)]
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break  # all-masked k-step: P = 0 exactly, no accumulator change
        prow = _Coll(A16, concat([(pb[k0 + j] if k0 + j < NVIS else zero16).refs for j in range(16)]))
        vblk = v[k0:k0 + 16]  # Array<16, Array<D>>
        # V column d over the 16 keys: strided view [D][16]
        vt_refs = strided_view(vblk, 0, 16, 1, D, 16 * D) if isinstance(vblk.refs, Affine) else \
            Explicit(tuple(vblk.refs[j * D + d] for d in range(D) for j in range(16)))
        vt = _Coll(Array(D, A16), vt_refs)
        o = list(B.batch(DOT, array_of(o), prow, vt, axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(o))


@composite("AttentionHead", 2, ["T", "D", "BN", "DOT", "INV"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="exact-model-tested")
def AttentionHeadV2(B, S, q, k, v, *extra, block=AttnBlockV2):
    """`AttentionHead_v1` with kBlockN, the tensor-core step and the epilogue inverse sum as statics: causal attention
    of one query at position T-1 over keys 0..T-1, ceil(T/BN) blocks visited last -> first; epilogue
    `sum = (l0 + l2) + (l1 + l3)`, `inv = INV(sum)`, `out = bf16(o * inv)`.  `extra` / `block`: `AttentionHead_v12`'s."""
    T, D, BN, DOT, INV = S.T, S.D, S.BN, S.DOT, S.INV
    nb = (T + BN - 1) // BN
    m = l = o = None
    zero_row = None
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:  # pad the block's K/V rows with zeros (Clear_OOB_MN); masked columns are never read
            if zero_row is None:
                zero_row = _Coll(Array(D, BF16), concat([B.call(ZERO16).refs] * D))
            pad = [zero_row.refs] * (BN - nvis)
            kb = _Coll(Array(BN, Array(D, BF16)), concat([kb.refs] + pad))
            vb = _Coll(Array(BN, Array(D, BF16)), concat([vb.refs] + pad))
        first = b == nb - 1
        blk = bind(block, D=D, NVIS=nvis, FIRST=first, BN=BN, DOT=DOT)
        if first:
            z = B.call(ZERO32)
            m0 = z
            l0 = _Coll(Array(4, F32), concat([z.refs] * 4))
            o0 = _Coll(Array(D, F32), concat([z.refs] * D))
            st = B.call(blk, q, kb, vb, m0, l0, o0, *extra)
        else:
            st = B.call(blk, q, kb, vb, m, l, o, *extra)
        m, l, o = st[0], st[1], st[2]
    total = B.call(P.F32AddFtz, B.call(P.F32AddFtz, l[0], l[2]), B.call(P.F32AddFtz, l[1], l[3]))
    inv = B.call(INV, total)
    y = B.batch(P.F32MulFtz, o, inv, axes=(0, None))
    return B.batch(P.F2fpBf16, y, axes=(0,))


@composite("Attention", 2, ["T", "NH", "KVH", "D", "BN", "DOT", "INV"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="exact-model-tested")
def AttentionV2(B, S, q, kc, vc, *extra, head=AttentionHeadV2):
    """`Attention_v1` over `AttentionHead_v2{T, D, BN, DOT, INV}`: GQA, head h reads KV head h // (NH/KVH).  `extra`: `Attention_v12`'s."""
    T, NH, KVH, D = S.T, S.NH, S.KVH, S.D
    G = NH // KVH
    outs = []
    for g in range(KVH):
        kg = _Coll(Array(T, Array(D, BF16)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = _Coll(Array(T, Array(D, BF16)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(bind(head, T=T, D=D, BN=S.BN, DOT=S.DOT, INV=S.INV), q[g * G:(g + 1) * G], kg, vg, *extra,
                            axes=(0, None, None) + (None,) * len(extra)))
    return _Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


def norm_bundle(C: dict, l0: CompositeDefinition = RMSNormTriton, fused: CompositeDefinition = RMSNormFusedCuda) -> dict:
    """The `NORM` record of a profile at configuration `C`: the layer-0 input norm (`L0`, residual-free) and the fused
    add+norm (`FUSED`) as complete specialisations at width `C["H"]`."""
    return {"L0": bind(l0, N=C["H"], EPS=C["EPS"]), "FUSED": bind(fused, N=C["H"], EPS=C["EPS"])}


@composite("LayerPre", 2, ["C", "L0", "GEMM", "NORM"],
           lambda S: ((("h", Array(S.C["H"], BF16)), ("res", Array(S.C["H"], BF16)), ("w", layer_weights_type(S.C)), ("cs", Array(S.C["D"], BF16))),
                      Tuple(Array(S.C["H"], BF16), Array(S.C["NH"], Array(S.C["D"], BF16)),
                            Array(S.C["KVH"], Array(S.C["D"], BF16)), Array(S.C["KVH"], Array(S.C["D"], BF16)))),
           conformance="exact-model-tested")
def LayerPreV2(B, S, h, res, w, cs):
    """`LayerPre_v1` with the linear and norm implementations as statics (`GEMM["DOT"]`, `NORM["L0"]`/`NORM["FUSED"]`)."""
    C = S.C
    H, NH, KVH, D = C["H"], C["NH"], C["KVH"], C["D"]
    if S.L0:
        x = h
        n1 = B.call(S.NORM["L0"], x, w["ln1"])
    else:
        nr = B.call(S.NORM["FUSED"], h, res, w["ln1"])
        n1, x = nr[0], nr[1]
    qkv = B.call(bind(GemmV2, K=H, N=C["QKV"], DOT=S.GEMM["DOT"]), n1, w["qkv_w"])
    if C.get("BIAS", True):
        qkv = B.call(bind(BiasAdd, N=C["QKV"]), qkv, w["qkv_b"])
    q = _Coll(Array(NH, Array(D, BF16)), qkv.refs.slice(0, NH * D))
    k = _Coll(Array(KVH, Array(D, BF16)), qkv.refs.slice(NH * D, (NH + KVH) * D))
    v = _Coll(Array(KVH, Array(D, BF16)), qkv.refs.slice((NH + KVH) * D, (NH + 2 * KVH) * D))
    qr = B.call(bind(RoPE, NHEADS=NH, D=D), q, cs)
    kr = B.call(bind(RoPE, NHEADS=KVH, D=D), k, cs)
    return tuple_of(x, qr, kr, v)


@composite("LayerPost", 2, ["C", "GEMM", "NORM"],
           lambda S: ((("attn", Array(S.C["NH"], Array(S.C["D"], BF16))), ("res", Array(S.C["H"], BF16)), ("w", layer_weights_type(S.C))),
                      Tuple(Array(S.C["H"], BF16), Array(S.C["H"], BF16))),
           conformance="exact-model-tested")
def LayerPostV2(B, S, attn, res, w):
    """`LayerPost_v1` with the linear and norm implementations as statics."""
    C = S.C
    H = C["H"]
    DOT = S.GEMM["DOT"]
    a = _Coll(Array(C["NH"] * C["D"], BF16), attn.refs)
    o = B.call(bind(GemmV2, K=C["NH"] * C["D"], N=H, DOT=DOT), a, w["o_w"])
    nr = B.call(S.NORM["FUSED"], o, res, w["ln2"])
    n2, r2 = nr[0], nr[1]
    gu = B.call(bind(GemmV2, K=H, N=C["GU"], DOT=DOT), n2, w["gate_up_w"])
    act = B.call(bind(SiluMul, I=C["I"]), gu)
    d = B.call(bind(GemmV2, K=C["I"], N=H, DOT=DOT), act, w["down_w"])
    return tuple_of(d, r2)


@composite("Final", 2, ["C", "GEMM", "NORM"],
           lambda S: ((("h", Array(S.C["H"], BF16)), ("res", Array(S.C["H"], BF16)), ("norm_w", Array(S.C["H"], BF16)),
                       ("lm_head", Array(S.C["V"], Array(S.C["H"], BF16)))), I32),
           conformance="model-untested")
def FinalV2(B, S, h, res, norm_w, lm_head):
    """`Final_v1` with the linear and norm implementations as statics."""
    C = S.C
    n = B.call(S.NORM["FUSED"], h, res, norm_w)[0]
    logits = B.call(bind(GemmV2, K=C["H"], N=C["V"], DOT=S.GEMM["DOT"]), n, lm_head)
    return B.call(bind(TokenSelect, V=C["V"]), logits)


def _forward_positions_v2(B, S, toks: Coll, weights: Coll, T0: int, n_pos: int, caches: list[tuple[Coll, Coll]] | None):
    """`_forward_positions` over the v2 layer functions, the implementation records passed down from `S`."""
    C = S.C
    NL, H = C["NL"], C["H"]
    hs = [B.call(bind(Embedding, V=C["V"], H=H), toks[i], weights["embed"], name=f"t={T0 + i}/embed") for i in range(n_pos)]
    res = [None] * n_pos
    new_kv: list[tuple[Coll, Coll]] = []
    for L in range(NL):
        w = weights["layers"][L]
        pre = [B.call(bind(LayerPreV2, C=C, L0=(L == 0), GEMM=S.GEMM, NORM=S.NORM), hs[i], hs[i] if L == 0 else res[i], w,
                      weights["cos_sin"][T0 + i], name=f"t={T0 + i}/layers.{L}/pre") for i in range(n_pos)]
        k_rows = _Coll(kv_type(C, n_pos), concat([p[2].refs for p in pre]))
        v_rows = _Coll(kv_type(C, n_pos), concat([p[3].refs for p in pre]))
        if caches is None:
            kc, vc = k_rows, v_rows
        else:
            kc = _Coll(kv_type(C, T0 + n_pos), concat([caches[L][0].refs, k_rows.refs]))
            vc = _Coll(kv_type(C, T0 + n_pos), concat([caches[L][1].refs, v_rows.refs]))
        new_kv.append((k_rows, v_rows))
        for i in range(n_pos):
            T = T0 + i + 1
            attn = B.call(bind(AttentionV2, T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"], **S.ATTENTION), pre[i][1], kc[0:T], vc[0:T],
                          name=f"t={T - 1}/layers.{L}/attn")
            post = B.call(bind(LayerPostV2, C=C, GEMM=S.GEMM, NORM=S.NORM), attn, pre[i][0], w, name=f"t={T - 1}/layers.{L}/post")
            hs[i], res[i] = post[0], post[1]
    return new_kv, list(zip(hs, res))


@composite("Serve", 2, ["C", "LP", "STEPS", "GEMM", "ATTENTION", "NORM"],
           lambda S: ((("prompt", Array(S.LP, I32)), ("weights", weights_type(S.C))), Array(S.STEPS + 1, I32)),
           conformance="model-untested")
def ServeV2(B, S, prompt, weights):
    """`Serve_v1` with the profile's implementation records as statics: `GEMM` (the linear layers' k16 step),
    `ATTENTION` (kBlockN, k16 step, epilogue inverse sum) and `NORM` (the two norm functions at width H). The Ampere
    eager program is `bind(Serve_v2, C=B1_CONFIG, LP, STEPS, GEMM=GEMM_AMPERE, ATTENTION=ATTENTION_FA2, NORM=norm_bundle(C))`;
    the Hopper eager program differs in `GEMM` and `ATTENTION` only (`hopper.GEMM_HOPPER`, `hopper.ATTENTION_FA3`)."""
    C, LP, STEPS = S.C, S.LP, S.STEPS
    fin = bind(FinalV2, C=C, GEMM=S.GEMM, NORM=S.NORM)
    caches, finals = _forward_positions_v2(B, S, prompt, weights, 0, LP, None)
    tok = B.call(fin, finals[-1][0], finals[-1][1], weights["norm"], weights["embed"], name=f"t={LP - 1}/final")
    toks = [tok]
    for s in range(STEPS):
        T0 = LP + s
        new_kv, last = _forward_positions_v2(B, S, _Coll(Array(1, I32), tok.refs), weights, T0, 1, caches)
        caches = [(_Coll(kv_type(C, T0 + 1), concat([caches[L][0].refs, new_kv[L][0].refs])),
                   _Coll(kv_type(C, T0 + 1), concat([caches[L][1].refs, new_kv[L][1].refs]))) for L in range(C["NL"])]
        tok = B.call(fin, last[0][0], last[0][1], weights["norm"], weights["embed"], name=f"t={T0}/final")
        toks.append(tok)
    return _Coll(Array(STEPS + 1, I32), concat([t.refs for t in toks]))


#: The Ampere (sm_80/sm_89 eager, TA1's profile) implementation records.
GEMM_AMPERE: dict = {"DOT": P.AmpereBF16TcDot16}
ATTENTION_FA2: dict = {"BN": BN, "DOT": P.AmpereBF16TcDot16, "INV": P.Fa2InvSum}  # the HEAD DIM 128 record (W11); see attention_fa2


def attention_fa2(D: int) -> dict:
    """[E7 D54] The FA2 `ATTENTION` record at head dim ``D``: `BN = fa2_kblock_n(D)` (128 at D <= 64, 64 at D = 128 =
    `ATTENTION_FA2`), the Ampere k16 step and epilogue -- `Serve_v2`'s binding record of the `Serve_v4` program."""
    return {"BN": fa2_kblock_n(D), "DOT": P.AmpereBF16TcDot16, "INV": P.Fa2InvSum}


def serve_v2(C: dict, LP: int, STEPS: int, gemm: dict = GEMM_AMPERE, attention: dict = ATTENTION_FA2, norm: dict | None = None):
    """`Serve_v2` bound for one profile; by default the Ampere eager profile (TA1's program as a binding record: corresponds to
    `Serve_v1`).  With `attention=attention_fa2(C["D"])` it is the D54-corrected program's binding record (corresponds to `Serve_v4`)."""
    return bind(ServeV2, C=C, LP=LP, STEPS=STEPS, GEMM=gemm, ATTENTION=attention, NORM=norm if norm is not None else norm_bundle(C))


def SERVE_AMPERE(C: dict, LP: int, STEPS: int):
    """The Ampere eager B1/B0 program: `Serve_v2` under `GEMM_AMPERE` / `ATTENTION_FA2` / `norm_bundle(C)`."""
    return serve_v2(C, LP, STEPS)


FACTORED_COMPOSITES = [DotBf16V2, GemmCoordinateV2, GemmV2, AttnBlockV2, AttentionHeadV2, AttentionV2, LayerPreV2, LayerPostV2, FinalV2, ServeV2]
#: v1 definition -> its factored v2 (the definitions whose bodies named a callee or read `BN`).
FACTORED_FROM = {DotBf16: DotBf16V2, GemmCoordinate: GemmCoordinateV2, Gemm: GemmV2, AttnBlock: AttnBlockV2, AttentionHead: AttentionHeadV2,
                 Attention: AttentionV2, LayerPre: LayerPreV2, LayerPost: LayerPostV2, Final: FinalV2, Serve: ServeV2}

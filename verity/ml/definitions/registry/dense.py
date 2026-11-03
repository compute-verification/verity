"""verity.ml.definitions.registry.dense — the generic elementwise / reduction / activation / softcap Definitions of the eager batch-invariant vLLM
profile that the fused b1 kernels do NOT cover: the operations a model runs UNFUSED when its norm is `forward_native`
(`ir.ops.rms_norm` / `ir.ops.fused_add_rms_norm` as an ATen chain on an f32 copy), its activation is GeGLU (`_C.gelu_tanh_and_mul`),
its attention has a logit soft cap (FA2 `Is_softcap`), or its logits are soft-capped (`LogitsProcessor(soft_cap)`).

PROMOTED from `verity_vllm.program.quarantine.dense`: every
Definition below is the quarantine module's body VERBATIM — same name, same version, same statics, same gates — so the
specialization ids are those the accepted `GEMMA2_2B-cov_gemma2_2b_b1_c256.r8h` record and the eager Gemma-2 (`profiles.generic`, formerly `gen_dense2_gemma2_2b`)
Match profile resolve to. The quarantine modules re-export these objects (their difftest adapters, `ALL_DEFINITIONS` and
`SEMANTIC_ID`s are unchanged). Each section keeps the quarantine module's docstring: the ATen / Triton / CUDA source the body
follows, the wave-1 exhaustive GPU tables and the pod difftests that are its correspondence evidence (see the `conformance=`
strings).

Bound as kinds of the pinned `b1-eager-v3` vocabulary (`frontend/rules/vocab.py::_dense_kinds`) and asked for by the op-shaped
rules of `frontend/rules/vllm_bindings.py` (`WidenBf16ToF32`, `Square`, `MeanDimBI`, `Rsqrt`, `AddScalar`, `AddWidened`, `MulF32`,
`NarrowF32ToBf16`, `GeluTanhAndMul`, `Bf16ScalarChain`, the softcap role of `UnifiedAttentionWithOutput`). Nothing here names a model.

Conventions (checked against wave-1's exhaustive GPU tables): every ATen CUDA kernel here converts f32 -> bf16 with the hardware
`cvt.rn.bf16.f32` (`c10::BFloat16(float)` on sm_80+ is `__float2bfloat16`), so NaN words are 0x7FFF (`F2fpBf16`), not the
software 0x7FC0. `MufuTanh` (the FA2 softcap's `tanh.approx.f32`) stays a primitive of `registry/prims.py` with its measured table.
"""
from __future__ import annotations

import math

import numpy as np

from verity.ml.circuits.defs import Coll, bind, composite, primitive
from verity.ml.circuits.refs import Affine, Coll as _Coll, Explicit, array_of, concat, strided_view, tuple_of
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.b1 import ZERO16, ZERO32, DotBf16, _butterfly, _lane, const, f32c
from verity.ml.definitions.registry.prims import A16
from verity.ml.circuits.types import Array, Tuple, Value

BF16, F32 = Value(16), Value(32)

#: `SEMANTIC_ID`s of the promoted Definitions (quarantine identity, one per Definition; the quarantine modules keep the same strings)
SEMANTIC_IDS = {
    "SquareBf16": "dense/SquareBf16@v1", "SquareF32": "dense/SquareF32@v1", "MeanTriton": "dense/MeanTriton@v1",
    "AddScalarF32": "dense/AddScalarF32@v1", "AddScalarBf16": "dense/AddScalarBf16@v1", "AddWidenedBf16": "dense/AddWidenedBf16@v1",
    "RsqrtF32": "dense/RsqrtF32@v1", "ScaleRowBf16": "dense/ScaleRowBf16@v1", "ScaleRowF32": "dense/ScaleRowF32@v1",
    "MulVecF32": "dense/MulVecF32@v1", "NarrowF32ToBf16": "dense/NarrowF32ToBf16@v1", "Bf16DivScalar": "dense/Bf16DivScalar@v1",
    "TanhF32Rn": "dense/TanhF32Rn@v1", "Bf16Tanh": "dense/Bf16Tanh@v1", "Bf16MulScalar": "dense/Bf16MulScalar@v1",
    "GeluTanhMulBf16": "dense/GeluTanhMulBf16@v1", "GeluTanhMul": "dense/GeluTanhMul@v1", "AttentionSoftcap": "dense/AttentionSoftcap@v1",
}


# ========================================================================================================================
# square_bf16  (SEMANTIC_ID dense/SquareBf16@v1)
# ========================================================================================================================
#: `aten.pow.Tensor_Scalar(x32, 2)` where `x32 = x.to(float32)` is the exact widening of a bf16 tensor (GemmaRMSNorm without
#: residual: `ir.ops.rms_norm`).  ATen's CUDA `pow_tensor_scalar_kernel` special-cases the exponent 2 as `base * base`
#: (one FMUL, RN); the widening is a Transparent noop, so the definition reads the bf16 source (wave-1 probe `pow2_f32`:
#: `aten.pow(x, 2) == x * x` on 2^20 f32 samples).

@composite("SquareBf16", 1, ["N"], lambda S: ((("x", Array(S.N, BF16)),), Array(S.N, F32)),
           conformance="quarantine (op-level evidence only): difftest vs aten.pow(x.float(), 2) on the pod")
def SquareBf16(B, S, x):
    """`out_j = RN32(f32(x_j) * f32(x_j))` (exact: a bf16 significand squared fits in 24 bits)."""
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    return B.batch(P.F32Mul, xf, xf, axes=(0, 0))


# ========================================================================================================================
# square_f32  (SEMANTIC_ID dense/SquareF32@v1)
# ========================================================================================================================
#: `aten.pow.Tensor_Scalar(s32, 2)` on a genuine f32 tensor (GemmaRMSNorm with residual: `ir.ops.fused_add_rms_norm`'s
#: `x = x.to(f32) + residual.to(f32)`).  ATen's CUDA `pow_tensor_scalar_kernel` special-cases the exponent 2 as `base * base`.

@composite("SquareF32", 1, ["N"], lambda S: ((("x", Array(S.N, F32)),), Array(S.N, F32)),
           conformance="quarantine (op-level evidence only): difftest vs aten.pow(x, 2) on the pod")
def SquareF32(B, S, x):
    """`out_j = RN32(x_j * x_j)` (FMUL)."""
    return B.batch(P.F32Mul, x, x, axes=(0, 0))


# ========================================================================================================================
# mean_triton  (SEMANTIC_ID dense/MeanTriton@v1)
# ========================================================================================================================
#: `aten.mean.dim(v32 [M,N], [-1], keepdim=True)` under `VLLM_BATCH_INVARIANT=1`: vLLM d9105ea80
#: `vllm/model_executor/determinism/batch_invariant.py::mean_dim` launches the Triton `mean_kernel` with `BLOCK_SIZE=1024`
#: (`num_warps=4`, the JIT default), the input viewed as `(M, N, 1)`, one program per output element:
#: 
#:     acc = 0.0
#:     for n_start in range(0, N, BLOCK_SIZE):
#:         vals = tl.load(input_ptr + idx, mask=n_offsets < N, other=0.0)
#:         acc += tl.sum(vals)
#:     mean_val = acc / N
#: 
#: The registered body writes that reduction out gate by gate (the same Triton 3.7.1 lowering the b1 `RMSNormTriton_v1` body
#: models for `_rms_norm_kernel`): a [1024] f32 block is held by 128 threads in the blocked layout `sizePerThread=4`
#: (128-bit loads), `threadsPerWarp=32`, `warpsPerCTA=4` — thread t owns elements `4t..4t+3` of each of the two 512-element
#: repetitions (registers in repetition-major order); `tl.sum` accumulates a thread's 8 registers sequentially
#: (`reduceWithinThreads`), then a warp xor-butterfly 16,8,4,2,1 (`reduceWithinWarps`), then the four warp results as
#: `(w0 + w2) + (w1 + w3)` through shared memory; masked lanes contribute `+0.0` literally.  `acc += ...` adds the chunk sums
#: in order from `0.0`; `acc / N` is Triton's f32 division, `div.full.f32` (`DivFullScaleA` * `DivFullRcp`, as in
#: `RMSNormTriton_v1`).  Output: `Array(1, F32)` — the row's `keepdim` cell.
#: 
#: The per-thread ownership (`SIZE_PER_THREAD = 4`, two repetitions) is the layout the coalescing pass picks for a contiguous
#: f32 load whose row pointer is provably 16-byte aligned (the same as the b1 `RMSNormTriton_v1` body's).  Evidence that it is
#: the production order: the difftest adapter (`mean_triton_difftest.py`) is bit-exact against `mean_batch_invariant` on the
#: pod at N = 2304 (Gemma-2's H: two full blocks + a masked 256-tail), 1024, 1280, 2048 and 256 with random f32 rows — a
#: different register order or per-thread width changes the rounding of most random rows.  The 32-case
#: seed-7 difftest found the layout is *specialisation-dependent*: for `N % 16 != 0` Triton cannot prove the alignment and
#: the layout is `sizePerThread=1` (`size_per_thread`, below); the body follows the same rule.

BLOCK_SIZE = 1024
NUM_WARPS = 4
THREADS = 32 * NUM_WARPS
SIZE_PER_THREAD = 4                      # f32 elements per thread per repetition (128-bit vectorised load), N % 16 == 0
TILE = SIZE_PER_THREAD * THREADS         # 512 elements covered by one pass of the CTA
REPS = BLOCK_SIZE // TILE                # 2 repetitions of the tile inside one [1024] block


def size_per_thread(N: int) -> int:
    """The blocked layout is a property of the *kernel specialisation*,
    not of the source: `mean_dim` passes `input_stride0 = N` (a `[1, N]` row) as a plain integer argument, and Triton
    specialises integer arguments on divisibility by 16 (`tt.divisibility=16`) only.  With `N % 16 == 0` the row pointer
    `input_ptr + m_idx * input_stride0 + n_offsets` is provably 16-byte aligned, so the coalescing pass gives every thread
    4 contiguous f32 (`sizePerThread=4`, the layout above); otherwise the provable alignment is 4 bytes and the layout is
    `sizePerThread=1`: thread t owns elements `t, t+128, ..., t+7*128` of the block.  Production evidence: the six N=4 `max`
    cases of the seed-7 difftest (`(128 + -MAX) + (-MAX + MAX)` = -MAX in production; the 4-wide layout's sequential chain
    gives -inf) and the random-normal small-N cases of the adapter; N = 2304 / 2048 / 1280 / 1024 / 256 are all in the
    16-divisible specialisation and were bit-exact before and after."""
    return SIZE_PER_THREAD if N % 16 == 0 else 1


def thread_elements(c0: int, t: int, spt: int = SIZE_PER_THREAD) -> list[int]:
    """Global element indices thread `t` (0..127) holds for the block starting at `c0`, in register order (`spt` contiguous
    elements per repetition, `BLOCK_SIZE / (spt * 128)` repetitions)."""
    tile = spt * THREADS
    return [c0 + rep * tile + t * spt + j for rep in range(BLOCK_SIZE // tile) for j in range(spt)]


@composite("MeanTriton", 1, ["N"], lambda S: ((("x", Array(S.N, F32)),), Array(1, F32)),
           conformance="quarantine (op-level evidence only): difftest vs batch_invariant.mean_dim on the pod")
def MeanTriton(B, S, x):
    """`mean_kernel` (BLOCK_SIZE=1024, 4 warps) over one row of N f32 values; see the module docstring."""
    N = S.N
    spt = size_per_thread(N)
    zero = B.call(ZERO32)
    acc = zero
    for c0 in range(0, N, BLOCK_SIZE):
        warps = []
        for w in range(NUM_WARPS):
            lanes = []
            for ln in range(32):
                regs = [x[i] if i < N else zero for i in thread_elements(c0, w * 32 + ln, spt)]
                a = regs[0]
                for r in regs[1:]:
                    a = B.call(P.F32Add, a, r)
                lanes.append(a)
            warps.append(_butterfly(B, lanes, (16, 8, 4, 2, 1), P.F32Add))
        chunk = B.call(P.F32Add, B.call(P.F32Add, warps[0], warps[2]), B.call(P.F32Add, warps[1], warps[3]))
        acc = B.call(P.F32Add, acc, chunk)
    nf = B.call(f32c(float(N)))
    mean = B.call(P.F32Mul, B.call(P.DivFullScaleA, acc, nf), B.call(P.DivFullRcp, nf))
    return mean.reshape(Array(1, F32))


# ========================================================================================================================
# add_scalar_f32  (SEMANTIC_ID dense/AddScalarF32@v1)
# ========================================================================================================================
#: `aten.add.Tensor(v32, c)` — an f32 tensor plus a Python scalar (GemmaRMSNorm: `variance + variance_epsilon`, N = 1 per
#: row, C = 1e-6).  ATen's CUDA `add_kernel` computes in opmath float: `a + alpha * b` with `alpha = 1` and the scalar
#: narrowed once to f32 (`f32(1e-6)` = 0x358637BD); one FADD, RN (wave-1 probe `add_eps_f32`: `aten.add(v, 1e-6) ==
#: f32(v + f32(1e-6))` on 2^20 samples).

@composite("AddScalarF32", 1, ["N", "C"], lambda S: ((("x", Array(S.N, F32)),), Array(S.N, F32)),
           conformance="quarantine (op-level evidence only): difftest vs aten.add(x, C) on the pod")
def AddScalarF32(B, S, x, c=None):
    """`out_j = RN32(x_j + f32(C))`; `c`: `AddScalarF32_v3`'s input (the f32 word `f32(C)`)."""
    c = B.call(f32c(float(S.C))) if c is None else c
    return B.batch(P.F32Add, x, c, axes=(0, None))


# ========================================================================================================================
# add_scalar_bf16  (SEMANTIC_ID dense/AddScalarBf16@v1)
# ========================================================================================================================
#: `aten.add.Tensor(w32, c)` where `w32 = w.to(float32)` is the exact widening of a bf16 weight (GemmaRMSNorm.forward_native:
#: `weight = self.weight.float() + 1.0`, N = H, C = 1.0).  The widening is a Transparent noop, so the definition reads the bf16
#: root; one FADD in f32, RN.

@composite("AddScalarBf16", 1, ["N", "C"], lambda S: ((("w", Array(S.N, BF16)),), Array(S.N, F32)),
           conformance="quarantine (op-level evidence only): difftest vs aten.add(w.float(), C) on the pod")
def AddScalarBf16(B, S, w, c=None):
    """`out_j = RN32(f32(w_j) + f32(C))`; `c`: `AddScalarBf16_v3`'s input (the f32 word `f32(C)`)."""
    wf = B.batch(P.Bf16ToF32, w, axes=(0,))
    c = B.call(f32c(float(S.C))) if c is None else c
    return B.batch(P.F32Add, wf, c, axes=(0, None))


# ========================================================================================================================
# add_widened_bf16  (SEMANTIC_ID dense/AddWidenedBf16@v1)
# ========================================================================================================================
#: `aten.add.Tensor(x32, r32)` where both operands are exact f32 widenings of bf16 tensors (GemmaRMSNorm with residual,
#: `ir.ops.fused_add_rms_norm`: `x = x.to(float32); x = x + x_residual.to(float32)`).  ATen's CUDA `add_kernel` on f32
#: (`alpha = 1`): one FADD per element, RN.  Both widenings are Transparent noops, so the definition reads the two bf16
#: sources.

@composite("AddWidenedBf16", 1, ["N"], lambda S: ((("x", Array(S.N, BF16)), ("r", Array(S.N, BF16))), Array(S.N, F32)),
           conformance="quarantine (op-level evidence only): difftest vs aten.add(x.float(), r.float()) on the pod")
def AddWidenedBf16(B, S, x, r):
    """`out_j = RN32(f32(x_j) + f32(r_j))`."""
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    rf = B.batch(P.Bf16ToF32, r, axes=(0,))
    return B.batch(P.F32Add, xf, rf, axes=(0, 0))


# ========================================================================================================================
# rsqrt_f32  (SEMANTIC_ID dense/RsqrtF32@v1)
# ========================================================================================================================
#: `aten.rsqrt.default(v32)` on f32 (GemmaRMSNorm: `torch.rsqrt(variance + eps)`, N = 1 per row).  ATen's CUDA
#: `rsqrt_kernel_cuda` calls `::rsqrt(float)` = libdevice `rsqrtf`, which ptxas emits as MUFU.RSQ with the subnormal range
#: scaling of `rsqrt.approx.f32` — the registered `RsqrtApprox_v1` (the same instruction sequence vLLM's CUDA RMSNorm kernels
#: use; wave-1 probe `rsqrt_f32`: equal to `RsqrtApprox` on 1,048,576 of 1,048,576 samples, and NOT the correctly rounded
#: 1/sqrt: 808,881 of them).

@composite("RsqrtF32", 1, ["N"], lambda S: ((("x", Array(S.N, F32)),), Array(S.N, F32)),
           conformance="quarantine (op-level evidence only): difftest vs aten.rsqrt on the pod")
def RsqrtF32(B, S, x):
    """`out_j = rsqrt.approx.f32(x_j)` (MUFU.RSQ + range scaling; `RsqrtApprox_v1`)."""
    return B.batch(P.RsqrtApprox, x, axes=(0,))


# ========================================================================================================================
# scale_row_bf16  (SEMANTIC_ID dense/ScaleRowBf16@v1)
# ========================================================================================================================
#: `aten.mul.Tensor(x32 [M,N], r [M,1])` — a row scaled by its own f32 scalar, where `x32 = x.to(float32)` is the exact widening
#: of a bf16 tensor (GemmaRMSNorm without residual: `x * torch.rsqrt(variance + eps)`).  ATen's CUDA `mul_kernel` on f32 with
#: the `[M,1]` operand broadcast along the row: one FMUL per element, RN.  The widening is a Transparent noop, so the definition
#: reads the bf16 source; `s` is the row's `[1]` cell of the `[M,1]` operand (PerRow).

@composite("ScaleRowBf16", 1, ["N"], lambda S: ((("x", Array(S.N, BF16)), ("s", Array(1, F32))), Array(S.N, F32)),
           conformance="quarantine (op-level evidence only): difftest vs aten.mul(x.float(), s[:, None]) on the pod")
def ScaleRowBf16(B, S, x, s):
    """`out_j = RN32(f32(x_j) * s)`."""
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    return B.batch(P.F32Mul, xf, s[0], axes=(0, None))


# ========================================================================================================================
# scale_row_f32  (SEMANTIC_ID dense/ScaleRowF32@v1)
# ========================================================================================================================
#: `aten.mul.Tensor(s32 [M,N], r [M,1])` — an f32 row scaled by its own f32 scalar (GemmaRMSNorm with residual:
#: `(x.float() + residual.float()) * torch.rsqrt(variance + eps)`).  ATen's CUDA `mul_kernel` on f32 with the `[M,1]` operand
#: broadcast along the row: one FMUL per element, RN; `s` is the row's `[1]` cell of the `[M,1]` operand (PerRow).

@composite("ScaleRowF32", 1, ["N"], lambda S: ((("x", Array(S.N, F32)), ("s", Array(1, F32))), Array(S.N, F32)),
           conformance="quarantine (op-level evidence only): difftest vs aten.mul(x, s[:, None]) on the pod")
def ScaleRowF32(B, S, x, s):
    """`out_j = RN32(x_j * s)`."""
    return B.batch(P.F32Mul, x, s[0], axes=(0, None))


# ========================================================================================================================
# mul_vec_f32  (SEMANTIC_ID dense/MulVecF32@v1)
# ========================================================================================================================
#: `aten.mul.Tensor(y32 [M,N], w1 [N])` — an f32 row times a shared f32 vector (GemmaRMSNorm: `x * (weight.float() + 1.0)`,
#: the `[N]` operand broadcast over rows).  ATen's CUDA `mul_kernel` on f32: one FMUL per element, RN.

@composite("MulVecF32", 1, ["N"], lambda S: ((("x", Array(S.N, F32)), ("w", Array(S.N, F32))), Array(S.N, F32)),
           conformance="quarantine (op-level evidence only): difftest vs aten.mul(x, w) on the pod")
def MulVecF32(B, S, x, w):
    """`out_j = RN32(x_j * w_j)`."""
    return B.batch(P.F32Mul, x, w, axes=(0, 0))


# ========================================================================================================================
# narrow_f32_bf16  (SEMANTIC_ID dense/NarrowF32ToBf16@v1)
# ========================================================================================================================
#: `aten.to.dtype(y32, bfloat16)` — the f32 -> bf16 narrowing copy (GemmaRMSNorm: `x.to(orig_dtype)`; the fused variant's
#: `residual = x.to(orig_dtype)`).  ATen's CUDA copy kernel converts with `c10::BFloat16(float)`, which on sm_80+ device code is
#: `__float2bfloat16` = `cvt.rn.bf16.f32`: RNE, NaN -> 0x7FFF — the registered `F2fpBf16_v1` (the same word the b1 GEMM /
#: RMSNorm stores use; wave-1's exhaustive tables of two other ATen bf16 kernels show the 0x7FFF NaN word).

@composite("NarrowF32ToBf16", 1, ["N"], lambda S: ((("x", Array(S.N, F32)),), Array(S.N, BF16)),
           conformance="quarantine (op-level evidence only): difftest vs x.to(torch.bfloat16) on the pod")
def NarrowF32ToBf16(B, S, x):
    """`out_j = cvt.rn.bf16.f32(x_j)`."""
    return B.batch(P.F2fpBf16, x, axes=(0,))


# ========================================================================================================================
# bf16_div_scalar  (SEMANTIC_ID dense/Bf16DivScalar@v1)
# ========================================================================================================================
#: `aten.div.Tensor(l [M,N] bf16, c)` — a bf16 tensor divided by a Python scalar (Gemma-2 `LogitsProcessor`, `soft_cap=30`:
#: `logits / self.soft_cap`).  ATen's CUDA `div_true_kernel_cuda` takes the CPU-scalar fast path: `inv_b = opmath_t(1) / b`
#: computed once on the host in f32 (correctly rounded: `f32(1)/f32(30)` = 0x3D088889) and then `f32(x) * inv_b` per element
#: (`BUnaryFunctor<MulFunctor<float>>`), converted with `cvt.rn.bf16.f32` (NaN -> 0x7FFF).  On the bf16 domain this equals
#: `bf16(f32(x) / 30)` at every one of the 65,536 words (checked offline; wave-1 probe `div30` compared the kernel with the
#: true-division form and found all 65,282 non-NaN words equal), so the two readings are indistinguishable here; the body
#: follows ATen's actual algorithm.

def reciprocal_bits(c: float) -> int:
    """`opmath_t(1.0) / c` in f32 (host, correctly rounded)."""
    return P.f32_div_bits(0x3F800000, P.bits_of(c))


@composite("Bf16DivScalar", 1, ["N", "C"], lambda S: ((("x", Array(S.N, BF16)),), Array(S.N, BF16)),
           conformance="quarantine (op-level evidence only): difftest vs aten.div(x, C) on the pod")
def Bf16DivScalar(B, S, x, inv=None):
    """`out_j = cvt.rn.bf16.f32(f32(x_j) * RN32(1 / C))`; `inv`: `Bf16DivScalar_v3`'s input (the f32 word `RN32(1 / C)`)."""
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    inv = B.call(const(32, reciprocal_bits(float(S.C)))) if inv is None else inv
    return B.batch(P.F2fpBf16, B.batch(P.F32Mul, xf, inv, axes=(0, None)), axes=(0,))


# ========================================================================================================================
# tanh_f32  (SEMANTIC_ID dense/TanhF32Rn@v1)
# ========================================================================================================================
#: `TanhF32Rn` — a stand-in for CUDA `tanhf` (libdevice `__nv_tanhf`, as ATen's `tanh_kernel_cuda` and vLLM's
#: `gelu_tanh_kernel` call it): RN32 of the exact real tanh(x), the same idiom as the registered `_silu_f32_bits` stand-in for
#: `expf` + division.  Total on encodings: NaN -> canonical NaN, ±inf -> ±1, ±0 -> ±0, |x| >= 10 -> ±1 exactly (tanh is within
#: 2^-28 of 1 there), subnormal x -> x.
#: 
#: Evidence that the stand-in is exact where it is used: both uses take a *bf16-valued* argument (`aten.tanh` on bf16;
#: `gelu_tanh_kernel`'s `inner` is a function of one bf16 word), so the composed bf16 -> bf16 functions have a finite domain of
#: 65,536 words, and wave-1's exhaustive GPU tables (`tanh_bf16_table.npy`, `gelu_tanh_bf16_table.npy`) agree with the
#: stand-in-based bodies on every non-NaN word (65,282 / 65,282 and 65,282 / 65,282; NaN inputs give the kernels' 0x7FFF).  CUDA's
#: `tanhf` (<= 2 ulp) is therefore not distinguishable from RN32(tanh) through either bf16 output on the whole input domain.






def tanh_f32_bits(x: int) -> int:
    c, v = P._decode(x)
    if c == 2:
        return P.F32_NAN
    if c == 1:
        return 0x3F800000
    if c == -1:
        return 0xBF800000
    if v == 0:
        return x & 0x80000000
    f = float(v)
    if abs(f) >= 10.0:
        return 0x3F800000 if f > 0 else 0xBF800000
    return P.bits_of(math.tanh(f))


@primitive("TanhF32Rn", 1, [("x", F32)], F32,
           conformance="quarantine (op-level evidence only): stand-in for libdevice tanhf; exact through both bf16 uses on all 65,536 words",
           tables=(("tanh_rn", "registered"),))
def TanhF32Rn(x: int) -> int:
    """RN32(tanh(x)) on encodings (float64 tanh, then one rounding to f32)."""
    return tanh_f32_bits(x)


# ========================================================================================================================
# bf16_tanh  (SEMANTIC_ID dense/Bf16Tanh@v1)
# ========================================================================================================================
#: `aten.tanh.default(t [M,N] bf16)` (Gemma-2 `LogitsProcessor`: `torch.tanh(logits / soft_cap)`).  ATen's CUDA `tanh_kernel_cuda`
#: computes `::tanh(float(x))` = libdevice `tanhf` in opmath float and converts once with `cvt.rn.bf16.f32` (NaN -> 0x7FFF).
#: wave-1's exhaustive GPU table (`tanh_bf16_table.npy`) equals `bf16(RN32(tanh(f32(x))))` on all 65,282 non-NaN words and
#: 0x7FFF on the 254 NaN words — the finite domain makes the `TanhF32Rn` stand-in exact here.

@composite("Bf16Tanh", 1, ["N"], lambda S: ((("x", Array(S.N, BF16)),), Array(S.N, BF16)),
           conformance="quarantine (op-level evidence only): exhaustive table match (wave-1) + difftest vs aten.tanh on the pod")
def Bf16Tanh(B, S, x):
    """`out_j = cvt.rn.bf16.f32(tanhf(f32(x_j)))`."""
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    return B.batch(P.F2fpBf16, B.batch(TanhF32Rn, xf, axes=(0,)), axes=(0,))


# ========================================================================================================================
# bf16_mul_scalar  (SEMANTIC_ID dense/Bf16MulScalar@v1)
# ========================================================================================================================
#: `aten.mul.Tensor(t [M,N] bf16, c)` — a bf16 tensor times a Python scalar (Gemma-2 `LogitsProcessor`: `logits * self.soft_cap`
#: after the tanh).  ATen's CUDA `mul_kernel` for bf16 computes `f32(x) * f32(c)` in opmath float and converts once with
#: `cvt.rn.bf16.f32` (NaN -> 0x7FFF).  wave-1 probe `mul30`: `aten.mul(bf16, 30.0) == bf16(f32(x) * 30)` on all 65,282 non-NaN
#: words.

@composite("Bf16MulScalar", 1, ["N", "C"], lambda S: ((("x", Array(S.N, BF16)),), Array(S.N, BF16)),
           conformance="quarantine (op-level evidence only): difftest vs aten.mul(x, C) on the pod")
def Bf16MulScalar(B, S, x, c=None):
    """`out_j = cvt.rn.bf16.f32(f32(x_j) * f32(C))`; `c`: `Bf16MulScalar_v3`'s input (the f32 word `f32(C)`)."""
    xf = B.batch(P.Bf16ToF32, x, axes=(0,))
    c = B.call(f32c(float(S.C))) if c is None else c
    return B.batch(P.F2fpBf16, B.batch(P.F32Mul, xf, c, axes=(0, None)), axes=(0,))


# ========================================================================================================================
# gelu_tanh_mul_bf16  (SEMANTIC_ID dense/GeluTanhMulBf16@v1)
# ========================================================================================================================
#: `GeluTanhMulBf16` — one output word of vLLM `_C.gelu_tanh_and_mul` (csrc/activation_kernels.cu, `act_and_mul_kernel<bf16,
#: gelu_tanh_kernel, true>`), the GeGLU of Gemma-2 (`GeluAndMul(approximate="tanh")`):
#: 
#:     const float f = (float)x;                                     // bf16 -> f32, exact
#:     constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;           // f32(sqrt(2/pi)) = 0x3F4C422A
#:     constexpr float KAPPA = 0.044715;                              // 0x3D372713
#:     float x_cube = f * f * f;                                      // (f*f)*f, two FMULs
#:     float inner = BETA * (f + KAPPA * x_cube);                     // FFMA(KAPPA, x_cube, f) (nvcc -fmad), then FMUL
#:     return (T)(0.5f * f * (1.0f + ::tanhf(inner)));               // (0.5f*f) * (1 + t), then cvt.rn.bf16.f32
#:     out = ACT_FN(x) * (y + 0.0f)                                   // f32(a) * (f32(y) + 0.0f), cvt.rn.bf16.f32
#: 
#: The pinned build (vLLM d9105ea80, `_C_stable_libtorch.abi3.so`, sm_89) compiles this as `masked_act_and_mul_kernel<
#: c10::BFloat16, gelu_tanh_kernel<c10::BFloat16>, true, false, false>(out, in, mask, d, n, float, float, float)` whose SASS is
#: (cuobjdump on the pod, `gelu_tanh_and_mul.sm89.sass`):
#: 
#:     PRMT R3, RZ, 0x5410, R0        ; f = (float)x            FMUL R0, R3, R3 ; FMUL R0, R3, R0      ; x_cube = (f*f)*f
#:     FFMA R0, R0, 0.044714998, R3   ; f + KAPPA*x_cube        FMUL R3, R3, 0.5                       ; 0.5f*f
#:     FADD R16, R16, c[0x0][0x188]   ; y + <float arg> (= +0.0f for gelu_tanh_and_mul: turns y = -0 into +0, identity otherwise)
#:     FMUL R13, R0, 0.79788458       ; inner = BETA*(...)      MUFU.EX2 / MUFU.RCP / FFMA...          ; libdevice tanhf(inner)
#:     FADD R18, R18, 1 ; FMUL R3, R3, R18 ; F2FP.BF16.F32.PACK_AB R3, RZ, R3 ; PRMT R3, RZ, 0x5410, R3   ; act = bf16(...) -> f32
#:     FMUL R3, R16, R3 ; F2FP.BF16.F32.PACK_AB R0, RZ, R3 ; STG.E.U16                                   ; bf16((y+0)*act)
#: 
#: The `y + 0.0f` was found by the difftest's `zeros` edge kind (every `y = -0` word came back `+0`; 16/20 -> 20/20 cases).
#: 
#: Evidence (wave-1 exhaustive table `gelu_tanh_bf16_table.npy`, T[x] for all 65,536 bf16 words, and the factorisation check
#: `out == bf16(f32(T[x]) * f32(y))` on 2^20 random pairs): the body above reproduces T on every non-NaN word (65,282/65,282) and
#: gives the kernel's 0x7FFF for NaN, including x = -inf where `(0.5f * -inf) * 0` is NaN; the alternative association
#: `0.5f * (f * (1 + t))` is wrong on 129 words (overflow at |f| > 1.7e38), and FFMA vs separate FMUL+FADD for `inner` is not
#: distinguishable on the domain.  `tanhf` is the `TanhF32Rn` stand-in (see `tanh_f32.py`).

BETA_BITS = 0x3F4C422A       # f32(M_SQRT2 * M_2_SQRTPI * 0.5)
KAPPA_BITS = 0x3D372713      # f32(0.044715)
HALF_BITS = 0x3F000000
ONE_BITS = 0x3F800000


def gelu_tanh_bf16_bits(g: int) -> int:
    """`(T)gelu_tanh_kernel((float)g)`: the activation word before the multiply."""
    f = P.bf16_to_f32_bits(g)
    x2 = P.f32_mul_bits(f, f)
    x3 = P.f32_mul_bits(x2, f)
    inner = P.f32_mul_bits(BETA_BITS, P.f32_fma_bits(KAPPA_BITS, x3, f))
    t = tanh_f32_bits(inner)
    act = P.f32_mul_bits(P.f32_mul_bits(HALF_BITS, f), P.f32_add_bits(ONE_BITS, t))
    return P.f32_to_bf16_rn_bits(act, P.BF16_NAN_F2FP)


ZERO_BITS = 0x00000000


@primitive("GeluTanhMulBf16", 1, [("g", BF16), ("u", BF16)], BF16,
           conformance="quarantine (op-level evidence only): T[x] exact on 65,282/65,282 non-NaN bf16 words of wave-1's GPU table; "
                       "difftest vs torch.ops._C.gelu_tanh_and_mul on the pod (20/20 incl. zeros/max/subnormal/alternating)",
           tables=(("gelu_tanh_bf16", "registered"),))
def GeluTanhMulBf16(g: int, u: int) -> int:
    """`bf16(f32(bf16(gelu_tanh(f32(g)))) * (f32(u) + 0.0f))`, both conversions `cvt.rn.bf16.f32` (NaN -> 0x7FFF)."""
    a = gelu_tanh_bf16_bits(g)
    uf = P.f32_add_bits(P.bf16_to_f32_bits(u), ZERO_BITS)          # the kernel's FADD y, +0.0f: -0 -> +0, else identity
    return P.f32_to_bf16_rn_bits(P.f32_mul_bits(P.bf16_to_f32_bits(a), uf), P.BF16_NAN_F2FP)


# ========================================================================================================================
# gelu_tanh_mul  (SEMANTIC_ID dense/GeluTanhMul@v1)
# ========================================================================================================================
#: `_C.gelu_tanh_and_mul(out [M,I], gu [M,2I])` — vLLM's GeGLU (`GeluAndMul(approximate="tanh")`, Gemma-2 MLP): `out[i] =
#: bf16(bf16(gelu_tanh(gate[i])) * up[i])`, gate = gu[:I], up = gu[I:] (`act_and_mul_kernel<..., act_first=true>`), the same
#: shape as the registered `SiluMul_v1` with the element `GeluTanhMulBf16` (see `gelu_tanh_mul_bf16.py`).

@composite("GeluTanhMul", 1, ["I"], lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)),
           conformance="quarantine (op-level evidence only): difftest vs torch.ops._C.gelu_tanh_and_mul on the pod")
def GeluTanhMul(B, S, gu):
    """`out[i] = GeluTanhMulBf16(gu[i], gu[I + i])`."""
    return B.batch(GeluTanhMulBf16, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


# ========================================================================================================================
# softcap  (SEMANTIC_ID dense/AttentionSoftcap@v1)
# ========================================================================================================================
#: `AttentionSoftcap_v1{T, NH, KVH, D, BN, CAP}` — FA2 attention WITH logit soft-capping (`flash_fwd_splitkv_kernel<..., Is_softcap=true>`,
#: vLLM + Gemma-2 `attn_logit_softcapping = 50`), STRUCTURE ONLY: the body of `Attention_v3` ->
#: `AttentionHead_v3` -> `AttnBlock_v3` with the one stage FA2 adds, in the place FA2 adds it.
#: 
#: FA2 (`flash_api.cpp::set_params_fprop`, `flash_fwd_kernel.h::compute_attn_1rowblock`, `softmax.h::apply_softcap`):
#: 
#:     params.softcap           = softmax_scale / softcap            (float / float -> RN32)
#:     params.scale_softmax_log2 = softcap * M_LOG2E                  (float * double -> double, narrowed to float)
#:     acc_s = Q K^T                                                  (mma, f32 accumulators)          -- stage A  (unchanged)
#:     acc_s(i) = tanh.approx.f32(acc_s(i) * params.softcap)          (`cutlass::fast_tanh`, one MUFU.TANH) -- stage A'  (NEW)
#:     mask(acc_s); softmax_rescale_o(acc_s, acc_o, params.scale_softmax_log2)                         -- stages B1..B5, C (unchanged)
#: 
#: so relative to `AttnBlock_v3` (b1.py): every score `s[c] = DotBf16{K=D}(q, k[c])` becomes `s'[c] = MufuTanh(F32MulFtz(s[c], SOFTCAP_MUL))`
#: BEFORE the row max (B1), the rescale factor (B2) and the exp2 stage (B3), and the exp2 stage's constant is `CAP * log2e` instead of
#: `D^-0.5 * log2e`; everything else (lane sums, P -> bf16, O accumulation, epilogue `Fa2InvSum`) is word for word `AttnBlock_v3`.
#: The mask is applied after the softcap in the kernel, but masked columns contribute P = 0 exactly and are skipped here as in v3.
#: 
#: Conformance: NOT established.  `MufuTanh` (prims.py) is a quarantine primitive whose table is a measured-shard pilot; until the full
#: sm_89 table lands, evaluating this composite raises `MufuTanhUnmeasured` on any score word outside the measured shards.  The tests in
#: `softcap_tests.py` check the STRUCTURE (gate population = v3 + NVIS x {MufuTanh, F32MulFtz} + the two constants, the constants'
#: bit patterns, static wiring); `softcap_difftest.py` is the adapter for the pod difftest vs `_vllm_fa2_C.varlen_fwd(softcap=CAP)`.
#: 

M_LOG2E = 1.4426950408889634  # <math.h> M_LOG2E (double), as FA2 uses it


def softcap_mul_bits(D: int, CAP: float) -> int:
    """`params.softcap = softmax_scale / softcap`: both `float`, one RN32 division.  `softmax_scale` reaches FA2 as `float(D ** -0.5)`
    (vLLM passes the Python double, the binding narrows it) — the same narrowing `b1.scale_log2_bits` uses."""
    return P.bits_of(np.float32(np.float32(D ** -0.5) / np.float32(CAP)))


def softcap_scale_log2_bits(CAP: float) -> int:
    """`params.scale_softmax_log2 = softcap * M_LOG2E`: float promoted to double, times the double constant, narrowed to float."""
    return P.bits_of(np.float32(float(np.float32(CAP)) * M_LOG2E))


@composite("AttnBlockSoftcap", 1, ["D", "NVIS", "FIRST", "BN", "CAP"],
           lambda S: (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                        ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))),
                      Tuple(F32, Array(4, F32), Array(S.D, F32))),
           conformance="quarantine (op-level evidence only): difftest vs vLLM FA2 flash_attn_varlen_func(softcap=CAP) 25/25 cases bitwise (43,520 bf16 words; T 5/17/64/65/130, D 128/256, CAP 30/50, all edge kinds; RTX 4090 sm_89 driver 595.91.07, vllm 0.28.1rc1.dev472); MufuTanh from the in-tree measured table; not yet reached by the capture patterns (AttentionScope softcap branch)")
def AttnBlockSoftcap(B, S, q, k, v, m_prev, l_prev, o_prev, softcap_mul=None, scale_log2=None):
    """`AttnBlock_v3`'s body with FA2's softcap stage A' in front of the online softmax: `s'[c] = MufuTanh(F32MulFtz(s[c], SOFTCAP_MUL))`,
    exp2 constant `CAP * log2e`.  One BN-key block of one query row; `NVIS` visible columns; `FIRST` = kernel `Is_first`.
    `softcap_mul` / `scale_log2`: `AttnBlockSoftcap_v4`'s inputs (no `CAP`)."""
    D, NVIS, FIRST, BN, CAP = S.D, S.NVIS, S.FIRST, S.BN, getattr(S, "CAP", None)
    if BN % 16 or not 1 <= NVIS <= BN:
        raise ValueError(f"AttnBlockSoftcap_v1: BN={BN} must be a multiple of 16 and 1 <= NVIS={NVIS} <= BN")
    if softcap_mul is None and not (isinstance(CAP, (int, float)) and CAP > 0):
        raise ValueError(f"AttnBlockSoftcap_v1: CAP={CAP!r} must be a positive softcap (FA2 takes the Is_softcap path iff softcap > 0)")
    dot = bind(DotBf16, K=D)
    softcap_mul = B.call(const(32, softcap_mul_bits(D, CAP))) if softcap_mul is None else softcap_mul
    # A: S = Q K^T;  A' (NEW): S' = tanh.approx(S * softmax_scale / CAP), one MUFU.TANH per score, BEFORE the row max
    s = [B.call(P.MufuTanh, B.call(P.F32MulFtz, B.call(dot, q, k[c]), softcap_mul)) for c in range(NVIS)]
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = B.call(P.F32Max, bmax, s[c])  # B1: row max (of the capped scores)
    m_new = bmax if FIRST else B.call(P.F32Max, m_prev, bmax)
    m_use = B.call(P.GuardNegInfZero, m_new)
    if scale_log2 is None:
        scale_log2 = B.call(const(32, softcap_scale_log2_bits(CAP)))  # CAP * log2e, not D^-0.5 * log2e
    if not FIRST:  # B2: rescale factor
        scores_scale = B.call(P.MufuEx2Ftz, B.call(P.F32MulFtz, B.call(P.F32SubFtz, m_prev, m_use), scale_log2))
    max_scaled = B.call(P.F32MulFtz, m_use, scale_log2)
    p = [B.call(P.MufuEx2Ftz, B.call(P.F32FmaSubFtz, s[c], scale_log2, max_scaled)) for c in range(NVIS)]  # B3
    # B4: per-lane sequential sums; the block's first add is contracted with the rescale  (= AttnBlock_v3)
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
    # B5 + C: P -> bf16, O = (O * scale) + P V in BN/16 ascending k-steps of 16 keys  (= AttnBlock_v3)
    pb = [B.call(P.F2fpBf16, x) for x in p]
    zero16 = B.call(ZERO16) if NVIS < BN else None
    o = [B.call(ZERO32) for _ in range(D)] if FIRST else [B.call(P.F32MulFtz, o_prev[d], scores_scale) for d in range(D)]
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break  # all-masked k-step: P = 0 exactly, no accumulator change
        prow = _Coll(A16, concat([(pb[k0 + j] if k0 + j < NVIS else zero16).refs for j in range(16)]))
        vblk = v[k0:k0 + 16]
        vt_refs = strided_view(vblk, 0, 16, 1, D, 16 * D) if isinstance(vblk.refs, Affine) else \
            Explicit(tuple(vblk.refs[j * D + d] for d in range(D) for j in range(16)))
        vt = _Coll(Array(D, A16), vt_refs)
        o = list(B.batch(P.AmpereBF16TcDot16, array_of(o), prow, vt, axes=(0, None, 0)))
    return tuple_of(m_new, array_of(lanes), array_of(o))


@composite("AttentionHeadSoftcap", 1, ["T", "D", "BN", "CAP"],
           lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))),
                      Array(S.D, BF16)),
           conformance="quarantine (op-level evidence only): difftest vs vLLM FA2 flash_attn_varlen_func(softcap=CAP) 25/25 cases bitwise (43,520 bf16 words; T 5/17/64/65/130, D 128/256, CAP 30/50, all edge kinds; RTX 4090 sm_89 driver 595.91.07, vllm 0.28.1rc1.dev472); MufuTanh from the in-tree measured table; not yet reached by the capture patterns (AttentionScope softcap branch)")
def AttentionHeadSoftcap(B, S, q, k, v, *extra, block=AttnBlockSoftcap):
    """`AttentionHead_v3`'s body over `AttnBlockSoftcap_v1{..., BN, CAP}`: causal attention of one query at position T-1 over keys
    0..T-1, ceil(T / BN) blocks visited last -> first; epilogue `Fa2InvSum` + `bf16(o * inv)` unchanged.  `extra` / `block`:
    `AttentionHeadSoftcap_v4`'s."""
    T, D, BN, CAP = S.T, S.D, S.BN, getattr(S, "CAP", None)
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
        blk = bind(block, D=D, NVIS=nvis, FIRST=first, BN=BN, **({} if extra else {"CAP": CAP}))
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


@composite("AttentionSoftcap", 1, ["T", "NH", "KVH", "D", "BN", "CAP"],
           lambda S: ((("q", Array(S.NH, Array(S.D, BF16))),
                       ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                       ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))),
                      Array(S.NH, Array(S.D, BF16))),
           conformance="quarantine (op-level evidence only): difftest vs vLLM FA2 flash_attn_varlen_func(softcap=CAP) 25/25 cases bitwise (43,520 bf16 words; T 5/17/64/65/130, D 128/256, CAP 30/50, all edge kinds; RTX 4090 sm_89 driver 595.91.07, vllm 0.28.1rc1.dev472); MufuTanh from the in-tree measured table; not yet reached by the capture patterns (AttentionScope softcap branch)")
def AttentionSoftcap(B, S, q, kc, vc, *extra, head=AttentionHeadSoftcap):
    """`Attention_v3`'s body over `AttentionHeadSoftcap_v1{T, D, BN, CAP}` (GQA: head h reads KV head h // (NH/KVH); caches `[T][KVH][D]`).
    Gemma-2-2B: NH = 8, KVH = 4, D = 256, CAP = 50.0, BN = `fa2_kblock_n(256)`; sliding-window layers need nothing more below 4096
    tokens (FA2 `Is_local = false` there)."""
    T, NH, KVH, D, BN = S.T, S.NH, S.KVH, S.D, S.BN
    cap = {} if extra else {"CAP": S.CAP}
    G = NH // KVH
    outs = []
    for g in range(KVH):
        kg = _Coll(Array(T, Array(D, BF16)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = _Coll(Array(T, Array(D, BF16)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(bind(head, T=T, D=D, BN=BN, **cap), q[g * G:(g + 1) * G], kg, vg, *extra, axes=(0, None, None) + (None,) * len(extra)))
    return _Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


ALL_DEFINITIONS = [SquareBf16, SquareF32, MeanTriton, AddScalarF32, AddScalarBf16, RsqrtF32, ScaleRowBf16, ScaleRowF32, MulVecF32,
                   NarrowF32ToBf16, AddWidenedBf16, Bf16DivScalar, Bf16Tanh, Bf16MulScalar, GeluTanhMul,
                   AttnBlockSoftcap, AttentionHeadSoftcap, AttentionSoftcap]
ALL_PRIMITIVES = [TanhF32Rn, GeluTanhMulBf16]

__all__ = ["SquareBf16", "SquareF32", "MeanTriton", "AddScalarF32", "AddScalarBf16", "AddWidenedBf16", "RsqrtF32", "ScaleRowBf16",
           "ScaleRowF32", "MulVecF32", "NarrowF32ToBf16", "Bf16DivScalar", "TanhF32Rn", "tanh_f32_bits", "Bf16Tanh", "Bf16MulScalar",
           "GeluTanhMulBf16", "gelu_tanh_bf16_bits", "GeluTanhMul", "AttnBlockSoftcap", "AttentionHeadSoftcap", "AttentionSoftcap",
           "softcap_mul_bits", "softcap_scale_log2_bits", "reciprocal_bits", "size_per_thread", "thread_elements", "M_LOG2E",
           "BLOCK_SIZE", "NUM_WARPS", "THREADS", "SIZE_PER_THREAD", "TILE", "REPS", "BETA_BITS", "KAPPA_BITS", "HALF_BITS", "ONE_BITS",
           "ZERO_BITS", "SEMANTIC_IDS", "ALL_DEFINITIONS", "ALL_PRIMITIVES", "BF16", "F32"]

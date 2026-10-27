"""verity_vllm.program.quarantine.dense — Gemma-2-2B's provisional exact
definitions for the operations vLLM d9105ea80 runs UNFUSED for `Gemma2ForCausalLM` under `VLLM_BATCH_INVARIANT=1` (eager,
sm_89), one module per definition (one `SEMANTIC_ID` each), a `*_difftest.py` adapter next to each.

What Gemma-2 adds over the Llama/Qwen profile:

  GemmaRMSNorm = `ir.ops.rms_norm` / `ir.ops.fused_add_rms_norm` (vllm/ir/ops/layernorm.py) as ATen ops on an f32 copy:
      x32 = x.to(f32)                     -> Transparent noop (exact widening; consumers re-point to the bf16 source)
      [s32 = x32 + res32; res' = s32.to(bf16)]   AddWidenedBf16, NarrowF32ToBf16        (fused_add variant)
      v = pow(x32 | s32, 2)                       SquareBf16 | SquareF32
      m = mean(v, -1, keepdim)                    MeanTriton   (batch-invariant Triton `mean_kernel`, BLOCK_SIZE=1024, 4 warps)
      r = rsqrt(m + eps)                          AddScalarF32{C=eps}, RsqrtF32
      y = (x32 | s32) * r                         ScaleRowBf16 | ScaleRowF32
      w1 = w.to(f32) + 1.0                        AddScalarBf16{C=1.0}   (`GemmaRMSNorm.forward_native`: weight.float() + 1)
      out = (y * w1).to(bf16)                     MulVecF32, NarrowF32ToBf16
  GeGLU `_C.gelu_tanh_and_mul`                    GeluTanhMulBf16 (primitive), GeluTanhMul (composite)
  embedding scale `embed * model.normalizer`      Bf16MulScalarTensor   (the 0-d bf16 buffer is a root of class "unknown": owner decision)
  final softcap `30 * tanh(logits / 30)`          Bf16DivScalar{C=30}, TanhF32Rn (primitive), Bf16Tanh, Bf16MulScalar{C=30}

NOT here (mechanism gap): the FA2 attention body with `Is_softcap=true` (`tanh.approx.f32` = MUFU.TANH on the scores) — no
registered hardware primitive models MUFU.TANH; the base `AttentionScope` keeps reporting it as `Unsupported: softcap is not
modelled`.

Conventions (checked against exhaustive GPU tables): every
ATen CUDA kernel here converts f32 -> bf16 with the hardware `cvt.rn.bf16.f32` (c10::BFloat16(float) on sm_80+ is
`__float2bfloat16`), so NaN words are 0x7FFF (`F2fpBf16`), not the software 0x7FC0.
"""
from __future__ import annotations

from verity_vllm.program.quarantine.dense import (add_scalar_bf16, add_scalar_f32, add_widened_bf16, bf16_div_scalar, bf16_mul_scalar, bf16_mul_scalar_tensor,
               bf16_tanh, gelu_tanh_mul, gelu_tanh_mul_bf16, mean_triton, mul_vec_f32, narrow_f32_bf16, rsqrt_f32,
               scale_row_bf16, scale_row_f32, square_bf16, square_f32, tanh_f32)

ALL_DEFINITIONS = [
    square_bf16.SquareBf16, square_f32.SquareF32, mean_triton.MeanTriton, add_scalar_f32.AddScalarF32, add_scalar_bf16.AddScalarBf16,
    rsqrt_f32.RsqrtF32, scale_row_bf16.ScaleRowBf16, scale_row_f32.ScaleRowF32, mul_vec_f32.MulVecF32,
    narrow_f32_bf16.NarrowF32ToBf16, add_widened_bf16.AddWidenedBf16, bf16_mul_scalar_tensor.Bf16MulScalarTensor,
    bf16_div_scalar.Bf16DivScalar, bf16_tanh.Bf16Tanh, bf16_mul_scalar.Bf16MulScalar, gelu_tanh_mul.GeluTanhMul,
]
ALL_PRIMITIVES = [tanh_f32.TanhF32Rn, gelu_tanh_mul_bf16.GeluTanhMulBf16]

"""`aten.mul.Tensor(e [M,N] bf16, s [] bf16)` — a bf16 tensor times a 0-d bf16 *tensor* (Gemma2Model.forward:
`self.embed_tokens(input_ids) * self.normalizer`, the `sqrt(hidden_size)` buffer registered at model dtype).  ATen's CUDA
`mul_kernel` for bf16 computes in opmath float — `f32(e) * f32(s)` (exact: 16 significant bits) — and converts once with
`c10::BFloat16(float)` = `cvt.rn.bf16.f32` (NaN -> 0x7FFF).  wave-1 probe `normalizer`: `aten.mul(bf16 x, bf16 0-d) ==
bf16(f32(x) * 48)` on all 65,282 non-NaN bf16 words.

[R13 X-08] PROMOTED: the Definition now lives in the accepted registry (`verity_vllm.registry.b1.Bf16MulScalarTensor`, bound as
kind `scale_tensor` by the pinned `b1-eager-v3` vocabulary); this module re-exports the same object so the quarantine's
difftest adapter and `ALL_DEFINITIONS` are unchanged.

The 0-d operand is the `model.normalizer` buffer root — a config-derived constant with no checkpoint tensor; the fold binds
it as `Shared(view)` so the Program reads the buffer honestly, and `root_policy` reports its class as "unknown" (an owner-level
root-class decision, REPORT §GEMMA2_2B)."""
from __future__ import annotations

from verity_vllm.registry.b1 import Bf16MulScalarTensor

SEMANTIC_ID = "dense/Bf16MulScalarTensor@v1"

__all__ = ["Bf16MulScalarTensor", "SEMANTIC_ID"]

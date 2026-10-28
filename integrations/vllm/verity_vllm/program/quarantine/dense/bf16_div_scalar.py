"""`aten.div.Tensor(l [M,N] bf16, c)` — a bf16 tensor divided by a Python scalar (Gemma-2 `LogitsProcessor`, `soft_cap=30`:
`logits / self.soft_cap`).  ATen's CUDA `div_true_kernel_cuda` takes the CPU-scalar fast path: `inv_b = opmath_t(1) / b`
computed once on the host in f32 (correctly rounded: `f32(1)/f32(30)` = 0x3D088889) and then `f32(x) * inv_b` per element
(`BUnaryFunctor<MulFunctor<float>>`), converted with `cvt.rn.bf16.f32` (NaN -> 0x7FFF).  On the bf16 domain this equals
`bf16(f32(x) / 30)` at every one of the 65,536 words (checked offline; probe `div30` compared the kernel with the
true-division form and found all 65,282 non-NaN words equal), so the two readings are indistinguishable here; the body
follows ATen's actual algorithm.

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.Bf16DivScalar`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import Bf16DivScalar, reciprocal_bits

SEMANTIC_ID = "dense/Bf16DivScalar@v1"

__all__ = ['Bf16DivScalar', 'reciprocal_bits', 'SEMANTIC_ID']

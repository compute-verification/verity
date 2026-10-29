"""`TanhF32Rn` — the stand-in for CUDA `tanhf` (libdevice `__nv_tanhf`, as ATen's `tanh_kernel_cuda` and vLLM's
`gelu_tanh_kernel` call it): RN32 of the exact real tanh(x), the same idiom as the registered `_silu_f32_bits` stand-in for
`expf` + division.  Total on encodings: NaN -> canonical NaN, ±inf -> ±1, ±0 -> ±0, |x| >= 10 -> ±1 exactly (tanh is within
2^-28 of 1 there), subnormal x -> x.

Evidence that the stand-in is exact where the quarantine uses it: both uses take a *bf16-valued* argument (`aten.tanh` on bf16;
`gelu_tanh_kernel`'s `inner` is a function of one bf16 word), so the composed bf16 -> bf16 functions have a finite domain of
65,536 words, and the exhaustive GPU tables (`tanh_bf16_table.npy`, `gelu_tanh_bf16_table.npy`) agree with the
stand-in-based bodies on every non-NaN word (65,282 / 65,282 and 65,282 / 65,282; NaN inputs give the kernels' 0x7FFF).  CUDA's
`tanhf` (<= 2 ulp) is therefore not distinguishable from RN32(tanh) through either bf16 output on the whole input domain.

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.TanhF32Rn`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import TanhF32Rn, tanh_f32_bits

SEMANTIC_ID = "dense/TanhF32Rn@v1"

__all__ = ['TanhF32Rn', 'tanh_f32_bits', 'SEMANTIC_ID']

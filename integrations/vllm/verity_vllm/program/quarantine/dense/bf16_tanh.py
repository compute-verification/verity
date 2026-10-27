"""`aten.tanh.default(t [M,N] bf16)` (Gemma-2 `LogitsProcessor`: `torch.tanh(logits / soft_cap)`).  ATen's CUDA `tanh_kernel_cuda`
computes `::tanh(float(x))` = libdevice `tanhf` in opmath float and converts once with `cvt.rn.bf16.f32` (NaN -> 0x7FFF).
The exhaustive GPU table (`tanh_bf16_table.npy`) equals `bf16(RN32(tanh(f32(x))))` on all 65,282 non-NaN words and
0x7FFF on the 254 NaN words — the finite domain makes the `TanhF32Rn` stand-in exact here.

PROMOTED: the Definition now lives in the accepted registry (`verity.ml.definitions.registry.dense.Bf16Tanh`, bound by the pinned `b1-eager-v3` vocabulary; the X-08 pattern) — this module re-exports
the same object so the quarantine's difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID` are unchanged (no registry clash).
"""
from __future__ import annotations

from verity.ml.definitions.registry.dense import Bf16Tanh

SEMANTIC_ID = "dense/Bf16Tanh@v1"

__all__ = ['Bf16Tanh', 'SEMANTIC_ID']

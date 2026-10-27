"""The residual add, `ops.rows.Bf16AddRow_v1{N}`: units_add, elementwise, and its twin.

OPS: y = bf16(RN(widen(a) + widen(b))): the words widened to f32 (exact), one add.rn.f32 with subnormals kept, then
cvt.rn.bf16.f32 (RNE, every NaN 0x7FFF): `fp32.Bf16AddF2fp_v1` per element.  The sweep (op 8) holds it to the twin on
every pair of bf16 words.
"""
from __future__ import annotations

import numpy as np

import words as W


def twin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return W.bf16(W.fadd(W.widen(a), W.widen(b)))


def gpu(a, b, *, out=None):
    """units_add on two bf16 CUDA tensors of one shape -> bf16 of that shape."""
    import torch

    import lib as L
    shape = tuple(a.shape)
    a = L.words(a, torch.bfloat16, shape, "add a")
    b = L.words(b, torch.bfloat16, shape, "add b")
    out = L.out_or_new(out, torch.bfloat16, shape, a, "add")
    L.aligned("add", a, b, out)
    n = a.numel()
    if n:
        L.lib().launch("units_add", min(-(-n // 2048), 1 << 16), 256, [L.ptr(a), L.ptr(b), L.ptr(out), L.u64(n)])
    return out

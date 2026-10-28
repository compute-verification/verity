"""ReLU on bf16 words, the MLP's activation: units_relu, elementwise, and its twin.  The frozen step folds it
into down_quant (quant.py with RELU on, the same relu16); this is the standalone unit.

OPS: y = +0 (0x0000) where 0 > x strictly (sign set, magnitude nonzero, not NaN), else x: -0 and every NaN are
themselves.  Integers only.
"""
from __future__ import annotations

import numpy as np

import words as W


def twin(x: np.ndarray) -> np.ndarray:
    return W.relu16(x, True)


def gpu(x, *, out=None):
    """units_relu on any contiguous bf16 CUDA tensor -> bf16 of its shape."""
    import torch

    import lib as L
    x = L.words(x, torch.bfloat16, tuple(x.shape), "relu x")
    out = L.out_or_new(out, torch.bfloat16, tuple(x.shape), x, "relu")
    L.aligned("relu", x, out)
    n = x.numel()
    if n:
        L.lib().launch("units_relu", min(-(-n // 2048), 1 << 16), 256, [L.ptr(x), L.ptr(out), L.u64(n)])
    return out

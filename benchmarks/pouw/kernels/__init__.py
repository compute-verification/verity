"""The PoUW honest GEMMs for the RTX 4090 (``pouw_gemm.cu``), JIT-built as a torch extension on first use.

NCP's kernel runs the s8 x s8 schedule, so its dumped checkpoints are the unbiased sums; `route_u_checkpoints` turns them
into ``ncp-v1``'s route-U values with the exact public shift (Lean `checkedU_eq`).  A native u8 x s8 kernel started at c₀
is the same values at the same issue rate."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


def route_u_checkpoints(ck: np.ndarray, y: np.ndarray, beta: int = 128, d: int = 16) -> np.ndarray:
    """``ck``: [m, n, T] unbiased checkpoints mod 2^32; ``y``: [n, 3k] (Y's columns).  Returns C^U = C − β·(suffix column
    sums of Y past each checkpoint), mod 2^32, as uint32."""
    steps = y.astype(np.int64).reshape(y.shape[0], -1, d).sum(axis=2)          # [n, T]
    suffix = np.cumsum(steps[:, ::-1], axis=1)[:, ::-1] - steps                 # Σ over the steps after t
    return ((ck.astype(np.int64) - beta * suffix[None, :, :]) & 0xFFFFFFFF).astype(np.uint32)


@lru_cache(maxsize=1)
def load():
    from torch.utils.cpp_extension import load as _load

    return _load(name="pouw_gemm", sources=[str(Path(__file__).with_name("pouw_gemm.cu"))],
                 extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89", "-lineinfo"], verbose=False)

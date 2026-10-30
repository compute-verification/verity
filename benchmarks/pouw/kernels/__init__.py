"""The PoUW honest GEMMs for the RTX 4090 (``pouw_gemm.cu``), JIT-built as a torch extension on first use.

The native NCP route-U path runs u8 x s8 from ``c0 = -128*colsum(Y)`` and dumps every depth-16 checked value.
The older signed path stays as a timing/control route; ``route_u_checkpoints`` is its CPU shift identity."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np


def route_u_checkpoints(ck: np.ndarray, y: np.ndarray, beta: int = 128, d: int = 16) -> np.ndarray:
    """``ck``: [m, n, T] unbiased checkpoints mod 2^32; ``y``: [n, 3k] (Y's columns).  Returns C^U = C − β·(suffix column
    sums of Y past each checkpoint), mod 2^32, as uint32."""
    steps = y.astype(np.int64).reshape(y.shape[0], -1, d).sum(axis=2)          # [n, T]
    suffix = np.cumsum(steps[:, ::-1], axis=1)[:, ::-1] - steps                 # Σ over the steps after t
    return ((ck.astype(np.int64) - beta * suffix[None, :, :]) & 0xFFFFFFFF).astype(np.uint32)


def ncp_route_u_tiles(ext, x_u8, y_s8, cfg: int = 2) -> list[tuple[bytes, list[list[int]]]]:
    """Native route-U kernel contract.

    ``x_u8`` is ``X + 128`` and ``y_s8`` is Y's signed bytes, each as contiguous torch uint8 rows.
    The kernel starts at ``c0 = -128*colsum(Y)`` and dumps every depth-16 checkpoint.  This wrapper returns one
    ``(digest, exact int32 tile)`` per 16x16 output tile, in row-major tile order.  Digest words are step-major and
    row-major within a step, exactly ``ncp-v1``'s ``digest`` framing.
    """
    if x_u8.ndim != 2 or y_s8.ndim != 2 or x_u8.shape[1] != y_s8.shape[1]:
        raise ValueError("route-U operands are two matrices with the same stacked depth")
    if x_u8.shape[0] % 16 or y_s8.shape[0] % 16:
        raise ValueError("route-U output tiles are 16x16")
    c, ck = ext.ncp_route_u(x_u8, y_s8, cfg)
    cn, kn = c.cpu().numpy(), ck.cpu().numpy().astype(np.uint32)
    out: list[tuple[bytes, list[list[int]]]] = []
    for i in range(0, cn.shape[0], 16):
        for j in range(0, cn.shape[1], 16):
            block = np.ascontiguousarray(kn[i:i + 16, j:j + 16].transpose(2, 0, 1)).astype("<u4")
            digest = hashlib.shake_256(b"verity/pouw/ncp/tile/v1\0" + block.tobytes()).digest(32)
            out.append((digest, cn[i:i + 16, j:j + 16].astype(np.int64).tolist()))
    return out


@lru_cache(maxsize=1)
def load():
    from torch.utils.cpp_extension import load as _load

    return _load(name="pouw_gemm", sources=[str(Path(__file__).with_name("pouw_gemm.cu"))],
                 extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89", "-lineinfo"], verbose=False)

"""PoUW's NCP operands on the host, for the executor (`verity_vllm.protocols.pouw`): the int7 row quantizer
(`quantize`), the scheme's noise rows, a weight's Y and c₀ (`Operands`, `stack_y`) and one unit's tiles (`ncp_tiles`)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from verity.core.protocols.pouw.protocol import Context as UnitContext
from verity.core.protocols.pouw.schemes import ncp

_INV63 = np.float32(1.0) / np.float32(63.0)
_M32 = 0xFFFFFFFF


def quantize(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`pouw.QUANTIZER` per row: scale = RN_f32(rowmax · RN_f32(1/63)) (1 for a zero row),
    q = clamp(RNE(f32(v / scale)), −63, 63). Returns int8 rows and their FP32 scales; non-finite input is refused."""
    v = np.asarray(v, dtype=np.float32)
    if not np.isfinite(v).all():
        raise ValueError("pouw: the quantizer refuses NaN and infinity")
    scale = (np.abs(v).max(axis=1) * _INV63).astype(np.float32)
    scale[scale == 0] = np.float32(1.0)
    return np.clip(np.rint(v / scale[:, None]), -63, 63).astype(np.int8), scale


def _noise(tag: str, count: int, k: int, parts) -> np.ndarray:
    """The scheme's noise rows (`ncp.noise_row`: one XOF byte's low six bits minus 31 per entry), as int64."""
    return np.stack([np.frombuffer(ncp.xof(tag, k, *parts(i)), dtype=np.uint8).astype(np.int64) % 64 - 31 for i in range(count)])


@dataclass(frozen=True)
class Operands:
    """One weight's operands for an epoch: its column scales, Y = [B + F₁ ; B + F₂ ; B + F₃] as n rows of 3k, and
    c₀ = −β·Σ_l Y[l][j], the route-U accumulator's initial value (Lean `initU`)."""

    scales: np.ndarray
    y: np.ndarray
    c0: np.ndarray


def stack_y(scheme: Any, salt: bytes, w: int, b: np.ndarray, noise: tuple, first: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Y (as int8 rows: every entry lies in [−128, 127], Lean `rangeU`) and c₀ for the int7 weight rows ``b`` (n x k, the
    weight's rows from ``first``), with F₁ from the epoch salt (per epoch, weight and row, as `ncp.stack` draws it),
    F₂ = −P·F₁ and F₃ = F₁ + F₂ (`ncp.assemble`)."""
    n, k = b.shape
    pi, sigma = np.asarray(noise[0]), np.asarray(noise[1], dtype=np.int64)
    f1 = _noise("verity/pouw/ncp/F1/v1", n, k, lambda j: (salt, w, first + j))
    f2 = np.empty_like(f1)
    f2[:, pi] = -(f1 * sigma)
    b = b.astype(np.int64)
    y = np.concatenate([b + f1, b + f2, b + f1 + f2], axis=1)
    return y.astype(np.int8), (-scheme.beta * y.sum(axis=1)).astype(np.float64)


def ncp_tiles(scheme: Any, ctx: UnitContext, noise: tuple, a: np.ndarray, ops: Operands,
              chunk: int = 256) -> dict[tuple[int, int], tuple[bytes, np.ndarray]]:
    """One unit's tiles as the scheme's Prover kernel contract gives them: (first row, first column) -> (digest, int32
    output tile), for int7 ``a`` (m x k) and the weight's `Operands`.  X + β is formed as `ncp.assemble` forms X, the
    accumulator starts at c₀ (Lean `mmWU β`), and a tile's digest covers every depth-16 checkpoint, step-major and
    row-major within a step (`NCP.digest`).  Float64 products are exact here: every partial sum is an integer under 2^53."""
    u = ctx.matmul
    (m, k), n = a.shape, ops.y.shape[0]
    pi, sigma = np.asarray(noise[0]), np.asarray(noise[1], dtype=np.int64)
    a = a.astype(np.int64)
    e1 = _noise("verity/pouw/ncp/E1/v1", m, k, lambda i: (ctx.salt, u.index, ctx.root_a, u.weight, i))
    e2 = e1[:, pi] * sigma
    x = np.concatenate([a + e1, a + e2, -(a + e1 + e2)], axis=1) + scheme.beta
    y, c0 = ops.y, ops.c0
    steps = 3 * k // ncp.D
    out: dict[tuple[int, int], tuple[bytes, np.ndarray]] = {}
    for r in range(0, m, 16):
        xs = x[r:r + 16].reshape(16, steps, ncp.D).transpose(1, 0, 2).astype(np.float64)       # [T, 16, d]
        for c in range(0, n, chunk):
            ys = y[c:c + chunk].reshape(-1, steps, ncp.D).transpose(1, 2, 0).astype(np.float64)  # [T, d, w]
            run = np.cumsum(np.matmul(xs, ys), axis=0) + c0[None, None, c:c + chunk]
            words = (run.astype(np.int64) & _M32).astype("<u4")                                  # [T, 16, w]
            for j in range(0, words.shape[2], 16):
                block = np.ascontiguousarray(words[:, :, j:j + 16])
                digest = hashlib.shake_256(b"verity/pouw/ncp/tile/v1\0" + block.tobytes()).digest(32)
                out[(r, c + j)] = (digest, block[-1].astype(np.int64) - ((block[-1].astype(np.int64) >> 31) << 32))
    return out

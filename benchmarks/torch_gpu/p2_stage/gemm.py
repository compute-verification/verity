"""The check of the exact W8A8 GEMM on a staged matrix against `ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}`. The GEMM is
`int_mm.linear` (`benchmarks/torch_gpu/s8_linear/int_mm.py`): the s32 sum by `torch._int_mm`
(cuBLASLt's s8 IMMA, exact for K < 2^17 whatever its k order), then the Definition's epilogue
`cvt.rn.bf16.f32((cvt.rn.f32.s32(acc) * sx) * sw)`, the token's scale first.

`twin` computes every output word again by another road and counts the words that differ:

- the sum as an f64 GEMM: each partial sum is an integer of magnitude below 2^31, exact in a double in any order;
- `cvt.rn.f32.s32` as the f64 -> f32 rounding of that exact integer;
- each f32 product exactly in f64 (24 x 24 bits) and rounded once to f32, the correctly rounded f32 product;
- `cvt.rn.bf16.f32` as integer round-to-nearest-even on the f32 bits, `(u + 0x7FFF + ((u >> 16) & 1)) >> 16`.

`samples` picks coordinates and their inputs, which `check.py` hands to the Definition's own evaluator
(`verity.ml.circuits.evaluation.evaluate_batch`, the plain-integer reference) on the CPU: that ties the twin's formula, and
the GPU's words, to the registered Definition.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def bf16_rn(f: Any) -> Any:
    """`cvt.rn.bf16.f32` of finite float32 `f` as int32 words, by integer round-to-nearest-even on its bits."""
    u = f.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).to(torch.int32)


def twin_words(x: Any, sx: Any, w: Any, sw: Any) -> Any:
    """The twin's [m, n] output words (int32) for one block: f64 sum, exact f64 products each rounded to f32, bf16_rn."""
    acc = x.to(torch.float64) @ w.to(torch.float64).t()
    f = acc.to(torch.float32)
    f = (f.to(torch.float64) * sx.to(torch.float64)[:, None]).to(torch.float32)
    f = (f.to(torch.float64) * sw.to(torch.float64)[None, :]).to(torch.float32)
    return bf16_rn(f)


def twin(x: Any, sx: Any, w: Any, sw: Any, y: Any, block: int = 4096, keep: int = 8) -> dict:
    """Every word of `y` ([M, N] bfloat16) against the twin: the count that differ and the first `keep` of them."""
    if not (torch.isfinite(sx).all() and torch.isfinite(sw).all()):
        raise ValueError("the twin takes finite scales")
    M, N = y.shape
    bad, first = 0, []
    for n0 in range(0, N, block):
        n1 = min(N, n0 + block)
        wb = w[n0:n1]
        for m0 in range(0, M, block):
            m1 = min(M, m0 + block)
            t = twin_words(x[m0:m1], sx[m0:m1], wb, sw[n0:n1])
            g = y[m0:m1, n0:n1].view(torch.int16).to(torch.int32) & 0xFFFF
            ne = g != t
            k = int(ne.sum())
            if k:
                bad += k
                for i, j in ne.nonzero()[:keep - len(first)].tolist():
                    first.append({"m": m0 + i, "n": n0 + j, "got": int(g[i, j]), "twin": int(t[i, j])})
    return {"words": M * N, "mismatches": bad, "first": first}


def samples(x: Any, sx: Any, w: Any, sw: Any, y: Any, count: int, gen: torch.Generator) -> dict[str, np.ndarray]:
    """`count` coordinates (the corners first, then uniform) with their inputs and `y`'s word, as numpy arrays."""
    M, N = y.shape
    corners = [(0, 0), (M - 1, N - 1), (0, N - 1), (M - 1, 0)][:count]
    rest = count - len(corners)
    ms = torch.randint(0, M, (rest,), generator=gen).tolist()
    ns = torch.randint(0, N, (rest,), generator=gen).tolist()
    mm = torch.tensor([c[0] for c in corners] + ms)
    nn = torch.tensor([c[1] for c in corners] + ns)
    dev = y.device
    md, nd = mm.to(dev), nn.to(dev)
    return {"m": mm.numpy(), "n": nn.numpy(),
            "x": x[md].cpu().numpy(), "sx": sx[md].cpu().numpy(),
            "w": w[nd].cpu().numpy(), "sw": sw[nd].cpu().numpy(),
            "y": (y[md, nd].view(torch.int16).cpu().numpy().view(np.uint16))}


def definition_words(K: int, x: np.ndarray, sx: np.ndarray, w: np.ndarray, sw: np.ndarray) -> np.ndarray:
    """`ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}` on each sample by the Definition's plain-integer reference (CPU)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import batch as E
    from verity.ml.circuits.ops import gemm as G

    target = bind(G.ScaledMmS8Coordinate, K=K, **G.GEMM_SM90_S8)
    out = E.evaluate_batch(target, np.ascontiguousarray(x).view(np.uint8), np.ascontiguousarray(sx).view(np.uint32),
                           np.ascontiguousarray(w).view(np.uint8), np.ascontiguousarray(sw).view(np.uint32),
                           kernel="reference")
    return np.asarray(out).astype(np.uint16).reshape(-1)

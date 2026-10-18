"""The exact-int8 linears (`TargetProfile.int8_exact`): the plain `int8` step of the perplexity bench, as torch ops.

The plain step is `benchmarks/ppl/steps.py`'s `int8_linear` with its defaults (act = weight = "round", act_top = weight_top
= 127, one scale per token and per output row, no group, no GPTQ), and steps.py run on the CPU is the oracle the tests
compare with.  For one call's activations x [M, K] (bf16, widened exactly to float32) and a weight W [N, K] (int8, |W| <= 127)
with per-output-row float32 scales s_w [N] (`engine.int8_checkpoint`, steps.py's `quant_rows(w, 127)`):

    a[i]   = max_k |x[i, k]|                                   exact
    s_a[i] = RN32(max(a[i], RN32(1e-30)) / 127)                steps.py's zero-row guard, then one division
    X[i,k] = clamp(round(RN32(x[i, k] / s_a[i])), -127, 127)   round half to even; a division, as steps.py, not x * (1 / s)
    Y      = X W^T in int32, exact                             |Y| <= 127 * 127 * K < 2^31 for K <= MAX_K
    y[i,n] = RN_bf16(RN32(RN32(RN32(Y[i,n]) * s_a[i]) * s_w[n]))   in that order: the activation scale first

RN32 is IEEE float32 round to nearest even, one rounding per operation; RN_bf16 is float32 to bfloat16, nearest even.  Every
row's value reads only that row, so a row's output does not depend on the rest of its batch.  The division by 127 is
tensor / tensor on x's device: on CUDA torch turns `t / 127` into a multiplication by RN32(1 / 127), which rounds twice (steps.py
run on a GPU does that; its CPU run, the oracle, divides).

The product is `torch._int_mm` (cuBLASLt's s8 GEMM with s32 accumulation on CUDA), which on CUDA needs M > 16 and K, N
multiples of 8 (`Int8ExactConfig` refuses a layer whose K or N is not).  An M < MIN_ROWS call gets zero rows appended to X up
to MIN_ROWS and keeps the product's first M rows: row i of an integer product reads only row i of X, so padding changes no
row's value.  CPU and CUDA run the same padding.
"""
from __future__ import annotations

from typing import Any

import torch

from verity_vllm.program.frontend.kernel_readback import INT8_ARM

TOP = 127
SCALE_FLOOR = 1e-30
W_MAX = TOP
MIN_ROWS = 17
MAX_K = (2 ** 31 - 1) // (TOP * W_MAX)
ARM = INT8_ARM


def row_scales(x: torch.Tensor, top: int = TOP) -> torch.Tensor:
    """s [M] float32 of x [M, K] float32: max |x[i]| floored at 1e-30, divided by `top` (a tensor division)."""
    t = torch.full((), float(top), dtype=torch.float32, device=x.device)
    return x.abs().amax(dim=1).clamp(min=SCALE_FLOOR) / t


def quantize_rows(x: torch.Tensor, top: int = TOP) -> tuple[torch.Tensor, torch.Tensor]:
    """(X [M, K] int8, s [M] float32) of x [M, K]: steps.py's `quant_rows(x.float(), top)`."""
    x = x.to(torch.float32)
    s = row_scales(x, top)
    return torch.round(x / s[:, None]).clamp_(-top, top).to(torch.int8), s


def int_mm(xq: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """X [M, K] int8 times W [N, K] int8 transposed, exact in int32, with X padded to MIN_ROWS rows (see the module doc)."""
    m = xq.shape[0]
    if m < MIN_ROWS:
        xq = torch.cat([xq, xq.new_zeros(MIN_ROWS - m, xq.shape[1])])
    return torch._int_mm(xq, w.t())[:m]


def dequantize(y: torch.Tensor, s_a: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    """RN_bf16(RN32(RN32(RN32(Y) * s_a[i]) * s_w[n]))."""
    return (y.to(torch.float32) * s_a[:, None] * s_w[None, :]).to(torch.bfloat16)


def plain_linear(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    """The plain step's linear y [..., N] (bf16) of x [..., K] (bf16): one activation scale per row."""
    lead, k = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, k)
    if x2.shape[0] == 0:
        return x.new_empty(*lead, w.shape[0])
    xq, s_a = quantize_rows(x2)
    return dequantize(int_mm(xq, w), s_a, s_w).reshape(*lead, w.shape[0])


def applied_record(model: Any) -> dict[str, dict[str, Any]]:
    """What every linear layer ran, by module name: its quant method's class, int8 arm (None for a method without one) and the
    calls it served (None for a method that does not count them).  Structural, as `kernel_readback.int8_exact_of`."""
    out = {}
    for name, m in model.named_modules():
        qm = getattr(m, "quant_method", None)
        if qm is not None and any(c.__name__ == "LinearMethodBase" for c in type(qm).__mro__):
            out[name] = {"method": type(qm).__name__, "arm": getattr(qm, "int8_arm", None), "calls": getattr(qm, "calls", None)}
    return out

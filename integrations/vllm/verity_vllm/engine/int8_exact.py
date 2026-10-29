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

**The z/y records** (`scaled_mm_zy`; `Accounting.Served.ValueChecked`): the call's product over
its T slots (`rows`, H1's static layout: slot i < M is row i under s_a[i], every later slot an empty row, zero codes under
scale 0) in `UNIT` = 32 x 32 units, one record each, unit (row block, column block) in row-major order (the compiled
Program's), each `ZY_BYTES` bytes: its exact s32 z_u = X_u W_u^T over the whole K, row-major, then y_u = dq(z_u, s_a, s_w)
in bf16, row-major, little-endian (`zy_records`).  An empty slot's z is 0 and its y dq(0, +0, s_w) (`h1zy.dequant`): +0,
so zero bytes, under a finite positive s_w, 0x8000 under a negative one and a NaN (0x7FFF on CUDA, 0x7FC0 or 0xFFFF from
torch's CPU cast) under an infinite or NaN one.  The z and y words are the record's two halves, and each unit's z is its own
cells.
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
#: H1's receipt unit (`pouw_h1.TILE`): a record's slot rows by weight rows
UNIT = (32, 32)
#: a unit's record: its z, s32, then its y, bf16
ZY_BYTES = 6 * UNIT[0] * UNIT[1]


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
    """RN_bf16(RN32(RN32(RN32(Y) * s_a[i]) * s_w[n])); s_w [N] or [N, 1]."""
    return (y.to(torch.float32) * s_a[:, None] * s_w.reshape(1, -1)).to(torch.bfloat16)


def plain_linear(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    """The plain step's linear y [..., N] (bf16) of x [..., K] (bf16): one activation scale per row."""
    lead, k = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, k)
    if x2.shape[0] == 0:
        return x.new_empty(*lead, w.shape[0])
    xq, s_a = quantize_rows(x2)
    return dequantize(int_mm(xq, w), s_a, s_w).reshape(*lead, w.shape[0])


def zy_units(m: int, n: int, rows: int) -> int:
    """The records of a call of m rows by n weight rows over `rows` slots; ValueError unless they are whole units and m fits."""
    ur, uc = UNIT
    if rows % ur or n % uc or rows <= 0:
        raise ValueError(f"T = {rows} slots by N = {n} are not whole {ur} x {uc} units")
    if m > rows:
        raise ValueError(f"{m} rows past the call's {rows} slots")
    return (rows // ur) * (n // uc)


def zy_records(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The records [units, ZY_BYTES] uint8 of z [T, N] int32 and y [T, N] bf16 over a call's slots (see the module doc)."""
    ur, uc = UNIT
    t, n = z.shape

    def cells(v: torch.Tensor) -> torch.Tensor:
        return v.reshape(t // ur, ur, n // uc, uc).transpose(1, 2).reshape(-1, ur * uc).contiguous().view(torch.uint8)
    return torch.cat([cells(z), cells(y.view(torch.int16))], dim=1)


def zy_split(zy: torch.Tensor, rows: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`zy_records` undone: (z [rows, n] int32, y [rows, n] bf16)."""
    ur, uc = UNIT
    half = 4 * ur * uc

    def grid(v: torch.Tensor) -> torch.Tensor:
        return v.reshape(rows // ur, n // uc, ur, uc).transpose(1, 2).reshape(rows, n)
    return (grid(zy[:, :half].contiguous().view(torch.int32)),
            grid(zy[:, half:].contiguous().view(torch.int16)).view(torch.bfloat16))


def scaled_mm_zy(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor,
                 rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(y [M, N] bf16, zy [units, ZY_BYTES] uint8): the product's y, `dequantize(int_mm(xq, w), s_a, s_w)`, and its records
    over `rows` slots, the empty ones zero codes under scale 0 (see the module doc)."""
    m, k = xq.shape
    zy_units(m, w.shape[0], rows)
    slots, scales = xq.new_zeros((rows, k)), s_a.new_zeros(rows)
    slots[:m], scales[:m] = xq, s_a.reshape(-1)
    z = int_mm(slots, w)
    y = dequantize(z, scales, s_w)
    return y[:m].clone(), zy_records(z, y)


def applied_record(model: Any) -> dict[str, dict[str, Any]]:
    """What every linear layer and fused-MoE layer ran, by module name: its quant method's class, int8 arm (None for a method
    without one) and the calls it served (None for a method that does not count them).  Structural, as
    `kernel_readback.int8_exact_of`."""
    out = {}
    for name, m in model.named_modules():
        qm = getattr(m, "quant_method", None)
        if qm is not None and any(c.__name__ in ("LinearMethodBase", "FusedMoEMethodBase") for c in type(qm).__mro__):
            out[name] = {"method": type(qm).__name__, "arm": getattr(qm, "int8_arm", None), "calls": getattr(qm, "calls", None)}
    return out

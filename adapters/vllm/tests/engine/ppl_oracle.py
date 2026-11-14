"""The plain int8 linear's oracle: the registered Definition's equations (`verity/definitions/registry/int8_exact.py`,
`Int8RowQuant` then `ScaledMmS8`), in torch on the CPU, one rounding per operation.

    s    = RN32(max(max_k |x_k|, RN32(1e-30)) / 127)            per row of x and per output row of the weight
    X_k  = clamp(rint(RN32(x_k / s)), -127, 127)
    Y_n  = sum_k X_k W_nk                                      exact, in int64; k max|X| max|W| within s32
    y_n  = bf16(RN32(RN32(RN32(Y_n) s_a) s_w[n]))

`int8_linear(x, w)` is y [M, N] bf16 of x [M, K] bf16 and the bf16 weight w [N, K], both read as float32; it is the perplexity
bench's `[[linear]] step = "int8"` arm with its defaults.
"""
from __future__ import annotations

import torch

TOP = 127
S32 = (1 << 31) - 1
#: the scale floor, RN32(1e-30): a float32 clamp rounds the bound to float32
FLOOR = 1e-30


def quant_rows(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes [R, K] int8, scales [R] float32) of the rows of m, read as float32."""
    m = m.float()
    s = m.abs().amax(dim=1).clamp(min=FLOOR) / TOP
    return torch.round(m / s[:, None]).clamp(-TOP, TOP).to(torch.int8), s


def int_product(xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """Y [M, N] int64 = xq wq^T, exact; refused unless k max|xq| max|wq| is within s32."""
    bound = xq.shape[1] * int(xq.abs().max()) * int(wq.abs().max())
    assert bound <= S32, f"the s32 accumulator's bound {bound} exceeds {S32}"
    return xq.long() @ wq.long().T


def dequantize(y: torch.Tensor, sa: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """bf16(RN32(RN32(RN32(Y) s_a) s_w)) of the integer product y [M, N]."""
    return (y.float() * sa[:, None] * sw[None, :]).to(torch.bfloat16)


def int8_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y [M, N] bf16 of x [M, K] bf16 and the bf16 weight w [N, K]."""
    xq, sa = quant_rows(x.reshape(-1, x.shape[-1]))
    wq, sw = quant_rows(w)
    return dequantize(int_product(xq, wq), sa, sw)


def int8_linear_scaled(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The dequantization (bf16) of a given integer product y [M, N] (int64), on x's and w's scales."""
    return dequantize(y, quant_rows(x.reshape(-1, x.shape[-1]))[1], quant_rows(w)[1])


def int8_weights(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(int8 [N, K], scales [N]) the bf16 weight w quantizes to."""
    return quant_rows(w)

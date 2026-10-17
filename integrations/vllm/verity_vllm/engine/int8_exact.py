"""The exact-int8 linears (VLLM-18, `TargetProfile.int8_exact = "plain"`): the edited GEMM's plain arm, composed of torch ops.

The plain arm is `benchmarks/pouw/edited_gemm/egemm.cu`'s `plainq` (its reference `egemm_ref.py`): the activations quantized by
the plain quantizer under one per-call scalar `inv_s`, then an exact s8 x s8 -> s32 product with an int8 weight whose entries
are at most 63 in magnitude.  egemm.cu takes `inv_s` as an argument (1.0 in its gates) and leaves dequantization to its caller;
how `inv_s` is formed and the dequantization are this module's.  For one call's activations x [M, K] (bf16) and a weight W [N, K]
(int8, |W| <= 63) with per-output-channel float32 scales s_w [N] (`engine.int8_checkpoint`):

    amax  = max |x| over the call's M x K values             exact (bf16 abs and max, widened to float32 exactly)
    inv_s = RN32(127 / amax),  s_x = RN32(amax / 127)        both 1 when amax < 2^-120 (0 included: 127 / amax overflows)
    Xq    = clamp(rint(RN32(x * inv_s)), -127, 127)          rint: round half to even (egemm_ref.py's plain quantizer)
    Y     = Xq W^T in int32, exact                           |Y| <= 127 * 63 * K < 2^31 for K <= MAX_K
    c[n]  = RN32(s_x * s_w[n])
    y     = RN_bf16(RN32(RN32(Y) * c[n]))                    RN32(Y) rounds only when |Y| > 2^24

RN32 is IEEE float32 round to nearest even, one rounding per operation, and RN_bf16 the round to nearest even of float32 to
bfloat16.  Both divisions are tensor / tensor on x's device: on CUDA torch turns `t / float` into a multiplication by the
reciprocal, and `float / t` into `reciprocal(t) * float`, both two roundings.

The product is `torch._int_mm` (cuBLASLt's s8 GEMM with s32 accumulation on CUDA), which on CUDA needs M > 16 and K, N multiples
of 8 (`Int8ExactConfig` refuses a layer whose K or N is not).  An M < MIN_ROWS call gets zero rows appended to Xq up to MIN_ROWS
and keeps the product's first M rows.  Row i of an integer product reads only row i of Xq, and amax is taken over the M real rows
before any padding, so padding changes no other row's value.  CPU and CUDA run the same padding.
"""
from __future__ import annotations

from typing import Any

import torch

X_MAX = 127
AMAX_MIN = 2.0 ** -120
W_MAX = 63
MIN_ROWS = 17
MAX_K = (2 ** 31 - 1) // (X_MAX * W_MAX)
ARM = "plain"


def plain_scales(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(inv_s, s_x), float32 0-d tensors on x's device, of one call's activations."""
    amax = x.abs().amax().to(torch.float32)
    one = torch.ones((), dtype=torch.float32, device=x.device)
    q = torch.full((), float(X_MAX), dtype=torch.float32, device=x.device)
    zero = amax < AMAX_MIN
    a = torch.where(zero, one, amax)
    return torch.where(zero, one, q / a), torch.where(zero, one, a / q)


def quantize_plain(x: torch.Tensor, inv_s: torch.Tensor) -> torch.Tensor:
    """Xq = clamp(rint(RN32(x * inv_s)), -127, 127) as int8."""
    return torch.round(x.to(torch.float32) * inv_s).clamp_(-X_MAX, X_MAX).to(torch.int8)


def int_mm(xq: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Xq [M, K] int8 times W [N, K] int8 transposed, exact in int32, with Xq padded to MIN_ROWS rows (see the module doc)."""
    m = xq.shape[0]
    if m < MIN_ROWS:
        xq = torch.cat([xq, xq.new_zeros(MIN_ROWS - m, xq.shape[1])])
    return torch._int_mm(xq, w.t())[:m]


def dequantize(y: torch.Tensor, s_x: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    """RN_bf16(RN32(RN32(Y) * RN32(s_x * s_w[n])))."""
    return (y.to(torch.float32) * (s_x * s_w)).to(torch.bfloat16)


def plain_linear(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> torch.Tensor:
    """The plain arm's linear y [..., N] (bf16) of x [..., K] (bf16), one call: every leading row shares one activation scale."""
    lead, k = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, k)
    if x2.shape[0] == 0:
        return x.new_empty(*lead, w.shape[0])
    inv_s, s_x = plain_scales(x2)
    return dequantize(int_mm(quantize_plain(x2, inv_s), w), s_x, s_w).reshape(*lead, w.shape[0])


def applied_record(model: Any) -> dict[str, dict[str, Any]]:
    """What every linear layer ran, by module name: its quant method's class, int8 arm (None for a method without one) and the
    calls it served (None for a method that does not count them).  Structural, as `target_profile.int8_exact_arm_of`."""
    out = {}
    for name, m in model.named_modules():
        qm = getattr(m, "quant_method", None)
        if qm is not None and any(c.__name__ == "LinearMethodBase" for c in type(qm).__mro__):
            out[name] = {"method": type(qm).__name__, "arm": getattr(qm, "int8_arm", None), "calls": getattr(qm, "calls", None)}
    return out

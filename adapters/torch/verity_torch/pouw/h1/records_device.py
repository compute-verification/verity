"""`records.holds_on` on the records' own device, over torch tensors: what the PyTorch adapter's `tensors.H1Device.holds_on`
is."""
from __future__ import annotations

from typing import Any

from verity.protocols.resource_accounting.compute.pouw.prover.serving import int8_arm as ARM
from verity.protocols.resource_accounting.compute.pouw.prover.serving.h1.records import UNIT, ZY_BYTES


def holds_on(zy: Any, rows: int, n: int, y: Any, s_a: Any, s_w: Any) -> bool:
    """`records.holds_on` on torch tensors: whether `records.reason` finds nothing, read on the records' own device from the call's tensors (``scaled_mm_zy``'s zy, the
    served y [m, n] bf16, s_a [m] and s_w [n]) with one sync for the answer. dq there is the arm's `dequantize` (`int8_arm.Int8Arm`): an
    s32 to f32 conversion, two f32 products and a bf16 rounding, each to nearest, so its bits are the host's but for a NaN's.
    False, for `refusal` to read the host's copy, when they aren't a call's tensors on one device, s_w isn't finite, a dq is
    a NaN, or anything doesn't hold."""
    import torch

    ur, uc = UNIT
    if not all(torch.is_tensor(t) for t in (zy, y, s_a, s_w)):
        return False
    m, dev = s_a.numel(), zy.device
    if not (zy.dtype == torch.uint8 and y.dtype == torch.bfloat16 and rows > 0 and rows % ur == 0 and n > 0
            and n % uc == 0 and tuple(zy.shape) == ((rows // ur) * (n // uc), ZY_BYTES) and m <= rows
            and y.numel() == m * n and s_w.numel() == n and all(t.device == dev for t in (y, s_a, s_w))):
        return False

    def grid(v: Any) -> Any:
        return v.reshape(rows // ur, n // uc, ur, uc).permute(0, 2, 1, 3).reshape(rows, n)
    zy = zy.detach().contiguous()
    z, yr = grid(zy.view(torch.int32)[:, :ur * uc]), grid(zy.view(torch.int16)[:, 2 * ur * uc:])
    sw = s_w.detach().reshape(-1).to(torch.float32)
    sa = torch.zeros(rows, dtype=torch.float32, device=dev)
    sa[:m] = s_a.detach().reshape(-1).to(torch.float32)
    dq = ARM.arm().dequantize(z, sa, sw).view(torch.int16)
    bad = (~torch.isfinite(sw).all() | ((dq & 0x7FFF) > 0x7F80).any() | (dq != yr).any() | (z[m:] != 0).any()
           | (yr[:m] != y.detach().contiguous().view(torch.int16).reshape(m, n)).any())
    return not bool(bad)


__all__ = ["holds_on"]

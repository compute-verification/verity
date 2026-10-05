"""``F_c = BabyBear^6 = F_p[X] / (X^6 - 31)`` on torch int64 tensors whose LAST axis holds the 6 coordinates (plain,
reduced representatives; the same tower as ``encode/buckets_ref.py`` / ``buckets_v2.py``, ``W_NR = 31``).

Inversion by the norm: ``a^-1 = a^(p + p^2 + .. + p^5) / N(a)`` with ``N(a) = a^(1 + p + .. + p^5) in F_p``; the Frobenius
``a -> a^p`` is diagonal in the power basis because ``p = 1 mod 6`` (``X^p = X * 31^((p-1)/6)``), so it costs one
coordinate-wise multiply.  Zero has inverse 0 here (callers exclude poles by the public redraw rule).
"""
from __future__ import annotations

import torch

from ..field import P, batch_modinv

D = 6
NR = 31
OMEGA6 = pow(NR, (P - 1) // 6, P)          # a primitive 6th root of unity: X^p = OMEGA6 * X mod (X^6 - 31)
assert pow(OMEGA6, 6, P) == 1 and pow(OMEGA6, 3, P) != 1 and pow(OMEGA6, 2, P) != 1

__all__ = ["D", "NR", "fc_from_ints", "fc_inv", "fc_mul", "fc_one", "fc_pow_seq", "fc_scalar_mul", "fc_sum"]


def fc_one(shape: tuple, device) -> torch.Tensor:
    out = torch.zeros(tuple(shape) + (D,), dtype=torch.int64, device=device)
    out[..., 0] = 1
    return out


def fc_from_ints(coords: list[int], device) -> torch.Tensor:
    assert len(coords) == D
    return torch.tensor([c % P for c in coords], dtype=torch.int64, device=device)


def fc_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(..., 6) x (..., 6) -> (..., 6)``, broadcasting; every product reduced before the (<= 6-term) sums."""
    a, b = torch.broadcast_tensors(a, b)
    t = [None] * (2 * D - 1)
    for i in range(D):
        for j in range(D):
            v = (a[..., i] * b[..., j]) % P
            t[i + j] = v if t[i + j] is None else (t[i + j] + v)
    out = []
    for c in range(D):
        v = t[c]
        if c + D < 2 * D - 1:
            v = v + (t[c + D] % P) * NR
        out.append(v % P)
    return torch.stack(out, dim=-1)


def fc_scalar_mul(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """``F_t`` scalar(s) ``(...)`` times ``F_c`` ``(..., 6)``."""
    return (a * (s % P)[..., None]) % P


def fc_sum(a: torch.Tensor, dim: int) -> torch.Tensor:
    return a.sum(dim) % P


def _frob(a: torch.Tensor) -> torch.Tensor:
    w = torch.tensor([pow(OMEGA6, i, P) for i in range(D)], dtype=torch.int64, device=a.device)
    return (a * w) % P


_FAST: dict = {}
R_MOD_P = pow(2, 32, P)
R_INV = pow(R_MOD_P, P - 2, P)


def _fast_inverter(device):
    """b-v2's blocked Montgomery batch inverter (``buckets_v2.BatchInverterBlocked``, cupy) when importable on CUDA;
    ``V2_FAST_INV=0`` disables it.  Memoised per device; None when unavailable."""
    import os

    key = str(device)
    if key in _FAST:
        return _FAST[key]
    inv = None
    if str(device).startswith("cuda") and os.environ.get("V2_FAST_INV", "1") != "0":
        try:
            from archive.direct.encode.buckets_v2 import BatchInverterBlocked

            inv = BatchInverterBlocked()
        except Exception:  # noqa: BLE001 -- no cupy / kernel build failure: the torch path stays
            inv = None
    _FAST[key] = inv
    return inv


def fc_inv(a: torch.Tensor) -> torch.Tensor:
    """Elementwise inverse over the last axis (0 -> 0).  On CUDA with cupy: the b-v2 blocked Montgomery kernel."""
    a = a % P
    inv = _fast_inverter(a.device)
    if inv is not None and a.numel() >= 6:
        import cupy as cp

        shape = a.shape
        flat = a.reshape(-1, D)
        zero = ~flat.any(dim=1)                                                # a zero breaks a Montgomery chain: mask
        if bool(zero.any()):
            flat = flat.clone()
            flat[zero, 0] = 1
        mont = ((flat * R_MOD_P) % P).t().contiguous().to(torch.int32)        # (D, N) Montgomery form
        out = inv(cp.from_dlpack(mont).view(cp.uint32))
        res = torch.from_dlpack(out.view(cp.int32)).to(torch.int64) & 0xFFFFFFFF   # (D, N) = mont(a^-1)
        res = ((res * R_INV) % P).t()
        res[zero] = 0
        return res.reshape(shape)
    return _fc_inv_torch(a)


def _fc_inv_torch(a: torch.Tensor) -> torch.Tensor:
    # r = a^(p + p^2 + ... + p^5): Frobenius powers multiplied together
    f = _frob(a)                       # a^p
    r = f
    for _ in range(4):
        f = _frob(f)                   # a^(p^k)
        r = fc_mul(r, f)
    norm = fc_mul(a, r)                # in F_p: coordinates 1..5 vanish
    n0 = norm[..., 0]
    ninv = batch_modinv(n0)            # 0 -> 0 (0^(p-2) = 0)
    return fc_scalar_mul(ninv, r)


def fc_pow_seq(base: torch.Tensor, count: int) -> torch.Tensor:
    """``[1, base, base^2, ..., base^(count-1)]`` -> ``(count, 6)``."""
    out = [fc_one((), base.device)]
    for _ in range(count - 1):
        out.append(fc_mul(out[-1], base))
    return torch.stack(out)

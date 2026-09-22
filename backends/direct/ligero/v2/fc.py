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


def fc_inv(a: torch.Tensor) -> torch.Tensor:
    """Elementwise inverse over the last axis (0 -> 0)."""
    a = a % P
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

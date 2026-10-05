"""BabyBear arithmetic and the batched radix-2 NTT in torch (int64 elementwise; every product of two reduced values
fits in 63 bits, so one ``%`` per multiply suffices).

Layout of the Reed-Solomon code (PROTOCOL.md section 3): a row of l values is read as the evaluations of a
polynomial of degree < l on the subgroup H = <w_l> (natural order, index i -> w_l^i); its codeword is the
evaluations on the coset g * <w_n>, g = 31 (the multiplicative generator, not in <w_n>), so H and the codeword
domain are disjoint and Z_H(x) = x^l - 1 never vanishes on the coset.  ``encode`` = INTT_l, zero-pad, scale by
g^k, NTT_n.
"""
from __future__ import annotations

from typing import ClassVar

import torch

P = 15 * 2**27 + 1
GEN = 31                     # multiplicative generator of BabyBear
TWO_ADICITY = 27

__all__ = ["GEN", "Domain", "P", "batch_modinv", "coset_points", "encode", "eval_on_coset", "intt", "modinv", "ntt", "poly_eval", "root_of_unity", "zh_on_coset"]


def modpow(a: int, e: int, m: int = P) -> int:
    return pow(a, e, m)


def modinv(a: int) -> int:
    return pow(a, P - 2, P)


def root_of_unity(n: int) -> int:
    assert n & (n - 1) == 0 and n <= 2**TWO_ADICITY
    return pow(GEN, (P - 1) // n, P)


def _bitrev(n: int, device) -> torch.Tensor:
    bits = n.bit_length() - 1
    idx = torch.arange(n, device=device)
    rev = torch.zeros_like(idx)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


class Domain:
    """Precomputed tables for one size ``n`` on one device."""

    _cache: ClassVar[dict] = {}

    def __init__(self, n: int, device):
        self.n = n
        self.device = device
        w = root_of_unity(n)
        self.w = w
        self.bitrev = _bitrev(n, device)
        # twiddles per stage: for length L, w_L^j for j < L/2
        self.stages = []
        self.stages_inv = []
        L = 2
        while L <= n:
            wL = pow(w, n // L, P)
            tw = [1]
            for _ in range(L // 2 - 1):
                tw.append(tw[-1] * wL % P)
            t = torch.tensor(tw, dtype=torch.int64, device=device)
            self.stages.append(t)
            wLi = modinv(wL)
            twi = [1]
            for _ in range(L // 2 - 1):
                twi.append(twi[-1] * wLi % P)
            self.stages_inv.append(torch.tensor(twi, dtype=torch.int64, device=device))
            L *= 2
        self.n_inv = modinv(n)

    @classmethod
    def get(cls, n: int, device) -> Domain:
        key = (n, str(device))
        if key not in cls._cache:
            cls._cache[key] = cls(n, device)
        return cls._cache[key]


def _butterfly_eager(a: torch.Tensor, b: torch.Tensor, tw: torch.Tensor) -> torch.Tensor:
    bt = (b * tw) % P
    return torch.cat(((a + bt) % P, (a - bt) % P), dim=-1)


_COMPILED: dict = {}


def compiled(fn, device):
    """``torch.compile(fn, dynamic=True)`` on CUDA (elementwise chains fuse into one pass), ``fn`` itself elsewhere
    or if the compiler is unavailable.  Memoised per (function, device)."""
    key = (fn, str(device))
    out = _COMPILED.get(key)
    if out is None:
        out = fn
        if str(device).startswith("cuda"):
            try:
                out = torch.compile(fn, dynamic=True)
            except Exception:  # noqa: BLE001 -- no compiler: stay eager
                out = fn
        _COMPILED[key] = out
    return out


def _butterfly(device):
    """The stage kernel: fused (one pass instead of seven) on CUDA."""
    return compiled(_butterfly_eager, device)


def ntt(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Batched NTT over the last dimension (size n, a power of two): natural order in, natural order out.
    Forward: X[k] = sum_i x[i] w^(ik).  Inverse divides by n."""
    n = x.shape[-1]
    dom = Domain.get(n, x.device)
    lead = x.shape[:-1]
    x = x.index_select(-1, dom.bitrev)
    bf = _butterfly(x.device)
    L = 2
    for tw in (dom.stages_inv if inverse else dom.stages):
        half = L // 2
        x = x.reshape(lead + (n // L, L))
        x = bf(x[..., :half], x[..., half:], tw)
        L *= 2
    x = x.reshape(lead + (n,))
    if inverse:
        x = (x * dom.n_inv) % P
    return x


def intt(x: torch.Tensor) -> torch.Tensor:
    return ntt(x, inverse=True)


def coset_scale(n: int, device) -> torch.Tensor:
    key = ("coset_scale", n, str(device))
    c = Domain._cache.get(key)
    if c is None:
        pw = [1]
        for _ in range(n - 1):
            pw.append(pw[-1] * GEN % P)
        c = torch.tensor(pw, dtype=torch.int64, device=device)
        Domain._cache[key] = c
    return c


def encode(rows: torch.Tensor, n: int) -> torch.Tensor:
    """(R, l) message rows (evaluations on H) -> (R, n) codewords (evaluations on g * <w_n>)."""
    l = rows.shape[-1]
    coeffs = intt(rows % P)
    if n > l:
        pad = torch.zeros(rows.shape[:-1] + (n - l,), dtype=torch.int64, device=rows.device)
        coeffs = torch.cat((coeffs, pad), dim=-1)
    coeffs = (coeffs * coset_scale(n, rows.device)) % P
    return ntt(coeffs)


def eval_on_coset(coeffs: torch.Tensor, n: int) -> torch.Tensor:
    """(R, K <= n) coefficient rows -> (R, n) evaluations on g * <w_n> (one NTT; replaces Horner at many points)."""
    K = coeffs.shape[-1]
    if K < n:
        pad = torch.zeros(coeffs.shape[:-1] + (n - K,), dtype=torch.int64, device=coeffs.device)
        coeffs = torch.cat((coeffs, pad), dim=-1)
    return ntt((coeffs % P * coset_scale(n, coeffs.device)) % P)


def coset_points(n: int, idx: torch.Tensor) -> torch.Tensor:
    """eta_c = g * w_n^c for the given column indices (int64 tensor on any device)."""
    w = root_of_unity(n)
    out = []
    for c in idx.tolist():
        out.append(GEN * pow(w, c, P) % P)
    return torch.tensor(out, dtype=torch.int64, device=idx.device)


def zh_on_coset(n: int, l: int, device) -> torch.Tensor:
    """Z_H(eta_c) = eta_c^l - 1 for every column c (nonzero on the coset).  Cached per (n, l, device)."""
    key = ("zh", n, l, str(device))
    hit = Domain._cache.get(key)
    if hit is None:
        hit = _zh_on_coset(n, l, device)
        Domain._cache[key] = hit
    return hit


def _zh_on_coset(n: int, l: int, device) -> torch.Tensor:
    w = root_of_unity(n)
    gl = pow(GEN, l, P)
    wl = pow(w, l, P)          # order n / l
    vals = []
    cur = gl
    for _ in range(n):
        vals.append((cur - 1) % P)
        cur = cur * wl % P
    z = torch.tensor(vals, dtype=torch.int64, device=device)
    assert (z != 0).all()
    return z


def batch_modinv(x: torch.Tensor) -> torch.Tensor:
    """Elementwise inverse by Fermat (square-and-multiply on the exponent p - 2)."""
    result = torch.ones_like(x)
    base = x % P
    e = P - 2
    while e:
        if e & 1:
            result = (result * base) % P
        base = (base * base) % P
        e >>= 1
    return result


def poly_eval(coeffs: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """(D, K) coefficient rows evaluated at (t,) points -> (D, t), Horner."""
    D, K = coeffs.shape
    acc = torch.zeros((D, xs.shape[0]), dtype=torch.int64, device=xs.device)
    for k in range(K - 1, -1, -1):
        acc = (acc * xs + coeffs[:, k:k + 1]) % P
    return acc

"""BabyBear and BabyBear^6 arithmetic on torch tensors (int64 compute, int32 storage) plus the small-value
Python-int versions the transcript-side logic uses.

Base field ``p = 2^31 - 2^27 + 1``; challenge field ``F_{p^6} = F_p[X] / (X^6 - W)``, ``W = 31`` (the same
``W`` as ``backends/gkr/tensor/reference.py``: the smallest non-square non-cube).  An extension element is a
tensor whose last dimension is 6 (planar coordinate order ``c0..c5``); a base element is any int64 tensor of
canonical values in ``[0, p)``.  Every product of two canonical values (< 2^62) fits int64, so one product is
reduced before it is accumulated; sums of up to 2^32 canonical values fit as well.

The heavy contractions (``Σ_g w_g X_g``, ``Σ_c eq_c V_c``, the Reed-Solomon row combinations) are done as exact
float64 matmuls on 16-bit limbs (``mm_mod``): with ``k`` terms the partial sums are below ``k · 2^32``, exact in
float64 for ``k ≤ 2^21``.  On the H100 those run on the FP64 tensor cores; on a laptop CPU the same code runs.

Hypercube conventions are the Rust crate's (``field.rs``): variables are bound from the top bit down (``fold``
halves a table by combining ``v[i]`` and ``v[i + half]``), a point is the list of challenges in that order, and
``eq_table(point)`` is indexed accordingly.
"""
from __future__ import annotations

import torch

P = 2**31 - 2**27 + 1
W = 31
DEG = 6
_R32 = (2**32) % P          # 2^32 mod p
_R16 = 2**16

Ext = tuple[int, int, int, int, int, int]
ZERO: Ext = (0,) * DEG
ONE: Ext = (1,) + (0,) * (DEG - 1)

# 36 -> 6 reduction matrix for the product coordinates (i, j) -> i + j (mod X^6 = W)
_RED_ROWS = []
for _i in range(DEG):
    for _j in range(DEG):
        row = [0] * DEG
        if _i + _j < DEG:
            row[_i + _j] = 1
        else:
            row[_i + _j - DEG] = W
        _RED_ROWS.append(row)
_RED_CACHE: dict[torch.device, torch.Tensor] = {}


def red_matrix(device) -> torch.Tensor:
    m = _RED_CACHE.get(device)
    if m is None:
        m = torch.tensor(_RED_ROWS, dtype=torch.float64, device=device)
        _RED_CACHE[device] = m
    return m


# ---------------------------------------------------------------------------------------------------------------------
# Python-int extension arithmetic (small values: challenges, claims, interpolation)
# ---------------------------------------------------------------------------------------------------------------------

def e_base(a: int) -> Ext:
    return (a % P,) + (0,) * (DEG - 1)


def e_add(a: Ext, b: Ext) -> Ext:
    return tuple((x + y) % P for x, y in zip(a, b))


def e_sub(a: Ext, b: Ext) -> Ext:
    return tuple((x - y) % P for x, y in zip(a, b))


def e_neg(a: Ext) -> Ext:
    return tuple((-x) % P for x in a)


def e_scale(a: Ext, c: int) -> Ext:
    c %= P
    return tuple(x * c % P for x in a)


def e_mul(a: Ext, b: Ext) -> Ext:
    acc = [0] * (2 * DEG - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                acc[i + j] += x * y
    return tuple((acc[d] + W * acc[d + DEG]) % P for d in range(DEG - 1)) + (acc[DEG - 1] % P,)


def e_pow(a: Ext, n: int) -> Ext:
    r, b = ONE, a
    while n:
        if n & 1:
            r = e_mul(r, b)
        b = e_mul(b, b)
        n >>= 1
    return r


def e_is_zero(a: Ext) -> bool:
    return all(x % P == 0 for x in a)


def e_inv(a: Ext) -> Ext:
    """Inverse by Fermat in F_{p^6} (only used in tests)."""
    return e_pow(a, P**DEG - 2)


def interpolate(vals: list[Ext], r: Ext) -> Ext:
    """``s(r)`` from ``s(0..d)`` by Lagrange interpolation (``field.rs::interpolate``)."""
    d = len(vals) - 1
    acc = ZERO
    for i in range(d + 1):
        num = ONE
        den = 1
        for j in range(d + 1):
            if j == i:
                continue
            num = e_mul(num, e_sub(r, e_base(j)))
            den = den * (i - j)
        acc = e_add(acc, e_scale(e_mul(vals[i], num), pow(den % P, P - 2, P)))
    return acc


def eq_point(a: list[Ext], b: list[Ext]) -> Ext:
    e = ONE
    for x, y in zip(a, b):
        e = e_mul(e, e_add(e_mul(x, y), e_mul(e_sub(ONE, x), e_sub(ONE, y))))
    return e


def to_bytes(x: Ext) -> bytes:
    return b"".join(int(c).to_bytes(4, "little") for c in x)


def from_bytes(b: bytes) -> Ext:
    return tuple(int.from_bytes(b[4 * i:4 * i + 4], "little") for i in range(DEG))


# ---------------------------------------------------------------------------------------------------------------------
# Tensor arithmetic
# ---------------------------------------------------------------------------------------------------------------------

def t_ext(x: Ext, device) -> torch.Tensor:
    return torch.tensor([int(c) % P for c in x], dtype=torch.int64, device=device)


def t_exts(xs: list[Ext], device) -> torch.Tensor:
    if not xs:
        return torch.zeros((0, DEG), dtype=torch.int64, device=device)
    return torch.tensor([[int(c) % P for c in x] for x in xs], dtype=torch.int64, device=device)


def py_ext(t: torch.Tensor) -> Ext:
    return tuple(int(v) for v in t.reshape(-1).tolist())


def py_exts(t: torch.Tensor) -> list[Ext]:
    return [tuple(int(v) for v in row) for row in t.reshape(-1, DEG).tolist()]


def ext_from_base(a: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(a.shape + (DEG,), dtype=torch.int64, device=a.device)
    out[..., 0] = a
    return out


def base_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b) % P


def ext_scale(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Extension tensor times a base tensor (broadcast over the coordinate axis)."""
    return (a * c.unsqueeze(-1)) % P


def ext_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Coordinate-wise product of two extension tensors (broadcasting), reduced mod ``X^6 - W`` and ``p``.
    On CUDA the fused Triton kernel (``kernels.ext_mul``) is used when available; the torch path is the reference."""
    if _fast_ext_mul is not None and a.is_cuda:
        return _fast_ext_mul(a, b)
    return _ref_ext_mul(a, b)


def _ref_ext_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.to(torch.int64)
    b = b.to(torch.int64)
    prod = (a.unsqueeze(-1) * b.unsqueeze(-2)) % P           # (..., 6, 6): [i, j] = a_i b_j
    shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    flat = prod.reshape(-1, DEG * DEG).to(torch.float64)
    out = flat @ red_matrix(a.device)                          # exact: < 36 * 31 * 2^31 < 2^53
    return (out.to(torch.int64) % P).reshape(*shape, DEG)


def ext_mul_chunked(a: torch.Tensor, b: torch.Tensor, chunk_elems: int = 1 << 22) -> torch.Tensor:
    """``ext_mul`` for large operands of equal leading shape, bounded temporaries."""
    shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    n = 1
    for s in shape:
        n *= s
    if n <= chunk_elems:
        return ext_mul(a, b)
    a = a.expand(*shape, DEG).reshape(n, DEG)
    b = b.expand(*shape, DEG).reshape(n, DEG)
    out = torch.empty((n, DEG), dtype=torch.int64, device=a.device)
    for lo in range(0, n, chunk_elems):
        hi = min(n, lo + chunk_elems)
        out[lo:hi] = ext_mul(a[lo:hi], b[lo:hi])
    return out.reshape(*shape, DEG)


def ext_sum(a: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return a.to(torch.int64).sum(dim=dim) % P


class Limbs:
    """A canonical int64 matrix pre-split into its two 16-bit float64 limb matrices (reused across ``mm_mod`` calls)."""

    def __init__(self, B: torch.Tensor) -> None:
        B = B.to(torch.int64)
        self.shape = B.shape
        self.lo = (B & 0xFFFF).to(torch.float64)
        self.hi = (B >> 16).to(torch.float64)


def mm_mod(A: torch.Tensor | Limbs, B: torch.Tensor | Limbs) -> torch.Tensor:
    """``(A @ B) mod p`` for canonical int64 matrices via exact float64 limb matmuls (``k ≤ 2^21``)."""
    if isinstance(A, Limbs):
        a_lo, a_hi = A.lo, A.hi
    else:
        A = A.to(torch.int64)
        a_lo = (A & 0xFFFF).to(torch.float64)
        a_hi = (A >> 16).to(torch.float64)
    k = a_lo.shape[-1]
    if k > (1 << 21):
        raise ValueError("mm_mod: contraction too long for exact float64 limbs")
    if isinstance(B, Limbs):
        b_lo, b_hi = B.lo, B.hi
    else:
        B = B.to(torch.int64)
        b_lo = (B & 0xFFFF).to(torch.float64)
        b_hi = (B >> 16).to(torch.float64)
    ll = (a_lo @ b_lo).to(torch.int64) % P
    lh = ((a_lo @ b_hi).to(torch.int64) + (a_hi @ b_lo).to(torch.int64)) % P
    hh = (a_hi @ b_hi).to(torch.int64) % P
    return (ll + lh * _R16 + hh * _R32) % P


def ext_contract(X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``Σ_g X[..., g, :] · w[g, :]`` in the extension: ``X`` ``(m, k, 6)``, ``w`` ``(k, 6)`` -> ``(m, 6)``."""
    k = w.shape[0]
    wmat = torch.zeros((k, DEG, DEG), dtype=torch.int64, device=w.device)   # [g, i, d] = coefficient of X_i w_j
    w = w.to(torch.int64)
    for i in range(DEG):
        for j in range(DEG):
            d = i + j
            if d < DEG:
                wmat[:, i, d] = w[:, j]
            else:
                wmat[:, i, d - DEG] = (W * w[:, j]) % P
    return mm_mod(X.reshape(X.shape[0], k * DEG), wmat.reshape(k * DEG, DEG))


def base_contract_ext(X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``Σ_g X[m, g] · w[g, :]`` with base ``X`` ``(m, k)`` and extension ``w`` ``(k, 6)`` -> ``(m, 6)``."""
    return mm_mod(X, w)


def fold(v: torch.Tensor, r: torch.Tensor, chunk_elems: int = 1 << 22) -> torch.Tensor:
    """``v[i] + (v[i + half] - v[i]) r`` on the leading axis (extension ``v``, ``(2h, ..., 6)``); bounded temporaries."""
    if _fast_fold is not None and v.is_cuda:
        return _fast_fold(v, r)
    half = v.shape[0] // 2
    per_row = v[0].numel() // DEG
    step = max(1, chunk_elems // per_row)
    out = torch.empty((half,) + tuple(v.shape[1:]), dtype=torch.int64, device=v.device)
    for lo_i in range(0, half, step):
        hi_i = min(half, lo_i + step)
        lo = v[lo_i:hi_i].to(torch.int64)
        hi = v[half + lo_i:half + hi_i].to(torch.int64)
        out[lo_i:hi_i] = (lo + ext_mul((hi - lo) % P, r)) % P
    return out


def fold_base(v: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """First fold of a base-field table into the extension: ``lo + (hi - lo) r``."""
    if _fast_fold is not None and v.is_cuda:
        return _fast_fold(v, r, base=True)
    half = v.shape[0] // 2
    lo = v[:half].to(torch.int64)
    d = (v[half:].to(torch.int64) - lo) % P
    out = (d.unsqueeze(-1) * r) % P
    out[..., 0] = (out[..., 0] + lo) % P
    return out


def eq_table(point: list[Ext], device) -> torch.Tensor:
    """``eq(point, x)`` for every ``x`` on the hypercube ``(2^n, 6)``, top-bit-first (``field.rs::eq_table``)."""
    t = torch.zeros((1, DEG), dtype=torch.int64, device=device)
    t[0, 0] = 1
    for r in reversed(point):
        rt = t_ext(r, device)
        one_minus = (t_ext(ONE, device) - rt) % P
        t = torch.cat([ext_mul_chunked(t, one_minus), ext_mul_chunked(t, rt)], dim=0)
    return t


def eval_mle(values: torch.Tensor, point: list[Ext]) -> Ext:
    v = values.to(torch.int64)
    for r in point:
        v = fold(v, t_ext(r, v.device))
    return py_ext(v[0])


def geometric(rho: Ext, geo: Ext, n: int, device) -> torch.Tensor:
    """``rho · geo^k`` for ``k < n`` as an ``(n, 6)`` tensor (doubling steps)."""
    pw = t_ext(rho, device).reshape(1, DEG)
    g = geo
    while pw.shape[0] < n:
        pw = torch.cat([pw, ext_mul(pw, t_ext(g, device))], dim=0)
        g = e_mul(g, g)
    return pw[:n]


# ---------------------------------------------------------------------------------------------------------------------
# Base-field helpers (roots of unity for the Reed-Solomon code)
# ---------------------------------------------------------------------------------------------------------------------

GENERATOR = 31  # a primitive element of BabyBear (Plonky3's choice); order p - 1 = 2^27 * 15
TWO_ADICITY = 27


def root_of_unity(order: int) -> int:
    if order & (order - 1) or order > 2**TWO_ADICITY:
        raise ValueError("order must be a power of two <= 2^27")
    return pow(GENERATOR, (P - 1) // order, P)


def inv_mod(a: int) -> int:
    return pow(a % P, P - 2, P)


# the fused kernel (Triton, CUDA); ``VERITY_GPU_TORCH_ONLY=1`` disables it
_fast_ext_mul = None
_fast_fold = None
_fast_logup_round = None
_fast_phase1 = None
_fast_scatter = None
_fast_gate_eval = None
_fast_combine_level = None
try:
    import os as _os

    if not _os.environ.get("VERITY_GPU_TORCH_ONLY"):
        from . import kernels as _kernels

        if _kernels.AVAILABLE:
            _fast_ext_mul = _kernels.ext_mul
            _fast_fold = _kernels.fold
            _fast_logup_round = _kernels.logup_round
            _fast_phase1 = _kernels.phase1_round
            _fast_scatter = _kernels.scatter_terms
            _fast_gate_eval = _kernels.gate_eval
            _fast_combine_level = _kernels.combine_level
except Exception:  # noqa: BLE001 -- no Triton / no CUDA: the torch path is used
    _fast_ext_mul = _fast_fold = _fast_logup_round = _fast_phase1 = _fast_scatter = _fast_gate_eval = _fast_combine_level = None

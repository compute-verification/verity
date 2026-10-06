"""Exact mod-p linear algebra in torch for the extension-field rounds (i >= 2), where the matrices are small
(<= 2^22 entries) but the values are F_{p^6}: an extension matrix is kept as 6 base coordinate planes ``(6, ...)``.

``matmul_mod_p``: (M, K) @ (K, N) over F_p through fp64 GEMMs on 16-bit limbs -- every partial sum is an integer
< K 2^32 < 2^53, so cuBLAS DGEMM is exact (K is chunked at 2^20).  ``ext_mul_planes``: schoolbook F_{p^6} product
of two plane tensors.  ``eq_planes``: the multilinear eq tensor of a point, built on the device by doubling.
"""
from __future__ import annotations

import numpy as np
import torch

from .ext import BETA, D, P

_K_CHUNK = 1 << 20
_2_16 = pow(2, 16, P)
_2_32 = pow(2, 32, P)


def _limbs(x: torch.Tensor):
    x = x.to(torch.int64) % P
    return (x >> 16).to(torch.float64), (x & 0xFFFF).to(torch.float64)


def matmul_mod_p(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A (M, K), B (K, N) integer tensors with entries in [0, p) -> (M, N) int64 in [0, p)."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    out = torch.zeros((M, N), dtype=torch.int64, device=A.device)
    for k0 in range(0, K, _K_CHUNK):
        ah, al = _limbs(A[:, k0:k0 + _K_CHUNK])
        bh, bl = _limbs(B[k0:k0 + _K_CHUNK])
        hh = (ah @ bh).to(torch.int64) % P
        hl = ((ah @ bl) + (al @ bh)).to(torch.int64) % P
        ll = (al @ bl).to(torch.int64) % P
        out = (out + (hh * _2_32) % P + (hl * _2_16) % P + ll) % P
    return out


def matmul_mod_p_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The numpy twin (verifier side)."""
    A = np.asarray(A, dtype=np.int64) % P
    B = np.asarray(B, dtype=np.int64) % P
    K = A.shape[1]
    out = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    for k0 in range(0, K, _K_CHUNK):
        a = A[:, k0:k0 + _K_CHUNK]
        b = B[k0:k0 + _K_CHUNK]
        ah, al = (a >> 16).astype(np.float64), (a & 0xFFFF).astype(np.float64)
        bh, bl = (b >> 16).astype(np.float64), (b & 0xFFFF).astype(np.float64)
        hh = (ah @ bh).astype(np.int64) % P
        hl = ((ah @ bl) + (al @ bh)).astype(np.int64) % P
        ll = (al @ bl).astype(np.int64) % P
        out = (out + (hh * _2_32) % P + (hl * _2_16) % P + ll) % P
    return out


def ext_mul_planes(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b (6, ...) int64 planes (broadcast on the trailing dims) -> (6, ...) int64."""
    shape = torch.broadcast_shapes(a.shape[1:], b.shape[1:])
    out = torch.zeros((D,) + tuple(shape), dtype=torch.int64, device=a.device)
    for i in range(D):
        for j in range(D):
            prod = (a[i] * b[j]) % P
            k = i + j
            if k >= D:
                out[k - D] += prod * BETA
            else:
                out[k] += prod
    return out % P


def _eq_planes_doubling(z: np.ndarray, device) -> torch.Tensor:
    """Small tables on the host (numpy doubling, 3 numpy ops per step) then one upload as (6, 2^k) planes."""
    from .claims import eq_vector
    vec = eq_vector(np.asarray(z, dtype=np.int64) % P)                # (2^k, 6)
    return torch.as_tensor(np.ascontiguousarray(vec.T), device=device)


def eq_planes(z: np.ndarray, device) -> torch.Tensor:
    """z (k, 6) numpy -> (6, 2^k) int64: eq(x, z) with bit j of x <-> z_j.  Two half tables (doubling, <= 2^12 each)
    and ONE broadcast extension product on the device: eq(x) = eq_lo(x_lo) eq_hi(x_hi), x = x_lo + 2^h x_hi."""
    k = z.shape[0]
    if k <= 12:
        return _eq_planes_doubling(z, device)
    h = k // 2
    lo = eq_planes(z[:h], device)                    # (6, 2^h)
    hi = eq_planes(z[h:], device)                    # (6, 2^{k-h})
    out = ext_mul_planes(hi[:, :, None], lo[:, None, :])          # (6, 2^{k-h}, 2^h)
    return out.reshape(D, 1 << k)


def _powers_rows(e: np.ndarray, count: int) -> np.ndarray:
    """(T, count): e[t]^i, doubling (log2 count numpy ops)."""
    T = len(e)
    out = np.empty((T, count), dtype=np.int64)
    out[:, 0] = 1
    step, cur = 1, e % P
    while step < count:
        m = min(step, count - step)
        out[:, step:step + m] = out[:, :m] * cur[:, None] % P
        step *= 2
        cur = cur * cur % P
    return out


def powers_matrix(etas: np.ndarray, R: int, device) -> torch.Tensor:
    """(R, T) int64: etas[t]^row, from two tables per eta (row = lo + 2^11 hi) and one broadcast product."""
    e = np.asarray(etas, dtype=np.int64) % P
    lo_bits = min(11, R.bit_length() - 1)
    lo_n = 1 << lo_bits
    hi_n = R // lo_n
    lo_t = torch.as_tensor(_powers_rows(e, lo_n), device=device)                         # (T, lo_n)
    e_hi = np.array([pow(int(x), lo_n, P) for x in e], dtype=np.int64)
    hi_t = torch.as_tensor(_powers_rows(e_hi, hi_n), device=device)                      # (T, hi_n)
    prod = (hi_t.t()[:, None, :] * lo_t.t()[None, :, :]) % P                             # (hi_n, lo_n, T)
    return prod.reshape(R, -1)


def contract_columns_ext(M: torch.Tensor, geom_rowparts: torch.Tensor | None, eq_rowpart: torch.Tensor | None):
    """M (6, C, R) planes; geom_rowparts (R, Tg) base; eq_rowpart (6, R) planes.
    -> (u_geom (Tg, C, 6) numpy or None, u_eq (C, 6) numpy or None)."""
    _, C, R = M.shape
    cols = []
    if geom_rowparts is not None:
        cols.append(geom_rowparts)
    if eq_rowpart is not None:
        cols.append(eq_rowpart.t())
    Bmat = torch.cat(cols, dim=1) if len(cols) > 1 else cols[0]           # (R, Tg [+ 6])
    res = matmul_mod_p(M.reshape(D * C, R), Bmat).view(D, C, -1)          # one GEMM for the 6 planes: (6, C, Tg + 6)
    Tg = geom_rowparts.shape[1] if geom_rowparts is not None else 0
    u_geom = u_eq = None
    if Tg:
        # u_geom[t, c] coordinate a = res[a][c, t]
        u_geom = torch.stack([res[a][:, :Tg] for a in range(D)], dim=-1).permute(1, 0, 2).contiguous().cpu().numpy()
    if eq_rowpart is not None:
        acc = torch.zeros((C, D), dtype=torch.int64, device=M.device)
        for a in range(D):
            for b in range(D):
                prod = res[a][:, Tg + b]
                k = a + b
                if k >= D:
                    acc[:, k - D] = (acc[:, k - D] + prod * BETA) % P
                else:
                    acc[:, k] = (acc[:, k] + prod) % P
        u_eq = acc.cpu().numpy()
    return u_geom, u_eq


def contract_rows_ext(M: torch.Tensor, rbar: np.ndarray) -> torch.Tensor:
    """M (6, C, R) planes, rbar (C, 6) numpy -> y (6, R) int32 planes: y[row] = sum_c M[c, row] rbar[c]."""
    _, C, R = M.shape
    rb = torch.as_tensor(np.asarray(rbar, dtype=np.int64) % P, device=M.device)      # (C, 6)
    acc = torch.zeros((D, R), dtype=torch.int64, device=M.device)
    for a in range(D):
        part = matmul_mod_p(M[a].t(), rb)                                             # (R, 6): sum_c M_a[c,row] rb[c,b]
        for b in range(D):
            k = a + b
            if k >= D:
                acc[k - D] = (acc[k - D] + part[:, b] * BETA) % P
            else:
                acc[k] = (acc[k] + part[:, b]) % P
    return acc.to(torch.int32)

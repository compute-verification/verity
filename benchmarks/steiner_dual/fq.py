"""Batched linear algebra over a prime field F_p, numpy only (no scipy on the VMs)."""

import numpy as np


def rank(m, p):
    """Ranks of a batch of matrices m[..., r, c] over F_p."""
    a = np.array(m, dtype=np.int64) % p
    if a.ndim == 2:
        a = a[None]
    a = a.reshape((-1,) + a.shape[-2:]).copy()
    b, rows, cols = a.shape
    inv = np.zeros(p, dtype=np.int64)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    rk = np.zeros(b, dtype=np.int64)
    ar = np.arange(b)
    for c in range(cols):
        if (rk >= rows).all():
            break
        # pivot: first row >= rk with nonzero entry in column c
        rowidx = np.arange(rows)[None, :]
        cand = (a[:, :, c] != 0) & (rowidx >= rk[:, None])
        has = cand.any(axis=1)
        piv = np.argmax(cand, axis=1)
        sel = ar[has]
        if sel.size == 0:
            continue
        pr, tr = piv[sel], rk[sel]
        tmp = a[sel, pr].copy()
        a[sel, pr] = a[sel, tr]
        a[sel, tr] = tmp
        pv = a[sel, tr, c]
        a[sel, tr] = (a[sel, tr] * inv[pv][:, None]) % p
        f = a[sel, :, c].copy()
        f[np.arange(sel.size), tr] = 0
        a[sel] = (a[sel] - f[:, :, None] * a[sel, tr][:, None, :]) % p
        rk[sel] += 1
    return rk


def matmul_targets(n, m=None, k=None):
    """Targets T_(r,c) = sum_j x_{rj} y_{jc} of the <n,k,m> product as (n*k) x (k*m) matrices.

    Rows index x_{r j} as r*k + j, columns index y_{j c} as j*m + c.
    """
    m = n if m is None else m
    k = n if k is None else k
    out = {}
    for r in range(n):
        for c in range(m):
            t = np.zeros((n * k, k * m), dtype=np.int64)
            for j in range(k):
                t[r * k + j, j * m + c] = 1
            out[(r, c)] = t
    return out

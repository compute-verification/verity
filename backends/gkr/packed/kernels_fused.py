"""Round-fused kernels for the packed checker sumcheck (lane ``a-fusion``).

**One pass for the ``k`` packed rounds** (:func:`packed_u_inner_product`).  Round ``i <= k`` of the eq-weighted sumcheck
``sum_b eq(gamma, b) g(b) h(b)`` needs ``sum_b eq(gamma_{>=i}, b) g(r_{<i}, x, b) h(r_{<i}, x, b)``; the challenges enter
only through ``eq(r_{<i}, a)`` coefficients, so every packed round is a linear combination of the same ``2^k x 2^k``
base-field tensor

    U[c, c'] = sum_{b'} eq(gamma_{>=k}, b') g(c, b') h(c', b'),    c, c' in {0,1}^k (bits 0..k-1 of the table index)

(``DESIGN.md`` 3 with all ``k`` rounds expanded at once).  Three separate packed rounds read ``g, h`` three times and
form ``1.5 t + 3 t + 6 t = 10.5 t`` Hadamards; ``U`` is one read and ``4^k / 2^k . t = 8 t`` Hadamards at ``k = 3``, and the
rounds become 64-term contractions of ``U`` (:func:`round_from_U`).  The messages are the same field elements in a
different summation order: transcripts stay bit-identical to ``reference.packed_prover``.

Kernel: per ``(b2, split)`` program, tiles of ``BLOCK_K`` values of ``b'`` load ``G, H`` as ``(BLOCK_K, 2^k)`` int32, form the
``(BLOCK_K, 2^k, 2^k)`` Montgomery products by broadcasting (whole-tile ``tl.join`` for the limbs, no 192-name join tree),
``A (BLOCK_K, 4^k . 4)`` int8, ``tl.dot(A^T, E)`` against the ``R``-scaled eq-weight limbs; INT32 partials out in
:func:`kernels_triton.reduce_contract`'s row layout ``(b2, r = c . 2^k + c', j)``.

The fold-fused extension round lives in the generated module (``kernels_triton.ext_fold_inner_product``).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import reference as ref
from .kernels_triton import _limbs, mmul

P = ref.P


@triton.jit
def _packed_u_kernel(g_ptr, h_ptr, e_ptr, z_ptr, K1, SPLIT_LEN, TWOK: tl.constexpr, ROWS4: tl.constexpr, BLOCK_K: tl.constexpr):
    b2 = tl.program_id(0)
    sp = tl.program_id(1)
    offs_n = tl.arange(0, 32)
    offs_c = tl.arange(0, TWOK)
    acc = tl.zeros((ROWS4, 32), dtype=tl.int32)
    for c in range(0, SPLIT_LEN, BLOCK_K):
        b1 = sp * SPLIT_LEN + c + tl.arange(0, BLOCK_K)
        base = (b2.to(tl.int64) * K1 + b1) * TWOK
        G = tl.load(g_ptr + base[:, None] + offs_c[None, :])                     # (BLOCK_K, TWOK)
        H = tl.load(h_ptr + base[:, None] + offs_c[None, :])
        E = tl.load(e_ptr + b1[:, None] * 32 + offs_n[None, :])                  # (BLOCK_K, 32) int8
        Gb = tl.broadcast_to(tl.expand_dims(G, 2), (BLOCK_K, TWOK, TWOK))
        Hb = tl.broadcast_to(tl.expand_dims(H, 1), (BLOCK_K, TWOK, TWOK))
        prod = tl.reshape(mmul(Gb, Hb), (BLOCK_K, TWOK * TWOK))                  # g(c) h(c') R^-1 at column c . TWOK + c'
        l0, l1, l2, l3 = _limbs(prod)
        # join(join(l0, l2), join(l1, l3)) -> (.., a, b) with limb 2 a + b: row-major column = 4 col + limb
        A = tl.reshape(tl.join(tl.join(l0, l2), tl.join(l1, l3)), (BLOCK_K, ROWS4), can_reorder=False)
        acc += tl.dot(tl.trans(A), E, out_dtype=tl.int32)
    offs_m = tl.arange(0, ROWS4)
    z = z_ptr + ((sp * tl.num_programs(0) + b2).to(tl.int64) * ROWS4) * 24
    tl.store(z + offs_m[:, None] * 24 + offs_n[None, :], acc, mask=offs_n[None, :] < 24)


def packed_u_inner_product(g: torch.Tensor, h: torch.Tensor, e_lo32: torch.Tensor, k: int, K1: int, K2: int, split: int,
                           BLOCK_K: int = 128, num_warps: int = 4) -> torch.Tensor:
    """``(split, K2 . 4^k . 4, 24) int32`` partial inner products of the ``U`` rows (``r = c . 2^k + c'``, 4 limb rows each)
    against the eq-weight limbs of ``b' = (b1, b2)``; ``g, h (t,)`` base plain, ``t = 2^k . K1 . K2``."""
    if not 1 <= k <= 3:
        raise ValueError("packed U pass covers k = 1..3 (4^k . 4 <= 256 limb rows)")
    if K1 % (split * BLOCK_K):
        raise ValueError("K1 must be a multiple of split * BLOCK_K")
    twok = 1 << k
    rows4 = twok * twok * 4
    Z = torch.empty((split, K2 * rows4, 24), dtype=torch.int32, device=g.device)
    _packed_u_kernel[(K2, split)](g, h, e_lo32, Z, K1, K1 // split, TWOK=twok, ROWS4=rows4, BLOCK_K=BLOCK_K, num_warps=num_warps)
    return Z


def u_round_terms(i: int, k: int) -> dict:
    """Host constants of :func:`round_from_U` for round ``i`` of ``k``: index vectors over the terms ``(a, a', d)`` and the ``U``
    row indices for ``(x1, x2) in {0,1}^2``; challenge-independent, built once per ``(i, k)``."""
    half = 1 << (i - 1)
    nd = 1 << (k - i)
    ia, iap, idd = [], [], []
    for a in range(half):
        for ap in range(half):
            for d in range(nd):
                ia.append(a)
                iap.append(ap)
                idd.append(d)
    ia_t, iap_t, idd_t = (torch.tensor(v, dtype=torch.int64) for v in (ia, iap, idd))
    twok = 1 << k
    uidx = {}
    for x1 in range(2):
        for x2 in range(2):
            c = ia_t + x1 * half + idd_t * (2 * half)
            cp = iap_t + x2 * half + idd_t * (2 * half)
            uidx[(x1, x2)] = c * twok + cp
    return {"ia": ia_t, "iap": iap_t, "id": idd_t, "uidx": uidx}


def round_from_U(U: torch.Tensor, eq_r: torch.Tensor, wd: torch.Tensor, terms: dict, ext_mul_rows) -> torch.Tensor:
    """``Q_i(0..2)`` as ``(3, 1, 6) int32`` plain from ``U (4^k, 6)`` plain, ``eq_r (2^(i-1), 6) = eq(r_1..r_{i-1}, a)``,
    ``wd (2^(k-i), 6) = eq(gamma_i..gamma_{k-1}, d)`` and the constants of :func:`u_round_terms`:
    ``V[x1, x2] = sum_{a, a', d} eq_r[a] eq_r[a'] wd[d] U[(a, x1, d), (a', x2, d)]``, ``Q(0) = V00``, ``Q(1) = V11``,
    ``Q(2) = 4 V11 - 2 V10 - 2 V01 + V00`` (``g(2) = 2 g(1) - g(0)`` on both factors).  Exact field arithmetic
    (``ext_mul_rows`` = the Triton plain ``F_{p^6}`` Hadamard; sums in INT64)."""
    dev = U.device
    ia, iap, idd = (terms[n].to(dev) for n in ("ia", "iap", "id"))
    coef = ext_mul_rows(ext_mul_rows(eq_r[ia], eq_r[iap]), wd[idd])                    # (terms, 6)
    V = {}
    for key, ui in terms["uidx"].items():
        V[key] = (ext_mul_rows(U[ui.to(dev)], coef).to(torch.int64).sum(0) % P).to(torch.int32)
    q0 = V[(0, 0)]
    q1 = V[(1, 1)]
    v11_4 = _add(_add(q1, q1), _add(q1, q1))
    cross = _add(_add(V[(1, 0)], V[(1, 0)]), _add(V[(0, 1)], V[(0, 1)]))
    q2 = _add(_sub(v11_4, cross), q0)
    return torch.stack([q0, q1, q2], 0).view(3, 1, 6)


def _add(a, b):
    s = (a - P) + b
    return torch.where(s < 0, s + P, s)


def _sub(a, b):
    d = a - b
    return torch.where(d < 0, d + P, d)

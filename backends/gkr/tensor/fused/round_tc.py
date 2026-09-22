"""Fused degree-3 round with the fold on INT8 tensor cores (``tl.dot``), everything else as in ``round_simt``.

The fold ``(1 - r) x_e + r x_o`` of a pair is TensorZKP's vector-scalar product: the 48 limbs of the pair (2 elements x 6
coefficients x 4 limbs) times the 48 x 48 convolution matrix ``[C_(1-r); C_r]`` (``sumcheck_int8.scalar_matrix``, ``X^6 = W``
applied inside the matrix) gives the 48 limb-position sums of the folded element, recomposed in registers.  What is
new against the previous lane's composition:

* **no limb tensors in HBM**: the little-endian bytes of a u32 coefficient *are* its unsigned 8-bit limbs, so the
  ``[BLOCK, 64]`` int8 operand tile is loaded straight from the table's bytes (48 useful columns); the unsigned -> signed
  mismatch of ``tl.dot`` (int8 x int8) is fixed by flipping the top bit (``x - 128``) and adding the constant row
  ``128 * colsum(C)`` to the INT32 result -- one dot, no second correction dot;
* **recomposition in registers**: ``sum_m z_m 2^(8m) mod p`` as int64 multiply-adds followed by one 64-bit Montgomery
  reduction (the matrix carries the challenge in Montgomery form, so the reduction lands on the plain value), then the
  six coefficient columns are peeled off with ``tl.split`` and written as 24 B per element;
* the round polynomial is evaluated on the folded coefficients with the SIMT extension arithmetic of ``round_simt``
  (the ``g o h`` Hadamards are not a matrix product in any layout; a tensor-core inner product for ``<f, g o h>`` would
  add 4 SIMT Hadamards + 96-column limb decompositions per pair to save 4 of the 8 SIMT products: not built, see README).

Constraints met on sm_89 / Triton 3.2: ``tl.dot`` needs ``M >= 16`` (``BLOCK >= 16``), K and N multiples of 16 (the 48
limb columns are padded to 64: 4096 MACs performed per folded element for 2304 useful); INT32 accumulation is far
inside its bound (48 products of |.| <= 2^14 per entry).  Extension-field input only (the base -> extension fold of
round 2 is 8 x 48 limbs: not built; ``supports`` says so and the SIMT kernel runs that round).
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from .. import reference as ref
from .. import sumcheck_int8 as legacy
from . import field
from .round_simt import DEG, FusedRound, FusedTable, eval_ext, pairs_of

PRIME = tl.constexpr(field.P)
MU = tl.constexpr(field.MU_I32)
BIG = tl.constexpr(field.P * (1 << 23))         # added before the 64-bit Montgomery reduction so the operand is positive


@triton.jit
def mont_reduce64(y):
    """``y R^-1 mod p`` for int64 ``0 <= y < p 2^32``."""
    lo = y.to(tl.int32)
    hi = (y >> 32).to(tl.int32)
    m = lo * MU
    u = tl.umulhi(m, PRIME)
    r = hi - u
    return tl.where(r < 0, r + PRIME, r)


@triton.jit
def tc_fold(bptr, out_ptr, idx, m, n_in, n_out, C, corr, pow2, BLOCK: tl.constexpr):
    """Fold ``BLOCK`` pairs of one extension table on tensor cores; stores and returns the 6 folded coefficients."""
    cols = tl.arange(0, 64)
    s_of = cols // 8                                            # coefficient of the limb column
    w_of = cols % 8                                             # byte within the 8-byte (even, odd) pair of that coefficient
    off = s_of[None, :] * (n_in * 4) + idx[:, None] * 8 + w_of[None, :]
    xb = tl.load(bptr + off, mask=m[:, None] & (cols[None, :] < 48), other=0)      # [BLOCK, 64] int8: the u32 bytes
    xs = (xb.to(tl.int16) ^ 128).to(tl.int8)                                       # unsigned limb - 128 in [-128, 127]
    z = tl.dot(xs, C, out_dtype=tl.int32) + corr[None, :]                          # [BLOCK, 64] limb-position sums
    z3 = tl.reshape(z, (BLOCK, 8, 8))
    y = tl.sum(z3.to(tl.int64) * pow2[None, None, :], axis=2) + BIG                # [BLOCK, 8] int64 (< 2^55)
    r = mont_reduce64(y)                                                           # [BLOCK, 8] int32 in [0, p)
    a, b = tl.split(tl.reshape(r, (BLOCK, 4, 2)))                                  # a: cols 0 2 4 6, b: 1 3 5 7
    a0, a1 = tl.split(tl.reshape(a, (BLOCK, 2, 2)))                                # a0: 0 4, a1: 2 6
    b0, b1 = tl.split(tl.reshape(b, (BLOCK, 2, 2)))                                # b0: 1 5, b1: 3 7
    x0, x4 = tl.split(a0)
    x2, _ = tl.split(a1)
    x1, x5 = tl.split(b0)
    x3, _ = tl.split(b1)
    tl.store(out_ptr + 0 * n_out + idx, x0, mask=m)
    tl.store(out_ptr + 1 * n_out + idx, x1, mask=m)
    tl.store(out_ptr + 2 * n_out + idx, x2, mask=m)
    tl.store(out_ptr + 3 * n_out + idx, x3, mask=m)
    tl.store(out_ptr + 4 * n_out + idx, x4, mask=m)
    tl.store(out_ptr + 5 * n_out + idx, x5, mask=m)
    return x0, x1, x2, x3, x4, x5


@triton.jit
def tc_round_kernel(f_ptr, g_ptr, h_ptr, fo_ptr, go_ptr, ho_ptr, C_ptr, corr_ptr, pow2_ptr, s_ptr, n_in, n_out,
                    EVAL: tl.constexpr, RELOAD: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_out
    rows = tl.arange(0, 64)
    C = tl.load(C_ptr + rows[:, None] * 64 + rows[None, :])                       # [64, 64] int8
    corr = tl.load(corr_ptr + rows)                                                # [64] int32
    pow2 = tl.load(pow2_ptr + tl.arange(0, 8))                                     # [8] int64
    f0, f1, f2, f3, f4, f5 = tc_fold(f_ptr, fo_ptr, idx, m, n_in, n_out, C, corr, pow2, BLOCK)
    g0, g1, g2, g3, g4, g5 = tc_fold(g_ptr, go_ptr, idx, m, n_in, n_out, C, corr, pow2, BLOCK)
    h0, h1, h2, h3, h4, h5 = tc_fold(h_ptr, ho_ptr, idx, m, n_in, n_out, C, corr, pow2, BLOCK)
    if EVAL and RELOAD:
        # the dot's MMA layout makes the SIMT evaluation replicate work; re-read the just-written coefficients
        # (this program's own rows, L1/L2-hot) to get them in a plain blocked layout
        tl.debug_barrier()
        f0 = tl.load(fo_ptr + 0 * n_out + idx, mask=m, other=0)
        f1 = tl.load(fo_ptr + 1 * n_out + idx, mask=m, other=0)
        f2 = tl.load(fo_ptr + 2 * n_out + idx, mask=m, other=0)
        f3 = tl.load(fo_ptr + 3 * n_out + idx, mask=m, other=0)
        f4 = tl.load(fo_ptr + 4 * n_out + idx, mask=m, other=0)
        f5 = tl.load(fo_ptr + 5 * n_out + idx, mask=m, other=0)
        g0 = tl.load(go_ptr + 0 * n_out + idx, mask=m, other=0)
        g1 = tl.load(go_ptr + 1 * n_out + idx, mask=m, other=0)
        g2 = tl.load(go_ptr + 2 * n_out + idx, mask=m, other=0)
        g3 = tl.load(go_ptr + 3 * n_out + idx, mask=m, other=0)
        g4 = tl.load(go_ptr + 4 * n_out + idx, mask=m, other=0)
        g5 = tl.load(go_ptr + 5 * n_out + idx, mask=m, other=0)
        h0 = tl.load(ho_ptr + 0 * n_out + idx, mask=m, other=0)
        h1 = tl.load(ho_ptr + 1 * n_out + idx, mask=m, other=0)
        h2 = tl.load(ho_ptr + 2 * n_out + idx, mask=m, other=0)
        h3 = tl.load(ho_ptr + 3 * n_out + idx, mask=m, other=0)
        h4 = tl.load(ho_ptr + 4 * n_out + idx, mask=m, other=0)
        h5 = tl.load(ho_ptr + 5 * n_out + idx, mask=m, other=0)
    if EVAL:
        fa0, fb0 = pairs_of(f0, BLOCK // 2)
        fa1, fb1 = pairs_of(f1, BLOCK // 2)
        fa2, fb2 = pairs_of(f2, BLOCK // 2)
        fa3, fb3 = pairs_of(f3, BLOCK // 2)
        fa4, fb4 = pairs_of(f4, BLOCK // 2)
        fa5, fb5 = pairs_of(f5, BLOCK // 2)
        ga0, gb0 = pairs_of(g0, BLOCK // 2)
        ga1, gb1 = pairs_of(g1, BLOCK // 2)
        ga2, gb2 = pairs_of(g2, BLOCK // 2)
        ga3, gb3 = pairs_of(g3, BLOCK // 2)
        ga4, gb4 = pairs_of(g4, BLOCK // 2)
        ga5, gb5 = pairs_of(g5, BLOCK // 2)
        ha0, hb0 = pairs_of(h0, BLOCK // 2)
        ha1, hb1 = pairs_of(h1, BLOCK // 2)
        ha2, hb2 = pairs_of(h2, BLOCK // 2)
        ha3, hb3 = pairs_of(h3, BLOCK // 2)
        ha4, hb4 = pairs_of(h4, BLOCK // 2)
        ha5, hb5 = pairs_of(h5, BLOCK // 2)
        eval_ext(s_ptr + pid * 24, fa0, fa1, fa2, fa3, fa4, fa5, fb0, fb1, fb2, fb3, fb4, fb5,
                 ga0, ga1, ga2, ga3, ga4, ga5, gb0, gb1, gb2, gb3, gb4, gb5,
                 ha0, ha1, ha2, ha3, ha4, ha5, hb0, hb1, hb2, hb3, hb4, hb5)


# ---------------------------------------------------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------------------------------------------------


def fold_operands(r: ref.Ext, device):
    """``(C (64, 64) int8, corr (64,) int32, pow2 (8,) int64)`` for the fold with ``r``: rows of ``C`` are indexed by the
    kernel's limb column ``8 s + 4 e + j`` (coefficient ``s``, even/odd ``e``, limb ``j``); the scalars are ``(1 - r) R`` and
    ``r R`` (Montgomery form) so the in-kernel reduction yields the plain fold; ``corr = 128 colsum(C)`` undoes the
    ``x - 128`` limb shift."""
    one_minus_r = ref.sub(ref.ONE, r)
    Ce = legacy.scalar_matrix(field.to_mont(one_minus_r), DEG, True)          # (24, 48), row 4 s + j
    Co = legacy.scalar_matrix(field.to_mont(r), DEG, True)
    C = np.zeros((64, 64), dtype=np.int8)
    for s in range(DEG):
        for j in range(4):
            C[8 * s + j, :48] = Ce[4 * s + j]
            C[8 * s + 4 + j, :48] = Co[4 * s + j]
    corr = 128 * C.astype(np.int32).sum(0)
    pow2 = np.array([pow(2, 8 * mm, field.P) for mm in range(8)], dtype=np.int64)
    return (torch.from_numpy(C).to(device), torch.from_numpy(corr.astype(np.int32)).to(device), torch.from_numpy(pow2).to(device))


class FusedRoundTC:
    name = "tc"

    def __init__(self, device, block: int = 32, num_warps: int = 2, reload: bool = True):
        if block < 16:
            raise ValueError("tl.dot needs M >= 16")
        self.device = device
        self.block = block
        self.num_warps = num_warps
        self.reload = reload

    def alloc_out(self, n_out: int):
        return torch.empty((DEG, n_out), dtype=torch.int32, device=self.device)

    def grid_size(self, n_in: int, fold: bool) -> int:
        return triton.cdiv(n_in // 2, self.block)

    @staticmethod
    def supports(deg_in: int, mode: str) -> bool:
        return deg_in == DEG and mode in ("fold+eval", "fold")

    @staticmethod
    def macs_performed(n_in: int, deg_in: int, mode: str) -> int:
        return 3 * (n_in // 2) * 64 * 64

    def launch(self, f: FusedTable, g: FusedTable, h: FusedTable, r_ops, eval: bool = True, outs=None, s_buf=None):
        """``r_ops`` = ``fold_operands(r)`` (the fold is mandatory here).  Same return shape as ``FusedRound.launch``."""
        n_in = f.t
        if r_ops is None or f.deg != DEG or g.t != n_in or h.t != n_in or n_in % 2:
            raise ValueError("TC round: extension tables of equal even length, with a challenge")
        C, corr, pow2 = r_ops
        n_out = n_in // 2
        grid = (self.grid_size(n_in, True),)
        if outs is None:
            outs = [self.alloc_out(n_out) for _ in range(3)]
        if s_buf is None:
            s_buf = torch.zeros((grid[0] if eval else 1, 24), dtype=torch.int64, device=self.device)
        tc_round_kernel[grid](f.coeffs.view(torch.int8), g.coeffs.view(torch.int8), h.coeffs.view(torch.int8), outs[0], outs[1], outs[2],
                              C, corr, pow2, s_buf, n_in, n_out, EVAL=eval, RELOAD=self.reload, BLOCK=self.block, num_warps=self.num_warps)
        return (s_buf if eval else None), [FusedTable(o) for o in outs]

    finish_s = staticmethod(FusedRound.finish_s)

    def challenge(self, r: ref.Ext):
        return fold_operands(r, self.device)

    def round(self, f, g, h, r_prev: ref.Ext | None, eval: bool = True):
        s, outs = self.launch(f, g, h, self.challenge(r_prev), eval)
        return (self.finish_s(s) if s is not None else None), outs

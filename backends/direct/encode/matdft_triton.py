"""Fused Triton kernel: one program per row, four 64 x 64 two-stage limb-GEMM DFTs in registers (``encode_matdft_int8``)."""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from . import babybear as bb
from . import matdft_ref as mr

P = bb.P


@triton.jit
def _split(x):
    """int32 tile in [0, p) -> four balanced int8 limb tiles."""
    c = x
    s0 = ((c + 128) & 255) - 128
    c = (c - s0) >> 8
    s1 = ((c + 128) & 255) - 128
    c = (c - s1) >> 8
    s2 = ((c + 128) & 255) - 128
    c = (c - s2) >> 8
    return s0.to(tl.int8), s1.to(tl.int8), s2.to(tl.int8), c.to(tl.int8)


@triton.jit
def _limb_dot(f0, f1, f2, f3, d0, d1, d2, d3, W1: tl.constexpr, W2: tl.constexpr, W3: tl.constexpr,
              W4: tl.constexpr, W5: tl.constexpr, W6: tl.constexpr, PP: tl.constexpr):
    """``(F @ D) mod p`` from limb tiles: 16 INT8 dots, INT64 recombination with 2^(8s) mod p, one reduction -> int32."""
    y0 = tl.dot(f0, d0, out_dtype=tl.int32)
    y1 = tl.dot(f0, d1, out_dtype=tl.int32) + tl.dot(f1, d0, out_dtype=tl.int32)
    y2 = tl.dot(f0, d2, out_dtype=tl.int32) + tl.dot(f1, d1, out_dtype=tl.int32) + tl.dot(f2, d0, out_dtype=tl.int32)
    y3 = (tl.dot(f0, d3, out_dtype=tl.int32) + tl.dot(f1, d2, out_dtype=tl.int32)
          + tl.dot(f2, d1, out_dtype=tl.int32) + tl.dot(f3, d0, out_dtype=tl.int32))
    y4 = tl.dot(f1, d3, out_dtype=tl.int32) + tl.dot(f2, d2, out_dtype=tl.int32) + tl.dot(f3, d1, out_dtype=tl.int32)
    y5 = tl.dot(f2, d3, out_dtype=tl.int32) + tl.dot(f3, d2, out_dtype=tl.int32)
    y6 = tl.dot(f3, d3, out_dtype=tl.int32)
    acc = (y0.to(tl.int64) + y1.to(tl.int64) * W1 + y2.to(tl.int64) * W2 + y3.to(tl.int64) * W3
           + y4.to(tl.int64) * W4 + y5.to(tl.int64) * W5 + y6.to(tl.int64) * W6)
    r = acc % PP
    r = tl.where(r < 0, r + PP, r)
    return r.to(tl.int32)


@triton.jit
def _dft(d, f0, f1, f2, f3, t, W1: tl.constexpr, W2: tl.constexpr, W3: tl.constexpr, W4: tl.constexpr,
         W5: tl.constexpr, W6: tl.constexpr, PP: tl.constexpr):
    """Two-stage size-N^2 DFT of the row held as tile ``d[n2, n1]``; returns ``X[k1, k2]`` (natural order)."""
    d0, d1, d2, d3 = _split(d)
    at = _limb_dot(f0, f1, f2, f3, d0, d1, d2, d3, W1, W2, W3, W4, W5, W6, PP)      # A^T[k2, n1]
    bt = (at.to(tl.int64) * t.to(tl.int64)) % PP                                    # twiddle w^(n1 k2), symmetric
    b = tl.trans(bt.to(tl.int32))                                                   # B[n1, k2]
    b0, b1, b2, b3 = _split(b)
    return _limb_dot(f0, f1, f2, f3, b0, b1, b2, b3, W1, W2, W3, W4, W5, W6, PP)    # X[k1, k2]


@triton.jit
def rs_matdft_kernel(in_ptr, out_ptr, finv_ptr, ffwd_ptr, tinv_ptr, tfwd_ptr, shift_ptr, rows,
                     N: tl.constexpr, K: tl.constexpr, W1: tl.constexpr, W2: tl.constexpr, W3: tl.constexpr,
                     W4: tl.constexpr, W5: tl.constexpr, W6: tl.constexpr, PP: tl.constexpr, WRITE0: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.arange(0, N)[:, None] * N + tl.arange(0, N)[None, :]
    fi0 = tl.load(finv_ptr + idx)
    fi1 = tl.load(finv_ptr + K + idx)
    fi2 = tl.load(finv_ptr + 2 * K + idx)
    fi3 = tl.load(finv_ptr + 3 * K + idx)
    tinv = tl.load(tinv_ptr + idx)
    d = tl.load(in_ptr + pid.to(tl.int64) * K + idx)
    if WRITE0:
        tl.store(out_ptr + pid.to(tl.int64) * (4 * K) + idx, d)
    coef = _dft(d, fi0, fi1, fi2, fi3, tinv, W1, W2, W3, W4, W5, W6, PP)
    ff0 = tl.load(ffwd_ptr + idx)
    ff1 = tl.load(ffwd_ptr + K + idx)
    ff2 = tl.load(ffwd_ptr + 2 * K + idx)
    ff3 = tl.load(ffwd_ptr + 3 * K + idx)
    tfwd = tl.load(tfwd_ptr + idx)
    for c in tl.static_range(1, 4):
        sh = tl.load(shift_ptr + (c - 1) * K + idx)
        y = ((coef.to(tl.int64) * sh.to(tl.int64)) % PP).to(tl.int32)
        x = _dft(y, ff0, ff1, ff2, ff3, tfwd, W1, W2, W3, W4, W5, W6, PP)
        tl.store(out_ptr + pid.to(tl.int64) * (4 * K) + c * K + idx, x)


class TritonMatDFT:
    def __init__(self, enc, device, num_warps: int = 8):
        self.enc = enc
        self.num_warps = num_warps
        N, k = enc.N1, enc.k

        def limb_tiles(F):                       # (4, N, N) int8: limb i of F[a, b]
            return torch.from_numpy(np.ascontiguousarray(mr.balanced_limbs(F).transpose(2, 0, 1))).to(device)

        self.finv = limb_tiles(enc.F_inv)
        self.ffwd = limb_tiles(enc.F_fwd)
        self.tinv = torch.from_numpy(enc.T_inv.astype(np.int32)).to(device).contiguous()
        self.tfwd = torch.from_numpy(enc.T_fwd.astype(np.int32)).to(device).contiguous()
        self.shift = torch.from_numpy(enc.shifts.astype(np.int32)).to(device).contiguous()
        self.W = [int(w) for w in mr.LIMB_WEIGHTS_MOD_P]
        self.N, self.k = N, k

    def __call__(self, rows: torch.Tensor, out: torch.Tensor, write_systematic: bool = True) -> torch.Tensor:
        R = rows.shape[0]
        rs_matdft_kernel[(R,)](rows, out, self.finv, self.ffwd, self.tinv, self.tfwd, self.shift, R,
                               N=self.N, K=self.k, W1=self.W[1], W2=self.W[2], W3=self.W[3], W4=self.W[4],
                               W5=self.W[5], W6=self.W[6], PP=P, WRITE0=write_systematic, num_warps=self.num_warps)
        return out

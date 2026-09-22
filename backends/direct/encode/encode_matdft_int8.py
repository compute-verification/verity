"""Two-stage matrix-DFT Reed-Solomon encoder on INT8 tensor cores (``matdft_ref`` for the algorithm).

Two variants of the same arithmetic (four 64 x 64 two-stage DFTs per row at ``k = 4096``, 16 limb GEMMs each):

* ``cublas``: ``torch._int_mm`` (cuBLASLt INT8 -> INT32).  The DFT matrices' limb tensors are decomposed ONCE at
  construction (``F^T`` limbs stacked along N: (64, 256) int8, resident on the device); per call only the data rows are
  decomposed, stacked along M ((4 R 64, 64) int8), multiplied, and recombined from the (4 R 64, 256) INT32 result with
  ``2^(8(i+j)) mod p`` in INT64.  Every intermediate (limbs, INT32 products, transposes, twiddles) is a separate
  torch kernel and a full pass over HBM: this is the "unfused" measurement.
* ``triton``: one Triton program per row keeps the row in registers/shared memory through all four DFTs: the 64 x 64
  int32 tile is limb-split in registers (balanced signed 8-bit digits), ``tl.dot`` against the constant limb tiles of
  the DFT matrix (16 dots, INT32 accumulate), recombined in INT64 and reduced mod p, twiddled, transposed
  (``tl.trans``), split again, dotted again; the coset shift and the write of coset ``c`` follow.  Nothing but the
  ``k`` input and ``n`` output words touches HBM.

Both are bit-exact against ``babybear.encode_rows`` (``tests/``).  The Triton variant is skipped if Triton is missing.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import babybear as bb
from . import matdft_ref as mr

P = bb.P


def _balanced_limbs_torch(x: torch.Tensor) -> torch.Tensor:
    """(..., 4) int8 balanced limbs of int32/int64 ``x`` in [0, p)."""
    cur = x.to(torch.int64)
    outs = []
    for _ in range(4):
        s = ((cur + 128) & 255) - 128
        outs.append(s.to(torch.int8))
        cur = (cur - s) >> 8
    return torch.stack(outs, dim=-1)


def _weights(device) -> torch.Tensor:
    return torch.tensor(mr.LIMB_WEIGHTS_MOD_P, dtype=torch.int64, device=device)


class _Stage:
    """A constant (N x N) DFT matrix ``F`` prepared for ``X @ F``: limbs of ``F`` stacked along N -> (N, 4N) int8."""

    def __init__(self, F: np.ndarray, device):
        self.N = F.shape[0]
        fl = mr.balanced_limbs(F)                                     # (N, N, 4)
        self.limbs = torch.from_numpy(np.ascontiguousarray(fl.transpose(0, 2, 1).reshape(self.N, 4 * self.N))).to(device)  # col block j = limb j
        self.F = torch.from_numpy(F).to(device)

    def matmul(self, X: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``X (M, N) int64 in [0,p)`` -> ``X @ F mod p`` via one INT8 GEMM of shape (4M, N) x (N, 4N)."""
        M = X.shape[0]
        xl = _balanced_limbs_torch(X)                                 # (M, N, 4)
        xs = xl.permute(2, 0, 1).reshape(4 * M, self.N)              # row block i = limb i
        y = torch._int_mm(xs, self.limbs)                             # (4M, 4N) int32
        y = y.view(4, M, 4, self.N).to(torch.int64)                   # [i, m, j, n]: column block j = limb j of F
        acc = torch.zeros(M, self.N, dtype=torch.int64, device=X.device)
        for i in range(4):
            for j in range(4):
                acc += y[i, :, j, :] * weights[i + j]
        return torch.remainder(acc, P)


class RSEncoderMatDFT:
    def __init__(self, k: int = bb.K_CODE, n: int = bb.N_CODE, variant: str = "cublas", device="cuda"):
        self.k, self.n, self.variant, self.device = k, n, variant, device
        self.N1 = 1 << (int(math.log2(k)) // 2)
        self.N2 = k // self.N1
        if self.N1 != self.N2:
            raise ValueError("square two-stage split expected (k = 4^m)")
        N = self.N1
        wn = bb.root_of_unity(n)
        wk = pow(wn, n // k, P)
        wk_inv = pow(wk, -1, P)
        kinv = pow(k, -1, P)
        self.F_fwd = mr.dft_matrix(N, pow(wk, N, P))
        self.F_inv = mr.dft_matrix(N, pow(wk_inv, N, P))
        idx = (np.arange(N)[:, None] * np.arange(N)[None, :]) % k
        self.T_fwd = bb.powers(wk, k).astype(np.int64)[idx]                 # w_k^(n1 k2)
        self.T_inv = bb.powers(wk_inv, k).astype(np.int64)[idx]
        self.shifts = np.stack([(bb.powers(pow(wn, c, P), k).astype(np.int64) * kinv) % P for c in range(1, n // k)])
        self.weights = _weights(device)
        if variant == "cublas":
            self.st_fwd = _Stage(self.F_fwd, device)
            self.st_inv = _Stage(self.F_inv, device)
            self.t_fwd = torch.from_numpy(self.T_fwd).to(device)
            self.t_inv = torch.from_numpy(self.T_inv).to(device)
            self.shift_t = torch.from_numpy(self.shifts).to(device)
        elif variant == "triton":
            from . import matdft_triton as mt

            self.tri = mt.TritonMatDFT(self, device)
        else:
            raise ValueError(variant)

    @staticmethod
    def available() -> list[str]:
        out = ["cublas"]
        try:
            import triton  # noqa: F401

            out.append("triton")
        except ImportError:
            pass
        return out

    # -- cuBLAS path ---------------------------------------------------------------------------------------------------
    def _dft(self, d: torch.Tensor, st: _Stage, T: torch.Tensor) -> torch.Tensor:
        """Row-wise size-k DFT of ``d`` (R, k) int64, natural order (matdft_ref.dft_two_stage)."""
        R = d.shape[0]
        N1, N2 = self.N1, self.N2
        D = d.view(R, N2, N1).transpose(1, 2).reshape(R * N1, N2)
        A = st.matmul(D, self.weights).view(R, N1, N2)
        B = torch.remainder(A * T[None], P)
        Bt = B.transpose(1, 2).reshape(R * N2, N1)
        Xt = st.matmul(Bt, self.weights).view(R, N2, N1)
        return Xt.transpose(1, 2).reshape(R, self.k)

    def __call__(self, rows: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """``rows`` (R, k) int32 (BabyBear residues as bit patterns) -> (R, n) int32 codewords, coset layout."""
        R = rows.shape[0]
        if out is None:
            out = torch.empty(R, self.n, dtype=torch.int32, device=rows.device)
        if self.variant == "triton":
            return self.tri(rows, out)
        d = rows.to(torch.int64)
        out[:, :self.k] = rows
        coef = self._dft(d, self.st_inv, self.t_inv)
        for c in range(1, self.n // self.k):
            y = torch.remainder(coef * self.shift_t[c - 1][None], P)
            out[:, c * self.k:(c + 1) * self.k] = self._dft(y, self.st_fwd, self.t_fwd).to(torch.int32)
        return out

    @staticmethod
    def limb_macs_per_row(k: int) -> int:
        N = 1 << (int(math.log2(k)) // 2)
        return 4 * 2 * k * N * 16

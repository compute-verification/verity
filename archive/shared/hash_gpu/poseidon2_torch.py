"""Poseidon2 / BabyBear as batched torch tensor ops (the permutation of ``poseidon2_ref``, bit-exact).

State: ``int64`` tensor ``(S, W)`` with canonical values in ``[0, P)``.  Every step is an elementwise torch op
(memory-bound: each pass reads and writes the whole batch), except the external linear layer, which has three
implementations selected by ``mds``:

* ``blocks``: Plonky3's shift-and-add form (M4 on every 4-block, then the four block sums), int64, ~12 passes.
* ``fp64``: dense ``x @ M_E^T`` in FP64 -- exact because ``|entries| <= 6`` and ``W <= 24`` keep every sum below
  ``2^38 < 2^53``; the FP64 units of a consumer GPU are 1/64 rate, so this is a correctness baseline.
* ``int8``: the tensor-core route.  ``x`` (< 2^31, stored int32) is viewed as 4 signed bytes ``s_l``; with
  ``c_l = [s_l < 0]`` the unsigned byte is ``s_l + 256 c_l``.  One ``torch._int_mm`` of ``[s | c]`` ``(S, 8W)``
  by the block matrix ``B[(i, l), (j, l)] = M_E[j][i]`` gives ``Y_s, Y_c`` in INT32, and
  ``y_j = sum_l 2^{8l} (Y_s[j, l] + 256 Y_c[j, l]) mod P`` (int64 combine, one reduction).  INT8 x INT8 -> INT32
  MMA, K = 8W = 128 (width 16) or 192 (width 24).

``leaf_tree`` hashes the columns of an ``(R, N)`` matrix (blocks of 8 rows, binary compression tree per column,
rightmost odd block promoted, a lone block compressed with a zero block), ``merkle_level`` compresses adjacent
digests of a ``(count, 8)`` level.  Digest = first 8 lanes of ``permute([left || right || 0])``.
"""
from __future__ import annotations

import torch

from .poseidon2_ref import INTERNAL_DIAG, P, Poseidon2, external_matrix, instance

DIGEST = 8


class Poseidon2Torch:
    def __init__(self, params: Poseidon2 | None = None, device: str | torch.device = "cuda", mds: str = "blocks",
                 chunk_states: int = 1 << 20) -> None:
        self.p = params or instance(16)
        self.W = self.p.width
        self.device = torch.device(device)
        self.mds = mds
        self.chunk_states = chunk_states
        d, W = self.device, self.W
        self.ext_init = torch.tensor(self.p.ext_init, dtype=torch.int64, device=d)
        self.ext_final = torch.tensor(self.p.ext_final, dtype=torch.int64, device=d)
        self.internal = torch.tensor(self.p.internal, dtype=torch.int64, device=d)
        self.diag = torch.tensor(INTERNAL_DIAG[W], dtype=torch.int64, device=d)
        me = external_matrix(W)
        self.me_t_f64 = torch.tensor(me, dtype=torch.float64, device=d).t().contiguous()
        # int8 block matrix: rows (i, l) -> cols (j, l) with value M_E[j][i]; twice (s part, c part)
        b = torch.zeros((8 * W, 8 * W), dtype=torch.int8, device=d)
        for i in range(W):
            for j in range(W):
                for l in range(4):
                    b[i * 4 + l, j * 4 + l] = me[j][i]
                    b[4 * W + i * 4 + l, 4 * W + j * 4 + l] = me[j][i]
        self.b_int8 = b
        self.pow256 = torch.tensor([1, 256, 65536, 1 << 24], dtype=torch.int64, device=d)
        # sm_90's cuBLASLt INT8 GEMM (torch 2.6 / cu124) takes only row counts that are multiples of 32
        self.int_mm_rows = 32 if d.type == "cuda" and torch.cuda.get_device_capability(d)[0] >= 9 else 1

    # -- layers ------------------------------------------------------------------------------------------------
    @staticmethod
    def _sbox(x: torch.Tensor) -> torch.Tensor:
        x2 = torch.remainder(x * x, P)
        x4 = torch.remainder(x2 * x2, P)
        x6 = torch.remainder(x4 * x2, P)
        return torch.remainder(x6 * x, P)

    def _mds_blocks(self, s: torch.Tensor) -> torch.Tensor:
        S, W = s.shape
        t = s.view(S, W // 4, 4)
        a, b, c, d = t[..., 0], t[..., 1], t[..., 2], t[..., 3]
        s0123 = a + b + c + d
        y = torch.stack((s0123 + a + 2 * b, s0123 + b + 2 * c, s0123 + c + 2 * d, s0123 + d + 2 * a), dim=-1)
        sums = y.sum(dim=1, keepdim=True)
        return torch.remainder((y + sums).view(S, W), P)

    def _mds_fp64(self, s: torch.Tensor) -> torch.Tensor:
        return torch.remainder((s.to(torch.float64) @ self.me_t_f64).to(torch.int64), P)

    def _mds_int8(self, s: torch.Tensor) -> torch.Tensor:
        S, W = s.shape
        if S < 17:                                                          # _int_mm needs > 16 rows
            return self._mds_int8(torch.cat((s, torch.zeros((17 - S, W), dtype=s.dtype, device=s.device))))[:S]
        s8 = s.contiguous().to(torch.int32).view(torch.int8).view(S, 4 * W)     # little-endian bytes of each lane
        c8 = (s8 < 0).to(torch.int8)
        a = torch.cat((s8, c8), dim=1)                                      # (S, 8W) int8
        y = torch._int_mm(a, self.b_int8)                                   # (S, 8W) int32
        ys = y[:, : 4 * W].reshape(S, W, 4).to(torch.int64)
        yc = y[:, 4 * W:].reshape(S, W, 4).to(torch.int64)
        comb = ((ys + 256 * yc) * self.pow256).sum(dim=-1)
        return torch.remainder(comb, P)

    def _mds(self, s: torch.Tensor) -> torch.Tensor:
        if self.mds == "blocks":
            return self._mds_blocks(s)
        if self.mds == "fp64":
            return self._mds_fp64(s)
        if self.mds == "int8":
            return self._mds_int8(s)
        raise ValueError(self.mds)

    def _internal(self, s: torch.Tensor) -> torch.Tensor:
        total = s.sum(dim=1, keepdim=True)
        return torch.remainder(total + s * self.diag, P)

    def _permute_chunk(self, s: torch.Tensor) -> torch.Tensor:
        pad = -s.shape[0] % self.int_mm_rows if self.mds == "int8" else 0
        if pad:
            return self._permute_chunk(torch.cat((s, s.new_zeros((pad, s.shape[1])))))[:-pad]
        s = self._mds(s)
        for rc in self.ext_init:
            s = self._mds(self._sbox(torch.remainder(s + rc, P)))
        for c in self.internal:
            s0 = self._sbox(torch.remainder(s[:, 0] + c, P))
            s = torch.cat((s0[:, None], s[:, 1:]), dim=1)
            s = self._internal(s)
        for rc in self.ext_final:
            s = self._mds(self._sbox(torch.remainder(s + rc, P)))
        return s

    def permute(self, states: torch.Tensor) -> torch.Tensor:
        """``(S, W)`` int64 -> ``(S, W)`` int64; processed in ``chunk_states`` slices."""
        if states.dtype != torch.int64 or states.shape[1] != self.W:
            raise ValueError("states must be int64 (S, W)")
        out = torch.empty_like(states)
        for i in range(0, states.shape[0], self.chunk_states):
            out[i:i + self.chunk_states] = self._permute_chunk(states[i:i + self.chunk_states])
        return out

    def compress(self, pairs: torch.Tensor) -> torch.Tensor:
        """``(S, 16)`` -> ``(S, 8)``: the 2:1 compression (zero-padded to the width)."""
        S = pairs.shape[0]
        if self.W > 16:
            pairs = torch.cat((pairs, torch.zeros((S, self.W - 16), dtype=torch.int64, device=pairs.device)), 1)
        return self.permute(pairs)[:, :DIGEST]

    # -- trees ---------------------------------------------------------------------------------------------------
    def leaf_tree(self, m: torch.Tensor) -> torch.Tensor:
        """Column leaves of an ``(R, N)`` matrix of BabyBear elements (int32/uint32/int64 values in ``[0, P)``):
        returns ``(N, 8)`` int64 digests."""
        R, N = m.shape
        m = m.to(torch.int64)
        pad = -R % DIGEST
        if pad:
            m = torch.cat((m, torch.zeros((pad, N), dtype=torch.int64, device=m.device)), 0)
        count = m.shape[0] // DIGEST
        if count == 1:
            m = torch.cat((m, torch.zeros((DIGEST, N), dtype=torch.int64, device=m.device)), 0)
            count = 2
        level = m                                             # (count * 8, N)
        while count > 1:
            npairs = count // 2
            # states for (pair k, column x): rows [16k, 16k + 16) of column x
            states = level[: npairs * 16].view(npairs, 16, N).permute(0, 2, 1).reshape(npairs * N, 16)
            dig = self.compress(states).view(npairs, N, DIGEST).permute(0, 2, 1).reshape(npairs * DIGEST, N)
            if count % 2:
                dig = torch.cat((dig, level[npairs * 16:]), 0)
            level, count = dig, (count + 1) // 2
        return level.t().contiguous()                          # (N, 8)

    def merkle_level(self, level: torch.Tensor) -> torch.Tensor:
        """``(count, 8)`` -> ``((count + 1) // 2, 8)`` with the odd last digest promoted."""
        count = level.shape[0]
        npairs = count // 2
        out = self.compress(level[: npairs * 2].reshape(npairs, 16))
        if count % 2:
            out = torch.cat((out, level[-1:]), 0)
        return out


def tree_states_per_column(rows: int) -> int:
    """Number of permutations the column tree spends per column of ``rows`` elements."""
    count = max(2, -(-rows // DIGEST))
    total = 0
    while count > 1:
        total += count // 2
        count = (count + 1) // 2
    return total

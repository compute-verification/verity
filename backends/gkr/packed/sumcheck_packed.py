"""Candidate A's eq-weighted degree-3 GKR sumcheck over BabyBear^6 as INT8 tensor-core matmuls, with ``k`` base-field
("packed") rounds first.  torch; CUDA or CPU (the CPU path is the laptop test path).

Algorithm: ``reference.packed_prover`` (DESIGN.md 3).  Layout: TensorZKP 4.1 with balanced signed limbs (lane
``tc-sumcheck``'s convention, ``reference.limbs_balanced``): a coefficient is 4 INT8 limbs ``l_j in [-128, 128)``,
an ``F_{p^6}`` element 24 limbs, column ``(s, j) = 4 s + j``.  A limb product is ``< 2^14`` so an INT32 accumulator
holds ``< 2^17`` products; the inner-product depth is ``K1 <= 32768`` (the paper's chunk) and the rest of the sum is
finished in the field.

The eq weight ``eq(gamma_{>=i}, b)`` is a tensor product, so the weighted inner product over ``b = (b2, b1)`` is done as
ONE well-shaped matmul over ``b1`` (``M = rows x 4 limbs``, ``K = K1``, ``N = 24``) followed by a short extension-field
contraction over ``b2`` (``K2 = m / K1`` terms): the weights never have to be folded, and the skinny ``M x K x 24``
split-K problem of the plain layout does not arise.

Per round (``m = t / 2^i`` surviving hypercube points):

* packed round ``i <= k`` (:meth:`Kernel.packed_round`): SIMT forms the ``3 . 4^(i-1)`` base x base Hadamards
  ``g(a, x, b) h(a', x, b) mod p`` (one INT64 multiply and one reduction each) and their 4 limbs; tensor cores do
  ``A^T x E_lo`` (``4 x 24 = 96`` MACs per Hadamard element); SIMT reduces the ``4 x 24`` limb block per output and
  contracts over ``b2``.
* extension round (:meth:`Kernel.ext_round`): SIMT forms 3 ``F_{p^6}`` Hadamards per pair (36 coefficient products,
  ``ext_mul``) and 24 limbs each; tensor cores do the ``24 x 24 = 576``-MAC weighted inner product and the fold
  ``[g_e | g_o] x [C_(1-r); C_r]`` (``48 x 48`` per pair, TensorZKP eq. 14 with ``X^6 = W`` inside the matrix).
* the one fold after the packed rounds (:meth:`Kernel.fold_k`): ``limbs(g) (m x 4.2^k) x C (4.2^k x 48)`` with
  ``C`` the stacked convolution matrices of ``eq(r_1..r_k, a)``: base x extension, ``192`` MACs per input element.

Every matmul is ``torch._int_mm`` (cuBLAS INT8, INT32 accumulate) on CUDA and an exact INT64 matmul on CPU.  A round
can be captured in a CUDA graph (``Prover.capture``) so the measured time is device time, not launch overhead;
``torch.compile`` of the SIMT passes is optional (``compile=True``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import reference as ref

P = ref.P
W = ref.W
DEGREE = ref.DEGREE
LIMBS = ref.LIMBS_PER_BASE
DL = DEGREE * LIMBS
POW2 = [pow(2, 8 * m, P) for m in range(2 * LIMBS)]
MIN_DIM = 32            # torch._int_mm: M > 16, K > 16, K and N multiples of 8; we pad to 32
K1_LOG2_DEFAULT = 13    # inner-product depth 2^13 = 8192 (<= 32768; balanced limbs allow < 2^17)


# ---------------------------------------------------------------------------------------------------------------------
# field passes (SIMT, elementwise)
# ---------------------------------------------------------------------------------------------------------------------


def add_mod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(a + b) mod p`` for int32 tensors in ``[0, p)`` without leaving int32 (``a + b`` could exceed 2^31)."""
    s = (a - P) + b
    return torch.where(s < 0, s + P, s)


def sub_mod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = a - b
    return torch.where(d < 0, d + P, d)


def ext_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Broadcast Hadamard in ``F_{p^6}``: ``(..., 6) x (..., 6) -> (..., 6)`` int32, 36 coefficient products (INT64,
    reduced each), ``X^6 = W``."""
    a64 = a.to(torch.int64)
    b64 = b.to(torch.int64)
    acc: list[Any] = [None] * DEGREE
    for i in range(DEGREE):
        ai = a64[..., i]
        for j in range(DEGREE):
            prod = (ai * b64[..., j]) % P
            d = i + j
            if d >= DEGREE:
                prod = (prod * W) % P
                d -= DEGREE
            acc[d] = prod if acc[d] is None else acc[d] + prod
    return torch.stack([x % P for x in acc], dim=-1).to(torch.int32)


def ext_mul_base(a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``(..., 6) x (...) base -> (..., 6)``."""
    return ((a.to(torch.int64) * s.to(torch.int64)[..., None]) % P).to(torch.int32)


def limbs_of(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The 4 balanced INT8 limbs of an int32 tensor in ``[0, p)`` (each the same shape as ``x``)."""
    v = x
    out = []
    for _ in range(LIMBS):
        l = ((v + 128) & 255) - 128
        out.append(l.to(torch.int8))
        v = (v - l) >> 8
    return tuple(out)  # type: ignore[return-value]


def decompose(coeffs: torch.Tensor) -> torch.Tensor:
    """``(..., deg) int32 -> (..., 4 deg) int8``, column ``4 s + j``."""
    ls = limbs_of(coeffs)
    return torch.stack(ls, dim=-1).reshape(*coeffs.shape[:-1], coeffs.shape[-1] * LIMBS)


def decompose_rows(x: torch.Tensor) -> torch.Tensor:
    """``(..., K) int32 -> (..., 4, K) int8``: limb index before the contraction axis (matmul rows ``(.., j)``)."""
    return torch.stack(limbs_of(x), dim=-2)


def recompose(z: torch.Tensor, deg_out: int, pow2_row: torch.Tensor) -> torch.Tensor:
    """Fold output ``(n, 8 deg_out) int32`` (coefficient block ``d`` = columns ``8d..8d+7``) -> ``(n, deg_out)`` mod p."""
    n = z.shape[0]
    z64 = z.to(torch.int64).reshape(n, deg_out, 2 * LIMBS)
    return (((z64 * pow2_row) % P).sum(-1) % P).to(torch.int32)


def to_dev(v: ref.Ext | list, device) -> torch.Tensor:
    return torch.tensor(np.asarray(v, dtype=np.int64), dtype=torch.int32, device=device)


def eq_table_dev(gammas: list[ref.Ext], device) -> torch.Tensor:
    """``eq(gammas, b)`` for ``b in {0,1}^m`` as ``(2^m, 6) int32`` (bit ``j`` of ``b`` = variable ``j``), by doubling."""
    tbl = to_dev([ref.ONE], device)
    for gm in gammas:
        g = to_dev([gm], device)
        one_m = to_dev([ref.sub(ref.ONE, gm)], device)
        tbl = torch.cat([ext_mul(tbl, one_m), ext_mul(tbl, g)], dim=0)
    return tbl


def scalar_matrix(c: ref.Ext, in_deg: int) -> np.ndarray:
    """``C_c``: ``(4 in_deg) x 48`` int8 with ``limbs(v) @ C_c`` = the unreduced limb-position sums of ``c v`` in 6
    coefficient blocks of 8 limb positions (``X^6 = W`` applied inside: wrapped terms use the limbs of ``W c_t mod p``).
    An output entry sums at most ``2 in_deg`` limb products."""
    C = np.zeros((in_deg * LIMBS, DEGREE * 2 * LIMBS), dtype=np.int8)
    limbs_c = [ref.limbs_balanced(ct) for ct in c]
    limbs_wc = [ref.limbs_balanced(W * ct % P) for ct in c]
    for s in range(in_deg):
        for t in range(DEGREE):
            d = s + t
            lc = limbs_c[t]
            if d >= DEGREE:
                d -= DEGREE
                lc = limbs_wc[t]
            for j in range(LIMBS):
                for jp in range(LIMBS):
                    C[s * LIMBS + j, d * 2 * LIMBS + j + jp] = lc[jp]
    return C


def fold_matrix_ext(r: ref.Ext) -> np.ndarray:
    """``[C_(1-r); C_r]`` (48 x 48) for the extension-table fold ``(1 - r) x_e + r x_o``."""
    return np.concatenate([scalar_matrix(ref.sub(ref.ONE, r), DEGREE), scalar_matrix(r, DEGREE)], axis=0)


def fold_matrix_k(rs: list[ref.Ext]) -> np.ndarray:
    """``(4 . 2^k) x 48``: row block ``a`` is the convolution matrix of ``eq(r_1..r_k, a)`` (base input)."""
    eq_r = ref.eq_table(rs)
    return np.concatenate([scalar_matrix(c, 1) for c in eq_r], axis=0)


# ---------------------------------------------------------------------------------------------------------------------
# the matmul
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Stats:
    macs_performed: int = 0     # M K N of the matmuls as issued (the cost model's unit)
    macs_padded: int = 0        # what the device multiplied (padding to _int_mm's minimum shapes)
    hadamard_base: int = 0      # base x base products on SIMT
    hadamard_ext: int = 0       # F_{p^6} x F_{p^6} products on SIMT (36 coefficient products each)
    matmuls: list = field(default_factory=list)   # (label, M, K, N)

    def add(self, label: str, M: int, K: int, N: int, padded: int) -> None:
        self.macs_performed += M * K * N
        self.macs_padded += padded
        self.matmuls.append((label, M, K, N))


def mm_int8(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, int]:
    """``a (M, K) int8 @ b (K, N) int8 -> (M, N) int32`` exactly; CUDA: ``torch._int_mm`` on operands padded to its
    minimum shapes (padding counted); CPU: INT64 matmul."""
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError("shape mismatch")
    if a.is_cuda:
        # cuBLASLt IMMA on the H100 (torch 2.6 / cu124) rejects M not a multiple of 32 (probed: 48, 144, 1000 fail; 32k ok)
        Mp, Kp, Np = max(MIN_DIM, -(-M // 32) * 32), max(MIN_DIM, -(-K // 8) * 8), max(MIN_DIM, -(-N // 8) * 8)
        if (Mp, Kp, Np) != (M, K, N):
            ap = torch.zeros((Mp, Kp), dtype=torch.int8, device=a.device)
            ap[:M, :K] = a
            bp = torch.zeros((Kp, Np), dtype=torch.int8, device=b.device)
            bp[:K, :N] = b
            return torch._int_mm(ap, bp)[:M, :N], Mp * Kp * Np
        return torch._int_mm(a, b), M * K * N
    return torch.mm(a.to(torch.int64), b.to(torch.int64)).to(torch.int32), M * K * N


# ---------------------------------------------------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class RoundInputs:
    """Device tensors a round consumes (per-round: the eq weights, the challenge coefficients, the prefix)."""

    e_lo_limbs: torch.Tensor      # (K1, 24) int8: limbs of eq(gamma_{i..i+kappa-1}, b1)
    e_hi: torch.Tensor            # (K2, 6) int32: eq(gamma_{i+kappa..n-1}, b2)
    eq_x: torch.Tensor            # (4, 6) int32: eq(gamma_{i-1}, x), x = 0..3
    prefix: torch.Tensor          # (6,) int32: eq(gamma_{<i-1}, r_{<i})
    coeff: torch.Tensor | None    # packed rounds: (4^(i-1), 6) int32 = eq(r_{<i}, a) eq(r_{<i}, a'), index a . 2^(i-1) + a'
    fold_C: torch.Tensor | None   # ext rounds: (48, 48) int8 fold matrix of r_i; fold_k: (4 . 2^k, 48)
    fold_coeff: torch.Tensor | None = None   # triton fold_k: (2^k, 6) int32 = eq(r_1..r_k, a)
    fold_C64: torch.Tensor | None = None     # triton fold_ext: (64, 64) int8 embedding of fold_C


class Kernel:
    """``mode``: ``"eager"`` (torch ops, the reference path), ``"compile"`` (torch.compile of the Hadamards) or
    ``"triton"`` (fused kernels from :mod:`kernels_triton` for the Hadamards and both folds; CUDA only)."""

    def __init__(self, device: str = "cpu", k1_log2: int = K1_LOG2_DEFAULT, compile: bool = False, mode: str | None = None):
        self.device = torch.device(device)
        self.k1_log2 = k1_log2
        self.mode = mode or ("compile" if compile else "eager")
        self.pow2_row = torch.tensor(POW2, dtype=torch.int64, device=self.device)
        pm = np.array([[POW2[j + jp] for jp in range(LIMBS)] for j in range(LIMBS)], dtype=np.int64)
        self.pow2_pair = torch.tensor(pm, dtype=torch.int64, device=self.device)   # [j, j'] -> 2^(8(j+j')) mod p
        self._ext_hadamard3 = self._ext_hadamard3_eager
        self._packed_hadamard = self._packed_hadamard_eager
        self._tk = None
        if self.mode == "compile":
            self._ext_hadamard3 = torch.compile(self._ext_hadamard3_eager, dynamic=True)
            self._packed_hadamard = torch.compile(self._packed_hadamard_eager, dynamic=True)
        elif self.mode == "triton":
            if self.device.type != "cuda":
                raise ValueError("triton mode needs a CUDA device")
            from . import kernels_triton as tk

            self._tk = tk
            self._ext_hadamard3 = tk.ext_hadamard3_limbs
            self._packed_hadamard = tk.packed_hadamard_limbs
        elif self.mode != "eager":
            raise ValueError(f"unknown mode {self.mode!r}")

    # -- shapes -------------------------------------------------------------------------------------------------------

    def split(self, m: int) -> tuple[int, int]:
        """``m = K1 . K2`` with ``K1 = min(m, 2^k1_log2)`` (the inner-product depth)."""
        K1 = min(m, 1 << self.k1_log2)
        return K1, m // K1

    # -- per-round inputs (host side: the verifier's coins) ----------------------------------------------------------

    def inputs(self, gammas: list[ref.Ext], rs: list[ref.Ext], i: int, k: int, r_i: ref.Ext | None = None) -> RoundInputs:
        """Round ``i`` (1-based) inputs from the eq point ``gammas`` and the challenges so far ``rs = [r_1..r_{i-1}]``;
        ``r_i`` (known only after the round) builds the fold matrix of an extension round, or of ``fold_k`` when ``i = k``."""
        n = len(gammas)
        m = 1 << (n - i)
        K1, _K2 = self.split(m)
        kappa = int(math.log2(K1))
        e_lo = eq_table_dev(gammas[i:i + kappa], self.device)
        e_hi = eq_table_dev(gammas[i + kappa:], self.device)
        gm = gammas[i - 1]
        eq_x = to_dev([ref.eval_at(ref.sub(ref.ONE, gm), gm, x) for x in range(4)], self.device)
        prefix = to_dev(ref.eq_points(gammas[:i - 1], rs), self.device)
        coeff = None
        if i <= k:
            eq_r = ref.eq_table(rs)
            coeff = to_dev([ref.mul(a, b) for a in eq_r for b in eq_r], self.device)
        fold_C = None
        inp = RoundInputs(decompose(e_lo), e_hi, eq_x, prefix, coeff, fold_C)
        if r_i is not None:
            self.set_fold(inp, rs, i, k, r_i)
        return inp

    def set_fold(self, inp: RoundInputs, rs: list[ref.Ext], i: int, k: int, r_i: ref.Ext) -> None:
        """Fill the fold operands of ``inp`` once ``r_i`` is known (ext round ``i > k``, or ``fold_k`` at ``i == k``)."""
        if i > k:
            C48 = fold_matrix_ext(r_i)
            inp.fold_C = torch.tensor(C48, device=self.device)
            if self._tk is not None:
                inp.fold_C64 = self._tk.fold_ext_matrix(C48, self.device)
        elif i == k:
            inp.fold_C = torch.tensor(fold_matrix_k(rs + [r_i]), device=self.device)
            if self._tk is not None:
                inp.fold_coeff = to_dev(ref.eq_table(rs + [r_i]), self.device)

    # -- SIMT pieces --------------------------------------------------------------------------------------------------

    @staticmethod
    def _three_points(G: torch.Tensor, dim: int) -> torch.Tensor:
        """``x in {0, 1, 2}`` values along ``dim`` (size 2): ``[G0, G1, 2 G1 - G0]``."""
        G0, G1 = G.select(dim, 0), G.select(dim, 1)
        G2 = sub_mod(add_mod(G1, G1), G0)
        return torch.stack([G0, G1, G2], dim=dim)

    def _packed_hadamard_eager(self, g: torch.Tensor, h: torch.Tensor, i: int, K1: int, K2: int) -> torch.Tensor:
        """``A[(b2, x, a, a', j), b1] = limb_j( g(a, x, b) h(a', x, b) mod p )`` as ``(K2 . 3 . 4^(i-1) . 4, K1) int8``."""
        half = 1 << (i - 1)
        G = self._three_points(g.view(K2, K1, 2, half).permute(0, 2, 3, 1), 1)      # (K2, 3, half, K1)
        H = self._three_points(h.view(K2, K1, 2, half).permute(0, 2, 3, 1), 1)
        A = ((G.unsqueeze(3).to(torch.int64) * H.unsqueeze(2).to(torch.int64)) % P).to(torch.int32)   # (K2, 3, half, half, K1)
        A = A.reshape(K2, 3 * half * half, K1)
        return decompose_rows(A).reshape(K2 * 3 * half * half * LIMBS, K1)

    def _ext_hadamard3_eager(self, gE: torch.Tensor, hE: torch.Tensor, K1: int, K2: int) -> torch.Tensor:
        """``A[(b2, x, s, j), b1] = limb_j( coeff_s( g_i(x, b) h_i(x, b) ) )`` as ``(K2 . 72, K1) int8`` from the
        ``(m, 2, 6)`` even/odd extension tables."""
        G = self._three_points(gE.view(K2, K1, 2, DEGREE).permute(0, 2, 1, 3), 1)    # (K2, 3, K1, 6)
        H = self._three_points(hE.view(K2, K1, 2, DEGREE).permute(0, 2, 1, 3), 1)
        A = ext_mul(G, H)                                                             # (K2, 3, K1, 6)
        Al = decompose(A)                                                             # (K2, 3, K1, 24)
        return Al.permute(0, 1, 3, 2).reshape(K2 * 3 * DL, K1)

    def _reduce_base_ext(self, Z: torch.Tensor, K2: int, rows: int) -> torch.Tensor:
        """``(K2 . rows . 4, 24) int32`` limb blocks (base x ext) -> ``(K2, rows, 6)`` coefficients mod p."""
        z = Z.to(torch.int64).view(K2, rows, LIMBS, DEGREE, LIMBS)                    # [b2, r, j, s, j']
        z = (z * self.pow2_pair[None, None, :, None, :]) % P
        return (z.sum((2, 4)) % P).to(torch.int32)

    def _reduce_ext_ext(self, Z: torch.Tensor, K2: int, rows: int) -> torch.Tensor:
        """``(K2 . rows . 24, 24) int32`` limb blocks (ext x ext) -> ``(K2, rows, 6)`` mod p, ``X^6 = W``."""
        z = Z.to(torch.int64).view(K2, rows, DEGREE, LIMBS, DEGREE, LIMBS)            # [b2, r, s, j, s', j']
        z = (z * self.pow2_pair[None, None, None, :, None, :]) % P
        z = z.sum((3, 5)) % P                                                          # [b2, r, s, s'] < p
        out = []
        for d in range(DEGREE):
            lo = sum(z[..., s, d - s] for s in range(d + 1))
            hi = sum((z[..., s, d + DEGREE - s] for s in range(d + 1, DEGREE)), torch.zeros_like(lo))
            out.append((lo + (hi % P) * W) % P)
        return torch.stack(out, dim=-1).to(torch.int32)

    def _contract_b2(self, X: torch.Tensor, e_hi: torch.Tensor) -> torch.Tensor:
        """``sum_{b2} e_hi(b2) X[b2, r]``: ``(K2, rows, 6) -> (rows, 6)``."""
        prod = ext_mul(X, e_hi[:, None, :])
        return (prod.to(torch.int64).sum(0) % P).to(torch.int32)

    def _q_to_s(self, Q: torch.Tensor, inp: RoundInputs) -> torch.Tensor:
        """``S(x) = prefix . eq(gamma_i, x) . Q(x)``, ``x = 0..3``, from ``Q(0..2)`` (degree 2): ``(4, 6) int32``."""
        Q3 = sub_mod(add_mod(add_mod(Q[2], Q[2]), Q[2]), sub_mod(add_mod(add_mod(Q[1], Q[1]), Q[1]), Q[0]))
        Q4 = torch.cat([Q, Q3[None]], dim=0)
        return ext_mul(ext_mul(inp.prefix[None, :], inp.eq_x), Q4)

    # -- rounds -------------------------------------------------------------------------------------------------------

    def packed_round(self, g: torch.Tensor, h: torch.Tensor, i: int, inp: RoundInputs, stats: Stats | None = None,
                     marks: dict | None = None) -> torch.Tensor:
        """Round ``i <= k`` on the untouched base tables ``g, h`` (``(t,) int32``): returns ``S_i(0..3)`` as ``(4, 6)``."""
        t = g.numel()
        m = t >> i
        K1, K2 = self.split(m)
        half = 1 << (i - 1)
        rows = 3 * half * half
        A = self._packed_hadamard(g, h, i, K1, K2)                                    # (K2 rows 4, K1) int8
        if marks is not None:
            marks["hadamard+limbs"] = self._mark()
        Z, padded = mm_int8(A, inp.e_lo_limbs)                                       # (K2 rows 4, 24) int32
        if stats:
            stats.add("inner.packed", A.shape[0], K1, DL, padded)
            stats.hadamard_base += K2 * rows * K1
        if marks is not None:
            marks["matmul.inner"] = self._mark()
        X = self._reduce_base_ext(Z, K2, rows)                                        # (K2, rows, 6)
        T = self._contract_b2(X, inp.e_hi)                                            # (rows, 6) = T[x, a, a']
        Tv = T.view(3, half * half, DEGREE)
        Q = (ext_mul(Tv, inp.coeff[None]).to(torch.int64).sum(1) % P).to(torch.int32)  # (3, 6)
        S = self._q_to_s(Q, inp)
        if marks is not None:
            marks["reduce+contract"] = self._mark()
        return S

    def fold_k(self, g: torch.Tensor, k: int, inp: RoundInputs, stats: Stats | None = None) -> torch.Tensor:
        """``g'(b) = sum_a eq(r_1..r_k, a) g(a, b)``: ``(t,) base -> (t / 2^k, 6)`` extension."""
        t = g.numel()
        m = t >> k
        if self._tk is not None:
            if stats:
                stats.add("fold_k", m, LIMBS << k, 2 * DL, 0)
            return self._tk.fold_k(g, k, inp.fold_coeff)
        L = decompose(g.view(m, 1 << k))                                              # (m, 4 . 2^k), column (a, j)
        Z, padded = mm_int8(L, inp.fold_C)                                            # (m, 48)
        if stats:
            stats.add("fold_k", m, L.shape[1], 2 * DL, padded)
        return recompose(Z, DEGREE, self.pow2_row)

    def ext_round(self, gE: torch.Tensor, hE: torch.Tensor, i: int, inp: RoundInputs, stats: Stats | None = None,
                  marks: dict | None = None, fold: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Round ``i > k`` on extension tables ``(t_i, 6)``: returns ``(S_i (4, 6), g', h')``; ``g', h'`` are the tables
        folded by ``r_i`` (``inp.fold_C``) when ``fold``."""
        t_i = gE.shape[0]
        m = t_i // 2
        K1, K2 = self.split(m)
        A = self._ext_hadamard3(gE, hE, K1, K2)                                       # (K2 . 72, K1)
        if marks is not None:
            marks["hadamard+limbs"] = self._mark()
        Z, padded = mm_int8(A, inp.e_lo_limbs)                                       # (K2 . 72, 24)
        if stats:
            stats.add("inner.ext", A.shape[0], K1, DL, padded)
            stats.hadamard_ext += 3 * m
        if marks is not None:
            marks["matmul.inner"] = self._mark()
        X = self._reduce_ext_ext(Z, K2, 3)                                            # (K2, 3, 6)
        Q = self._contract_b2(X, inp.e_hi)                                            # (3, 6)
        S = self._q_to_s(Q, inp)
        if marks is not None:
            marks["reduce+contract"] = self._mark()
        if not fold:
            return S, gE, hE
        g2, h2 = self.fold_pair(gE, hE, inp, stats, marks)
        return S, g2, h2

    def fold_pair(self, gE: torch.Tensor, hE: torch.Tensor, inp: RoundInputs, stats: Stats | None = None,
                  marks: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Both extension tables folded by ``r_i`` in one matmul: ``[x_e | x_o] (2m x 48) @ [C_(1-r); C_r] (48 x 48)``."""
        m = gE.shape[0] // 2
        if self._tk is not None:
            if stats:
                stats.add("fold.ext", 2 * m, 2 * DL, 2 * DL, 0)
            F0 = self._tk.fold_ext(gE.view(m, 2 * DEGREE), inp.fold_C64, self.pow2_row)
            F1 = self._tk.fold_ext(hE.view(m, 2 * DEGREE), inp.fold_C64, self.pow2_row)
            if marks is not None:
                marks["matmul.fold"] = self._mark()
            return F0, F1
        GH = torch.stack([gE, hE], 0)                                                 # (2, t_i, 6)
        L = decompose(GH.view(2 * m, 2 * DEGREE))                                     # (2m, 48): (parity, s, j)
        Zf, padded = mm_int8(L, inp.fold_C)                                           # (2m, 48)
        if stats:
            stats.add("fold.ext", 2 * m, 2 * DL, 2 * DL, padded)
        if marks is not None:
            marks["matmul.fold"] = self._mark()
        F = recompose(Zf, DEGREE, self.pow2_row).view(2, m, DEGREE)
        if marks is not None:
            marks["recompose.fold"] = self._mark()
        return F[0], F[1]

    # -- timing marks (CUDA events; no host sync) ----------------------------------------------------------------------

    def _mark(self):
        if self.device.type == "cuda":
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        import time

        return time.perf_counter()


# ---------------------------------------------------------------------------------------------------------------------
# the prover (all rounds)
# ---------------------------------------------------------------------------------------------------------------------


def S_to_ext(S: torch.Tensor) -> list[ref.Ext]:
    return [tuple(int(v) for v in row) for row in S.cpu().tolist()]


def prove(kernel: Kernel, gammas: list[ref.Ext], g: torch.Tensor, h: torch.Tensor, k: int, challenger: ref.Challenger,
          stats: Stats | None = None) -> tuple[list[list[ref.Ext]], list[ref.Ext], tuple[ref.Ext, ref.Ext, ref.Ext]]:
    """The whole sumcheck on the device (host round trip per round for the challenge): the same triple as
    ``reference.packed_prover``.  ``k = 0`` runs every round as an extension round (round 1 on 6-coefficient tables
    with zero high coefficients: the ordinary algorithm as the cost model prices it, minus the eq fold)."""
    n = len(gammas)
    rounds, rs = [], []
    for i in range(1, k + 1):
        inp = kernel.inputs(gammas, rs, i, k)
        S = S_to_ext(kernel.packed_round(g, h, i, inp, stats))
        r = challenger(i, S)
        rounds.append(S)
        rs.append(r)
    if k:
        kernel.set_fold(inp, rs[:-1], k, k, rs[-1])
        gE, hE = kernel.fold_k(g, k, inp, stats), kernel.fold_k(h, k, inp, stats)
    else:
        gE = torch.zeros((g.numel(), DEGREE), dtype=torch.int32, device=kernel.device)
        gE[:, 0] = g
        hE = torch.zeros_like(gE)
        hE[:, 0] = h
    for i in range(k + 1, n + 1):
        inp = kernel.inputs(gammas, rs, i, k)
        S, _, _ = kernel.ext_round(gE, hE, i, inp, stats, fold=False)
        S = S_to_ext(S)
        r = challenger(i, S)
        rounds.append(S)
        rs.append(r)
        kernel.set_fold(inp, rs[:-1], i, k, r)
        gE, hE = kernel.fold_pair(gE, hE, inp, stats)
    final = (ref.eq_points(gammas, rs), tuple(int(v) for v in gE[0].tolist()), tuple(int(v) for v in hE[0].tolist()))
    return rounds, rs, final

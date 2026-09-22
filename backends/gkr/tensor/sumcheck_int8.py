"""Candidate A's GKR c-round over BabyBear^6 as INT8 tensor-core matrix multiplications (TensorZKP's layout).

Layout (TensorZKP 4.1, adapted to signed INT8).  A field element is ``deg`` BabyBear coefficients, each 4 signed
8-bit limbs in balanced form ``l_j in [-128, 128)`` with ``a = sum l_j 2^(8j)`` (``reference.limbs_balanced``): cuBLAS /
``torch._int_mm`` multiply *signed* INT8, so the unsigned limbs of the paper are replaced by balanced ones.  A product
of two limbs is at most ``2^14`` in magnitude, so an INT32 accumulator holds ``N < 2^17`` products (the paper's UINT8
bound is ``N < 33,025``); chunks are ``CHUNK = 32768`` rows as in the paper and chunk sums are INT64.  A table of ``t``
elements is the INT8 matrix ``t x (4 deg)``, column ``(s, j) = 4 s + j`` (coefficient ``s``, limb ``j``).

Round (``round_step``): tables ``f, g, h`` (``t`` elements; base ``deg = 1`` or extension ``deg = 6``), challenge ``r``
in ``F_{p^6}``:

1. Hadamards ``gh_bc = g_b o h_c`` for ``b, c in {even, odd}`` (SIMT: coefficient arithmetic in INT64, ``ext_mul``).
2. Limb decomposition of ``f`` and of the four ``gh`` tables (SIMT, elementwise).
3. Inner products as ONE matmul per chunk: ``F^T x GH`` with ``F = [f_e | f_o]`` (``t/2 x 8 deg_f``) and
   ``GH = [gh_ee | gh_eo | gh_oe | gh_oo]`` (``t/2 x 16 deg_gh``): the ``8 deg_f x 16 deg_gh`` INT32 block holds the
   eight triple inner products ``T_abc = <f_a, g_b o h_c>`` as ``4 deg_f x 4 deg_gh`` sub-blocks.  Chunks are summed in
   INT64 and each sub-block is reduced once (``reduce_inner``): limb-pair diagonals times ``2^(8(j+j'))`` mod p, then the
   coefficient convolution folded by ``X^6 = W``.  ``S(0..3)`` are fixed integer combinations of the eight ``T_abc``
   (TensorZKP 5.1's higher-degree recipe with points ``0, 1, 2, 3``).
4. Fold as ONE matmul: ``[f_e | f_o ; g_e | g_o ; h_e | h_o]`` (``3t/2 x 8 deg``) times ``[C_(1-r) ; C_r]``
   (``8 deg x 48``), where ``C_c`` is the scalar's convolution matrix (TensorZKP 4.1, eq. 14) with ``X^6 = W`` already
   applied inside the matrix (the wrapped coefficients use the limbs of ``W c_t mod p``), so the output has 6 coefficient
   blocks of 8 limb positions (48 INT32 columns) instead of the paper's ``4 k L = 96``; ``reduce_in_matrix=False``
   reproduces the paper's 96-column layout.  The INT32 output is recomposed (``recompose``) to coefficients mod p.

``torch._int_mm`` needs ``M > 16``, ``K > 16``, ``K, N`` multiples of 8: operands are zero-padded to those shapes and the
padding is counted (``Stats.macs_padded`` vs ``Stats.macs_performed``).  The same code runs on CPU with numpy or torch
(``Ops``), which is the path the reference-agreement test exercises.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # the kernel's device path; the CPU reference path works without it
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:  # split-K inner product (cuBLAS via torch._int_mm runs a 48 x K x 96 GEMM on one SM: no split-K for INT8)
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

from . import reference as ref

P = ref.P
W = ref.W
DEGREE = ref.DEGREE
LIMBS = ref.LIMBS_PER_BASE           # 4 limbs per BabyBear coefficient
CHUNK = 32768                         # rows per INT32 accumulation (TensorZKP B); balanced limbs would allow < 131072
MIN_DIM = 24                          # smallest dimension torch._int_mm accepts (> 16, multiple of 8)
POW2 = [pow(2, 8 * m, P) for m in range(2 * LIMBS)]   # 2^(8m) mod p for limb positions m = 0..7 (m = 7 is padding)


# ---------------------------------------------------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------------------------------------------------


class Ops:
    """The dozen array operations the kernel needs, over numpy or torch (CPU or CUDA)."""

    def __init__(self, backend: str = "numpy", device: str = "cpu"):
        if backend not in ("numpy", "torch"):
            raise ValueError(backend)
        if backend == "torch" and torch is None:
            raise RuntimeError("torch is not installed")
        self.backend = backend
        self.device = device
        self.cuda = backend == "torch" and device.startswith("cuda")
        if backend == "torch":
            self.i8, self.i32, self.i64 = torch.int8, torch.int32, torch.int64
        else:
            self.i8, self.i32, self.i64 = np.int8, np.int32, np.int64

    # construction
    def from_numpy(self, a: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(a)).to(self.device) if self.backend == "torch" else np.ascontiguousarray(a)

    def to_numpy(self, x) -> np.ndarray:
        return x.detach().cpu().numpy() if self.backend == "torch" else np.asarray(x)

    def zeros(self, shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=self.device) if self.backend == "torch" else np.zeros(shape, dtype=dtype)

    def randint(self, shape, high: int, seed: int):
        rng = np.random.default_rng(seed)
        return self.from_numpy(rng.integers(0, high, size=shape, dtype=np.int64))

    # elementwise / shape
    def astype(self, x, dtype):
        return x.to(dtype) if self.backend == "torch" else x.astype(dtype)

    def cat(self, xs, axis: int):
        return torch.cat(xs, dim=axis) if self.backend == "torch" else np.concatenate(xs, axis=axis)

    def sum(self, x, axis: int):
        return x.sum(dim=axis, dtype=self.i64) if self.backend == "torch" else x.sum(axis=axis, dtype=np.int64)

    def contiguous(self, x):
        return x.contiguous() if self.backend == "torch" else np.ascontiguousarray(x)

    def synchronize(self) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)

    # the tensor-core instruction
    def mm_int8(self, a, b, out=None) -> tuple[Any, int]:
        """``a (M, K) int8 @ b (K, N) int8 -> (M, N) int32`` and the number of MACs the device performed (padding
        included).  CUDA: ``torch._int_mm`` on zero-padded operands; CPU: exact integer matmul, no padding."""
        M, K = a.shape
        K2, N = b.shape
        if K != K2:
            raise ValueError("shape mismatch")
        if self.cuda:
            Mp, Kp, Np = max(MIN_DIM, M), max(MIN_DIM, -(-K // 8) * 8), max(MIN_DIM, -(-N // 8) * 8)
            if (Mp, Kp, Np) != (M, K, N):
                ap = torch.zeros((Mp, Kp), dtype=torch.int8, device=a.device)
                ap[:M, :K] = a
                bp = torch.zeros((Kp, Np), dtype=torch.int8, device=b.device)
                bp[:K, :N] = b
                z = torch._int_mm(ap, bp)
                z = z[:M, :N]
                if out is not None:
                    out.copy_(z)
                    z = out
            elif out is not None:
                torch._int_mm(a, b, out=out)
                z = out
            else:
                z = torch._int_mm(a, b)
            return z, Mp * Kp * Np
        if self.backend == "torch":
            z = torch.mm(a.to(torch.int64), b.to(torch.int64)).to(torch.int32)
        else:
            z = (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)
        if out is not None:
            out[...] = z
            z = out
        return z, M * K * N


if triton is not None:

    @triton.jit
    def _splitk_int8_kernel(f_ptr, gh_ptr, out_ptr, rows, rows_per_block, M, N, stride_f, stride_gh,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """Block ``b`` accumulates ``F[b-th row slab]^T @ GH[b-th row slab]`` (``M x N`` INT32) with INT8 tensor-core
        dots over ``BK``-row tiles; ``out[b]`` is the block's partial sum (summed in INT64 by the caller).  ``BM, BN``
        are the power-of-two tiles; columns beyond ``M, N`` are masked to zero."""
        b = tl.program_id(0)
        row0 = b * rows_per_block
        row1 = tl.minimum(row0 + rows_per_block, rows)
        m = tl.arange(0, BM)
        n = tl.arange(0, BN)
        kk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for r0 in range(row0, row1, BK):
            rr = r0 + kk
            rmask = rr < row1
            f = tl.load(f_ptr + rr[:, None] * stride_f + m[None, :], mask=rmask[:, None] & (m[None, :] < M), other=0)     # (BK, BM)
            g = tl.load(gh_ptr + rr[:, None] * stride_gh + n[None, :], mask=rmask[:, None] & (n[None, :] < N), other=0)  # (BK, BN)
            acc += tl.dot(tl.trans(f), g, out_dtype=tl.int32)
        tl.store(out_ptr + b * M * N + m[:, None] * N + n[None, :], acc, mask=(m[:, None] < M) & (n[None, :] < N))


def splitk_inner(F, GH, rows_per_block: int, BK: int = 64) -> tuple[Any, int]:
    """``F (rows, M) int8``, ``GH (rows, N) int8`` -> ``(nblocks, M, N) int32`` partial products over row slabs of
    ``rows_per_block`` (<= the INT32 bound), on tensor cores via Triton ``tl.dot``.  Returns the partials and the MACs the
    device performed (tiles padded to powers of two)."""
    rows, M = F.shape
    N = GH.shape[1]
    BM, BN = max(16, triton.next_power_of_2(M)), max(16, triton.next_power_of_2(N))
    nblocks = -(-rows // rows_per_block)
    out = torch.empty((nblocks, M, N), dtype=torch.int32, device=F.device)
    _splitk_int8_kernel[(nblocks,)](F, GH, out, rows, rows_per_block, M, N, F.stride(0), GH.stride(0), BM=BM, BN=BN, BK=BK, num_warps=4)
    tiles = sum(-(-min(rows_per_block, rows - b * rows_per_block) // BK) for b in range(nblocks))
    return out, tiles * BK * BM * BN


@dataclass
class Stats:
    """MAC and element counts of the matmuls a round performed."""

    macs_performed: int = 0      # M K N of the unpadded matmuls (the cost model's unit)
    macs_padded: int = 0         # what the device actually multiplied
    hadamard_products: int = 0   # field products done on SIMT (coefficient arithmetic)
    inner_products: int = 0      # reductions of a 4 deg x 4 deg block to one field element
    fold_elements: int = 0       # elements recomposed after the fold
    segments: dict = field(default_factory=dict)

    def add_mm(self, M: int, K: int, N: int, padded: int) -> None:
        self.macs_performed += M * K * N
        self.macs_padded += padded


class Timer:
    """Segment timer: ``mark(name)`` closes the current segment.  CUDA events on the device path (no host sync until
    ``finish``), ``perf_counter`` otherwise."""

    def __init__(self, ops: Ops, stats: Stats | None = None):
        self.ops = ops
        self.stats = stats
        self.events: list[tuple[str, Any]] = []
        self._start()

    def _now(self):
        if self.ops.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def _start(self) -> None:
        self.events = [("start", self._now())]

    def mark(self, name: str) -> None:
        self.events.append((name, self._now()))

    def finish(self) -> dict[str, float]:
        """Seconds per segment (name -> s); synchronises the device."""
        self.ops.synchronize()
        out: dict[str, float] = {}
        for (_, a), (name, b) in zip(self.events, self.events[1:]):
            dt = a.elapsed_time(b) / 1e3 if self.ops.cuda else b - a
            out[name] = out.get(name, 0.0) + dt
        if self.stats is not None:
            for k, v in out.items():
                self.stats.segments[k] = self.stats.segments.get(k, 0.0) + v
        return out


# ---------------------------------------------------------------------------------------------------------------------
# limb passes (SIMT, elementwise)
# ---------------------------------------------------------------------------------------------------------------------


def decompose(ops: Ops, coeffs, deg: int):
    """``(n, deg) int32`` in ``[0, p)`` -> ``(n, 4 deg) int8`` balanced limbs, column ``4 s + j``."""
    v = coeffs
    parts = []
    for _ in range(LIMBS):
        l = ((v + 128) & 255) - 128
        parts.append(ops.astype(l, ops.i8)[..., None])
        v = (v - l) >> 8
    return ops.cat(parts, -1).reshape(coeffs.shape[0], deg * LIMBS)


def recompose(ops: Ops, z, deg_out: int, pow2):
    """Fold output ``(n, 8 deg_out) int32`` (coefficient block ``d`` = columns ``8d .. 8d+7``, limb positions) ->
    ``(n, deg_out) int32`` mod p.  Entries are < 2^20 in magnitude, so ``z * 2^(8m) mod p`` stays below 2^51."""
    n = z.shape[0]
    z64 = ops.astype(z, ops.i64).reshape(n, deg_out, 2 * LIMBS)
    return ops.astype(ops.sum(z64 * pow2, -1) % P, ops.i32)


def ext_mul(ops: Ops, a, da: int, b, db: int):
    """Hadamard product of two coefficient tables (``(n, da)`` and ``(n, db)``, int32 in ``[0, p)``) in ``F_{p^6}``:
    ``da db`` coefficient products with a reduction each (products < 2^62), wrapped by ``X^6 = W``.  Returns
    ``(n, deg_out)`` with ``deg_out = 1`` for base x base, else 6."""
    deg_out = 1 if da == 1 and db == 1 else DEGREE
    acc = [None] * deg_out
    for i in range(da):
        for j in range(db):
            prod = ops.astype(a[:, i], ops.i64) * ops.astype(b[:, j], ops.i64) % P
            d = i + j
            if d >= DEGREE:
                prod = prod * W % P
                d -= DEGREE
            acc[d] = prod if acc[d] is None else acc[d] + prod
    cols = [(x % P if x is not None else ops.zeros((a.shape[0],), ops.i64))[:, None] for x in acc]
    return ops.astype(ops.cat(cols, -1), ops.i32)


# ---------------------------------------------------------------------------------------------------------------------
# scalar (convolution) matrix, TensorZKP 4.1
# ---------------------------------------------------------------------------------------------------------------------


def scalar_matrix(c: ref.Ext, in_deg: int, reduce_in_matrix: bool = True) -> np.ndarray:
    """``C_c``: ``(4 in_deg) x (8 deg_out)`` int8 with ``limbs(v) @ C_c`` = the unreduced limb-position sums of ``c v``:
    entry ``[(s, j), (d, j + j')] = limb_j'(c_t)`` for ``d = s + t`` (``deg_out = 11 -> 12`` blocks), or with
    ``X^6 = W`` applied (``deg_out = 6``; wrapped terms use the limbs of ``W c_t mod p``).  Every output entry is a sum
    of at most ``2 in_deg`` limb products (< 2^14 each), far inside INT32."""
    deg_out = DEGREE if reduce_in_matrix else 2 * DEGREE
    C = np.zeros((in_deg * LIMBS, deg_out * 2 * LIMBS), dtype=np.int8)
    limbs_c = [ref.limbs_balanced(ct) for ct in c]
    limbs_wc = [ref.limbs_balanced(W * ct % P) for ct in c]
    for s in range(in_deg):
        for t in range(DEGREE):
            d = s + t
            if reduce_in_matrix and d >= DEGREE:
                d -= DEGREE
                lc = limbs_wc[t]
            else:
                lc = limbs_c[t]
            for j in range(LIMBS):
                for jp in range(LIMBS):
                    C[s * LIMBS + j, d * 2 * LIMBS + j + jp] = lc[jp]
    return C


def fold_matrix(ops: Ops, r: ref.Ext, in_deg: int, reduce_in_matrix: bool = True):
    """``[C_(1-r) ; C_r]`` for the fold ``(1 - r) x_even + r x_odd`` of ``[x_e | x_o]`` rows."""
    one_minus_r = ref.sub(ref.ONE, r)
    return ops.from_numpy(np.concatenate([scalar_matrix(one_minus_r, in_deg, reduce_in_matrix),
                                          scalar_matrix(r, in_deg, reduce_in_matrix)], axis=0))


# ---------------------------------------------------------------------------------------------------------------------
# the round
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Table:
    """A sumcheck table in coefficient form: ``(t, deg) int32`` in ``[0, p)`` (p < 2^31); ``deg`` 1 (base) or 6 (extension)."""

    coeffs: Any
    deg: int

    @property
    def t(self) -> int:
        return int(self.coeffs.shape[0])


SPLITK_BLOCKS = 256   # target number of row slabs (CTAs) for the split-K inner product


class Kernel:
    def __init__(self, ops: Ops, chunk: int = CHUNK, reduce_in_matrix: bool = True, inner_impl: str = "auto"):
        """``inner_impl``: ``"int_mm"`` (chunk loop of ``torch._int_mm`` / CPU matmul), ``"triton"`` (split-K tensor-core
        kernel, CUDA only), ``"auto"`` (triton when available on CUDA)."""
        self.ops = ops
        self.chunk = chunk
        self.reduce_in_matrix = reduce_in_matrix
        if inner_impl == "auto":
            inner_impl = "triton" if (ops.cuda and triton is not None) else "int_mm"
        if inner_impl == "triton" and not (ops.cuda and triton is not None):
            raise RuntimeError("the triton split-K path needs CUDA and triton")
        self.inner_impl = inner_impl
        self.pow2_row = ops.from_numpy(np.array(POW2, dtype=np.int64))                         # (8,)
        pm = np.array([[POW2[j + jp] for jp in range(LIMBS)] for j in range(LIMBS)], dtype=np.int64)
        self.pow2_pair = ops.from_numpy(pm).reshape(1, 1, LIMBS, 1, 1, LIMBS)                  # 2^(8(j+j')) mod p

    # -- tables -------------------------------------------------------------------------------------------------------

    def table_from_ints(self, values: list[ref.Ext], deg: int) -> Table:
        arr = np.array([v[:deg] for v in values], dtype=np.int32).reshape(len(values), deg)
        return Table(self.ops.from_numpy(arr), deg)

    def table_to_ints(self, tb: Table) -> list[ref.Ext]:
        a = self.ops.to_numpy(tb.coeffs)
        return [tuple(int(x) for x in row) + (0,) * (DEGREE - tb.deg) for row in a]

    def random_table(self, t: int, deg: int, seed: int) -> Table:
        return Table(self.ops.astype(self.ops.randint((t, deg), P, seed), self.ops.i32), deg)

    # -- pieces -------------------------------------------------------------------------------------------------------

    def even_odd_limbs(self, tb: Table):
        """``[x_e | x_o]`` as ``(t/2, 8 deg) int8``."""
        ops = self.ops
        limbs = decompose(ops, tb.coeffs, tb.deg).reshape(tb.t // 2, 2 * tb.deg * LIMBS)  # row 2k, 2k+1 side by side
        return ops.contiguous(limbs)

    def inner_products(self, F, GH, deg_f: int, deg_gh: int, stats: Stats, timer: Timer | None):
        """``T_abc`` as ``(2, 4, 6) int64`` (a, bc, coefficient): the chunked ``F^T x GH`` matmuls, INT64 chunk sums,
        one reduction per 4 deg_f x 4 deg_gh block."""
        ops = self.ops
        half = F.shape[0]
        M, N = 2 * deg_f * LIMBS, 4 * deg_gh * LIMBS
        if self.inner_impl == "triton":
            # split-K on tensor cores: one block per slab of <= chunk rows, ~SPLITK_BLOCKS slabs
            rows_per_block = min(self.chunk, max(64, -(-half // SPLITK_BLOCKS)))
            buf, padded = splitk_inner(F, GH, rows_per_block)
            stats.add_mm(M, half, N, padded)
        else:
            nchunks = -(-half // self.chunk)
            FT = ops.contiguous(F.T)
            if ops.backend == "torch":
                buf = torch.zeros((nchunks, M, N), dtype=torch.int32, device=ops.device)
            else:
                buf = np.zeros((nchunks, M, N), dtype=np.int32)
            for c in range(nchunks):
                lo, hi = c * self.chunk, min(half, (c + 1) * self.chunk)
                _, padded = ops.mm_int8(FT[:, lo:hi], GH[lo:hi], out=buf[c])
                stats.add_mm(M, hi - lo, N, padded)
        if timer:
            timer.mark("matmul.inner")
        Z = ops.sum(buf, 0)                                                  # (M, N) int64
        if timer:
            timer.mark("chunk_sum")
        T = self.reduce_inner(Z, deg_f, deg_gh)
        stats.inner_products += 8
        if timer:
            timer.mark("reduce.inner")
        return T

    def reduce_inner(self, Z, deg_f: int, deg_gh: int):
        ops = self.ops
        z = (Z % P).reshape(2, deg_f, LIMBS, 4, deg_gh, LIMBS)            # [a, s, j, bc, t, j']
        z = (z * self.pow2_pair) % P                                          # < 2^62 before the mod
        z = ops.sum(ops.sum(z, 5), 2)                                         # [a, s, bc, t], < 16 p
        out = [None] * DEGREE
        for s in range(deg_f):
            for t in range(deg_gh):
                term = z[:, s, :, t]
                d = s + t
                if d >= DEGREE:
                    term = term * W % P
                    d -= DEGREE
                out[d] = term if out[d] is None else out[d] + term
        cols = [(x % P if x is not None else ops.zeros((2, 4), ops.i64))[..., None] for x in out]
        return ops.cat(cols, -1)                                              # (2, 4, 6)

    def round_values(self, T):
        """``S(0), S(1), S(2), S(3)`` from ``T[a, bc]`` (a: e=0 o=1; bc: ee=0 eo=1 oe=2 oo=3) as ``(4, 6) int64``.
        ``x(2) = 2 x_o - x_e``, ``x(3) = 3 x_o - 2 x_e`` expanded over the triple product."""
        Teee, Teeo, Teoe, Teoo = T[0, 0], T[0, 1], T[0, 2], T[0, 3]
        Toee, Toeo, Tooe, Tooo = T[1, 0], T[1, 1], T[1, 2], T[1, 3]
        two_odd = Tooe + Toeo + Teoo
        one_odd = Toee + Teoe + Teeo
        s2 = (8 * Tooo - 4 * two_odd + 2 * one_odd - Teee) % P
        s3 = (27 * Tooo - 18 * two_odd + 12 * one_odd - 8 * Teee) % P
        return self.ops.cat([Teee[None], Tooo[None], s2[None], s3[None]], 0)

    def fold(self, EO_f, EO_g, EO_h, deg: int, r: ref.Ext, stats: Stats, timer: Timer | None, C=None):
        """The three tables folded by ``r`` in one matmul; returns three ``(t/2, 6)`` coefficient tables.  ``C`` is
        the precomputed ``fold_matrix`` (needed inside a CUDA-graph capture, where the host-to-device copy cannot run)."""
        ops = self.ops
        if C is None:
            C = fold_matrix(ops, r, deg, self.reduce_in_matrix)                # (8 deg, 48 | 96)
        Zs = []
        for rows in (EO_f, EO_g, EO_h):                                        # (t/2, 8 deg) each; one GEMM per table
            Z, padded = ops.mm_int8(rows, C)
            stats.add_mm(rows.shape[0], rows.shape[1], C.shape[1], padded)
            Zs.append(Z)
        if timer:
            timer.mark("matmul.fold")
        deg_out = DEGREE if self.reduce_in_matrix else 2 * DEGREE
        out = []
        for Z in Zs:
            coeffs = recompose(ops, Z, deg_out, self.pow2_row)                 # (t/2, deg_out)
            if not self.reduce_in_matrix:
                lo, hi = coeffs[:, :DEGREE], coeffs[:, DEGREE:]                # block 11 is the paper's zero padding
                coeffs = ops.astype((ops.astype(lo, ops.i64) + ops.astype(hi, ops.i64) * W) % P, ops.i32)
            stats.fold_elements += Z.shape[0]
            out.append(Table(coeffs, DEGREE))
        if timer:
            timer.mark("reduce.fold")
        return out

    # -- one round ----------------------------------------------------------------------------------------------------

    def round_step(self, f: Table, g: Table, h: Table, r: ref.Ext, stats: Stats | None = None, timer: Timer | None = None,
                   fold_C=None):
        """One degree-3 round: returns ``(S (4, 6) int64, f', g', h')``."""
        ops = self.ops
        stats = stats if stats is not None else Stats()
        t = f.t
        if t % 2 or g.t != t or h.t != t or not (f.deg == g.deg == h.deg):
            raise ValueError("tables must have equal even length and equal degree")
        deg = f.deg
        half = t // 2
        # 1. Hadamards g_b o h_c (SIMT)
        ge, go = g.coeffs[0::2], g.coeffs[1::2]
        he, ho = h.coeffs[0::2], h.coeffs[1::2]
        gh = [ext_mul(ops, gb, deg, hc, deg) for gb in (ge, go) for hc in (he, ho)]   # ee, eo, oe, oo
        deg_gh = gh[0].shape[1]
        stats.hadamard_products += 4 * half
        if timer:
            timer.mark("hadamard")
        # 2. limb decomposition
        EO_f, EO_g, EO_h = (self.even_odd_limbs(x) for x in (f, g, h))
        GH = ops.cat([decompose(ops, x, deg_gh) for x in gh], -1)                   # (t/2, 16 deg_gh)
        if timer:
            timer.mark("limb.decompose")
        # 3. inner products on tensor cores, reduce
        T = self.inner_products(EO_f, GH, deg, deg_gh, stats, timer)
        S = self.round_values(T)
        # 4. fold on tensor cores, recompose
        f2, g2, h2 = self.fold(EO_f, EO_g, EO_h, deg, r, stats, timer, fold_C)
        return S, f2, g2, h2


def default_ops(prefer_cuda: bool = True) -> Ops:
    """torch+CUDA when available, else torch CPU, else numpy."""
    if torch is not None:
        if prefer_cuda and torch.cuda.is_available():
            return Ops("torch", "cuda")
        return Ops("torch", "cpu")
    return Ops("numpy")

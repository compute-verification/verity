"""B-Ligero's linear test through the structure of ``r^T A`` (b-lintest lane), bit-exact against ``buckets.Combiner.linear``.

The test (``PROTOCOL.md`` section 5 step 5, AHIV22 4.7): ``q(x) = sum_i R_i(x) U_i(x)`` where ``U_i`` is the ``deg < k``
interpolant of Ligero row ``i`` of the committed matrix and ``R_i`` the ``deg < k`` interpolant of row ``i`` of ``r^T A``
(an ``F_c`` row: 6 BabyBear coordinate rows), both on the 8,192-point domain (cosets 0 and 1).  The b-commit lane timed
the general form: 6 coordinate encodings of ``r^T A`` + 1 re-encoding of the witness row per row (14 transforms).

Structure (``LINTEST.md``): the protocol's combination coefficients are POWERS ``r_add = (1, rho, rho^2, ...)`` over the
linear constraints (``PROTOCOL.md`` section 6), every unit satisfies the same local constraint pattern, and a Ligero row
holds whole units.  So ``r^T A`` restricted to a regular row ``i`` is ``s_i * E`` -- a per-row ``F_c`` scalar times ONE
fixed ``F_c`` row (``E``, the rho-combined per-unit pattern replicated over the row's unit slots); helper rows and
table rows give a few more fixed rows.  Interpolation is linear, so with ``M`` fixed rows ``E_m`` and scalars
``s[i][m]``:

    q(x) = sum_m G_m(x) * V_m(x),   G_m = enc(E_m) (once per batch),   V_m = enc( sum_i s[i][m] W_i )  (one k-wide pass over W)

plus the generic ``sum_i R_i U_i`` for the few EXCEPTIONAL rows (boundary units, table-side helper equations) whose
``r^T A`` is not of that form.  Per-row transforms: 14 -> 0 (the pass over ``W`` is the IRS kernel's memory-bound GEMV;
``6 M + 6`` transforms per batch, not per row).  Same test, same ``r``, same ``q``: a prover-side algorithm, not a
protocol change.  This module provides the numpy reference (``expand`` -> dense ``r^T A`` -> ``buckets_ref.linear_test``),
the toy unit-pattern system that exhibits the structure (``unit_pattern_system``), the GPU implementation
(``StructuredLinearTest``: fused ``M``-vector row combination, the encoder, an ``F_c`` pointwise product, the generic path for
exceptional rows), a tensor-core INT8 limb variant of the row combination (``combine_int8_tc``, torch ``_int_mm``), and the
recorded bench (``python3 -m encode.lintest_structured --reps 7 --out ...``).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encode import babybear as bb
from encode import buckets_ref as ref

P = bb.P
D = 6
NR = ref.NR
K = bb.K_CODE
L_PAYLOAD = 3840
U64 = np.uint64


# -- F_c arithmetic on numpy coordinate arrays (last axis = 6 coordinates, plain uint64) --------------------------------
def fc_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a``, ``b`` ``(..., 6)`` uint64 < p -> product ``(..., 6)``; broadcasts."""
    a, b = np.asarray(a, dtype=U64), np.asarray(b, dtype=U64)
    shape = np.broadcast_shapes(a.shape, b.shape)
    t = np.zeros(shape[:-1] + (2 * D - 1,), dtype=U64)
    for i in range(D):
        for j in range(D):
            t[..., i + j] = (t[..., i + j] + (a[..., i] * b[..., j]) % U64(P)) % U64(P)
    out = t[..., :D].copy()
    for i in range(D, 2 * D - 1):
        out[..., i - D] = (out[..., i - D] + (t[..., i] * U64(NR)) % U64(P)) % U64(P)
    return out


def fc_scale_row(s: np.ndarray, row: np.ndarray) -> np.ndarray:
    """``F_c`` scalar ``s`` ``(6,)`` times an ``F_c`` row ``(6, n)`` -> ``(6, n)``."""
    return fc_mul_np(np.asarray(s)[None, :], np.asarray(row).T).T


def fc_pow_seq(rho: list[int], count: int, step: int = 1, start: int = 0) -> np.ndarray:
    """``[rho^start, rho^(start+step), ...]`` ``(count, 6)``."""
    out = np.empty((count, D), dtype=U64)
    cur = ref.fc_pow(rho, start)
    stp = ref.fc_pow(rho, step)
    for i in range(count):
        out[i] = cur
        cur = ref.fc_mul(cur, stp)
    return out


# -- the structured description of r^T A --------------------------------------------------------------------------------
@dataclass
class StructuredRTA:
    """``r^T A`` for ``rows`` Ligero rows: regular row ``i`` = ``sum_m s[i][m] E[m]``; ``exceptional[i]`` = dense ``(6, k)`` row."""

    E: np.ndarray                                   # (M, 6, k) plain: the fixed F_c rows (zero at the k - l randomizer positions)
    s: np.ndarray                                   # (rows, M, 6) plain: per-row F_c scalars (zero rows for exceptional rows)
    exceptional: dict[int, np.ndarray] = field(default_factory=dict)   # row -> (6, k) plain

    @property
    def rows(self) -> int:
        return self.s.shape[0]

    @property
    def M(self) -> int:
        return self.E.shape[0]

    def expand(self) -> np.ndarray:
        """Dense ``(6, rows, k)`` plain ``r^T A`` -- the input of the general path (``buckets_ref.linear_test``, ``Combiner.linear``)."""
        out = np.zeros((D, self.rows, self.E.shape[2]), dtype=U64)
        for i in range(self.rows):
            if i in self.exceptional:
                out[:, i] = self.exceptional[i]
                continue
            acc = np.zeros((D, self.E.shape[2]), dtype=U64)
            for m in range(self.M):
                acc = (acc + fc_scale_row(self.s[i, m], self.E[m])) % U64(P)
            out[:, i] = acc
        return out


def random_structured(rows: int, M: int, n_exceptional: int, k: int = K, l: int = L_PAYLOAD, seed: int = 0) -> StructuredRTA:
    """Pseudo-random instance of the structured form (for bit-exactness and timing; data-independent cost)."""
    rng = np.random.default_rng(seed)
    E = rng.integers(0, P, (M, D, k)).astype(U64)
    E[:, :, l:] = 0
    s = rng.integers(0, P, (rows, M, D)).astype(U64)
    exc = {}
    for i in sorted(rng.choice(rows, min(n_exceptional, rows), replace=False).tolist()):
        row = rng.integers(0, P, (D, k)).astype(U64)
        row[:, l:] = 0
        exc[i] = row
        s[i] = 0
    return StructuredRTA(E, s, exc)


# -- a toy chain constraint system exhibiting the structure (laptop-sized; numpy) -----------------------------------------
@dataclass
class UnitPatternSystem:
    """``U`` units of ``w`` slots each; per unit ``C`` local linear constraints (``own[c]``: coefficients on the unit's slots) of
    which the ``transition`` ones also touch the NEXT unit's slots (``nxt[c]``); the last unit has no transition constraints
    (no successor).  Boundary constraints on unit 0 (``boundary``: a few coefficient vectors on unit 0's slots) follow.
    Constraint order (public convention, ``LINTEST.md``): ``u * C + c`` for the per-unit constraints, then the boundary
    ones.  Ligero rows: ``S`` whole units per row, payload ``S w`` <= ``l``, zeros at positions ``>= l`` (randomizers)."""

    U: int
    w: int
    C: int
    transition: list[int]
    own: np.ndarray             # (C, w) F_t coefficients
    nxt: np.ndarray             # (C, w) F_t coefficients on the next unit (zero for non-transition constraints)
    boundary: np.ndarray        # (nb, w) F_t coefficients on unit 0
    S: int
    k: int
    l: int

    @classmethod
    def random(cls, U=10, w=7, C=5, transition=(3, 4), nb=2, S=2, k=16, l=14, density=2, seed=0) -> UnitPatternSystem:
        rng = np.random.default_rng(seed)
        own = np.zeros((C, w), dtype=np.int64)
        nxt = np.zeros((C, w), dtype=np.int64)
        for c in range(C):
            own[c, rng.choice(w, density, replace=False)] = rng.integers(1, P, density)
            if c in transition:
                nxt[c, rng.choice(w, 1)] = rng.integers(1, P, 1)
        boundary = np.zeros((nb, w), dtype=np.int64)
        for b in range(nb):
            boundary[b, rng.choice(w, 1)] = rng.integers(1, P, 1)
        assert S * w <= l <= k
        return cls(U, w, C, list(transition), own, nxt, boundary, S, k, l)

    @property
    def n_constraints(self) -> int:
        return self.U * self.C - len(self.transition) + self.boundary.shape[0]

    def matrix(self) -> np.ndarray:
        """Dense ``A`` ``(n_constraints, U w)`` int64 (the per-unit block structure made explicit)."""
        A = np.zeros((self.U * self.C + self.boundary.shape[0], self.U * self.w), dtype=np.int64)
        keep = []
        for u in range(self.U):
            for c in range(self.C):
                idx = u * self.C + c
                if u == self.U - 1 and c in self.transition:
                    continue                                        # the last unit has no successor: constraint absent
                keep.append(idx)
                A[idx, u * self.w:(u + 1) * self.w] = self.own[c]
                if c in self.transition:
                    A[idx, (u + 1) * self.w:(u + 2) * self.w] = self.nxt[c]
        for b in range(self.boundary.shape[0]):
            idx = self.U * self.C + b
            keep.append(idx)
            A[idx, : self.w] = self.boundary[b]
        return A[keep]

    def powers_r(self, rho: list[int]) -> np.ndarray:
        """``r_add = (1, rho, rho^2, ...)`` over the constraints in the order of ``matrix()``: ``(n_constraints, 6)``."""
        n = self.U * self.C + self.boundary.shape[0]
        allp = fc_pow_seq(rho, n)
        keep = [u * self.C + c for u in range(self.U) for c in range(self.C) if not (u == self.U - 1 and c in self.transition)]
        keep += [self.U * self.C + b for b in range(self.boundary.shape[0])]
        return allp[keep]

    def rows(self) -> int:
        return -(-self.U // self.S)

    def dense_rta(self, rho: list[int]) -> np.ndarray:
        """``r^T A`` computed the general way (``F_c x F_t`` sums over the explicit matrix), laid out as ``(6, rows, k)``."""
        A = self.matrix()
        r = self.powers_r(rho)
        rta = np.zeros((self.U * self.w, D), dtype=U64)
        for idx in range(A.shape[0]):
            nz = np.nonzero(A[idx])[0]
            for col in nz:
                rta[col] = (rta[col] + (r[idx] * U64(A[idx, col])) % U64(P)) % U64(P)
        out = np.zeros((D, self.rows(), self.k), dtype=U64)
        for u in range(self.U):
            i, sl = divmod(u, self.S)
            out[:, i, sl * self.w:(sl + 1) * self.w] = rta[u * self.w:(u + 1) * self.w].T
        return out

    def structured(self, rho: list[int]) -> StructuredRTA:
        """The shortcut: ``pi = sum_c rho^c own[c] + rho^-C sum_c rho^c nxt[c]`` (the unit pattern under the powers
        combination, the neighbour part arriving from the previous unit's constraints), ``E = sum_s rho^(s C) shift_s(pi)``,
        ``s_i = rho^(i S C)``; exceptional rows: those holding unit 0 (no predecessor, boundary constraints) and the last
        unit (no transition constraints)."""
        rc = fc_pow_seq(rho, self.C)                                                  # rho^c
        pi = np.zeros((self.w, D), dtype=U64)
        for c in range(self.C):
            pi = (pi + (rc[c][None, :] * self.own[c].astype(U64)[:, None]) % U64(P)) % U64(P)
        rinvC = ref.fc_inv(ref.fc_pow(rho, self.C))
        nb = np.zeros((self.w, D), dtype=U64)
        for c in self.transition:
            nb = (nb + (rc[c][None, :] * self.nxt[c].astype(U64)[:, None]) % U64(P)) % U64(P)
        pi = (pi + fc_mul_np(np.asarray(rinvC, dtype=U64)[None, :], nb)) % U64(P)
        E = np.zeros((1, D, self.k), dtype=U64)
        rsC = fc_pow_seq(rho, self.S, step=self.C)                                    # rho^(s C)
        for sl in range(self.S):
            E[0, :, sl * self.w:(sl + 1) * self.w] = fc_mul_np(rsC[sl][None, :], pi).T
        rows = self.rows()
        s = fc_pow_seq(rho, rows, step=self.S * self.C).reshape(rows, 1, D)
        dense = self.dense_rta(rho)
        exc_rows = {0, (self.U - 1) // self.S}
        exc = {i: dense[:, i].copy() for i in sorted(exc_rows)}
        for i in exc:
            s[i] = 0
        return StructuredRTA(E, s, exc)


# -- numpy reference of the structured algorithm ----------------------------------------------------------------------------
def structured_linear_test_ref(st: StructuredRTA, W: np.ndarray) -> np.ndarray:
    """``q`` ``(6, 2k)`` by the shortcut in numpy: ``sum_m enc(E_m) * enc(sum_i s_im W_i)`` + the generic sum over exceptional rows."""
    Wd = np.asarray(W, dtype=U64)
    q = np.zeros((D, 2 * st.E.shape[2]), dtype=U64)
    for m in range(st.M):
        V = ref.coords_times_rows(st.s[:, m, :], Wd)                     # (6, k): sum_i s[i][m] W_i, coordinate-wise
        Vc = ref.encode2(V)                                              # (6, 2k)
        Gc = ref.encode2(st.E[m])                                        # (6, 2k)
        q = (q + fc_mul_np(Gc.T, Vc.T).T) % U64(P)
    if st.exceptional:
        idx = sorted(st.exceptional)
        rA = np.stack([np.stack([st.exceptional[i][c] for i in idx]) for c in range(D)])   # (6, X, k)
        q = (q + ref.linear_test(rA, Wd[idx])) % U64(P)
    return q


# -- GPU ---------------------------------------------------------------------------------------------------------------------
_SRC = r"""
#define P 2013265921u
#define PINV %(pinv)du
#define D 6
#define M %(m)d
#define W_NR 31u
typedef unsigned u32;
__device__ __forceinline__ u32 add_p(u32 a, u32 b) { u32 s = a + b; return s >= P ? s - P : s; }
__device__ __forceinline__ u32 mont_mul(u32 a, u32 w) {
    unsigned long long x = (unsigned long long)a * w;
    u32 lo = (u32)x, hi = (u32)(x >> 32);
    u32 t = lo * PINV;
    u32 u = __umulhi(t, P);
    u32 r = hi - u;
    return hi < u ? r + P : r;
}

/* V_m[c][j] = sum_i s[i][m][c] W[i][j] for all M fixed rows at once: one read of W, 6 M accumulators per thread.
   Thread (j, chunk); s is (rows, M, D) Montgomery; acc (nchunk, M, D, ncols). */
extern "C" __global__ void combine_multi(const u32* __restrict__ W, const u32* __restrict__ s, u32* __restrict__ acc,
                                         int rows, int chunk_rows, int ncols)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = blockIdx.y;
    if (j >= ncols) return;
    int i0 = chunk * chunk_rows, i1 = min(rows, i0 + chunk_rows);
    u32 a[M * D];
    #pragma unroll
    for (int c = 0; c < M * D; ++c) a[c] = 0;
    for (int i = i0; i < i1; ++i) {
        u32 w = __ldg(&W[(size_t)i * ncols + j]);
        const u32* si = s + (size_t)i * M * D;
        #pragma unroll
        for (int c = 0; c < M * D; ++c) a[c] = add_p(a[c], mont_mul(w, __ldg(&si[c])));
    }
    #pragma unroll
    for (int c = 0; c < M * D; ++c) {
        u32* o = acc + ((size_t)chunk * M * D + c) * ncols + j;
        *o = add_p(*o, a[c]);
    }
}

/* acc (nchunk, M*D, ncols) -> out (M*D, ncols) */
extern "C" __global__ void reduce_multi(const u32* __restrict__ acc, u32* __restrict__ out, int nchunk, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * D * ncols) return;
    u32 s = 0;
    for (int ch = 0; ch < nchunk; ++ch) s = add_p(s, acc[(size_t)ch * M * D * ncols + idx]);
    out[idx] = s;
}

/* q[c][j] += (G(x_j) * V(x_j))_c over F_c = F_p[X]/(X^6 - 31), j < 2k; G, V (D, 2k) plain (sys | coset 1 concatenated
   by the caller through the two pointers: j < k reads *_sys[c][j], else *_c1[c][j - k]).  Plain x plain: mont_mul gives
   a b R^-1, fixed by a second mont_mul with R^2. */
extern "C" __global__ void fc_pointwise_accum(const u32* __restrict__ g_sys, const u32* __restrict__ g_c1,
                                              const u32* __restrict__ v_sys, const u32* __restrict__ v_c1,
                                              u32* __restrict__ q, int k, u32 r2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= 2 * k) return;
    const bool sys = j < k;
    const int jj = sys ? j : j - k;
    const u32* G = sys ? g_sys : g_c1;
    const u32* V = sys ? v_sys : v_c1;
    u32 g[D], v[D], t[2 * D - 1];
    #pragma unroll
    for (int c = 0; c < D; ++c) { g[c] = __ldg(&G[(size_t)c * k + jj]); v[c] = mont_mul(__ldg(&V[(size_t)c * k + jj]), r2); }
    #pragma unroll
    for (int c = 0; c < 2 * D - 1; ++c) t[c] = 0;
    #pragma unroll
    for (int a = 0; a < D; ++a)
        #pragma unroll
        for (int b = 0; b < D; ++b) t[a + b] = add_p(t[a + b], mont_mul(g[a], v[b]));
    #pragma unroll
    for (int c = 0; c < D; ++c) {
        u32 o = t[c];
        if (c + D < 2 * D - 1) {
            unsigned long long x = (unsigned long long)t[c + D] * W_NR;
            o = add_p(o, (u32)(x %% P));
        }
        q[(size_t)c * 2 * k + j] = add_p(q[(size_t)c * 2 * k + j], o);
    }
}

extern "C" __global__ void add_mod_inplace(u32* __restrict__ q, const u32* __restrict__ x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) q[i] = add_p(q[i], x[i]);
}
"""


def mont(a: np.ndarray) -> np.ndarray:
    return ((np.asarray(a, dtype=U64) % U64(P)) * U64(bb.R_MOD_P) % U64(P)).astype(np.uint32)


class StructuredLinearTest:
    """GPU shortcut.  ``__init__`` compiles for ``M``; ``prepare(E)`` encodes the fixed rows (once per batch);
    ``__call__(W, s_mont, exceptional)`` returns ``q`` ``(6, 2k)`` plain, the same object as ``Combiner.reduce`` of the general path."""

    def __init__(self, enc, M: int, chunk_rows: int = 1024, block: int = 128, k: int = K):
        import cupy as cp

        from encode import buckets as bk

        self.cp, self.enc, self.M, self.k, self.chunk_rows, self.block = cp, enc, M, k, chunk_rows, block
        self.mod = cp.RawModule(code=_SRC % {"pinv": bb.P_INV_MOD_R, "m": M}, options=("-std=c++17",))
        self.k_combine = self.mod.get_function("combine_multi")
        self.k_reduce = self.mod.get_function("reduce_multi")
        self.k_point = self.mod.get_function("fc_pointwise_accum")
        self.k_add = self.mod.get_function("add_mod_inplace")
        self.generic = bk.Combiner(chunk_rows=32)
        self.r2 = np.uint32(bb.R2_MOD_P)
        self.G_sys = self.G_c1 = None

    def prepare(self, E: np.ndarray):
        """``E`` ``(M, 6, k)`` plain -> device systematic + coset-1 evaluations of every fixed row (6 M encoder calls)."""
        cp = self.cp
        self.G_sys = cp.asarray(np.ascontiguousarray(E).astype(np.uint32))                        # (M, 6, k)
        self.G_c1 = cp.empty((self.M, D, self.k), dtype=cp.uint32)
        for m in range(self.M):
            self.enc(self.G_sys[m], self.G_c1[m], write_systematic=False, ncosets=1)

    def nchunks(self, rows: int) -> int:
        return -(-rows // self.chunk_rows)

    def combine(self, W, s_mont, acc):
        rows, ncols = W.shape
        grid = ((ncols + self.block - 1) // self.block, self.nchunks(rows))
        self.k_combine(grid, (self.block,), (W, s_mont, acc, np.int32(rows), np.int32(self.chunk_rows), np.int32(ncols)))

    def reduce(self, acc, out):
        nchunk, md, ncols = acc.shape
        total = md * ncols
        self.k_reduce(((total + 255) // 256,), (256,), (acc, out, np.int32(nchunk), np.int32(ncols)))

    def pointwise(self, V_sys, V_c1, q):
        """``q (6, 2k) += G_m * V_m`` for every ``m``; ``V_* (M, 6, k)``."""
        grid = ((2 * self.k + self.block - 1) // self.block,)
        for m in range(self.M):
            self.k_point(grid, (self.block,), (self.G_sys[m], self.G_c1[m], V_sys[m], V_c1[m], q, np.int32(self.k), self.r2))

    def workspace(self, rows: int, n_exceptional: int) -> dict:
        cp = self.cp
        ws = {"acc": cp.zeros((self.nchunks(rows), self.M * D, self.k), dtype=cp.uint32),
              "V_sys": cp.empty((self.M, D, self.k), dtype=cp.uint32), "V_c1": cp.empty((self.M, D, self.k), dtype=cp.uint32),
              "q": cp.zeros((D, 2 * self.k), dtype=cp.uint32)}
        if n_exceptional:
            X = n_exceptional
            ws.update({"x_acc": cp.zeros((self.generic.nchunks(X), D, 2 * self.k), dtype=cp.uint32),
                       "x_u_c1": cp.empty((X, self.k), dtype=cp.uint32), "x_rA_c1": cp.empty((D, X, self.k), dtype=cp.uint32),
                       "x_q": cp.empty((D, 2 * self.k), dtype=cp.uint32)})
        return ws

    def run(self, W, s_mont, ws: dict, x_rows=None, x_W=None, x_rA=None, stream=None):
        """One linear test.  ``W`` ``(rows, k)`` plain device; ``s_mont`` ``(rows, M, 6)`` Montgomery; exceptional rows: ``x_W``
        ``(X, k)`` (their witness rows), ``x_rA`` ``(6, X, k)`` plain (their dense ``r^T A`` rows) -> ``ws['q']`` ``(6, 2k)`` plain."""
        ws["acc"].fill(0)
        ws["q"].fill(0)
        self.combine(W, s_mont, ws["acc"])
        self.reduce(ws["acc"], ws["V_sys"].reshape(self.M * D, self.k))
        for m in range(self.M):
            self.enc(ws["V_sys"][m], ws["V_c1"][m], write_systematic=False, ncosets=1)
        self.pointwise(ws["V_sys"], ws["V_c1"], ws["q"])
        if x_W is not None and x_W.shape[0]:
            X = x_W.shape[0]
            ws["x_acc"].fill(0)
            self.enc(x_W, ws["x_u_c1"][:X], write_systematic=False, ncosets=1)
            for c in range(D):
                self.enc(x_rA[c], ws["x_rA_c1"][c, :X], write_systematic=False, ncosets=1)
            self.generic.linear(x_W, ws["x_u_c1"], x_rA, ws["x_rA_c1"], ws["x_acc"])
            self.generic.reduce(ws["x_acc"], ws["x_q"])
            n = D * 2 * self.k
            self.k_add(((n + 255) // 256,), (256,), (ws["q"], ws["x_q"], np.int32(n)))
        return ws["q"]


def general_path_gpu(enc, W_d, rA_d):
    """The b-commit lane's path (the reference kernel): 7 encoder calls per row + ``Combiner.linear`` -> ``q`` ``(6, 2k)``."""
    import cupy as cp

    from encode import buckets as bk

    rows = W_d.shape[0]
    cb = bk.Combiner(chunk_rows=32)
    u_c1 = enc(W_d, cp.empty((rows, K), dtype=cp.uint32), write_systematic=False, ncosets=1)
    rA_c1 = cp.empty((D, rows, K), dtype=cp.uint32)
    for c in range(D):
        enc(rA_d[c], rA_c1[c], write_systematic=False, ncosets=1)
    acc = cb.acc(rows, 2 * K)
    cb.linear(W_d, u_c1, rA_d, rA_c1, acc)
    return cb.reduce(acc)


def split_structured(st: StructuredRTA, W: np.ndarray):
    """Host arrays for ``StructuredLinearTest.run``: ``(s_mont (rows, M, 6) u32, x_idx, x_W (X, k), x_rA (6, X, k))``."""
    idx = sorted(st.exceptional)
    x_W = np.asarray(W, dtype=np.uint32)[idx] if idx else np.zeros((0, st.E.shape[2]), np.uint32)
    x_rA = (np.stack([np.stack([st.exceptional[i][c] for i in idx]) for c in range(D)]).astype(np.uint32) if idx
            else np.zeros((D, 0, st.E.shape[2]), np.uint32))
    return mont(st.s), idx, x_W, x_rA


# -- tensor-core variant of the row combination: INT8 limbs, torch._int_mm ---------------------------------------------------
def combine_int8_tc(W_t, s_plain: np.ndarray, chunk_rows: int = 65536, xor_done: bool = False, W8=None, A_t=None):
    """``V[c][j] = sum_i s[i][c] W[i][j] mod p`` on the tensor cores.  ``W_t`` torch ``(rows, k)`` int32 view of the u32 rows;
    the four bytes of a row element are its base-2^8 limbs, so ``W.view(int8)`` is a zero-copy ``(rows, 4k)`` limb matrix;
    ``x ^ 0x80`` per byte turns the unsigned byte into ``byte - 128`` (int8), a one-time pass unless the witness is stored so.
    ``s`` limbs likewise (``(24 + 1) x rows``: 6 coordinates x 4 limbs, shifted by 128, plus an all-ones row for the shift
    correction, padded to 32).  Exact integer arithmetic: INT32 accumulation over ``chunk_rows <= 2^17`` rows (|product| <=
    2^14), INT64 across chunks, limb recombination ``sum_{a,b} 2^(8(a+b)) * total_ab mod p``.  Returns ``(6, k)`` int64 plain."""
    import torch

    rows, k = W_t.shape
    dev = W_t.device
    if W8 is None:
        W8 = W_t.view(torch.int8) if xor_done else (W_t ^ torch.tensor(-0x7F7F7F80, dtype=torch.int32, device=dev)).view(torch.int8)
    # -0x7F7F7F80 as int32 has bit pattern 0x80808080
    if A_t is None:
        A_t = s_limb_matrix(s_plain).to(dev)
    n = A_t.shape[0]
    tot = torch.zeros((n, 4 * k), dtype=torch.int64, device=dev)
    main = rows - rows % 8                                                               # _int_mm wants the inner dim % 8 == 0
    for r0 in range(0, main, chunk_rows):
        r1 = min(main, r0 + chunk_rows)
        C = torch._int_mm(A_t[:, r0:r1].contiguous(), W8[r0:r1])                         # (n, 4k) int32, exact (|ab| <= 2^14, <= 2^17 rows)
        tot += C
    if main < rows:                                                                      # < 8 tail rows: exact int64 GEMM on the host (tiny)
        tail = A_t[:, main:].cpu().to(torch.int64) @ W8[main:].cpu().to(torch.int64)
        tot += tail.to(dev)
    M = 1 if np.ndim(s_plain) == 2 else np.shape(s_plain)[1]
    return recombine_limbs(tot, A_t, k, M)


def s_limb_matrix(s_plain: np.ndarray):
    """``s`` ``(rows, 6)`` or ``(rows, M, 6)`` plain -> ``(n, rows)`` int8: row ``(m * 4 + a) * 6 + c`` = limb ``a`` of coordinate ``c`` of
    ``s[:, m]`` minus 128; row ``24 M`` = ones (the shift correction); zero-padded to ``n`` = a multiple of 8, ``>= 32``."""
    import torch

    s = np.asarray(s_plain, dtype=np.int64)
    if s.ndim == 2:
        s = s[:, None, :]
    rows, M, _ = s.shape
    limbs = np.stack([(s >> (8 * a)) & 0xFF for a in range(4)], axis=2)                  # (rows, M, 4, 6)
    n = max(32, -(-(24 * M + 1) // 8) * 8)
    A = np.zeros((n, rows), dtype=np.int8)
    A[:24 * M] = (limbs.reshape(rows, 24 * M).T - 128).astype(np.int8)
    A[24 * M] = 1
    return torch.from_numpy(A)


def recombine_limbs(tot, A_t, k: int, M: int = 1):
    """``sum_i s_a[i] W_b[i][j] = C[a] + 128 C[ones] + 128 sum_i s_a[i]`` (``W_b = W8_b + 128``, ``s_a = A[a] + 128``), then
    ``V[m][c][j] = sum_{a,b} 2^(8(a+b)) (.) mod p``; vectorised torch int64, every term reduced before the 16-term sum.
    Returns ``(6, k)`` for ``M = 1``, else ``(M, 6, k)``."""
    import torch

    dev = tot.device
    ssum = (A_t[:24 * M].to(torch.int64) + 128).sum(dim=1).view(M, 4, D)                 # sum_i s_a[i], (m, a, c)
    T = tot[:24 * M].view(M, 4, D, k, 4)                                                 # (m, a, c, j, b)
    ones = tot[24 * M].view(k, 4)                                                        # column sums of W8, (j, b)
    t = T + 128 * ones[None, None, None] + 128 * ssum[:, :, :, None, None]               # exact, |t| < 2^41
    coef = torch.tensor([[pow(2, 8 * (a + b), P) for b in range(4)] for a in range(4)], dtype=torch.int64, device=dev)
    terms = ((t % P) * coef[None, :, None, None, :]) % P                                 # < p each
    out = terms.sum(dim=(1, 4)) % P                                                      # (m, c, j)
    return out[0] if M == 1 else out


def combine_int8_tc_ref(W: np.ndarray, s: np.ndarray) -> np.ndarray:
    return ref.coords_times_rows(s, W)


# -- bench --------------------------------------------------------------------------------------------------------------------
ROWS_B = 310_602
PEAK_BW = 1.008e12


def _timed(fn, reps: int):
    import cupy as cp

    fn()
    cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        s, e = cp.cuda.Event(), cp.cuda.Event()
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(cp.cuda.get_elapsed_time(s, e))
    return statistics.median(ts), ts


def checks(enc) -> dict:
    """Bit-exactness on small shapes: (a) toy chain system: structured == dense r^T A; (b) numpy shortcut == numpy general;
    (c) GPU shortcut == the b-commit GPU path == numpy, with exceptional rows; (d) INT8 tensor-core combination == numpy."""
    import cupy as cp

    out = {}
    rng = np.random.default_rng(0)
    sysm = UnitPatternSystem.random(U=12, w=7, C=5, transition=(3, 4), nb=2, S=2, k=16, l=14, seed=1)
    rho = [int(v) for v in rng.integers(0, P, D)]
    st = sysm.structured(rho)
    out["toy_structure_matches_dense_rta"] = bool(np.array_equal(st.expand(), sysm.dense_rta(rho)))
    out["toy_exceptional_rows"] = sorted(st.exceptional)
    rows, M, X = 40, 2, 3
    st = random_structured(rows, M, X, seed=2)
    W = bb.fill_rows(rows, seed=3)
    rA = st.expand()
    q_np_general = ref.linear_test(rA, W)
    q_np_struct = structured_linear_test_ref(st, W)
    out["numpy_shortcut_equals_numpy_general"] = bool(np.array_equal(q_np_general, q_np_struct))
    Wd = cp.asarray(W)
    q_gpu_general = cp.asnumpy(general_path_gpu(enc, Wd, cp.asarray(rA.astype(np.uint32)))).astype(U64)
    out["gpu_general_equals_numpy"] = bool(np.array_equal(q_gpu_general, q_np_general))
    slt = StructuredLinearTest(enc, M, chunk_rows=8)
    slt.prepare(st.E)
    s_mont, idx, x_W, x_rA = split_structured(st, W)
    ws = slt.workspace(rows, len(idx))
    q_gpu = slt.run(Wd, cp.asarray(s_mont), ws, x_W=cp.asarray(x_W), x_rA=cp.asarray(x_rA))
    out["gpu_shortcut_equals_gpu_general"] = bool(np.array_equal(cp.asnumpy(q_gpu).astype(U64), q_gpu_general))
    try:
        import torch

        W42 = bb.fill_rows(42, seed=11)                                                    # 42 % 8 != 0: exercises the tail path
        W_t = torch.from_numpy(W42.view(np.int32)).cuda()
        s = rng.integers(0, P, (42, D)).astype(U64)
        got = combine_int8_tc(W_t, s).cpu().numpy().astype(U64)
        ok = np.array_equal(got, combine_int8_tc_ref(W42, s))
        s2 = rng.integers(0, P, (42, 2, D)).astype(U64)
        got2 = combine_int8_tc(W_t, s2).cpu().numpy().astype(U64)
        ok = ok and all(np.array_equal(got2[m], combine_int8_tc_ref(W42, s2[:, m])) for m in range(2))
        out["int8_tc_combine_equals_numpy"] = bool(ok)
    except Exception as e:                                                                 # noqa: BLE001
        out["int8_tc_combine_equals_numpy"] = f"error: {e}"
    return out


def _bench_structured(enc, W, R: int, M: int, X: int, a, stream, bbench) -> dict:
    import cupy as cp

    slt = StructuredLinearTest(enc, M, chunk_rows=a.chunk_rows)
    E = np.random.default_rng(5).integers(0, P, (M, D, K)).astype(U64)
    E[:, :, L_PAYLOAD:] = 0
    t_prep, _ = _timed(lambda: slt.prepare(E), 3)
    s_mont = cp.random.RandomState(2).randint(0, P, size=(R, M, D), dtype=cp.int32).view(cp.uint32)
    x_W = W[:X] if X else None
    x_rA = cp.random.RandomState(3).randint(0, P, size=(D, X, K), dtype=cp.int32).view(cp.uint32) if X else None
    ws = slt.workspace(R, X)
    with stream:
        g = bbench.graph_of(lambda: slt.run(W, s_mont, ws, x_W=x_W, x_rA=x_rA), stream)
        med, ts = _timed(lambda: g.launch(stream=stream), a.reps)

        def comb():
            ws["acc"].fill(0)
            slt.combine(W, s_mont, ws["acc"])
            slt.reduce(ws["acc"], ws["V_sys"].reshape(M * D, K))
        t_comb, _ = _timed(comb, a.reps)
        t_enc, _ = _timed(lambda: [slt.enc(ws["V_sys"][m], ws["V_c1"][m], write_systematic=False, ncosets=1) for m in range(M)], a.reps)
        t_pt, _ = _timed(lambda: slt.pointwise(ws["V_sys"], ws["V_c1"], ws["q"]), a.reps)
    r = {"name": f"linear_test_structured_M{M}_X{X}", "bucket": "linear_test_structured", "shape": "B", "rows": R, "M": M, "exceptional_rows": X,
         "chunk_rows": a.chunk_rows, "ms_median": med, "ms_all": ts, "us_per_row": med * 1e3 / R, "transforms_per_row": 0,
         "transforms_per_batch": 6 * M + 6 * M + 7 * X, "encoder_calls_per_row": 0,
         "split_ms": {"prepare_G_once_per_batch": t_prep, "combine_multi": t_comb, "encode_V": t_enc, "pointwise": t_pt,
                      "exceptional_generic": max(0.0, med - t_comb - t_enc - t_pt)},
         "read_floor_ms": R * K * 4 / PEAK_BW * 1e3, "device_memory_in_use_bytes": bbench.mem()}
    r["fraction_of_read_floor"] = r["read_floor_ms"] / med
    print(f"B {r['name']}: {med:.2f} ms ({r['us_per_row']:.4f} us/row); combine {t_comb:.2f} + encode {t_enc:.2f} + pointwise {t_pt:.3f} "
          f"+ exceptional {r['split_ms']['exceptional_generic']:.2f}; G prep {t_prep:.2f} ms; read floor {r['read_floor_ms']:.1f} ms", flush=True)
    return r


def bench(a) -> dict:
    import cupy as cp

    from encode import buckets_bench as bbench
    from encode.encode_ntt_simt import RSEncoderSIMT

    enc = RSEncoderSIMT()
    res = {"device": bbench.device_info(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(a), "results": [],
           "bench_kind": "bucket", "bucket": "linear_test",
           "scope": "component: B-Ligero linear test q_add through the structure of r^T A (powers-of-rho combination, per-unit "
                    "constraint pattern): M fixed F_c rows + per-row F_c scalars + X exceptional rows, measured standalone on "
                    "resident pseudo-random data; the r^T A scalars/pattern generation is not included (it is O(rows + |pattern|))"}
    res["checks"] = checks(enc)
    print("checks:", json.dumps(res["checks"]), flush=True)
    R = a.rows
    W = cp.random.RandomState(1).randint(0, P, size=(R, K), dtype=cp.int32).view(cp.uint32)
    stream = cp.cuda.Stream(non_blocking=True)
    for M in a.M:
        for X in a.exceptional:
            res["results"].append(_bench_structured(enc, W, R, M, X, a, stream, bbench))
            bbench.free_all()
    # tensor-core INT8 limb combination (M = 1), with and without the byte-xor pass
    if a.tc:
        try:
            import torch

            W_t = torch.from_dlpack(W.view(cp.int32))                                   # zero-copy view of the resident rows

            xor_c = torch.tensor(-0x7F7F7F80, dtype=torch.int32, device=W_t.device)
            W8 = (W_t ^ xor_c).view(torch.int8)
            torch.cuda.synchronize()

            def tm(fn, reps):
                fn()
                torch.cuda.synchronize()
                ts = []
                for _ in range(reps):
                    s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s0.record()
                    fn()
                    e0.record()
                    torch.cuda.synchronize()
                    ts.append(s0.elapsed_time(e0))
                return statistics.median(ts), ts
            t_xor, _ = tm(lambda: (W_t ^ xor_c).view(torch.int8), a.reps)
            for M in a.M:
                sM = np.random.default_rng(6).integers(0, P, (R, M, D)).astype(U64)
                A_t = s_limb_matrix(sM).to(W_t.device)                                   # the s limbs: prepared once (24 M B / row)
                n = A_t.shape[0]
                for chunk in ((32768, 65536, 131072) if M == 1 else (65536,)):
                    med, ts = tm(lambda chunk=chunk, A_t=A_t, sM=sM: combine_int8_tc(W_t, sM, chunk_rows=chunk, W8=W8, A_t=A_t), a.reps)
                    res["results"].append({"name": f"combine_int8_tensorcore_M{M}_chunk{chunk}", "bucket": "combine_int8_tensorcore", "shape": "B",
                                           "rows": R, "M": M, "n_limb_rows": n, "chunk_rows": chunk,
                                           "ms_median": med, "ms_all": ts, "xor_pass_ms": t_xor, "read_floor_ms": R * K * 4 / PEAK_BW * 1e3,
                                           "fraction_of_read_floor": R * K * 4 / PEAK_BW * 1e3 / med,
                                           "limb_macs": n * R * 4 * K, "t_limb_macs_per_s": n * R * 4 * K / (med / 1e3) / 1e12,
                                           "note": f"torch._int_mm, A = {n} x rows int8 ({24 * M} s-limb rows shifted by 128 + ones row), B = W bytes ^ 0x80; "
                                                   "INT32 per chunk, INT64 across chunks, limb recombination mod p; bit-exact vs numpy (checks)"})
                    print(f"B combine_int8_tensorcore M{M} chunk {chunk}: {med:.2f} ms (+ xor pass {t_xor:.2f} ms), "
                          f"{res['results'][-1]['t_limb_macs_per_s']:.1f} T limb-MAC/s, {res['results'][-1]['fraction_of_read_floor']:.2f} of read floor", flush=True)
        except Exception as e:                                                             # noqa: BLE001
            res["tc_error"] = repr(e)
            print("tensor-core variant failed:", repr(e), flush=True)
        finally:
            W_t = W8 = A_t = None                                                          # noqa: F841 - release the torch views before cupy allocates again
            import torch

            torch.cuda.empty_cache()
    bbench.free_all()
    # the general path (the b-commit kernels), same run, same device, for the before/after table
    if a.baseline:
        res["results"] += bbench.bench_linear("B", R, a.reps, enc, slice_list=(256,))
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["run_id"] = os.environ.get("RESEARCH_RUN_ID")
    res["all_bit_exact"] = all(v is True for k_, v in res["checks"].items() if k_ != "toy_exceptional_rows")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="lintest_bench.json")
    ap.add_argument("--rows", type=int, default=ROWS_B)
    ap.add_argument("--M", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--exceptional", type=int, nargs="+", default=[0, 256])
    ap.add_argument("--chunk-rows", type=int, default=1024)
    ap.add_argument("--no-tc", dest="tc", action="store_false")
    ap.add_argument("--baseline", action="store_true", help="also time the b-commit general path (14 transforms/row) in this run")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        a.rows, a.reps, a.M, a.exceptional = 20_000, 3, [1, 2], [0, 64]
    res = bench(a)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        Path(rd, "lintest_bench.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

"""The one B-Ligero bucket kernel the checker-v2 shape adds (b-v2 lane): the quadratic test over ``F_c`` triples.

At v2 the lookup helper identity ``h_j (z_T - w_j(beta_T)) = 1`` (``PROTOCOL.md`` 4.2) is the dominant quadratic
constraint: 95 per unit against 34 ``F_t`` products.  ``h_j`` is an ``F_c`` element (6 committed coordinate rows in the
transposed layout, ``V2_SHAPE.md``); ``w_j(beta_T) = sum_c beta^c w_{j,c}`` is an ``F_c``-linear combination of the query's
``C`` witness rows (the table's columns: key and outputs), so ``y_j = z_T - w_j(beta_T)`` and ``z_j = 1`` are VIRTUAL rows
-- the verifier evaluates them at the opened columns from the opened witness rows -- and the triple ``(h_j, y_j, 1)`` is a
product OVER ``F_c``: the Ligero quadratic test with ``F_c`` arithmetic at every point of the 8,192-point domain
(the code over ``F_c`` is the coordinate-wise code over ``F_t``; interpolation is linear).

``p_0[j] += r_i * ( H_i[j] * Y_i[j] - ONE[j] e_0 )`` with ``Y_i[j] = z_T ONE[j] - sum_c beta^c W_{i,c}[j]``, ``ONE`` the
interpolant of ones on the payload and zeros on the randomizers (``ONE[j] = [j < l]`` on coset 0; its coset-1 evaluation
is one fixed row per batch).  Per point per triple: ``6 C`` mont-muls (``Y``), 36 (``H Y``), 36 (``r * .``) + 6 (``r * ONE``).
Every operand plain except the constants (``beta`` coordinates, ``z_T``, ``r R`` / ``r R^2``) in Montgomery form; ``H Y``
comes out as ``H Y R^-1`` and is multiplied by ``r R^2`` so no extra reduction is needed (as ``combine_quadratic``).

Reference: ``quadratic_fc_ref`` (numpy uint64, ``lintest_structured.fc_mul_np``).
"""
from __future__ import annotations

import numpy as np

from . import babybear as bb
from . import buckets as bk
from . import buckets_ref as ref

D = 6
U64 = np.uint64
P = bb.P

_SRC = bk._COMMON + r"""
#define C %(cols)d
/* (a plain-ish, b Montgomery-ish): out[c] = sum_{i+j=c} mont_mul(a_i, b_j) + 31 * sum_{i+j=c+6} ... over F_c = F_p[X]/(X^6 - 31) */
__device__ __forceinline__ void fc_mul_mm(const u32* a, const u32* b, u32* out) {
    u32 t[2 * D - 1];
    #pragma unroll
    for (int i = 0; i < 2 * D - 1; ++i) t[i] = 0;
    #pragma unroll
    for (int i = 0; i < D; ++i)
        #pragma unroll
        for (int j = 0; j < D; ++j) t[i + j] = add_p(t[i + j], mont_mul(a[i], b[j]));
    #pragma unroll
    for (int i = 0; i < D; ++i) out[i] = t[i];
    #pragma unroll
    for (int i = D; i < 2 * D - 1; ++i) {
        unsigned long long v = (unsigned long long)t[i] * W_NR;
        out[i - D] = add_p(out[i - D], (u32)(v %% P));
    }
}

/* Thread (j, chunk), j < 2K.  H_in / H_c1: (D, rows, K) plain (helper coordinate rows, systematic / coset 1);
   W_in / W_c1: (C, rows, K) plain (the query's witness rows);  beta: (C, D) Montgomery = coordinates of beta^c;
   z: (D) Montgomery;  one_c1: (K) plain = coset-1 evaluation of the ONE row;  r: (rows, 2D) = [r R, r R^2];
   acc: (nchunk, D, 2K). */
extern "C" __global__ void combine_quadratic_fc(const u32* __restrict__ H_in, const u32* __restrict__ H_c1,
                                                const u32* __restrict__ W_in, const u32* __restrict__ W_c1,
                                                const u32* __restrict__ beta, const u32* __restrict__ z,
                                                const u32* __restrict__ one_c1, const u32* __restrict__ r,
                                                u32* __restrict__ acc, int rows, int chunk_rows, int l_payload)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = blockIdx.y;
    if (j >= 2 * K) return;
    const bool sys = j < K;
    const int jj = sys ? j : j - K;
    const u32* H = sys ? H_in : H_c1;
    const u32* W = sys ? W_in : W_c1;
    const u32 one = sys ? (jj < l_payload ? 1u : 0u) : __ldg(&one_c1[jj]);
    u32 zc[D], bt[C][D];
    #pragma unroll
    for (int c = 0; c < D; ++c) zc[c] = __ldg(&z[c]);
    #pragma unroll
    for (int q = 0; q < C; ++q)
        #pragma unroll
        for (int c = 0; c < D; ++c) bt[q][c] = __ldg(&beta[q * D + c]);
    int i0 = chunk * chunk_rows, i1 = min(rows, i0 + chunk_rows);
    u32 a[D];
    #pragma unroll
    for (int c = 0; c < D; ++c) a[c] = 0;
    for (int i = i0; i < i1; ++i) {
        u32 h[D], y[D], w[C], hy[D], t[D], rr[D], r2[D];
        #pragma unroll
        for (int c = 0; c < D; ++c) h[c] = __ldg(&H[((size_t)c * rows + i) * K + jj]);
        #pragma unroll
        for (int q = 0; q < C; ++q) w[q] = __ldg(&W[((size_t)q * rows + i) * K + jj]);
        /* y_c = z_c one - sum_q beta[q][c] w_q  (plain) */
        #pragma unroll
        for (int c = 0; c < D; ++c) {
            u32 s = mont_mul(one, zc[c]);
            #pragma unroll
            for (int q = 0; q < C; ++q) s = sub_p(s, mont_mul(w[q], bt[q][c]));
            y[c] = s;
        }
        fc_mul_mm(h, y, hy);                                   /* (H Y) R^-1 */
        #pragma unroll
        for (int c = 0; c < D; ++c) { rr[c] = __ldg(&r[(size_t)i * 2 * D + c]); r2[c] = __ldg(&r[(size_t)i * 2 * D + D + c]); }
        fc_mul_mm(hy, r2, t);                                  /* r (H Y) */
        #pragma unroll
        for (int c = 0; c < D; ++c) a[c] = add_p(a[c], sub_p(t[c], mont_mul(one, rr[c])));   /* - r * (one, 0, ..., 0) */
    }
    #pragma unroll
    for (int c = 0; c < D; ++c) {
        u32* o = acc + ((size_t)chunk * D + c) * (2 * K) + j;
        *o = add_p(*o, a[c]);
    }
}
"""


_INVERSE_BLOCKED = bk._INVERSE + r"""
#define G %(g)d
/* Blocked Montgomery trick (b-v2): thread t owns d[s T + t] (s < m) as before, but stores one prefix per GROUP of G
   consecutive s instead of one per element.  Forward: running product over the group, prefix_g[q T + t] = product of
   everything up to the end of group q.  Backward: the group's G elements are re-read into registers, the within-group
   prefixes recomputed (G - 1 extra F_c products per group), then out[s] = prefix(s - 1) * inv, inv *= d[s] as before.
   Traffic (3 + 2/G) x 24 N bytes instead of 5 x 24 N; F_c products ~ (4 - 1/G) per element instead of 3. */
extern "C" __global__ void batch_inverse_blocked(const u32* __restrict__ d, u32* __restrict__ prefix_g, u32* __restrict__ out,
                                                 long long N, int m, const u32* __restrict__ exp_words, int nbits, u32 one_mont)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long T = (long long)gridDim.x * blockDim.x;
    if (t >= T) return;
    const int ngroups = (m + G - 1) / G;
    const long long NG = (long long)ngroups * T;       /* prefix_g is (D, NG) */
    u32 run[D], e[D], tmp[D];
    #pragma unroll
    for (int i = 0; i < D; ++i) run[i] = 0;
    run[0] = one_mont;
    for (int q = 0; q < ngroups; ++q) {
        #pragma unroll
        for (int gi = 0; gi < G; ++gi) {
            long long idx = ((long long)q * G + gi) * T + t;
            if (idx < N) {
                #pragma unroll
                for (int i = 0; i < D; ++i) e[i] = __ldg(&d[(size_t)i * N + idx]);
                fc_mul(run, e, tmp);
                #pragma unroll
                for (int i = 0; i < D; ++i) run[i] = tmp[i];
            }
        }
        #pragma unroll
        for (int i = 0; i < D; ++i) prefix_g[(size_t)i * NG + (size_t)q * T + t] = run[i];
    }
    u32 inv[D];
    fc_inv(run, inv, exp_words, nbits, one_mont);
    for (int q = ngroups - 1; q >= 0; --q) {
        u32 base[D], el[G][D], pf[G][D];
        if (q == 0) {
            #pragma unroll
            for (int i = 0; i < D; ++i) base[i] = 0;
            base[0] = one_mont;
        } else {
            #pragma unroll
            for (int i = 0; i < D; ++i) base[i] = prefix_g[(size_t)i * NG + (size_t)(q - 1) * T + t];
        }
        /* recompute the within-group prefixes pf[gi] = base * e_0 ... e_gi (elements beyond N count as 1) */
        #pragma unroll
        for (int gi = 0; gi < G; ++gi) {
            long long idx = ((long long)q * G + gi) * T + t;
            const u32* prev = gi == 0 ? base : pf[gi - 1];
            if (idx < N) {
                #pragma unroll
                for (int i = 0; i < D; ++i) el[gi][i] = __ldg(&d[(size_t)i * N + idx]);
                fc_mul(prev, el[gi], pf[gi]);
            } else {
                #pragma unroll
                for (int i = 0; i < D; ++i) { el[gi][i] = 0; pf[gi][i] = prev[i]; }
                el[gi][0] = one_mont;
            }
        }
        #pragma unroll
        for (int gi = G - 1; gi >= 0; --gi) {
            long long idx = ((long long)q * G + gi) * T + t;
            if (idx < N) {
                const u32* prev = gi == 0 ? base : pf[gi - 1];
                fc_mul(prev, inv, tmp);
                #pragma unroll
                for (int i = 0; i < D; ++i) out[(size_t)i * N + idx] = tmp[i];
                fc_mul(inv, el[gi], tmp);
                #pragma unroll
                for (int i = 0; i < D; ++i) inv[i] = tmp[i];
            }
        }
    }
}
"""


class BatchInverterBlocked:
    """``BatchInverter`` with ``G`` elements per stored prefix (``_INVERSE_BLOCKED``); same interface and result."""

    def __init__(self, g: int = 4, threads_total: int = 128 * 1024, block: int = 256):
        import cupy as cp

        self.cp, self.T, self.block, self.g = cp, threads_total, block, g
        self.mod = cp.RawModule(code=_INVERSE_BLOCKED % {**bk._params(), "g": g}, options=("-std=c++17",))
        self.kernel = self.mod.get_function("batch_inverse_blocked")
        e = bb.P**D - 2
        self.nbits = e.bit_length()
        self.exp_words = cp.asarray(np.array([(e >> (32 * i)) & 0xFFFFFFFF for i in range(-(-self.nbits // 32))], dtype=np.uint32))
        self.one_mont = np.uint32(bb.R_MOD_P)
        self.scratch = None

    def __call__(self, d_mont, out=None):
        cp = self.cp
        N = d_mont.shape[1]
        m = -(-N // self.T)
        ngroups = -(-m // self.g)
        need = (D, max(ngroups * self.T, 1))
        if self.scratch is None or self.scratch.shape != need:
            self.scratch = cp.empty(need, dtype=cp.uint32)
        if out is None:
            out = cp.empty_like(d_mont)
        self.kernel((self.T // self.block,), (self.block,), (d_mont, self.scratch, out, np.int64(N), np.int32(m), self.exp_words,
                                                             np.int32(self.nbits), self.one_mont))
        return out

    def bytes_moved(self, N: int) -> int:
        return int((3 + 2 / self.g) * D * N * 4)


def fc_mul_np(a, b):
    from .lintest_structured import fc_mul_np as f

    return f(a, b)


def one_row(k: int = bb.K_CODE, l: int = 3840) -> np.ndarray:
    o = np.zeros((1, k), dtype=np.uint32)
    o[0, :l] = 1
    return o


def beta_coords(beta: list[int], cols: int) -> np.ndarray:
    """``(cols, 6)`` plain: coordinates of ``beta^c`` for ``c < cols`` (``beta^0 = 1``)."""
    out = np.zeros((cols, D), dtype=U64)
    cur = np.array([1, 0, 0, 0, 0, 0], dtype=U64)
    for c in range(cols):
        out[c] = cur
        cur = fc_mul_np(cur, np.asarray(beta, dtype=U64))
    return out


def quadratic_fc_ref(r: np.ndarray, H: np.ndarray, W: np.ndarray, beta: np.ndarray, z: np.ndarray, l: int = 3840) -> np.ndarray:
    """``r (rows, 6)``, ``H (6, rows, k)``, ``W (C, rows, k)``, ``beta (C, 6)``, ``z (6)`` plain -> ``p_0 (6, 2k)``."""
    rows = H.shape[1]
    He = np.stack([ref.encode2(H[c].astype(np.uint32)) for c in range(D)], axis=-1).astype(U64)       # (rows, 2k, 6)
    We = np.stack([ref.encode2(W[q].astype(np.uint32)) for q in range(W.shape[0])], axis=-1).astype(U64)  # (rows, 2k, C)
    one = ref.encode2(one_row(l=l)).astype(U64)[0]                                                     # (2k,)
    Y = (one[None, :, None] * np.asarray(z, dtype=U64)[None, None, :]) % U64(P)
    for q in range(W.shape[0]):
        Y = (Y + U64(P) - (We[:, :, q][:, :, None] * np.asarray(beta[q], dtype=U64)[None, None, :]) % U64(P)) % U64(P)
    HY = fc_mul_np(He, Y)                                                                             # (rows, 2k, 6)
    Z = np.zeros_like(HY)
    Z[:, :, 0] = one[None, :]
    term = (HY + U64(P) - Z) % U64(P)
    out = np.zeros((2 * bb.K_CODE, D), dtype=U64)
    for i in range(rows):
        out = (out + fc_mul_np(np.asarray(r[i], dtype=U64)[None, :], term[i])) % U64(P)
    return out.T.copy()


class QuadraticFc:
    """``cols`` witness rows per triple (the table's column count); compiled per ``cols``."""

    def __init__(self, cols: int, k: int = bb.K_CODE, chunk_rows: int = 32, block: int = 128):
        import cupy as cp

        self.cp, self.k, self.cols, self.chunk_rows, self.block = cp, k, cols, chunk_rows, block
        self.mod = cp.RawModule(code=_SRC % {"pinv": bb.P_INV_MOD_R, "k": k, "cols": cols}, options=("-std=c++17",))
        self.kernel = self.mod.get_function("combine_quadratic_fc")
        self.reduce_mod = bk.Combiner(k=k, chunk_rows=chunk_rows)

    def nchunks(self, rows: int) -> int:
        return -(-rows // self.chunk_rows)

    def acc(self, rows: int):
        return self.cp.zeros((self.nchunks(rows), D, 2 * self.k), dtype=self.cp.uint32)

    def __call__(self, H_in, H_c1, W_in, W_c1, beta_mont, z_mont, one_c1, r12, acc, l: int = 3840):
        rows = H_in.shape[1]
        grid = ((2 * self.k + self.block - 1) // self.block, self.nchunks(rows))
        self.kernel(grid, (self.block,), (H_in, H_c1, W_in, W_c1, beta_mont, z_mont, one_c1, r12, acc,
                                          np.int32(rows), np.int32(self.chunk_rows), np.int32(l)))

    def reduce(self, acc, out=None):
        return self.reduce_mod.reduce(acc, out)

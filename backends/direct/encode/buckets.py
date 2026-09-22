"""B-Ligero's remaining prover buckets as standalone CUDA kernels (b-commit lane), bit-exact against ``buckets_ref``.

Everything is BabyBear ``F_t`` data with challenge-field ``F_c = F_t[X]/(X^6 - 31)`` coefficients (31 is a generator of
``F_t^*``, hence a non-square and non-cube, so ``X^6 - 31`` is irreducible).  An ``F_c`` element is stored as 6
coordinate lanes; ``F_c x F_t`` products are 6 independent ``F_t`` products, so the tests below never multiply two
``F_c`` elements -- only the helper inversion does.  Montgomery ``R = 2^32``: challenge coefficients are passed in
Montgomery form (``x R mod p``), data in plain form, so ``mont_mul(data, coeff)`` is the plain product.

Buckets (``PROTOCOL.md`` section 5, steps 5 and 3):

* ``irs_row_combination`` (step 5, ``v = r^T [W_1; W_2]``): scalar ``F_c`` coefficient per row, over the *unencoded*
  ``k``-wide rows (encoding is linear: the prover encodes the 6 combined rows afterwards, 6 encoder calls).  A tall-skinny
  GEMV: reads the 5.09 GB matrix once.
* ``quadratic_test`` (step 5, ``p_0``): ``p_0[j] = sum_i r_i (X_i[j] Y_i[j] - Z_i[j])`` over the 8,192-point domain
  (cosets 0 and 1: ``deg p_0 < 2k - 1``); ``X_i`` is row ``x_i`` re-encoded on coset 1 (the codeword is not kept), coset 0
  is the row itself.  Per triple: 3 encoder calls (``ncosets=1``) + the product-accumulate.
* ``linear_test`` (step 5, ``q_add``): ``q[j] = sum_i R_i[j] U_i[j]`` with ``R_i`` the ``deg < k`` interpolant of row ``i``
  of ``r^T A`` (an ``F_c`` row: 6 coordinate rows, each encoded on coset 1) and ``U_i`` the witness row re-encoded on
  coset 1: 7 encoder calls per row + the product-accumulate.  ``r^T A`` itself (the sparse per-unit constraint
  combination) is NOT generated here: the bench feeds a resident pseudo-random ``r^T A`` slice (labelled).
* ``batch_inverse`` (step 3, helpers ``h_j = 1 / (z_T - w_j)``): Montgomery's trick over ``F_c`` with strided
  ownership (thread ``t`` owns elements ``t, t + T, ...``; prefix products in a global scratch), one ``F_c`` inversion per
  thread by ``a^-1 = a^(p^6 - 2)`` (square-and-multiply, 186 bits) -- 3 ``F_c`` products per element + 1/m inversions.
* ``open_columns`` (step 7): the ``t`` opened columns of the codeword, either gathered from a kept cosets store or
  re-encoded with the v2 kernel in gather mode (``RSEncoderSIMT.gather``: the row is encoded, only the selected
  positions are written); Merkle paths from the tree levels.
"""
from __future__ import annotations

import numpy as np

from . import babybear as bb

EXT_DEG = 6
EXT_NONRESIDUE = 31          # F_c = F_t[X] / (X^6 - 31)

_COMMON = r"""
#define P 2013265921u
#define PINV %(pinv)du
#define K %(k)d
#define D 6
#define W_NR 31u
typedef unsigned u32;
__device__ __forceinline__ u32 add_p(u32 a, u32 b) { u32 s = a + b; return s >= P ? s - P : s; }
__device__ __forceinline__ u32 sub_p(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }
/* a in [0, p), w = w' R (Montgomery): a w' mod p. */
__device__ __forceinline__ u32 mont_mul(u32 a, u32 w) {
    unsigned long long x = (unsigned long long)a * w;
    u32 lo = (u32)x, hi = (u32)(x >> 32);
    u32 t = lo * PINV;
    u32 u = __umulhi(t, P);
    u32 r = hi - u;
    return hi < u ? r + P : r;
}
"""

_COMBINE = _COMMON + r"""
/* Thread (j, chunk): column j of the 2k-wide (or k-wide) domain, rows [chunk * chunk_rows, +chunk_rows) of the slice;
   acc[chunk][c][j] += sum over its rows of the c-th coordinate of the term.  Column j < K reads the systematic source
   (the input rows, stride K), j >= K the coset-1 buffer (stride K, column j - K). */

/* MODE 0: term = r[i] * W[i][j] (k-wide, scalar F_c coefficient per row; r is (rows, 6) Montgomery) */
extern "C" __global__ void combine_scalar(const u32* __restrict__ W, const u32* __restrict__ r, u32* __restrict__ acc,
                                          int rows, int chunk_rows, int ncols)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = blockIdx.y;
    if (j >= ncols) return;
    int i0 = chunk * chunk_rows, i1 = min(rows, i0 + chunk_rows);
    u32 a[D];
    #pragma unroll
    for (int c = 0; c < D; ++c) a[c] = 0;
    for (int i = i0; i < i1; ++i) {
        u32 w = __ldg(&W[(size_t)i * ncols + j]);
        #pragma unroll
        for (int c = 0; c < D; ++c) a[c] = add_p(a[c], mont_mul(w, __ldg(&r[(size_t)i * D + c])));
    }
    #pragma unroll
    for (int c = 0; c < D; ++c) {
        u32* o = acc + ((size_t)chunk * D + c) * ncols + j;
        *o = add_p(*o, a[c]);
    }
}

/* MODE 1: term = r[i] * (X[i][j] Y[i][j] - Z[i][j]), j < 2K; X = j < K ? x_in : x_c1 (coset 1 of the re-encoded row) */
extern "C" __global__ void combine_quadratic(const u32* __restrict__ x_in, const u32* __restrict__ y_in, const u32* __restrict__ z_in,
                                             const u32* __restrict__ x_c1, const u32* __restrict__ y_c1, const u32* __restrict__ z_c1,
                                             const u32* __restrict__ r, u32* __restrict__ acc, int rows, int chunk_rows)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = blockIdx.y;
    if (j >= 2 * K) return;
    const bool sys = j < K;
    const int jj = sys ? j : j - K;
    const u32* X = sys ? x_in : x_c1;
    const u32* Y = sys ? y_in : y_c1;
    const u32* Z = sys ? z_in : z_c1;
    int i0 = chunk * chunk_rows, i1 = min(rows, i0 + chunk_rows);
    u32 a[D];
    #pragma unroll
    for (int c = 0; c < D; ++c) a[c] = 0;
    for (int i = i0; i < i1; ++i) {
        size_t o = (size_t)i * K + jj;
        u32 x = __ldg(&X[o]), y = __ldg(&Y[o]), z = __ldg(&Z[o]);
        /* x y in plain form: mont_mul(x, y) gives x y R^-1; multiply by r R^2 instead (r2 = r R^2 mod p) */
        u32 xy = mont_mul(x, y);                       /* = x y R^-1 */
        #pragma unroll
        for (int c = 0; c < D; ++c) {
            u32 rc = __ldg(&r[(size_t)i * (2 * D) + c]);          /* r_c R   (for z) */
            u32 rc2 = __ldg(&r[(size_t)i * (2 * D) + D + c]);     /* r_c R^2 (for x y R^-1) */
            a[c] = add_p(a[c], sub_p(mont_mul(xy, rc2), mont_mul(z, rc)));
        }
    }
    #pragma unroll
    for (int c = 0; c < D; ++c) {
        u32* o = acc + ((size_t)chunk * D + c) * (2 * K) + j;
        *o = add_p(*o, a[c]);
    }
}

/* MODE 2: term_c = R_c[i][j] * U[i][j], j < 2K; R_c = coordinate c of r^T A row i (rA_in: (D, rows, K) plain; rA_c1: its
   coset-1 encoding (D, rows, K) plain), U = j < K ? u_in : u_c1.  Both plain: u is first lifted to Montgomery form
   (mont_mul(u, R^2) = u R), then mont_mul(RA, u R) = RA u. */
extern "C" __global__ void combine_linear(const u32* __restrict__ u_in, const u32* __restrict__ u_c1,
                                          const u32* __restrict__ rA_in, const u32* __restrict__ rA_c1,
                                          u32* __restrict__ acc, int rows, int chunk_rows, u32 r2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk = blockIdx.y;
    if (j >= 2 * K) return;
    const bool sys = j < K;
    const int jj = sys ? j : j - K;
    const u32* U = sys ? u_in : u_c1;
    const u32* RA = sys ? rA_in : rA_c1;
    int i0 = chunk * chunk_rows, i1 = min(rows, i0 + chunk_rows);
    u32 a[D];
    #pragma unroll
    for (int c = 0; c < D; ++c) a[c] = 0;
    for (int i = i0; i < i1; ++i) {
        u32 u = mont_mul(__ldg(&U[(size_t)i * K + jj]), r2);          /* u R: Montgomery form of u */
        #pragma unroll
        for (int c = 0; c < D; ++c) a[c] = add_p(a[c], mont_mul(__ldg(&RA[((size_t)c * rows + i) * K + jj]), u));
    }
    #pragma unroll
    for (int c = 0; c < D; ++c) {
        u32* o = acc + ((size_t)chunk * D + c) * (2 * K) + j;
        *o = add_p(*o, a[c]);
    }
}

/* acc (nchunk, D, ncols) -> out (D, ncols): sum over chunks */
extern "C" __global__ void reduce_chunks(const u32* __restrict__ acc, u32* __restrict__ out, int nchunk, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= D * ncols) return;
    u32 s = 0;
    for (int ch = 0; ch < nchunk; ++ch) s = add_p(s, acc[(size_t)ch * D * ncols + idx]);
    out[idx] = s;
}
"""

_INVERSE = _COMMON + r"""
/* F_c = F_p[X]/(X^6 - 31), coordinate-major storage (D, N).  Elements in Montgomery form throughout the kernel. */
__device__ __forceinline__ void fc_mul(const u32* a, const u32* b, u32* out) {
    u32 t[2 * D - 1];
    #pragma unroll
    for (int i = 0; i < 2 * D - 1; ++i) t[i] = 0;
    #pragma unroll
    for (int i = 0; i < D; ++i)
        #pragma unroll
        for (int j = 0; j < D; ++j) t[i + j] = add_p(t[i + j], mont_mul(a[i], b[j]));
    #pragma unroll
    for (int i = 0; i < D; ++i) out[i] = t[i];
    /* X^6 = 31: fold t[6..10] * 31 into out[0..4] (31 a = 32 a - a, all mod p, plain small-constant multiply) */
    #pragma unroll
    for (int i = D; i < 2 * D - 1; ++i) {
        unsigned long long v = (unsigned long long)t[i] * W_NR;
        out[i - D] = add_p(out[i - D], (u32)(v %% P));
    }
}
__device__ __forceinline__ void fc_sqr(const u32* a, u32* out) {
    u32 tmp[D];
    fc_mul(a, a, tmp);
    #pragma unroll
    for (int i = 0; i < D; ++i) out[i] = tmp[i];
}

/* a^(p^6 - 2) by square-and-multiply over the exponent's 186 bits (exp words, little-endian, 6 x 32 bits) */
__device__ __forceinline__ void fc_inv(const u32* a, u32* out, const u32* __restrict__ exp_words, int nbits, u32 one_mont) {
    u32 acc[D], base[D];
    #pragma unroll
    for (int i = 0; i < D; ++i) { acc[i] = 0; base[i] = a[i]; }
    acc[0] = one_mont;
    for (int b = 0; b < nbits; ++b) {
        if ((__ldg(&exp_words[b >> 5]) >> (b & 31)) & 1u) {
            u32 t[D];
            fc_mul(acc, base, t);
            #pragma unroll
            for (int i = 0; i < D; ++i) acc[i] = t[i];
        }
        fc_sqr(base, base);
    }
    #pragma unroll
    for (int i = 0; i < D; ++i) out[i] = acc[i];
}

/* Thread t owns d[s * T + t], s < m (T = total threads).  prefix[s] = d[0] ... d[s] (global scratch, same layout);
   inv = prefix[m-1]^-1; backward: out[s] = prefix[s-1] * inv; inv *= d[s].  N must be a multiple of T (the bench pads). */
extern "C" __global__ void batch_inverse(const u32* __restrict__ d, u32* __restrict__ prefix, u32* __restrict__ out,
                                         long long N, int m, const u32* __restrict__ exp_words, int nbits, u32 one_mont)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long T = (long long)gridDim.x * blockDim.x;
    if (t >= T) return;
    u32 run[D], e[D], tmp[D];
    #pragma unroll
    for (int i = 0; i < D; ++i) run[i] = 0;
    run[0] = one_mont;
    for (int s = 0; s < m; ++s) {
        long long idx = (long long)s * T + t;
        if (idx >= N) break;
        #pragma unroll
        for (int i = 0; i < D; ++i) e[i] = __ldg(&d[(size_t)i * N + idx]);
        fc_mul(run, e, tmp);
        #pragma unroll
        for (int i = 0; i < D; ++i) { run[i] = tmp[i]; prefix[(size_t)i * N + idx] = tmp[i]; }
    }
    u32 inv[D];
    fc_inv(run, inv, exp_words, nbits, one_mont);
    for (int s = m - 1; s >= 0; --s) {
        long long idx = (long long)s * T + t;
        if (idx >= N) continue;
        #pragma unroll
        for (int i = 0; i < D; ++i) e[i] = __ldg(&d[(size_t)i * N + idx]);
        if (s == 0) {
            #pragma unroll
            for (int i = 0; i < D; ++i) out[(size_t)i * N + idx] = inv[i];
        } else {
            u32 pf[D];
            #pragma unroll
            for (int i = 0; i < D; ++i) pf[i] = prefix[(size_t)i * N + idx - T];
            fc_mul(pf, inv, tmp);
            #pragma unroll
            for (int i = 0; i < D; ++i) out[(size_t)i * N + idx] = tmp[i];
            fc_mul(inv, e, tmp);
            #pragma unroll
            for (int i = 0; i < D; ++i) inv[i] = tmp[i];
        }
    }
}
"""

_GATHER = _COMMON + r"""
/* gather columns sel[q] (q < t) of a (rows, ncols) row-major matrix into out[q][row] (t, rows): thread per (row, q) */
extern "C" __global__ void gather_columns(const u32* __restrict__ M, long long ncols, const int* __restrict__ sel, int t,
                                          long long rows, u32* __restrict__ out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * t) return;
    int q = (int)(idx %% t);
    long long row = idx / t;
    out[(size_t)q * rows + row] = __ldg(&M[row * ncols + sel[q]]);
}
"""


def _cp():
    import cupy as cp

    return cp


def _params(k: int = bb.K_CODE) -> dict:
    return {"pinv": bb.P_INV_MOD_R, "k": k}


def mont(a: np.ndarray) -> np.ndarray:
    return ((np.asarray(a, dtype=np.uint64) % np.uint64(bb.P)) * np.uint64(bb.R_MOD_P) % np.uint64(bb.P)).astype(np.uint32)


def mont2(a: np.ndarray) -> np.ndarray:
    return ((np.asarray(a, dtype=np.uint64) % np.uint64(bb.P)) * np.uint64(bb.R2_MOD_P) % np.uint64(bb.P)).astype(np.uint32)


class Combiner:
    """The three product-accumulate kernels plus the encoder calls that feed them; ``chunk_rows`` rows per thread."""

    def __init__(self, k: int = bb.K_CODE, chunk_rows: int = 32, block: int = 128):
        cp = _cp()
        self.cp, self.k, self.chunk_rows, self.block = cp, k, chunk_rows, block
        self.mod = cp.RawModule(code=_COMBINE % _params(k), options=("-std=c++17",))
        self.k_scalar = self.mod.get_function("combine_scalar")
        self.k_quad = self.mod.get_function("combine_quadratic")
        self.k_lin = self.mod.get_function("combine_linear")
        self.k_reduce = self.mod.get_function("reduce_chunks")
        self.r2 = np.uint32(bb.R2_MOD_P)               # R^2 mod p: mont_mul(u, r2) = u R, the Montgomery form of u

    def nchunks(self, rows: int) -> int:
        return -(-rows // self.chunk_rows)

    def acc(self, rows: int, ncols: int):
        return self.cp.zeros((self.nchunks(rows), EXT_DEG, ncols), dtype=self.cp.uint32)

    def scalar(self, W, r_mont, acc):
        """``acc[chunk][c][j] += sum_i r[i][c] W[i][j]`` over the rows of ``W`` ``(rows, ncols)``; ``r_mont`` ``(rows, 6)``."""
        rows, ncols = W.shape
        grid = ((ncols + self.block - 1) // self.block, self.nchunks(rows))
        self.k_scalar(grid, (self.block,), (W, r_mont, acc, np.int32(rows), np.int32(self.chunk_rows), np.int32(ncols)))

    def quadratic(self, x_in, y_in, z_in, x_c1, y_c1, z_c1, r_mont12, acc):
        """``r_mont12`` ``(rows, 12)``: ``[r R, r R^2]`` per row."""
        rows = x_in.shape[0]
        grid = ((2 * self.k + self.block - 1) // self.block, self.nchunks(rows))
        self.k_quad(grid, (self.block,), (x_in, y_in, z_in, x_c1, y_c1, z_c1, r_mont12, acc, np.int32(rows), np.int32(self.chunk_rows)))

    def linear(self, u_in, u_c1, rA_in, rA_c1, acc):
        rows = u_in.shape[0]
        grid = ((2 * self.k + self.block - 1) // self.block, self.nchunks(rows))
        self.k_lin(grid, (self.block,), (u_in, u_c1, rA_in, rA_c1, acc, np.int32(rows), np.int32(self.chunk_rows), self.r2))

    def reduce(self, acc, out=None):
        nchunk, d, ncols = acc.shape
        if out is None:
            out = self.cp.empty((d, ncols), dtype=self.cp.uint32)
        total = d * ncols
        self.k_reduce(((total + 255) // 256,), (256,), (acc, out, np.int32(nchunk), np.int32(ncols)))
        return out


class BatchInverter:
    def __init__(self, threads_total: int = 128 * 1024, block: int = 256):
        cp = _cp()
        self.cp, self.T, self.block = cp, threads_total, block
        self.mod = cp.RawModule(code=_INVERSE % _params(), options=("-std=c++17",))
        self.kernel = self.mod.get_function("batch_inverse")
        e = bb.P**EXT_DEG - 2
        self.nbits = e.bit_length()
        self.exp_words = cp.asarray(np.array([(e >> (32 * i)) & 0xFFFFFFFF for i in range(-(-self.nbits // 32))], dtype=np.uint32))
        self.one_mont = np.uint32(bb.R_MOD_P)
        self.scratch = None

    def __call__(self, d_mont, out=None):
        """``d_mont`` ``(6, N)`` uint32 in Montgomery form (nonzero elements); returns ``(6, N)`` inverses in Montgomery form."""
        cp = self.cp
        N = d_mont.shape[1]
        m = -(-N // self.T)
        if self.scratch is None or self.scratch.shape != d_mont.shape:
            self.scratch = cp.empty_like(d_mont)
        if out is None:
            out = cp.empty_like(d_mont)
        self.kernel((self.T // self.block,), (self.block,), (d_mont, self.scratch, out, np.int64(N), np.int32(m), self.exp_words,
                                                             np.int32(self.nbits), self.one_mont))
        return out


class ColumnGather:
    def __init__(self):
        cp = _cp()
        self.cp = cp
        self.mod = cp.RawModule(code=_GATHER % _params(), options=("-std=c++17",))
        self.kernel = self.mod.get_function("gather_columns")

    def __call__(self, M, sel, out=None):
        """``M`` ``(rows, ncols)`` row-major; ``sel`` int32 device array of column indices; returns ``(t, rows)``."""
        cp = self.cp
        rows, ncols = M.shape
        t = sel.shape[0]
        if out is None:
            out = cp.empty((t, rows), dtype=cp.uint32)
        total = rows * t
        self.kernel(((total + 255) // 256,), (256,), (M, np.int64(ncols), sel, np.int32(t), np.int64(rows), out))
        return out


def merkle_paths(levels, columns) -> list:
    """Authentication paths from ``hash_gpu``-style levels (device arrays), one device->host copy per level."""
    host = [lvl.get() if hasattr(lvl, "get") else np.asarray(lvl) for lvl in levels[:-1]]
    paths = []
    for c in columns:
        path, idx = [], int(c)
        for lvl in host:
            sib = idx ^ 1
            path.append(lvl[sib] if sib < lvl.shape[0] else None)
            idx >>= 1
        paths.append(path)
    return paths

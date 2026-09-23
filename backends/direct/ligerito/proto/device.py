"""Device kernels for the fast Ligerito prover (cupy RawModule, launched on the current torch stream): F_{p^6} primitives,
the column sumcheck, and every contraction of the recursion -- exact mod-p arithmetic in u32/u64 integer registers, no fp64.

* ``EXT_CUDA``: ``__device__`` F_p / F_{p^6} helpers (``mulm``, ``fold``, ``mont_mul``, ``ext_mul``, ``ext_add``, ``ext_sub``) for
  x^6 - 31 over BabyBear, coordinates in [0, p).  Self-contained; other lanes may paste it into their own kernels.
* ``contract_geom(A, etas)``: ``out[p, t] = sum_r A[p, r] eta_t^r`` -- a tiled SIMT GEMM whose right factor (the RS generator rows of the
  opened positions) is generated on the fly (one Montgomery multiply per tile element), u64 accumulators folded every 4 products.
* ``contract_planes(A, Bp)``: ``out[p, w] = sum_r A[p, r] Bp[w, r]`` for w < 6 m (m <= 4 eq tensors as 6 planes each);
  ``contract_eq(A, zs)`` wraps it for m evaluation points.
* ``fold_ext(M, rbar)``: ``y[r] = sum_c M~[c, r] rbar[c]`` for an F_{p^6} matrix kept as 6 planes.
* ``eq_table(lo, hi)``: the eq tensor ``lo[x_lo] hi[x_hi]`` as 6 planes.
* ``build_tables`` + ``sc_step``: the column sumcheck on (T, C, 6) tables resident on the device; one single-block launch per sumcheck
  variable (fold with the previous challenge + the next round message), 18 words back to the host.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from .ext import BETA, D, P

try:
    import cupy as cp
except Exception:  # noqa: BLE001
    cp = None

PINV = pow(P, -1, 2**32)
R_MOD_P = pow(2, 32, P)

EXT_CUDA = r"""
typedef unsigned int u32;
typedef unsigned long long u64;
#define PP 2013265921u
#define PL 2013265921ull
#define BETA_ 31ull
#define PINV_ %(pinv)du
#define R32 268435454ull            /* 2^32 mod p */

/* a < 2^64 -> a' == a (mod p), a' < 2^60 + 2^32 */
__device__ __forceinline__ u64 fold(u64 a) { return (a & 0xFFFFFFFFull) + (a >> 32) * R32; }
__device__ __forceinline__ u32 addm(u32 a, u32 b) { u32 s = a + b; return min(s, s - PP); }
__device__ __forceinline__ u32 subm(u32 a, u32 b) { u32 d = a - b; return min(d, d + PP); }
__device__ __forceinline__ u32 mulm(u32 a, u32 b) { return (u32)(((u64)a * b) %% PL); }
/* a (standard) * w (Montgomery, w' 2^32) -> a w' (standard), in [0, p) */
__device__ __forceinline__ u32 mont_mul(u32 a, u32 w) {
    u64 x = (u64)a * w;
    u32 lo = (u32)x, hi = (u32)(x >> 32);
    u32 t = lo * PINV_;
    u32 u = __umulhi(t, PP);
    u32 r = hi - u;
    return min(r, r + PP);
}
/* c = a * b in F_p[x]/(x^6 - 31); c may alias neither a nor b */
__device__ __forceinline__ void ext_mul(const u32* a, const u32* b, u32* c) {
    u64 lo[6] = {0, 0, 0, 0, 0, 0}, hi[5] = {0, 0, 0, 0, 0};
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        #pragma unroll
        for (int j = 0; j < 6; ++j) {
            const u64 pr = fold((u64)a[i] * b[j]);
            if (i + j < 6) lo[i + j] += pr; else hi[i + j - 6] += pr;
        }
    }
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        u64 v = lo[k] %% PL;
        if (k < 5) v += BETA_ * (hi[k] %% PL);
        c[k] = (u32)(v %% PL);
    }
}
__device__ __forceinline__ void ext_add(const u32* a, const u32* b, u32* c) {
    #pragma unroll
    for (int k = 0; k < 6; ++k) c[k] = addm(a[k], b[k]);
}
__device__ __forceinline__ void ext_sub(const u32* a, const u32* b, u32* c) {
    #pragma unroll
    for (int k = 0; k < 6; ++k) c[k] = subm(a[k], b[k]);
}
"""

_SRC = EXT_CUDA + r"""
/* ------------------------------------------------------------------ geometric GEMM: out[p, t] += sum_r A[p, r] eta_t^r
   grid (R chunks, T tiles of 64, P tiles of 16 MP); block 256 = 16 (terms, 4 each) x 16 (polys, MP each); BK = 32 rows per step */
#define BT 64
#define BK 32
#define BP (16 * MP)
__device__ __forceinline__ u32 pow_std(u32 base_mont, u32 e) {       /* base^e, base given in Montgomery form, result standard */
    u32 res = 1u, b = base_mont;
    while (e) { if (e & 1u) res = mont_mul(res, b); b = mont_mul(b, b); e >>= 1; }
    return res;
}
extern "C" __global__ void __launch_bounds__(256)
contract_geom(const u32* __restrict__ A, long long lda, int Ptot, int R, int rchunk,
              const u32* __restrict__ eta_mont, const u32* __restrict__ eta_bk_mont, int Ttot, u64* __restrict__ out)
{
    __shared__ u32 As[BP][BK + 1];
    __shared__ __align__(16) u32 Bs[BK][BT];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int r_begin = blockIdx.x * rchunk, r_end = min(R, r_begin + rchunk);
    const int t0 = blockIdx.y * BT, p0 = blockIdx.z * BP;
    const int tb = tid & 63, kb0 = tid >> 6;                 /* generator: term tb, rows kb0 + 4 j */
    u32 cur[8], step = 0u;
    const bool tval = (t0 + tb) < Ttot;
    if (tval) {
        const u32 em = eta_mont[t0 + tb];
        step = eta_bk_mont[t0 + tb];
        #pragma unroll
        for (int j = 0; j < 8; ++j) cur[j] = pow_std(em, (u32)(r_begin + kb0 + 4 * j));
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) cur[j] = 0u;
    }
    u64 acc[MP][4];
    #pragma unroll
    for (int i = 0; i < MP; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0ull;
    for (int r0 = r_begin; r0 < r_end; r0 += BK) {
        #pragma unroll
        for (int i = 0; i < BP / 8; ++i) {
            const int p = (tid >> 5) + 8 * i, k = tid & 31;
            u32 v = 0u;
            if (p0 + p < Ptot && r0 + k < r_end) v = A[(long long)(p0 + p) * lda + r0 + k];
            As[p][k] = v;
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) { Bs[kb0 + 4 * j][tb] = cur[j]; cur[j] = mont_mul(cur[j], step); }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            const uint4 b = *reinterpret_cast<const uint4*>(&Bs[k][tx * 4]);
            #pragma unroll
            for (int i = 0; i < MP; ++i) {
                const u64 a = As[ty * MP + i][k];
                acc[i][0] += a * b.x; acc[i][1] += a * b.y; acc[i][2] += a * b.z; acc[i][3] += a * b.w;
            }
            if ((k & 3) == 3) {
                #pragma unroll
                for (int i = 0; i < MP; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] = fold(acc[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < MP; ++i) {
        const int p = p0 + ty * MP + i;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = t0 + tx * 4 + j;
            if (p < Ptot && t < Ttot) atomicAdd(&out[(long long)p * Ttot + t], acc[i][j] %% PL);
        }
    }
}

/* ------------------------------------------------------------------ planes: out[p, w] += sum_r A[p, r] Bp[w, r], w < W
   (W = 6 m: m eq tensors as 6 planes each, one pass over A).  grid (P tiles of 8 PT, R chunks); block 256 = 8 warps; warp g owns
   rows p0 + g PT + i (i < PT), its lanes stride r by 32 (coalesced A and Bp rows; the 8 warps share each Bp line through L1);
   PT x W u64 accumulators folded every 4 products, warp-shuffle reduced, one atomic per (row, w) per block */
template <int W, int PT>
__device__ __forceinline__ void planes_body(const u32* __restrict__ A, long long lda, int Ptot, int R, int rchunk,
                                            const u32* __restrict__ Bp, long long ldb, u64* __restrict__ out)
{
    const int lane = threadIdx.x & 31, g = threadIdx.x >> 5;
    const int r_begin = blockIdx.y * rchunk, r_end = min(R, r_begin + rchunk);
    const int p0 = blockIdx.x * (8 * PT) + g * PT;
    u64 acc[PT][W];
    #pragma unroll
    for (int i = 0; i < PT; ++i)
        #pragma unroll
        for (int w = 0; w < W; ++w) acc[i][w] = 0ull;
    int cnt = 0;
    for (int r = r_begin + lane; r < r_end; r += 32) {
        u32 a[PT];
        #pragma unroll
        for (int i = 0; i < PT; ++i) a[i] = (p0 + i < Ptot) ? A[(long long)(p0 + i) * lda + r] : 0u;
        #pragma unroll
        for (int w = 0; w < W; ++w) {
            const u64 b = Bp[(long long)w * ldb + r];
            #pragma unroll
            for (int i = 0; i < PT; ++i) acc[i][w] += (u64)a[i] * b;
        }
        if (++cnt == 4) {
            cnt = 0;
            #pragma unroll
            for (int i = 0; i < PT; ++i)
                #pragma unroll
                for (int w = 0; w < W; ++w) acc[i][w] = fold(acc[i][w]);
        }
    }
    #pragma unroll
    for (int i = 0; i < PT; ++i)
        #pragma unroll
        for (int w = 0; w < W; ++w) {
            u64 v = acc[i][w] %% PL;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (lane == 0 && p0 + i < Ptot) atomicAdd(&out[(long long)(p0 + i) * W + w], v %% PL);
        }
}
#define PLANES(W, PT) \
extern "C" __global__ void __launch_bounds__(256) \
contract_planes_w##W(const u32* __restrict__ A, long long lda, int Ptot, int R, int rchunk, \
                     const u32* __restrict__ Bp, long long ldb, u64* __restrict__ out) \
{ planes_body<W, PT>(A, lda, Ptot, R, rchunk, Bp, ldb, out); }
PLANES(6, 8)
PLANES(12, 4)
PLANES(18, 2)
PLANES(24, 2)

/* ------------------------------------------------------------------ ext fold: y[k][r] = sum_c sum_{a+b} M[a C + c][r] rbar[c][b] */
extern "C" __global__ void __launch_bounds__(256)
fold_ext(const u32* __restrict__ M, int C, int R, const u32* __restrict__ rbar, u32* __restrict__ y)
{
    extern __shared__ u32 rb[];
    for (int i = threadIdx.x; i < 6 * C; i += blockDim.x) rb[i] = rbar[i];
    __syncthreads();
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    u64 acc[11];
    #pragma unroll
    for (int k = 0; k < 11; ++k) acc[k] = 0ull;
    for (int c = 0; c < C; ++c) {
        u32 m[6];
        #pragma unroll
        for (int a = 0; a < 6; ++a) m[a] = M[((long long)a * C + c) * R + r];
        #pragma unroll
        for (int a = 0; a < 6; ++a)
            #pragma unroll
            for (int b = 0; b < 6; ++b) acc[a + b] += fold((u64)m[a] * rb[c * 6 + b]);
        #pragma unroll
        for (int k = 0; k < 11; ++k) acc[k] = fold(acc[k]);
    }
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        u64 v = acc[k] %% PL;
        if (k < 5) v += BETA_ * (acc[k + 6] %% PL);
        y[(long long)k * R + r] = (u32)(v %% PL);
    }
}

/* ------------------------------------------------------------------ eq tensor: out[d][x] = (lo[x & (L - 1)] hi[x >> logL])_d */
extern "C" __global__ void __launch_bounds__(256)
eq_table(const u32* __restrict__ lo, int logL, const u32* __restrict__ hi, long long n, u32* __restrict__ out)
{
    const long long x = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    const long long xl = x & ((1ll << logL) - 1), xh = x >> logL;
    u32 a[6], b[6], c[6];
    #pragma unroll
    for (int d = 0; d < 6; ++d) { a[d] = lo[xl * 6 + d]; b[d] = hi[xh * 6 + d]; }
    ext_mul(b, a, c);
    #pragma unroll
    for (int d = 0; d < 6; ++d) out[(long long)d * n + x] = c[d];
}

/* ------------------------------------------------------------------ eq tensor directly (small k): out[x] = prod_j (x_j ? z_j : 1 - z_j),
   bit j of x <-> z_j; rowmajor: out[x * 6 + d], else planes out[d * 2^k + x] */
extern "C" __global__ void __launch_bounds__(256)
eq_direct(const u32* __restrict__ z, int k, int rowmajor, u32* __restrict__ out)
{
    const int n = 1 << k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    u32 acc[6] = {1u, 0u, 0u, 0u, 0u, 0u}, f[6], t[6];
    for (int j = 0; j < k; ++j) {
        if ((x >> j) & 1) {
            #pragma unroll
            for (int d = 0; d < 6; ++d) f[d] = z[j * 6 + d];
        } else {
            f[0] = subm(1u, z[j * 6]);
            #pragma unroll
            for (int d = 1; d < 6; ++d) f[d] = subm(0u, z[j * 6 + d]);
        }
        ext_mul(acc, f, t);
        #pragma unroll
        for (int d = 0; d < 6; ++d) acc[d] = t[d];
    }
    #pragma unroll
    for (int d = 0; d < 6; ++d) out[rowmajor ? (long long)x * 6 + d : (long long)d * n + x] = acc[d];
}

/* ------------------------------------------------------------------ sumcheck tables (T, Cs, 6): terms 0..Te-1 = eq, Te.. = geom
   A[t][c] = coef[t] u_t[c];  u_eq(t)[c]_k = sum_{a+b=k} ue[t][aC+c][b] + 31 sum_{a+b=k+6} ...;  u_geom(t)[c]_a = ug[aC+c][t-Te];
   B[t < Te][c] = cole[t][c], B[t][c] = (colg[t-Te][c], 0, ..) */
extern "C" __global__ void __launch_bounds__(256)
build_tables(const u64* __restrict__ ug, int Tg, const u64* __restrict__ ue, int Te, const u32* __restrict__ coef,
             const u32* __restrict__ colg, const u32* __restrict__ cole, int T, int C, u32* __restrict__ A, u32* __restrict__ B)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * C) return;
    const int t = i / C, c = i - t * C;
    u32 u[6], cf[6], out[6];
    if (t < Te) {
        const u64* uet = ue + (long long)t * 36 * C;
        u64 lo[6] = {0, 0, 0, 0, 0, 0}, hi[5] = {0, 0, 0, 0, 0};
        #pragma unroll
        for (int a = 0; a < 6; ++a)
            #pragma unroll
            for (int b = 0; b < 6; ++b) {
                const u64 v = uet[((long long)a * C + c) * 6 + b] %% PL;
                if (a + b < 6) lo[a + b] += v; else hi[a + b - 6] += v;
            }
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            u64 v = lo[k] %% PL;
            if (k < 5) v += BETA_ * (hi[k] %% PL);
            u[k] = (u32)(v %% PL);
        }
        #pragma unroll
        for (int d = 0; d < 6; ++d) B[(long long)i * 6 + d] = cole[((long long)t * C + c) * 6 + d];
    } else {
        #pragma unroll
        for (int a = 0; a < 6; ++a) u[a] = (u32)(ug[((long long)a * C + c) * Tg + (t - Te)] %% PL);
        B[(long long)i * 6] = colg[(long long)(t - Te) * C + c];
        #pragma unroll
        for (int d = 1; d < 6; ++d) B[(long long)i * 6 + d] = 0u;
    }
    #pragma unroll
    for (int d = 0; d < 6; ++d) cf[d] = coef[t * 6 + d];
    ext_mul(cf, u, out);
    #pragma unroll
    for (int d = 0; d < 6; ++d) A[(long long)i * 6 + d] = out[d];
}

/* ------------------------------------------------------------------ one column-sumcheck variable, single block.
   Tables (T, Cs, 6) with a fixed row stride Cs; the live width is C.  do_fold: first fold (C -> C/2) with r:
   X <- X_lo + r (X_hi - X_lo) in place; then g(0), g(1), g(2) of the top variable of the (new) width -> g[18]. */
extern "C" __global__ void __launch_bounds__(1024)
sc_step(u32* __restrict__ A, u32* __restrict__ B, int T, int Cs, int C, int do_fold,
        u32 r0, u32 r1, u32 r2, u32 r3, u32 r4, u32 r5, u32* __restrict__ g)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    if (do_fold) {
        const u32 r[6] = {r0, r1, r2, r3, r4, r5};
        const int h = C >> 1;
        for (int i = tid; i < 2 * T * h; i += nt) {
            u32* X = (i < T * h) ? A : B;
            const int j = (i < T * h) ? i : i - T * h;
            const int t = j / h, x = j - t * h;
            u32* lo = X + ((long long)t * Cs + x) * 6;
            const u32* hi = X + ((long long)t * Cs + x + h) * 6;
            u32 l[6], d[6], pr[6];
            #pragma unroll
            for (int k = 0; k < 6; ++k) l[k] = lo[k];
            #pragma unroll
            for (int k = 0; k < 6; ++k) d[k] = subm(hi[k], l[k]);
            ext_mul(r, d, pr);
            #pragma unroll
            for (int k = 0; k < 6; ++k) lo[k] = addm(l[k], pr[k]);
        }
        C = h;
        __syncthreads();
    }
    const int h = C >> 1;
    u64 s[18];
    #pragma unroll
    for (int k = 0; k < 18; ++k) s[k] = 0ull;
    for (int j = tid; j < T * h; j += nt) {
        const int t = j / h, x = j - t * h;
        const u32* al = A + ((long long)t * Cs + x) * 6;
        const u32* ah = A + ((long long)t * Cs + x + h) * 6;
        const u32* bl = B + ((long long)t * Cs + x) * 6;
        const u32* bh = B + ((long long)t * Cs + x + h) * 6;
        u32 a0[6], a1[6], a2[6], b0[6], b1[6], b2[6], pr[6];
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            a0[k] = al[k]; a1[k] = ah[k]; b0[k] = bl[k]; b1[k] = bh[k];
            a2[k] = subm(addm(a1[k], a1[k]), a0[k]); b2[k] = subm(addm(b1[k], b1[k]), b0[k]);
        }
        ext_mul(a0, b0, pr);
        #pragma unroll
        for (int k = 0; k < 6; ++k) s[k] += pr[k];
        ext_mul(a1, b1, pr);
        #pragma unroll
        for (int k = 0; k < 6; ++k) s[6 + k] += pr[k];
        ext_mul(a2, b2, pr);
        #pragma unroll
        for (int k = 0; k < 6; ++k) s[12 + k] += pr[k];
    }
    __shared__ u64 red[32][18];
    const int lane = tid & 31, warp = tid >> 5;
    #pragma unroll
    for (int k = 0; k < 18; ++k) {
        u64 v = s[k] %% PL;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (tid < 18) {
        u64 v = 0;
        for (int w = 0; w < (nt >> 5); ++w) v += red[w][tid];
        g[tid] = (u32)(v %% PL);
    }
}
"""

_MODS: dict = {}


def ext_source() -> str:
    """``EXT_CUDA`` ready to paste into another RawModule (the %-substitution done)."""
    return EXT_CUDA % {"pinv": PINV}


def available(t: torch.Tensor | None = None) -> bool:
    return cp is not None and torch.cuda.is_available() and (t is None or t.is_cuda)


_SHARED_ALLOC = False


def share_torch_allocator() -> None:
    """Route cupy's allocations (Merkle levels, gathers) through torch's caching allocator: one pool, so memory freed by
    either side is reusable by the other (two pools OOM at 2^29-2^30 on 24 GB with most of the card idle in the other cache).
    Idempotent; process-wide."""
    global _SHARED_ALLOC
    if _SHARED_ALLOC or cp is None:
        return

    def _alloc(size: int, device_id: int) -> int:
        return torch.cuda.caching_allocator_alloc(size, device_id, torch.cuda.current_stream(device_id).cuda_stream)

    def _free(ptr: int, device_id: int) -> None:
        torch.cuda.caching_allocator_delete(ptr)

    cp.cuda.set_allocator(cp.cuda.memory.PythonFunctionAllocator(_alloc, _free).malloc)
    _SHARED_ALLOC = True


def _module(mp: int = 6):
    m = _MODS.get(mp)
    if m is None:
        src = ("#define MP %d\n" % mp) + (_SRC % {"pinv": PINV})
        m = cp.RawModule(code=src, options=("-std=c++17",))
        _MODS[mp] = m
    return m


def _launch(name: str, grid, block, args, smem: int = 0, mp: int = 6):
    ts = torch.cuda.current_stream()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cs = cp.cuda.ExternalStream(ts.cuda_stream)
    k = _module(mp).get_function(name)
    with cs:
        k(grid, block, args, shared_mem=smem)


def _ptr(t: torch.Tensor):
    return np.uint64(t.data_ptr())


def mont(x: np.ndarray) -> np.ndarray:
    """Standard -> Montgomery (x 2^32 mod p), uint32."""
    return (np.asarray(x, dtype=np.int64) % P * R_MOD_P % P).astype(np.uint32)


def _pow_vec(e: np.ndarray, k: int) -> np.ndarray:
    out = np.ones_like(e)
    b = e % P
    while k:
        if k & 1:
            out = out * b % P
        b = b * b % P
        k >>= 1
    return out


# ------------------------------------------------------------------------------------------------ contractions

def contract_geom(A: torch.Tensor, etas: np.ndarray, out: torch.Tensor | None = None, rchunk: int | None = None) -> torch.Tensor:
    """A (P, R) int32 device (entries in [0, p), rows contiguous, any row stride); etas (T,) base elements ->
    (P, T) int64 device: sum_r A[p, r] eta_t^r mod p (unreduced partial sums < 2^63 if ``out`` is accumulated into; reduce with % p)."""
    Pn, R = A.shape
    assert A.stride(1) == 1 and A.dtype == torch.int32
    T = len(etas)
    e = np.asarray(etas, dtype=np.int64) % P
    em = torch.as_tensor(mont(e).view(np.int32), device=A.device)
    ebk = torch.as_tensor(mont(_pow_vec(e, 32)).view(np.int32), device=A.device)
    if out is None:
        out = torch.zeros((Pn, T), dtype=torch.int64, device=A.device)
    mp = 6 if Pn % 96 == 0 else (4 if Pn % 64 == 0 else 2)
    bp = 16 * mp
    if rchunk is None:
        # enough CTAs to fill the part (~4 per SM) without shrinking a chunk below 256 rows
        tiles = ((T + 63) // 64) * ((Pn + bp - 1) // bp)
        want = max(1, 512 // max(1, tiles))
        rchunk = max(256, min(R, ((R + want - 1) // want + 31) // 32 * 32))
    grid = ((R + rchunk - 1) // rchunk, (T + 63) // 64, (Pn + bp - 1) // bp)
    _launch("contract_geom", grid, (256,), (_ptr(A), np.int64(A.stride(0)), np.int32(Pn), np.int32(R), np.int32(rchunk),
                                            _ptr(em), _ptr(ebk), np.int32(T), _ptr(out)), mp=mp)
    return out


_PLANES_PT = {6: 8, 12: 4, 18: 2, 24: 2}


def contract_planes(A: torch.Tensor, Bp: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """A (P, R) int32 device, Bp (W, R) int32 device (standard form), W in {6, 12, 18, 24} -> (P, W) int64:
    sum_r A[p, r] Bp[w, r] (unreduced; % p)."""
    Pn, R = A.shape
    W = Bp.shape[0]
    assert A.stride(1) == 1 and Bp.stride(1) == 1 and Bp.shape[1] == R and W in _PLANES_PT
    pt = _PLANES_PT[W]
    if out is None:
        out = torch.zeros((Pn, W), dtype=torch.int64, device=A.device)
    ptiles = (Pn + 8 * pt - 1) // (8 * pt)
    want = max(1, 1024 // ptiles)
    rchunk = max(512, ((R + want - 1) // want + 127) // 128 * 128)
    grid = (ptiles, (R + rchunk - 1) // rchunk)
    _launch("contract_planes_w%d" % W, grid, (256,), (_ptr(A), np.int64(A.stride(0)), np.int32(Pn), np.int32(R),
                                                      np.int32(rchunk), _ptr(Bp), np.int64(Bp.stride(0)), _ptr(out)))
    return out


def contract_eq(A: torch.Tensor, zs: np.ndarray) -> torch.Tensor:
    """A (P, 2^k) int32 device, zs (m, k, 6) -> (m, P, 6) int64 (unreduced): sum_r A[p, r] eq(r, z_j) -- up to 4 eq tensors
    per pass over A."""
    Pn, R = A.shape
    zs = np.asarray(zs, dtype=np.int64)
    m = zs.shape[0]
    outs = []
    for a in range(0, m, 4):
        mc = min(4, m - a)
        B = torch.empty((D * mc, R), dtype=torch.int32, device=A.device)
        for j in range(mc):
            eq_table(zs[a + j], A.device, out=B[D * j:D * (j + 1)])
        outs.append(contract_planes(A, B).view(Pn, mc, D).permute(1, 0, 2))
        del B
    return torch.cat(outs, dim=0)


def fold_ext(M: torch.Tensor, rbar: np.ndarray) -> torch.Tensor:
    """M (6, C, R) int32 planes, rbar (C, 6) -> y (6, R) int32: y[r] = sum_c M~[c, r] rbar[c] in F_{p^6}."""
    _, C, R = M.shape
    M = M.contiguous()
    rb = torch.as_tensor(np.ascontiguousarray(np.asarray(rbar, dtype=np.int64) % P).astype(np.int32), device=M.device)
    y = torch.empty((D, R), dtype=torch.int32, device=M.device)
    _launch("fold_ext", ((R + 255) // 256,), (256,), (_ptr(M), np.int32(C), np.int32(R), _ptr(rb), _ptr(y)), smem=4 * 6 * C)
    return y


def _eq_direct(z: np.ndarray, device, rowmajor: bool, out: torch.Tensor | None = None) -> torch.Tensor:
    k = z.shape[0]
    n = 1 << k
    zt = torch.as_tensor(np.ascontiguousarray(z).astype(np.int32).reshape(-1), device=device)
    if out is None:
        out = torch.empty((n, D) if rowmajor else (D, n), dtype=torch.int32, device=device)
    assert out.is_contiguous() and out.numel() == D * n
    _launch("eq_direct", ((n + 255) // 256,), (256,), (_ptr(zt), np.int32(k), np.int32(int(rowmajor)), _ptr(out)))
    return out


def eq_table(z: np.ndarray, device, out: torch.Tensor | None = None) -> torch.Tensor:
    """z (k, 6) -> (6, 2^k) int32 planes of eq(x, z), bit j of x <-> z_j (same values as ``linalg.eq_planes``); ``out``: a
    contiguous (6, 2^k) int32 destination."""
    z = np.asarray(z, dtype=np.int64) % P
    k = z.shape[0]
    if k <= 12:
        return _eq_direct(z, device, False, out)
    h = (k + 1) // 2
    lo = _eq_direct(z[:h], device, True) if h <= 12 else eq_table(z[:h], device).t().contiguous()        # (2^h, 6)
    hi = _eq_direct(z[h:], device, True) if k - h <= 12 else eq_table(z[h:], device).t().contiguous()    # (2^{k-h}, 6)
    n = 1 << k
    if out is None:
        out = torch.empty((D, n), dtype=torch.int32, device=device)
    assert out.shape == (D, n) and out.is_contiguous()
    _launch("eq_table", ((n + 255) // 256,), (256,), (_ptr(lo), np.int32(h), _ptr(hi), np.int64(n), _ptr(out)))
    return out


# ------------------------------------------------------------------------------------------------ device column sumcheck

class DeviceSumcheck:
    """The k' column rounds of one committed round on (T, C, 6) device tables (terms 0..Te-1 = eq terms, then the geom terms).
    ``ue``: (6C, 6) for one eq term or (Te, 6C, 6); ``cole``: (C, 6) or (Te, C, 6); ``coef``: (Te + Tg, 6).
    ``message()`` -> (3, 6) numpy (the first call: the top variable's g; later calls fold with the previous challenge first)."""

    def __init__(self, ug: torch.Tensor | None, ue: torch.Tensor, coef: np.ndarray, colg: np.ndarray | None,
                 cole: np.ndarray, C: int, device):
        T = coef.shape[0]
        Te = 1 if ue.dim() == 2 else ue.shape[0]
        Tg = T - Te
        ue = ue.contiguous()
        self.T, self.Cs, self.C, self.device = T, C, C, device
        self.A = torch.empty((T, C, D), dtype=torch.int32, device=device)
        self.B = torch.empty((T, C, D), dtype=torch.int32, device=device)
        self.g = torch.empty(18, dtype=torch.int32, device=device)
        self.g_host = torch.empty(18, dtype=torch.int32, pin_memory=True)
        coef_d = torch.as_tensor(np.ascontiguousarray(coef % P).astype(np.int32), device=device)
        cole_d = torch.as_tensor(np.ascontiguousarray(cole % P).astype(np.int32), device=device)
        if Tg:
            colg_d = torch.as_tensor(np.ascontiguousarray(colg % P).astype(np.int32), device=device)
            ug_p = _ptr(ug)
        else:
            colg_d = cole_d
            ug_p = _ptr(ue)
        n = T * C
        _launch("build_tables", ((n + 255) // 256,), (256,), (ug_p, np.int32(Tg), _ptr(ue), np.int32(Te), _ptr(coef_d), _ptr(colg_d),
                                                              _ptr(cole_d), np.int32(T), np.int32(C), _ptr(self.A), _ptr(self.B)))
        self._keep = (coef_d, cole_d, colg_d)
        self.r = None

    def message(self) -> np.ndarray:
        if self.r is None:
            args = (0,) * 6
            fold = 0
        else:
            args = tuple(int(v) for v in np.asarray(self.r, dtype=np.int64) % P)
            fold = 1
        _launch("sc_step", (1,), (1024,), (_ptr(self.A), _ptr(self.B), np.int32(self.T), np.int32(self.Cs), np.int32(self.C),
                                          np.int32(fold)) + tuple(np.uint32(a) for a in args) + (_ptr(self.g),))
        if fold:
            self.C //= 2
        self.g_host.copy_(self.g, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self.g_host.numpy().astype(np.int64).reshape(3, D)

    def fold(self, r: np.ndarray) -> None:
        self.r = r

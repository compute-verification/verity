"""Fused device kernels for the prover's tests (the 52% bucket after the SIMT encoder): the D-fold linear combination
of a row matrix and the batched quadratic test, each in ONE pass over the matrix instead of D (or 3 + D) torch passes.

  lincomb(C, X)      (D, cols) = C (D, R) @ X (R, cols) mod p                              -- protocol._combine
  lincomb(C, X, sq)  (D, cols) = sum_j C[d, j] (X_j^2 - X_j) mod p                         -- witness._bool_row x D
  quad_sparse(...)   (D, cols) = sum_q rho[d, q] (A_q(X) B_q(X) - C_q(X))  for the general (non-booleanity) constraints
                     with A_q = const + sum coef X_j read straight from the CSR tables    -- witness.apply_sparse x 3

One thread per column, a grid split over the rows (S partial sums, reduced mod p afterwards) so the L40S has
> 500K threads in flight; every X read is coalesced (a row across the warp) and X is read once.  Products
c * x < 2^62 are accumulated three at a time in 64 bits before one reduction.  Results are exact mod p: bit-exact
against the torch paths (tests_fused_test.py, and the gates run through them).  X may be int32 (the SIMT codeword)
or int64 (coefficient matrices); C and the tables are uint32 < p.
"""

from __future__ import annotations

import numpy as np
import torch

from .field import P

_SRC = r"""
#define P 2013265921u
typedef unsigned long long u64;

/* partial[s][d][col] = sum over rows [s*chunk, (s+1)*chunk) of C[d][j] * f(X[j][col]) */
template <typename T, int D, int SQ>
__device__ __forceinline__ void lincomb_body(const unsigned* __restrict__ C, const T* __restrict__ X,
                                             unsigned* __restrict__ partial, int R, int cols, int chunk)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (col >= cols) return;
    const int j0 = s * chunk, j1 = min(R, j0 + chunk);
    u64 acc[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) acc[d] = 0;
    int cnt = 0;
    for (int j = j0; j < j1; ++j) {
        u64 x = (u64)(unsigned)X[(size_t)j * cols + col];
        if (SQ) x = (x * x + (u64)P - x) % P;              /* x^2 - x mod p (x < p) */
        #pragma unroll
        for (int d = 0; d < D; ++d) acc[d] += (u64)C[(size_t)d * R + j] * x;
        if (++cnt == 3) {
            cnt = 0;
            #pragma unroll
            for (int d = 0; d < D; ++d) acc[d] %= P;
        }
    }
    #pragma unroll
    for (int d = 0; d < D; ++d) partial[((size_t)s * D + d) * cols + col] = (unsigned)(acc[d] % P);
}

extern "C" __global__ void lincomb_u32_D@D@(const unsigned* C, const unsigned* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_body<unsigned, @D@, 0>(C, X, out, R, cols, chunk); }
extern "C" __global__ void lincomb_u64_D@D@(const unsigned* C, const u64* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_body<u64, @D@, 0>(C, X, out, R, cols, chunk); }
extern "C" __global__ void boolcomb_u32_D@D@(const unsigned* C, const unsigned* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_body<unsigned, @D@, 1>(C, X, out, R, cols, chunk); }
extern "C" __global__ void boolcomb_u64_D@D@(const unsigned* C, const u64* X, unsigned* out, int R, int cols, int chunk)
{ lincomb_body<u64, @D@, 1>(C, X, out, R, cols, chunk); }

/* general quadratic constraints from CSR tables: for side s in {a,b,c}: ptr_s[q]..ptr_s[q+1] -> (j, coef), const_s[q] */
template <typename T>
__device__ __forceinline__ unsigned sparse_eval(const T* __restrict__ X, int cols, int col,
                                                const int* __restrict__ ptr, const int* __restrict__ jj,
                                                const unsigned* __restrict__ cc, unsigned cst, int q)
{
    u64 acc = cst;
    int cnt = 0;
    for (int e = ptr[q]; e < ptr[q + 1]; ++e) {
        acc += (u64)cc[e] * (u64)(unsigned)X[(size_t)jj[e] * cols + col];
        if (++cnt == 3) { cnt = 0; acc %= P; }
    }
    return (unsigned)(acc % P);
}

template <typename T, int D>
__device__ __forceinline__ void quad_body(const unsigned* __restrict__ rho, const T* __restrict__ X,
                                          const int* pa, const int* ja, const unsigned* ca, const unsigned* ka,
                                          const int* pb, const int* jb, const unsigned* cb, const unsigned* kb,
                                          const int* pc, const int* jc, const unsigned* ccn, const unsigned* kc,
                                          unsigned* __restrict__ partial, int Q, int cols, int chunk)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (col >= cols) return;
    const int q0 = s * chunk, q1 = min(Q, q0 + chunk);
    u64 acc[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) acc[d] = 0;
    int cnt = 0;
    for (int q = q0; q < q1; ++q) {
        u64 a = sparse_eval<T>(X, cols, col, pa, ja, ca, ka[q], q);
        u64 b = sparse_eval<T>(X, cols, col, pb, jb, cb, kb[q], q);
        u64 c = sparse_eval<T>(X, cols, col, pc, jc, ccn, kc[q], q);
        u64 t = (a * b + (u64)P - c) % P;
        #pragma unroll
        for (int d = 0; d < D; ++d) acc[d] += (u64)rho[(size_t)d * Q + q] * t;
        if (++cnt == 3) {
            cnt = 0;
            #pragma unroll
            for (int d = 0; d < D; ++d) acc[d] %= P;
        }
    }
    #pragma unroll
    for (int d = 0; d < D; ++d) partial[((size_t)s * D + d) * cols + col] = (unsigned)(acc[d] % P);
}

extern "C" __global__ void quad_u32_D@D@(const unsigned* rho, const unsigned* X,
    const int* pa, const int* ja, const unsigned* ca, const unsigned* ka,
    const int* pb, const int* jb, const unsigned* cb, const unsigned* kb,
    const int* pc, const int* jc, const unsigned* ccn, const unsigned* kc, unsigned* out, int Q, int cols, int chunk)
{ quad_body<unsigned, @D@>(rho, X, pa, ja, ca, ka, pb, jb, cb, kb, pc, jc, ccn, kc, out, Q, cols, chunk); }
extern "C" __global__ void quad_u64_D@D@(const unsigned* rho, const u64* X,
    const int* pa, const int* ja, const unsigned* ca, const unsigned* ka,
    const int* pb, const int* jb, const unsigned* cb, const unsigned* kb,
    const int* pc, const int* jc, const unsigned* ccn, const unsigned* kc, unsigned* out, int Q, int cols, int chunk)
{ quad_body<u64, @D@>(rho, X, pa, ja, ca, ka, pb, jb, cb, kb, pc, jc, ccn, kc, out, Q, cols, chunk); }
"""

THREADS = 256
MIN_THREADS_IN_FLIGHT = 1 << 19     # split the rows until this many threads are resident (L40S: 142 SMs x 2048)


def available() -> bool:
    try:
        import cupy  # noqa: F401
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class FusedTests:
    def __init__(self, D: int, device):
        import cupy as cp
        self.cp = cp
        self.D = D
        self.device = torch.device(device)
        self.mod = cp.RawModule(code=_SRC.replace("@D@", str(D)), options=("-std=c++17",))
        self.k = {name: self.mod.get_function(f"{name}_D{D}")
                  for name in ("lincomb_u32", "lincomb_u64", "boolcomb_u32", "boolcomb_u64", "quad_u32", "quad_u64")}

    def _cp(self, t: torch.Tensor):
        return self.cp.asarray(t)

    def _stream(self):
        import warnings
        ts = torch.cuda.current_stream(self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return self.cp.cuda.ExternalStream(ts.cuda_stream)

    @staticmethod
    def _grid(cols: int, R: int):
        splits = max(1, min(R, -(-MIN_THREADS_IN_FLIGHT // max(cols, 1))))
        chunk = -(-R // splits)
        splits = -(-R // chunk)
        return (-(-cols // THREADS), splits), chunk

    def _reduce(self, partial: torch.Tensor) -> torch.Tensor:
        # (S, D, cols) uint32 partial sums -> (D, cols) int64 mod p; S <= 512 terms < 2^31 fit in int64
        return partial.to(torch.int64).sum(0) % P

    def lincomb(self, C: torch.Tensor, X: torch.Tensor, square_minus: bool = False) -> torch.Tensor:
        """(D, cols) int64 = C (D, R) @ f(X) (R, cols) mod p; f = identity, or x^2 - x when ``square_minus``."""
        D, R = C.shape
        assert D == self.D and X.shape[0] == R
        cols = X.shape[1]
        X = X.contiguous()
        C32 = (C % P).to(torch.int32).contiguous()
        (gx, S), chunk = self._grid(cols, R)
        partial = torch.empty((S, D, cols), dtype=torch.int32, device=self.device)
        name = ("boolcomb" if square_minus else "lincomb") + ("_u32" if X.dtype == torch.int32 else "_u64")
        if X.dtype not in (torch.int32, torch.int64):
            raise TypeError(X.dtype)
        with self._stream():
            self.k[name]((gx, S), (THREADS,), (self._cp(C32), self._cp(X), self._cp(partial),
                                               np.int32(R), np.int32(cols), np.int32(chunk)))
        return self._reduce(partial)

    def quad_general(self, rho: torch.Tensor, X: torch.Tensor, csr: dict) -> torch.Tensor:
        """(D, cols) = sum_q rho[d, q] (A_q B_q - C_q)(X) over the general constraints in ``csr`` (see ``csr_tables``)."""
        D, Q = rho.shape
        assert D == self.D and Q == csr["Q"]
        cols = X.shape[1]
        X = X.contiguous()
        rho32 = (rho % P).to(torch.int32).contiguous()
        (gx, S), chunk = self._grid(cols, Q)
        partial = torch.empty((S, D, cols), dtype=torch.int32, device=self.device)
        name = "quad_u32" if X.dtype == torch.int32 else "quad_u64"
        args = [self._cp(rho32), self._cp(X)]
        for side in ("a", "b", "c"):
            args += [self._cp(csr[side][k]) for k in ("ptr", "j", "coef", "const")]
        args += [self._cp(partial), np.int32(Q), np.int32(cols), np.int32(chunk)]
        with self._stream():
            self.k[name]((gx, S), (THREADS,), tuple(args))
        return self._reduce(partial)


def csr_tables(gen: dict, Q: int, device) -> dict:
    """The (qq, jj, cc, consts) triples of witness._triples for the general constraints, as CSR (rows sorted by q)."""
    out = {"Q": Q}
    for side in ("a", "b", "c"):
        qq, jj, cc, consts = gen[side]
        order = torch.argsort(qq, stable=True)
        qq_s = qq[order]
        counts = torch.bincount(qq_s, minlength=Q)
        ptr = torch.zeros(Q + 1, dtype=torch.int64, device=device)
        ptr[1:] = torch.cumsum(counts, 0)
        out[side] = {"ptr": ptr.to(torch.int32).contiguous(), "j": jj[order].to(torch.int32).contiguous(),
                     "coef": (cc[order] % P).to(torch.int32).contiguous(), "const": (consts % P).to(torch.int32).contiguous()}
    return out


_FT: dict = {}


def fused_for(D: int, device) -> FusedTests | None:
    """The kernels for this D on this CUDA device, or None (CPU / no cupy)."""
    if not str(device).startswith("cuda") or not available():
        return None
    key = (D, str(device))
    f = _FT.get(key)
    if f is None:
        try:
            f = FusedTests(D, device)
        except Exception:  # noqa: BLE001 -- no NVRTC: the torch path
            return None
        _FT[key] = f
    return f

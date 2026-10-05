"""Tall-axis Reed-Solomon encoder for Ligerito: coefficient vectors of length R (one per column of M~) evaluated on
the subgroup <w_n>, n = 4 R (rate 1/4), as a 4-step Cooley-Tukey NTT n = n1 x n2 built from ONE cupy kernel that
transforms a (B, n, inner) tensor along its middle axis inside shared memory (radix-2 DIT on bit-reversed input,
BabyBear Montgomery twiddles in the layout of ``encode_simt.tables``: tw[m + j] = w_{2m}^j).

Message layout: ``msg`` is (Bm, R) row-major -- Bm messages (the columns of M~, or the 6 x C_i coordinate planes of
an extension matrix), each a contiguous coefficient vector.  Codeword: ``U`` (Bm, n) row-major.  Position
``s = k' n1 + k1`` of every message holds its polynomial at w_n^{k' + n2 k1} (``eta_index``): the 4-step writes the
inner transform's output index k' as the slow index.  For n <= 4096 a single step is used (n1 = n, n2 = 1, natural
order).

4-step (x[j1 + n1 j2], j1 < n1, j2 < n2; k = k' + n2 k1):
  A. view msg as (Bm, R / n1, n1) [j2, j1]; NTT along j2 of length n2 (input length R / n1 = n2 / 4, the rest zero),
     store x w_n^{j1 k'} (the post-twiddle table (n1, n2) is fused into the store)  -> (Bm, n2, n1) [k', j1]
  B. view as (Bm n2, n1, 1); NTT along j1 of length n1 in place                    -> (Bm, n2, n1) [k', k1]

Kernel: CTA = (b, group of G adjacent inner columns); G = 8 (one 32-byte sector per row) when n G 4 B <= 64 KiB,
else 4 / 2 / 1; threads 256; shared n x G words.  Step B (inner = 1) reads n contiguous words per CTA.
Without cupy (laptop tests) ``encode`` falls back to ``field.ntt`` on the last axis (n1 = n, n2 = 1).
"""
from __future__ import annotations

import numpy as np
import torch

from ._ligero import encode_simt as _es
from ._ligero import field as _field

_mont, _powers = _es._mont, _es._powers
P, root_of_unity = _field.P, _field.root_of_unity

try:
    import cupy as cp
except Exception:  # noqa: BLE001
    cp = None

_P_INV_MOD_R = pow(P, -1, 2**32)

_SRC = r"""
#define P 2013265921u
#define PINV %(pinv)du
#define N %(n)d
#define LOGN %(logn)d
#define G %(g)d
#define T %(threads)d

__device__ __forceinline__ unsigned add_p(unsigned a, unsigned b) { unsigned s = a + b; return min(s, s - P); }
__device__ __forceinline__ unsigned sub_p(unsigned a, unsigned b) { unsigned d = a - b; return min(d, d + P); }
__device__ __forceinline__ unsigned mont_mul(unsigned a, unsigned w) {
    unsigned long long x = (unsigned long long)a * w;
    unsigned lo = (unsigned)x, hi = (unsigned)(x >> 32);
    unsigned t = lo * PINV;
    unsigned u = __umulhi(t, P);
    unsigned r = hi - u;
    return min(r, r + P);
}

/* in: (B, n_in, inner); out: (B, N, inner); transform along the middle axis; post_tw (inner, N) Montgomery or null */
extern "C" __global__ void __launch_bounds__(T)
ntt_axis(const unsigned* __restrict__ in, unsigned* __restrict__ out, const unsigned* __restrict__ tw,
         const unsigned* __restrict__ post_tw, int n_in, int inner, int swap)
{
    extern __shared__ unsigned sh[];
    const int t = threadIdx.x;
    /* consecutive CTAs should walk ADJACENT column groups of the same batch (TLB locality: a batch spans n_in * inner
       words); grid.y is limited to 65535, so when the batch count exceeds it the roles are swapped */
    const int cg = (swap ? blockIdx.y : blockIdx.x) * G;
    const long long b = swap ? blockIdx.x : blockIdx.y;
    const unsigned* src = in + b * (long long)n_in * inner;
    unsigned* dst = out + b * (long long)N * inner;
    for (int e = t; e < N * G; e += T) {
        const int i = e / G, g = e - i * G;
        unsigned v = 0u;
        if (i < n_in) v = src[(long long)i * inner + cg + g];
        const int r = __brev((unsigned)i) >> (32 - LOGN);
        sh[r * G + g] = v;
    }
    __syncthreads();
    #pragma unroll 1
    for (int logm = 0; logm < LOGN; ++logm) {
        const int m = 1 << logm;
        for (int u = t; u < (N / 2) * G; u += T) {
            const int q = u / G, g = u - q * G;
            const int j = q & (m - 1);
            const int k = ((q >> logm) << (logm + 1)) + j;
            const unsigned w = tw[m + j];
            const unsigned a = sh[k * G + g];
            const unsigned c = mont_mul(sh[(k + m) * G + g], w);
            sh[k * G + g] = add_p(a, c);
            sh[(k + m) * G + g] = sub_p(a, c);
        }
        __syncthreads();
    }
    for (int e = t; e < N * G; e += T) {
        const int i = e / G, g = e - i * G;
        unsigned v = sh[i * G + g];
        if (post_tw != 0) v = mont_mul(v, post_tw[(long long)(cg + g) * N + i]);
        dst[(long long)i * inner + cg + g] = v;
    }
}
"""

_SRC_A = r"""
#define P 2013265921u
#define PINV %(pinv)du
#define ONE_M 268435454u              /* 1 in Montgomery form (2^32 mod p) */
#define NS %(ns)d                     /* sub-transform length = message rows per column (n2 / NC) */
#define LNS %(lns)d
#define G 8                           /* adjacent inner columns per CTA: one 32-byte sector per row */
#define T %(threads)d
#define TC (T / G)                    /* threads per column */
#define LTC %(ltc)d
#define E (NS / TC)                   /* elements per thread; E >= TC */
#define LE %(le)d
#define NC %(nc)d                     /* cosets = n2 / NS = 2^rate */

__device__ __forceinline__ unsigned add_p(unsigned a, unsigned b) { unsigned s = a + b; return min(s, s - P); }
__device__ __forceinline__ unsigned sub_p(unsigned a, unsigned b) { unsigned d = a - b; return min(d, d + P); }
__device__ __forceinline__ unsigned mont_mul(unsigned a, unsigned w) {
    unsigned long long x = (unsigned long long)a * w;
    unsigned lo = (unsigned)x, hi = (unsigned)(x >> 32);
    unsigned t = lo * PINV;
    unsigned u = __umulhi(t, P);
    unsigned r = hi - u;
    return min(r, r + P);
}
__device__ __forceinline__ unsigned pow_mont(unsigned b, unsigned e) {
    unsigned r = ONE_M;
    while (e) { if (e & 1u) r = mont_mul(r, b); b = mont_mul(b, b); e >>= 1; }
    return r;
}
/* bank-conflict-free for the scatter (bit-reversed rows), the E-contiguous phase and the stride-E phase */
__device__ __forceinline__ unsigned swz(unsigned r) { return r ^ ((r >> LE) & 3u); }

/* Step A of the 4-step, specialised to a zero-padded input: in (B, NS, inner) -> out (B, NC NS, inner), transform along the
   middle axis, x w_n^{j1 k'} fused into the store.  X[NC k0 + c] = NTT_NS(x_j w_{n2}^{j c})[k0]: the NS-point transform once per
   coset c with the input kept in registers; each transform is two register phases (LE stages on E-contiguous positions, LTC stages
   on stride-E positions) around one shared-memory transpose. */
extern "C" __global__ void __launch_bounds__(T)
ntt_pass_a(const unsigned* __restrict__ in, unsigned* __restrict__ out, const unsigned* __restrict__ tw_ns,
           const unsigned* __restrict__ tw_n2, unsigned wn_mont, int inner)
{
    extern __shared__ unsigned sh[];
    unsigned* stw = sh + NS * G;
    const int t = threadIdx.x;
    const int cg = blockIdx.x * G;
    const long long b = blockIdx.y;
    for (int i = t; i < NS; i += T) stw[i] = tw_ns[i];
    const int gl = t & (G - 1);
    const int ul = (((t >> 3) & 3) << (LTC - 2)) | (t >> 5);
    const unsigned* src = in + b * (long long)NS * inner + cg + gl;
    unsigned x[E];
    #pragma unroll
    for (int s = 0; s < E; ++s) x[s] = __ldg(src + (long long)(ul + TC * s) * inner);
    const int g = t & (G - 1), u = t >> 3;
    const unsigned a = pow_mont(wn_mont, (unsigned)(cg + g));
    const unsigned st = pow_mont(a, (unsigned)(NC * E));
    unsigned* dst = out + b * (long long)(NS * NC) * inner + cg + g;
    #pragma unroll 1
    for (int c = 0; c < NC; ++c) {
        __syncthreads();
        {
            unsigned wr = ONE_M, ws = ONE_M;
            if (c) { wr = __ldg(tw_n2 + ((ul * c) & (NS * NC - 1))); ws = __ldg(tw_n2 + ((TC * c) & (NS * NC - 1))); }
            #pragma unroll
            for (int s = 0; s < E; ++s) {
                unsigned v = x[s];
                if (c) { v = mont_mul(v, wr); wr = mont_mul(wr, ws); }
                const unsigned r = __brev((unsigned)(ul + TC * s)) >> (32 - LNS);
                sh[swz(r) * G + gl] = v;
            }
        }
        __syncthreads();
        {
            unsigned z[E];
            #pragma unroll
            for (int s = 0; s < E; ++s) z[s] = sh[swz(E * u + s) * G + g];
            #pragma unroll
            for (int lm = 0; lm < LE; ++lm) {
                const int m = 1 << lm;
                #pragma unroll
                for (int q = 0; q < E / 2; ++q) {
                    const int jj = q & (m - 1);
                    const int k = ((q >> lm) << (lm + 1)) + jj;
                    const unsigned aa = z[k], cc = mont_mul(z[k + m], stw[m + jj]);
                    z[k] = add_p(aa, cc);
                    z[k + m] = sub_p(aa, cc);
                }
            }
            #pragma unroll
            for (int s = 0; s < E; ++s) sh[swz(E * u + s) * G + g] = z[s];
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < E / TC; ++q) {
            const int lo = u + TC * q;
            unsigned y[TC];
            #pragma unroll
            for (int h = 0; h < TC; ++h) y[h] = sh[swz(lo + E * h) * G + g];
            #pragma unroll
            for (int lm = 0; lm < LTC; ++lm) {
                const int mh = 1 << lm;
                #pragma unroll
                for (int p = 0; p < TC / 2; ++p) {
                    const int jh = p & (mh - 1);
                    const int k = ((p >> lm) << (lm + 1)) + jh;
                    const unsigned aa = y[k], cc = mont_mul(y[k + mh], stw[E * mh + lo + E * jh]);
                    y[k] = add_p(aa, cc);
                    y[k + mh] = sub_p(aa, cc);
                }
            }
            unsigned w = pow_mont(a, (unsigned)(NC * lo + c));
            #pragma unroll
            for (int h = 0; h < TC; ++h) {
                dst[((long long)NC * (lo + E * h) + c) * inner] = mont_mul(y[h], w);
                w = mont_mul(w, st);
            }
        }
    }
}
"""

_KERNELS: dict = {}
_TABLES: dict = {}
_PASS_A_THREADS = {128: 64, 256: 128, 512: 128, 1024: 256, 2048: 256}


def _kernel_a(ns: int, nc: int):
    key = ("A", ns, nc)
    k = _KERNELS.get(key)
    if k is None:
        th = _PASS_A_THREADS[ns]
        tc = th // 8
        e = ns // tc
        src = _SRC_A % {"pinv": _P_INV_MOD_R, "ns": ns, "lns": ns.bit_length() - 1, "threads": th,
                        "ltc": tc.bit_length() - 1, "le": e.bit_length() - 1, "nc": nc}
        k = cp.RawKernel(src, "ntt_pass_a", options=("-std=c++17",))
        smem = 4 * (ns * 8 + ns)
        if smem > 48 * 1024:
            k.max_dynamic_shared_size_bytes = smem
        _KERNELS[key] = (k, th, smem)
    return _KERNELS[key]


def pass_a_supported(ns: int, nc: int, inner: int) -> bool:
    return cp is not None and ns in _PASS_A_THREADS and 1 <= nc <= 16 and inner % 8 == 0


def root_powers(n: int, device) -> torch.Tensor:
    """(n,): w_n^q (Montgomery), int32 on ``device``."""
    key = ("pow", n, str(device))
    t = _TABLES.get(key)
    if t is None:
        t = torch.from_numpy(_mont(_powers(root_of_unity(n), n)).view(np.int32)).to(device)
        _TABLES[key] = t
    return t


def ntt_pass_a(x: torch.Tensor, n2: int, n: int, out: torch.Tensor, stream: torch.cuda.Stream | None = None) -> torch.Tensor:
    """x (B, NS, inner) -> out (B, n2, inner): the 4-step's step A (NTT along the middle axis zero-padded to n2, times
    w_n^{j1 k'}), same output as ``ntt_axis(x, n2, out, post_twiddles(inner, n2))`` with inner = n1 = n / n2."""
    B, ns, inner = x.shape
    nc = n2 // ns
    k, th, smem = _kernel_a(ns, nc)
    tw = stage_twiddles(ns, x.device)
    tw2 = root_powers(n2, x.device)
    wn = int(_mont(np.array([root_of_unity(n)], dtype=np.uint64))[0])
    ts = stream if stream is not None else torch.cuda.current_stream(x.device)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cs = cp.cuda.ExternalStream(ts.cuda_stream)
    with cs:
        k((inner // 8, B), (th,),
          (np.uint64(x.data_ptr()), np.uint64(out.data_ptr()), np.uint64(tw.data_ptr()), np.uint64(tw2.data_ptr()),
           np.uint32(wn), np.int32(inner)), shared_mem=smem)
    return out


def _kernel(n: int, g: int, threads: int = 256):
    key = (n, g, threads)
    k = _KERNELS.get(key)
    if k is None:
        src = _SRC % {"pinv": _P_INV_MOD_R, "n": n, "logn": n.bit_length() - 1, "g": g, "threads": threads}
        k = cp.RawKernel(src, "ntt_axis", options=("-std=c++17",))
        smem = 4 * n * g
        if smem > 48 * 1024:
            k.max_dynamic_shared_size_bytes = smem
        _KERNELS[key] = k
    return k


def stage_twiddles(n: int, device) -> torch.Tensor:
    """tw[m + j] = w_n^{(n / 2m) j} (Montgomery), m = 1, 2, .., n/2; int32 on ``device``."""
    key = ("tw", n, str(device))
    t = _TABLES.get(key)
    if t is None:
        w = root_of_unity(n)
        tw = np.zeros(n, dtype=np.uint64)
        m = 1
        while m < n:
            tw[m:2 * m] = _powers(pow(w, n // (2 * m), P), m)
            m *= 2
        t = torch.from_numpy(_mont(tw).view(np.int32)).to(device)
        _TABLES[key] = t
    return t


def post_twiddles(n1: int, n2: int, device) -> torch.Tensor:
    """(n1, n2): w_n^{j1 k'} (Montgomery), n = n1 n2."""
    key = ("post", n1, n2, str(device))
    t = _TABLES.get(key)
    if t is None:
        w = root_of_unity(n1 * n2)
        rows = [_powers(pow(w, j1, P), n2) for j1 in range(n1)]
        t = torch.from_numpy(_mont(np.stack(rows, axis=0).reshape(-1)).view(np.int32)).to(device).reshape(n1, n2)
        _TABLES[key] = t
    return t


def _group(n: int, inner: int) -> int:
    for g in (8, 4, 2, 1):
        if g <= inner and inner % g == 0 and 4 * n * g <= 64 * 1024:
            return g
    raise ValueError(f"no CTA shape for n = {n}, inner = {inner}")


def ntt_axis(x: torch.Tensor, n: int, out: torch.Tensor | None = None, post_tw: torch.Tensor | None = None,
             stream: torch.cuda.Stream | None = None) -> torch.Tensor:
    """x (B, n_in, inner) int32 -> out (B, n, inner) int32: NTT along the middle axis (zero-padded to n)."""
    B, n_in, inner = x.shape
    assert n & (n - 1) == 0 and n_in <= n
    if out is None:
        out = torch.empty((B, n, inner), dtype=torch.int32, device=x.device)
    g = _group(n, inner)
    k = _kernel(n, g)
    tw = stage_twiddles(n, x.device)
    ts = stream if stream is not None else torch.cuda.current_stream(x.device)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cs = cp.cuda.ExternalStream(ts.cuda_stream)
    swap = B > 65535
    grid = (B, inner // g) if swap else (inner // g, B)
    with cs:
        k(grid, (256,),
          (np.uint64(x.data_ptr()), np.uint64(out.data_ptr()), np.uint64(tw.data_ptr()),
           np.uint64(post_tw.data_ptr() if post_tw is not None else 0), np.int32(n_in), np.int32(inner), np.int32(int(swap))),
          shared_mem=4 * n * g)
    return out


def split(n: int) -> tuple[int, int]:
    """(n1, n2) with n1 n2 = n, both <= 16384 (64 KiB shared memory at G = 1), n1 >= n2 (n <= 4096: (n, 1))."""
    if n <= 4096:
        return n, 1
    logn = n.bit_length() - 1
    log1 = (logn + 1) // 2
    n1, n2 = 1 << log1, 1 << (logn - log1)
    assert n1 <= 16384 and n2 <= 16384, f"codeword length {n} needs three factors (not implemented)"
    return n1, n2


class Encoder:
    """RS rate-2^-rate_log2 encoder for (Bm, R) coefficient rows on one device."""

    def __init__(self, R: int, device, rate_log2: int = 2):
        self.R, self.n = R, R << rate_log2
        self.device = torch.device(device)
        self.n1, self.n2 = split(self.n)
        self.gpu = cp is not None and self.device.type == "cuda"

    def eta_index(self, s) -> np.ndarray:
        """Codeword position(s) -> exponent e with eta_s = w_n^e."""
        s = np.asarray(s, dtype=np.int64)
        return (s // self.n1) + self.n2 * (s % self.n1)

    def etas(self, s) -> np.ndarray:
        w = root_of_unity(self.n)
        return np.array([pow(w, int(e), P) for e in self.eta_index(s)], dtype=np.int64)

    def encode(self, msg: torch.Tensor, out: torch.Tensor | None = None, pad: torch.Tensor | None = None) -> torch.Tensor:
        """msg (Bm, R) int32/int64 in [0, p) -> U (Bm, n) int32.  ``pad`` (Bm, t) in [0, p): ZK column padding, the message
        coefficients R .. R + t - 1 (R + t <= n)."""
        Bm, R = msg.shape
        assert R == self.R
        tp = 0 if pad is None else pad.shape[1]
        assert tp == 0 or (pad.shape[0] == Bm and R + tp <= self.n)
        if not self.gpu or (tp and self.n2 == 1):
            full = msg.to(torch.int64) if not tp else torch.cat((msg.to(torch.int64), pad.to(torch.int64)), dim=1)
            if not self.gpu:
                zeros = torch.zeros((Bm, self.n - R - tp), dtype=torch.int64, device=msg.device)
                return _field.ntt(torch.cat((full, zeros), dim=1)).to(torch.int32)
            msg, R = full.to(torch.int32).contiguous(), R + tp
            tp = 0
        if msg.dtype != torch.int32:
            msg = msg.to(torch.int32)
        msg = msg.contiguous()
        if out is None:
            out = torch.empty((Bm, self.n), dtype=torch.int32, device=self.device)
        n1, n2 = self.n1, self.n2
        if n2 == 1:
            ntt_axis(msg.view(Bm * 1, R, 1), self.n, out=out.view(Bm, self.n, 1))
            return out
        # step A: (Bm, R / n1, n1) -> (Bm, n2, n1) with the post twiddle
        if pass_a_supported(R // n1, n2 // (R // n1), n1):
            ntt_pass_a(msg.view(Bm, R // n1, n1), n2, self.n, out=out.view(Bm, n2, n1))
        else:
            ptw = post_twiddles(n1, n2, self.device)
            ntt_axis(msg.view(Bm, R // n1, n1), n2, out=out.view(Bm, n2, n1), post_tw=ptw)
        if tp:
            self._add_pad(out.view(Bm, n2, n1), pad)
        # step B: (Bm n2, n1, 1) in place
        v = out.view(Bm * n2, n1, 1)
        ntt_axis(v, n1, out=v)
        return out

    def _add_pad(self, X: torch.Tensor, pad: torch.Tensor) -> None:
        """Step A's output of coefficient j = R + t alone is X[:, k2, j mod n1] = pad_t w_n^{j k2}; add it (linearity)."""
        Bm, n2, n1 = X.shape
        tp = pad.shape[1]
        pad = pad.to(torch.int32).contiguous()
        k = _pad_kernel()
        ts = torch.cuda.current_stream(X.device)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cs = cp.cuda.ExternalStream(ts.cuda_stream)
        for t0 in range(0, tp, n1):                                             # n1 consecutive j: distinct slots j mod n1
            t1 = min(tp, t0 + n1)
            W = self._pad_twiddles(np.arange(self.R + t0, self.R + t1, dtype=np.int64))   # (n2, t1 - t0) int32
            cnt = Bm * n2 * (t1 - t0)
            with cs:
                k(((cnt + 255) // 256,), (256,),
                  (np.uint64(X.data_ptr()), np.uint64(pad.data_ptr()), np.uint64(W.data_ptr()), np.int32(Bm), np.int32(n2),
                   np.int32(n1), np.int32(tp), np.int32(t0), np.int32(t1 - t0), np.int32((self.R + t0) % n1)))

    def _pad_twiddles(self, j: np.ndarray) -> torch.Tensor:
        key = ("padtw", self.n, self.n2, int(j[0]), len(j), str(self.device))
        t = _TABLES.get(key)
        if t is None:
            w = root_of_unity(self.n)
            e = (np.arange(self.n2, dtype=np.int64)[:, None] * j[None, :]) % self.n
            lo_bits = 12
            lo = _powers(w, 1 << lo_bits).astype(np.int64)
            hi = _powers(pow(w, 1 << lo_bits, P), max(1, self.n >> lo_bits)).astype(np.int64)
            vals = hi[e >> lo_bits] * lo[e & ((1 << lo_bits) - 1)] % P
            t = torch.as_tensor(np.ascontiguousarray(vals).astype(np.int32), device=self.device)
            _TABLES[key] = t
        return t


_PAD_SRC = r"""
extern "C" __global__ void __launch_bounds__(256)
pad_add(unsigned int* X, const unsigned int* pad, const unsigned int* W, int Bm, int n2, int n1, int tp, int t0, int tc, int pos0)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)Bm * n2 * tc) return;
    const int t = (int)(i % tc);
    const long long bk = i / tc;
    const int k2 = (int)(bk % n2), b = (int)(bk / n2);
    const unsigned long long P = 2013265921ull;
    unsigned int* x = X + ((long long)b * n2 + k2) * n1 + (pos0 + t) % n1;
    const unsigned long long v = (unsigned long long)pad[(long long)b * tp + t0 + t] * W[(long long)k2 * tc + t] % P;
    *x = (unsigned int)((*x + v) % P);
}
"""


def _pad_kernel():
    k = _KERNELS.get("pad_add")
    if k is None:
        k = cp.RawKernel(_PAD_SRC, "pad_add")
        _KERNELS["pad_add"] = k
    return k


def reference_encode(msg: torch.Tensor, n: int, enc: Encoder) -> torch.Tensor:
    """Natural-order evaluations permuted to the encoder's position order (for the bit-exactness test)."""
    ntt = _field.ntt
    Bm, R = msg.shape
    pad = torch.zeros((Bm, n - R), dtype=torch.int64, device=msg.device)
    nat = ntt(torch.cat((msg.to(torch.int64), pad), dim=1))
    idx = torch.as_tensor(enc.eta_index(np.arange(n)), device=msg.device)
    return nat[:, idx].to(torch.int32)

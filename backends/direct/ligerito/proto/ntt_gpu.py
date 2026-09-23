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
         const unsigned* __restrict__ post_tw, int n_in, int inner)
{
    extern __shared__ unsigned sh[];
    const int t = threadIdx.x;
    const int cg = blockIdx.y * G;          /* batch on grid.x (2^31 limit), column groups on grid.y */
    const long long b = blockIdx.x;
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

_KERNELS: dict = {}
_TABLES: dict = {}


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
    with cs:
        k((B, inner // g), (256,),
          (np.uint64(x.data_ptr()), np.uint64(out.data_ptr()), np.uint64(tw.data_ptr()),
           np.uint64(post_tw.data_ptr() if post_tw is not None else 0), np.int32(n_in), np.int32(inner)),
          shared_mem=4 * n * g)
    return out


def split(n: int) -> tuple[int, int]:
    """(n1, n2) with n1 n2 = n, both <= 4096, n1 >= n2 (n <= 4096: (n, 1))."""
    if n <= 4096:
        return n, 1
    logn = n.bit_length() - 1
    log1 = (logn + 1) // 2
    n1, n2 = 1 << log1, 1 << (logn - log1)
    assert n1 <= 4096 and n2 <= 4096, f"codeword length {n} needs three factors (not implemented)"
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

    def encode(self, msg: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """msg (Bm, R) int32/int64 in [0, p) -> U (Bm, n) int32."""
        Bm, R = msg.shape
        assert R == self.R
        if not self.gpu:
            ntt = _field.ntt
            pad = torch.zeros((Bm, self.n - R), dtype=torch.int64, device=msg.device)
            return ntt(torch.cat((msg.to(torch.int64), pad), dim=1)).to(torch.int32)
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
        ptw = post_twiddles(n1, n2, self.device)
        ntt_axis(msg.view(Bm, R // n1, n1), n2, out=out.view(Bm, n2, n1), post_tw=ptw)
        # step B: (Bm n2, n1, 1) in place
        v = out.view(Bm * n2, n1, 1)
        ntt_axis(v, n1, out=v)
        return out


def reference_encode(msg: torch.Tensor, n: int, enc: Encoder) -> torch.Tensor:
    """Natural-order evaluations permuted to the encoder's position order (for the bit-exactness test)."""
    ntt = _field.ntt
    Bm, R = msg.shape
    pad = torch.zeros((Bm, n - R), dtype=torch.int64, device=msg.device)
    nat = ntt(torch.cat((msg.to(torch.int64), pad), dim=1))
    idx = torch.as_tensor(enc.eta_index(np.arange(n)), device=msg.device)
    return nat[:, idx].to(torch.int32)

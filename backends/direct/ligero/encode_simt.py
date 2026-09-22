"""The prover's Reed-Solomon encoder on SIMT cores: BabyBear int32 Montgomery butterflies, radix-4 fused shared-memory
stages, one CUDA block per row (the measured ``backends/direct/encode/encode_ntt_simt.py`` v2/radix4 kernel of the
b-encode / b-v2 lanes, 32.2 ms per 310,602 x 4096 rows on the 4090), specialised to THIS protocol's code:

* evaluation domain the coset ``g <w_n>`` (``g = 31``, disjoint from ``H = <w_l>``; PROTOCOL.md 3 / 8b, red-team F4),
  **not** the systematic ``<w_n>`` of the shared kernel: the four coset tables are ``l^-1 (g w_n^c)^i``, ``c < 4``;
* output in the protocol's natural order (position ``i`` holds ``p(g w_n^i)``): coset ``c``, index ``j`` is written to
  position ``4 j + c`` (the shared kernel writes coset blocks) -- bit-exact with ``field.encode`` / ``eval_on_coset``;
* ZK (8b): row ``j`` is committed as ``interp_H(W_j) + Z_H s_j`` with ``deg s_j < t_pad``.  On coset ``c`` the factor
  ``Z_H(x) = x^l - 1 = g^l w_n^{c l} - 1 =: z_c`` is a CONSTANT (``w_l^{j l} = 1``), so the mask enters as
  ``z_c s_j(x)``: the kernel adds ``s_{j,i} z_c (g w_n^c)^i`` to the (bit-reversed) coefficient ``i < t_pad`` before each
  coset NTT -- 256 multiply-adds per coset per row, no second NTT, no length-``l + 256`` transform;
* optionally the natural-order coefficients ``interp_H(W_j)`` (``l^-1`` x the DIF output, bit-reversal undone on the
  store), which the ZK prover needs for the coefficient-form messages ``w`` and ``v``.

Register / shared-memory layout, Montgomery arithmetic (``R = 2^32``, Plonky3-style signed reduction) and the radix-4
stage fusion are the shared kernel's; ``K`` here is ``l`` (a power of two, ``>= 4 x threads``), ``n = 4 l``.  Shared
memory is dynamic (``l = 16384`` needs 64 KiB > the 48 KiB static limit).  Tested bit-exact against ``field.encode``
and ``eval_on_coset`` (``encode_simt_test``); the verifier keeps the torch encoder on its ``D`` rows.
"""
from __future__ import annotations

import numpy as np
import torch

from .field import GEN, P, root_of_unity

__all__ = ["LigeroEncoderSIMT", "available", "encoder_for"]

_R_MOD_P = pow(2, 32, P)
_P_INV_MOD_R = pow(P, -1, 2**32)

_SRC = r"""
#define P 2013265921u
#define PINV %(pinv)du        /* p^{-1} mod 2^32 */
#define K %(k)d
#define LOGK %(logk)d
#define THREADS %(threads)d
#define LOGT %(logt)d
#define PER (K / THREADS)
#define TP %(tp)d             /* mask coefficients per row (0: no mask) */
#define KINV_MONT %(kinv_mont)du

__device__ __forceinline__ unsigned add_p(unsigned a, unsigned b) { unsigned s = a + b; return s >= P ? s - P : s; }
__device__ __forceinline__ unsigned sub_p(unsigned a, unsigned b) { return a >= b ? a - b : a + P - b; }
/* a in [0,p), w = w' R mod p (Montgomery): a w' mod p.  x = a w < p 2^32; t = x p^{-1} mod 2^32; (x - t p) / 2^32 =
   hi(x) - hi(t p) in (-p, p). */
__device__ __forceinline__ unsigned mont_mul(unsigned a, unsigned w) {
    unsigned long long x = (unsigned long long)a * w;
    unsigned lo = (unsigned)x, hi = (unsigned)(x >> 32);
    unsigned t = lo * PINV;
    unsigned u = __umulhi(t, P);
    unsigned r = hi - u;
    return hi < u ? r + P : r;
}

extern "C" __global__ void __launch_bounds__(THREADS)
rs_encode_ligero(const unsigned* __restrict__ in, unsigned* __restrict__ out, unsigned* __restrict__ coef_out,
                 const unsigned* __restrict__ mask,
                 const unsigned* __restrict__ tw_inv, const unsigned* __restrict__ tw_fwd,
                 const unsigned* __restrict__ coset_tw,   /* 4 x K: l^{-1} (g w_n^c)^i at bit-reversed position, Montgomery */
                 const unsigned* __restrict__ mask_tw,    /* 4 x TP: z_c (g w_n^c)^i, Montgomery */
                 int rows)
{
    extern __shared__ unsigned x[];
    unsigned r[PER], coef[PER];
    const int t = threadIdx.x;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const unsigned* src = in + (size_t)row * K;
        unsigned* dst = out + (size_t)row * (4 * K);
        #pragma unroll
        for (int s = 0; s < PER; ++s) r[s] = __ldg(&src[t + THREADS * s]);
        /* INTT, DIF: register stages m = K/2 .. THREADS */
        #pragma unroll
        for (int lm = LOGK - 1; lm >= LOGT; --lm) {
            const int m = 1 << lm, ms = m / THREADS;
            #pragma unroll
            for (int s = 0; s < PER; ++s) {
                if (!(s & ms)) {
                    const unsigned a = r[s], c = r[s + ms];
                    r[s] = add_p(a, c);
                    r[s + ms] = mont_mul(sub_p(a, c), __ldg(&tw_inv[m + t + THREADS * (s & (ms - 1))]));
                }
            }
        }
        #pragma unroll
        for (int s = 0; s < PER; ++s) x[t + THREADS * s] = r[s];
        __syncthreads();
        /* INTT, DIF: shared-memory stages, radix-4 fused (lm, lm - 1) */
        #pragma unroll
        for (int lm = LOGT - 1; lm >= 1; lm -= 2) {
            const int m = 1 << lm, h = m >> 1;
            #pragma unroll
            for (int s = 0; s < PER / 4; ++s) {
                const int q = t + s * THREADS;
                const int tt = q & (h - 1);
                const int i = ((q >> (lm - 1)) << (lm + 1)) | tt;
                const unsigned a0 = x[i], a1 = x[i + h], a2 = x[i + m], a3 = x[i + m + h];
                const unsigned b0 = add_p(a0, a2), b2 = mont_mul(sub_p(a0, a2), __ldg(&tw_inv[m + tt]));
                const unsigned b1 = add_p(a1, a3), b3 = mont_mul(sub_p(a1, a3), __ldg(&tw_inv[m + tt + h]));
                const unsigned wh = __ldg(&tw_inv[h + tt]);
                x[i] = add_p(b0, b1);
                x[i + h] = mont_mul(sub_p(b0, b1), wh);
                x[i + m] = add_p(b2, b3);
                x[i + m + h] = mont_mul(sub_p(b2, b3), wh);
            }
            __syncthreads();
        }
        if (LOGT & 1) {   /* one radix-2 stage left: m = 1 */
            #pragma unroll
            for (int s = 0; s < PER / 2; ++s) {
                const int b = t + s * THREADS;
                const int i = b << 1;
                const unsigned a = x[i], c = x[i + 1];
                x[i] = add_p(a, c);
                x[i + 1] = mont_mul(sub_p(a, c), __ldg(&tw_inv[1]));
            }
            __syncthreads();
        }
        #pragma unroll
        for (int s = 0; s < PER; ++s) coef[s] = x[t + THREADS * s];      /* bit-reversed, unscaled (x l) coefficients */
        if (coef_out != 0) {
            unsigned* cd = coef_out + (size_t)row * K;
            #pragma unroll
            for (int s = 0; s < PER; ++s) {
                const int pos = t + THREADS * s;
                cd[__brev((unsigned)pos) >> (32 - LOGK)] = mont_mul(coef[s], KINV_MONT);
            }
        }
        #pragma unroll 1
        for (int c = 0; c < 4; ++c) {
            const unsigned* ct = coset_tw + (size_t)c * K;
            #pragma unroll
            for (int s = 0; s < PER; ++s) x[t + THREADS * s] = mont_mul(coef[s], __ldg(&ct[t + THREADS * s]));
            __syncthreads();
#if TP > 0
            {   /* + z_c s_i (g w_n^c)^i at coefficient i < TP (bit-reversed position) */
                const unsigned* mk = mask + (size_t)row * TP;
                const unsigned* mt = mask_tw + (size_t)c * TP;
                for (int i = t; i < TP; i += THREADS) {
                    const int pos = __brev((unsigned)i) >> (32 - LOGK);
                    x[pos] = add_p(x[pos], mont_mul(__ldg(&mk[i]), __ldg(&mt[i])));
                }
                __syncthreads();
            }
#endif
            /* NTT, DIT: shared-memory stages, radix-4 fused (lm, lm + 1) */
            #pragma unroll
            for (int lm = 0; lm + 1 < LOGT; lm += 2) {
                const int m = 1 << lm, m2 = m << 1;
                #pragma unroll
                for (int s = 0; s < PER / 4; ++s) {
                    const int q = t + s * THREADS;
                    const int tt = q & (m - 1);
                    const int i = ((q >> lm) << (lm + 2)) | tt;
                    const unsigned a0 = x[i], a1 = x[i + m], a2 = x[i + m2], a3 = x[i + m2 + m];
                    const unsigned wm = __ldg(&tw_fwd[m + tt]);
                    const unsigned c1 = mont_mul(a1, wm), c3 = mont_mul(a3, wm);
                    const unsigned b0 = add_p(a0, c1), b1 = sub_p(a0, c1), b2 = add_p(a2, c3), b3 = sub_p(a2, c3);
                    const unsigned d2 = mont_mul(b2, __ldg(&tw_fwd[m2 + tt])), d3 = mont_mul(b3, __ldg(&tw_fwd[m2 + tt + m]));
                    x[i] = add_p(b0, d2);
                    x[i + m2] = sub_p(b0, d2);
                    x[i + m] = add_p(b1, d3);
                    x[i + m2 + m] = sub_p(b1, d3);
                }
                __syncthreads();
            }
            if (LOGT & 1) {   /* one radix-2 stage left: m = THREADS / 2 */
                const int m = THREADS >> 1;
                #pragma unroll
                for (int s = 0; s < PER / 2; ++s) {
                    const int b = t + s * THREADS;
                    const int tt = b & (m - 1);
                    const int i = ((b >> (LOGT - 1)) << LOGT) | tt;
                    const unsigned a = x[i];
                    const unsigned cc = mont_mul(x[i + m], __ldg(&tw_fwd[m + tt]));
                    x[i] = add_p(a, cc);
                    x[i + m] = sub_p(a, cc);
                }
                __syncthreads();
            }
            #pragma unroll
            for (int s = 0; s < PER; ++s) r[s] = x[t + THREADS * s];
            /* NTT, DIT: register stages m = THREADS .. K/2 */
            #pragma unroll
            for (int lm = LOGT; lm < LOGK; ++lm) {
                const int m = 1 << lm, ms = m / THREADS;
                #pragma unroll
                for (int s = 0; s < PER; ++s) {
                    if (!(s & ms)) {
                        const unsigned a = r[s];
                        const unsigned cc = mont_mul(r[s + ms], __ldg(&tw_fwd[m + t + THREADS * (s & (ms - 1))]));
                        r[s] = add_p(a, cc);
                        r[s + ms] = sub_p(a, cc);
                    }
                }
            }
            /* natural protocol order: p(g w_n^{4 j + c}) at position 4 j + c */
            #pragma unroll
            for (int s = 0; s < PER; ++s) dst[((size_t)(t + THREADS * s) << 2) | c] = r[s];
        }
    }
}
"""


def _powers(w: int, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.uint64)
    out[0] = 1
    step, cur = 1, w % P
    while step < count:
        m = min(step, count - step)
        out[step:step + m] = (out[:m] * np.uint64(cur)) % np.uint64(P)
        step *= 2
        cur = cur * cur % P
    return out


def _bitrev(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    idx = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def _mont(a: np.ndarray) -> np.ndarray:
    return ((a % np.uint64(P)) * np.uint64(_R_MOD_P) % np.uint64(P)).astype(np.uint32)


def tables(l: int, n: int, t_pad: int) -> dict[str, np.ndarray]:
    """Stage twiddles (DIF inverse / DIT forward over ``<w_l>``), the four coset tables ``l^-1 (g w_n^c)^i`` at
    bit-reversed position, and the mask tables ``z_c (g w_n^c)^i`` (``i < t_pad``), all Montgomery uint32."""
    assert n == 4 * l
    wn = root_of_unity(n)
    wl = pow(wn, 4, P)
    wl_inv = pow(wl, -1, P)
    tw_fwd = np.zeros(l, dtype=np.uint64)
    tw_inv = np.zeros(l, dtype=np.uint64)
    m = 1
    while m < l:
        tw_fwd[m:2 * m] = _powers(pow(wl, l // (2 * m), P), m)
        tw_inv[m:2 * m] = _powers(pow(wl_inv, l // (2 * m), P), m)
        m *= 2
    rev = _bitrev(l)
    linv = pow(l, -1, P)
    cos, msk = [], []
    for c in range(4):
        base = GEN * pow(wn, c, P) % P
        pw = _powers(base, l)
        cos.append(((pw * np.uint64(linv)) % np.uint64(P))[rev])
        if t_pad:
            z_c = (pow(base, l, P) - 1) % P
            assert z_c != 0
            msk.append((pw[:t_pad] * np.uint64(z_c)) % np.uint64(P))
    return {"tw_fwd": _mont(tw_fwd), "tw_inv": _mont(tw_inv), "coset": _mont(np.concatenate(cos)),
            "mask": _mont(np.concatenate(msk)) if t_pad else np.zeros(1, dtype=np.uint32),
            "kinv_mont": int(_mont(np.array([linv], dtype=np.uint64))[0])}


def available() -> bool:
    try:
        import cupy  # noqa: F401
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class LigeroEncoderSIMT:
    """``encode(rows (R, l) int64/int32 torch on the device, mask (R, t_pad) or None, want_coefs) -> (U (R, n) int32,
    coefs (R, l) int32 | None)`` in the protocol's domain and order."""

    def __init__(self, l: int, n: int, t_pad: int = 0, device="cuda", threads: int | None = None):
        import cupy as cp

        if n != 4 * l or l & (l - 1) or l < 256:
            raise ValueError("the kernel is specialised to n = 4 l, l a power of two >= 256")
        self.cp, self.l, self.n, self.t_pad, self.device = cp, l, n, t_pad, torch.device(device)
        threads = threads or min(1024, l // 4)
        if l < 4 * threads or threads & (threads - 1) or threads > 1024:
            raise ValueError("threads must be a power of two with l >= 4 threads")
        self.threads = threads
        tabs = tables(l, n, t_pad)
        params = {"pinv": _P_INV_MOD_R, "k": l, "logk": l.bit_length() - 1, "threads": threads,
                  "logt": threads.bit_length() - 1, "tp": t_pad, "kinv_mont": tabs["kinv_mont"]}
        self.kernel = cp.RawKernel(_SRC % params, "rs_encode_ligero", options=("-std=c++17",))
        self.smem = 4 * l
        if self.smem > 48 * 1024:
            self.kernel.max_dynamic_shared_size_bytes = self.smem
        with cp.cuda.Device(self.device.index or 0):
            self.tw_fwd = cp.asarray(tabs["tw_fwd"])
            self.tw_inv = cp.asarray(tabs["tw_inv"])
            self.coset = cp.asarray(tabs["coset"])
            self.mask_tw = cp.asarray(tabs["mask"])
            sms = cp.cuda.runtime.getDeviceProperties(self.device.index or 0)["multiProcessorCount"]
        self.blocks = sms * 8

    def _u32(self, t: torch.Tensor):
        return self.cp.asarray(t).view(self.cp.uint32)

    def encode(self, rows: torch.Tensor, mask: torch.Tensor | None = None, want_coefs: bool = False,
               out: torch.Tensor | None = None):
        cp = self.cp
        R = rows.shape[0]
        if rows.shape != (R, self.l):
            raise ValueError(rows.shape)
        if (mask is not None) != (self.t_pad > 0):
            raise ValueError("mask must be given iff the encoder was built with t_pad > 0")
        rows32 = rows if rows.dtype == torch.int32 else rows.to(torch.int32)      # values < p < 2^31
        rows32 = rows32.contiguous()
        if out is None:
            out = torch.empty((R, self.n), dtype=torch.int32, device=self.device)
        elif out.shape != (R, self.n) or out.dtype != torch.int32 or not out.is_contiguous():
            raise ValueError("out must be a contiguous (R, n) int32 tensor")
        coefs = torch.empty((R, self.l), dtype=torch.int32, device=self.device) if want_coefs else None
        if mask is not None:
            if mask.shape != (R, self.t_pad):
                raise ValueError(mask.shape)
            mask32 = (mask if mask.dtype == torch.int32 else mask.to(torch.int32)).contiguous()
            mask_arg = self._u32(mask32)
        else:
            mask_arg = self.tw_fwd            # unused: TP = 0
        coef_arg = self._u32(coefs) if coefs is not None else np.uint64(0)
        # torch's current stream vs cupy's: launch on torch's stream
        import warnings
        ts = torch.cuda.current_stream(self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            stream = cp.cuda.ExternalStream(ts.cuda_stream)
        with stream:
            self.kernel((min(self.blocks, R),), (self.threads,),
                        (self._u32(rows32), self._u32(out), coef_arg, mask_arg, self.tw_inv, self.tw_fwd, self.coset,
                         self.mask_tw, np.int32(R)), shared_mem=self.smem)
        return out, coefs


_ENC: dict = {}


def encoder_for(l: int, n: int, t_pad: int, device) -> LigeroEncoderSIMT:
    key = (l, n, t_pad, str(device))
    e = _ENC.get(key)
    if e is None:
        e = LigeroEncoderSIMT(l, n, t_pad, device)
        _ENC[key] = e
    return e

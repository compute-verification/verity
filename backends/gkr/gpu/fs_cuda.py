"""The Fiat-Shamir sponge of :mod:`transcript` on the device (cupy ``RawKernel``, NVRTC), so a CUDA graph can draw its
own challenges: ``step(msg, out0[, out1])`` absorbs the int32 words ``msg`` and draws one or two extension challenges,
bit-identical to ``Transcript.absorb_exts`` followed by ``Transcript.challenge()`` calls.

Byte model (``transcript.py``): the state is the 32-byte SHA-256 digest, kept here as its eight words ``H0..H7`` (the
big-endian serialisation is the digest); ``absorb``: ``state <- SHA256(state || msg)`` with every coordinate a
little-endian u32 (so its SHA-256 word is the byte swap); ``challenge``: SHA-256 of ``state || "chal" || ctr_le64``
for ``ctr, ctr + 1, ...``, each digest read as eight little-endian u32 (the byte swaps of ``H_j``), a word accepted iff
below ``p``, the first six accepted words form the challenge and the rest of the last digest is dropped (each
``challenge()`` starts a new generator).  The kernel hashes eight counters at once (one thread each, one block) and
consumes them in order, so a challenge costs one compression of latency instead of ~2.

The SHA-256 compression is ``hash_gpu/sha256_cuda.py``'s (FIPS 180-4 constants and round functions)."""
from __future__ import annotations

import numpy as np
import torch

try:
    import cupy as cp
except ImportError:            # pragma: no cover - laptop
    cp = None

from .field import P, W

SRC = f"#define P_ {P}ull\n#define W_ {W}ull\n" + r"""
typedef unsigned int u32;
typedef unsigned long long u64;

__constant__ u32 K256[64] = {
 0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
 0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
 0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
 0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
 0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
 0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
 0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
 0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ u32 rotr(u32 x, int n) { return __funnelshift_r(x, x, n); }
__device__ __forceinline__ u32 bswap(u32 x) { return __byte_perm(x, 0, 0x0123); }
#define CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)  (rotr(x,2) ^ rotr(x,13) ^ rotr(x,22))
#define EP1(x)  (rotr(x,6) ^ rotr(x,11) ^ rotr(x,25))
#define SIG0(x) (rotr(x,7) ^ rotr(x,18) ^ ((x) >> 3))
#define SIG1(x) (rotr(x,17) ^ rotr(x,19) ^ ((x) >> 10))

__device__ __forceinline__ void iv(u32 h[8]) {
    h[0]=0x6a09e667; h[1]=0xbb67ae85; h[2]=0x3c6ef372; h[3]=0xa54ff53a;
    h[4]=0x510e527f; h[5]=0x9b05688c; h[6]=0x1f83d9ab; h[7]=0x5be0cd19;
}

__device__ __forceinline__ void sha256_compress(u32 st[8], u32 w[16]) {
    u32 a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        u32 wi;
        if (i < 16) wi = w[i];
        else {
            wi = SIG1(w[(i-2)&15]) + w[(i-7)&15] + SIG0(w[(i-15)&15]) + w[i&15];
            w[i&15] = wi;
        }
        u32 t1 = h + EP1(e) + CH(e,f,g) + K256[i] + wi;
        u32 t2 = EP0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

// c = a b in F_p[X] / (X^6 - W), canonical in and out
__device__ __forceinline__ void ext_mul(const u32 a[6], const u32 b[6], u32 c[6]) {
    u64 lo[6] = {0, 0, 0, 0, 0, 0}, hi[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            const u64 t = ((u64)a[i] * (u64)b[j]) % P_;
            if (i + j < 6) lo[i + j] += t; else hi[i + j - 6] += t;
        }
    for (int d = 0; d < 6; ++d) c[d] = (u32)((lo[d] + (hi[d] % P_) * W_) % P_);
}

// One block of 8 threads.  st_a: u32[8] state words, ctr_a: u64 counter, msg_a: nmsg u32 words (LE coordinates),
// out0_a / out1_a: u32[6] challenge destinations (nchal = 1 or 2).  pfx_a (optional): u32[6] running product; with
// rho_a (u32[6]) it is multiplied by eq(rho, r) = 1 - rho - r + 2 rho r for the first challenge r; reset_pfx sets it to 1.
extern "C" __global__ void fs_step(u64 st_a, u64 ctr_a, u64 msg_a, int nmsg, u64 out0_a, u64 out1_a, int nchal, u32 p,
                                   u64 rho_a, u64 pfx_a, int reset_pfx)
{
    u32 r0[6] = {0, 0, 0, 0, 0, 0};
    u32* st_g = (u32*)st_a;
    u64* ctr_g = (u64*)ctr_a;
    const u32* msg = (const u32*)msg_a;
    __shared__ u32 st[8];
    __shared__ u32 dig[8][8];
    __shared__ u64 ctr;
    __shared__ int done;
    const int t = threadIdx.x;
    if (t == 0) {
        u32 h[8];
        iv(h);
        const int total = 8 + nmsg;
        const int padded = ((total + 3 + 15) / 16) * 16;
        const u64 bitlen = (u64)total * 32ull;
        u32 w[16];
        for (int blk = 0; blk < padded; blk += 16) {
            for (int j = 0; j < 16; ++j) {
                const int i = blk + j;
                u32 v;
                if (i < 8) v = st_g[i];
                else if (i < total) v = bswap(msg[i - 8]);
                else if (i == total) v = 0x80000000u;
                else v = 0u;
                if (i == padded - 2) v = (u32)(bitlen >> 32);
                if (i == padded - 1) v = (u32)bitlen;
                w[j] = v;
            }
            sha256_compress(h, w);
        }
        for (int j = 0; j < 8; ++j) { st[j] = h[j]; st_g[j] = h[j]; }
        ctr = *ctr_g;
        done = 0;
    }
    __syncthreads();
    int c = 0, have = 0;
    u32 acc[6];
    u32* outs[2] = {(u32*)out0_a, (u32*)out1_a};
    while (true) {
        u32 h[8], w[16];
        iv(h);
        const u64 cc = ctr + (u64)t;
        for (int j = 0; j < 8; ++j) w[j] = st[j];
        w[8] = 0x6368616cu;                    // "chal"
        w[9] = bswap((u32)cc);
        w[10] = bswap((u32)(cc >> 32));
        w[11] = 0x80000000u;
        w[12] = 0u; w[13] = 0u; w[14] = 0u;
        w[15] = 44u * 8u;
        sha256_compress(h, w);
        for (int j = 0; j < 8; ++j) dig[t][j] = h[j];
        __syncthreads();
        if (t == 0) {
            u64 used = 0;
            for (int b = 0; b < 8 && c < nchal; ++b) {
                ++used;
                for (int j = 0; j < 8; ++j) {
                    const u32 v = bswap(dig[b][j]);
                    if (v < p) {
                        acc[have++] = v;
                        if (have == 6) break;
                    }
                }
                if (have == 6) {
                    for (int j = 0; j < 6; ++j) outs[c][j] = acc[j];
                    if (c == 0)
                        for (int j = 0; j < 6; ++j) r0[j] = acc[j];
                    ++c;
                    have = 0;
                }
            }
            ctr += used;
            done = c == nchal;
        }
        __syncthreads();
        if (done) break;
    }
    if (t == 0) {
        *ctr_g = ctr;
        if (pfx_a) {
            u32* pfx = (u32*)pfx_a;
            if (rho_a) {
                const u32* a = (const u32*)rho_a;
                u32 ab[6], e[6], pv[6], np[6];
                ext_mul(a, r0, ab);
                for (int d = 0; d < 6; ++d)
                    e[d] = (u32)((2ull * ab[d] + (d == 0 ? 1ull : 0ull) + 2ull * P_ - a[d] - r0[d]) % P_);
                for (int d = 0; d < 6; ++d) pv[d] = pfx[d];
                ext_mul(pv, e, np);
                for (int d = 0; d < 6; ++d) pfx[d] = np[d];
            }
            if (reset_pfx) {
                pfx[0] = 1u;
                for (int d = 1; d < 6; ++d) pfx[d] = 0u;
            }
        }
    }
}
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(SRC, "fs_step", options=("--std=c++14",))
    return _KERNEL


def available() -> bool:
    return cp is not None and torch.cuda.is_available()


class DeviceTranscript:
    """The sponge state on ``device``: ``st`` (8 u32 words as int32) and ``ctr`` (one int64).  ``load`` / ``store`` move
    it to / from a host :class:`transcript.Transcript`; ``step`` is stream-ordered on the current torch stream (and so
    capturable in a CUDA graph)."""

    def __init__(self, device) -> None:
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.buf = torch.zeros(10, dtype=torch.int32, device=self.device)   # st[0:8] | ctr (two int32 halves)
        self.st = self.buf[:8]
        self.ctr = self.buf[8:].view(torch.int64)
        self.kernel = _kernel()

    def load(self, tr) -> None:
        h = np.empty(10, dtype=np.int32)
        h[:8] = np.frombuffer(tr.state, dtype=">u4").astype(np.uint32).view(np.int32)
        h[8:] = np.array([tr.ctr], dtype=np.int64).view(np.int32)
        self.buf.copy_(torch.from_numpy(h).pin_memory(), non_blocking=True)

    @staticmethod
    def store(tr, words: np.ndarray, slots: int) -> None:
        """``words``: the 10 int32 of ``buf`` read back; ``slots``: challenges drawn on the device since ``load``."""
        words = np.ascontiguousarray(words, dtype=np.int32)
        tr.state = words[:8].view(np.uint32).astype(">u4").tobytes()
        tr.ctr = int(words[8:].view(np.int64)[0])
        tr.slots += slots

    def step(self, msg: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor | None = None, *,
             rho: torch.Tensor | None = None, pfx: torch.Tensor | None = None, reset: bool = False) -> None:
        """Absorb ``msg``, draw ``out0`` (and ``out1``); with ``pfx``: ``pfx *= eq(rho, out0)`` when ``rho`` is given, then
        ``pfx = 1`` when ``reset``."""
        for t in (msg, out0, out1, rho, pfx):
            assert t is None or (t.dtype == torch.int32 and t.is_contiguous() and t.device == self.device)
        assert out0.numel() == 6 and (out1 is None or out1.numel() == 6)
        assert (rho is None and not reset) or pfx is not None

        def ptr(t):
            return np.uint64(0 if t is None else t.data_ptr())

        with cp.cuda.ExternalStream(torch.cuda.current_stream(self.device).cuda_stream):
            self.kernel((1,), (8,), (ptr(self.st), ptr(self.ctr), ptr(msg), np.int32(msg.numel()), ptr(out0), ptr(out1),
                                     np.int32(1 if out1 is None else 2), np.uint32(P), ptr(rho), ptr(pfx), np.int32(reset)))


__all__ = ["DeviceTranscript", "available"]

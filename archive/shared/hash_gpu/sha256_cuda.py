"""SHA-256 on the GPU (cupy ``RawKernel``, NVRTC), one thread per message.

Written from FIPS 180-4 following the structure of Brad Conte's public-domain ``crypto-algorithms/sha256.c``
(the round constants and the ``EP0/EP1/SIG0/SIG1`` decomposition); rotations use ``__funnelshift_r``.

Message model (``hash_strided``): thread ``(x, y)`` hashes ``nwords`` 32-bit words read from
``base[y * ystride + x * xstride + i * wstride]`` for ``i < nwords``.  With ``bswap = 1`` each word is a
little-endian ``u32`` element whose 4 bytes are the message bytes (so the SHA-256 big-endian word is the byte
swap); with ``bswap = 0`` the words are SHA-256 words already (digests stored as ``H0..H7`` values).  The message
length is ``4 * nwords`` bytes, so padding starts on a word boundary.  Output: ``out[(y * nx + x) * 8 + j] = H_j``.

Uses: column leaves (``x`` = column, ``xstride = 1``, ``wstride = ncols``, ``nwords = rows``), 1 KiB sub-chunks
(``y`` = chunk, ``ystride = 256 * ncols``), tree nodes (``xstride = 16``, ``wstride = 1``, ``nwords = 16``).
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:            # pragma: no cover - laptop
    cp = None

SRC = r"""
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
#define CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)  (rotr(x,2) ^ rotr(x,13) ^ rotr(x,22))
#define EP1(x)  (rotr(x,6) ^ rotr(x,11) ^ rotr(x,25))
#define SIG0(x) (rotr(x,7) ^ rotr(x,18) ^ ((x) >> 3))
#define SIG1(x) (rotr(x,17) ^ rotr(x,19) ^ ((x) >> 10))

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

extern "C" __global__
void sha256_strided(const u32* __restrict__ base, long long xstride, long long ystride, long long wstride,
                    int nwords, int nx, int ny, int bswap, int out_xy, u32* __restrict__ out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= nx || y >= ny) return;
    const u32* m = base + (long long)y * ystride + (long long)x * xstride;
    u32 st[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u64 bitlen = (u64)nwords * 32ull;
    // total words after padding: nwords + 1 (0x80) + zeros + 2 (length), rounded up to a multiple of 16
    int total = ((nwords + 3 + 15) / 16) * 16;
    u32 w[16];
    for (int blk = 0; blk < total; blk += 16) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            int i = blk + j;
            u32 v;
            if (i < nwords) { v = m[(long long)i * wstride]; if (bswap) v = __byte_perm(v, 0, 0x0123); }
            else if (i == nwords) v = 0x80000000u;
            else if (i == total - 2) v = (u32)(bitlen >> 32);
            else if (i == total - 1) v = (u32)bitlen;
            else v = 0u;
            w[j] = v;
        }
        sha256_compress(st, w);
    }
    u32* o = out + (out_xy ? ((long long)x * ny + y) : ((long long)y * nx + x)) * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = st[j];
}
"""

_kernel = None


def kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(SRC, "sha256_strided", options=("-std=c++17",))
    return _kernel


def hash_strided(base, xstride: int, ystride: int, wstride: int, nwords: int, nx: int, ny: int, bswap: bool,
                 out=None, block: int = 128, out_xy: bool = False):
    """Launch ``nx * ny`` SHA-256 hashes of ``nwords`` words each (module docstring).  ``base`` is a cupy uint32
    array (or any ``__cuda_array_interface__`` device buffer of u32); returns ``out`` of shape ``(ny * nx, 8)``
    indexed ``y * nx + x`` (default) or ``x * ny + y`` (``out_xy``)."""
    base = cp.asarray(base)
    if base.dtype != cp.uint32:
        base = base.view(cp.uint32)
    if out is None:
        out = cp.empty((ny * nx, 8), dtype=cp.uint32)
    grid = ((nx + block - 1) // block, ny)
    kernel()(grid, (block,), (base, np.int64(xstride), np.int64(ystride), np.int64(wstride), np.int32(nwords),
                              np.int32(nx), np.int32(ny), np.int32(1 if bswap else 0), np.int32(1 if out_xy else 0), out))
    return out


def digests_to_bytes(d) -> list[bytes]:
    """``(n, 8)`` SHA-256 word digests -> the standard 32-byte digests (big-endian words)."""
    a = cp.asnumpy(d) if cp is not None and isinstance(d, cp.ndarray) else np.asarray(d)
    a = a.astype(">u4")
    return [row.tobytes() for row in a]

"""SHA-512 Merkle commitment on the GPU (cupy ``RawKernel``, NVRTC): the Ligero column tree of ``ligero.Merkle``.

Written from FIPS 180-4 (section 4.1.3 functions, 4.2.3 constants, 5.1.2 padding, 5.3.5 IV, 6.4 computation) in the
structure of ``hash_gpu/sha256_cuda.py``: one thread per message.  Digests are kept as the eight state words
``H0..H7`` (u64); the standard 64-byte digest is their big-endian serialisation (``digest_bytes``).

  leaf ``j``   SHA-512 of column ``j``'s ``rows`` values, each a little-endian u32 (4 message bytes), top row first
  node         SHA-512(left || right) over the two 64-byte child digests (one full block + the padding block)

A tree over ``N`` leaves promotes an unpaired rightmost node unchanged (``N`` is a power of two for Ligero).
Why SHA-512 and not SHA-256: the commitment's binding term is the generic ``q^2 / 2^h`` with ``q = 2^64`` hash
evaluations (security.accounting ``hash_collision_bound``); at ``h = 256`` that term alone is ``2^-128``, so the
protocol total could not reach ``2^-128`` (gpu/README.md "Soundness").
"""
from __future__ import annotations

import hashlib

import numpy as np

try:
    import cupy as cp
except ImportError:            # pragma: no cover - laptop
    cp = None

DIGEST_BYTES = 64

SRC = r"""
typedef unsigned int u32;
typedef unsigned long long u64;

__constant__ u64 K512[80] = {
 0x428a2f98d728ae22ull,0x7137449123ef65cdull,0xb5c0fbcfec4d3b2full,0xe9b5dba58189dbbcull,0x3956c25bf348b538ull,
 0x59f111f1b605d019ull,0x923f82a4af194f9bull,0xab1c5ed5da6d8118ull,0xd807aa98a3030242ull,0x12835b0145706fbeull,
 0x243185be4ee4b28cull,0x550c7dc3d5ffb4e2ull,0x72be5d74f27b896full,0x80deb1fe3b1696b1ull,0x9bdc06a725c71235ull,
 0xc19bf174cf692694ull,0xe49b69c19ef14ad2ull,0xefbe4786384f25e3ull,0x0fc19dc68b8cd5b5ull,0x240ca1cc77ac9c65ull,
 0x2de92c6f592b0275ull,0x4a7484aa6ea6e483ull,0x5cb0a9dcbd41fbd4ull,0x76f988da831153b5ull,0x983e5152ee66dfabull,
 0xa831c66d2db43210ull,0xb00327c898fb213full,0xbf597fc7beef0ee4ull,0xc6e00bf33da88fc2ull,0xd5a79147930aa725ull,
 0x06ca6351e003826full,0x142929670a0e6e70ull,0x27b70a8546d22ffcull,0x2e1b21385c26c926ull,0x4d2c6dfc5ac42aedull,
 0x53380d139d95b3dfull,0x650a73548baf63deull,0x766a0abb3c77b2a8ull,0x81c2c92e47edaee6ull,0x92722c851482353bull,
 0xa2bfe8a14cf10364ull,0xa81a664bbc423001ull,0xc24b8b70d0f89791ull,0xc76c51a30654be30ull,0xd192e819d6ef5218ull,
 0xd69906245565a910ull,0xf40e35855771202aull,0x106aa07032bbd1b8ull,0x19a4c116b8d2d0c8ull,0x1e376c085141ab53ull,
 0x2748774cdf8eeb99ull,0x34b0bcb5e19b48a8ull,0x391c0cb3c5c95a63ull,0x4ed8aa4ae3418acbull,0x5b9cca4f7763e373ull,
 0x682e6ff3d6b2b8a3ull,0x748f82ee5defb2fcull,0x78a5636f43172f60ull,0x84c87814a1f0ab72ull,0x8cc702081a6439ecull,
 0x90befffa23631e28ull,0xa4506cebde82bde9ull,0xbef9a3f7b2c67915ull,0xc67178f2e372532bull,0xca273eceea26619cull,
 0xd186b8c721c0c207ull,0xeada7dd6cde0eb1eull,0xf57d4f7fee6ed178ull,0x06f067aa72176fbaull,0x0a637dc5a2c898a6ull,
 0x113f9804bef90daeull,0x1b710b35131c471bull,0x28db77f523047d84ull,0x32caab7b40c72493ull,0x3c9ebe0a15c9bebcull,
 0x431d67c49c100d4cull,0x4cc5d4becb3e42b6ull,0x597f299cfc657e2aull,0x5fcb6fab3ad6faecull,0x6c44198c4a475817ull};

__device__ __forceinline__ u64 rotr64(u64 x, int n) { return (x >> n) | (x << (64 - n)); }
#define CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define BS0(x) (rotr64(x,28) ^ rotr64(x,34) ^ rotr64(x,39))
#define BS1(x) (rotr64(x,14) ^ rotr64(x,18) ^ rotr64(x,41))
#define SS0(x) (rotr64(x,1) ^ rotr64(x,8) ^ ((x) >> 7))
#define SS1(x) (rotr64(x,19) ^ rotr64(x,61) ^ ((x) >> 6))

__device__ __forceinline__ void iv(u64 st[8]) {
    st[0]=0x6a09e667f3bcc908ull; st[1]=0xbb67ae8584caa73bull; st[2]=0x3c6ef372fe94f82bull; st[3]=0xa54ff53a5f1d36f1ull;
    st[4]=0x510e527fade682d1ull; st[5]=0x9b05688c2b3e6c1full; st[6]=0x1f83d9abfb41bd6bull; st[7]=0x5be0cd19137e2179ull;
}

__device__ __forceinline__ void compress(u64 st[8], u64 w[16]) {
    u64 a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    #pragma unroll
    for (int i = 0; i < 80; ++i) {
        u64 wi;
        if (i < 16) wi = w[i];
        else {
            wi = SS1(w[(i-2)&15]) + w[(i-7)&15] + SS0(w[(i-15)&15]) + w[i&15];
            w[i&15] = wi;
        }
        u64 t1 = h + BS1(e) + CH(e,f,g) + K512[i] + wi;
        u64 t2 = BS0(a) + MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

// half-word i (big-endian 32-bit half of a 64-bit message word) of the padded leaf message
__device__ __forceinline__ u32 leaf_hw(const u32* m, long long wstride, int i, int nwords, int total, u64 bitlen) {
    if (i < nwords) return __byte_perm(m[(long long)i * wstride], 0, 0x0123);
    if (i == nwords) return 0x80000000u;
    if (i == total - 2) return (u32)(bitlen >> 32);
    if (i == total - 1) return (u32)bitlen;
    return 0u;
}

extern "C" __global__
void sha512_leaves(const u32* __restrict__ base, long long xstride, long long wstride, int nwords, int nx,
                   u64* __restrict__ out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const u32* m = base + (long long)x * xstride;
    u64 st[8];
    iv(st);
    u64 bitlen = (u64)nwords * 32ull;
    // half-words after padding: nwords + 1 (0x80) + zeros + 4 (128-bit length), rounded up to 32 per block
    int total = ((nwords + 5 + 31) / 32) * 32;
    u64 w[16];
    for (int blk = 0; blk < total; blk += 32) {
        if (blk + 32 <= nwords) {
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                u32 hi = __byte_perm(m[(long long)(blk + 2 * j) * wstride], 0, 0x0123);
                u32 lo = __byte_perm(m[(long long)(blk + 2 * j + 1) * wstride], 0, 0x0123);
                w[j] = ((u64)hi << 32) | lo;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                u32 hi = leaf_hw(m, wstride, blk + 2 * j, nwords, total, bitlen);
                u32 lo = leaf_hw(m, wstride, blk + 2 * j + 1, nwords, total, bitlen);
                w[j] = ((u64)hi << 32) | lo;
            }
        }
        compress(st, w);
    }
    u64* o = out + (long long)x * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = st[j];
}

extern "C" __global__
void sha512_nodes(const u64* __restrict__ lvl, int npairs, u64* __restrict__ out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= npairs) return;
    const u64* m = lvl + (long long)x * 16;
    u64 st[8];
    iv(st);
    u64 w[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) w[j] = m[j];
    compress(st, w);
    #pragma unroll
    for (int j = 0; j < 16; ++j) w[j] = 0ull;
    w[0] = 0x8000000000000000ull;
    w[15] = 1024ull;
    compress(st, w);
    u64* o = out + (long long)x * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = st[j];
}
"""

_mod = None


def _module():
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=SRC, options=("-std=c++17",))
    return _mod


def leaves(m, block: int = 128):
    """``m`` an ``(R, N)`` row-major cupy uint32 matrix -> ``(N, 8)`` uint64 leaf digests (column ``j``'s ``R`` values)."""
    R, N = m.shape
    out = cp.empty((N, 8), dtype=cp.uint64)
    _module().get_function("sha512_leaves")(((N + block - 1) // block,), (block,),
                                            (m, np.int64(1), np.int64(N), np.int32(R), np.int32(N), out))
    return out


def node_level(level, block: int = 128):
    """``(count, 8)`` uint64 -> ``((count + 1) // 2, 8)``: SHA-512(left || right), an unpaired last node promoted."""
    count = level.shape[0]
    npairs = count // 2
    out = cp.empty((npairs + (count & 1), 8), dtype=cp.uint64)
    if npairs:
        _module().get_function("sha512_nodes")(((npairs + block - 1) // block,), (block,),
                                               (cp.ascontiguousarray(level), np.int32(npairs), out))
    if count & 1:
        out[npairs] = level[count - 1]
    return out


def commit(m) -> list:
    """All tree levels, leaves first, root last (``(1, 8)``)."""
    lvl = leaves(m)
    levels = [lvl]
    while lvl.shape[0] > 1:
        lvl = node_level(lvl)
        levels.append(lvl)
    return levels


def digest_bytes(words) -> bytes:
    """``(8,)`` state words -> the standard 64-byte SHA-512 digest (big-endian words)."""
    a = np.asarray(words.get() if hasattr(words, "get") else words, dtype=np.uint64)
    return a.astype(">u8").tobytes()


# ---- host reference (hashlib), the definition the kernels are tested against ------------------------------------------

def leaf_host(column_u32: np.ndarray) -> bytes:
    return hashlib.sha512(np.ascontiguousarray(column_u32, dtype="<u4").tobytes()).digest()


def node_host(left: bytes, right: bytes) -> bytes:
    return hashlib.sha512(left + right).digest()

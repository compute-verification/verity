"""Blake3 on the GPU (cupy ``RawKernel``), bit-exact with the ``blake3`` reference for messages of whole u32 words.

Written from the Blake3 specification / ``reference_impl.rs`` (BLAKE3 team, CC0 / Apache-2.0): the compression
function ``g`` / ``round`` / ``MSG_PERMUTATION``, the chunk state (1 KiB chunks of 16 x 64-byte blocks, counter =
chunk index, ``CHUNK_START`` / ``CHUNK_END``), and the parent nodes (``PARENT``, counter 0, ``ROOT`` on the last).

Parallel layout: ``blake3_chunks`` gives one thread per (message ``x``, chunk ``y``) and writes the chunk's
chaining value; ``blake3_parents`` combines adjacent chaining values of each message level by level, promoting an
unpaired rightmost node unchanged -- which is exactly the spec's tree shape (left subtree = largest power of two
of chunks), so the final chaining value equals ``blake3(message)``.  A one-chunk message (<= 1 KiB, e.g. a
64-byte Merkle node ``left || right``) is finished in ``blake3_chunks`` with the ``ROOT`` flag.

Message words are little-endian ``u32`` (Blake3 is little-endian), so BabyBear columns need no byte swap and a
digest ``(8,)`` of words is the standard 32-byte digest read little-endian.  Level layout: ``(ny, nx, 8)``.
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:            # pragma: no cover
    cp = None

CHUNK_WORDS = 256

SRC = r"""
typedef unsigned int u32;
typedef unsigned long long u64;

#define CHUNK_START 1u
#define CHUNK_END   2u
#define PARENT      4u
#define ROOT        8u

__constant__ u32 IV[8] = {0x6A09E667u,0xBB67AE85u,0x3C6EF372u,0xA54FF53Au,0x510E527Fu,0x9B05688Cu,0x1F83D9ABu,0x5BE0CD19u};
__constant__ int MSG_PERM[16] = {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8};

__device__ __forceinline__ u32 rotr32(u32 x, int n) { return __funnelshift_r(x, x, n); }

__device__ __forceinline__ void g(u32* s, int a, int b, int c, int d, u32 mx, u32 my) {
    s[a] = s[a] + s[b] + mx; s[d] = rotr32(s[d] ^ s[a], 16);
    s[c] = s[c] + s[d];      s[b] = rotr32(s[b] ^ s[c], 12);
    s[a] = s[a] + s[b] + my; s[d] = rotr32(s[d] ^ s[a], 8);
    s[c] = s[c] + s[d];      s[b] = rotr32(s[b] ^ s[c], 7);
}

__device__ __forceinline__ void round_fn(u32* s, const u32* m) {
    g(s,0,4, 8,12,m[0], m[1]);  g(s,1,5, 9,13,m[2], m[3]);
    g(s,2,6,10,14,m[4], m[5]);  g(s,3,7,11,15,m[6], m[7]);
    g(s,0,5,10,15,m[8], m[9]);  g(s,1,6,11,12,m[10],m[11]);
    g(s,2,7, 8,13,m[12],m[13]); g(s,3,4, 9,14,m[14],m[15]);
}

// cv[8] in/out (new chaining value = s[i] ^ s[i+8]); block[16] is permuted in place.
__device__ __forceinline__ void compress(u32 cv[8], u32 block[16], u64 counter, u32 block_len, u32 flags) {
    u32 s[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = cv[i];
    s[8]=IV[0]; s[9]=IV[1]; s[10]=IV[2]; s[11]=IV[3];
    s[12]=(u32)counter; s[13]=(u32)(counter>>32); s[14]=block_len; s[15]=flags;
    u32 m[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = block[i];
    #pragma unroll
    for (int r = 0; r < 7; ++r) {
        round_fn(s, m);
        if (r < 6) {
            u32 p[16];
            #pragma unroll
            for (int i = 0; i < 16; ++i) p[i] = m[MSG_PERM[i]];
            #pragma unroll
            for (int i = 0; i < 16; ++i) m[i] = p[i];
        }
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = s[i] ^ s[i + 8];
}

// thread (x, y): chunk y of message x; message words at base + y*ystride + x*xstride + i*wstride, total_words per message.
extern "C" __global__
void blake3_chunks(const u32* __restrict__ base, long long xstride, long long ystride, long long wstride,
                   int total_words, int nx, int nchunks, u32* __restrict__ out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= nx || y >= nchunks) return;
    const u32* m = base + (long long)y * ystride + (long long)x * xstride;
    int words = total_words - y * 256; if (words > 256) words = 256;
    u32 cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = IV[i];
    int nblocks = (words + 15) / 16; if (nblocks == 0) nblocks = 1;
    u32 block[16];
    for (int b = 0; b < nblocks; ++b) {
        int have = words - b * 16; if (have > 16) have = 16; if (have < 0) have = 0;
        #pragma unroll
        for (int j = 0; j < 16; ++j) block[j] = (j < have) ? m[(long long)(b * 16 + j) * wstride] : 0u;
        u32 flags = 0;
        if (b == 0) flags |= CHUNK_START;
        if (b == nblocks - 1) { flags |= CHUNK_END; if (nchunks == 1) flags |= ROOT; }
        compress(cv, block, (u64)y, (u32)have * 4u, flags);
    }
    u32* o = out + ((long long)y * nx + x) * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = cv[i];
}

// level (count, nx, 8) -> ((count+1)/2, nx, 8): pair (2k, 2k+1) of message x; odd last node copied.
extern "C" __global__
void blake3_parents(const u32* __restrict__ in, int nx, int count, u32* __restrict__ out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int nout = (count + 1) / 2;
    if (idx >= (long long)nout * nx) return;
    int k = (int)(idx / nx), x = (int)(idx % nx);
    u32* o = out + ((long long)k * nx + x) * 8;
    const u32* l = in + ((long long)(2 * k) * nx + x) * 8;
    if (2 * k + 1 >= count) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = l[i];
        return;
    }
    const u32* r = in + ((long long)(2 * k + 1) * nx + x) * 8;
    u32 cv[8]; u32 block[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { cv[i] = IV[i]; block[i] = l[i]; block[8 + i] = r[i]; }
    u32 flags = PARENT; if (count == 2) flags |= ROOT;
    compress(cv, block, 0ull, 64u, flags);
    #pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = cv[i];
}
"""

_mod = None


def module():
    global _mod
    if _mod is None:
        _mod = cp.RawModule(code=SRC, options=("-std=c++17",))
    return _mod


def hash_strided(base, xstride: int, ystride_chunk: int, wstride: int, nwords: int, nx: int, block: int = 128,
                 workspace=None):
    """Blake3 of ``nx`` messages of ``nwords`` little-endian words each; message ``x`` word ``i`` sits at
    ``base[x * xstride + i * wstride]`` and chunk ``y`` starts ``y * ystride_chunk`` words later
    (``ystride_chunk = 256 * wstride``).  Returns ``(nx, 8)`` digests (uint32).  ``workspace`` may hold a
    preallocated ``(nchunks, nx, 8)`` buffer to avoid allocation in benchmarks."""
    base = cp.asarray(base)
    if base.dtype != cp.uint32:
        base = base.view(cp.uint32)
    nchunks = max(1, (nwords + CHUNK_WORDS - 1) // CHUNK_WORDS)
    mod = module()
    grid_x = (nx + block - 1) // block
    cvs = workspace if workspace is not None else cp.empty((nchunks, nx, 8), dtype=cp.uint32)
    mod.get_function("blake3_chunks")((grid_x, nchunks), (block,),
                                      (base, np.int64(xstride), np.int64(ystride_chunk), np.int64(wstride),
                                       np.int32(nwords), np.int32(nx), np.int32(nchunks), cvs))
    count = nchunks
    cur = cvs
    while count > 1:
        cur, count = parents(cur, nx, count, block)
    return cur.reshape(nx, 8)


def parents(level, nx: int, count: int, block: int = 256):
    """One level of Blake3 parent nodes: ``(count, nx, 8) -> ((count + 1) // 2, nx, 8)``; ``nx = 1`` for a Merkle
    level (the root gets the ``ROOT`` flag, i.e. node = the parent-node compression, not ``blake3(left || right)``)."""
    nout = (count + 1) // 2
    nxt = cp.empty((nout, nx, 8), dtype=cp.uint32)
    total = nout * nx
    module().get_function("blake3_parents")(((total + block - 1) // block,), (block,),
                                            (level, np.int32(nx), np.int32(count), nxt))
    return nxt, nout


def digests_to_bytes(d) -> list[bytes]:
    a = cp.asnumpy(d) if cp is not None and isinstance(d, cp.ndarray) else np.asarray(d)
    return [row.astype("<u4").tobytes() for row in a]

"""The prover's Merkle commitment on the GPU, specialised to this protocol (lane enc-hopper): leaf ``c`` = Blake3 of
column ``c`` of the codeword matrix ``U`` (``m`` little-endian uint32 words), node = Blake3(left || right) -- the same
bytes as ``merkle.verify_path`` / the ``blake3`` package / ``verity/ml/kernels/hash_gpu`` (which stays the reference
the tests compare against), in three launches instead of ~20 and without the local-memory round trips of the shared
kernel:

* ``leaves``: one CUDA block per 32 columns, one warp per Blake3 chunk (256 rows = 1 KiB of one column), so a warp's
  loads are 128 contiguous bytes of a row; the chunk chaining values meet in shared memory and the block finishes the
  spec's chunk tree (left subtree = largest power of two of chunks, i.e. pairwise with the unpaired rightmost node
  promoted) -- the leaf digests never round-trip through HBM.  The message permutation is a ``constexpr`` table, so
  the 7-round schedule is resolved at compile time (the shared kernel indexes a ``__constant__`` table and ptxas puts
  the 16-word block in local memory: 96 LDL + 24 STL per compression).
* ``tree``: level ``k+1`` from level ``k`` for all ``N`` (a power of two) leaves in two launches: blocks of 512 leaves
  reduce nine levels in shared memory (every level is written out -- the openings need the siblings), one block
  reduces the rest to the root.

``commit(U_int32) -> Tree`` (``levels[k]``: ``(N >> k, 8)`` uint32 cupy views, leaves first; ``root``), the object
``merkle.open_device`` / ``merkle.commit_device`` consume.  ``stream=``: launch on that torch stream (default: the
current one), so lane hp2-host's pipelining can overlap the commitment with the next sub-batch's encode.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["Tree", "available", "commit", "leaves", "tree_levels"]

_SRC = r"""
typedef unsigned int u32;
typedef unsigned long long u64;

#define CHUNK_START 1u
#define CHUNK_END   2u
#define PARENT      4u
#define ROOT        8u
#define ROWS_PER_CHUNK 256
#define COLS_PER_BLOCK 32

__device__ __forceinline__ u32 rotr32(u32 x, int n) { return __funnelshift_r(x, x, n); }

__device__ __forceinline__ void g(u32* s, int a, int b, int c, int d, u32 mx, u32 my) {
    s[a] = s[a] + s[b] + mx; s[d] = rotr32(s[d] ^ s[a], 16);
    s[c] = s[c] + s[d];      s[b] = rotr32(s[b] ^ s[c], 12);
    s[a] = s[a] + s[b] + my; s[d] = rotr32(s[d] ^ s[a], 8);
    s[c] = s[c] + s[d];      s[b] = rotr32(s[b] ^ s[c], 7);
}

/* the message schedule of round r: word i of round r is block[SCHED[r][i]] (MSG_PERMUTATION iterated r times);
   a constexpr table indexed by unrolled loop counters resolves to register names at compile time. */
__device__ __forceinline__ void compress(u32 cv[8], const u32 blk[16], u64 counter, u32 block_len, u32 flags) {
    constexpr int SCHED[7][16] = {
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
        {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
        {10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
        {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
        {9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
        {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13}};
    u32 s[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = cv[i];
    s[8] = 0x6A09E667u; s[9] = 0xBB67AE85u; s[10] = 0x3C6EF372u; s[11] = 0xA54FF53Au;
    s[12] = (u32)counter; s[13] = (u32)(counter >> 32); s[14] = block_len; s[15] = flags;
    #pragma unroll
    for (int r = 0; r < 7; ++r) {
        g(s,0,4, 8,12, blk[SCHED[r][0]],  blk[SCHED[r][1]]);
        g(s,1,5, 9,13, blk[SCHED[r][2]],  blk[SCHED[r][3]]);
        g(s,2,6,10,14, blk[SCHED[r][4]],  blk[SCHED[r][5]]);
        g(s,3,7,11,15, blk[SCHED[r][6]],  blk[SCHED[r][7]]);
        g(s,0,5,10,15, blk[SCHED[r][8]],  blk[SCHED[r][9]]);
        g(s,1,6,11,12, blk[SCHED[r][10]], blk[SCHED[r][11]]);
        g(s,2,7, 8,13, blk[SCHED[r][12]], blk[SCHED[r][13]]);
        g(s,3,4, 9,14, blk[SCHED[r][14]], blk[SCHED[r][15]]);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = s[i] ^ s[i + 8];
}

__device__ __forceinline__ void iv(u32 cv[8]) {
    cv[0] = 0x6A09E667u; cv[1] = 0xBB67AE85u; cv[2] = 0x3C6EF372u; cv[3] = 0xA54FF53Au;
    cv[4] = 0x510E527Fu; cv[5] = 0x9B05688Cu; cv[6] = 0x1F83D9ABu; cv[7] = 0x5BE0CD19u;
}

/* blake3(left || right): one chunk, one 64-byte block. */
__device__ __forceinline__ void node_hash(const u32* l, const u32* r, u32 out[8]) {
    u32 blk[16], cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { blk[i] = l[i]; blk[8 + i] = r[i]; }
    iv(cv);
    compress(cv, blk, 0ull, 64u, CHUNK_START | CHUNK_END | ROOT);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = cv[i];
}

/* leaves: block b, warp w = chunk w of columns [32 b, 32 b + 32); lane = column.  rows x N matrix (row-major u32). */
extern "C" __global__ void __launch_bounds__(512)
blake3_leaves(const u32* __restrict__ U, int rows, int N, int nchunks, u32* __restrict__ out)
{
    __shared__ u32 cvs[16][8][COLS_PER_BLOCK];     /* [chunk][word][lane]: nchunks <= 16 (rows <= 4096) */
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int col = blockIdx.x * COLS_PER_BLOCK + lane;
    if (w < nchunks && col < N) {
        const int row0 = w * ROWS_PER_CHUNK;
        int words = rows - row0; if (words > ROWS_PER_CHUNK) words = ROWS_PER_CHUNK;
        const int nblocks = (words + 15) >> 4;
        const u32* p = U + (size_t)row0 * N + col;
        u32 cv[8]; iv(cv);
        u32 blk[16];
#if PREFETCH
        #pragma unroll
        for (int j = 0; j < 16; ++j) blk[j] = (j < words) ? __ldg(p + (size_t)j * N) : 0u;
#endif
        for (int b = 0; b < nblocks; ++b) {
            const int have = min(words - b * 16, 16);
#if PREFETCH
            u32 nxt[16];
            if (b + 1 < nblocks) {                      /* prefetch the next block's rows while this one compresses */
                const int have2 = min(words - (b + 1) * 16, 16);
                const u32* q = p + (size_t)(b + 1) * 16 * N;
                #pragma unroll
                for (int j = 0; j < 16; ++j) nxt[j] = (j < have2) ? __ldg(q + (size_t)j * N) : 0u;
            }
#else
            {
                const u32* q = p + (size_t)b * 16 * N;
                #pragma unroll
                for (int j = 0; j < 16; ++j) blk[j] = (j < have) ? __ldg(q + (size_t)j * N) : 0u;
            }
#endif
            u32 flags = 0;
            if (b == 0) flags |= CHUNK_START;
            if (b == nblocks - 1) { flags |= CHUNK_END; if (nchunks == 1) flags |= ROOT; }
            compress(cv, blk, (u64)w, (u32)have * 4u, flags);
#if PREFETCH
            if (b + 1 < nblocks) {
                #pragma unroll
                for (int j = 0; j < 16; ++j) blk[j] = nxt[j];
            }
#endif
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) cvs[w][i][lane] = cv[i];
    }
    if (nchunks > 1) {
        /* the spec's chunk tree: pairwise, unpaired rightmost promoted; parent flags, ROOT on the last */
        __syncthreads();
        int count = nchunks;
        while (count > 1) {                                 /* barriers block-uniform: lanes with col >= N idle in step */
            const int npairs = count >> 1, nout = (count + 1) >> 1;
            const bool pair = (w < npairs) && (col < N);
            u32 cv[8];
            if (pair) {
                u32 blk[16];
                #pragma unroll
                for (int i = 0; i < 8; ++i) { blk[i] = cvs[2 * w][i][lane]; blk[8 + i] = cvs[2 * w + 1][i][lane]; }
                iv(cv);
                compress(cv, blk, 0ull, 64u, (count == 2) ? (PARENT | ROOT) : PARENT);
            }
            __syncthreads();                                /* everyone has read its pair */
            if (pair) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) cvs[w][i][lane] = cv[i];
            } else if (w == npairs && (count & 1) && col < N) {   /* promote the odd node */
                #pragma unroll
                for (int i = 0; i < 8; ++i) cvs[w][i][lane] = cvs[count - 1][i][lane];
            }
            __syncthreads();
            count = nout;
        }
    }
    if (w == 0 && col < N) {
        u32* o = out + (size_t)col * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = cvs[0][i][lane];
    }
}

/* tree_block: block b reduces leaves [512 b, 512 b + 512) of level `in` (count nodes) for LEVELS levels, writing every
   level: out_k at outs[k] (k = 1..LEVELS), each of count >> k nodes.  N a power of two, count >= 512 * 2^0. */
#define TREE_LEAVES 512
#define TREE_LEVELS 9
extern "C" __global__ void __launch_bounds__(256)
blake3_tree(const u32* __restrict__ in, int count, u32* __restrict__ out, int levels)
{
    __shared__ u32 buf[TREE_LEAVES][8];
    const int t = threadIdx.x;
    const int base = blockIdx.x * TREE_LEAVES;
    if (base >= count) return;
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int i = t + k * 256;
        if (base + i < count) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) buf[i][j] = in[(size_t)(base + i) * 8 + j];
        }
    }
    __syncthreads();
    int n = min(TREE_LEAVES, count - base);          /* nodes of this block at the current level */
    size_t off = 0;                                 /* output offset of level k+1 within `out` */
    int lvl_count = count;
    for (int k = 0; k < levels && n > 1; ++k) {
        const int npairs = n >> 1;
        u32 h[8];
        if (t < npairs) node_hash(buf[2 * t], buf[2 * t + 1], h);
        __syncthreads();
        const int nout_level = lvl_count >> 1;
        if (t < npairs) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) buf[t][j] = h[j];
            u32* o = out + (off + (size_t)(base >> (k + 1)) + t) * 8;
            #pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = h[j];
        }
        __syncthreads();
        off += nout_level;
        lvl_count = nout_level;
        n = npairs;
    }
}
"""

_MOD: dict = {}
PREFETCH = 0        # leaf kernel: double-buffer the next 16 rows in registers while compressing (A/B switch; measured
                    # on the H100: 0.670 ms without vs 0.694 ms with -- the extra 5 registers cost more than the latency they hide)


def _module(device_index: int):
    import cupy as cp
    key = (device_index, PREFETCH)
    mod = _MOD.get(key)
    if mod is None:
        with cp.cuda.Device(device_index):
            mod = cp.RawModule(code=_SRC, options=("-std=c++17", f"-DPREFETCH={int(PREFETCH)}"))
            mod.compile()
        _MOD[key] = mod
    return mod


def available() -> bool:
    try:
        import cupy  # noqa: F401
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class Tree:
    """``levels[k]``: ``(N >> k, 8)`` uint32 cupy views (leaves first, root last); ``root``: ``(8,)``."""

    def __init__(self, rows: int, cols: int, levels: list):
        self.hash, self.rows, self.cols, self.levels = "blake3", rows, cols, levels

    @property
    def root(self):
        return self.levels[-1][0]

    @property
    def depth(self) -> int:
        return len(self.levels) - 1


def _stream(device, stream):
    import cupy as cp
    import warnings
    ts = stream if stream is not None else torch.cuda.current_stream(device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return cp.cuda.ExternalStream(ts.cuda_stream)


def leaves(U: torch.Tensor, out=None, stream=None):
    """``U``: (rows, N) contiguous int32/uint32 torch tensor on the device -> ``(N, 8)`` uint32 cupy leaf digests."""
    import cupy as cp
    if U.ndim != 2 or not U.is_contiguous() or U.element_size() != 4:
        raise ValueError("U must be a contiguous (rows, N) 32-bit tensor")
    rows, N = U.shape
    nchunks = max(1, (rows + 255) // 256)
    if nchunks > 16:
        raise ValueError("leaf kernel handles rows <= 4096")
    dev = U.device.index or 0
    mod = _module(dev)
    with cp.cuda.Device(dev):
        if out is None:
            out = cp.empty((N, 8), dtype=cp.uint32)
        u = cp.asarray(U).view(cp.uint32)
        with _stream(U.device, stream):
            mod.get_function("blake3_leaves")(((N + 31) // 32,), (32 * nchunks,),
                                              (u, np.int32(rows), np.int32(N), np.int32(nchunks), out))
    return out


def tree_levels(lv, stream=None, device_index: int = 0) -> list:
    """All levels above the ``(N, 8)`` leaves (``N`` a power of two >= 2): ``[(N/2, 8), ..., (1, 8)]``."""
    import cupy as cp
    N = int(lv.shape[0])
    if N & (N - 1) or N < 2:
        raise ValueError("N must be a power of two >= 2")
    depth = N.bit_length() - 1
    mod = _module(device_index)
    with cp.cuda.Device(device_index):
        flat = cp.empty((N - 1, 8), dtype=cp.uint32)          # all upper levels, level 1 first
        views, off = [], 0
        for k in range(1, depth + 1):
            cnt = N >> k
            views.append(flat[off:off + cnt])
            off += cnt
        with _stream(torch.device("cuda", device_index), stream):
            fn = mod.get_function("blake3_tree")
            cur, count, done = lv, N, 0
            while count > 1:
                lv_here = min(9, depth - done)
                fn(((count + 511) // 512,), (256,), (cur, np.int32(count), views[done], np.int32(lv_here)))
                done += lv_here
                count >>= lv_here
                cur = views[done - 1]
    return views


def commit(U: torch.Tensor, stream=None) -> Tree:
    """Merkle tree over the columns of ``U`` ((rows, N) contiguous int32 on the device, N a power of two)."""
    lv = leaves(U, stream=stream)
    N = int(U.shape[1])
    levels = [lv] + (tree_levels(lv, stream=stream, device_index=U.device.index or 0) if N > 1 else [])
    return Tree(int(U.shape[0]), N, levels)

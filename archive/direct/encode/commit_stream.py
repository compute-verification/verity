"""Fused encode -> column-hash streaming commitment: the Ligero codeword is never materialised in HBM.

The commitment of the ``(R, k)`` witness matrix is the Merkle root over the ``n = 16384`` column digests of its
``(R, n)`` Reed-Solomon codeword (``PROTOCOL.md`` section 3; leaf = H(column), column-major bytes).  The unfused path
(b-encode + hash-gpu lanes) writes the 20.4 GB codeword and reads it back: 32.2 + 45.7 = 78 ms per B batch on the
RTX 4090, both kernels well under the card's SIMT rate because they are memory-streaming.  Here the batch is cut into
row slices small enough for the slice codeword to stay in the 72 MB L2 (default 512 rows = 32 MiB) and every slice
is fed straight into ``n`` running column-hash states:

* SHA-256: one 64-byte block = 16 rows of a column; ``sha256_cols_update`` (one thread per column) folds ``rows / 16``
  blocks of the slice into ``state[column]`` (8 words), reading coset 0 from the *input* rows (the code is systematic)
  and cosets 1-3 from the slice buffer; ``sha256_cols_final`` appends the < 16 tail rows and the FIPS padding.  The
  digests equal ``hash_gpu.merkle.leaves(codeword, "sha256")`` bit for bit (tests).
* Blake3: a 1 KiB chunk = 256 rows; ``blake3_cols_chunks`` computes the chunk chaining values of every chunk of the
  slice with the *global* chunk counter, into a ``(nchunks, n, 8)`` table (636 MB at the B shape); the spec tree is
  built by ``hash_gpu.blake3_cuda.parents`` at the end.  Equals ``hash_gpu.merkle.leaves(codeword, "blake3")``.

Two streams overlap the encoder (many CTAs, SIMT + shared memory) with the hash update (16,384 threads, ALU-bound):
double-buffered slices, events for the hand-over; the whole slice loop is captured once into a CUDA graph so the
Python launch overhead does not gate the ~100 us per slice.  Options: ``keep_codeword=True`` additionally streams
the three non-systematic cosets to a resident ``(R, 3k)`` store (15.3 GB at the B shape: the quadratic test and the
column openings then read it instead of re-encoding) -- a pure write that the hash-bound pipeline absorbs.

Digest conventions (for the Rust/CUDA port): column ``x`` of the *coset-major* codeword (``babybear`` module: column
``c k + j`` is the evaluation at ``omega^(4 j + c)``; columns ``< k`` are the message values); leaf ``x`` = H(the
``R`` elements of column ``x``, each as 4 little-endian bytes, row 0 first); SHA-256 digests are stored as the 8
big-endian words ``H0..H7`` (``sha256_cuda.digests_to_bytes``), Blake3 digests as 8 little-endian words; the tree is
``hash_gpu.merkle`` (SHA-256(left || right) / Blake3 parent nodes; unpaired node promoted), leaves in column order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "ml" / "kernels"))

from hash_gpu import merkle, sha256_cuda

from . import babybear as bb
from .encode_ntt_simt import RSEncoderSIMT

_SHA_SRC = sha256_cuda.SRC.split("extern \"C\" __global__")[0] + r"""
/* column x, rows [0, nrows) of a slice whose systematic part (columns < k) is in sys[row * sys_stride + x] and whose
   non-systematic part (columns >= k) is in cos[row * cos_stride + (x - k)].  nrows is a multiple of 16: whole blocks. */
__device__ __forceinline__ u32 col_word(const u32* __restrict__ sys, long long sys_stride, const u32* __restrict__ cos,
                                        long long cos_stride, int k, int x, int row) {
    u32 v = (x < k) ? sys[(long long)row * sys_stride + x] : cos[(long long)row * cos_stride + (x - k)];
    return __byte_perm(v, 0, 0x0123);
}

extern "C" __global__
void sha256_cols_update(const u32* __restrict__ sys, long long sys_stride, const u32* __restrict__ cos, long long cos_stride,
                        int k, int ncols, int nrows, u32* __restrict__ state)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= ncols) return;
    u32 st[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) st[j] = state[(long long)x * 8 + j];
    u32 w[16];
    for (int r0 = 0; r0 < nrows; r0 += 16) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) w[j] = col_word(sys, sys_stride, cos, cos_stride, k, x, r0 + j);
        sha256_compress(st, w);
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) state[(long long)x * 8 + j] = st[j];
}

/* tail: the last (total_rows mod 16) rows, then 0x80, zeros and the 64-bit bit length; out[x * 8 + j] = H_j. */
extern "C" __global__
void sha256_cols_final(const u32* __restrict__ sys, long long sys_stride, const u32* __restrict__ cos, long long cos_stride,
                       int k, int ncols, int tail_rows, long long total_rows, const u32* __restrict__ state, u32* __restrict__ out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= ncols) return;
    u32 st[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) st[j] = state[(long long)x * 8 + j];
    u64 bitlen = (u64)total_rows * 32ull;
    int total = ((tail_rows + 3 + 15) / 16) * 16;          /* 16 or 32 words */
    u32 w[16];
    for (int blk = 0; blk < total; blk += 16) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            int i = blk + j;
            u32 v;
            if (i < tail_rows) v = col_word(sys, sys_stride, cos, cos_stride, k, x, i);
            else if (i == tail_rows) v = 0x80000000u;
            else if (i == total - 2) v = (u32)(bitlen >> 32);
            else if (i == total - 1) v = (u32)bitlen;
            else v = 0u;
            w[j] = v;
        }
        sha256_compress(st, w);
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) out[(long long)x * 8 + j] = st[j];
}
"""

def _blake3_src() -> str:
    """Blake3 compression with the 7-round message schedule unrolled to literal indices (the reference's
    ``m[MSG_PERM[i]]`` permutation loop indexes a register array with a constant-memory value and spills to local
    memory: hash_gpu's Blake3 leaf ran at 359 GB/s, below SHA-256, for that reason)."""
    perm = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
    sched = [list(range(16))]
    for _ in range(6):
        sched.append([sched[-1][perm[i]] for i in range(16)])
    rounds = []
    for sc in sched:
        rounds.append("\n".join(
            f"    G(s,{a},{b},{c},{d},m[{sc[2 * i]}],m[{sc[2 * i + 1]}]);"
            for i, (a, b, c, d) in enumerate([(0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),
                                              (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)])))
    body = "\n".join(rounds)
    return r"""
typedef unsigned int u32;
typedef unsigned long long u64;
#define CHUNK_START 1u
#define CHUNK_END   2u
#define PARENT      4u
#define ROOT        8u
#define IV0 0x6A09E667u
#define IV1 0xBB67AE85u
#define IV2 0x3C6EF372u
#define IV3 0xA54FF53Au
#define IV4 0x510E527Fu
#define IV5 0x9B05688Cu
#define IV6 0x1F83D9ABu
#define IV7 0x5BE0CD19u
__device__ __forceinline__ u32 rotr32(u32 x, int n) { return __funnelshift_r(x, x, n); }
#define G(s,a,b,c,d,mx,my) \
    s[a] = s[a] + s[b] + (mx); s[d] = rotr32(s[d] ^ s[a], 16); s[c] = s[c] + s[d]; s[b] = rotr32(s[b] ^ s[c], 12); \
    s[a] = s[a] + s[b] + (my); s[d] = rotr32(s[d] ^ s[a], 8);  s[c] = s[c] + s[d]; s[b] = rotr32(s[b] ^ s[c], 7);

__device__ __forceinline__ void compress(u32 cv[8], const u32 m[16], u64 counter, u32 block_len, u32 flags) {
    u32 s[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = cv[i];
    s[8]=IV0; s[9]=IV1; s[10]=IV2; s[11]=IV3;
    s[12]=(u32)counter; s[13]=(u32)(counter>>32); s[14]=block_len; s[15]=flags;
""" + body + r"""
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = s[i] ^ s[i + 8];
}

/* chunk (chunk0 + y) of column x: rows [256 y, 256 y + 256) of the slice; total_words = the whole column's length, so
   the last chunk may be short (words = total_words - 256 (chunk0 + y), clipped to 256). */
extern "C" __global__
void blake3_cols_chunks(const u32* __restrict__ sys, long long sys_stride, const u32* __restrict__ cos, long long cos_stride,
                        int k, int ncols, int chunk0, int nchunks_total, long long total_words, u32* __restrict__ cvs)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= ncols) return;
    int chunk = chunk0 + y;
    long long words_ll = total_words - (long long)chunk * 256; int words = words_ll > 256 ? 256 : (int)words_ll;
    u32 cv[8] = {IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7};
    int nblocks = (words + 15) / 16; if (nblocks == 0) nblocks = 1;
    u32 block[16];
    const u32* src = (x < k) ? sys + x : cos + (x - k);
    long long stride = (x < k) ? sys_stride : cos_stride;
    for (int b = 0; b < nblocks; ++b) {
        int have = words - b * 16; if (have > 16) have = 16; if (have < 0) have = 0;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            int row = y * 256 + b * 16 + j;
            block[j] = (j < have) ? src[(long long)row * stride] : 0u;
        }
        u32 flags = 0;
        if (b == 0) flags |= CHUNK_START;
        if (b == nblocks - 1) { flags |= CHUNK_END; if (nchunks_total == 1) flags |= ROOT; }
        compress(cv, block, (u64)chunk, (u32)have * 4u, flags);
    }
    u32* o = cvs + ((long long)chunk * ncols + x) * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = cv[i];
}

/* one level of the spec's chunk tree for every column: (count, ncols, 8) -> ((count + 1) / 2, ncols, 8); the unpaired
   last node is promoted; ROOT on the final parent. */
extern "C" __global__
void blake3_cols_parents(const u32* __restrict__ in, int ncols, int count, u32* __restrict__ out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int nout = (count + 1) / 2;
    if (idx >= (long long)nout * ncols) return;
    int kk = (int)(idx / ncols), x = (int)(idx % ncols);
    u32* o = out + ((long long)kk * ncols + x) * 8;
    const u32* l = in + ((long long)(2 * kk) * ncols + x) * 8;
    if (2 * kk + 1 >= count) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = l[i];
        return;
    }
    const u32* r = in + ((long long)(2 * kk + 1) * ncols + x) * 8;
    u32 cv[8] = {IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7};
    u32 block[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { block[i] = l[i]; block[8 + i] = r[i]; }
    compress(cv, block, 0ull, 64u, count == 2 ? (PARENT | ROOT) : PARENT);
    #pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = cv[i];
}
"""


SHA_INIT = np.array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19], dtype=np.uint32)


class StreamingCommitter:
    """``commit(rows) -> (leaves (n, 8), root (8,))`` with the codeword streamed through L2 slice by slice.

    ``hash`` in ``("sha256", "blake3")``; ``slice_rows`` a multiple of 16 (SHA-256) / 256 (Blake3); ``overlap`` runs the
    encoder and the hash on two streams with double-buffered slices; ``graph`` captures the slice loop into a CUDA graph
    (per row count, cached); ``keep_codeword`` also writes cosets 1-3 to ``self.store`` ``(R, 3k)``."""

    def __init__(self, hash: str = "sha256", slice_rows: int = 512, overlap: bool = True, graph: bool = True,
                 keep_codeword: bool = False, encoder: RSEncoderSIMT | None = None, hash_block: int = 128):
        import cupy as cp

        self.cp = cp
        if hash not in ("sha256", "blake3"):
            raise ValueError(hash)
        align = 16 if hash == "sha256" else 256
        if slice_rows % align:
            raise ValueError(f"slice_rows must be a multiple of {align} for {hash}")
        self.hash, self.slice_rows, self.overlap, self.graph, self.keep = hash, slice_rows, overlap, graph, keep_codeword
        if encoder is None:
            sms = cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
            # 64 registers, one 32-warp CTA per SM; radix-4 shared-memory stages (b-v2: same arithmetic, 1.07x on the fused commitment)
            encoder = RSEncoderSIMT(threads=1024, min_blocks_per_sm=2, blocks=sms, radix4=True)
        self.enc = encoder
        self.k, self.n = self.enc.k, self.enc.n
        self.hash_block = hash_block
        if hash == "sha256":
            self.mod = cp.RawModule(code=_SHA_SRC, options=("-std=c++17",))
            self.k_update = self.mod.get_function("sha256_cols_update")
            self.k_final = self.mod.get_function("sha256_cols_final")
        else:
            self.mod = cp.RawModule(code=_blake3_src(), options=("-std=c++17",))
            self.k_chunks = self.mod.get_function("blake3_cols_chunks")
            self.k_parents = self.mod.get_function("blake3_cols_parents")
        self.s_enc = cp.cuda.Stream(non_blocking=True)
        self.s_hash = cp.cuda.Stream(non_blocking=True) if overlap else self.s_enc
        self.bufs = [cp.empty((slice_rows, 3 * self.k), dtype=cp.uint32) for _ in range(2 if overlap else 1)]
        self.ev_encoded = [cp.cuda.Event(disable_timing=True) for _ in self.bufs]
        self.ev_hashed = [cp.cuda.Event(disable_timing=True) for _ in self.bufs]
        self.state = cp.empty((self.n, 8), dtype=cp.uint32)
        self.sha_init = cp.asarray(np.tile(SHA_INIT, (self.n, 1)))
        self.leaves = cp.empty((self.n, 8), dtype=cp.uint32)
        self.store = None
        self.cvs = None
        self._graphs = {}
        self.skip_encode = False     # diagnostics only (bench): hash whatever the slice buffers hold
        self.skip_hash = False       # diagnostics only: encode into the L2 slice buffers, no hashing

    # -- the slice loop (what gets captured) -------------------------------------------------------------------------
    def _slices(self, R: int):
        r0 = 0
        while r0 < R:
            r = min(self.slice_rows, R - r0)
            yield r0, r
            r0 += r

    def _loop(self, rows, R: int):
        cp = self.cp
        k, n = self.k, self.n
        grid = ((n + self.hash_block - 1) // self.hash_block,)
        blk = (self.hash_block,)
        if self.hash == "sha256":
            with self.s_enc:
                cp.copyto(self.state, self.sha_init)
        else:
            nchunks = -(-R // 256)
        recorded = set()
        for i, (r0, r) in enumerate(self._slices(R)):
            b = i % len(self.bufs)
            buf = self.bufs[b]
            with self.s_enc:
                if i >= len(self.bufs):
                    self.s_enc.wait_event(self.ev_hashed[b])
                src = self.store[r0:r0 + r] if self.keep else buf
                if not self.skip_encode:
                    self.enc(rows[r0:r0 + r], src[:r], write_systematic=False)
                self.ev_encoded[b].record(self.s_enc)
            with self.s_hash:
                self.s_hash.wait_event(self.ev_encoded[b])
                sys_ptr, cos_ptr = rows[r0:r0 + r], src
                if self.skip_hash:
                    pass
                elif self.hash == "sha256":
                    full = (r // 16) * 16
                    if full:
                        self.k_update(grid, blk, (sys_ptr, np.int64(k), cos_ptr, np.int64(3 * k), np.int32(k), np.int32(n),
                                                  np.int32(full), self.state))
                    if r0 + r == R:
                        tail = r - full
                        self.k_final(grid, blk, (sys_ptr[full:], np.int64(k), cos_ptr[full:], np.int64(3 * k), np.int32(k), np.int32(n),
                                                 np.int32(tail), np.int64(R), self.state, self.leaves))
                else:
                    c0 = r0 // 256
                    nch = -(-r // 256)
                    self.k_chunks((grid[0], nch), blk, (sys_ptr, np.int64(k), cos_ptr, np.int64(3 * k), np.int32(k), np.int32(n),
                                                        np.int32(c0), np.int32(nchunks), np.int64(R), self.cvs))
                self.ev_hashed[b].record(self.s_hash)
                recorded.add(b)
        if self.overlap:                       # join the hash stream back into the origin stream
            for b in recorded:
                self.s_enc.wait_event(self.ev_hashed[b])

    def _run_loop(self, rows, R: int):
        if not self.graph:
            self._loop(rows, R)
            return
        key = (R, rows.data.ptr, None if self.store is None else self.store.data.ptr)
        g = self._graphs.get(key)
        if g is None:
            self._graphs.clear()
            self._loop(rows, R)                # eager warm-up: compiles / loads every kernel before the capture
            self.s_enc.synchronize()
            self.s_enc.begin_capture()
            self._loop(rows, R)
            g = self.s_enc.end_capture()
            self._graphs[key] = g
        g.launch(stream=self.s_enc)

    # -- public --------------------------------------------------------------------------------------------------------
    def commit(self, rows, stream=None):
        """``rows``: ``(R, k)`` uint32 device array (values < p).  Returns ``(leaves, root)``; ``self.store`` holds the
        cosets when ``keep_codeword``.  All work is enqueued on ``self.s_enc`` (joined with ``s_hash``) and the caller's
        stream waits for it."""
        cp = self.cp
        rows = cp.ascontiguousarray(rows).view(cp.uint32)
        R = rows.shape[0]
        if rows.shape != (R, self.k):
            raise ValueError(rows.shape)
        cur = stream or cp.cuda.get_current_stream()
        ev = cp.cuda.Event(disable_timing=True)
        ev.record(cur)
        self.s_enc.wait_event(ev)
        if self.keep and (self.store is None or self.store.shape[0] < R):
            self.store = cp.empty((R, 3 * self.k), dtype=cp.uint32)
        if self.hash == "blake3":
            nchunks = -(-R // 256)
            if self.cvs is None or self.cvs.shape[0] != nchunks:
                self.cvs = cp.empty((nchunks, self.n, 8), dtype=cp.uint32)
        self._run_loop(rows, R)
        with self.s_enc:
            if self.hash == "blake3":
                cur_lvl, count = self.cvs, self.cvs.shape[0]
                while count > 1:
                    nout = (count + 1) // 2
                    nxt = cp.empty((nout, self.n, 8), dtype=cp.uint32)
                    total = nout * self.n
                    self.k_parents(((total + 255) // 256,), (256,), (cur_lvl, np.int32(self.n), np.int32(count), nxt))
                    cur_lvl, count = nxt, nout
                self.leaves[:] = cur_lvl.reshape(self.n, 8)
            lvl = self.leaves
            levels = [lvl]
            while lvl.shape[0] > 1:
                lvl = merkle.node_level(lvl, self.hash)
                levels.append(lvl)
        ev2 = cp.cuda.Event(disable_timing=True)
        ev2.record(self.s_enc)
        cur.wait_event(ev2)
        self.levels = levels
        return self.leaves, levels[-1][0]


def reference_leaves(rows_np: np.ndarray, hash: str) -> np.ndarray:
    """CPU reference: encode with the numpy encoder, hash every column with hashlib / blake3 (8 words each)."""
    import hashlib

    code = bb.encode_rows(rows_np, bb.N_CODE, layout="coset").astype(np.uint32)
    out = np.empty((code.shape[1], 8), dtype=np.uint32)
    for x in range(code.shape[1]):
        msg = np.ascontiguousarray(code[:, x]).astype("<u4").tobytes()
        if hash == "sha256":
            out[x] = np.frombuffer(hashlib.sha256(msg).digest(), dtype=">u4")
        else:
            import blake3

            out[x] = np.frombuffer(blake3.blake3(msg).digest(), dtype="<u4")
    return out

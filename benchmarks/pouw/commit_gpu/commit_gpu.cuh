// The GPU committer's device code and its host side (constants, Committer), included by commit_gpu.cu and by any
// kernel that commits rows it produces (an epilogue calls hash_after_block or its own row digest, then
// leaf_from_digest, after Committer::salts; Committer::nodes folds the tree). One translation unit: C is __constant__.
//
// A GPU committer for hm96-sha512/row/v2 and hm96-sha512/row-seg/v1 rows in frame-v3-sha512 trees, bit for bit the
// reference's (verity.ml.commitments: hm96.SHA512, rowleaf, merkle.CommitmentDomain and tree_levels; reference.py here).
// The frame's leaf, pad and node, the hm96 leaf's mask, salt digest and value, and the constants' builders are the GPU's one
// row commitment (`rc`, verity/ml/kernels/row_commit.cuh); this file keeps the committer's schedule: its rows' digests, its
// salts, its kernels and streams, and its constants in __constant__ memory.
//
// Per position of a domain of n leaves (rank = position, the tree padded to width = 2^depth):
//   x    = SHA-512(P(l) || row)                         the row's sha512/row/v2 digest
//   y    = salt (192 bytes): given, or y_r = SHA-512(Ks || u64be(pos) || u64be(r)), r = 0, 1, 2 (reference.prf_salt)
//   c    = SHA-512(salt_prefix || y);  b = x ^ M_key y;  h = SHA-512(leaf_prefix(key) || b || c)
//   leaf = SHA-512(frame || "leaf" || domain id || uint(pos) || uint(pos) || "hm96-sha512/row/v2" || h)   (framed parts)
//   pad  = SHA-512(frame || "pad" || domain id || uint(pos)) for n <= pos < width
// then node(level, index, left, right) up the tree. Every 64-byte digest is kept as 8 u64 words, each the big-endian
// value of its 8 bytes (the host swaps them back).
//
// Two phases: the salt phase (y, M_key y and c; salt_kernel) reads no data and can run before the operand exists; the data
// phase (x, b, h, the leaf, the tree; leaf_kernel and the node kernels) is all that sits on the critical path. The tree's
// narrow top can run on the CPU (Committer::host_tail), where a compression is far faster than on one GPU thread.
//
// M_key y uses M's Hankel form: column j of M is bits j .. j+511 of the key, so the reference's byte table (p, v) is bytes
// [p, p+64) of T_v = xor of (key >> s) over the set bits s of v: one 64 KB table T (reference.t_table), in shared memory.
//
// Row sources: `e2e` (the e2e slice's leaf: 32 int8 of an operand as bf16 words twice, l = 1024, leaf pos at bytes
// [32 pos, 32 pos + 32)), `rows` (n rows of l bits at a byte stride; an int8 operand's own memory is rows of 8 L bits) and
// `seg` (the same rows under sha512/row-seg/v1 and schema hm96-sha512/row-seg/v1: 1 KiB segments hashed a thread each, then
// the outer hash over their digests, so a long row's hash is not one serial chain; P is then Q(l)). `tile` and `tileseg`
// (Decision 46) commit an M x K int8 operand in place as tile leaves: leaf pos is block (pos % ceil(M/R), pos / ceil(M/R))
// of R rows x 32 values, its row the block's 32 R bytes in row-major order (rows past M zero), as sha512/row/v2 or
// sha512/row-seg/v1 rows: the reference's commitment of the permuted operand (reference.permute).
//
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../../verity/ml/kernels/row_commit.cuh"

using u64 = uint64_t;
using u32 = uint32_t;
#ifndef DEV
#define DEV __device__ __forceinline__
#endif
#ifndef HD
#define HD __host__ __device__ __forceinline__
#endif

#ifndef CK
#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); \
      exit(3);                                                                                  \
    }                                                                                           \
  } while (0)
#endif

#ifndef LEAF_THREADS
#define LEAF_THREADS 256
#endif
constexpr int TSTRIDE = 272;                 // bytes a T row in shared memory: 256 + 16, so rows start in different bank groups
constexpr int TSMEM = 256 * TSTRIDE;          // 69,632 bytes
constexpr int NODE_THREADS = 256;             // a node pass covers 512 children a block, up to 9 levels
constexpr int SMALL_THREADS = 128;            // the small path's CTAs (no shared memory): a CTA folds its 128 leaves' 7 levels
constexpr int SMALL_LEVELS = 7;
constexpr int SMALL_TOP = 256;                // the small path's top CTA takes a level of at most 256 nodes
#ifndef SMALL_MINB
#define SMALL_MINB 5                   // the small path's CTAs an SM by registers (5: <= 102 a thread), so they fit beside a GEMM's
#endif

struct Consts {
  u64 mid_P[8];             // after P(l)
  u64 kw_xpad[80];          // the row digest's last block when the row fills whole blocks (all padding)
  rc::Hm96K hm;             // the hm96 leaf's: after the salt prefix, after leaf_prefix(key), and its padding block
  u64 ks[8];                // the salt key
  u64 ks_m8[8];             // state after rounds 0..7 of a salt block (its words 0..7 are the key)
  u64 seg_mid[64][8];       // sha512/row-seg/v1: the chaining value after segment j's first block S(j, s, m_j)
};
// What the domain id fixes: the frame's leaf, pad and node messages. COMMIT_GPU_DOMS slots, so committers of different
// domains (an operand's leaf counts) can share one binary while they agree on Consts.
struct Dom : rc::FrameK, rc::NodeK {};
#ifndef COMMIT_GPU_DOMS
#define COMMIT_GPU_DOMS 1
#endif
__constant__ Consts C;
__constant__ Dom CD[COMMIT_GPU_DOMS];

using rc::cpy8;
using rc::ldg16;
using rc::nbytes;
using rc::or_be;
using rc::or_digest;
using sx::be64;   // 8 bytes loaded as two LE u32
using sx::bswap32;

// --- the frame: row_commit's leaf, pad and node over domain slot dom's constants, CD[dom] ------------------------------

// SX: the schema's length beyond "hm96-sha512/row/v2" (0; 4 for "hm96-sha512/row-seg/v1").
template <int SX>
DEV void frame_leaf(int dom, u64 pos, const u64* h, u64* out) {
  switch (nbytes(pos)) {
    case 1: rc::frame_leaf_nb<SX, 1>(CD[dom], pos, h, out); break;
    case 2: rc::frame_leaf_nb<SX, 2>(CD[dom], pos, h, out); break;
    case 3: rc::frame_leaf_nb<SX, 3>(CD[dom], pos, h, out); break;
    default: rc::frame_leaf_nb<SX, 4>(CD[dom], pos, h, out); break;
  }
}
// The frame leaf's first block holds pos but not h, so its compression can run ahead (the salt phase): frame_first is the
// state after it, frame_second the leaf from that state and h.
DEV void frame_first(int dom, u64 pos, u64* out) { rc::frame_first(CD[dom], pos, out); }
template <int SX>
DEV void frame_second(int dom, u64 pos, const u64* h, u64* out) { rc::frame_second<SX>(CD[dom], pos, h, out); }
DEV void pad(int dom, u64 rank, u64* out) { rc::pad(CD[dom], rank, out); }
// Each arm indexes CD itself: through rc::node's pointers the node kernel stages its constant-bank addresses in uniform
// registers and grows.
DEV void node(int dom, int level, u64 index, const u64* L, const u64* R, u64* out) {
  switch (nbytes(index)) {
    case 1: rc::node_nb_from<1>(CD[dom].node_tpl, CD[dom].node_mid[level], index, L, R, out); break;
    case 2: rc::node_nb_from<2>(CD[dom].node_tpl, CD[dom].node_mid[level], index, L, R, out); break;
    case 3: rc::node_nb_from<3>(CD[dom].node_tpl, CD[dom].node_mid[level], index, L, R, out); break;
    default: rc::node_nb_from<4>(CD[dom].node_tpl, CD[dom].node_mid[level], index, L, R, out); break;
  }
}

// --- the leaf's parts ----------------------------------------------------------------------------------------------

// x for the e2e leaf: the 32 int8 at A[32 pos ..] as bf16 words (little-endian), twice: one data block, then P(1024)'s
// fixed padding block.
DEV void row_e2e(const uint8_t* A, u64 pos, u64* x) {
  const uint4 a0 = ldg16(A + 32 * pos), a1 = ldg16(A + 32 * pos + 16);
  const u32 aw[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
  u64 W[16];
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    u32 h[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) h[t] = __float_as_uint((float)(int8_t)(aw[q] >> (8 * t))) >> 16;
    W[q] = W[q + 8] = be64(h[0] | h[1] << 16, h[2] | h[3] << 16);
  }
  cpy8(x, C.mid_P);
  sha::compress(x, W);
  sha::compress_kw(x, C.kw_xpad);
}

// A row's bytes as 16-byte loads: `Contig`, lb bytes at one 16-byte aligned address (reads may run up to 15 bytes past
// the row, inside the buffer); `TileSrc`, a tile leaf's block of an int8 operand (rows K bytes apart, 32 bytes of each in
// row-major order, rows at or past `rows` reading as zero).
// pf(off): an L1 prefetch of the 128 bytes at off (hash_after_block's next block)
DEV void pf_l1(const uint8_t* p) { asm volatile("prefetch.global.L1 [%0];" ::"l"(p)); }
struct Contig {
  const uint8_t* p;
  DEV uint4 ld16(u32 off) const { return ldg16(p + off); }
  DEV void pf(u32 off) const { pf_l1(p + off); }
};
struct TileSrc {
  const uint8_t* base;
  u64 K;
  long long rows;
  DEV uint4 ld16(u32 off) const { return (long long)(off >> 5) < rows ? ldg16(base + (off >> 5) * K + (off & 31)) : make_uint4(0, 0, 0, 0); }
  DEV void pf(u32 off) const {
#pragma unroll
    for (u32 r = off >> 5; r < (off >> 5) + 4; ++r)
      if ((long long)r < rows) pf_l1(base + r * K);
  }
};

// SHA-512 of a 128-byte first block (its chaining value cv) and lb row bytes from `src`. pf: each block's compression
// starts with an L1 prefetch of the next block's bytes, so their loads don't wait in the chain.
template <class Src>
DEV void hash_after_block(const u64* cv, const Src& src, u32 lb, u64* x, bool pf = false) {
  cpy8(x, cv);
  const u32 nblk = (lb + 17 + 127) / 128;
  for (u32 b = 0; b < nblk; ++b) {
    if (pf && 128 * (b + 1) < lb) src.pf(128 * (b + 1));
    u64 W[16];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const u32 off = 128 * b + 16 * c;
      const uint4 v = off < lb ? src.ld16(off) : make_uint4(0, 0, 0, 0);
      W[2 * c] = be64(v.x, v.y);
      W[2 * c + 1] = be64(v.z, v.w);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const u32 base = 128 * b + 8 * i;
      if (base + 8 > lb) {
        if (base <= lb) {
          const u32 j = lb - base;
          W[i] = (j ? W[i] & (~0ull << (64 - 8 * j)) : 0) | (0x80ull << (56 - 8 * j));
        } else {
          W[i] = 0;
        }
      }
    }
    if (b == nblk - 1) W[15] = (u64)(128 + lb) * 8;
    sha::compress(x, W);
  }
}

// The tile layout (Decision 46): an M x K int8 operand (row-major, K a multiple of 32) as ceil(M/R) x K/32 blocks of R rows
// x 32 values; leaf pos is block (b_r, b_c) = (pos % ceil(M/R), pos / ceil(M/R)), its bytes the block's in row-major order
// (32 R bytes; rows past M are zero), so a check's 128-row tile at one 32-step is 128/R adjacent leaves.
struct TileGeom {
  const uint8_t* A;
  u64 M, K, mr;   // mr = ceil(M/R)
  u32 R;
  DEV TileSrc block(u64 pos, u32 row0 = 0) const {
    const u64 br = pos % mr, bc = pos / mr, r = br * R + row0;
    return TileSrc{A + r * K + 32 * bc, K, (long long)M - (long long)r};
  }
};

// Segment sources for seg_kernel: rows at a byte stride, or tile leaves (segment j is the block's rows 32 j .. 32 j + 31).
struct SegRows {
  const uint8_t* rows;
  u64 stride;
  DEV Contig segment(u64 pos, u32 j) const { return Contig{rows + pos * stride + 1024ull * j}; }
};
struct SegTiles {
  TileGeom g;
  DEV TileSrc segment(u64 pos, u32 j) const { return g.block(pos, 32 * j); }
};

// sha512/row-seg/v1's segment digests: d_j = SHA-512(S(j, s, m_j) || segment j), a thread a segment, to D (64 bytes each,
// in digest byte order, row pos's s digests contiguous: the outer hash then reads them as a row of 64 s bytes).
template <class Segs>
__global__ void __launch_bounds__(256) seg_kernel(Segs segs, u64 n, u32 s, u32 last_bytes, uint8_t* __restrict__ D) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * s) return;
  const u64 pos = t / s;
  const u32 j = (u32)(t % s);
  u64 x[8];
  hash_after_block(C.seg_mid[j], segs.segment(pos, j), j + 1 < s ? 1024u : last_bytes, x);
  uint4* o = reinterpret_cast<uint4*>(D + 64 * t);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    o[i] = make_uint4(bswap32((u32)(x[2 * i] >> 32)), bswap32((u32)x[2 * i]), bswap32((u32)(x[2 * i + 1] >> 32)),
                      bswap32((u32)x[2 * i + 1]));
}

DEV void salt_prf(u64 pos, u64* Y) {
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    u64 w[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = C.ks[i];
    w[8] = pos; w[9] = r; w[10] = 0x8000000000000000ull;
    w[11] = w[12] = w[13] = w[14] = 0;
    w[15] = 80 * 8;
    sha::iv(Y + 8 * r);
    sha::compress_from<8>(Y + 8 * r, C.ks_m8, w);
  }
}
DEV void salt_given(const uint8_t* s, u64* Y) {
#pragma unroll
  for (int c = 0; c < 12; ++c) {
    const uint4 v = ldg16(s + 16 * c);
    Y[2 * c] = be64(v.x, v.y);
    Y[2 * c + 1] = be64(v.z, v.w);
  }
}

// c = SHA-512(salt_prefix || y)
DEV void salt_digest(const u64* Y, u64* c) { rc::salt_digest(C.hm, Y, c); }

// M_key y as 16 little-endian u32 (row_commit's mask, its loop unrolled) from T in shared memory (TS, the byte table's rows
// TSTRIDE apart) or read from global memory through L1 (rows 256 apart). NIB: T is the nibble table (Committer::Tn) through
// L1: 31 rows, 8 KB, which stays in L1 at a 100% shared-memory carveout, where all 256 rows (64 KB) came from L2, and a
// warp's lanes read one chunk of at most 16 rows: 256 bytes in two lines, where row-major rows took a line a row.
template <bool TS, bool NIB = false>
DEV void hm96_mask(const uint8_t* T, const u64* Y, u32* m) {
  if constexpr (NIB) rc::mask<false>(rc::NibTab<true>{T}, Y, m);
  else rc::mask<false>(rc::ByteTab<TS ? TSTRIDE : 256, !TS>{T}, Y, m);
}

struct RowE2E {
  const uint8_t* A;
  DEV void digest(u64 pos, u64* x) const { row_e2e(A, pos, x); }
};
struct RowBytes {
  const uint8_t* base;
  u64 stride;
  u32 lb;
  DEV void digest(u64 pos, u64* x) const { hash_after_block(C.mid_P, Contig{base + pos * stride}, lb, x); }
};
#ifdef COMMIT_GPU_DIAG_NOOP
// timing only: a leaf's bytes from registers, the same compressions without the operand's loads
struct FakeSrc {
  u32 seed;
  DEV uint4 ld16(u32 off) const { return make_uint4(seed ^ off, seed + off, seed * 3u ^ off, off); }
  DEV void pf(u32) const {}
};
#endif
struct RowTile {
  TileGeom g;
  bool pf = false;  // hash_after_block's prefetch of the next block (Committer::prefetch)
#ifdef COMMIT_GPU_DIAG_NOOP
  DEV void digest(u64 pos, u64* x) const { hash_after_block(C.mid_P, FakeSrc{(u32)pos}, 32 * g.R, x); }
#else
  DEV void digest(u64 pos, u64* x) const { hash_after_block(C.mid_P, g.block(pos), 32 * g.R, x, pf); }
#endif
};

DEV void ld8(const u64* p, u64* d) {
  const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 v = q[i];
    d[2 * i] = (u64)v.y << 32 | v.x;
    d[2 * i + 1] = (u64)v.w << 32 | v.z;
  }
}
// ld8 from L2 (ld.global.cg): nodes another CTA of the same kernel stored
DEV void ld8cg(const u64* p, u64* d) {
  const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 v = __ldcg(q + i);
    d[2 * i] = (u64)v.y << 32 | v.x;
    d[2 * i + 1] = (u64)v.w << 32 | v.z;
  }
}
DEV void st8(u64* p, const u64* d) {
  uint4* q = reinterpret_cast<uint4*>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) q[i] = make_uint4((u32)d[2 * i], (u32)(d[2 * i] >> 32), (u32)d[2 * i + 1], (u32)(d[2 * i + 1] >> 32));
}

// M_key y and c = SHA-512(salt_prefix || y) for pos to S[pos] (16 words), T at T (TS: in shared memory).
template <bool PRF, bool TS, bool NIB = false>
DEV void salt_one(u64 pos, const uint8_t* __restrict__ salts, const uint8_t* T, u64* __restrict__ S) {
  u64 Y[24], c[8], my[8];
  u32 m[16];
  if constexpr (PRF) salt_prf(pos, Y);
  else salt_given(salts + 192 * pos, Y);
#ifdef COMMIT_GPU_DIAG_NOMASK
  for (int w = 0; w < 16; ++w) m[w] = (u32)Y[w];
#else
  hm96_mask<TS, NIB>(T, Y, m);
#endif
  salt_digest(Y, c);
  rc::mask_words(m, my);
  st8(S + 16 * pos, my);
  st8(S + 16 * pos + 8, c);
}

// The salt phase, a thread a position: S[pos]. Nothing here reads the data, so it can run before the operand exists; the
// data phase (leaf_kernel) then has only the row on its path.
template <bool PRF, bool TS>
__global__ void __launch_bounds__(LEAF_THREADS, 1)
    salt_kernel(const uint8_t* __restrict__ salts, const uint4* __restrict__ Tg, u64 n, u64* __restrict__ S) {
  extern __shared__ uint4 smem[];
  if constexpr (TS) {
    for (int i = threadIdx.x; i < 256 * 16; i += blockDim.x) smem[(i >> 4) * (TSTRIDE / 16) + (i & 15)] = Tg[i];
    __syncthreads();
  }
  const u64 pos = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= n) return;
  salt_one<PRF, TS>(pos, salts, TS ? reinterpret_cast<const uint8_t*>(smem) : reinterpret_cast<const uint8_t*>(Tg), S);
}

// The tree leaf at pos from its row digest x and the salt phase's S[pos]: b || c = (x ^ M_key y) || c (to bc when given),
// h = SHA-512(leaf_prefix || b || c), then the frame leaf (F: from its first block's state F[pos], computed ahead). A
// producer that hashes its own rows (an epilogue) ends here.
template <int SX>
DEV void leaf_from_digest(int dom, u64 pos, const u64* x, const u64* __restrict__ S, u64* out, u64* __restrict__ bc,
                          const u64* __restrict__ F = nullptr) {
  u64 B[16];
#ifdef COMMIT_GPU_DIAG_NOS   // timing only: S[pos] and F[pos] from registers
#pragma unroll
  for (int i = 0; i < 16; ++i) B[i] = pos * (2 * i + 1);
#else
  ld8(S + 16 * pos, B);
  ld8(S + 16 * pos + 8, B + 8);
#endif
#pragma unroll
  for (int i = 0; i < 8; ++i) B[i] ^= x[i];
  if (bc) {
    st8(bc + 16 * pos, B);
    st8(bc + 16 * pos + 8, B + 8);
  }
  u64 h[8];
#ifdef COMMIT_GPU_DIAG_MERGED
  for (int i = 0; i < 8; ++i) h[i] = B[i] ^ B[8 + i];
#else
  rc::hm96_value(C.hm, B, h);
#endif
  if (F) {
#ifdef COMMIT_GPU_DIAG_NOS
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = pos ^ (u64)i << 40;
#else
    ld8(F + 8 * pos, out);
#endif
    frame_second<SX>(dom, pos, h, out);
  } else {
    frame_leaf<SX>(dom, pos, h, out);
  }
}

// The data phase, a thread a position: the tree leaf (or pad) to lv0[pos] from the row and S[pos]; b || c to bc[pos]
// when bc is given.
template <class Row, int SX>
__global__ void __launch_bounds__(LEAF_THREADS)
    leaf_kernel(int dom, Row row, const u64* __restrict__ S, u64 n, u64 width, u64* __restrict__ lv0, u64* __restrict__ bc) {
  const u64 pos = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= width) return;
  u64 out[8];
  if (pos < n) {
    u64 x[8];
    row.digest(pos, x);
    leaf_from_digest<SX>(dom, pos, x, S, out, bc);
  } else {
    pad(dom, pos, out);
  }
  st8(lv0 + 8 * pos, out);
}

// Level d+1 from level d (n_in nodes), a thread a node.
__global__ void __launch_bounds__(NODE_THREADS) level_kernel(int dom, const u64* __restrict__ in, u64* __restrict__ out, u64 n_in, int d) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_in / 2) return;
  u64 L[8], R[8], o[8];
  ld8(in + 16 * i, L);
  ld8(in + 16 * i + 8, R);
  node(dom, d, i, L, R, o);
  st8(out + 8 * i, o);
}

// --- the small path: CTAs of SMALL_THREADS with no shared memory, so they fit beside a GEMM's CTAs on an SM -------------

// level d of a tree of `width` leaves stored leaves first (Committer::levels)
DEV u64* level_at(u64* levels, u64 width, int d) { return levels + 8 * (2 * width - 2 * (width >> d)); }

// Levels 1 .. L of a CTA's cnt = 2^L aligned leaves (from leaf cnt * blockIdx.x), through global memory: what a thread
// stores before __syncthreads() is visible to the CTA's threads after it, so no shared memory is needed.
DEV void fold_cta(int dom, u64* levels, u64 width, u32 cnt, int L) {
  const u32 t = threadIdx.x;
  for (int l = 1; l <= L; ++l) {
    __syncthreads();
    const u32 m = cnt >> l;
    if (t < m) {
      const u64 i = (u64)blockIdx.x * m + t;
      const u64* in = level_at(levels, width, l - 1);
      u64 Lc[8], Rc[8], o[8];
      ld8(in + 16 * i, Lc);
      ld8(in + 16 * i + 8, Rc);
      node(dom, l - 1, i, Lc, Rc, o);
      st8(level_at(levels, width, l) + 8 * i, o);
    }
  }
}
#ifdef COMMIT_GPU_DIAG_NOTREE
// timing only: fold_cta's compressions and barriers on the thread's own node o, without the levels' loads and stores
// (the CTA's root, level L, is still stored for the levels above)
DEV void fold_cta_regs(int dom, u64* levels, u64 width, u32 cnt, int L, u64* o) {
  const u32 t = threadIdx.x;
  for (int l = 1; l <= L; ++l) {
    __syncthreads();
    const u32 m = cnt >> l;
    if (t < m) {
      const u64 i = (u64)blockIdx.x * m + t;
      u64 Lc[8], Rc[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) { Lc[j] = o[j]; Rc[j] = o[(j + 1) & 7]; }
      node(dom, l - 1, i, Lc, Rc, o);
      if (l == L) st8(level_at(levels, width, l) + 8 * i, o);
    }
  }
}
#endif

// The small path's salt phase for slots s0 .. s0 + gridDim.y - 1 of the commitments of one domain (S_k = S + 16 n k), and
// with slot 0 the frame leaves' first blocks F and the pads (positions n .. width) into level 0 with every CTA subtree that
// holds only pads, so the data phase computes only what the data decides (F and the pads don't depend on the slot).
template <bool PRF>
__global__ void __launch_bounds__(SMALL_THREADS, SMALL_MINB)
    salt_small(int dom, const uint8_t* __restrict__ salts, const uint4* __restrict__ Tg, u64 n, u64 width, u64* __restrict__ S,
               u64* __restrict__ F, u64* levels, int s0) {
  const u64 pos = (u64)blockIdx.x * SMALL_THREADS + threadIdx.x;
  const u32 cnt = (u32)(width < SMALL_THREADS ? width : SMALL_THREADS);
  const u64 slot = s0 + blockIdx.y;
  if (pos < n) {
    salt_one<PRF, false, true>(pos, salts, reinterpret_cast<const uint8_t*>(Tg), S + 16 * n * slot);
    if (slot == 0) {
      u64 f[8];
      frame_first(dom, pos, f);
      st8(F + 8 * pos, f);
    }
  } else if (pos < width && slot == 0) {
    u64 o[8];
    pad(dom, pos, o);
    st8(levels + 8 * pos, o);
  }
  if (slot == 0 && (u64)blockIdx.x * cnt >= n) {
    int L = 0;
    while ((1u << L) < cnt) ++L;
    fold_cta(dom, levels, width, cnt, L);
  }
}

// The small path's data phase over the CTAs that hold data (ceil(n / SMALL_THREADS)): each thread a leaf, then its CTA's
// levels 1 .. min(SMALL_LEVELS, depth). g: thread t takes the CTA's leaf (t % g) * (SMALL_THREADS / g) + t / g, so with
// tile leaves of mr = SMALL_THREADS / g row blocks a column (Committer::lanes) the g leaves of one row block at adjacent
// columns sit in adjacent lanes and a warp's 16-byte loads touch 32 / g lines, not 32. ctr (Committer::fused): the CTA
// that finishes last also folds the nl levels above the CTAs.
template <class Row, int SX>
__global__ void __launch_bounds__(SMALL_THREADS, SMALL_MINB)
    leaf_small(int dom, Row row, const u64* __restrict__ S, const u64* __restrict__ F, u64 n, u64 width, u64* levels,
               u64* __restrict__ bc, u32 g, u32* ctr, int nl) {
  const u64 pos = (u64)blockIdx.x * SMALL_THREADS + (threadIdx.x % g) * (SMALL_THREADS / g) + threadIdx.x / g;
  const u32 cnt = (u32)(width < SMALL_THREADS ? width : SMALL_THREADS);
#ifdef COMMIT_GPU_DIAG_NOTREE
  u64 out[8] = {};
  if (pos < n) {
    u64 x[8];
    row.digest(pos, x);
    leaf_from_digest<SX>(dom, pos, x, S, out, bc, F);
    st8(levels + 8 * pos, out);
  }
  int L = 0;
  while ((1u << L) < cnt) ++L;
  fold_cta_regs(dom, levels, width, cnt, L, out);
#else
  if (pos < n) {
    u64 x[8], out[8];
    row.digest(pos, x);
    leaf_from_digest<SX>(dom, pos, x, S, out, bc, F);
    st8(levels + 8 * pos, out);
  }
  int L = 0;
  while ((1u << L) < cnt) ++L;
  fold_cta(dom, levels, width, cnt, L);
#endif
  if (!ctr) return;
  // the levels above the CTAs (Committer::fused): the CTA that finishes last folds them, so no launch follows; its
  // count goes back to 0 for the next launch. Thread 0 stored the CTA's root (level SMALL_LEVELS), the one node another
  // CTA reads, so it alone fences before counting.
  u32 old = 0;
  if (threadIdx.x == 0) {
    __threadfence();
    old = atomicAdd(ctr, 1u);
  }
  if (!__syncthreads_or(threadIdx.x == 0 && old == gridDim.x - 1)) return;
  __threadfence();
  for (int l = 1; l <= nl; ++l) {
    if (l > 1) __syncthreads();
    const u64 m = (width >> SMALL_LEVELS) >> l;
    if (threadIdx.x < m) {
      const u64* in = level_at(levels, width, SMALL_LEVELS + l - 1);
      u64 Lc[8], Rc[8], o[8];
      ld8cg(in + 16 * threadIdx.x, Lc);
      ld8cg(in + 16 * threadIdx.x + 8, Rc);
      node(dom, SMALL_LEVELS + l - 1, threadIdx.x, Lc, Rc, o);
      st8(level_at(levels, width, SMALL_LEVELS + l) + 8 * threadIdx.x, o);
    }
  }
  if (threadIdx.x == 0) *ctr = 0;
}

// Levels d+1 .. d+nl from level d (at most SMALL_TOP nodes) in one CTA through global memory.
__global__ void __launch_bounds__(SMALL_TOP / 2, SMALL_MINB) top_small(int dom, u64* levels, u64 width, int d, int nl) {
  const u32 t = threadIdx.x;
  for (int l = 1; l <= nl; ++l) {
    if (l > 1) __syncthreads();
    const u64 m = (width >> d) >> l;
    if (t < m) {
      const u64* in = level_at(levels, width, d + l - 1);
      u64 Lc[8], Rc[8], o[8];
      ld8(in + 16 * t, Lc);
      ld8(in + 16 * t + 8, Rc);
      node(dom, d + l - 1, t, Lc, Rc, o);
      st8(level_at(levels, width, d + l) + 8 * t, o);
    }
  }
}

// The top: levels d+1 .. d+nl from level d (n_in <= 512 nodes) in one block, the levels in shared memory.
// out[l] is level d+1+l's array.
struct LevelPtrs { u64* p[10]; };
__global__ void __launch_bounds__(NODE_THREADS) node_kernel(int dom, const u64* __restrict__ in, LevelPtrs out, u64 n_in, int d, int nl) {
  __shared__ u64 s[NODE_THREADS * 8];
  const int t = threadIdx.x;
  const u64 n1 = n_in / 2, i1 = (u64)blockIdx.x * NODE_THREADS + t;
  u64 L[8], R[8], o[8];
  if (i1 < n1) {
    ld8(in + 16 * i1, L);
    ld8(in + 16 * i1 + 8, R);
    node(dom, d, i1, L, R, o);
    st8(s + 8 * t, o);
    st8(out.p[0] + 8 * i1, o);
  }
  u64 cnt = n1 < NODE_THREADS ? n1 : NODE_THREADS;
  for (int l = 1; l < nl && cnt > 1; ++l) {
    __syncthreads();
    cnt >>= 1;
    if (t < (int)cnt) {
      cpy8(L, s + 16 * t);
      cpy8(R, s + 16 * t + 8);
      node(dom, d + l, (u64)blockIdx.x * cnt + t, L, R, o);
    }
    __syncthreads();
    if (t < (int)cnt) {
      cpy8(s + 8 * t, o);
      st8(out.p[l] + 8 * ((u64)blockIdx.x * cnt + t), o);
    }
  }
}

#ifdef COMMIT_GPU_DIAG_QUAD
// Timing only: a 4-ary node, 3 compressions after the level's midstate (children 0-1, children 2-3, then the index and
// padding), as a 4-ary frame node's message would take; it stands for two binary levels' 2 + 2 serial compressions.
DEV void quad(int dom, int level, u64 index, const u64* a, const u64* b, const u64* c, const u64* e, u64* out) {
  u64 W[16];
  cpy8(out, CD[dom].node_mid[level]);
#pragma unroll
  for (int i = 0; i < 8; ++i) { W[i] = a[i]; W[8 + i] = b[i]; }
  sha::compress(out, W);
#pragma unroll
  for (int i = 0; i < 8; ++i) { W[i] = c[i]; W[8 + i] = e[i]; }
  sha::compress(out, W);
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = 0;
  W[0] = index; W[1] = 0x8000000000000000ull; W[15] = (128 + 256 + 8) * 8;
  sha::compress(out, W);
}
DEV void quad_from(int dom, const u64* p, int level, u64 index, u64* o) {
  u64 a[8], b[8], c[8], e[8];
  ld8(p, a); ld8(p + 8, b); ld8(p + 16, c); ld8(p + 24, e);
  quad(dom, level, index, a, b, c, e, o);
}
__global__ void __launch_bounds__(NODE_THREADS) quad_level_kernel(int dom, const u64* __restrict__ in, u64* __restrict__ out, u64 n_in, int d) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_in / 4) return;
  u64 o[8];
  quad_from(dom, in + 32 * i, d, i, o);
  st8(out + 8 * i, o);
}
// The top in one block from level d (n_in <= 1024 nodes, nl levels): 4-ary steps, then a binary one if one level is left.
__global__ void __launch_bounds__(NODE_THREADS) quad_node_kernel(int dom, const u64* __restrict__ in, LevelPtrs out, u64 n_in, int d, int nl) {
  __shared__ u64 s[NODE_THREADS * 8];
  const int t = threadIdx.x;
  u64 L[8], R[8], o[8];
  int l = nl >= 2 ? 2 : 1;
  u64 cnt = n_in >> l;
  if (t < (int)cnt) {
    if (l == 2) {
      quad_from(dom, in + 32 * t, d, t, o);
    } else {
      ld8(in + 16 * t, L);
      ld8(in + 16 * t + 8, R);
      node(dom, d, t, L, R, o);
    }
    st8(s + 8 * t, o);
    st8(out.p[l - 1] + 8 * t, o);
  }
  while (l < nl && cnt > 1) {
    __syncthreads();
    const int step = nl - l >= 2 ? 2 : 1;
    cnt >>= step;
    if (t < (int)cnt) {
      if (step == 2) {
        quad(dom, d + l, t, s + 32 * t, s + 32 * t + 8, s + 32 * t + 16, s + 32 * t + 24, o);
      } else {
        cpy8(L, s + 16 * t);
        cpy8(R, s + 16 * t + 8);
        node(dom, d + l, t, L, R, o);
      }
    }
    __syncthreads();
    l += step;
    if (t < (int)cnt) {
      cpy8(s + 8 * t, o);
      st8(out.p[l - 1] + 8 * t, o);
    }
  }
}
#endif
// --- host --------------------------------------------------------------------------------------------------------

static const char SCHEMA[] = "hm96-sha512/row/v2";
static const char SCHEMA_SEG[] = "hm96-sha512/row-seg/v1";   // 4 bytes longer: SX = 4
static const char SEG_TAG[] = "verity/sha512-row-seg/v1/segment";   // + its NUL

struct Params {
  uint8_t P[128], salt_prefix[128], leaf_prefix[128], domain_id[64], ks[64];
  std::vector<uint8_t> T;   // 256 x 256
};

static std::vector<uint8_t> readf(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
  std::vector<uint8_t> v;
  uint8_t buf[1 << 16];
  size_t r;
  while ((r = fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + r);
  fclose(f);
  return v;
}
static void writef(const std::string& path, const void* p, size_t n) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f || fwrite(p, 1, n, f) != n) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(2); }
  fclose(f);
}

static Params read_params(const std::string& path) {
  const auto v = readf(path);
  const size_t want = 4 + 128 * 3 + 64 * 2 + 65536;
  if (v.size() != want || memcmp(v.data(), "CGP1", 4)) { fprintf(stderr, "%s: not a CGP1 params file\n", path.c_str()); exit(2); }
  Params p;
  const uint8_t* q = v.data() + 4;
  memcpy(p.P, q, 128); q += 128;
  memcpy(p.salt_prefix, q, 128); q += 128;
  memcpy(p.leaf_prefix, q, 128); q += 128;
  memcpy(p.domain_id, q, 64); q += 64;
  memcpy(p.ks, q, 64); q += 64;
  p.T.assign(q, q + 65536);
  return p;
}

static void words_of(const uint8_t* block, u64* w, int n) { for (int i = 0; i < n; ++i) w[i] = sha::be64(block + 8 * i); }

// seg: sha512/row-seg/v1 rows of seg_bits bits (P is then Q(seg_bits), the outer hash's first block).
static Consts build_consts(const Params& p, u32 lb, bool seg, u64 seg_bits, Dom& dm) {
  Consts c{};
  dm = Dom{};
  rc::midstate(p.P, c.mid_P);
  rc::pad_schedule(128 + lb, c.kw_xpad);
  rc::midstate(p.salt_prefix, c.hm.mid_salt);
  rc::midstate(p.leaf_prefix, c.hm.mid_leaf);
  rc::pad_schedule(256, c.hm.kw_leafpad);
  words_of(p.ks, c.ks, 8);
  {
    u64 blk[16] = {}, st[8];
    memcpy(blk, c.ks, 64);
    sha::iv(st);
    sha::rounds_prefix(st, blk, 8, c.ks_m8);
  }
  const char* sname = seg ? SCHEMA_SEG : SCHEMA;
  const auto* schema = reinterpret_cast<const uint8_t*>(sname);
  if (rc::frame_plan(p.domain_id, schema, (int)strlen(sname), dm) || rc::node_plan(p.domain_id, dm)) {
    fprintf(stderr, "the frame's leaf, pad or node messages are not the blocks the kernels take\n");
    exit(2);
  }
  if (seg) {   // S(j, s, m_j) = tag || u64be j || u64be s || u64be m_j, zero padded to 128 bytes
    const u64 s = (seg_bits + 8191) / 8192;
    if (s > 64) { fprintf(stderr, "row-seg rows of at most 64 segments\n"); exit(2); }
    for (u64 j = 0; j < s; ++j) {
      uint8_t blk[128] = {};
      memcpy(blk, SEG_TAG, sizeof SEG_TAG);
      sha::put_be64(blk + sizeof SEG_TAG, j);
      sha::put_be64(blk + sizeof SEG_TAG + 8, s);
      sha::put_be64(blk + sizeof SEG_TAG + 16, j + 1 < s ? 8192 : seg_bits - 8192 * (s - 1));
      rc::midstate(blk, c.seg_mid[j]);
    }
  }
  return c;
}

static int depth_of(u64 n) { int d = 0; while ((1ull << d) < n) ++d; return d; }

// The tile layout's leaf count, after checking its geometry (K a multiple of 32; a leaf of at most 64 row-seg segments).
static u64 tile_leaves(u64 M, u64 K, u64 R) {
  if (!M || !K || K % 32 || !R || R > 2048) {
    fprintf(stderr, "tiles: M, K > 0, K a multiple of 32, 1 <= R <= 2048, not %llu %llu %llu\n", (unsigned long long)M,
            (unsigned long long)K, (unsigned long long)R);
    exit(2);
  }
  return (M + R - 1) / R * (K / 32);
}

// The salt phase's S[pos] (M_key y || c, 16 words, as salt_one<true> writes it) on the CPU from the same constants and the
// row-major T, so host threads can make the small path's S records (ovl.cu's cgh arms copy them in).
static void host_salt(const Consts& c, const uint8_t* T, u64 pos, u64* S) {
  u64 Y[24];
  for (int r = 0; r < 3; ++r) {
    u64 w[16] = {};
    for (int i = 0; i < 8; ++i) w[i] = c.ks[i];
    w[8] = pos; w[9] = (u64)r; w[10] = 0x8000000000000000ull; w[15] = 80 * 8;
    sha::iv(Y + 8 * r);
    sha::compress(Y + 8 * r, w);
  }
  u32 m[16];
  rc::mask<false>(rc::ByteTab<256, false>{T}, Y, m);
  rc::mask_words(m, S);
  rc::salt_digest(c.hm, Y, S + 8);
}

struct Committer {
  u64 n = 0, width = 0;
  int depth = 0;
  int dom = 0;                      // this domain's slot in CD
  bool small = false;               // the small path (salt_small, leaf_small, top_small)
  int slots = 1;                    // the small path's salt slots (commitments of this domain salted ahead in one launch)
  u64* levels = nullptr;            // every level, leaves first: level d at offset 8 (2 width - 2 width >> d) words
  u64* bc = nullptr;
  u64* S = nullptr;                 // the salt phase's M_key y || c, 16 words a leaf (a slot after another)
  u64* F = nullptr;                 // the small path's frame-leaf first-block states, 8 words a leaf
  uint4* T = nullptr;
  uint4* Tn = nullptr;              // the small path's nibble table (hm96_mask's NIB layout), 8 KB
  u32* ctr = nullptr;               // the small path's count of finished leaf CTAs (fused)
  bool fuse_top = false;            // the small path: the last leaf CTA folds the levels above the CTAs (fused)
  bool prefetch = false;            // tiles(): each leaf prefetches its next block into L1 (RowTile::pf)
  uint8_t* D = nullptr;             // row-seg: the segment digests
  u32 segs = 0, seg_last = 0;
  std::vector<u64*> lv;
  Consts hc;                        // the host copies of C and CD[dom]
  Dom hd;
  // lb: the row bytes after P; seg: row-seg rows of seg_bits bits (lb is then the outer hash's 64 s bytes; the empty row
  // has no segment)
  void init(const Params& p, u64 n_, u32 lb, bool keep_bc, bool seg = false, u64 seg_bits = 0, int dom_ = 0, bool small_ = false,
            int slots_ = 1) {
    n = n_; depth = depth_of(n); width = 1ull << depth; dom = dom_; small = small_; slots = small_ ? slots_ : 1;
    if (n == 0 || n > (1ull << 32)) { fprintf(stderr, "1 to 2^32 leaves, not %llu\n", (unsigned long long)n); exit(2); }
    if (dom < 0 || dom >= COMMIT_GPU_DOMS) { fprintf(stderr, "domain slot %d of %d\n", dom, COMMIT_GPU_DOMS); exit(2); }
    if (seg) {
      segs = (u32)((seg_bits + 8191) / 8192);
      seg_last = segs ? (u32)((seg_bits - 8192ull * (segs - 1) + 7) / 8) : 0;
      lb = 64 * segs;
      CK(cudaMalloc(&D, n * lb + 16));
    }
    hc = build_consts(p, lb, seg, seg_bits, hd);
    CK(cudaMemcpyToSymbol(C, &hc, sizeof hc));
    CK(cudaMemcpyToSymbol(CD, &hd, sizeof hd, sizeof(Dom) * dom));
    CK(cudaMalloc(&levels, (2 * width) * 64));
    if (keep_bc) CK(cudaMalloc(&bc, n * 128));
    CK(cudaMalloc(&S, n * 128 * slots));
    if (small) CK(cudaMalloc(&F, n * 64));
    CK(cudaMalloc(&T, 65536));
    CK(cudaMemcpy(T, p.T.data(), 65536, cudaMemcpyHostToDevice));
    if (small) {
      std::vector<uint8_t> tn(8192);
      for (int h = 0; h < 2; ++h)
        for (int c = 0; c < 16; ++c)
          for (int u = 0; u < 16; ++u) memcpy(&tn[4096 * h + 256 * c + 16 * u], &p.T[256 * (h ? 16 * u : u) + 16 * c], 16);
      CK(cudaMalloc(&Tn, tn.size()));
      CK(cudaMemcpy(Tn, tn.data(), tn.size(), cudaMemcpyHostToDevice));
      CK(cudaMalloc(&ctr, sizeof(u32)));
      CK(cudaMemset(ctr, 0, sizeof(u32)));
    }
    u64 off = 0;
    for (int d = 0; d <= depth; ++d) { lv.push_back(levels + 8 * off); off += width >> d; }
  }
  // The salt phase. Small path: every slot's S, F and the pads in CTAs of SMALL_THREADS, T read through L1/L2. Otherwise,
  // above 2^16 leaves T sits in shared memory (one block of LEAF_THREADS an SM); at or below, blocks of 64 threads read T
  // through L1, which spreads few leaves over the SMs and takes no shared memory.
  // Small path: slots s0 .. s0 + ns - 1 (default all); F and the pads come with slot 0, which runs first.
  template <bool PRF>
  void salts(const uint8_t* given, cudaStream_t st, int s0 = 0, int ns = -1) {
    if (small) {
      if (ns < 0) ns = slots - s0;
      const dim3 g((unsigned)((width + SMALL_THREADS - 1) / SMALL_THREADS), (unsigned)ns);
      salt_small<PRF><<<g, SMALL_THREADS, 0, st>>>(dom, given, Tn, n, width, S, F, levels, s0);
    } else if (n > (1u << 16)) {
      auto k = salt_kernel<PRF, true>;
      static bool attr = false;
      if (!attr) { CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, TSMEM)); attr = true; }
      k<<<(unsigned)((n + LEAF_THREADS - 1) / LEAF_THREADS), LEAF_THREADS, TSMEM, st>>>(given, T, n, S);
    } else {
      salt_kernel<PRF, false><<<(unsigned)((n + 63) / 64), 64, 0, st>>>(given, T, n, S);
    }
  }
  // The data phase's leaves (after salts(); slot: the small path's salt slot). Small path: leaf_small, which also folds each
  // CTA's levels. Otherwise blocks of 64 threads up to 2^16 leaves, so few leaves still spread over the SMs.
  // g: leaf_small's lane order (1: thread t leaf t).
  template <class Row, int SX = 0>
  void leaves(const Row& row, cudaStream_t st, int slot = 0, u32 g = 1) {
    if (small) {
      leaf_small<Row, SX><<<(unsigned)((n + SMALL_THREADS - 1) / SMALL_THREADS), SMALL_THREADS, 0, st>>>(
          dom, row, S + 16 * n * slot, F, n, width, levels, bc, g, fused() ? ctr : nullptr, depth - SMALL_LEVELS);
      return;
    }
    const unsigned t = width > (1u << 16) ? LEAF_THREADS : 64;
    leaf_kernel<Row, SX><<<(unsigned)((width + t - 1) / t), t, 0, st>>>(dom, row, S, n, width, lv[0], bc);
  }
  // row-seg rows at `rows` (stride bytes apart, 16-byte aligned): the segment digests, then the leaves over them
  void seg_leaves(const uint8_t* rows, u64 stride, cudaStream_t st) { seg_leaves_from(SegRows{rows, stride}, st); }
  template <class Segs>
  void seg_leaves_from(const Segs& sg, cudaStream_t st, int slot = 0) {
    const unsigned t = small ? SMALL_THREADS : 256;
    if (segs) seg_kernel<<<(unsigned)((n * segs + t - 1) / t), t, 0, st>>>(sg, n, segs, seg_last, D);
    leaves<RowBytes, 4>(RowBytes{D, 64ull * segs, 64u * segs}, st, slot);
  }
  // The tile layout over an M x K int8 operand at A (init with n = tile_leaves(M, K, R) and rows of 256 R bits): as
  // sha512/row/v2 rows, or (seg) as sha512/row-seg/v1 rows.
  void tiles(const uint8_t* A, u64 M, u64 K, u32 R, bool seg, cudaStream_t st, int slot = 0) {
    const TileGeom g{A, M, K, (M + R - 1) / R, R};
    if (seg) seg_leaves_from(SegTiles{g}, st, slot);
    else leaves<RowTile>(RowTile{g, prefetch}, st, slot, (u32)lanes(g.mr));
  }
  // Small path with fuse_top: when the levels above the CTAs fit one CTA (top_small's case), leaf_small's last CTA folds
  // them up to the root and nodes() launches nothing.
  bool fused() const { return small && fuse_top && depth > SMALL_LEVELS && (width >> SMALL_LEVELS) <= (u64)SMALL_TOP; }
  // The small path's lane order for tile leaves of mr row blocks a column: when mr divides SMALL_THREADS, a CTA's leaves
  // are SMALL_THREADS / mr whole columns, and the leaves of one row block go to adjacent lanes.
  static u64 lanes(u64 mr) { return mr <= (u64)SMALL_THREADS && SMALL_THREADS % mr == 0 ? SMALL_THREADS / mr : 1; }
  // Levels 1 .. top (default the root). Small path: above the CTAs' folded levels, a level a launch while a level is wider
  // than SMALL_TOP nodes, then the rest in one CTA (top_small). Otherwise a level a launch while a level is wider than 512
  // nodes (blocks of 64 threads below 2^16 nodes), then the rest in one block (COMMIT_GPU_DIAG_QUAD: the same with 4-ary
  // steps).
  void nodes(cudaStream_t st, int top = -1) {
    if (top < 0) top = depth;
    int d = 0;
    if (small) {
      if (fused()) return;
      d = std::min(SMALL_LEVELS, depth);
      for (; d < top && (width >> d) > SMALL_TOP; ++d) {
        const u64 n_out = width >> (d + 1);
        level_kernel<<<(unsigned)((n_out + SMALL_THREADS - 1) / SMALL_THREADS), SMALL_THREADS, 0, st>>>(dom, lv[d], lv[d + 1],
                                                                                                         width >> d, d);
      }
      if (d < top) top_small<<<1, (unsigned)std::max<u64>(1, (width >> d) / 2), 0, st>>>(dom, levels, width, d, top - d);
      return;
    }
#ifdef COMMIT_GPU_DIAG_QUAD
    for (; d + 2 <= top && (width >> d) > 4 * NODE_THREADS; d += 2) {
      const u64 n_out = width >> (d + 2);
      const unsigned t = n_out >= (1u << 16) ? NODE_THREADS : 64;
      quad_level_kernel<<<(unsigned)((n_out + t - 1) / t), t, 0, st>>>(dom, lv[d], lv[d + 2], width >> d, d);
    }
    if (d + 1 == top && (width >> d) > 2 * NODE_THREADS) {
      const u64 n_out = width >> (d + 1);
      const unsigned t = n_out >= (1u << 16) ? NODE_THREADS : 64;
      level_kernel<<<(unsigned)((n_out + t - 1) / t), t, 0, st>>>(dom, lv[d], lv[d + 1], width >> d, d);
      ++d;
    }
    if (d < top) {
      LevelPtrs o{};
      for (int l = 0; l < top - d; ++l) o.p[l] = lv[d + 1 + l];
      quad_node_kernel<<<1, NODE_THREADS, 0, st>>>(dom, lv[d], o, width >> d, d, top - d);
    }
#else
    for (; d < top && (width >> d) > 2 * NODE_THREADS; ++d) {
      const u64 n_out = width >> (d + 1);
      const unsigned t = n_out >= (1u << 16) ? NODE_THREADS : 64;
      level_kernel<<<(unsigned)((n_out + t - 1) / t), t, 0, st>>>(dom, lv[d], lv[d + 1], width >> d, d);
    }
    if (d < top) {
      LevelPtrs o{};
      for (int l = 0; l < top - d; ++l) o.p[l] = lv[d + 1 + l];
      node_kernel<<<1, NODE_THREADS, 0, st>>>(dom, lv[d], o, width >> d, d, top - d);
    }
#endif
  }
  // Levels from+1 .. depth on the CPU from level `from` (cur: its nodes' words, replaced by the root's); each level's words
  // are appended to `keep` when given. One GPU thread takes microseconds a compression, a CPU core a fraction of one, so
  // the narrow top of a tree is faster on the host, which needs the root anyway.
  void host_tail(int from, std::vector<u64>& cur, std::vector<std::vector<u64>>* keep = nullptr) const {
    std::vector<u64> nxt;
    for (int d = from; d < depth; ++d) {
      const u64 m = (width >> d) / 2;
      nxt.resize(8 * m);
      for (u64 i = 0; i < m; ++i) rc::node(hd, d, i, &cur[16 * i], &cur[16 * i + 8], &nxt[8 * i]);
      if (keep) keep->push_back(nxt);
      cur.swap(nxt);
    }
  }
  void release() { cudaFree(levels); cudaFree(bc); cudaFree(S); cudaFree(F); cudaFree(T); cudaFree(Tn); cudaFree(ctr); cudaFree(D); lv.clear(); }
};

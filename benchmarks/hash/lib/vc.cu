// libvc (vc.h, README.md): the served path's SHA-512 commitments, one code base. SHA-512's compression is
// verity/ml/kernels/sha512.cuh's (included by path); the rest is vendored from three seeds, each piece re-pointed
// from __constant__ memory to kernel parameters (__grid_constant__) so that plans of many domains run back to back on one
// stream with no symbol copy:
//   * native_rows.cu (934d0bcbd): LeafK / NodeK, the frame leaf, pad and node templates,
//     hash_after_block, salt_prf, salt_digest, the nibble-table mask, the leaf, the level kernels and the host builder;
//   * sharaw.cu (4369ae62): the byte-table mask (mask8) and the salt / data split (sr_salt8,
//     sr_data), with commit_gpu's tree schedule (a level a launch while a level is wider than 512 nodes, then one block for
//     the rest); sr_data's fold and climb (k_leaf_fold), its nodes by length or (sm_100) through compress_compact /
//     node_rt; sr_leaf4 (k_leaf_spec) with mask_roll;
//   * hash_fast.cuh (benchmarks/pouw/pearl_c_sm120): row-seg's segments a thread from their header's state, then a fold a unit;
//     rows take one launch with their digests in shared memory, as hash_h3.cuh's s512_rows (benchmarks/pouw/pearl_c) does.
// The device functions and the plan's builders are vc_device.cuh's, header only, for kernels of other libraries to call.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "vc.h"
#include "vc_device.cuh"

namespace {

using namespace vc;
#define DEV __device__ __forceinline__

constexpr u32 MAGIC = 0x76630001u;   // "vc", ABI 1
constexpr int THREADS = 256;
constexpr int NODE_THREADS = 256;
constexpr int T8_BYTES = 256 * 17 * 16;
constexpr int SEG = 1024;
constexpr int MAX_SEGS = 1024;       // a row-seg row of at most 1 MiB
constexpr int REC = 192;             // a salt record: M_key y (64), c (64), the frame leaf's first block (64)
#ifndef VC_LEAF_MIN_BLOCKS
#define VC_LEAF_MIN_BLOCKS 2
#endif

struct RowK {
  u64 mid_P[8];                      // row/v2: after P(l)
  u64 kw_xpad[80];                   // row/v2: the all-padding last block (lb % 128 == 0)
  u64 kw_segpad[80];                 // row-seg: a whole segment's padding block
  u64 kw_fold[80];                   // row-seg rows of an even s: the fold's all-padding last block
  u64 mid_Q[2][8];                   // row-seg: after Q, per shape (tiles: full tiles, the last row tile's)
  u32 scheme, lb, xpad;
  u32 s[2];                          // segments a unit, per shape
  u32 last[2];                       // the last segment's bytes, per shape
  u32 rows, cols;                    // tiles
};
struct TabK {
  uint8_t key[256];
  u32 has_key;
  u32 s[2];
  u32 last_bits[2];
};
struct SeedK {
  u64 mid[8];                        // after the seed-A label block
};
struct ArK {                         // the arity what-if: node(level, index, c_0 .. c_(a-1)) from its constant first 113 bytes
  u64 head[15];                      // frame || u32be(4) || "node" || u64be(64) || domain id, zero padded
};
struct Plan {
  u32 magic, has_hm96;
  LeafK leaf;
  NodeK node;
  RowK row;
  TabK tab;
  SeedK seed;
  ArK ar;
  uint8_t empty_root[64];
  u64 tab_bytes;
  u32 has_seed;
};

// --- kernels -------------------------------------------------------------------------------------------------------

// The plan's tables: Tn (8 KB, the nibble table of T_v = xor of (key >> s) over the set bits s of v), then per shape the state
// after each segment header S(j, s, m) (row-seg).
__constant__ uint8_t SEG_TAG[33] = {'v', 'e', 'r', 'i', 't', 'y', '/', 's', 'h', 'a', '5', '1', '2', '-', 'r', 'o', 'w', '-',
                                    's', 'e', 'g', '/', 'v', '1', '/', 's', 'e', 'g', 'm', 'e', 'n', 't', 0};
__global__ void k_tables(const __grid_constant__ TabK T, uint8_t* tab) {
  const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < 512) {
    if (T.has_key) nib_entry(T.key, t, tab);
    return;
  }
  const u32 q = t - 512;
  if (q >= T.s[0] + T.s[1]) return;
  const int sh = q < T.s[0] ? 0 : 1;
  const u64 j = sh ? q - T.s[0] : q, s = T.s[sh];
  const u64 m = j + 1 < s ? 8 * SEG : T.last_bits[sh];
  uint8_t blk[128];
  for (int i = 0; i < 128; ++i) blk[i] = 0;
  for (int i = 0; i < 33; ++i) blk[i] = SEG_TAG[i];
  for (int i = 0; i < 8; ++i) {
    blk[33 + i] = (uint8_t)(j >> (56 - 8 * i));
    blk[41 + i] = (uint8_t)(s >> (56 - 8 * i));
    blk[49 + i] = (uint8_t)(m >> (56 - 8 * i));
  }
  u64 w[16], st[8];
  for (int i = 0; i < 16; ++i) {
    u64 x = 0;
    for (int k = 0; k < 8; ++k) x = x << 8 | blk[8 * i + k];
    w[i] = x;
  }
  sha::iv(st);
  sha::compress(st, w);
  u64* o = reinterpret_cast<u64*>(tab + TN_BYTES) + 8 * q;
  for (int i = 0; i < 8; ++i) o[i] = st[i];
}

__global__ void __launch_bounds__(THREADS) k_sha512(const uint8_t* __restrict__ msgs, u64 stride, u64 len, u64 n, uint8_t* __restrict__ out) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u64 cv[8], x[8];
  sha::iv(cv);
  const uint8_t* p = msgs + i * stride;
  u64 done = 0;
  while (len - done >= (1u << 30)) {   // hash_tail takes a u32 length: whole 1 GiB pieces first
    cpy8(x, cv);
#pragma unroll 1
    for (u64 b = 0; b < (1u << 30) / 128; ++b) {
      u64 W[16];
      ld_be(p + done + 128 * b, W, 16);
      sha::compress(x, W);
    }
    cpy8(cv, x);
    done += 1u << 30;
  }
  hash_tail(cv, done, p + done, (u32)(len - done), x, nullptr);
  st_be(out + 64 * i, x, 8);
}

__global__ void __launch_bounds__(THREADS) k_compress(const uint8_t* __restrict__ states, const uint8_t* __restrict__ blocks, u64 n,
                                                      uint8_t* __restrict__ out) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u64 st[8], W[16];
  ld_be(states + 64 * i, st, 8);
  ld_be(blocks + 128 * i, W, 16);
  sha::compress(st, W);
  st_be(out + 64 * i, st, 8);
}

// sha512/row/v2: a thread a row
__global__ void __launch_bounds__(THREADS, VC_LEAF_MIN_BLOCKS) k_row_v2(const __grid_constant__ RowK R, const uint8_t* __restrict__ rows,
                                                                        u64 stride, u64 n, uint8_t* __restrict__ out) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u64 x[8];
  hash_tail(R.mid_P, 128, rows + i * stride, R.lb, x, R.xpad ? R.kw_xpad : nullptr);
  st_be(out + 64 * i, x, 8);
}

DEV void seg_hash(const RowK& R, const uint8_t* tab, int sh, u32 j, const uint8_t* p, u64* dst) {
  const u32 s = R.s[sh];
  const u64* head = reinterpret_cast<const u64*>(tab + TN_BYTES) + 8 * ((sh ? R.s[0] : 0) + j);
  u64 st[8], x[8];
  ld_w(head, st, 8);
  const bool whole = j + 1 < s || R.last[sh] == SEG;
  hash_tail(st, 128, p, whole ? SEG : R.last[sh], x, whole ? R.kw_segpad : nullptr);
  st_w(dst, x, 8);
}

// row-seg: a thread a segment, segment t = row t / s, segment t % s; its digest's words to dig + 8 t
__global__ void __launch_bounds__(128, 2) k_row_segs(const __grid_constant__ RowK R, const uint8_t* __restrict__ tab,
                                                     const uint8_t* __restrict__ rows, u64 stride, u64 n, u64* __restrict__ dig) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  const u32 s = R.s[0];
  if (t >= n * s) return;
  const u64 r = t / s;
  const u32 j = (u32)(t - r * s);
  seg_hash(R, tab, 0, j, rows + r * stride + (u64)SEG * j, dig + 8 * t);
}

// Pearl-C's tiles: segment j of tile (ti, tj) is the tile's rows 2j and 2j + 1, each row's two 32-column chunks as le32 C~ ||
// le32 U; the full row tiles' tiles first (32 segments each), then the last row tile's (s = R / 2).
__global__ void __launch_bounds__(128, 2) k_tile_segs(const __grid_constant__ RowK R, const uint8_t* __restrict__ tab,
                                                      const uint8_t* __restrict__ C, const uint8_t* __restrict__ U, u64* __restrict__ dig) {
  const u32 rows = R.rows, cols = R.cols, tn = cols / 64;
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x, n_full = (u64)(rows / 64) * tn * 32;
  u32 ti, tj, j;
  int sh;
  if (t < n_full) {
    const u64 tile = t / 32;
    sh = 0, j = (u32)(t % 32), ti = (u32)(tile / tn), tj = (u32)(tile % tn);
  } else {
    const u32 s = R.s[1];
    const u64 q = t - n_full;
    if (!s || q >= (u64)tn * s) return;
    sh = 1, ti = rows / 64, tj = (u32)(q / s), j = (u32)(q % s);
  }
  const u32 s = R.s[sh];
  const u64* head = reinterpret_cast<const u64*>(tab + TN_BYTES) + 8 * ((sh ? R.s[0] : 0) + j);
  u64 st[8];
  ld_w(head, st, 8);
#pragma unroll 1
  for (int c = 0; c < SEG / 128; ++c) {
    const u64 row = 64ull * ti + 2 * j + c / 4;
    const uint8_t* src = (c & 1) ? U : C;
    u64 W[16];
    ld_be(src + 4 * (row * cols + 64 * tj + 32 * (c >> 1 & 1)), W, 16);
    sha::compress(st, W);
  }
  (void)s;
  sha::compress_kw(st, R.kw_segpad);
  st_w(dig + 8 * t, st, 8);
}

// n words from global (__ldg) or, SH, shared memory, which __ldg can't read
template <bool SH> DEV void ld_dig(const u64* p, u64* d, int n) {
  if constexpr (SH) {
#pragma unroll
    for (int i = 0; i < n; ++i) d[i] = p[i];
  } else {
    ld_w(p, d, n);
  }
}
// a unit's digest over its s segment digests (words; SH, in shared memory) from the Q state, into out (byte order); an even
// s's all-padding last block through kw_pad, its schedule from the host, when given. k_rows_seg passes it: on sm_120 its fold
// compressing that block of constants came out wrong for some rows.
template <bool SH> DEV void seg_fold(const u64* qst, const u64* dig, u32 s, uint8_t* out, const u64* kw_pad) {
  u64 st[8], w[16];
  cpy8(st, qst);
  const int pairs = (int)(s / 2);
#pragma unroll 1
  for (int k = 0; k < pairs; ++k) {
    ld_dig<SH>(dig + 16 * k, w, 16);
    sha::compress(st, w);
  }
  if (s & 1) {
    ld_dig<SH>(dig + 16 * pairs, w, 8);
#pragma unroll
    for (int t = 8; t < 16; ++t) w[t] = 0;
    w[8] = 0x8000000000000000ull;
    w[15] = 8ull * (128 + 64ull * s);
    sha::compress(st, w);
  } else if (kw_pad) {
    sha::compress_kw(st, kw_pad);
  } else {
#pragma unroll
    for (int t = 0; t < 16; ++t) w[t] = 0;
    w[0] = 0x8000000000000000ull;
    w[15] = 8ull * (128 + 64ull * s);
    sha::compress(st, w);
  }
  st_be(out, st, 8);
}
// row-seg in one launch, its segment digests in shared memory: block b hashes rows g b .. g b + g - 1, g = blockDim / s (1 past
// s = 128); segment j of its row q on thread q s + j (and j + 128 i), then thread q folds row q
__global__ void __launch_bounds__(128) k_rows_seg(const __grid_constant__ RowK R, const uint8_t* __restrict__ tab,
                                                  const uint8_t* __restrict__ rows, u64 stride, u64 n, uint8_t* __restrict__ out) {
  extern __shared__ u64 sdig[];
  const u32 s = R.s[0], g = s < blockDim.x ? blockDim.x / s : 1;
  const u64 r0 = (u64)blockIdx.x * g;
  for (u32 t = threadIdx.x; t < g * s; t += blockDim.x) {
    const u32 q = t / s, j = t - q * s;
    if (r0 + q < n) seg_hash(R, tab, 0, j, rows + (r0 + q) * stride + (u64)SEG * j, sdig + 8 * t);
  }
  __syncthreads();
  if (threadIdx.x < g && r0 + threadIdx.x < n)
    seg_fold<true>(R.mid_Q[0], sdig + 8ull * s * threadIdx.x, s, out + 64 * (r0 + threadIdx.x), R.kw_fold);
}
constexpr u32 ROWS_SEG_MAX_S = 768;   // k_rows_seg's 64 s bytes of shared memory within the default 48 KiB

// units u < n0 have s[0] segments each, the rest s[1]
__global__ void __launch_bounds__(32) k_folds(const __grid_constant__ RowK R, const u64* __restrict__ dig, u64 n_units, u64 n0,
                                              uint8_t* __restrict__ out) {
  const u64 u = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n_units) return;
  const bool head = u < n0;
  const u32 s = head ? R.s[0] : R.s[1];
  const u64 off = head ? u * R.s[0] : n0 * R.s[0] + (u - n0) * R.s[1];
  seg_fold<false>(R.mid_Q[head ? 0 : 1], dig + 8 * off, s, out + 64 * u, nullptr);
}

// Pearl-C's row seeds: H_seed-A(salt || digest || root_B || u64le(unit << 32 | row)), the first 32 bytes
__global__ void __launch_bounds__(THREADS) k_row_seeds(const __grid_constant__ SeedK S, const uint8_t* __restrict__ digests, u64 n,
                                                       const uint8_t* __restrict__ salt, const uint8_t* __restrict__ root_b, u32 unit,
                                                       uint8_t* __restrict__ out) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  u64 st[8], W[16], rb[8];
  cpy8(st, S.mid);
  ld_be(salt, W, 4);
  ld_be(digests + 64 * i, W + 4, 8);
  ld_be(root_b, rb, 8);
#pragma unroll
  for (int k = 0; k < 4; ++k) W[12 + k] = rb[k];
  sha::compress(st, W);
#pragma unroll
  for (int k = 0; k < 4; ++k) W[k] = rb[4 + k];
  const u64 idx = (u64)unit << 32 | (u32)i;
  W[4] = (u64)bswap32((u32)idx) << 32 | bswap32((u32)(idx >> 32));
  W[5] = 0x8000000000000000ull;
#pragma unroll
  for (int k = 6; k < 15; ++k) W[k] = 0;
  W[15] = (128 + 168) * 8;
  sha::compress(st, W);
  st_be(out + 32 * i, st, 4);
}

// The salt phase, a thread a position (sharaw's sr_salt8): M_key y from the byte table each CTA builds in shared memory from Tn,
// c, the frame leaf's first block; a pad for n_total <= pos. Thread i < n_ctr zeroes k_leaf_fold's counter i.
template <int SX>
__global__ void __launch_bounds__(256, 2) k_salt8(const __grid_constant__ LeafK K, const uint8_t* __restrict__ tab,
                                                  const uint8_t* __restrict__ salts, u64 first, u64 m, u64 n_total, u64* __restrict__ rec,
                                                  uint8_t* __restrict__ lv0, u32* __restrict__ ctr, u64 n_ctr) {
  extern __shared__ uint4 T8[];
  const uint4* Tn = reinterpret_cast<const uint4*>(tab);
  for (int k = threadIdx.x; k < 256 * 16; k += blockDim.x) {
    const int b = k >> 4, c = k & 15;
    const uint4 lo = Tn[16 * c + (b & 15)], hi = Tn[256 + 16 * c + (b >> 4)];
    T8[17 * b + c] = make_uint4(lo.x ^ hi.x, lo.y ^ hi.y, lo.z ^ hi.z, lo.w ^ hi.w);
  }
  __syncthreads();
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_ctr) ctr[i] = 0;
  if (i >= m) return;
  const u64 pos = first + i;
  if (pos < n_total) {
    u64 Y[24], c[8], my[8], f[8];
    u32 mk[16];
    salt_of(K, salts, i, pos, Y);
    mask8(T8, Y, mk);
    salt_digest(K, Y, c);
    mask_words(mk, my);
    u64* r = rec + (REC / 8) * i;
    st_w(r, my, 8);
    st_w(r + 8, c, 8);
    frame_first(K, pos, f);
    st_w(r + 16, f, 8);
  } else {
    u64 o[8];
    pad(K, pos, o);
    st_be(lv0 + 64 * i, o, 8);
  }
}

// The data phase from x given, or (ROWS) from the row: a thread a position below n_total.
template <int SX, bool ROWS>
__global__ void __launch_bounds__(THREADS, VC_LEAF_MIN_BLOCKS)
    k_leaf_data(const __grid_constant__ LeafK K, const uint8_t* __restrict__ src, u64 stride, u64 first, u64 m, u64 n_total,
                const u64* __restrict__ rec, uint8_t* __restrict__ lv0, uint8_t* __restrict__ bc, uint8_t* __restrict__ hs) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const u64 pos = first + i;
  if (pos >= n_total) return;
  u64 x[8], my[8], c[8], f[8];
  if constexpr (ROWS) hash_tail(K.mid_P, 128, src + i * stride, K.lb, x, K.xpad ? K.kw_xpad : nullptr);
  else ld_be(src + 64 * i, x, 8);
  const u64* r = rec + (REC / 8) * i;
  ld_w(r, my, 8);
  ld_w(r + 8, c, 8);
  ld_w(r + 16, f, 8);
  finish_leaf<SX>(K, pos, x, my, c, f, bc ? bc + 128 * i : nullptr, hs ? hs + 64 * i : nullptr);
  st_be(lv0 + 64 * i, f, 8);
}

// The data phase from the rows over the whole tree, its levels folded in (sharaw's sr_data): a CTA's cnt = min(width, THREADS)
// leaves (the salt phase's pads past n), their subtree through shared memory, then thread 0 climbs while its CTA is the second
// of two siblings to finish. ctr: a count a node above the CTAs' level (level L + 1's first), zeroed by the salt phase.
// COMPACT: the nodes through node_rt<true> (one compression body), else node (one a length of the index).
template <int SX, bool COMPACT>
__global__ void __launch_bounds__(THREADS, VC_LEAF_MIN_BLOCKS)
    k_leaf_fold(const __grid_constant__ LeafK K, const __grid_constant__ NodeK N, const uint8_t* __restrict__ rows, u64 stride,
                u64 n, u64 width, int depth, const u64* __restrict__ rec, uint8_t* levels, uint8_t* __restrict__ bc, u32* ctr) {
  __shared__ u64 sm[THREADS * 8];
  const u32 t = threadIdx.x, cnt = blockDim.x;
  const u64 pos = (u64)blockIdx.x * cnt + t;
  u64 o[8];
  if (pos < n) {
    u64 x[8], my[8], c[8];
    hash_tail(K.mid_P, 128, rows + pos * stride, K.lb, x, K.xpad ? K.kw_xpad : nullptr);
    const u64* r = rec + (REC / 8) * pos;
    ld_w(r, my, 8);
    ld_w(r + 8, c, 8);
    ld_w(r + 16, o, 8);
    finish_leaf<SX>(K, pos, x, my, c, o, bc ? bc + 128 * pos : nullptr, nullptr);
    st_be(levels + 64 * pos, o, 8);
  } else {
    ld_be(levels + 64 * pos, o, 8);
  }
  cpy8(sm + 8 * t, o);
  int L = 0;
  while ((1u << L) < cnt) ++L;
  u64 off = 0;   // level l's first node in levels
  for (int l = 1; l <= L; ++l) {
    off += width >> (l - 1);
    __syncthreads();
    const u32 m = cnt >> l;
    u64 r[8];
    if (t < m) {
      if constexpr (COMPACT) node_rt<true>(N, l - 1, (u64)blockIdx.x * m + t, sm + 16 * t, sm + 16 * t + 8, r);
      else node(N, l - 1, (u64)blockIdx.x * m + t, sm + 16 * t, sm + 16 * t + 8, r);
    }
    __syncthreads();
    if (t < m) {
      cpy8(sm + 8 * t, r);
      st_be(levels + 64 * (off + (u64)blockIdx.x * m + t), r, 8);
    }
  }
  if (t != 0 || L >= depth) return;
  u64 cur[8], idx = blockIdx.x;
  cpy8(cur, sm);
  const u64 nL = width >> L;
  for (int l = L; l < depth; ++l) {
    const u64 p = idx >> 1;
    u32* c = ctr + (nL - (width >> l)) + p;
    __threadfence();
    if (atomicAdd(c, 1u) == 0) return;   // the sibling's subtree isn't done: its CTA climbs on
    __threadfence();
    u64 sib[8], a[8], b[8], r[8];
    ld_be_cg(levels + 64 * (off + (idx ^ 1)), sib, 8);
    const bool right = idx & 1;
#pragma unroll
    for (int k = 0; k < 8; ++k) { a[k] = right ? sib[k] : cur[k]; b[k] = right ? cur[k] : sib[k]; }
    if constexpr (COMPACT) node_rt<true>(N, l, p, a, b, r);
    else node(N, l, p, a, b, r);
    off += width >> l;
    st_be(levels + 64 * (off + p), r, 8);
    cpy8(cur, r);
    idx = p;
  }
}
u64 fold_ctrs(u64 width) { return width > THREADS ? width / THREADS - 1 : 0; }

// The whole commit in one launch (sharaw's sr_leaf4, v4): a CTA of 2 cnt threads for cnt = min(width, SPEC_R) leaves. Threads
// below cnt hash the rows; the others compute, for the same leaves at the same time, the salt record (M_key y from the nibble
// table in shared memory, c, the frame leaf's first block; or the pad) into shared memory. Then the row threads finish their
// leaves, fold their warp's 32 through shuffles, and lane 0 climbs while its warp is the second of two siblings to finish.
// The frame, pad and node hashes go through the _rt helpers (one compression body each) to keep the code small.
// ctr: a count a node above level log2(min(cnt, 32)), zeroed before the launch.
constexpr int SPEC_R = 128;
template <int SX>
__global__ void __launch_bounds__(2 * SPEC_R, 2)
    k_leaf_spec(const __grid_constant__ LeafK K, const __grid_constant__ NodeK N, const uint8_t* __restrict__ tab,
                const uint8_t* __restrict__ salts, const uint8_t* __restrict__ rows, u64 stride, u64 n, u64 width, int depth,
                uint8_t* levels, uint8_t* __restrict__ bc, u32* ctr) {
  __shared__ uint4 T[512];
  __shared__ u64 SF[SPEC_R * 24];
  const u32 cnt = blockDim.x / 2, t = threadIdx.x;
  for (u32 i = t; i < 512; i += blockDim.x) T[i] = reinterpret_cast<const uint4*>(tab)[i];
  __syncthreads();
  const bool rower = t < cnt;
  const u32 j = rower ? t : t - cnt;
  const u64 pos = (u64)blockIdx.x * cnt + j;
  u64 x[8];
  if (rower) {
    if (pos < n) hash_tail(K.mid_P, 128, rows + pos * stride, K.lb, x, K.xpad ? K.kw_xpad : nullptr);
  } else {
    u64* s = SF + 24 * j;
    if (pos < n) {
      u64 Y[24], my[8], c[8], f[8];
      u32 mk[16];
      salt_of(K, salts, pos, pos, Y);
      mask_nib_s(reinterpret_cast<const uint8_t*>(T), Y, mk);
      salt_digest(K, Y, c);
      mask_words(mk, my);
      frame_first_rt(K, pos, f);
      cpy8(s, my);
      cpy8(s + 8, c);
      cpy8(s + 16, f);
    } else {
      u64 o[8];
      pad_rt(K, pos, o);
      cpy8(s + 16, o);
    }
  }
  __syncthreads();
  if (!rower) return;
  const u64* s = SF + 24 * j;
  u64 o[8];
  cpy8(o, s + 16);
  if (pos < n) {
    u64 my[8], c[8];
    cpy8(my, s);
    cpy8(c, s + 8);
    finish_leaf<SX, true>(K, pos, x, my, c, o, bc ? bc + 128 * pos : nullptr, nullptr);
  }
  st_be(levels + 64 * pos, o, 8);
  const u32 lane = t & 31, wsz = cnt < 32 ? cnt : 32;
  const unsigned mask = wsz == 32 ? 0xFFFFFFFFu : (1u << wsz) - 1;
  int L = 0;
  while ((1u << L) < wsz) ++L;
  u64 off = 0;   // level l's first node in levels
  for (int l = 1; l <= L; ++l) {
    off += width >> (l - 1);
    u64 p[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) p[k] = __shfl_xor_sync(mask, o[k], 1 << (l - 1));
    if ((lane & ((1u << l) - 1)) == 0) {
      u64 r[8];
      node_rt<false>(N, l - 1, pos >> l, o, p, r);
      cpy8(o, r);
      st_be(levels + 64 * (off + (pos >> l)), o, 8);
    }
  }
  if (lane != 0 || L >= depth) return;
  u64 idx = pos >> L;
  const u64 nL = width >> L;
  for (int l = L; l < depth; ++l) {
    const u64 p = idx >> 1;
    u32* c = ctr + (nL - (width >> l)) + p;
    __threadfence();
    if (atomicAdd(c, 1u) == 0) return;   // the sibling's subtree isn't done: its warp climbs on
    __threadfence();
    u64 sib[8], a[8], b[8], r[8];
    ld_be_cg(levels + 64 * (off + (idx ^ 1)), sib, 8);
    const bool right = idx & 1;
#pragma unroll
    for (int k = 0; k < 8; ++k) { a[k] = right ? sib[k] : o[k]; b[k] = right ? o[k] : sib[k]; }
    node_rt<false>(N, l, p, a, b, r);
    off += width >> l;
    st_be(levels + 64 * (off + p), r, 8);
    cpy8(o, r);
    idx = p;
  }
}
u64 spec_ctrs(u64 width) { return width > 32 ? width / 32 - 1 : 0; }

// The current device's compute capability major, read once a device (reading it enqueues nothing); -1 if unknown.
int device_major() {
  static int major[64];
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return -1;
  if (!major[dev]) {
    int m = 0;
    major[dev] = cudaDeviceGetAttribute(&m, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && m ? m : -1;
  }
  return major[dev];
}
// AUTO's form for n rows on the current device, from bench/'s 4 KiB rows. Up to 40,000 rows a one- or two-launch form, which
// saves SPLIT's tree launches (a fixed cost) but idles through its subtree's barriers (a cost a row): SPEC on sm_90 (H200 at
// 34,737 rows: sharaw's v4 0.80 ms, FOLD 1.00), FOLD elsewhere (B200: FOLD 0.556, SPEC 0.760; SPLIT 8% faster at 65,536).
// Above, SPEC on sm_120 up to 200,000 rows (RTX PRO 6000 at 65,536: SPEC 0.748, FUSED 0.825, SPLIT 0.883, FOLD 0.979), else
// SPLIT.
int auto_form(u64 n) {
  const int m = device_major();
  if (n <= 40000) return m == 9 ? VC_FORM_SPEC : VC_FORM_FOLD;
  return m == 12 && n <= 200000 ? VC_FORM_SPEC : VC_FORM_SPLIT;
}

// Both phases a thread (native_rows' rows_leaf_kernel): the mask from the nibble table through L1.
template <int SX, bool ROWS>
__global__ void __launch_bounds__(THREADS, VC_LEAF_MIN_BLOCKS)
    k_leaf_fused(const __grid_constant__ LeafK K, const uint8_t* __restrict__ tab, const uint8_t* __restrict__ src, u64 stride,
                 const uint8_t* __restrict__ salts, u64 first, u64 m, u64 n_total, uint8_t* __restrict__ lv0, uint8_t* __restrict__ bc,
                 uint8_t* __restrict__ hs) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const u64 pos = first + i;
  u64 o[8];
  if (pos < n_total) {
    u64 x[8], Y[24], c[8], my[8];
    u32 mk[16];
    if constexpr (ROWS) hash_tail(K.mid_P, 128, src + i * stride, K.lb, x, K.xpad ? K.kw_xpad : nullptr);
    else ld_be(src + 64 * i, x, 8);
    salt_of(K, salts, i, pos, Y);
    mask_nib(tab, Y, mk);
    salt_digest(K, Y, c);
    mask_words(mk, my);
    frame_first(K, pos, o);
    finish_leaf<SX>(K, pos, x, my, c, o, bc ? bc + 128 * i : nullptr, hs ? hs + 64 * i : nullptr);
  } else {
    pad(K, pos, o);
  }
  st_be(lv0 + 64 * i, o, 8);
}

template <int SX>
__global__ void __launch_bounds__(THREADS) k_frame_leaves(const __grid_constant__ LeafK K, const uint8_t* __restrict__ values, u64 first,
                                                          u64 m, u64 n_total, uint8_t* __restrict__ out) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const u64 pos = first + i;
  u64 o[8];
  if (pos < n_total) {
    u64 v[8];
    ld_be(values + 64 * i, v, 8);
    frame_leaf<SX>(K, pos, v, o);
  } else {
    pad(K, pos, o);
  }
  st_be(out + 64 * i, o, 8);
}

constexpr u64 OPEN_HEAD = 16 + 192 + 128 + 64;   // position, leaf offset, y, b || c, the leaf; the path follows

DEV void cp16(uint8_t* d, const uint8_t* s, int n16) {
  for (int k = 0; k < n16; ++k) reinterpret_cast<uint4*>(d)[k] = ldg16(s + 16 * k);
}
DEV void zero16(uint8_t* d, u64 n16) {
  for (u64 k = 0; k < n16; ++k) reinterpret_cast<uint4*>(d)[k] = make_uint4(0, 0, 0, 0);
}

// An opening a thread (vc_open_launch's record): the salt from the plan's source, b || c, the leaf and its path from the levels.
__global__ void __launch_bounds__(64) k_open(const __grid_constant__ LeafK K, int hm, const uint8_t* __restrict__ tab,
                                             const uint8_t* __restrict__ x, const uint8_t* __restrict__ bc,
                                             const uint8_t* __restrict__ salts, const uint8_t* __restrict__ levels, u64 n, int depth,
                                             const u32* __restrict__ positions, u64 m, uint8_t* __restrict__ out) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const u64 pos = positions[i], rec = OPEN_HEAD + 64ull * depth;
  uint8_t* r = out + rec * i;
  reinterpret_cast<u64*>(r)[0] = pos;
  reinterpret_cast<u64*>(r)[1] = pos < n ? 64 * pos : ~0ull;
  if (pos >= n) {
    zero16(r + 16, (rec - 16) / 16);
    return;
  }
  if (hm) {
    u64 Y[24];
    salt_of(K, salts, pos, pos, Y);
    st_be(r + 16, Y, 24);
    if (bc) {
      cp16(r + 208, bc + 128 * pos, 8);
    } else if (x) {
      u64 xs[8], my[8], c[8], B[16];
      u32 mk[16];
      ld_be(x + 64 * pos, xs, 8);
      mask_nib(tab, Y, mk);
      salt_digest(K, Y, c);
      mask_words(mk, my);
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        B[w] = xs[w] ^ my[w];
        B[8 + w] = c[w];
      }
      st_be(r + 208, B, 16);
    } else {
      zero16(r + 208, 8);
    }
  } else {
    zero16(r + 16, 20);
  }
  cp16(r + 336, levels + 64 * pos, 4);
  u64 off = 0, w = 1ull << depth, cur = pos;
  for (int l = 0; l < depth; ++l) {
    cp16(r + OPEN_HEAD + 64 * l, levels + 64 * (off + (cur ^ 1)), 4);
    off += w;
    w >>= 1;
    cur >>= 1;
  }
}

// level `level` + 1 from level `level`, a thread a parent
__global__ void __launch_bounds__(NODE_THREADS) k_level(const __grid_constant__ NodeK N, const uint8_t* __restrict__ in,
                                                        uint8_t* __restrict__ out, u64 n_out, int level, u64 first_index) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  u64 L[8], R[8], o[8];
  ld_be(in + 128 * i, L, 8);
  ld_be(in + 128 * i + 64, R, 8);
  node(N, level, first_index + i, L, R, o);
  st_be(out + 64 * i, o, 8);
}

// n_levels levels above n_in nodes of level `level`, each level right after the previous one in out; block b folds its own
// slice of 2^n_levels nodes (native_rows' rows_levels_kernel)
__global__ void __launch_bounds__(NODE_THREADS) k_levels(const __grid_constant__ NodeK N, const uint8_t* in, uint8_t* out, u64 n_in,
                                                         int level, int n_levels) {
  const u64 first = (u64)blockIdx.x << n_levels;
  in += 64 * first;
  for (int k = 0; k < n_levels; ++k) {
    const u64 n_out = n_in / 2, len_out = 1ull << (n_levels - k - 1), at = first >> (k + 1);
    uint8_t* dst = out + 64 * at;
    for (u64 i = threadIdx.x; i < len_out; i += blockDim.x) {
      u64 L[8], R[8], o[8];
      ld_be_written(in + 128 * i, L, 8);
      ld_be_written(in + 128 * i + 64, R, 8);
      node(N, level + k, at + i, L, R, o);
      st_be(dst + 64 * i, o, 8);
    }
    __syncthreads();
    in = dst;
    out += 64 * n_out;
    n_in = n_out;
  }
}

__global__ void k_put64(uint8_t* dst, ulonglong4 a, ulonglong4 b) {
  if (threadIdx.x == 0) {
    u64 w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    for (int i = 0; i < 64; ++i) dst[i] = (uint8_t)(w[i / 8] >> (8 * (i % 8)));
  }
}

// vc_chunk_rows_launch: a thread a 16 bytes of a leaf (kc16 of codes, then the scale's, then zeros to the stride)
__global__ void __launch_bounds__(THREADS) k_chunk_rows(const uint4* __restrict__ codes, u64 cs16, const u32* __restrict__ scales,
                                                        u64 n_leaves, u32 chunks, u32 kc16, uint4* __restrict__ out, u64 os16) {
  const u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_leaves * os16) return;
  const u64 j = t / os16, w = t % os16, r = j / chunks;
  uint4 v = make_uint4(0, 0, 0, 0);
  if (w < kc16) v = codes[r * cs16 + (j % chunks) * kc16 + w];
  else if (w == kc16) v.x = scales[r];
  out[t] = v;
}

// The arity what-if: a thread a parent, its message assembled byte by byte (a slow path: it is not served)
DEV void put_byte(u64* W, u32 off, u32 b) { W[off >> 3] |= (u64)(b & 0xFF) << (56 - 8 * (off & 7)); }
DEV void put_be(u64* W, u32 off, u64 v, int n) {
  for (int k = 0; k < n; ++k) put_byte(W, off + k, (u32)(v >> (8 * (n - 1 - k))));
}
__global__ void __launch_bounds__(128) k_level_arity(const __grid_constant__ ArK A, const uint8_t* __restrict__ in, uint8_t* __restrict__ out,
                                                     u64 n_out, int level, int arity) {
  const u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_out) return;
  u64 W[16 * 6];
  for (int k = 0; k < 16 * 6; ++k) W[k] = 0;
  for (int k = 0; k < 14; ++k) W[k] = A.head[k];
  W[14] = A.head[14];                    // byte 112: the domain id's last
  const int nl = nbytes((u64)level), ni = nbytes(i);
  u32 off = 113;
  put_be(W, off, (u64)nl, 8); off += 8;
  put_be(W, off, (u64)level, nl); off += nl;
  put_be(W, off, (u64)ni, 8); off += 8;
  put_be(W, off, i, ni); off += ni;
  for (int c = 0; c < arity; ++c) {
    put_be(W, off, 64, 8); off += 8;
    const uint8_t* ch = in + 64 * ((u64)arity * i + c);
    for (int k = 0; k < 64; ++k) put_byte(W, off + k, ch[k]);
    off += 64;
  }
  const u32 len = off, nblk = (len + 17 + 127) / 128;
  put_byte(W, len, 0x80);
  W[16 * nblk - 1] = (u64)len * 8;
  u64 st[8];
  sha::iv(st);
  for (u32 b = 0; b < nblk; ++b) {
    u64 w[16];
    for (int k = 0; k < 16; ++k) w[k] = W[16 * b + k];
    sha::compress(st, w);
  }
  st_be(out + 64 * i, st, 8);
}

// --- host ----------------------------------------------------------------------------------------------------------

const char FRAME[] = "veritor/protocol/merkle/frame/v3";  // + its NUL: 33 bytes

// merkle._message: frame || u32be(len tag) || tag || (u64be(len part) || part)...
std::vector<uint8_t> message(const char* tag, const std::vector<std::vector<uint8_t>>& parts) {
  std::vector<uint8_t> m(FRAME, FRAME + sizeof FRAME);
  const size_t tl = strlen(tag);
  for (int i = 3; i >= 0; --i) m.push_back((uint8_t)(tl >> (8 * i)));
  m.insert(m.end(), tag, tag + tl);
  for (const auto& p : parts) {
    for (int i = 7; i >= 0; --i) m.push_back((uint8_t)((u64)p.size() >> (8 * i)));
    m.insert(m.end(), p.begin(), p.end());
  }
  return m;
}
void words_of(const uint8_t* block, u64* w, int n) { for (int i = 0; i < n; ++i) w[i] = sha::be64(block + 8 * i); }
u32 le32(const uint8_t* p) { return (u32)p[0] | (u32)p[1] << 8 | (u32)p[2] << 16 | (u32)p[3] << 24; }

const Plan* plan_of(const void* p) {
  const Plan* q = static_cast<const Plan*>(p);
  return q && q->magic == MAGIC ? q : nullptr;
}
unsigned blocks(u64 n, u64 t) { return (unsigned)((n + t - 1) / t); }
int depth_of(u64 n) { int d = 0; while ((1ull << d) < n) ++d; return d; }
int rc() { return (int)cudaGetLastError(); }

}  // namespace

extern "C" {

int vc_plan_bytes(void) { return (int)sizeof(Plan); }

int vc_plan_build(const vc_spec* sp, void* out) {
  if (!sp || !out || !sp->domain_id) return -1;
  Plan* p = new Plan();
  memset(p, 0, sizeof *p);
  p->magic = MAGIC;
  LeafK& c = p->leaf;
  NodeK& n = p->node;
  RowK& r = p->row;
  const std::vector<uint8_t> did(sp->domain_id, sp->domain_id + 64);
  // the frame
  if (sp->schema && (sp->schema_len == 18 || sp->schema_len == 22)) {
    if (leaf_frame(sp->domain_id, reinterpret_cast<const uint8_t*>(sp->schema), sp->schema_len, c)) {
      delete p;
      return -1;
    }
  } else if (sp->schema) {
    delete p;
    return -1;
  }
  pad_tpls(sp->domain_id, c);
  if (node_tpls(n.node_tpl)) { delete p; return -1; }
  node_mids(sp->domain_id, n.node_mid, LEVELS);
  {
    const auto e = message("empty", {did});
    sha::digest(e.data(), e.size(), p->empty_root);
    const auto h = message("node", {did});
    std::vector<uint8_t> head(h.begin(), h.begin() + 113);
    head.resize(120, 0);
    words_of(head.data(), p->ar.head, 15);
  }
  // hm96
  if (sp->salt_prefix && sp->leaf_prefix && sp->key) {
    p->has_hm96 = 1;
    midstate(sp->salt_prefix, c.mid_salt);
    midstate(sp->leaf_prefix, c.mid_leaf);
    pad_schedule(256, c.kw_leafpad);
    memcpy(p->tab.key, sp->key, 256);
    p->tab.has_key = 1;
    c.salt_src = (u32)sp->salt_source;
    if (sp->salt_source == VC_SALT_PRF) {
      if (!sp->salt_key) { delete p; return -1; }
      words_of(sp->salt_key, c.ks, 8);
      u64 blk[16] = {}, st[8];
      memcpy(blk, c.ks, 64);
      sha::iv(st);
      sha::rounds_prefix(st, blk, 8, c.ks_m8);
    } else if (sp->salt_source == VC_SALT_CHACHA64 || sp->salt_source == VC_SALT_CHACHA_RFC) {
      if (!sp->salt_key) { delete p; return -1; }
      for (int i = 0; i < 8; ++i) c.cc_key[i] = le32(sp->salt_key + 4 * i);
      if (sp->salt_source == VC_SALT_CHACHA64) {
        c.cc_ctr64 = 1;
        c.cc_w14 = (u32)sp->salt_nonce;
        c.cc_w15 = (u32)(sp->salt_nonce >> 32);
      } else {
        c.cc_w13 = (u32)sp->salt_nonce;
        c.cc_w14 = (u32)(sp->salt_nonce >> 32);
        c.cc_w15 = 0;
      }
    } else if (sp->salt_source != VC_SALT_GIVEN) {
      delete p;
      return -1;
    }
  }
  // rows
  r.scheme = (u32)sp->row_scheme;
  pad_schedule(128 + SEG, r.kw_segpad);
  if (sp->row_scheme == VC_ROW_V2 || sp->row_scheme == VC_ROW_SEG) {
    if (!sp->row_prefix || !sp->row_bits || sp->row_bits >= (1ull << 32)) { delete p; return -1; }
    r.lb = (u32)((sp->row_bits + 7) / 8);
    if (sp->row_scheme == VC_ROW_V2) {
      midstate(sp->row_prefix, r.mid_P);
      r.xpad = r.lb % 128 == 0;
      if (r.xpad) pad_schedule(128 + (u64)r.lb, r.kw_xpad);
      memcpy(c.mid_P, r.mid_P, 64);
      memcpy(c.kw_xpad, r.kw_xpad, sizeof c.kw_xpad);
      c.lb = r.lb;
      c.xpad = r.xpad;
    } else {
      const u64 s = (sp->row_bits + 8 * SEG - 1) / (8 * SEG);
      if (s > MAX_SEGS) { delete p; return -1; }
      r.s[0] = (u32)s;
      r.last[0] = (u32)(r.lb - SEG * (s - 1));
      p->tab.s[0] = (u32)s;
      p->tab.last_bits[0] = (u32)(sp->row_bits - 8ull * SEG * (s - 1));
      midstate(sp->row_prefix, r.mid_Q[0]);
      if (s % 2 == 0) pad_schedule(128 + 64 * s, r.kw_fold);
    }
  } else if (sp->row_scheme == VC_ROW_TILE) {
    if (!sp->row_prefix || sp->tile_rows % 2 || sp->tile_cols % 64 || !sp->tile_cols) { delete p; return -1; }
    r.rows = sp->tile_rows;
    r.cols = sp->tile_cols;
    r.s[0] = 32;
    r.s[1] = sp->tile_rows % 64 / 2;
    r.last[0] = r.last[1] = SEG;
    p->tab.s[0] = r.s[0];
    p->tab.s[1] = r.s[1];
    p->tab.last_bits[0] = p->tab.last_bits[1] = 8 * SEG;
    midstate(sp->row_prefix, r.mid_Q[0]);
    if (r.s[1]) {
      if (!sp->row_prefix_b) { delete p; return -1; }
      midstate(sp->row_prefix_b, r.mid_Q[1]);
    }
  } else if (sp->row_scheme != VC_ROW_NONE) {
    delete p;
    return -1;
  }
  if (sp->seed_label) {
    p->has_seed = 1;
    midstate(sp->seed_label, p->seed.mid);
  }
  p->tab_bytes = TN_BYTES + 64ull * (p->tab.s[0] + p->tab.s[1]);
  memcpy(out, p, sizeof *p);
  delete p;
  return 0;
}

uint64_t vc_tables_bytes(const void* plan) {
  const Plan* p = plan_of(plan);
  return p ? p->tab_bytes : 0;
}

int vc_tables_launch(const void* plan, void* tables, cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || !tables) return -1;
  const u64 n = 512 + p->tab.s[0] + p->tab.s[1];
  k_tables<<<blocks(n, 128), 128, 0, st>>>(p->tab, static_cast<uint8_t*>(tables));
  return rc();
}

int vc_empty_root(const void* plan, uint8_t* out64) {
  const Plan* p = plan_of(plan);
  if (!p || !out64) return -1;
  memcpy(out64, p->empty_root, 64);
  return 0;
}

int vc_init(void) {
  cudaError_t e = cudaFuncSetAttribute(k_salt8<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, T8_BYTES);
  if (!e) e = cudaFuncSetAttribute(k_salt8<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, T8_BYTES);
  return (int)e;
}

int vc_sha512_launch(const uint8_t* msgs, uint64_t stride, uint64_t len, uint64_t n, uint8_t* out, cudaStream_t st) {
  if (!n) return 0;
  k_sha512<<<blocks(n, THREADS), THREADS, 0, st>>>(msgs, stride, len, n, out);
  return rc();
}

int vc_compress_launch(const uint8_t* states, const uint8_t* blks, uint64_t n, uint8_t* out, cudaStream_t st) {
  if (!n) return 0;
  k_compress<<<blocks(n, THREADS), THREADS, 0, st>>>(states, blks, n, out);
  return rc();
}

int vc_chunk_rows_launch(const uint8_t* codes, uint64_t codes_stride, const uint8_t* scales, uint64_t n_rows, uint32_t k,
                         uint32_t chunks, uint8_t* out, uint64_t out_stride, cudaStream_t st) {
  if (!chunks || k % chunks || (k / chunks) % 16 || codes_stride % 16 || codes_stride < k || out_stride % 16 ||
      out_stride < k / chunks + 4 || (reinterpret_cast<uintptr_t>(codes) | reinterpret_cast<uintptr_t>(out)) % 16 ||
      reinterpret_cast<uintptr_t>(scales) % 4)
    return -1;
  if (!n_rows) return 0;
  const u64 n = n_rows * chunks, os16 = out_stride / 16;
  k_chunk_rows<<<blocks(n * os16, THREADS), THREADS, 0, st>>>(reinterpret_cast<const uint4*>(codes), codes_stride / 16,
                                                             reinterpret_cast<const u32*>(scales), n, chunks, k / chunks / 16,
                                                             reinterpret_cast<uint4*>(out), os16);
  return rc();
}

uint64_t vc_rows_scratch_bytes(const void* plan, uint64_t n_rows) {
  const Plan* p = plan_of(plan);
  if (!p) return 0;
  return p->row.scheme == VC_ROW_SEG ? 64ull * p->row.s[0] * n_rows : 0;
}

int vc_rows_launch(const void* plan, const void* tables, const uint8_t* rows, uint64_t stride, uint64_t n_rows, uint8_t* out,
                   void* scratch, cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p) return -1;
  if (!n_rows) return 0;
  if (p->row.scheme == VC_ROW_V2) {
    k_row_v2<<<blocks(n_rows, THREADS), THREADS, 0, st>>>(p->row, rows, stride, n_rows, out);
    return rc();
  }
  if (p->row.scheme != VC_ROW_SEG || !tables || !scratch) return -1;
  const u32 s = p->row.s[0];
  if (s <= ROWS_SEG_MAX_S) {
    const u32 threads = s < 128 ? (s + 31) / 32 * 32 : 128, g = s < threads ? threads / s : 1;
    k_rows_seg<<<blocks(n_rows, g), threads, 64ull * g * s, st>>>(p->row, static_cast<const uint8_t*>(tables), rows, stride, n_rows, out);
    return rc();
  }
  u64* dig = static_cast<u64*>(scratch);
  k_row_segs<<<blocks(n_rows * p->row.s[0], 128), 128, 0, st>>>(p->row, static_cast<const uint8_t*>(tables), rows, stride, n_rows, dig);
  if (int e = rc()) return e;
  k_folds<<<blocks(n_rows, 32), 32, 0, st>>>(p->row, dig, n_rows, n_rows, out);
  return rc();
}

uint64_t vc_tiles_count(const void* plan) {
  const Plan* p = plan_of(plan);
  if (!p || p->row.scheme != VC_ROW_TILE) return 0;
  return (u64)((p->row.rows + 63) / 64) * (p->row.cols / 64);
}

uint64_t vc_tiles_scratch_bytes(const void* plan) {
  const Plan* p = plan_of(plan);
  if (!p || p->row.scheme != VC_ROW_TILE) return 0;
  const u64 tn = p->row.cols / 64;
  return 64ull * ((u64)(p->row.rows / 64) * tn * 32 + tn * p->row.s[1]);
}

int vc_tiles_launch(const void* plan, const void* tables, const uint8_t* c, const uint8_t* u, uint8_t* out, void* scratch,
                    cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || p->row.scheme != VC_ROW_TILE || !tables || !scratch) return -1;
  const u64 tn = p->row.cols / 64, n_full = (u64)(p->row.rows / 64) * tn, segs = n_full * 32 + tn * p->row.s[1];
  if (!segs) return 0;
  u64* dig = static_cast<u64*>(scratch);
  k_tile_segs<<<blocks(segs, 128), 128, 0, st>>>(p->row, static_cast<const uint8_t*>(tables), c, u, dig);
  if (int e = rc()) return e;
  const u64 units = vc_tiles_count(plan);
  k_folds<<<blocks(units, 32), 32, 0, st>>>(p->row, dig, units, n_full, out);
  return rc();
}

int vc_row_seeds_launch(const void* plan, const uint8_t* digests, uint64_t n, const uint8_t* salt32, const uint8_t* root_b64,
                        uint32_t unit, uint8_t* out32, cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || !p->has_seed || n > (1ull << 32)) return -1;
  if (!n) return 0;
  k_row_seeds<<<blocks(n, THREADS), THREADS, 0, st>>>(p->seed, digests, n, salt32, root_b64, unit, out32);
  return rc();
}

uint64_t vc_tree_width(uint64_t n_leaves) { return n_leaves ? 1ull << depth_of(n_leaves) : 0; }
uint64_t vc_tree_bytes(uint64_t n_leaves) { return n_leaves ? 64 * (2 * vc_tree_width(n_leaves) - 1) : 64; }
uint64_t vc_salt_scratch_bytes(uint64_t m) { return REC * m; }

int vc_salts_launch(const void* plan, const void* tables, const uint8_t* salts, uint64_t first, uint64_t m, uint64_t n_total,
                    void* scratch, uint8_t* lv0, cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || !p->has_hm96 || !tables || first + m > (1ull << 32)) return -1;
  if (!m) return 0;
  const auto t = static_cast<const uint8_t*>(tables);
  u64* r = static_cast<u64*>(scratch);
  if (p->leaf.sx) k_salt8<4><<<blocks(m, 256), 256, T8_BYTES, st>>>(p->leaf, t, salts, first, m, n_total, r, lv0, nullptr, 0);
  else k_salt8<0><<<blocks(m, 256), 256, T8_BYTES, st>>>(p->leaf, t, salts, first, m, n_total, r, lv0, nullptr, 0);
  return rc();
}

int vc_leaves_x_launch(const void* plan, const uint8_t* x, uint64_t first, uint64_t m, uint64_t n_total, const void* scratch,
                       uint8_t* lv0, uint8_t* bc, uint8_t* hs, cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || !p->has_hm96 || first + m > (1ull << 32)) return -1;
  if (!m) return 0;
  const u64* r = static_cast<const u64*>(scratch);
  if (p->leaf.sx) k_leaf_data<4, false><<<blocks(m, THREADS), THREADS, 0, st>>>(p->leaf, x, 64, first, m, n_total, r, lv0, bc, hs);
  else k_leaf_data<0, false><<<blocks(m, THREADS), THREADS, 0, st>>>(p->leaf, x, 64, first, m, n_total, r, lv0, bc, hs);
  return rc();
}

int vc_leaves_rows_launch(const void* plan, const uint8_t* rows, uint64_t stride, uint64_t first, uint64_t m, uint64_t n_total,
                          const void* scratch, uint8_t* lv0, uint8_t* bc, uint8_t* hs, cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || !p->has_hm96 || p->row.scheme != VC_ROW_V2 || first + m > (1ull << 32)) return -1;
  if (!m) return 0;
  const u64* r = static_cast<const u64*>(scratch);
  if (p->leaf.sx) k_leaf_data<4, true><<<blocks(m, THREADS), THREADS, 0, st>>>(p->leaf, rows, stride, first, m, n_total, r, lv0, bc, hs);
  else k_leaf_data<0, true><<<blocks(m, THREADS), THREADS, 0, st>>>(p->leaf, rows, stride, first, m, n_total, r, lv0, bc, hs);
  return rc();
}

int vc_leaves_fused_launch(const void* plan, const void* tables, const uint8_t* rows, uint64_t stride, const uint8_t* x,
                           const uint8_t* salts, uint64_t first, uint64_t m, uint64_t n_total, uint8_t* lv0, uint8_t* bc,
                           uint8_t* hs, cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || !p->has_hm96 || !tables || first + m > (1ull << 32) || (rows && p->row.scheme != VC_ROW_V2)) return -1;
  if (!m) return 0;
  const auto t = static_cast<const uint8_t*>(tables);
  const unsigned g = blocks(m, THREADS);
  if (rows) {
    if (p->leaf.sx) k_leaf_fused<4, true><<<g, THREADS, 0, st>>>(p->leaf, t, rows, stride, salts, first, m, n_total, lv0, bc, hs);
    else k_leaf_fused<0, true><<<g, THREADS, 0, st>>>(p->leaf, t, rows, stride, salts, first, m, n_total, lv0, bc, hs);
  } else {
    if (p->leaf.sx) k_leaf_fused<4, false><<<g, THREADS, 0, st>>>(p->leaf, t, x, 64, salts, first, m, n_total, lv0, bc, hs);
    else k_leaf_fused<0, false><<<g, THREADS, 0, st>>>(p->leaf, t, x, 64, salts, first, m, n_total, lv0, bc, hs);
  }
  return rc();
}

int vc_frame_leaves_launch(const void* plan, const uint8_t* values, uint64_t first, uint64_t m, uint64_t n_total, uint8_t* out,
                           cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || first + m > (1ull << 32)) return -1;
  if (!m) return 0;
  if (p->leaf.sx) k_frame_leaves<4><<<blocks(m, THREADS), THREADS, 0, st>>>(p->leaf, values, first, m, n_total, out);
  else k_frame_leaves<0><<<blocks(m, THREADS), THREADS, 0, st>>>(p->leaf, values, first, m, n_total, out);
  return rc();
}

int vc_nodes_launch(const void* plan, const uint8_t* in, uint8_t* out, uint64_t n_out, int level, uint64_t first_index,
                    cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || level < 0 || level + 1 >= LEVELS || first_index + n_out > (1ull << 32)) return -1;
  if (!n_out) return 0;
  k_level<<<blocks(n_out, NODE_THREADS), NODE_THREADS, 0, st>>>(p->node, in, out, n_out, level, first_index);
  return rc();
}

// commit_gpu's schedule: a level a launch while the level is wider than 512 nodes, then one block for the rest
int vc_tree_launch(const void* plan, uint8_t* levels, uint64_t n_leaves, cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || n_leaves > (1ull << 32)) return -1;
  if (!n_leaves) {
    u64 w[8];
    for (int i = 0; i < 8; ++i) {
      w[i] = 0;
      for (int k = 0; k < 8; ++k) w[i] |= (u64)p->empty_root[8 * i + k] << (8 * k);
    }
    k_put64<<<1, 32, 0, st>>>(levels, make_ulonglong4(w[0], w[1], w[2], w[3]), make_ulonglong4(w[4], w[5], w[6], w[7]));
    return rc();
  }
  const int depth = depth_of(n_leaves);
  const u64 width = 1ull << depth;
  u64 off = 0;
  int d = 0;
  for (; d < depth && (width >> d) > 2 * NODE_THREADS; ++d) {
    const u64 n_in = width >> d, n_out = n_in / 2;
    const unsigned t = n_out >= (1u << 16) ? NODE_THREADS : 64;
    k_level<<<blocks(n_out, t), t, 0, st>>>(p->node, levels + 64 * off, levels + 64 * (off + n_in), n_out, d, 0);
    if (int e = rc()) return e;
    off += n_in;
  }
  if (d < depth) {
    const u64 n_in = width >> d;
    k_levels<<<1, NODE_THREADS, 0, st>>>(p->node, levels + 64 * off, levels + 64 * (off + n_in), n_in, d, depth - d);
    return rc();
  }
  return 0;
}

uint64_t vc_tree_arity_width(uint64_t n_leaves, int arity) {
  if (!n_leaves) return 0;
  u64 w = 1;
  while (w < n_leaves) w *= (u64)arity;
  return w;
}
uint64_t vc_tree_arity_bytes(uint64_t n_leaves, int arity) {
  if (!n_leaves) return 64;
  u64 total = 0;
  for (u64 w = vc_tree_arity_width(n_leaves, arity); w >= 1; w /= (u64)arity) {
    total += w;
    if (w == 1) break;
  }
  return 64 * total;
}

int vc_tree_arity_launch(const void* plan, uint8_t* levels, uint64_t n_leaves, int arity, cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || (arity != 2 && arity != 4 && arity != 8) || n_leaves > (1ull << 32)) return -1;
  if (!n_leaves) return vc_tree_launch(plan, levels, 0, st);
  u64 w = vc_tree_arity_width(n_leaves, arity), off = 0;
  for (int level = 0; w > 1; ++level) {
    const u64 n_out = w / (u64)arity;
    k_level_arity<<<blocks(n_out, 128), 128, 0, st>>>(p->ar, levels + 64 * off, levels + 64 * (off + w), n_out, level, arity);
    if (int e = rc()) return e;
    off += w;
    w = n_out;
  }
  return 0;
}

uint64_t vc_commit_scratch_bytes(uint64_t n) { return REC * n + (4 * spec_ctrs(vc_tree_width(n)) + 15) / 16 * 16; }

int vc_commit_rows_launch(const void* plan, const void* tables, const uint8_t* rows, uint64_t stride, const uint8_t* salts,
                          uint64_t n, uint8_t* levels, uint8_t* bc, void* scratch, int form, cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || !p->has_hm96 || p->row.scheme != VC_ROW_V2 || n > (1ull << 32)) return -1;
  if (form < VC_FORM_AUTO || form > VC_FORM_SPEC) return -1;
  if (!n) return vc_tree_launch(plan, levels, 0, st);
  const u64 width = vc_tree_width(n);
  if (form == VC_FORM_AUTO) form = auto_form(n);
  if (form == VC_FORM_SPEC) {
    if (!tables || !scratch) return -1;
    const auto t = static_cast<const uint8_t*>(tables);
    u32* ctr = reinterpret_cast<u32*>(static_cast<uint8_t*>(scratch) + REC * n);
    if (const u64 nc = spec_ctrs(width))
      if (const cudaError_t e = cudaMemsetAsync(ctr, 0, 4 * nc, st)) return static_cast<int>(e);
    const u32 cnt = (u32)(width < SPEC_R ? width : SPEC_R);
    const int depth = depth_of(n);
    if (p->leaf.sx)
      k_leaf_spec<4><<<(unsigned)(width / cnt), 2 * cnt, 0, st>>>(p->leaf, p->node, t, salts, rows, stride, n, width, depth, levels, bc, ctr);
    else
      k_leaf_spec<0><<<(unsigned)(width / cnt), 2 * cnt, 0, st>>>(p->leaf, p->node, t, salts, rows, stride, n, width, depth, levels, bc, ctr);
    return rc();
  }
  if (form == VC_FORM_FOLD) {
    if (!tables || !scratch) return -1;
    const auto t = static_cast<const uint8_t*>(tables);
    u64* r = static_cast<u64*>(scratch);
    u32* ctr = reinterpret_cast<u32*>(static_cast<uint8_t*>(scratch) + REC * n);
    const u64 nc = fold_ctrs(width);
    const u32 cnt = (u32)(width < THREADS ? width : THREADS);
    const int depth = depth_of(n);
    if (p->leaf.sx) {
      k_salt8<4><<<blocks(width, 256), 256, T8_BYTES, st>>>(p->leaf, t, salts, 0, width, n, r, levels, ctr, nc);
    } else {
      k_salt8<0><<<blocks(width, 256), 256, T8_BYTES, st>>>(p->leaf, t, salts, 0, width, n, r, levels, ctr, nc);
    }
    if (int e = rc()) return e;
    // the compact node on sm_100 only: at 34,737 rows sharaw's v1 (by length) against v1n (compact) is 0.563 vs 0.555 ms on
    // B200, 0.502 vs 0.520 on RTX PRO 6000 and 0.877 vs 0.973 on H200
    const bool compact = device_major() == 10;
    const auto fold = p->leaf.sx ? (compact ? k_leaf_fold<4, true> : k_leaf_fold<4, false>)
                                 : (compact ? k_leaf_fold<0, true> : k_leaf_fold<0, false>);
    fold<<<(unsigned)(width / cnt), cnt, 0, st>>>(p->leaf, p->node, rows, stride, n, width, depth, r, levels, bc, ctr);
    return rc();
  }
  int e;
  if (form == VC_FORM_FUSED) {
    e = vc_leaves_fused_launch(plan, tables, rows, stride, nullptr, salts, 0, width, n, levels, bc, nullptr, st);
  } else {
    e = vc_salts_launch(plan, tables, salts, 0, width, n, scratch, levels, st);
    if (!e) e = vc_leaves_rows_launch(plan, rows, stride, 0, n, n, scratch, levels, bc, nullptr, st);
  }
  if (e) return e;
  return vc_tree_launch(plan, levels, n, st);
}

uint64_t vc_open_bytes(uint64_t n_leaves) { return OPEN_HEAD + 64ull * (n_leaves ? depth_of(n_leaves) : 0); }

int vc_open_launch(const void* plan, const void* tables, const uint8_t* x, const uint8_t* bc, const uint8_t* salts,
                   const uint8_t* levels, uint64_t n_leaves, const uint32_t* positions, uint64_t m, uint8_t* out, cudaStream_t st) {
  const Plan* p = plan_of(plan);
  if (!p || !levels || !positions || !out || n_leaves > (1ull << 32)) return -1;
  // an empty tree has no position to open: every record comes back outside, and nothing but the positions is read
  if (n_leaves && p->has_hm96 && p->leaf.salt_src == VC_SALT_GIVEN && !salts) return -1;
  if (n_leaves && p->has_hm96 && !bc && x && !tables) return -1;
  if (!m) return 0;
  const int depth = n_leaves ? depth_of(n_leaves) : 0;
  k_open<<<blocks(m, 64), 64, 0, st>>>(p->leaf, p->has_hm96, static_cast<const uint8_t*>(tables), x, bc, salts, levels, n_leaves,
                                       depth, positions, m, out);
  return rc();
}

}  // extern "C"

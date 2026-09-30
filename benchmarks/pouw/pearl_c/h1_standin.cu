// The decode GEMM's memory traffic, for timing -h1's hashing around it on sm_120 (h1_bench.py). It is a stand-in, not a GEMM:
// it reads the n x k FP8 weights once (16 bytes a load, a warp per weight row) and writes C~, U (m x n FP32 words) and y
// (m x n BF16), each a function of the weight row read and of A's rows, so that every output depends on both; and
// derive_rows, the next call's A (m x k BF16, k = n) from y, which makes the calls a dependent chain.
#include "hash_h2.cuh"

// p0 W (n x k bytes), p1 A (m x k BF16), p2 C~ (m x n words), p3 U, p4 y (m x n BF16); i0 m, i1 n, i2 k
extern "C" __global__ void __launch_bounds__(256) standin_gemm(const Args a) {
  const int m = a.i[0], n = a.i[1], k = a.i[2];
  const int j = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (j >= n) return;
  const uint4* w = reinterpret_cast<const uint4*>(a.p[0] + (uint64_t)j * k);
  uint32_t acc = 0;
  for (int q = lane; q < k / 16; q += 32) {
    const uint4 v = w[q];
    acc ^= (v.x * 0x9E3779B1u + v.y) ^ (v.z * 0x85EBCA77u + v.w);
  }
  for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
  const uint32_t* A = reinterpret_cast<const uint32_t*>(a.p[1]);
  for (int r = lane; r < m; r += 32) {
    const uint32_t x = acc ^ A[(uint64_t)r * (k / 2) + (j % (k / 2))] ^ (uint32_t)r * 0x27D4EB2Du;
    reinterpret_cast<uint32_t*>(a.p[2])[(uint64_t)r * n + j] = x;
    reinterpret_cast<uint32_t*>(a.p[3])[(uint64_t)r * n + j] = x * 0x165667B1u;
    reinterpret_cast<uint16_t*>(a.p[4])[(uint64_t)r * n + j] = (uint16_t)(x >> 16);
  }
}

// p0 y (count BF16), p1 the next A (count BF16); i0 count
extern "C" __global__ void __launch_bounds__(256) derive_rows(const Args a) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (uint64_t)a.i[0]) return;
  reinterpret_cast<uint16_t*>(a.p[1])[t] = reinterpret_cast<const uint16_t*>(a.p[0])[t] ^ 0x3C00u;
}

// A read-only L2 flush: every thread sums its words of the buffer and writes one word only if the sum is a value it never is,
// so the flush leaves no dirty line.  p0 buffer, p1 sink; i0 words
extern "C" __global__ void __launch_bounds__(256) l2_flush(const Args a) {
  const uint32_t* b = reinterpret_cast<const uint32_t*>(a.p[0]);
  uint32_t s = 0;
  for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; t < (uint64_t)(uint32_t)a.i[0]; t += (uint64_t)gridDim.x * blockDim.x)
    s += b[t];
  if (s == 0x9E3779B9u) reinterpret_cast<uint32_t*>(a.p[1])[0] = s;
}

// ---------------------------------------------------------------- A's commitment at decode: candidates measured, not adopted
// derive_rows_hash: the producer of A hashing it, the "previous layer's epilogue": each warp writes its row of the next A from
// y (derive_rows' values, the row's first rb/2 BF16 words, the rest y's again) and hashes it at once (row_leaf_w: the row is
// in L1/L2), then the last CTA builds A's tree (hash_rows_wt's tail).  The same bytes as derive_rows then hash_rows_wt.
// p0 y (m x n BF16), p1 the next A (m x rb bytes), p2 the tree, p3 domain id, p4 level keys, p5 counter; i0 m, i1 rb, i2 width,
// i3 n.  128 threads (4 rows) a CTA; dynamic shared memory rows_w_words(rb) words per warp.
extern "C" __global__ void __launch_bounds__(128, 1) derive_rows_hash(const Args a) {
  extern __shared__ uint32_t sh[];
  __shared__ int last;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const uint64_t row = (uint64_t)blockIdx.x * (blockDim.x / 32) + warp;
  const uint32_t rb = (uint32_t)a.i[1];
  const int m = a.i[0], width = a.i[2], n = a.i[3];
  uint32_t key[8], h[8];
  load_key(a.p[3], key);
  if (row < (uint64_t)m) {
    const uint16_t* y = reinterpret_cast<const uint16_t*>(a.p[0]) + row * n;
    uint16_t* A = reinterpret_cast<uint16_t*>(a.p[1] + row * rb);
    for (uint32_t e = lane; e < rb / 2; e += 32) A[e] = y[e % n] ^ 0x3C00u;
    __syncwarp();
    __threadfence_block();
    row_leaf_w(reinterpret_cast<const uint8_t*>(A), rb, row, key,
               reinterpret_cast<uint32_t(*)[8]>(sh + (size_t)warp * rows_w_words((int)rb)), lane, h);
    if (lane == 0) store_cv(a.p[2] + row * 32, h);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(reinterpret_cast<unsigned*>(a.p[5]), 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  uint64_t below = a.p[2];
  int level = 0;
  for (uint64_t w = (uint64_t)width; w > 1; w >>= 1, level++) {
    const uint64_t above = below + 32 * w;
    uint32_t lk[8];
    load_key(a.p[4] + 32ull * level, lk);
    for (uint64_t i = threadIdx.x; i < w / 2; i += blockDim.x) {
      uint32_t mm[16], hh[8];
      load_key(below + 64 * i, mm);
      load_key(below + 64 * i + 32, mm + 8);
      blake3_block(lk, mm, hh);
      store_cv(above + 32 * i, hh);
    }
    __syncthreads();
    below = above;
  }
  if (threadIdx.x == 0) *reinterpret_cast<unsigned*>(a.p[5]) = 0u;
}

// hash_rows_blk: a candidate layout, not -h1's.  A row's commitment is a Merkle tree over its 64-byte blocks: block j's leaf is
// the one-block keyed BLAKE3 of (the block) under the row-leaf key, and a node at level l is one block under level key l,
// both standard BLAKE3; so a row is 1 + log2(blocks) compressions deep in place of a 16-block chunk and the chunk tree.  One
// CTA of 256 threads a row, each thread two blocks, the levels in shared memory; the row's root into p1.  p0 rows, p1 out
// (32 B a row), p2 the row-leaf key, p3 level keys (32 B a level); i0 rows, i1 row bytes (a power-of-two multiple of 64,
// at most 512 blocks)
extern "C" __global__ void __launch_bounds__(256) hash_rows_blk(const Args a) {
  __shared__ uint32_t lv[512][8];
  const uint64_t row = blockIdx.x;
  const int blocks = a.i[1] / 64;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(a.p[0] + row * (uint64_t)a.i[1]);
  uint32_t key[8];
  load_key(a.p[2], key);
  for (int j = threadIdx.x; j < blocks; j += blockDim.x) {
    uint32_t m[16], h[8];
#pragma unroll
    for (int w = 0; w < 16; w++) m[w] = src[16 * j + w];
    blake3_block(key, m, h);
#pragma unroll
    for (int i = 0; i < 8; i++) lv[j][i] = h[i];
  }
  __syncthreads();
  int level = 0;
  for (int w = blocks; w > 1; w >>= 1, level++) {
    uint32_t lk[8];
    load_key(a.p[3] + 32ull * level, lk);
    uint32_t m[16], h[8];
    const int i = threadIdx.x;
    if (i < w / 2) {
#pragma unroll
      for (int q = 0; q < 8; q++) { m[q] = lv[2 * i][q]; m[8 + q] = lv[2 * i + 1][q]; }
      blake3_block(lk, m, h);
    }
    __syncthreads();
    if (i < w / 2) {
#pragma unroll
      for (int q = 0; q < 8; q++) lv[i][q] = h[q];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) store_cv(a.p[1] + row * 32, lv[0]);
}

// hash_rows_seg: P1's prototype (a-commit-latency.md §4.1), not -h1's, timing only.  hash_rows_blk with s-byte segments
// (s = 64 b, b blocks): segment j's leaf is the standard keyed BLAKE3 of the segment under the row-leaf key (one chunk of b
// blocks, chunk counter 0), then hash_rows_blk's levels; a row is b + log2(rb / s) compressions deep.  One thread a segment,
// the next block's words loaded while this one compresses; launched with rb / s threads (a multiple of 32).  p0 rows, p1
// out (32 B a row), p2 the row-leaf key, p3 level keys; i0 rows, i1 row bytes, i2 s (at most 1,024: one chunk; rb / s a
// power of two, at most 512)
extern "C" __global__ void __launch_bounds__(512) hash_rows_seg(const Args a) {
  __shared__ uint32_t lv[512][8];
  const uint64_t row = blockIdx.x;
  const int nb = a.i[2] / 64, segs = a.i[1] / a.i[2], j = threadIdx.x;
  uint32_t key[8];
  load_key(a.p[2], key);
  if (j < segs) {
    const uint4* src = reinterpret_cast<const uint4*>(a.p[0] + row * (uint64_t)a.i[1] + (uint64_t)j * a.i[2]);
    uint4 nx[4];
#pragma unroll
    for (int w = 0; w < 4; w++) nx[w] = src[w];
    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    for (int b = 0; b < nb; b++) {
      uint32_t m[16];
#pragma unroll
      for (int w = 0; w < 4; w++) { m[4 * w] = nx[w].x; m[4 * w + 1] = nx[w].y; m[4 * w + 2] = nx[w].z; m[4 * w + 3] = nx[w].w; }
      if (b + 1 < nb) {
#pragma unroll
        for (int w = 0; w < 4; w++) nx[w] = src[4 * (b + 1) + w];
      }
      const uint32_t fl = B3_KEYED | (b == 0 ? B3_START : 0) | (b + 1 == nb ? B3_END | B3_ROOT : 0);
      b3c(cv, m, 0, 64, fl, cv);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) lv[j][i] = cv[i];
  }
  __syncthreads();
  int level = 0;
  for (int w = segs; w > 1; w >>= 1, level++) {
    uint32_t lk[8], m[16], h[8];
    load_key(a.p[3] + 32ull * level, lk);
    if (j < w / 2) {
#pragma unroll
      for (int q = 0; q < 8; q++) { m[q] = lv[2 * j][q]; m[8 + q] = lv[2 * j + 1][q]; }
      blake3_block(lk, m, h);
    }
    __syncthreads();
    if (j < w / 2) {
#pragma unroll
      for (int q = 0; q < 8; q++) lv[j][q] = h[q];
    }
    __syncthreads();
  }
  if (j == 0) store_cv(a.p[1] + row * 32, lv[0]);
}

// derive_rows_hash's A alone (its reference in the gate): row r of the next A is y's row r (n BF16 words, cycled) ^ 0x3C00 over
// rb / 2 words.  p0 y, p1 A; i0 m, i1 rb, i2 n
extern "C" __global__ void __launch_bounds__(256) derive_rows_wide(const Args a) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, per = (uint64_t)a.i[1] / 2;
  if (t >= (uint64_t)a.i[0] * per) return;
  const uint64_t r = t / per, e = t % per;
  reinterpret_cast<uint16_t*>(a.p[1])[t] = reinterpret_cast<const uint16_t*>(a.p[0])[r * a.i[2] + e % a.i[2]] ^ 0x3C00u;
}

// derive_rows_b3s: A's producer with -h2's segment leaves fused (P1 in the producer).  It writes derive_rows_wide's A (row r
// is y's row r, n BF16 words cycled, ^ 0x3C00, over rb / 2 words) a 256-byte segment a thread, and hashes each segment from
// the very words it stores (b3s_segment_regs), so the leaves bind the bytes forming reads.  The segment's key comes first:
// it needs the domain only.  i4 = 0: segment j of row r's leaf into p2 + 32 (r · segs + j) (any launch, a thread a segment);
// i4 = 1: a CTA a row (launched with exactly segs threads, at most 512), which then builds the row's segment tree in shared
// memory (hash_rows_b3s's levels) and writes the row leaf into p2 + 32 r; i4 = 2: and -h3's row seed into p9 + 32 r
// (b3_seeds' mode 1, the unit index i5).  p0 y (m x n BF16), p1 A (m x rb bytes), p2 leaves, p3 domain id, p4 the key template,
// p5 the segment level keys, p6 the seed key, p7 salt, p8 root_B, p9 the row seeds; i0 m, i1 rb (a multiple of 256 with
// 128 | n), i2 n, i3 the template's length, i4 the mode, i5 the unit index.
DEV void derive_segment(const Args& a, uint64_t r, uint32_t j, uint32_t* w) {
  const uint32_t rb = (uint32_t)a.i[1], n = (uint32_t)a.i[2];
  const uint4* y = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(a.p[0]) + r * n + (128u * j) % n);
#pragma unroll
  for (int q = 0; q < 16; q++) {
    const uint4 v = y[q];
    w[4 * q] = v.x ^ 0x3C003C00u; w[4 * q + 1] = v.y ^ 0x3C003C00u; w[4 * q + 2] = v.z ^ 0x3C003C00u; w[4 * q + 3] = v.w ^ 0x3C003C00u;
  }
  uint4* dst = reinterpret_cast<uint4*>(a.p[1] + r * rb + (uint64_t)B3S_SEG * j);
#pragma unroll
  for (int q = 0; q < 16; q++) dst[q] = make_uint4(w[4 * q], w[4 * q + 1], w[4 * q + 2], w[4 * q + 3]);
}

extern "C" __global__ void __launch_bounds__(512) derive_rows_b3s(const Args a) {
  __shared__ uint32_t lv[B3S_MAX_SEGS][8];
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t rb = (uint32_t)a.i[1], segs = rb / B3S_SEG;
  const int mode = a.i[4];
  if (mode == 0 && t >= (uint64_t)a.i[0] * segs) return;
  const uint64_t r = mode ? blockIdx.x : t / segs;
  const uint32_t j = mode ? threadIdx.x : (uint32_t)(t % segs);
  uint32_t dom[8], tmpl[16], key[8], w[64], h[8];
  load_key(a.p[3], dom);
#pragma unroll
  for (int q = 0; q < 16; q++) tmpl[q] = reinterpret_cast<const uint32_t*>(a.p[4])[q];
  b3s_segment_key(dom, tmpl, (uint32_t)a.i[3], r, r, rb, j, key);
  derive_segment(a, r, j, w);
  b3s_segment_regs(key, w, h);
  if (mode == 0) {
    store_cv(a.p[2] + 32 * t, h);
    return;
  }
#pragma unroll
  for (int i = 0; i < 8; i++) lv[j][i] = h[i];
  __syncthreads();
  int level = 0;
  for (uint32_t wd = segs; wd > 1; wd = (wd + 1) / 2, level++) {
    const uint32_t half = wd / 2;
    if (j < half) {
      uint32_t lk[8], m[16];
      load_key(a.p[5] + 32ull * level, lk);
#pragma unroll
      for (int q = 0; q < 8; q++) { m[q] = lv[2 * j][q]; m[8 + q] = lv[2 * j + 1][q]; }
      blake3_block(lk, m, h);
    } else if (j == half && (wd & 1)) {
#pragma unroll
      for (int q = 0; q < 8; q++) h[q] = lv[wd - 1][q];
    }
    __syncthreads();
    if (j < half || (j == half && (wd & 1))) {
#pragma unroll
      for (int q = 0; q < 8; q++) lv[j][q] = h[q];
    }
    __syncthreads();
  }
  if (j == 0) {
    store_cv(a.p[2] + 32 * r, lv[0]);
    if (mode == 2) {
      uint32_t sd[8];
      b3_seed(a.p[6], a.p[7], lv[0], a.p[8], (uint64_t)(uint32_t)a.i[5] << 32 | r, sd);
      store_cv(a.p[9] + 32 * r, sd);
    }
  }
}

// derive_rows_b3s's producer without the hashing: the same words, a 256-byte segment a thread (16-byte loads and stores), so
// the fused producer's cost over it is the hashing alone.  p0 y, p1 A; i0 m, i1 rb, i2 n
extern "C" __global__ void __launch_bounds__(128) derive_rows_vec(const Args a) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t segs = (uint32_t)a.i[1] / B3S_SEG;
  if (t >= (uint64_t)a.i[0] * segs) return;
  uint32_t w[64];
  derive_segment(a, t / segs, (uint32_t)(t % segs), w);
}

// ---- A's producer, coalesced: the chain's store layout with the -h2 / -h3 commitment fused ----------------------------
// derive_rows_b3s_coal: a CTA a row (1,024 threads).  Every thread derives and stores 16-byte pieces of the row, consecutive
// threads consecutive pieces (coalesced, as a plain producer stores), and stages each piece in shared memory, one 256-byte
// segment every 272 bytes (the pad spreads a segment's 16-byte reads over all banks).  Then thread j < rb / 256 reads
// segment j back, hashes it under its key (computed while the loads were in flight) with b3s_segment_regs, and the block builds
// the row's segment tree; i4 = 2 adds -h3's row seed.  After staging, the warps past the hashers return and the tree's
// barriers are a named barrier over the hashers' warps.  The hashed words are the stored words.  Arguments as derive_rows_b3s's
// mode 1 and 2 (p0 y, p1 A, p2 row leaves, p3 domain id, p4 key template, p5 segment level keys, p6 seed key, p7 salt, p8
// root_B, p9 row seeds; i0 m, i1 rb at most 32 KB, a multiple of 256 with 128 | n, i2 n, i3 template length, i4 mode, i5 index).
constexpr int COAL_SEGS = 128, COAL_PAD = 68;
// A barrier over the first `count` threads of the block (a multiple of 32): named barrier 1, so warps that have returned
// aren't waited for.  The host twin's block barrier drops returned threads, so there it is __syncthreads.
DEV void hashers_sync(uint32_t count) {
#ifdef __CUDA_ARCH__
  asm volatile("bar.sync 1, %0;" ::"r"(count) : "memory");
#else
  (void)count;
  __syncthreads();
#endif
}
extern "C" __global__ void __launch_bounds__(1024) derive_rows_b3s_coal(const Args a) {
  __shared__ __align__(16) uint32_t seg[COAL_SEGS * COAL_PAD];
  __shared__ uint32_t lv[COAL_SEGS][8];
  const uint64_t r = blockIdx.x;
  const uint32_t rb = (uint32_t)a.i[1], n = (uint32_t)a.i[2], segs = rb / B3S_SEG, pieces = rb / 16;
  const uint32_t tid = threadIdx.x;
  const uint4* y = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(a.p[0]) + r * n);
  uint4* dst = reinterpret_cast<uint4*>(a.p[1] + r * rb);
  uint4 v[2];
#pragma unroll
  for (int p = 0; p < 2; p++) {
    const uint32_t q = tid + p * blockDim.x;
    if (q < pieces) v[p] = y[(8 * q % n) / 8];
  }
  uint32_t key[8];
  if (tid < segs) {
    uint32_t dom[8], tmpl[16];
    load_key(a.p[3], dom);
#pragma unroll
    for (int w = 0; w < 16; w++) tmpl[w] = reinterpret_cast<const uint32_t*>(a.p[4])[w];
    b3s_segment_key(dom, tmpl, (uint32_t)a.i[3], r, r, rb, tid, key);
  }
#pragma unroll
  for (int p = 0; p < 2; p++) {
    const uint32_t q = tid + p * blockDim.x;
    if (q < pieces) {
      const uint4 x = make_uint4(v[p].x ^ 0x3C003C00u, v[p].y ^ 0x3C003C00u, v[p].z ^ 0x3C003C00u, v[p].w ^ 0x3C003C00u);
      dst[q] = x;
      *reinterpret_cast<uint4*>(&seg[(q / 16) * COAL_PAD + (q % 16) * 4]) = x;
    }
  }
  __syncthreads();
  const uint32_t hashers = 32 * ((segs + 31) / 32);
  if (tid >= hashers) return;                              // the store-only warps are done; the tree's barriers are the hashers'
  if (tid < segs) {
    uint32_t w[64], h[8];
#pragma unroll
    for (int q = 0; q < 16; q++) {
      const uint4 x = *reinterpret_cast<const uint4*>(&seg[tid * COAL_PAD + 4 * q]);
      w[4 * q] = x.x; w[4 * q + 1] = x.y; w[4 * q + 2] = x.z; w[4 * q + 3] = x.w;
    }
    b3s_segment_regs(key, w, h);
#pragma unroll
    for (int i = 0; i < 8; i++) lv[tid][i] = h[i];
  }
  hashers_sync(hashers);
  int level = 0;
  for (uint32_t wd = segs; wd > 1; wd = (wd + 1) / 2, level++) {
    const uint32_t half = wd / 2;
    uint32_t h[8];
    if (tid < half) {
      uint32_t lk[8], m[16];
      load_key(a.p[5] + 32ull * level, lk);
#pragma unroll
      for (int q = 0; q < 8; q++) { m[q] = lv[2 * tid][q]; m[8 + q] = lv[2 * tid + 1][q]; }
      blake3_block(lk, m, h);
    } else if (tid == half && (wd & 1)) {
#pragma unroll
      for (int q = 0; q < 8; q++) h[q] = lv[wd - 1][q];
    }
    hashers_sync(hashers);
    if (tid < half || (tid == half && (wd & 1))) {
#pragma unroll
      for (int q = 0; q < 8; q++) lv[tid][q] = h[q];
    }
    hashers_sync(hashers);
  }
  if (tid == 0) {
    store_cv(a.p[2] + 32 * r, lv[0]);
    if (a.i[4] == 2) {
      uint32_t sd[8];
      b3_seed(a.p[6], a.p[7], lv[0], a.p[8], (uint64_t)(uint32_t)a.i[5] << 32 | r, sd);
      store_cv(a.p[9] + 32 * r, sd);
    }
  }
}

// The same producer without the hashing (its matched baseline): a CTA a row, 1,024 threads, 16-byte coalesced stores.  p0 y,
// p1 A; i0 m, i1 rb, i2 n
extern "C" __global__ void __launch_bounds__(1024) derive_rows_coal(const Args a) {
  const uint64_t r = blockIdx.x;
  const uint32_t rb = (uint32_t)a.i[1], n = (uint32_t)a.i[2], pieces = rb / 16;
  const uint4* y = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(a.p[0]) + r * n);
  uint4* dst = reinterpret_cast<uint4*>(a.p[1] + r * rb);
  for (uint32_t q = threadIdx.x; q < pieces; q += blockDim.x) {
    const uint4 v = y[(8 * q % n) / 8];
    dst[q] = make_uint4(v.x ^ 0x3C003C00u, v.y ^ 0x3C003C00u, v.z ^ 0x3C003C00u, v.w ^ 0x3C003C00u);
  }
}

// The fastest plain whole-row producer here: a thread a 16-byte piece over the whole of A, 256-thread CTAs.  p0 y, p1 A; i0 m,
// i1 rb, i2 n
extern "C" __global__ void __launch_bounds__(256) derive_rows_v16(const Args a) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t rb = (uint32_t)a.i[1], n = (uint32_t)a.i[2], pieces = rb / 16;
  if (t >= (uint64_t)a.i[0] * pieces) return;
  const uint64_t r = t / pieces;
  const uint32_t q = (uint32_t)(t % pieces);
  const uint4 v = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(a.p[0]) + r * n)[(8 * q % n) / 8];
  reinterpret_cast<uint4*>(a.p[1] + r * rb)[q] = make_uint4(v.x ^ 0x3C003C00u, v.y ^ 0x3C003C00u, v.z ^ 0x3C003C00u, v.w ^ 0x3C003C00u);
}

// derive_rows_b3s_split: derive_rows_b3s's thread-a-segment producer in CTAs of 32 threads, so A's stores spread over
// m · segs / 32 SMs rather than m, each segment's leaf to p10 (32 B, row-major); the last CTA of each row (a u32 counter a row
// in p11, which it resets) reads the row's leaves back through L2 and builds the row's segment tree and, with i4 = 2, -h3's
// row seed.  The hashed words are the stored words.  Arguments as derive_rows_b3s's modes 1 and 2, plus p10 the segment leaves
// and p11 the counters; launched with m · (rb / 256) / 32 CTAs of 32 threads (rb / 256 a multiple of 32, at most 256).
DEV uint32_t ld_cg(const uint32_t* p) {
#ifdef __CUDA_ARCH__
  return __ldcg(p);
#else
  return *p;
#endif
}
extern "C" __global__ void __launch_bounds__(32) derive_rows_b3s_split(const Args a) {
  __shared__ uint32_t lv[256][8];
  __shared__ int last;
  const uint32_t rb = (uint32_t)a.i[1], segs = rb / B3S_SEG, per = segs / 32;
  const uint64_t r = blockIdx.x / per;
  const uint32_t j = (blockIdx.x % per) * 32 + threadIdx.x;
  uint32_t dom[8], tmpl[16], key[8], w[64], h[8];
  load_key(a.p[3], dom);
#pragma unroll
  for (int q = 0; q < 16; q++) tmpl[q] = reinterpret_cast<const uint32_t*>(a.p[4])[q];
  b3s_segment_key(dom, tmpl, (uint32_t)a.i[3], r, r, rb, j, key);
  derive_segment(a, r, j, w);
  b3s_segment_regs(key, w, h);
  store_cv(a.p[10] + 32 * (r * segs + j), h);
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(reinterpret_cast<unsigned*>(a.p[11]) + r, 1u) == per - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const uint32_t* leaves = reinterpret_cast<const uint32_t*>(a.p[10] + 32 * r * segs);
  for (uint32_t i = threadIdx.x; i < segs; i += 32)
#pragma unroll
    for (int q = 0; q < 8; q++) lv[i][q] = ld_cg(leaves + 8 * i + q);
  __syncthreads();
  int level = 0;
  for (uint32_t wd = segs; wd > 1; wd = (wd + 1) / 2, level++) {
    const uint32_t half = wd / 2;
    uint32_t lk[8], out[4][8];
    load_key(a.p[5] + 32ull * level, lk);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const uint32_t i = threadIdx.x + 32 * k;
      if (i < half) {
        uint32_t m[16];
#pragma unroll
        for (int q = 0; q < 8; q++) { m[q] = lv[2 * i][q]; m[8 + q] = lv[2 * i + 1][q]; }
        blake3_block(lk, m, out[k]);
      } else if (i == half && (wd & 1)) {
#pragma unroll
        for (int q = 0; q < 8; q++) out[k][q] = lv[wd - 1][q];
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const uint32_t i = threadIdx.x + 32 * k;
      if (i < half || (i == half && (wd & 1))) {
#pragma unroll
        for (int q = 0; q < 8; q++) lv[i][q] = out[k][q];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    store_cv(a.p[2] + 32 * r, lv[0]);
    if (a.i[4] == 2) {
      uint32_t sd[8];
      b3_seed(a.p[6], a.p[7], lv[0], a.p[8], (uint64_t)(uint32_t)a.i[5] << 32 | r, sd);
      store_cv(a.p[9] + 32 * r, sd);
    }
    reinterpret_cast<unsigned*>(a.p[11])[r] = 0;
  }
}

// derive_rows_b3s_wide: the fastest plain layout (derive_rows_v16's: a thread a 16-byte piece, 256-thread CTAs, 16 segments a
// CTA) with the commitment fused.  Each thread loads its piece first, and threads j < 16 compute their segment's key while the
// loads are in flight; every thread stores its piece (coalesced) and stages it in shared memory (a segment every 272 bytes);
// threads j < 16 hash their segments from shared memory (b3s_segment_regs) and the CTA builds its 16-leaf subtree, which is
// the row's segment tree at levels 0-3.  The last of the row's CTAs (a u32 counter a row in p11, which it resets) reads the
// subtree roots back through L2 (p10) and builds the levels above, then -h3's row seed with i4 = 2.  The hashed words are the
// stored words.  Arguments as derive_rows_b3s_split's; launched with m · rb / 4096 CTAs of 256 threads (rb a multiple of 4096,
// at most 64 KB).
extern "C" __global__ void __launch_bounds__(256) derive_rows_b3s_wide(const Args a) {
  __shared__ __align__(16) uint32_t seg[16 * COAL_PAD];
  __shared__ uint32_t lv[16][8];
  __shared__ int last;
  const uint32_t rb = (uint32_t)a.i[1], n = (uint32_t)a.i[2], per = rb / 4096;
  const uint64_t r = blockIdx.x / per;
  const uint32_t c = blockIdx.x % per, tid = threadIdx.x, q = 256 * c + tid;
  const uint4 v = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(a.p[0]) + r * n)[(8 * q % n) / 8];
  uint32_t key[8];
  if (tid < 16) {
    uint32_t dom[8], tmpl[16];
    load_key(a.p[3], dom);
#pragma unroll
    for (int w = 0; w < 16; w++) tmpl[w] = reinterpret_cast<const uint32_t*>(a.p[4])[w];
    b3s_segment_key(dom, tmpl, (uint32_t)a.i[3], r, r, rb, 16 * c + tid, key);
  }
  const uint4 x = make_uint4(v.x ^ 0x3C003C00u, v.y ^ 0x3C003C00u, v.z ^ 0x3C003C00u, v.w ^ 0x3C003C00u);
  reinterpret_cast<uint4*>(a.p[1] + r * rb)[q] = x;
  *reinterpret_cast<uint4*>(&seg[(tid / 16) * COAL_PAD + (tid % 16) * 4]) = x;
  __syncthreads();
  if (tid < 16) {
    uint32_t w[64], h[8];
#pragma unroll
    for (int k = 0; k < 16; k++) {
      const uint4 y4 = *reinterpret_cast<const uint4*>(&seg[tid * COAL_PAD + 4 * k]);
      w[4 * k] = y4.x; w[4 * k + 1] = y4.y; w[4 * k + 2] = y4.z; w[4 * k + 3] = y4.w;
    }
    b3s_segment_regs(key, w, h);
#pragma unroll
    for (int i = 0; i < 8; i++) lv[tid][i] = h[i];
  }
  __syncthreads();
  int level = 0;
  for (uint32_t wd = 16; wd > 1; wd /= 2, level++) {
    uint32_t h[8];
    if (tid < wd / 2) {
      uint32_t lk[8], m[16];
      load_key(a.p[5] + 32ull * level, lk);
#pragma unroll
      for (int k = 0; k < 8; k++) { m[k] = lv[2 * tid][k]; m[8 + k] = lv[2 * tid + 1][k]; }
      blake3_block(lk, m, h);
    }
    __syncthreads();
    if (tid < wd / 2) {
#pragma unroll
      for (int k = 0; k < 8; k++) lv[tid][k] = h[k];
    }
    __syncthreads();
  }
  if (tid == 0) {
    store_cv(a.p[10] + 32 * (r * per + c), lv[0]);
    __threadfence();
    last = atomicAdd(reinterpret_cast<unsigned*>(a.p[11]) + r, 1u) == per - 1;
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (tid < per) {
    const uint32_t* roots = reinterpret_cast<const uint32_t*>(a.p[10] + 32 * (r * per + tid));
#pragma unroll
    for (int k = 0; k < 8; k++) lv[tid][k] = ld_cg(roots + k);
  }
  __syncthreads();
  for (uint32_t wd = per; wd > 1; wd = (wd + 1) / 2, level++) {
    const uint32_t half = wd / 2;
    uint32_t h[8];
    if (tid < half) {
      uint32_t lk[8], m[16];
      load_key(a.p[5] + 32ull * level, lk);
#pragma unroll
      for (int k = 0; k < 8; k++) { m[k] = lv[2 * tid][k]; m[8 + k] = lv[2 * tid + 1][k]; }
      blake3_block(lk, m, h);
    } else if (tid == half && (wd & 1)) {
#pragma unroll
      for (int k = 0; k < 8; k++) h[k] = lv[wd - 1][k];
    }
    __syncthreads();
    if (tid < half || (tid == half && (wd & 1))) {
#pragma unroll
      for (int k = 0; k < 8; k++) lv[tid][k] = h[k];
    }
    __syncthreads();
  }
  if (tid == 0) {
    store_cv(a.p[2] + 32 * r, lv[0]);
    if (a.i[4] == 2) {
      uint32_t sd[8];
      b3_seed(a.p[6], a.p[7], lv[0], a.p[8], (uint64_t)(uint32_t)a.i[5] << 32 | r, sd);
      store_cv(a.p[9] + 32 * r, sd);
    }
    reinterpret_cast<unsigned*>(a.p[11])[r] = 0;
  }
}

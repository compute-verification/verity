// The decode GEMM's memory traffic, for timing -h1's hashing around it on sm_120 (h1_bench.py). It is a stand-in, not a GEMM:
// it reads the n x k FP8 weights once (16 bytes a load, a warp per weight row) and writes C~, U (m x n FP32 words) and y
// (m x n BF16), each a function of the weight row read and of A's rows, so that every output depends on both; and
// derive_rows, the next call's A (m x k BF16, k = n) from y, which makes the calls a dependent chain.
#include "hash_sm120.cuh"

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

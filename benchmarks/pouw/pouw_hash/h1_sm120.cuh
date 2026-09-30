// h1_sm120.cuh: -h1's commitments of A and of the tiles in fewer launches on sm_120, committing exactly what hash.cuh's
// hash_rows, hash_pad, hash_node_keys, hash_nodes and hash_tree_leaves write (#449's frame-b3 at 6a1a051f) and what
// pearl_c_sm120.cu's seed_a derives.  h1_sm120.cu builds them with hash.cuh and hash_sm120.cuh into hash_sm120a.cubin.
#pragma once
#include "../pearl_c/hash_sm120.cuh"

DEV uint4 ld_nc(const uint4* p) { return __ldg(p); }
DEV void ld_cv_cg(uint64_t p, uint32_t* h) {   // a node another CTA wrote, from L2
  const uint4 x = __ldcg(reinterpret_cast<const uint4*>(p)), y = __ldcg(reinterpret_cast<const uint4*>(p) + 1);
  h[0] = x.x; h[1] = x.y; h[2] = x.z; h[3] = x.w; h[4] = y.x; h[5] = y.y; h[6] = y.z; h[7] = y.w;
}

// A row's frame-b3 leaf by one warp, as row_leaf_w hashes it, with each block's 16 words as four 16-byte loads issued a
// block ahead.  Needs rb a multiple of 16 and src 16-byte aligned for the vector path; other blocks go through row_msg_word.
DEV void row_leaf_v(const uint8_t* src, uint32_t rb, uint64_t row, const uint32_t* key, uint32_t (*cvs)[8], int lane, uint32_t* h) {
  const uint64_t len = ROW_PREFIX + (uint64_t)rb;
  const int n = (int)((len + 1023) / 1024);
  const uint4* src4 = reinterpret_cast<const uint4*>(src);
  const bool aligned = rb % 16 == 0 && (reinterpret_cast<uint64_t>(src) & 15) == 0;
  for (int c = lane; c < n; c += 32) {
    const uint64_t left = len - 1024ull * c;
    const uint32_t clen = (uint32_t)(left < 1024 ? left : 1024);
    const int blocks = (int)((clen + 63) / 64);
    // block b's message words 256c + 16b … are row words 256c + 16b − 16 …, all in the row when fast(b)
    auto fast = [&](int b) { const uint64_t j0 = 256ull * c + 16ull * b; return aligned && j0 >= 16 && 4 * j0 <= rb; };
    uint32_t cv[8];
    uint4 nxt[4];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    bool nxt_fast = fast(0);
    if (nxt_fast) {
#pragma unroll
      for (int q = 0; q < 4; q++) nxt[q] = ld_nc(src4 + (64ull * c - 4) + q);
    }
    for (int b = 0; b < blocks; b++) {
      const uint64_t j0 = 256ull * c + 16ull * b;
      uint32_t m[16];
      const bool cur_fast = nxt_fast;
#pragma unroll
      for (int q = 0; q < 4; q++) { m[4 * q] = nxt[q].x; m[4 * q + 1] = nxt[q].y; m[4 * q + 2] = nxt[q].z; m[4 * q + 3] = nxt[q].w; }
      if (b + 1 < blocks) {
        nxt_fast = fast(b + 1);
        if (nxt_fast) {
#pragma unroll
          for (int q = 0; q < 4; q++) nxt[q] = ld_nc(src4 + j0 / 4 + q);
        }
      }
      if (!cur_fast) {
#pragma unroll
        for (int w = 0; w < 16; w++) m[w] = row_msg_word(src, rb, row, j0 + w);
      }
      const bool last = b == blocks - 1;
      const uint32_t f = B3_KEYED | (b == 0 ? B3_START : 0) | (last ? B3_END : 0) | (last && n == 1 ? B3_ROOT : 0);
      b3c(cv, m, (uint64_t)c, last ? clen - 64 * b : 64, f, cv);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) cvs[c][i] = cv[i];
  }
  __syncwarp();
  for (int s = 1; s < n; s <<= 1) {
    for (int i = 2 * s * lane; i < n; i += 64 * s)
      if (i + s < n && block_size(n, i) >= 2 * s) b3_parent(key, cvs[i], cvs[i + s], 2 * s == n ? B3_ROOT : 0, cvs[i]);
    __syncwarp();
  }
  if (lane != 0) return;
  const int top = 31 - __clz(n);
  if (n == 1 << top) {
#pragma unroll
    for (int i = 0; i < 8; i++) h[i] = cvs[0][i];
  } else {
    int off = n - (n & -n);
#pragma unroll
    for (int i = 0; i < 8; i++) h[i] = cvs[off][i];
    int rest = n - (n & -n);
    while (rest) {
      const int low = rest & -rest;
      off = rest - low;
      rest -= low;
      b3_parent(key, cvs[off], h, rest ? 0 : B3_ROOT, h);
    }
  }
}

// hash_rows, a warp per row and blockDim / 32 rows a CTA (row_leaf_v).  Dynamic shared memory: rows_w_words(rb) words a
// warp.  Same arguments as hash_rows.
extern "C" __global__ void __launch_bounds__(256) h1_rows(const Args a) {
  extern __shared__ uint32_t sh[];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const uint64_t row = (uint64_t)blockIdx.x * (blockDim.x / 32) + warp;
  const uint32_t rb = (uint32_t)a.i[1];
  if (row >= (uint64_t)a.i[0]) return;
  uint32_t key[8], h[8];
  load_key(a.p[2], key);
  row_leaf_v(reinterpret_cast<const uint8_t*>(a.p[0]) + row * rb, rb, row, key,
             reinterpret_cast<uint32_t(*)[8]>(sh + (size_t)warp * rows_w_words((int)rb)), lane, h);
  if (lane == 0) store_cv(a.p[1] + row * 32, h);
}

// a label of n <= 64 bytes as one message block (zero-padded, little-endian words)
DEV void h1_label(const char* s, int n, uint32_t w[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) {
    uint32_t x = 0;
#pragma unroll
    for (int j = 0; j < 4; j++)
      if (4 * i + j < n) x |= (uint32_t)(uint8_t)s[4 * i + j] << (8 * j);
    w[i] = x;
  }
}

// the levels above a run of `width` nodes in sh (a power of two, width <= 2 blockDim), level `l0` of the tree at `off`
// nodes into it (`first` the run's first node there), each level written back to sh[0 …) and to the tree; the root ends
// in sh[0].  keys: the level keys in shared memory.
DEV void h1_levels(uint32_t (*sh)[8], const uint32_t (*keys)[8], uint64_t tree, uint64_t tw, int l0, uint64_t first, int width) {
  for (int l = l0, w = width; w > 1; l++, w >>= 1) {
    const int j = threadIdx.x, above = w / 2;
    uint32_t h[8];
    if (j < above) {
      uint32_t m[16];
#pragma unroll
      for (int i = 0; i < 8; i++) m[i] = sh[2 * j][i], m[8 + i] = sh[2 * j + 1][i];
      blake3_block(keys[l], m, h);
    }
    __syncthreads();
    if (j < above) {
      const uint64_t lvl = 2 * tw - (2 * tw >> (l + 1)), f = first >> (l + 1 - l0);
#pragma unroll
      for (int i = 0; i < 8; i++) sh[j][i] = h[i];
      store_cv(tree + 32 * (lvl + f + j), h);
    }
    __syncthreads();
  }
}

// A whole frame-b3 tree in one launch: level 0's pads (and from p3 the tile tree's leaves), the level keys, every level,
// and under p4 seed_A from the root as seed_a derives it.  CTA c builds the subtree over leaves [c S, (c + 1) S), S = 2
// blockDim (or the width); the last CTA to finish, by p2's counter (zero at the first launch; atomicInc leaves it zero
// again), builds the levels above from the gridDim subtree roots, which needs gridDim <= 2 blockDim.  Dynamic shared memory:
// 32 (S + 32) bytes.
// p0 the tree (level 0 at offset 0), p1 the domain id, p2 the counter (u32), p3 tile leaves (32 B each; 0: level 0 holds the
// leaves), p4 out (seed_A ‖ line key ‖ jackpot key, 96 B; 0: none), p5 blake3("pearl-c/v0/seed-A"), p6 salt, p7 root_B,
// p8 the level keys out (32 B a level; 0: none); i0 leaves, i1 width (a power of two), i2 the matmul's index
extern "C" __global__ void __launch_bounds__(1024) h1_tree(const Args a) {
  extern __shared__ uint32_t smem[];
  uint32_t (*keys)[8] = reinterpret_cast<uint32_t(*)[8]>(smem);
  uint32_t (*sh)[8] = keys + 32;
  __shared__ bool last;
  const uint64_t tree = a.p[0], count = (uint64_t)a.i[0], tw = (uint64_t)a.i[1];
  const int depth = 63 - __clzll((long long)tw), S = (int)(tw < 2ull * blockDim.x ? tw : 2ull * blockDim.x), ls = 31 - __clz(S);
  const uint64_t first = (uint64_t)blockIdx.x * S;
  uint32_t dom[8];
  load_key(a.p[1], dom);
  for (int l = threadIdx.x; l < depth; l += blockDim.x) {
    b3_short<2>(dom, [&](int j) { return j == 0 ? 2u : (uint32_t)l; }, keys[l]);
    if (a.p[8] && blockIdx.x == 0) store_cv(a.p[8] + 32ull * l, keys[l]);
  }
  for (int j = threadIdx.x; j < S; j += blockDim.x) {
    const uint64_t i = first + j;
    uint32_t h[8];
    if (i >= count) {
      b3_short<9>(dom, [&](int q) { return q == 0 ? 3u : word_le(i, q - 1); }, h);
      store_cv(tree + 32 * i, h);
    } else if (a.p[3]) {
      uint32_t d[8];
      load_key(a.p[3] + 32 * i, d);
      b3_short<96>(dom, [&](int q) { return q < 64 ? header_byte(q, i, TILE_SCHEMA, 19) : cv_byte(d, q - 64); }, h);
      store_cv(tree + 32 * i, h);
    } else {
      load_key(tree + 32 * i, h);
    }
#pragma unroll
    for (int q = 0; q < 8; q++) sh[j][q] = h[q];
  }
  __syncthreads();
  h1_levels(sh, keys, tree, tw, 0, first, S);
  if (gridDim.x > 1) {
    if (threadIdx.x == 0) {
      __threadfence();
      last = atomicInc(reinterpret_cast<unsigned*>(a.p[2]), gridDim.x - 1) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const uint64_t lvl = 2 * tw - (2 * tw >> ls);
    for (int j = threadIdx.x; j < (int)gridDim.x; j += blockDim.x) ld_cv_cg(tree + 32 * (lvl + j), sh[j]);
    __syncthreads();
    h1_levels(sh, keys, tree, tw, ls, 0, gridDim.x);
  }
  if (!a.p[4] || threadIdx.x != 0) return;
  const uint32_t* salt = reinterpret_cast<const uint32_t*>(a.p[6]);
  const uint32_t* root_b = reinterpret_cast<const uint32_t*>(a.p[7]);
  uint32_t m[16], cv[8], seed[8], h[8];
  load_key(a.p[5], cv);
#pragma unroll
  for (int i = 0; i < 8; i++) m[i] = salt[i], m[8 + i] = sh[0][i];
  b3c(cv, m, 0, 64, B3_KEYED | B3_START, cv);
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = i < 8 ? root_b[i] : 0u;
  m[8] = (uint32_t)a.i[2];
  b3c(cv, m, 0, 40, B3_KEYED | B3_END | B3_ROOT, seed);
  uint32_t* out = reinterpret_cast<uint32_t*>(a.p[4]);
  const char line[] = "pearl/v4/FP8/noise-line", jackpot[] = "pearl/v4/FP8/jackpot";
  h1_label(line, sizeof line - 1, m);
  b3c(seed, m, 0, sizeof line - 1, B3_KEYED | B3_START | B3_END | B3_ROOT, h);
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = seed[i], out[8 + i] = h[i];
  h1_label(jackpot, sizeof jackpot - 1, m);
  b3c(seed, m, 0, sizeof jackpot - 1, B3_KEYED | B3_START | B3_END | B3_ROOT, h);
#pragma unroll
  for (int i = 0; i < 8; i++) out[16 + i] = h[i];
}

// hash_sm120.cuh: the -h1 kernels tuned for sm_120, beside hash.cuh's as pushed (whose code is unchanged): the same
// bytes in fewer serial steps.  hash.cu builds both into hash_sm120a.cubin; GPU 1's sm_120 port can include this header.
#pragma once
#include "hash.cuh"

// ---------------------------------------------------------------- -h1 on sm_120: the same bytes, fewer serial steps
// The kernels below commit exactly what hash_rows, hash_nodes, hash_leaf and hash_msg do; the host picks either form.

// one -h1 message digest from a lane's 32 C~ words and 32 U words (registers or memory), for hash_msg_v and GEMM epilogues
DEV void b3_msg_digest(const uint32_t* c, const uint32_t* u, const uint32_t* key, uint32_t* out) {
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = key[i];
#pragma unroll
  for (int b = 0; b < 4; b++)
    b3c(out, (b < 2 ? c : u) + 16 * (b & 1), 0, 64, B3_KEYED | (b == 0 ? B3_START : 0) | (b == 3 ? B3_END | B3_ROOT : 0), out);
}

// hash_msg with 16-byte loads.  Same arguments.
extern "C" __global__ void __launch_bounds__(128) hash_msg_v(const Args a) {
  const int per = a.i[1] / 32;
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)a.i[0] * per) return;
  const uint4* c4 = reinterpret_cast<const uint4*>(a.p[0]) + t * 8;
  const uint4* u4 = reinterpret_cast<const uint4*>(a.p[1]) + t * 8;
  uint32_t c[32], u[32], key[8], h[8];
#pragma unroll
  for (int q = 0; q < 8; q++) {
    const uint4 x = c4[q], y = u4[q];
    c[4 * q] = x.x; c[4 * q + 1] = x.y; c[4 * q + 2] = x.z; c[4 * q + 3] = x.w;
    u[4 * q] = y.x; u[4 * q + 1] = y.y; u[4 * q + 2] = y.z; u[4 * q + 3] = y.w;
  }
  load_key(a.p[3], key);
  b3_msg_digest(c, u, key, h);
  store_cv(a.p[2] + t * 32, h);
}

// A row's message byte i, i < ROW_PREFIX: the frame-b3 leaf prefix (rank = position = row)
DEV uint32_t row_prefix_byte(uint64_t row, int i) {
  if (i == 0) return 1u;
  if (i < 17) return word_le(row, (i - 1) % 8);
  if (i == 17) return 18u;
  if (i == 18) return 0u;
  return ROW_SCHEMA[i - 19];
}
// shared words hash_rows_w needs per warp for a row of rb bytes: its chunks' chaining values
__host__ __device__ inline int rows_w_words(int rb) { return 8 * ((ROW_PREFIX + rb + 1023) / 1024); }

// A row's frame-b3 leaf by one warp: lane c hashes chunks c, c + 32, … straight from the row (each message word one funnel
// shift of two row words when the row is word-aligned, the next block's words loaded while this block compresses; the
// prefix's and the tail's words byte by byte), the chaining values go to cvs (8 words a chunk), and the chunk tree is built
// as hash_rows builds it.  Lane 0 returns the leaf in h.
DEV void row_leaf_w(const uint8_t* src, uint32_t rb, uint64_t row, const uint32_t* key, uint32_t (*cvs)[8], int lane, uint32_t* h) {
  const uint64_t len = ROW_PREFIX + (uint64_t)rb;
  const int n = (int)((len + 1023) / 1024);
  const uint32_t* src_w = reinterpret_cast<const uint32_t*>(src);
  const bool aligned = rb % 4 == 0 && (reinterpret_cast<uint64_t>(src) & 3) == 0;
  for (int c = lane; c < n; c += 32) {
    const uint64_t left = len - 1024ull * c;
    const uint32_t clen = (uint32_t)(left < 1024 ? left : 1024);
    const int blocks = (int)((clen + 63) / 64);
    auto fast = [&](int b) { const uint64_t j0 = 256ull * c + 16ull * b; return aligned && j0 >= 10 && 4 * (j0 + 6) + 3 < rb; };
    uint32_t cv[8], nxt[17];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    bool nxt_fast = fast(0);
    if (nxt_fast) {
#pragma unroll
      for (int q = 0; q < 17; q++) nxt[q] = src_w[256ull * c - 10 + q];
    }
    for (int b = 0; b < blocks; b++) {
      const uint64_t j0 = 256ull * c + 16ull * b;             // the block's first message word
      uint32_t cur[17], m[16];
      const bool cur_fast = nxt_fast;
#pragma unroll
      for (int q = 0; q < 17; q++) cur[q] = nxt[q];
      if (b + 1 < blocks) {
        nxt_fast = fast(b + 1);
        if (nxt_fast) {
#pragma unroll
          for (int q = 0; q < 17; q++) nxt[q] = src_w[j0 + 16 - 10 + q];
        }
      }
      if (cur_fast) {                                         // message word j is bytes 4j − 37 … of the row
#pragma unroll
        for (int w = 0; w < 16; w++) m[w] = __funnelshift_r(cur[w], cur[w + 1], 24);
      } else {
#pragma unroll
        for (int w = 0; w < 16; w++) {
          uint32_t x = 0;
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const uint64_t i = 4 * (j0 + w) + k;
            const uint32_t v = i < ROW_PREFIX ? row_prefix_byte(row, (int)i) : i < len ? (uint32_t)src[i - ROW_PREFIX] : 0u;
            x |= v << (8 * k);
          }
          m[w] = x;
        }
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

// hash_rows, one warp per row and blockDim / 32 rows per CTA (row_leaf_w).  Dynamic shared memory: rows_w_words(rb) words per
// warp.  Same arguments as hash_rows.
extern "C" __global__ void __launch_bounds__(256) hash_rows_w(const Args a) {
  extern __shared__ uint32_t sh[];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const uint64_t row = (uint64_t)blockIdx.x * (blockDim.x / 32) + warp;
  const uint32_t rb = (uint32_t)a.i[1];
  if (row >= (uint64_t)a.i[0]) return;
  uint32_t key[8], h[8];
  load_key(a.p[2], key);
  row_leaf_w(reinterpret_cast<const uint8_t*>(a.p[0]) + row * rb, rb, row, key,
             reinterpret_cast<uint32_t(*)[8]>(sh + (size_t)warp * rows_w_words((int)rb)), lane, h);
  if (lane == 0) store_cv(a.p[1] + row * 32, h);
}

// every level of a frame-b3 tree in one CTA, for a level 0 (pads included, width a power of two) of at most 2 blockDim nodes:
// the levels hash_nodes writes one launch at a time, in the same places.  p0 the tree (level 0 at offset 0), p1 domain id;
// i0 the width of level 0
extern "C" __global__ void __launch_bounds__(1024) hash_tree_small(const Args a) {
  uint32_t key[8];
  load_key(a.p[1], key);
  uint64_t below = a.p[0];
  uint32_t level = 0;
  for (uint64_t w = (uint64_t)a.i[0]; w > 1; w >>= 1, level++) {
    const uint64_t above = below + 32 * w;
    for (uint64_t i = threadIdx.x; i < w / 2; i += blockDim.x) {
      uint32_t l[8], r[8], h[8];
      load_key(below + 64 * i, l);
      load_key(below + 64 * i + 32, r);
      b3_short<74>(key, [&](int j) {
        return j == 0 ? 2u : j == 1 ? level : j < 10 ? word_le(i, j - 2) : j < 42 ? cv_byte(l, j - 10) : cv_byte(r, j - 42);
      }, h);
      store_cv(above + 32 * i, h);
    }
    __syncthreads();
    below = above;
  }
}

// hash_leaf with each of a tile's chunks (32 digests) on its own thread, four threads a tile and 32 tiles a CTA of 128, and
// with partial tiles: a tile of r < 64 rows (the decode headline's m = 32) is its 2r digests in ⌈2r / 32⌉ chunks under
// BLAKE3's tree, as the reference's leaf() hashes it.  Launch ⌈tiles / 32⌉ CTAs of 128.  Same arguments as hash_leaf.
extern "C" __global__ void __launch_bounds__(128) hash_leaf_p(const Args a) {
  __shared__ uint32_t cvs[32][4][8];
  const int rows = a.i[0], cols = a.i[1], tn = cols / 64, per = cols / 32;
  const int lt = threadIdx.x / 4, ch = threadIdx.x % 4;
  const int t = blockIdx.x * 32 + lt, tiles = (rows + 63) / 64 * tn;
  const int ti = t / tn, tj = t % tn;
  const int r = t < tiles ? min(64, rows - 64 * ti) : 0, nd = 2 * r, nch = (nd + 31) / 32;
  uint32_t key[8];
  load_key(a.p[2], key);
  if (ch < nch) {
    const uint32_t* dg = reinterpret_cast<const uint32_t*>(a.p[0]);
    const int d0 = 32 * ch, dn = min(32, nd - d0), blocks = (dn + 1) / 2;
    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];
#pragma unroll 1
    for (int b = 0; b < blocks; b++) {
      uint32_t m[16];
#pragma unroll
      for (int w = 0; w < 16; w++) {
        const int d = d0 + 2 * b + w / 8;
        m[w] = d < nd ? dg[((size_t)(64 * ti + d / 2) * per + 2 * tj + d % 2) * 8 + w % 8] : 0u;
      }
      const bool last = b == blocks - 1;
      const uint32_t f = B3_KEYED | (b == 0 ? B3_START : 0) | (last ? B3_END | (nch == 1 ? B3_ROOT : 0) : 0);
      b3c(cv, m, (uint64_t)ch, last && (dn & 1) ? 32 : 64, f, cv);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) cvs[lt][ch][i] = cv[i];
  }
  __syncthreads();
  if (ch != 0 || t >= tiles) return;
  uint32_t l[8], h[8];
  if (nch == 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) h[i] = cvs[lt][0][i];
  } else if (nch == 2) {
    b3_parent(key, cvs[lt][0], cvs[lt][1], B3_ROOT, h);
  } else if (nch == 3) {
    b3_parent(key, cvs[lt][0], cvs[lt][1], 0, l);
    b3_parent(key, l, cvs[lt][2], B3_ROOT, h);
  } else {
    uint32_t r2[8];
    b3_parent(key, cvs[lt][0], cvs[lt][1], 0, l);
    b3_parent(key, cvs[lt][2], cvs[lt][3], 0, r2);
    b3_parent(key, l, r2, B3_ROOT, h);
  }
  store_cv(a.p[1] + (size_t)t * 32, h);
}

// hash_sm120.cuh: the -h1 kernels tuned for sm_120, beside hash.cuh's as pushed (whose code is unchanged): the same
// bytes in fewer serial steps.  hash.cu builds both into hash_sm120a.cubin; GPU 1's sm_120 port can include this header.
#pragma once
#include "hash.cuh"

// ---------------------------------------------------------------- -h1 on sm_120: the same bytes, fewer serial steps
// The kernels below commit exactly what hash_rows, hash_nodes, hash_leaf and hash_msg do (#449's frame-b3 at 6a1a051f: one-block
// leaf headers, one-compression nodes under level keys); the host picks either form.

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

// shared words hash_rows_w needs per warp for a row of rb bytes: its chunks' chaining values
__host__ __device__ inline int rows_w_words(int rb) { return 8 * ((ROW_PREFIX + rb + 1023) / 1024); }

// A row's frame-b3 leaf by one warp: lane c hashes chunks c, c + 32, … (the header's block, then the row's, block-aligned: a
// block of a row of whole words is 16 loads, the next block's issued while this one compresses; the rest through
// row_msg_word), the chaining values go to cvs (8 words a chunk), and the chunk tree is built as hash_rows builds it.  Lane 0
// returns the leaf in h.
DEV void row_leaf_w(const uint8_t* src, uint32_t rb, uint64_t row, const uint32_t* key, uint32_t (*cvs)[8], int lane, uint32_t* h) {
  const uint64_t len = ROW_PREFIX + (uint64_t)rb;
  const int n = (int)((len + 1023) / 1024);
  const uint32_t* src_w = reinterpret_cast<const uint32_t*>(src);
  const bool aligned = rb % 4 == 0 && (reinterpret_cast<uint64_t>(src) & 3) == 0;
  for (int c = lane; c < n; c += 32) {
    const uint64_t left = len - 1024ull * c;
    const uint32_t clen = (uint32_t)(left < 1024 ? left : 1024);
    const int blocks = (int)((clen + 63) / 64);
    // block b's 16 message words are row words j0 − 16 … j0 − 1 when they all lie in the row
    auto fast = [&](int b) { const uint64_t j0 = 256ull * c + 16ull * b; return aligned && j0 >= 16 && 4 * j0 <= rb; };
    uint32_t cv[8], nxt[16];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    bool nxt_fast = fast(0);
    if (nxt_fast) {
#pragma unroll
      for (int q = 0; q < 16; q++) nxt[q] = src_w[256ull * c - 16 + q];
    }
    for (int b = 0; b < blocks; b++) {
      const uint64_t j0 = 256ull * c + 16ull * b;
      uint32_t m[16];
      const bool cur_fast = nxt_fast;
#pragma unroll
      for (int q = 0; q < 16; q++) m[q] = nxt[q];
      if (b + 1 < blocks) {
        nxt_fast = fast(b + 1);
        if (nxt_fast) {
#pragma unroll
          for (int q = 0; q < 16; q++) nxt[q] = src_w[j0 + 16 - 16 + q];
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
// the levels hash_nodes writes one launch at a time, in the same places, each node one compression under its level's key.
// p0 the tree (level 0 at offset 0), p1 the level keys (hash_node_keys: 32 bytes a level); i0 the width of level 0
extern "C" __global__ void __launch_bounds__(1024) hash_tree_small(const Args a) {
  uint64_t below = a.p[0];
  int level = 0;
  for (uint64_t w = (uint64_t)a.i[0]; w > 1; w >>= 1, level++) {
    const uint64_t above = below + 32 * w;
    uint32_t key[8];
    load_key(a.p[1] + 32ull * level, key);
    for (uint64_t i = threadIdx.x; i < w / 2; i += blockDim.x) {
      uint32_t m[16], h[8];
      load_key(below + 64 * i, m);
      load_key(below + 64 * i + 32, m + 8);
      blake3_block(key, m, h);
      store_cv(above + 32 * i, h);
    }
    __syncthreads();
    below = above;
  }
}

// hash_leaf with each of a tile's chunks (32 digests) on its own thread, four threads a tile and blockDim / 4 tiles a CTA (32
// at 128 threads, the most cvs holds), and
// with partial tiles: a tile of r < 64 rows (the decode headline's m = 32) is its 2r digests in ⌈2r / 32⌉ chunks under
// BLAKE3's tree, as the reference's leaf() hashes it.  Launch ⌈tiles / (blockDim / 4)⌉ CTAs of a multiple of 4 threads, at
// most 128.  Same arguments as hash_leaf.
extern "C" __global__ void __launch_bounds__(128) hash_leaf_p(const Args a) {
  __shared__ uint32_t cvs[32][4][8];
  const int rows = a.i[0], cols = a.i[1], tn = cols / 64, per = cols / 32;
  const int lt = threadIdx.x / 4, ch = threadIdx.x % 4;
  const int t = blockIdx.x * (blockDim.x / 4) + lt, tiles = (rows + 63) / 64 * tn;
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

// ---------------------------------------------------------------- A's commitment at decode: latency
// b3c4: one BLAKE3 compression by the four lanes q = lane % 4 of a group, lane q holding column q of the state (v[q], v[4+q],
// v[8+q], v[12+q]); the diagonal step's operands come by three shuffles and go back by three.  Every lane of the group
// passes the same cv, msg, counter, len and flags and gets the whole output.  The group's lanes must be converged.
DEV uint32_t sel4(int q, uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3) { return q == 0 ? x0 : q == 1 ? x1 : q == 2 ? x2 : x3; }
DEV void g1(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t x, uint32_t y) {
  a = a + b + x; d = rotr(d ^ a, 16); c = c + d; b = rotr(b ^ c, 12);
  a = a + b + y; d = rotr(d ^ a, 8); c = c + d; b = rotr(b ^ c, 7);
}
DEV void b3c4(const uint32_t* cv, const uint32_t* msg, uint64_t counter, uint32_t len, uint32_t flags, uint32_t* out) {
  constexpr int PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  const int lane = threadIdx.x % 32, q = lane % 4, base = lane - q;
  const unsigned mask = 0xFu << base;
  uint32_t m[16], t[16];
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = msg[i];
  uint32_t a = sel4(q, cv[0], cv[1], cv[2], cv[3]), b = sel4(q, cv[4], cv[5], cv[6], cv[7]);
  uint32_t c = sel4(q, B3_IV[0], B3_IV[1], B3_IV[2], B3_IV[3]);
  uint32_t d = sel4(q, (uint32_t)counter, (uint32_t)(counter >> 32), len, flags);
#pragma unroll
  for (int r = 0; r < 7; r++) {
    g1(a, b, c, d, sel4(q, m[0], m[2], m[4], m[6]), sel4(q, m[1], m[3], m[5], m[7]));
    b = __shfl_sync(mask, b, base + ((q + 1) & 3));
    c = __shfl_sync(mask, c, base + ((q + 2) & 3));
    d = __shfl_sync(mask, d, base + ((q + 3) & 3));
    g1(a, b, c, d, sel4(q, m[8], m[10], m[12], m[14]), sel4(q, m[9], m[11], m[13], m[15]));
    b = __shfl_sync(mask, b, base + ((q + 3) & 3));
    c = __shfl_sync(mask, c, base + ((q + 2) & 3));
    d = __shfl_sync(mask, d, base + ((q + 1) & 3));
#pragma unroll
    for (int i = 0; i < 16; i++) t[i] = m[PERM[i]];
#pragma unroll
    for (int i = 0; i < 16; i++) m[i] = t[i];
  }
  const uint32_t lo = a ^ c, hi = b ^ d;                    // output words q and 4 + q
#pragma unroll
  for (int i = 0; i < 4; i++) {
    out[i] = __shfl_sync(mask, lo, base + i);
    out[4 + i] = __shfl_sync(mask, hi, base + i);
  }
}

// latency: n dependent compressions on one thread (i1 = 0, b3c) or on one group of four lanes (i1 = 1, b3c4), the SM clock
// cycles they took into p0 (u64), and the last chaining value into p0 + 16 (store_cv's uint4 stores need 16-byte
// alignment).  i0 n
extern "C" __global__ void __launch_bounds__(32) lat_chain(const Args a) {
  uint32_t cv[8], m[16];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = 0x9E3779B9u * (i + 1);
  const int n = a.i[0];
  const long long t0 = clock64();
  for (int k = 0; k < n; k++) {
    if (a.i[1]) b3c4(cv, m, (uint64_t)k, 64, B3_KEYED, cv);
    else b3c(cv, m, (uint64_t)k, 64, B3_KEYED, cv);
    m[0] ^= cv[0];
  }
  const long long t1 = clock64();
  if (threadIdx.x == 0) {
    *reinterpret_cast<unsigned long long*>(a.p[0]) = (unsigned long long)(t1 - t0);
    store_cv(a.p[0] + 16, cv);
  }
}

// A's row leaves and its tree in one launch: hash_rows_w's warps, then the last CTA to finish (a counter in p4, which it
// resets) writes the pads and every level as hash_pad and hash_tree_small do; launched with 128 threads (4 rows) a CTA.  p0 rows, p1 the tree, p2 domain id, p3 the level
// keys, p4 a zeroed u32 counter; i0 rows, i1 row bytes, i2 width (level 0, a power of two).  Dynamic shared memory:
// rows_w_words(rb) words per warp.
extern "C" __global__ void __launch_bounds__(128, 1) hash_rows_wt(const Args a) {
  extern __shared__ uint32_t sh[];
  __shared__ int last;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const uint64_t row = (uint64_t)blockIdx.x * (blockDim.x / 32) + warp;
  const uint32_t rb = (uint32_t)a.i[1];
  const int m = a.i[0], width = a.i[2];
  uint32_t key[8], h[8];
  load_key(a.p[2], key);
  if (row < (uint64_t)m) {
    row_leaf_w(reinterpret_cast<const uint8_t*>(a.p[0]) + row * rb, rb, row, key,
               reinterpret_cast<uint32_t(*)[8]>(sh + (size_t)warp * rows_w_words((int)rb)), lane, h);
    if (lane == 0) store_cv(a.p[1] + row * 32, h);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(reinterpret_cast<unsigned*>(a.p[4]), 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int r = m + (int)threadIdx.x; r < width; r += blockDim.x) {
    uint32_t pm[16];                                        // the pad's message, 0x03 ‖ u64 rank: one block of 9 bytes
#pragma unroll
    for (int w = 0; w < 16; w++) pm[w] = 0u;
    pm[0] = 3u | ((uint32_t)r << 8);
    pm[1] = (uint32_t)r >> 24;
    b3c(key, pm, 0, 9, B3_KEYED | B3_START | B3_END | B3_ROOT, h);
    store_cv(a.p[1] + (uint64_t)r * 32, h);
  }
  __syncthreads();
  uint64_t below = a.p[1];
  int level = 0;
  for (uint64_t w = (uint64_t)width; w > 1; w >>= 1, level++) {
    const uint64_t above = below + 32 * w;
    uint32_t lk[8];
    load_key(a.p[3] + 32ull * level, lk);
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
  if (threadIdx.x == 0) *reinterpret_cast<unsigned*>(a.p[4]) = 0u;
}

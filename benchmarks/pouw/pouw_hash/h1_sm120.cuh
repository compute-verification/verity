// h1_sm120.cuh: -h1's commitments of A and of the tiles in fewer launches on sm_120, committing exactly what hash.cuh's
// hash_rows, hash_pad, hash_node_keys, hash_nodes and hash_tree_leaves and hash_sm120.cuh's hash_msg_v and hash_leaf_p write
// (#449's frame-b3 at 6a1a051f) and what pearl_c_sm120.cu's seed_line_a derives.  pearl_c_sm120.cu includes it.
#pragma once
#include "../pearl_c/hash_sm120.cuh"

DEV uint4 ld_nc(const uint4* p) { return __ldg(p); }
// the same load, and an L2 miss fetches the 256 aligned bytes around it (the chunk's next blocks) from DRAM
DEV uint4 ld_nc_256(const uint4* p) {
#ifdef __CUDA_ARCH__
  uint4 r;
  asm("ld.global.nc.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
#else
  return __ldg(p);
#endif
}
DEV void ld_cv_cg(uint64_t p, uint32_t* h) {   // a node another CTA wrote, from L2
  const uint4 x = __ldcg(reinterpret_cast<const uint4*>(p)), y = __ldcg(reinterpret_cast<const uint4*>(p) + 1);
  h[0] = x.x; h[1] = x.y; h[2] = x.z; h[3] = x.w; h[4] = y.x; h[5] = y.y; h[6] = y.z; h[7] = y.w;
}

// A row's frame-b3 leaf by one warp, as row_leaf_w hashes it, with each block's 16 words as four 16-byte loads issued a
// block ahead (P256: with L2::256B).  Needs rb a multiple of 16 and src 16-byte aligned for the vector path; other blocks go
// through row_msg_word.
template <bool P256>
DEV void row_leaf_v(const uint8_t* src, uint32_t rb, uint64_t row, const uint32_t* key, uint32_t (*cvs)[8], int lane, uint32_t* h) {
  auto ld = [](const uint4* p) { return P256 ? ld_nc_256(p) : ld_nc(p); };
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
      for (int q = 0; q < 4; q++) nxt[q] = ld(src4 + (64ull * c - 4) + q);
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
          for (int q = 0; q < 4; q++) nxt[q] = ld(src4 + j0 / 4 + q);
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

// hash_rows, a warp per row and blockDim / 32 rows a CTA (row_leaf_v; h1_rows_p loads with L2::256B).  Dynamic shared
// memory: rows_w_words(rb) words a warp.  Same arguments as hash_rows.
template <bool P256> DEV void h1_rows_body(const Args& a, uint32_t* sh) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const uint64_t row = (uint64_t)blockIdx.x * (blockDim.x / 32) + warp;
  const uint32_t rb = (uint32_t)a.i[1];
  if (row >= (uint64_t)a.i[0]) return;
  uint32_t key[8], h[8];
  load_key(a.p[2], key);
  row_leaf_v<P256>(reinterpret_cast<const uint8_t*>(a.p[0]) + row * rb, rb, row, key,
                   reinterpret_cast<uint32_t(*)[8]>(sh + (size_t)warp * rows_w_words((int)rb)), lane, h);
  if (lane == 0) store_cv(a.p[1] + row * 32, h);
}
extern "C" __global__ void __launch_bounds__(256) h1_rows(const Args a) {
  extern __shared__ uint32_t sh[];
  h1_rows_body<false>(a, sh);
}
extern "C" __global__ void __launch_bounds__(256) h1_rows_p(const Args a) {
  extern __shared__ uint32_t sh[];
  h1_rows_body<true>(a, sh);
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

// level l's key of a frame-b3 tree, keyed(domain id, 0x02 ‖ u8 l): from the level-key table lk (32 B a level) or, at lk = 0,
// from the domain id
DEV void h1_level_key(uint64_t lk, const uint32_t* dom, int l, uint32_t* k) {
  if (lk) load_key(lk + 32ull * l, k);
  else b3_short<2>(dom, [&](int j) { return j == 0 ? 2u : (uint32_t)l; }, k);
}

// The level-key table of i0 domains (p0, 32 B each): domain d's key of level l < i1 at p1 + 32 (i1 d + l), hash_node_keys'
// bytes.  The keys depend on the domain id alone, so one launch before the calls (every unit of an epoch whose domains are
// known) takes them off each call's path: h1_tree's p9, h1_commit's p7 and h1_tiles' p9 read a domain's row of it.
extern "C" __global__ void __launch_bounds__(128) h1_level_keys(const Args a) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x, depth = a.i[1];
  if (t >= a.i[0] * depth) return;
  uint32_t dom[8], h[8];
  load_key(a.p[0] + 32ull * (t / depth), dom);
  h1_level_key(0, dom, t % depth, h);
  store_cv(a.p[1] + 32ull * t, h);
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
// p8 the level keys out (32 B a level; 0: none), p9 the level keys in (h1_level_keys' row; 0: from p1); i0 leaves, i1 width
// (a power of two), i2 the matmul's index
extern "C" __global__ void __launch_bounds__(1024) h1_tree(const Args a) {
  extern __shared__ uint32_t sh[];
  uint32_t (*keys)[8] = reinterpret_cast<uint32_t(*)[8]>(sh);
  uint32_t (*nodes)[8] = keys + 32;
  __shared__ bool last;
  const uint64_t tree = a.p[0], count = (uint64_t)a.i[0], tw = (uint64_t)a.i[1];
  const int depth = 63 - __clzll((long long)tw), S = (int)(tw < 2ull * blockDim.x ? tw : 2ull * blockDim.x), ls = 31 - __clz(S);
  const uint64_t first = (uint64_t)blockIdx.x * S;
  uint32_t dom[8];
  load_key(a.p[1], dom);
  for (int l = threadIdx.x; l < depth; l += blockDim.x) {
    h1_level_key(a.p[9], dom, l, keys[l]);
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
    for (int q = 0; q < 8; q++) nodes[j][q] = h[q];
  }
  __syncthreads();
  h1_levels(nodes, keys, tree, tw, 0, first, S);
  if (gridDim.x > 1) {
    if (threadIdx.x == 0) {
      __threadfence();
      last = atomicInc(reinterpret_cast<unsigned*>(a.p[2]), gridDim.x - 1) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const uint64_t lvl = 2 * tw - (2 * tw >> ls);
    for (int j = threadIdx.x; j < (int)gridDim.x; j += blockDim.x) ld_cv_cg(tree + 32 * (lvl + j), nodes[j]);
    __syncthreads();
    h1_levels(nodes, keys, tree, tw, ls, 0, gridDim.x);
  }
  if (!a.p[4] || threadIdx.x != 0) return;
  const uint32_t* salt = reinterpret_cast<const uint32_t*>(a.p[6]);
  const uint32_t* root_b = reinterpret_cast<const uint32_t*>(a.p[7]);
  uint32_t m[16], cv[8], seed[8], h[8];
  load_key(a.p[5], cv);
#pragma unroll
  for (int i = 0; i < 8; i++) m[i] = salt[i], m[8 + i] = nodes[0][i];
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

// ---------------------------------------------------------------- decode: a commitment in one launch, its tree by the last CTA
// Whether this CTA is the grid's last to arrive at counter p (u32, zero between launches: atomicInc wraps it back), after
// every thread's global stores; the last CTA then reads the others' through L2 (ld_cv_cg).
DEV bool h1_arrive(uint64_t p) {
  __shared__ bool last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicInc(reinterpret_cast<unsigned*>(p), gridDim.x - 1) == gridDim.x - 1;
  __syncthreads();
  if (last) __threadfence();
  return last;
}

// A frame-b3 tree of `width` (a power of two, <= 2 blockDim) over level 0's first `count` nodes, which the grid wrote: the
// pads, the level keys (from the table lk, or the domain id at lk = 0) and every level above (h1_levels), in one CTA.  The
// root ends in nodes[0].
DEV void h1_top(uint64_t tree, const uint32_t* dom, uint64_t count, uint64_t width, uint32_t (*nodes)[8], uint32_t (*keys)[8],
                uint64_t lk) {
  const int depth = 63 - __clzll((long long)width);
  for (int l = threadIdx.x; l < depth; l += blockDim.x) h1_level_key(lk, dom, l, keys[l]);
  for (uint64_t j = threadIdx.x; j < width; j += blockDim.x) {
    uint32_t h[8];
    if (j >= count) {
      b3_short<9>(dom, [&](int q) { return q == 0 ? 3u : word_le(j, q - 1); }, h);
      store_cv(tree + 32 * j, h);
    } else {
      ld_cv_cg(tree + 32 * j, h);
    }
#pragma unroll
    for (int q = 0; q < 8; q++) nodes[j][q] = h[q];
  }
  __syncthreads();
  h1_levels(nodes, keys, tree, width, 0, 0, (int)width);
}

// A's commitment in one launch, for a tree of width <= 2 blockDim (decode): h1_rows_p's row leaves (a warp a row), then in
// the last CTA by p3's counter the tree (h1_top), and from its root seed_A and E_A's line key as seed_line_a derives them.
// Dynamic shared memory: max(4 warps rows_w_words(rb), 32 (width + 32)) bytes.
// p0 A's rows, p1 the tree (level 0 at offset 0), p2 the domain id, p3 the counter (u32), p4 E_A's line key out, p5 the unit
// (salt ‖ root_B ‖ u64 index ‖ blake3("pearl-c/v0/seed-A"), 104 B), p6 seed_A out (0: none), p7 the level keys (h1_level_keys'
// row; 0: from p2); i0 rows, i1 row bytes, i2 width
extern "C" __global__ void __launch_bounds__(256) h1_commit(const Args a) {
  extern __shared__ uint32_t sh[];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const uint64_t count = (uint64_t)a.i[0], width = (uint64_t)a.i[2], row = (uint64_t)blockIdx.x * (blockDim.x / 32) + warp;
  const uint32_t rb = (uint32_t)a.i[1];
  uint32_t dom[8];
  load_key(a.p[2], dom);
  if (row < count) {
    uint32_t h[8];
    row_leaf_v<true>(reinterpret_cast<const uint8_t*>(a.p[0]) + row * rb, rb, row, dom,
                     reinterpret_cast<uint32_t(*)[8]>(sh + (size_t)warp * rows_w_words((int)rb)), lane, h);
    if (lane == 0) store_cv(a.p[1] + row * 32, h);
  }
  if (!h1_arrive(a.p[3])) return;
  uint32_t (*keys)[8] = reinterpret_cast<uint32_t(*)[8]>(sh);
  uint32_t (*nodes)[8] = keys + 32;
  h1_top(a.p[1], dom, count, width, nodes, keys, a.p[7]);
  if (threadIdx.x) return;
  const uint8_t* unit = reinterpret_cast<const uint8_t*>(a.p[5]);
  uint32_t key[8], seed[8], h[8], m[16];
  load_key(a.p[5] + 72, key);
  b3_short<104>(key, [&](int j) { return j < 32 ? (uint32_t)unit[j] : j < 64 ? cv_byte(nodes[0], j - 32) : (uint32_t)unit[j - 32]; },
                seed);
  const char line[] = "pearl/v4/FP8/noise-line";
  h1_label(line, sizeof line - 1, m);
  b3c(seed, m, 0, sizeof line - 1, B3_KEYED | B3_START | B3_END | B3_ROOT, h);
  store_cv(a.p[4], h);
  if (a.p[6]) store_cv(a.p[6], seed);
}

// The tile tree in one launch, for m_real <= 64 and a tree of width <= 256 (decode): a CTA of 128 a tile of r real rows,
// its 2r message digests a thread each (hash_msg_v's), its leaf a thread a chunk of 32 of them (hash_leaf_p's), its node at
// level 0 (h1_tree's from p3), and in the last CTA by p8's counter the levels above (h1_top).  The digests and the tile
// leaves are written too.  Launch one CTA a tile; dynamic shared memory 32 max(132, width + 32) bytes.
// p0 C~ (rows x cols FP32), p1 U, p2 digests out, p3 the message key, p4 tile leaves out, p5 the leaf key, p6 the tile tree
// (level 0 at offset 0), p7 its domain id, p8 the counter (u32), p9 its level keys (h1_level_keys' row; 0: from p7); i0 rows,
// i1 cols, i2 the tree's width
extern "C" __global__ void __launch_bounds__(128) h1_tiles(const Args a) {
  extern __shared__ uint32_t sh[];
  uint32_t (*dg)[8] = reinterpret_cast<uint32_t(*)[8]>(sh);
  uint32_t (*cvs)[8] = dg + 128;
  const int rows = a.i[0], cols = a.i[1], tn = cols / 64, per = cols / 32;
  const int t = blockIdx.x, ti = t / tn, tj = t % tn, d = threadIdx.x;
  const int r = min(64, rows - 64 * ti), nd = 2 * r, nch = (nd + 31) / 32;
  if (d < nd) {
    const size_t g = (size_t)(64 * ti + d / 2) * per + 2 * tj + d % 2;
    const uint4* c4 = reinterpret_cast<const uint4*>(a.p[0]) + g * 8;
    const uint4* u4 = reinterpret_cast<const uint4*>(a.p[1]) + g * 8;
    uint32_t c[32], u[32], key[8], h[8];
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const uint4 x = c4[q], y = u4[q];
      c[4 * q] = x.x; c[4 * q + 1] = x.y; c[4 * q + 2] = x.z; c[4 * q + 3] = x.w;
      u[4 * q] = y.x; u[4 * q + 1] = y.y; u[4 * q + 2] = y.z; u[4 * q + 3] = y.w;
    }
    load_key(a.p[3], key);
    b3_msg_digest(c, u, key, h);
    store_cv(a.p[2] + g * 32, h);
#pragma unroll
    for (int i = 0; i < 8; i++) dg[d][i] = h[i];
  }
  __syncthreads();
  uint32_t lk[8];
  load_key(a.p[5], lk);
  if (d < nch) {
    const int d0 = 32 * d, dn = min(32, nd - d0), blocks = (dn + 1) / 2;
    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = lk[i];
#pragma unroll 1
    for (int b = 0; b < blocks; b++) {
      uint32_t m[16];
#pragma unroll
      for (int w = 0; w < 16; w++) {
        const int e = d0 + 2 * b + w / 8;
        m[w] = e < nd ? dg[e][w % 8] : 0u;
      }
      const bool last = b == blocks - 1;
      const uint32_t f = B3_KEYED | (b == 0 ? B3_START : 0) | (last ? B3_END | (nch == 1 ? B3_ROOT : 0) : 0);
      b3c(cv, m, (uint64_t)d, last && (dn & 1) ? 32 : 64, f, cv);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) cvs[d][i] = cv[i];
  }
  __syncthreads();
  uint32_t dom[8];
  load_key(a.p[7], dom);
  if (d == 0) {
    uint32_t l[8], h[8];
    if (nch == 1) {
#pragma unroll
      for (int i = 0; i < 8; i++) h[i] = cvs[0][i];
    } else if (nch == 2) {
      b3_parent(lk, cvs[0], cvs[1], B3_ROOT, h);
    } else if (nch == 3) {
      b3_parent(lk, cvs[0], cvs[1], 0, l);
      b3_parent(lk, l, cvs[2], B3_ROOT, h);
    } else {
      uint32_t r2[8];
      b3_parent(lk, cvs[0], cvs[1], 0, l);
      b3_parent(lk, cvs[2], cvs[3], 0, r2);
      b3_parent(lk, l, r2, B3_ROOT, h);
    }
    store_cv(a.p[4] + (size_t)t * 32, h);
    b3_short<96>(dom, [&](int q) { return q < 64 ? header_byte(q, (uint64_t)t, TILE_SCHEMA, 19) : cv_byte(h, q - 64); }, l);
    store_cv(a.p[6] + (size_t)t * 32, l);
  }
  if (!h1_arrive(a.p[8])) return;
  uint32_t (*keys)[8] = reinterpret_cast<uint32_t(*)[8]>(sh);
  h1_top(a.p[6], dom, gridDim.x, (uint64_t)a.i[2], keys + 32, keys, a.p[9]);
}

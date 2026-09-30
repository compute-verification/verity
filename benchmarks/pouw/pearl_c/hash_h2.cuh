// Hashing format -h2 (frame-b3s, `verity.commitments.merkle`'s hash="blake3-s256"): -h1 with each tree leaf a Merkle tree
// over its value's 256-byte segments.  Segment j of a value of L bytes is the standard keyed BLAKE3 of its exact bytes (the
// last one short, the empty segment when L = 0) under keyed(domain id, 0x05 ‖ u64 rank ‖ u64 position ‖ u64 L ‖ u32 j ‖
// u16 len(schema) ‖ schema); a segment-tree node at level l is one block under keyed(domain id, 0x06 ‖ u8 l); an odd last
// node is promoted unchanged.  The tree over the leaves is frame-b3's (hash_node_keys, hash_tree_small, hash_nodes).
#pragma once
#include "hash_sm120.cuh"

constexpr int B3S_SEG = 256, B3S_MAX_SEGS = 512;

// The segment key's one-block message: the host's template (0x05, zeros where the fields go, u16 len(schema) at byte 29, the
// schema from byte 31; tlen = 31 + len(schema) bytes) with the rank, position, L and j filled in.
DEV void put_le(uint32_t* m, int off, uint64_t v, int n) {
#pragma unroll
  for (int i = 0; i < n; i++) m[(off + i) >> 2] |= (uint32_t)((v >> (8 * i)) & 0xFF) << (8 * ((off + i) & 3));
}
DEV void b3s_segment_key(const uint32_t* dom, const uint32_t* tmpl, uint32_t tlen, uint64_t rank, uint64_t position,
                         uint64_t len, uint32_t j, uint32_t* key) {
  uint32_t m[16];
#pragma unroll
  for (int w = 0; w < 16; w++) m[w] = tmpl[w];
  put_le(m, 1, rank, 8);
  put_le(m, 9, position, 8);
  put_le(m, 17, len, 8);
  put_le(m, 25, j, 4);
  b3c(dom, m, 0, tlen, B3_START | B3_END | B3_ROOT | B3_KEYED, key);
}

// Segment j of the value at v (L bytes): its keyed BLAKE3 under key into out.  aligned: v is 16-byte aligned and L a
// multiple of 16, so a whole 64-byte block is four 16-byte loads (the next one issued while this one compresses).
DEV void b3s_segment(const uint8_t* v, uint32_t L, uint32_t j, const uint32_t* key, bool aligned, uint32_t* out) {
  const uint32_t off = B3S_SEG * j, seg = L > off ? min((uint32_t)B3S_SEG, L - off) : 0;
  const int nb = seg == 0 ? 1 : (int)((seg + 63) / 64);
  uint32_t cv[8];
#pragma unroll
  for (int i = 0; i < 8; i++) cv[i] = key[i];
  if (aligned && seg == B3S_SEG) {
    const uint4* src = reinterpret_cast<const uint4*>(v + off);
    uint4 nx[4];
#pragma unroll
    for (int w = 0; w < 4; w++) nx[w] = src[w];
#pragma unroll
    for (int b = 0; b < 4; b++) {
      uint32_t m[16];
#pragma unroll
      for (int w = 0; w < 4; w++) { m[4 * w] = nx[w].x; m[4 * w + 1] = nx[w].y; m[4 * w + 2] = nx[w].z; m[4 * w + 3] = nx[w].w; }
      if (b < 3) {
#pragma unroll
        for (int w = 0; w < 4; w++) nx[w] = src[4 * (b + 1) + w];
      }
      b3c(cv, m, 0, 64, B3_KEYED | (b == 0 ? B3_START : 0) | (b == 3 ? B3_END | B3_ROOT : 0), cv);
    }
  } else {
    for (int b = 0; b < nb; b++) {
      const uint32_t base = off + 64 * b, blen = seg > 64u * b ? min(64u, seg - 64u * b) : 0;
      uint32_t m[16];
#pragma unroll
      for (int w = 0; w < 16; w++) {
        uint32_t word = 0;
#pragma unroll
        for (int t = 0; t < 4; t++)
          if ((uint32_t)(4 * w + t) < blen) word |= (uint32_t)v[base + 4 * w + t] << (8 * t);
        m[w] = word;
      }
      b3c(cv, m, 0, blen, B3_KEYED | (b == 0 ? B3_START : 0) | (b == nb - 1 ? B3_END | B3_ROOT : 0), cv);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// A whole 256-byte segment's hash from the 64 words a producer holds in registers, exactly as it stores them (after every
// cast and packing, in the row's byte order), under its key from b3s_segment_key: the leaves then bind the stored bytes.
DEV void b3s_segment_regs(const uint32_t* key, const uint32_t* w, uint32_t* out) {
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = key[i];
#pragma unroll
  for (int b = 0; b < 4; b++) b3c(out, w + 16 * b, 0, 64, B3_KEYED | (b == 0 ? B3_START : 0) | (b == 3 ? B3_END | B3_ROOT : 0), out);
}

// The segment tree's level keys: keyed(domain id, 0x06 ‖ u8 l) into p0 + 32 l for l < i0.  p0 out, p1 domain id; i0 levels
extern "C" __global__ void __launch_bounds__(32) b3s_level_keys(const Args a) {
  const int l = threadIdx.x;
  if (l >= a.i[0]) return;
  uint32_t dom[8], m[16] = {0}, h[8];
  load_key(a.p[1], dom);
  m[0] = 0x06u | ((uint32_t)l << 8);
  b3c(dom, m, 0, 2, B3_START | B3_END | B3_ROOT | B3_KEYED, h);
  store_cv(a.p[0] + 32ull * l, h);
}

// Row leaves: a CTA a row, a thread a segment (at least ⌈L / 256⌉ threads, a multiple of 32, at most 512), then the segment
// tree in shared memory with the odd-node rule; row r's leaf (rank = position = i2 + r) into p1 + 32 r.  p0 rows (L bytes
// each, back to back), p1 out, p2 domain id, p3 the segment level keys (32 B a level), p4 the key template (64 B); i0 rows,
// i1 L (at most 512 segments), i2 the first row's rank, i3 the template's length
extern "C" __global__ void __launch_bounds__(512) hash_rows_b3s(const Args a) {
  __shared__ uint32_t lv[B3S_MAX_SEGS][8];
  const uint64_t row = blockIdx.x;
  const uint32_t L = (uint32_t)a.i[1], nseg = L == 0 ? 1 : (L + B3S_SEG - 1) / B3S_SEG;
  const int j = threadIdx.x;
  const uint64_t rank = (uint64_t)a.i[2] + row;
  const uint8_t* v = reinterpret_cast<const uint8_t*>(a.p[0]) + row * (uint64_t)L;
  if ((uint32_t)j < nseg) {
    uint32_t dom[8], tmpl[16], key[8], h[8];
    load_key(a.p[2], dom);
#pragma unroll
    for (int w = 0; w < 16; w++) tmpl[w] = reinterpret_cast<const uint32_t*>(a.p[4])[w];
    b3s_segment_key(dom, tmpl, (uint32_t)a.i[3], rank, rank, L, (uint32_t)j, key);
    const bool aligned = (L % 16 == 0) && ((a.p[0] & 15) == 0);
    b3s_segment(v, L, (uint32_t)j, key, aligned, h);
#pragma unroll
    for (int i = 0; i < 8; i++) lv[j][i] = h[i];
  }
  __syncthreads();
  int level = 0;
  for (uint32_t w = nseg; w > 1; w = (w + 1) / 2, level++) {
    const uint32_t half = w / 2;
    uint32_t h[8];
    if ((uint32_t)j < half) {
      uint32_t lk[8], m[16];
      load_key(a.p[3] + 32ull * level, lk);
#pragma unroll
      for (int q = 0; q < 8; q++) { m[q] = lv[2 * j][q]; m[8 + q] = lv[2 * j + 1][q]; }
      blake3_block(lk, m, h);
    } else if ((uint32_t)j == half && (w & 1)) {
#pragma unroll
      for (int q = 0; q < 8; q++) h[q] = lv[w - 1][q];
    }
    __syncthreads();
    if ((uint32_t)j < half || ((uint32_t)j == half && (w & 1))) {
#pragma unroll
      for (int q = 0; q < 8; q++) lv[j][q] = h[q];
    }
    __syncthreads();
  }
  if (j == 0) store_cv(a.p[1] + row * 32, lv[0]);
}

// Leaves of short values (at most 256 bytes: one segment, so the leaf is the segment's hash), a thread a value: value t
// (rank = position = i2 + t) into p1 + 32 t.  p0 values (L bytes each, back to back), p1 out, p2 domain id, p3 the key
// template; i0 values, i1 L, i2 the first value's rank, i3 the template's length
extern "C" __global__ void __launch_bounds__(128) hash_leaves_b3s(const Args a) {
  const uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (uint64_t)a.i[0]) return;
  const uint32_t L = (uint32_t)a.i[1];
  uint32_t dom[8], tmpl[16], key[8], h[8];
  load_key(a.p[2], dom);
#pragma unroll
  for (int w = 0; w < 16; w++) tmpl[w] = reinterpret_cast<const uint32_t*>(a.p[3])[w];
  const uint64_t rank = (uint64_t)a.i[2] + t;
  b3s_segment_key(dom, tmpl, (uint32_t)a.i[3], rank, rank, L, 0, key);
  b3s_segment(reinterpret_cast<const uint8_t*>(a.p[0]) + t * L, L, 0, key, (L % 16 == 0) && ((a.p[0] & 15) == 0), h);
  store_cv(a.p[1] + t * 32, h);
}

// Seeds: -h3's row seeds and a unit's seed_A, each the standard keyed BLAKE3 under the seed-A label's key (p2,
// blake3("pearl-c/v0/seed-A")) of salt ‖ root ‖ root_B ‖ u64 LE ι (104 bytes: two blocks).  i2 = 1: row r of i0 rows, root its
// leaf (p0 + 32 r) and ι = i1 · 2^32 + r (`pearl_c.row_seed`, the one-row unit's seed_A); i2 = 0: the unit's, root p0 and
// ι = i1 (`pearl_c.unit_seed_a`).  p0 leaves or the root, p1 out (32 B a seed), p2 the key, p3 salt, p4 root_B; i0 rows,
// i1 the unit index, i2 the mode
DEV void b3_seed(uint64_t key_p, uint64_t salt_p, const uint32_t* root, uint64_t root_b_p, uint64_t iota, uint32_t* out) {
  uint32_t key[8], m[16], m2[16] = {0};
  load_key(key_p, key);
  load_key(salt_p, m);
#pragma unroll
  for (int i = 0; i < 8; i++) m[8 + i] = root[i];
  b3c(key, m, 0, 64, B3_KEYED | B3_START, out);
  load_key(root_b_p, m2);
  m2[8] = (uint32_t)iota;
  m2[9] = (uint32_t)(iota >> 32);
  b3c(out, m2, 0, 40, B3_KEYED | B3_END | B3_ROOT, out);
}
extern "C" __global__ void __launch_bounds__(128) b3_seeds(const Args a) {
  const uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= (uint64_t)a.i[0]) return;
  uint32_t root[8], cv[8];
  load_key(a.p[0] + 32 * r, root);
  const uint64_t iota = a.i[2] ? ((uint64_t)(uint32_t)a.i[1] << 32 | r) : (uint64_t)(uint32_t)a.i[1];
  b3_seed(a.p[2], a.p[3], root, a.p[4], iota, cv);
  store_cv(a.p[1] + 32 * r, cv);
}

// Rows' segment trees over segment leaves already hashed (by A's producer, derive_rows_b3s), a CTA a row, a thread a segment,
// hash_rows_b3s's levels and odd-node rule; row r's leaf into p1 + 32 r, and with i3 = 1 its -h3 row seed (b3_seeds' mode 1,
// the unit index i2) into p6 + 32 r.  p0 the segment leaves (32 B each, row-major), p1 the row leaves, p2 the segment level
// keys, p3 the seed key, p4 salt, p5 root_B, p6 the row seeds; i0 rows, i1 segments a row (at most 512), i2 unit index, i3
extern "C" __global__ void __launch_bounds__(512) b3s_seg_trees(const Args a) {
  __shared__ uint32_t lv[B3S_MAX_SEGS][8];
  const uint64_t row = blockIdx.x;
  const uint32_t nseg = (uint32_t)a.i[1];
  const int j = threadIdx.x;
  if ((uint32_t)j < nseg) {
    uint32_t h[8];
    load_key(a.p[0] + 32 * (row * nseg + j), h);
#pragma unroll
    for (int i = 0; i < 8; i++) lv[j][i] = h[i];
  }
  __syncthreads();
  int level = 0;
  for (uint32_t w = nseg; w > 1; w = (w + 1) / 2, level++) {
    const uint32_t half = w / 2;
    uint32_t h[8];
    if ((uint32_t)j < half) {
      uint32_t lk[8], m[16];
      load_key(a.p[2] + 32ull * level, lk);
#pragma unroll
      for (int q = 0; q < 8; q++) { m[q] = lv[2 * j][q]; m[8 + q] = lv[2 * j + 1][q]; }
      blake3_block(lk, m, h);
    } else if ((uint32_t)j == half && (w & 1)) {
#pragma unroll
      for (int q = 0; q < 8; q++) h[q] = lv[w - 1][q];
    }
    __syncthreads();
    if ((uint32_t)j < half || ((uint32_t)j == half && (w & 1))) {
#pragma unroll
      for (int q = 0; q < 8; q++) lv[j][q] = h[q];
    }
    __syncthreads();
  }
  if (j == 0) {
    store_cv(a.p[1] + row * 32, lv[0]);
    if (a.i[3]) {
      uint32_t sd[8];
      b3_seed(a.p[3], a.p[4], lv[0], a.p[5], (uint64_t)(uint32_t)a.i[2] << 32 | row, sd);
      store_cv(a.p[6] + row * 32, sd);
    }
  }
}

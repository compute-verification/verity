// Hashing format -h2 (frame-b3s, `verity.ml.commitments.merkle`'s hash="blake3-s256"): -h1 with each tree leaf a Merkle tree
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

// STATS's part of block b (its FP32 words 16b .. 16b + 15) of a segment: the words' largest |x|, and words 0 and 8
DEV void b3s_block_stats(const uint32_t* m, int b, float& mx, float* smp) {
#pragma unroll
  for (int w = 0; w < 16; w++) mx = fmaxf(mx, fabsf(__uint_as_float(m[w])));
  smp[2 * b] = __uint_as_float(m[0]);
  smp[2 * b + 1] = __uint_as_float(m[8]);
}

// Segment j of the value at v (L bytes): its keyed BLAKE3 under key into out.  aligned: v is 16-byte aligned and L a
// multiple of 16, so a whole 64-byte block is four 16-byte loads (the next one issued while this one compresses).  STATS
// (a whole segment of 64 FP32 words): also mx = max(mx, |x|) over its words, and its every 8th word x[8t] into smp[t], t < 8,
// from the words as they are hashed.
template <bool STATS>
DEV void b3s_segment_t(const uint8_t* v, uint32_t L, uint32_t j, const uint32_t* key, bool aligned, uint32_t* out, float& mx,
                       float* smp) {
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
      if constexpr (STATS) b3s_block_stats(m, b, mx, smp);
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
      if constexpr (STATS) b3s_block_stats(m, b, mx, smp);
      b3c(cv, m, 0, blen, B3_KEYED | (b == 0 ? B3_START : 0) | (b == nb - 1 ? B3_END | B3_ROOT : 0), cv);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = cv[i];
}
DEV void b3s_segment(const uint8_t* v, uint32_t L, uint32_t j, const uint32_t* key, bool aligned, uint32_t* out) {
  float mx = 0.0f;
  b3s_segment_t<false>(v, L, j, key, aligned, out, mx, nullptr);
}

// The segment tree's level-l node key, keyed(domain id, 0x06 ‖ u8 l)
DEV void b3s_level_key(const uint32_t* dom, int l, uint32_t* h) {
  uint32_t m[16] = {0};
  m[0] = 0x06u | ((uint32_t)l << 8);
  b3c(dom, m, 0, 2, B3_START | B3_END | B3_ROOT | B3_KEYED, h);
}

// The segment tree's level keys (b3s_level_key) into p0 + 32 l for l < i0.  p0 out, p1 domain id; i0 levels
extern "C" __global__ void __launch_bounds__(32) b3s_level_keys(const Args a) {
  const int l = threadIdx.x;
  if (l >= a.i[0]) return;
  uint32_t dom[8], h[8];
  load_key(a.p[1], dom);
  b3s_level_key(dom, l, h);
  store_cv(a.p[0] + 32ull * l, h);
}

// Row leaves: a CTA a row, a thread a segment (at least ⌈L / 256⌉ threads, a multiple of 32, at most 512), the segment
// tree's ⌈log2 segments⌉ level keys by as many threads after their segments (b3s_level_key, into shared memory), then the
// tree in shared memory with the odd-node rule; row r's leaf (rank = position = i2 + r) into p1 + 32 r.  p0 rows (L bytes
// each, back to back), p1 out, p2 domain id, p3 (or 0) out, the level keys (32 B a level, b3s_level_keys' bytes, by the
// first CTA), p4 the key template (64 B); i0 rows, i1 L (at most 512 segments), i2 the first row's rank, i3 the template's
// length
constexpr int B3S_MAX_LEVELS = 9;
static_assert(1 << B3S_MAX_LEVELS == B3S_MAX_SEGS, "a row's segment tree has at most B3S_MAX_LEVELS levels");

// hash_rows_b3s's CTA over row blockIdx.x, its segment tree in lv (a segment's 32 bytes each) and its level keys in lks.
// STATS, for rows of FP32 words (L a multiple of 1,024, p0 16-byte aligned): also thread j's max |x| over its segment,
// returned, and its every 8th word x[64 j + 8 t] into samp[33 (j / 4) + 8 (j % 4) + t], t < 8, so that samp[33 i + c] is
// x[256 i + 8 c] (stats_s5's chain c at step i) and neither the writes nor a warp's reads over c conflict; and a row at or
// past i0 (m_real: m's filler rows) is read for its stats alone.
template <bool STATS>
DEV float rows_b3s(const Args& a, uint32_t (*lv)[8], uint32_t (*lks)[8], float* samp) {
  const uint64_t row = blockIdx.x;
  const uint32_t L = (uint32_t)a.i[1], nseg = L == 0 ? 1 : (L + B3S_SEG - 1) / B3S_SEG;
  const int j = threadIdx.x, levels = nseg > 1 ? 32 - __clz((int)(nseg - 1)) : 0;
  const uint64_t rank = (uint64_t)a.i[2] + row;
  const uint8_t* v = reinterpret_cast<const uint8_t*>(a.p[0]) + row * (uint64_t)L;
  const bool hashed = !STATS || row < (uint64_t)a.i[0];
  float mx = 0.0f;
  float* smp = STATS ? samp + 33 * (j >> 2) + 8 * (j & 3) : nullptr;
  if ((uint32_t)j < nseg) {
    if (hashed) {
      uint32_t dom[8], tmpl[16], key[8], h[8];
      load_key(a.p[2], dom);
#pragma unroll
      for (int w = 0; w < 16; w++) tmpl[w] = reinterpret_cast<const uint32_t*>(a.p[4])[w];
      b3s_segment_key(dom, tmpl, (uint32_t)a.i[3], rank, rank, L, (uint32_t)j, key);
      const bool aligned = (L % 16 == 0) && ((a.p[0] & 15) == 0);
      b3s_segment_t<STATS>(v, L, (uint32_t)j, key, aligned, h, mx, smp);
#pragma unroll
      for (int i = 0; i < 8; i++) lv[j][i] = h[i];
      if (j < levels) {
        b3s_level_key(dom, j, h);
#pragma unroll
        for (int i = 0; i < 8; i++) lks[j][i] = h[i];
        if (row == 0 && a.p[3]) store_cv(a.p[3] + 32ull * j, h);
      }
    } else if constexpr (STATS) {
      const float4* x = reinterpret_cast<const float4*>(v + B3S_SEG * j);
#pragma unroll
      for (int q = 0; q < 16; q++) {
        const float4 f = x[q];
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
        if (q % 2 == 0) smp[q / 2] = f.x;
      }
    }
  }
  if (!hashed) return mx;
  __syncthreads();
  int level = 0;
  for (uint32_t w = nseg; w > 1; w = (w + 1) / 2, level++) {
    const uint32_t half = w / 2;
    uint32_t h[8];
    if ((uint32_t)j < half) {
      uint32_t m[16];
#pragma unroll
      for (int q = 0; q < 8; q++) { m[q] = lv[2 * j][q]; m[8 + q] = lv[2 * j + 1][q]; }
      blake3_block(lks[level], m, h);
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
  return mx;
}

extern "C" __global__ void __launch_bounds__(512) hash_rows_b3s(const Args a) {
  __shared__ uint32_t lv[B3S_MAX_SEGS][8];
  __shared__ uint32_t lks[B3S_MAX_LEVELS][8];
  rows_b3s<false>(a, lv, lks, nullptr);
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

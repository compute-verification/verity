// Pearl-C -h3's A tree past decode's rows, in hash_decode.cuh's code form, writing the bytes hash_h3.cuh's launches write
// (hm_leaves and s512_frame_leaves, s512_nodes and s512_tree_small, s512_seeds; every format theirs, and every compression,
// builder and label midstate hash_decode.cuh's):
//
// * pl_leaves, A's level 0: PL_ROWS rows a CTA.  Each warp makes its rows' hm96 salt masks across its lanes (`pd_hm_mask`),
//   then warp 0's lane j makes row j's c, hm96 leaf and frame leaf, five compressions paired with warp 1, while warp 2's
//   lane j builds the frame leaf's first block and its state.
// * pl_levels, PL_LEVELS levels a launch: a CTA's 2^levels nodes of the level below to the one node above them, a level's
//   nodes from its template (`pl_level_template`, as pd_level_template) on the two pairs' consumers, each level's states
//   kept in shared memory for the next.
// * pl_seed_a, seed_A and line_a from the root (pd_seed_a); pl_seeds, the rows' seeds and line keys (pd_seed), a row a lane.
//
// hash_decode.cuh's messages hold a rank or index below 256, whose `merkle._uint` is one byte.  Here one of 256 to 65,535
// (`wide`) is two, which moves every part after it a byte later: the frame leaf's hl at PD_LEAF_AT + 2, a node's children
// at PD_LEFT_AT + 1 and PD_RIGHT_AT + 1, its index bytes 130-131 (block 1's word 0, bits 32-47).  A CTA's rows, and a level's
// nodes in a CTA, are all below 256 or all above.  The variant V gives the compressions' form (hash_decode.cuh's bits);
// PD_PAIR is required.  pearl_c_sm90.cu builds PL_INSTANCE(851) (run.py's PL_VARIANT), and run.py's Pipeline launches
// them past decode's rows (its `chain`, "pl"); hash_prefill_host.cpp builds them for the host tests.
#pragma once
#include "../pearl_c_sm120/hash_decode.cuh"

constexpr int PL_ROWS = 16, PL_LEVELS = 7, PL_MAX_ROWS = 1 << 16;
constexpr int PL_LEAF_WIDE = PD_LEAF_AT + 2, PL_LEFT_WIDE = PD_LEFT_AT + 1, PL_RIGHT_WIDE = PD_RIGHT_AT + 1;
constexpr int PL_FTW = 33;   // a row's frame leaf blocks in shared memory, a word apart so the builders' lanes miss each other's banks

// the frame leaf of rank r's two blocks into f (32 words, shared), as pd_commit's frame builder; its hl at PD_LEAF_AT, or
// PL_LEAF_WIDE past rank 255
DEV void pl_frame_msg(uint64_t dom, uint64_t r, uint64_t* f) {
  for (int t = 0; t < 32; t++) f[t] = 0;
  WMsg m{f, 0};
  wm_frame(m, "leaf", dom);
  rc::put_uint(m, r);
  rc::put_uint(m, r);
  rc::put_part(m, HM96_SCHEMA, 22);
  const int at = wm_slot64(m);
  if (rc::put_padding(m) != 2 || at - 128 != (r >= 256 ? PL_LEAF_WIDE : PD_LEAF_AT)) __trap();
}

// node (level, i)'s message (48 words: 3 blocks), its children zero, i < 65,536: pd_node_msg's, with the wide offsets
DEV void pl_node_msg(uint64_t dom, int level, int i, uint64_t* buf) {
  for (int t = 0; t < 48; t++) buf[t] = 0;
  WMsg m{buf, 0};
  wm_frame(m, "node", dom);
  rc::put_uint(m, (uint64_t)level);
  rc::put_uint(m, (uint64_t)i);
  const int left = wm_slot64(m), right = wm_slot64(m), wide = i >= 256;
  if (rc::put_padding(m) != 3 || left - 128 != PD_LEFT_AT + wide || right - 128 != PD_RIGHT_AT + wide) __trap();
}
// level l's node template (PD_TM words) for indices below 256 or (wide) from 256 on: the state after its first block, then
// blocks 1 and 2 with the index's bytes zero.  Checked against the builder at the next index.  buf: 96 words
template <int V> DEV void pl_level_template(uint64_t dom, int level, bool wide, uint64_t* tm, uint64_t* buf) {
  const int i0 = wide ? 256 : 0;
  pl_node_msg(dom, level, i0, buf);
  pl_node_msg(dom, level, i0 + 1, buf + 48);
  const uint64_t bit = wide ? 1ull << 32 : 1ull << 40;
  for (int t = 0; t < 48; t++)
    if ((buf[t] ^ buf[48 + t]) != (t == 16 ? bit : 0ull)) __trap();
  for (int t = 0; t < 32; t++) tm[8 + t] = buf[16 + t];
  if (wide) tm[8] &= ~(0xFFFFull << 32);
  uint64_t st[8];
  rs_iv(st);
  pd_cmp<V>(st, buf);
#pragma unroll
  for (int t = 0; t < 8; t++) tm[t] = st[t];
}
// node i of a level from its children's states l and r and the level's template tm (wide: i from 256 on); out its state
template <int V, int P = 0>
DEV void pl_node(const uint64_t* tm, int i, bool wide, const uint64_t* l, const uint64_t* r, uint64_t* st) {
  uint64_t w[32];
#pragma unroll
  for (int t = 0; t < 32; t++) w[t] = tm[8 + t];
  if (wide) {
    w[0] |= (uint64_t)i << 32;
    wm_put64<PL_LEFT_WIDE>(w, l);
    wm_put64<PL_RIGHT_WIDE>(w, r);
  } else {
    w[0] |= (uint64_t)i << 40;
    wm_put64<PD_LEFT_AT>(w, l);
    wm_put64<PD_RIGHT_AT>(w, r);
  }
#pragma unroll
  for (int t = 0; t < 8; t++) st[t] = tm[t];
#pragma unroll 1
  for (int b = 0; b < 2; b++) {
    uint64_t m[16];
#pragma unroll
    for (int t = 0; t < 16; t++) m[t] = b ? w[16 + t] : w[t];
    pd_cmp<V, P>(st, m);
  }
}

// pl_leaves_V: A's level 0, PL_ROWS rows a CTA of 128 threads (8 PD_PAIR_WORDS bytes of dynamic shared memory: pair 0).  p0
// the row digests (64 B a row), p1 A's tree (level 0 out, 64 B a node), p2 the prover's salt key, p3 the hm96 constants, p4
// the unit block (its u64 LE index at byte 96 the salt's nonce), p5 dom_a; i0 rows (at most PL_MAX_ROWS)
template <int V> DEV void pl_leaves_t(const Args a) {
  static_assert((V & PD_PAIR) != 0, "pl_leaves' leaf chain is paired");
  __shared__ uint32_t ys[PL_ROWS][48];
  __shared__ uint64_t mk[PL_ROWS][8], ftw[PL_ROWS * PL_FTW], ftm[PL_ROWS][8];
  const int count = a.i[0], tid = threadIdx.x, warp = tid / 32, lane = tid % 32, r0 = PL_ROWS * blockIdx.x;
  if (count > PL_MAX_ROWS || blockDim.x != 128 || pd_dyn_bytes() < 8u * PD_PAIR_WORDS) __trap();
  const uint32_t* unit = reinterpret_cast<const uint32_t*>(a.p[4]);
  const uint64_t nonce = (uint64_t)unit[24] | (uint64_t)unit[25] << 32, hm = a.p[3];
  for (int j = warp; j < PL_ROWS; j += 4)
    if (r0 + j < count) pd_hm_mask(a.p[2], nonce, (uint64_t)(r0 + j), hm, ys[j], lane, mk[j]);
  __syncthreads();
  const int j = lane, r = r0 + j;
  const bool on = j < PL_ROWS && r < count, wide = r0 >= 256;
  if (warp == 1) {
    pd_serve<V>(0);
    return;
  }
  if (warp == 2) {                                  // the frame leaf's first block, and its state
    if (on) {
      uint64_t* f = ftw + PL_FTW * j;
      pl_frame_msg(a.p[5], (uint64_t)r, f);
      uint64_t st[8], w[16];
#pragma unroll
      for (int q = 0; q < 16; q++) w[q] = f[q];
      rs_iv(st);
      pd_cmp<V>(st, w);
#pragma unroll
      for (int q = 0; q < 8; q++) ftm[j][q] = st[q];
    }
    pd_bar_arrive(PD_LAST, 64);
    return;
  }
  if (warp != 0) return;
  // c over y, the hm96 leaf over (x ⊕ M·y, c), then the frame leaf over hl: five paired compressions at one site
  const int jj = on ? j : 0;
  const uint64_t* mid = reinterpret_cast<const uint64_t*>(hm);
  const uint32_t* y = ys[jj];
  uint64_t xr[8], st[8], cc[8], w[17];
  load_be64x8(a.p[0] + 64ull * (on ? r : 0), xr);
#pragma unroll 1
  for (int b = 0; b < 5; b++) {
    if (b == 0) {
#pragma unroll
      for (int q = 0; q < 8; q++) st[q] = mid[q];
#pragma unroll
      for (int q = 0; q < 16; q++) w[q] = rs_be64(y[2 * q], y[2 * q + 1]);
    } else if (b == 1) {
#pragma unroll
      for (int q = 0; q < 16; q++) w[q] = q < 8 ? rs_be64(y[32 + 2 * q], y[33 + 2 * q]) : 0;
      w[8] = 0x8000000000000000ull;
      w[15] = 8 * (128 + 192);
    } else if (b == 2) {
#pragma unroll
      for (int q = 0; q < 8; q++) {
        cc[q] = st[q];
        st[q] = mid[8 + q];
        w[q] = rs_bswap64(rs_bswap64(xr[q]) ^ mk[jj][q]);
        w[8 + q] = cc[q];
      }
    } else if (b == 3) {
      pd_padw(w, 256);
    } else {
      pd_bar_sync(PD_LAST, 64);
      const uint64_t* f = ftw + PL_FTW * jj;
#pragma unroll
      for (int q = 0; q < 16; q++) w[q] = f[16 + q];
      w[16] = 0;
      if (wide) wm_put64<PL_LEAF_WIDE>(w, st); else wm_put64<PD_LEAF_AT>(w, st);
#pragma unroll
      for (int q = 0; q < 8; q++) st[q] = ftm[jj][q];
    }
    pd_cmp<V, 1>(st, w);
  }
  pd_pstop();
  if (on) s512_store(a.p[1] + 64ull * r, st, 8);
}

// pl_levels_V: levels i1 + 1 … i1 + i2 of A's tree (i2 <= PL_LEVELS), CTA c from nodes c 2^i2 … (c + 1) 2^i2 − 1 of level
// i1 to node c of level i1 + i2, a CTA of 128 threads: warps 0 and 2 the consumers of pairs 0 and 1, warps 1 and 3 their
// producers (PD_PAIR_SMEM bytes of dynamic shared memory).  p0 level i1 (each level above it after it), p1 dom_a; i0 level
// i1's width (a power of two, at most PL_MAX_ROWS), i1 its level, i2 the levels
template <int V> DEV void pl_levels_t(const Args a) {
  static_assert((V & PD_PAIR) != 0, "pl_levels' nodes are paired");
  __shared__ uint64_t tm[PL_LEVELS * PD_TM], nodes[2 * 64 * 8];
  static_assert(2 * 64 * 8 >= 96 * PL_LEVELS, "nodes holds the templates' builders");
  const int width = a.i[0], base_level = a.i[1], levels = a.i[2], c = blockIdx.x, warp = threadIdx.x / 32;
  if (width > PL_MAX_ROWS || levels > PL_LEVELS || (width >> levels) << levels != width || blockDim.x != 128 ||
      pd_dyn_bytes() < (uint32_t)PD_PAIR_SMEM)
    __trap();
  const int t = threadIdx.x;
  if (t < levels) {   // the template over level base_level + t (a node's message holds its children's level), in its nodes' form
    const int n = (1 << levels) >> (t + 1);
    pl_level_template<V>(a.p[1], base_level + t, c * n >= 256, tm + PD_TM * t, nodes + 96 * t);
  }
  __syncthreads();
  if (warp == 1 || warp == 3) {
    pd_serve<V>(warp == 3);
    return;
  }
  const int ci = pd_ci();
  uint64_t below = a.p[0], above = a.p[0], l[8], r[8], st[8];
  const uint64_t* prev = nullptr;
  uint64_t* out = nodes;
  for (int j = 1, w = width; j <= levels; j++, w >>= 1) {
    above += 64ull * w;
    const int n = (1 << levels) >> j, g0 = c * n;
    if (pd_warp_has(0, n)) {
      const bool on = ci < n;
      const int at = on ? ci : 0;
      if (j == 1) {
        load_be64x8(below + 64ull * (2 * (g0 + at)), l);
        load_be64x8(below + 64ull * (2 * (g0 + at) + 1), r);
      } else {
#pragma unroll
        for (int q = 0; q < 8; q++) { l[q] = prev[16 * at + q]; r[q] = prev[16 * at + 8 + q]; }
      }
      pl_node<V, 1>(tm + PD_TM * (j - 1), g0 + at, g0 >= 256, l, r, st);
      if (on) {
        s512_store(above + 64ull * (g0 + at), st, 8);
#pragma unroll
        for (int q = 0; q < 8; q++) out[8 * at + q] = st[q];
      }
    }
    pd_bar_sync(PD_CONSUMERS, 64);
    prev = out;
    out = out == nodes ? nodes + 64 * 8 : nodes;
    below = above;
  }
  pd_pstop();
}

// pl_seed_a_V: seed_A and line_a from A's root (pd_seed_a), warp 0 paired with warp 1 (8 PD_PAIR_WORDS bytes of dynamic
// shared memory).  p0 the root (64 B), p1 the unit block, p2 seed_A out, p3 line_a out
template <int V> DEV void pl_seed_a_t(const Args a) {
  __shared__ uint32_t lk[8];
  if (blockDim.x != 64 || pd_dyn_bytes() < 8u * PD_PAIR_WORDS) __trap();
  if (threadIdx.x >= 32) {
    pd_serve<V>(0);
    return;
  }
  uint64_t root[8];
  load_be64x8(a.p[0], root);
  pd_seed_a<V, 1>(a.p[1], root, a.p[2], a.p[3], lk, threadIdx.x == 0);
  pd_pstop();
}

// pl_seeds_V: rows 32 c … 32 c + 31's seeds and line keys (s512_seeds' mode 1, pd_seed), row 32 c + j on warp 0's lane j,
// paired with warp 1 (8 PD_PAIR_WORDS bytes of dynamic shared memory).  p0 the row digests (64 B a row), p1 seeds out (32 B
// a row), p2 line keys out, p3 the unit block; i0 rows
template <int V> DEV void pl_seeds_t(const Args a) {
  if (blockDim.x != 64 || pd_dyn_bytes() < 8u * PD_PAIR_WORDS) __trap();
  if (threadIdx.x >= 32) {
    pd_serve<V>(0);
    return;
  }
  const int r = 32 * blockIdx.x + threadIdx.x;
  const bool on = r < a.i[0];
  const uint32_t* unit = reinterpret_cast<const uint32_t*>(a.p[3]);
  const uint64_t index = (uint64_t)unit[24] | (uint64_t)unit[25] << 32;
  uint64_t x[8];
  uint32_t seed[8], key[8];
  load_be64x8(a.p[0] + 64ull * (on ? r : 0), x);
  pd_seed<V, 1>(unit, x, index << 32 | (uint64_t)r, seed, key);
  pd_pstop();
  if (on) {
    store_cv(a.p[1] + 32ull * r, seed);
    store_cv(a.p[2] + 32ull * r, key);
  }
}

#define PL_INSTANCE(V)                                                                                                    \
  extern "C" __global__ void __launch_bounds__(128) pl_leaves_##V(const Args a) { pl_leaves_t<V>(a); }                  \
  extern "C" __global__ void __launch_bounds__(128) pl_levels_##V(const Args a) { pl_levels_t<V>(a); }                  \
  extern "C" __global__ void __launch_bounds__(64) pl_seed_a_##V(const Args a) { pl_seed_a_t<V>(a); }                   \
  extern "C" __global__ void __launch_bounds__(64) pl_seeds_##V(const Args a) { pl_seeds_t<V>(a); }
// PL_INSTANCE(V)'s entries in hash_decode_host.cpp's kernel list
#define PL_KERNELS(V)                                                                                                     \
  {"pl_leaves_" #V, pl_leaves_##V}, {"pl_levels_" #V, pl_levels_##V}, {"pl_seed_a_" #V, pl_seed_a_##V},                 \
      {"pl_seeds_" #V, pl_seeds_##V},

// Pearl-C -h3's per-call hashing at decode in fewer, shorter launches, writing the bytes Pearl-C's launches write
// (hash_h3.cuh, hash_fast.cuh; every format is theirs, and every constant byte comes from their tables):
//
// * pd_commit, one launch for A's commitment: a CTA a row.  Its first warps hash the row's 1 KiB segments (a thread a
//   segment, from hash_fast.cuh's header table) while two helper warps do what no data reaches: the row's hm96 salt
//   (ChaCha20, the byte-table mask M·y and c = SHA-512(salt prefix ‖ y)), its frame leaf's first block, and this CTA's share
//   of the tree's node and pad messages (a node's first block is (dom_a, level) alone below 256 nodes a level; its
//   template, written to scratch, waits only for the children).  Then thread 0 folds the row digest (`pd_fold`) and makes
//   the hm96 leaf and frame leaf (3 compressions), while another warp makes the row's seed and line key from the labels'
//   midstates (PD_LABELS; 3 compressions).  The last CTA to finish (a counter in scratch, reset by that CTA) builds the
//   tree's levels, 2 compressions a level, then seed_A and line_a (3).  Mode 1 stops after level 0, and pd_tree builds
//   the rest: the tree and seed_A on a lane of their own when nothing in the call reads line_a.
// * pd_tiles, one launch for the tile leaves: a warp a tile, a lane a segment, then the fold (`pd_fold`).
//
// pd_fold's rounds-only form (variant 1): the fold's message is all there before it starts, so the lanes of the folding
// warp compute every block's schedule with its round constants (K + W) into shared memory at once, and the folding lane
// runs the 80 rounds alone (rs_rounds).  Variant 0 is hash_h3.cuh's rs_fold.
//
// The builder (WMsg) writes `merkle._message`'s bytes as big-endian words, the data's 64-byte parts left zero and filled
// later (wm_put64, at any byte offset).  pd_host.cpp builds this file for the host tests.
#pragma once
#include "../../pouw/pearl_c/hash_h3.cuh"
#include "../../pouw/pearl_c_sm120/hash_fast.cuh"

#ifndef PD_HOST
#define PD_LDCG(p) __ldcg(p)
#else
#define PD_LDCG(p) (*(p))
#endif

// ---------------------------------------------------------------- tables, once per module (pd_heads)
enum { PD_SEED_A = 0, PD_NOISE_LINE = 1 };
__device__ uint64_t PD_LABELS[2][8];   // the states after the seed-A and noise-line label blocks

// ---------------------------------------------------------------- the message builder
struct WMsg {
  uint64_t* w;   // zeroed before
  int n;
};
DEV void wm_byte(WMsg& m, uint32_t v) {
  m.w[m.n >> 3] |= (uint64_t)(v & 0xFF) << (56 - 8 * (m.n & 7));
  m.n++;
}
DEV void wm_be(WMsg& m, uint64_t v, int bytes) {
  for (int i = bytes - 1; i >= 0; i--) wm_byte(m, (uint32_t)(v >> (8 * i)));
}
DEV void wm_uint(WMsg& m, uint64_t v) {   // `merkle._uint`, as sm_uint
  const int bytes = v ? (71 - __clzll((long long)v)) / 8 : 1;
  wm_be(m, (uint64_t)bytes, 8);
  wm_be(m, v, bytes);
}
DEV int wm_slot64(WMsg& m) {   // a 64-byte part left zero: its offset
  wm_be(m, 64, 8);
  const int at = m.n;
  m.n += 64;
  return at;
}
DEV void wm_frame(WMsg& m, const char* tag, int len, uint64_t dom) {   // as sm_frame
  for (int i = 0; i < 33; i++) wm_byte(m, FRAME_V3[i]);
  wm_be(m, (uint64_t)len, 4);
  for (int i = 0; i < len; i++) wm_byte(m, (uint32_t)(uint8_t)tag[i]);
  wm_be(m, 64, 8);
  const uint32_t* q = reinterpret_cast<const uint32_t*>(dom);
  for (int i = 0; i < 16; i++) {
    const uint32_t x = q[i];
    for (int j = 0; j < 4; j++) wm_byte(m, x >> (8 * j));
  }
}
DEV int wm_finish(WMsg& m) {   // SHA-512's padding, as sm_hash: the message's blocks
  const int total = m.n;
  wm_byte(m, 0x80);
  while ((m.n & 127) != 112) wm_byte(m, 0);
  wm_be(m, 0, 8);
  wm_be(m, 8ull * total, 8);
  return m.n / 128;
}
// a 64-byte value (8 big-endian words: a SHA-512 state, or a digest's bytes) into zeroed bytes at byte offset AT, a
// constant so the words stay in registers (the builders trap on any other offset)
template <int AT> DEV void wm_put64(uint64_t* w, const uint64_t* v) {
  constexpr int a = AT >> 3, r = 8 * (AT & 7);
#pragma unroll
  for (int t = 0; t < 8; t++) {
    if (r == 0) {
      w[a + t] |= v[t];
    } else {
      w[a + t] |= v[t] >> r;
      w[a + t + 1] |= v[t] << ((64 - r) & 63);
    }
  }
}
// the data's offsets in the blocks after the first, below 256 leaves (`merkle._uint` of a rank, level or index < 256 is
// one byte): the frame leaf's hl, a node's children
constexpr int PD_LEAF_AT = 41, PD_LEFT_AT = 11, PD_RIGHT_AT = 83;
DEV void load_be64x8(uint64_t p, uint64_t* v) {   // 64 bytes at device address p as 8 big-endian words
  const uint32_t* q = reinterpret_cast<const uint32_t*>(p);
#pragma unroll
  for (int t = 0; t < 8; t++) v[t] = rs_be64(PD_LDCG(q + 2 * t), PD_LDCG(q + 2 * t + 1));
}

// ---------------------------------------------------------------- the rounds-only compression
// one block's 80 round inputs K_i + W_i from its 16 message words
DEV void rs_sched(const uint64_t* m16, uint64_t* kw) {
  uint64_t w[16];
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = m16[i];
#pragma unroll
  for (int i = 0; i < 80; ++i) {
    uint64_t wi;
    if (i < 16) {
      wi = w[i];
    } else {
      const uint64_t w15 = w[(i - 15) & 15], w2 = w[(i - 2) & 15];
      wi = (rs_rotr(w2, 19) ^ rs_rotr(w2, 61) ^ (w2 >> 6)) + w[(i - 7) & 15] + (rs_rotr(w15, 1) ^ rs_rotr(w15, 8) ^ (w15 >> 7))
           + w[i & 15];
      w[i & 15] = wi;
    }
    kw[i] = RS_K[i] + wi;
  }
}
DEV void rs_rounds(uint64_t* st, const uint64_t* kw) {
  uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
  for (int i = 0; i < 80; ++i) {
    const uint64_t t1 = h + (rs_rotr(e, 14) ^ rs_rotr(e, 18) ^ rs_rotr(e, 41)) + ((e & f) ^ (~e & g)) + kw[i];
    const uint64_t t2 = (rs_rotr(a, 28) ^ rs_rotr(a, 34) ^ rs_rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

// ---------------------------------------------------------------- the pieces
// the fold's block b < s / 2 + 1 of s segment digests (rs_fold's message after Q)
DEV void pd_fold_block(const uint64_t* dig, int s, int b, uint64_t* w) {
  if (2 * b + 1 < s) {
#pragma unroll
    for (int t = 0; t < 16; t++) w[t] = dig[16 * b + t];
  } else if (2 * b < s) {
#pragma unroll
    for (int t = 0; t < 8; t++) { w[t] = dig[16 * b + t]; w[t + 8] = 0; }
    w[8] = 0x8000000000000000ull;
    w[15] = 8ull * (128 + 64 * s);
  } else {
#pragma unroll
    for (int t = 0; t < 16; t++) w[t] = 0;
    w[0] = 0x8000000000000000ull;
    w[15] = 8ull * (128 + 64 * s);
  }
}
// rs_fold's digest state over s digests (dig, shared), variant 0 by one thread; variant 1 by a whole warp, lane `lane` (its
// lanes compute the s / 2 + 1 blocks' schedules into kw, 80 words each, then lane 0 the rounds): st out on lane 0
DEV void pd_fold(const uint64_t* qst, const uint64_t* dig, int s, uint64_t* st, int variant, int lane, uint64_t* kw) {
  const int nb = s / 2 + 1;
  if (variant == 1) {
    uint64_t w[16];
    for (int b = lane; b < nb; b += 32) {
      pd_fold_block(dig, s, b, w);
      rs_sched(w, kw + 80 * b);
    }
    __syncwarp();
    if (lane) return;
#pragma unroll
    for (int t = 0; t < 8; t++) st[t] = qst[t];
#pragma unroll 1
    for (int b = 0; b < nb; b++) rs_rounds(st, kw + 80 * b);
    return;
  }
  if (lane) return;
  uint64_t w[16];
#pragma unroll
  for (int t = 0; t < 8; t++) st[t] = qst[t];
#pragma unroll 1
  for (int b = 0; b < nb; b++) {
    pd_fold_block(dig, s, b, w);
    rs_compress(st, w);
  }
}
// the hm96 salt pieces of leaf i (hm_leaves' work before x): the mask M·y (8 little-endian limbs) and c's state
DEV void pd_hm_salt(uint64_t salt_key, uint64_t nonce, uint64_t i, uint64_t hm, uint64_t* mk, uint64_t* c) {
  const uint64_t* mid = reinterpret_cast<const uint64_t*>(hm);
  const uint64_t* tables = reinterpret_cast<const uint64_t*>(hm + 128);
  uint32_t key[8], y[48];
  load_key(salt_key, key);
#pragma unroll
  for (int j = 0; j < 3; j++) chacha20_block(key, 3 * i + j, nonce, y + 16 * j);
#pragma unroll
  for (int k = 0; k < 8; k++) mk[k] = 0;
#pragma unroll
  for (int q = 0; q < 48; q++) {
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const uint4* e = reinterpret_cast<const uint4*>(tables + ((size_t)(4 * q + r) * 256 + ((y[q] >> (8 * r)) & 0xFF)) * 8);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const uint4 v = e[k];
        mk[2 * k] ^= (uint64_t)v.y << 32 | v.x;
        mk[2 * k + 1] ^= (uint64_t)v.w << 32 | v.z;
      }
    }
  }
  uint64_t w[16];
#pragma unroll
  for (int j = 0; j < 8; j++) c[j] = mid[j];
#pragma unroll
  for (int j = 0; j < 16; j++) w[j] = rs_be64(y[2 * j], y[2 * j + 1]);
  rs_compress(c, w);
#pragma unroll
  for (int j = 0; j < 16; j++) w[j] = j < 8 ? rs_be64(y[32 + 2 * j], y[33 + 2 * j]) : 0;
  w[8] = 0x8000000000000000ull;
  w[15] = 8 * (128 + 192);
  rs_compress(c, w);
}
// H_seed-A(salt ‖ x ‖ root_B ‖ u64 LE iota) from the label's midstate, and its line key (s512_seeds' bytes); x as 8 words
DEV void pd_seed(const uint32_t* unit, const uint64_t* x, uint64_t iota, uint32_t* seed, uint32_t* key) {
  uint64_t st[8], w[16];
#pragma unroll
  for (int t = 0; t < 8; t++) st[t] = PD_LABELS[PD_SEED_A][t];
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = i < 4 ? rs_be64(unit[2 * i], unit[2 * i + 1]) : i < 12 ? x[i - 4]
                                            : rs_be64(unit[8 + 2 * (i - 12)], unit[9 + 2 * (i - 12)]);
  rs_compress(st, w);
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = i < 4 ? rs_be64(unit[16 + 2 * i], unit[17 + 2 * i]) : 0;
  w[4] = rs_be64((uint32_t)iota, (uint32_t)(iota >> 32));
  w[5] = 0x8000000000000000ull;
  w[15] = 8 * (128 + 168);
  rs_compress(st, w);
  s512_out32(st, seed);
#pragma unroll
  for (int t = 0; t < 8; t++) st[t] = PD_LABELS[PD_NOISE_LINE][t];
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = i < 4 ? rs_be64(seed[2 * i], seed[2 * i + 1]) : 0;
  w[4] = 0x8000000000000000ull;
  w[15] = 8 * (128 + 32);
  rs_compress(st, w);
  s512_out32(st, key);
}

// ---------------------------------------------------------------- the tree's messages
// scratch: u32 counter at 0; node q's template at PD_TMPL + PD_TSTRIDE q: its first block's state (8 words), then blocks 1
// and 2 with the children zero (32 words)
constexpr int PD_TMPL = 256, PD_TSTRIDE = 384, PD_MAX_ROWS = 256;
constexpr int PD_KW_BLOCKS = 33;   // pd_commit's rounds-only fold: s / 2 + 1 <= 33 blocks (k <= 16384)
// node q's (level, index): levels in order, level l holding width >> (l + 1) nodes
DEV void pd_node_of(int q, int width, int* level, int* index) {
  int l = 0, nodes = width >> 1;
  while (q >= nodes) { q -= nodes; nodes >>= 1; l++; }
  *level = l;
  *index = q;
}
DEV void pd_node_template(uint64_t dom, int width, int q, uint64_t scratch, uint64_t* buf /*48 words*/) {
  int level, i;
  pd_node_of(q, width, &level, &i);
  for (int t = 0; t < 48; t++) buf[t] = 0;
  WMsg m{buf, 0};
  wm_frame(m, "node", 4, dom);
  wm_uint(m, (uint64_t)level);
  wm_uint(m, (uint64_t)i);
  const int left = wm_slot64(m), right = wm_slot64(m);
  if (wm_finish(m) != 3 || left - 128 != PD_LEFT_AT || right - 128 != PD_RIGHT_AT) __trap();
  uint64_t st[8];
  rs_iv(st);
  rs_compress(st, buf);
  uint64_t* out = reinterpret_cast<uint64_t*>(scratch + PD_TMPL + (uint64_t)PD_TSTRIDE * q);
#pragma unroll
  for (int t = 0; t < 8; t++) out[t] = st[t];
  for (int t = 0; t < 32; t++) out[8 + t] = buf[16 + t];
}
DEV void pd_pad(uint64_t dom, uint64_t rank, uint64_t tree, uint64_t* buf /*48 words*/) {   // as s512_pads
  for (int t = 0; t < 48; t++) buf[t] = 0;
  WMsg m{buf, 0};
  wm_frame(m, "pad", 3, dom);
  wm_uint(m, rank);
  const int blocks = wm_finish(m);
  uint64_t st[8];
  rs_iv(st);
  for (int b = 0; b < blocks; b++) rs_compress(st, buf + 16 * b);
  s512_store(tree + 64 * rank, st, 8);
}
// node q (at level `level`, index i) from its children in `below`, into `above`: blocks 1 and 2 from its template; out its state
DEV void pd_node(uint64_t scratch, int q, uint64_t below, uint64_t above, int i, uint64_t* st) {
  const uint64_t* tp = reinterpret_cast<const uint64_t*>(scratch + PD_TMPL + (uint64_t)PD_TSTRIDE * q);
  uint64_t w[32], l[8], r[8];
#pragma unroll
  for (int t = 0; t < 32; t++) w[t] = PD_LDCG(tp + 8 + t);
  load_be64x8(below + 128ull * i, l);
  load_be64x8(below + 128ull * i + 64, r);
  wm_put64<PD_LEFT_AT>(w, l);
  wm_put64<PD_RIGHT_AT>(w, r);
#pragma unroll
  for (int t = 0; t < 8; t++) st[t] = PD_LDCG(tp + t);
  rs_compress(st, w);
  rs_compress(st, w + 16);
  s512_store(above + 64ull * i, st, 8);
}
// the levels above level 0 of a tree of `width` leaves at `tree`, by the CTA's threads; out the root's state on thread 0
DEV void pd_levels(uint64_t scratch, uint64_t tree, int width, uint64_t* root) {
  uint64_t below = tree, st[8];
  int q0 = 0;
  for (int w = width; w > 1; w >>= 1) {
    const uint64_t above = below + 64ull * w;
    for (int i = threadIdx.x; i < w / 2; i += blockDim.x) {
      pd_node(scratch, q0 + i, below, above, i, st);
      if (w == 2) {
#pragma unroll
        for (int t = 0; t < 8; t++) root[t] = st[t];
      }
    }
    q0 += w / 2;
    below = above;
    __syncthreads();
  }
  if (width == 1 && threadIdx.x == 0) load_be64x8(tree, root);
}
DEV void pd_seed_a(uint64_t unit, const uint64_t* root, uint64_t seed_a, uint64_t line_a) {
  const uint32_t* u = reinterpret_cast<const uint32_t*>(unit);
  const uint64_t index = (uint64_t)u[24] | (uint64_t)u[25] << 32;
  uint32_t seed[8], key[8];
  pd_seed(u, root, index, seed, key);
  store_cv(seed_a, seed);
  store_cv(line_a, key);
}

// ---------------------------------------------------------------- the launches
// pd_heads: the label midstates (once per module, after s512_heads)
extern "C" __global__ void __launch_bounds__(32) pd_heads(const Args) {
  const int t = threadIdx.x;
  if (t < 2) {
    uint64_t st[8];
    s512_label(st, t == 0 ? S512_SEED_A : S512_NOISE_LINE);
#pragma unroll
    for (int q = 0; q < 8; q++) PD_LABELS[t][q] = st[q];
  }
}

// pd_commit: a CTA a row r < i0, 32 (ceil(s / 32) + 2) threads (s = k / 256).  p0 A's rows (FP32, k words each), p1 row
// digests out (64 B a row), p2 A's tree (level 0, then each level above), p3 row seeds out (32 B a row), p4 their line keys
// out, p5 the unit block (salt ‖ root_B ‖ u64 LE index), p6 dom_a (64 B), p7 the prover's salt key, p8 the hm96 constants,
// p9 scratch (PD_TMPL + PD_TSTRIDE (width − 1) bytes, its counter zero before the first launch), p10 seed_A out, p11 line_a
// out; i0 rows (at most PD_MAX_ROWS), i1 k, i2 mode (0: the tree and seed_A too, 1: level 0 only), i3 pd_fold's variant
extern "C" __global__ void __launch_bounds__(192) pd_commit(const Args a) {
  __shared__ uint64_t dig[RS_MAX_SEGS * 8], xr[8], mk[8], cc[8], ftm[8], ftw[32], hbuf[48], root[8];
  __shared__ uint64_t kw[PD_KW_BLOCKS * 80];
  __shared__ int last;
  const int count = a.i[0], k = a.i[1], s = k / 256, cw = (s + 31) / 32, variant = a.i[3];
  const int r = blockIdx.x, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int width = count <= 1 ? 1 : 1 << (32 - __clz(count - 1));
  if (count > PD_MAX_ROWS || (variant == 1 && s / 2 + 1 > PD_KW_BLOCKS)) __trap();
  const uint64_t dom = a.p[6], scratch = a.p[9], tree = a.p[2];
  const uint32_t* unit = reinterpret_cast<const uint32_t*>(a.p[5]);
  const uint64_t index = (uint64_t)unit[24] | (uint64_t)unit[25] << 32;
  if (warp < cw) {                                  // the row's segments
    if (tid < s) {
      uint64_t st[8], w[16];
      rs_seg_head(st, tid, s);
      const uint4* seg = reinterpret_cast<const uint4*>(a.p[0] + 4ull * k * r + (uint64_t)RS_SEG * tid);
#pragma unroll 1
      for (int c = 0; c < RS_SEG / 128; c++) rs_block(st, w, seg + 8 * c);
      rs_pad(st, 128 + RS_SEG);
#pragma unroll
      for (int q = 0; q < 8; q++) dig[8 * tid + q] = st[q];
    }
  } else if (warp == cw) {                          // the row's hm96 salt
    if (lane == 0) {
      uint64_t m_[8], c_[8];
      pd_hm_salt(a.p[7], index, (uint64_t)r, a.p[8], m_, c_);
#pragma unroll
      for (int q = 0; q < 8; q++) { mk[q] = m_[q]; cc[q] = c_[q]; }
    }
  } else if (lane == 0) {                           // the frame leaf's first block, this CTA's node templates and pads
    for (int t = 0; t < 32; t++) ftw[t] = 0;
    WMsg m{ftw, 0};
    wm_frame(m, "leaf", 4, dom);
    wm_uint(m, (uint64_t)r);
    wm_uint(m, (uint64_t)r);
    wm_be(m, 22, 8);
    for (int j = 0; j < 22; j++) wm_byte(m, HM96_SCHEMA[j]);
    const int at = wm_slot64(m);
    if (wm_finish(m) != 2 || at - 128 != PD_LEAF_AT) __trap();
    uint64_t st[8];
    rs_iv(st);
    rs_compress(st, ftw);
#pragma unroll
    for (int q = 0; q < 8; q++) ftm[q] = st[q];
    for (int q = r; q < width - 1; q += count) pd_node_template(dom, width, q, scratch, hbuf);
    for (int p = count + r; p < width; p += count) pd_pad(dom, (uint64_t)p, tree, hbuf);
  }
  __syncthreads();
  if (warp == 0) {                                  // the row digest
    uint64_t st[8];
    pd_fold(RS_Q_HEADS + 8 * s, dig, s, st, variant, lane, kw);
    if (lane == 0) {
      s512_store(a.p[1] + 64ull * r, st, 8);
#pragma unroll
      for (int q = 0; q < 8; q++) xr[q] = st[q];
    }
  }
  __syncthreads();
  if (tid == 0) {                                   // the hm96 leaf, then the frame leaf: level 0's node r
    const uint64_t* mid = reinterpret_cast<const uint64_t*>(a.p[8]);
    uint64_t lt[8], w[17];
#pragma unroll
    for (int q = 0; q < 8; q++) { lt[q] = mid[8 + q]; w[q] = rs_bswap64(rs_bswap64(xr[q]) ^ mk[q]); w[8 + q] = cc[q]; }
    rs_compress(lt, w);
    rs_pad(lt, 256);
#pragma unroll
    for (int q = 0; q < 16; q++) w[q] = ftw[16 + q];
    w[16] = 0;
    wm_put64<PD_LEAF_AT>(w, lt);
#pragma unroll
    for (int q = 0; q < 8; q++) lt[q] = ftm[q];
    rs_compress(lt, w);
    s512_store(tree + 64ull * r, lt, 8);
  } else if (tid == 32 * cw) {                      // the row's seed and line key
    uint64_t x[8];
    uint32_t seed[8], key[8];
#pragma unroll
    for (int q = 0; q < 8; q++) x[q] = xr[q];
    pd_seed(unit, x, index << 32 | (uint64_t)r, seed, key);
    store_cv(a.p[3] + 32ull * r, seed);
    store_cv(a.p[4] + 32ull * r, key);
  }
  if (a.i[2] == 1) return;
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(reinterpret_cast<unsigned*>(scratch), 1u) == (unsigned)count - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  pd_levels(scratch, tree, width, root);
  if (tid == 0) {
    uint64_t rt[8];
#pragma unroll
    for (int q = 0; q < 8; q++) rt[q] = root[q];
    pd_seed_a(a.p[5], rt, a.p[10], a.p[11]);
    *reinterpret_cast<unsigned*>(scratch) = 0;
  }
}

// pd_tree: pd_commit's mode 0 tail after its mode 1, one CTA: the levels above level 0, then seed_A and line_a.  p2 the tree,
// p5 the unit block, p9 scratch (pd_commit's templates), p10 seed_A out, p11 line_a out; i0 rows
extern "C" __global__ void __launch_bounds__(128) pd_tree(const Args a) {
  __shared__ uint64_t root[8];
  const int count = a.i[0];
  const int width = count <= 1 ? 1 : 1 << (32 - __clz(count - 1));
  pd_levels(a.p[9], a.p[2], width, root);
  if (threadIdx.x == 0) {
    uint64_t rt[8];
#pragma unroll
    for (int q = 0; q < 8; q++) rt[q] = root[q];
    pd_seed_a(a.p[5], rt, a.p[10], a.p[11]);
  }
}

// pd_tiles: the tile leaves (s512_tile_leaves' bytes), a CTA of 32 threads a tile (row-major; a last row block of R = 16, 32
// or 48 rows partial), lane j < s = R / 2 its segment j from the header table, then pd_fold.  p0 C~ (rows x cols), p1 U,
// p2 out (64 B a tile); i0 rows (a multiple of 16), i1 cols (a multiple of 64), i3 pd_fold's variant
extern "C" __global__ void __launch_bounds__(32) pd_tiles(const Args a) {
  __shared__ uint64_t dig[32 * 8], kw[17 * 80];
  const int cols = a.i[1], tn = cols / 64, t = blockIdx.x, j = threadIdx.x;
  const int ti = t / tn, tj = t % tn, R = min(64, a.i[0] - 64 * ti), s = R / 2;
  if (j < s) {
    uint64_t st[8], w[16];
    rs_seg_head(st, j, s);
#pragma unroll 1
    for (int c = 0; c < RS_SEG / 128; c++) {
      const size_t row = 64 * (size_t)ti + 2 * j + c / 4;
      rs_block(st, w, reinterpret_cast<const uint4*>(a.p[c & 1]) + (row * cols + 64 * tj + 32 * (c >> 1 & 1)) / 4);
    }
    rs_pad(st, 128 + RS_SEG);
#pragma unroll
    for (int q = 0; q < 8; q++) dig[8 * j + q] = st[q];
  }
  __syncthreads();
  uint64_t st[8];
  pd_fold(RS_Q_HEADS + 8 * s, dig, s, st, a.i[3], j, kw);
  if (j == 0) s512_store(a.p[2] + 64ull * t, st, 8);
}

// pd_probe: a compression's cost on one warp, alone on its SM: i0 compressions chained, rs_compress (i1 = 0) or rs_rounds
// over schedules computed beforehand (i1 = 1); p0 out: the clock64 cycles (u64) and the final state (8 u64)
extern "C" __global__ void __launch_bounds__(32) pd_probe(const Args a) {
  __shared__ uint64_t kw[80];
  uint64_t st[8], w[16];
  rs_iv(st);
  for (int t = 0; t < 16; t++) w[t] = 0x0123456789abcdefull * (t + 1 + threadIdx.x);
  rs_sched(w, kw);
  __syncwarp();
  const int n = a.i[0];
  const long long t0 = clock64();
  if (a.i[1] == 0) {
#pragma unroll 1
    for (int i = 0; i < n; i++) {
#pragma unroll
      for (int t = 0; t < 16; t++) w[t] ^= st[t & 7];
      rs_compress(st, w);
    }
  } else {
#pragma unroll 1
    for (int i = 0; i < n; i++) rs_rounds(st, kw);
  }
  const long long t1 = clock64();
  if (threadIdx.x == 0) {
    uint64_t* out = reinterpret_cast<uint64_t*>(a.p[0]);
    out[0] = (uint64_t)(t1 - t0);
    for (int t = 0; t < 8; t++) out[1 + t] = st[t];
  }
}

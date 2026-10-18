// Pearl-C -h3's per-call hashing at decode in fewer, shorter launches, writing the bytes Pearl-C's launches write
// (hash_h3.cuh, hash_fast.cuh; every format is theirs, and every constant byte comes from their tables):
//
// * pd_commit, one launch for A's commitment: a CTA a row.  Its first warps hash the row's 1 KiB segments (a thread a
//   segment, from hash_fast.cuh's header table) while two helper warps do what no data reaches: one the row's hm96 salt
//   (ChaCha20, the byte-table mask M·y over the warp's lanes, and c = SHA-512(salt prefix ‖ y)), the other its frame leaf's
//   first block, its share of the pads, and a lane a level the tree's node templates (below 256 a node's first block is
//   (dom_a, level) alone and the rest differs in the index byte).  Then thread 0 folds the row digest (`pd_fold`) and makes
//   the hm96 leaf and frame leaf (3 compressions), while another warp makes the row's seed and line key from the labels'
//   midstates (PD_LABELS; 3 compressions).  The last CTA to finish (a counter in scratch, reset by that CTA) builds the
//   tree's levels, 2 compressions a level, each level's states kept in shared memory for the next, then seed_A and line_a
//   (3).  Mode 1 stops after level 0, and pd_tree builds the rest: the tree and seed_A on a lane of their own when nothing
//   in the call reads line_a.
// * pd_tiles, one launch for the tile leaves: a warp a tile, a lane a segment, then the fold (`pd_fold`).
//
// Every launch is built once per variant (PD_VARIANT_LIST: pd_commit_V, pd_tree_V, pd_tiles_V, pd_probe_V), each with its
// own code and registers.  A variant's bits: PD_ROUNDS, pd_fold's rounds-only form (the fold's message is all there before
// it starts, so the folding warp's lanes compute every block's round inputs K + W into shared memory at once and the folding
// lane runs the 80 rounds alone); PD_ROLLED, the rl_* compressions (one 16-round body run five times, about a fifth of
// rs_compress's 3,500 instructions); PD_CALL, one copy of each compression a kernel, called from every site rather than
// inlined at each (a call's serial compressions are each at a site of their own, and an unrolled site's 56 KB of code
// fetched cold costs about as much again as the compression); PD_PREFETCH, a segment's 8 lines prefetched into L1 before its
// first block.
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

constexpr int PD_ROUNDS = 1, PD_ROLLED = 2, PD_CALL = 4, PD_PREFETCH = 8;
#define PD_VARIANT_LIST(X) X(1) X(3) X(5) X(7) X(13) X(15)

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
DEV void pd_prefetch(const void* p) {   // the 128-byte line at p into L1
#ifndef PD_HOST
  asm volatile("prefetch.global.L1 [%0];" ::"l"(p));
#else
  (void)p;
#endif
}

// ---------------------------------------------------------------- the compressions
// rs_compress (sha512.cuh) is Pearl-C's.  rl_compress is the same function with one 16-round body run five times, the
// schedule's next 16 words computed in place after each.  rs_rounds / rl_rounds run a block's rounds alone over its 80 round
// inputs K_i + W_i (rs_sched / rl_sched).
DEV void rs_round(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d, uint64_t& e, uint64_t& f, uint64_t& g, uint64_t& h,
                  uint64_t kwi) {
  const uint64_t t1 = h + (rs_rotr(e, 14) ^ rs_rotr(e, 18) ^ rs_rotr(e, 41)) + ((e & f) ^ (~e & g)) + kwi;
  const uint64_t t2 = (rs_rotr(a, 28) ^ rs_rotr(a, 34) ^ rs_rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c));
  h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
}
DEV void rs_next16(uint64_t* w) {   // W_{t+16} for the ring's 16 words W_t, in place, in order
#pragma unroll
  for (int i = 0; i < 16; i++) {
    const uint64_t w15 = w[(i + 1) & 15], w2 = w[(i + 14) & 15];
    w[i] += (rs_rotr(w2, 19) ^ rs_rotr(w2, 61) ^ (w2 >> 6)) + w[(i + 9) & 15] + (rs_rotr(w15, 1) ^ rs_rotr(w15, 8) ^ (w15 >> 7));
  }
}
DEV void rs_sched(const uint64_t* m16, uint64_t* kw) {   // one block's 80 round inputs from its 16 message words
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
DEV void rl_sched(const uint64_t* m16, uint64_t* kw) {
  uint64_t w[16];
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = m16[i];
#pragma unroll 1
  for (int r = 0;; r += 16) {
#pragma unroll
    for (int i = 0; i < 16; i++) kw[r + i] = RS_K[r + i] + w[i];
    if (r == 64) break;
    rs_next16(w);
  }
}
DEV void rs_rounds(uint64_t* st, const uint64_t* kw) {
  uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll
  for (int i = 0; i < 80; ++i) rs_round(a, b, c, d, e, f, g, h, kw[i]);
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}
DEV void rl_rounds(uint64_t* st, const uint64_t* kw) {
  uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
#pragma unroll 1
  for (int r = 0; r < 80; r += 16) {
#pragma unroll
    for (int i = 0; i < 16; i++) rs_round(a, b, c, d, e, f, g, h, kw[r + i]);
  }
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}
DEV void rl_compress(uint64_t* st, const uint64_t* m16) {   // the ring in registers: m16 may be shared memory
  uint64_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7], w[16];
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = m16[i];
#pragma unroll 1
  for (int r = 0;; r += 16) {
#pragma unroll
    for (int i = 0; i < 16; i++) rs_round(a, b, c, d, e, f, g, h, RS_K[r + i] + w[i]);
    if (r == 64) break;
    rs_next16(w);
  }
  st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}
// a variant's forms (R: PD_ROLLED's bit), inlined, or (PD_CALL) one copy a kernel; the called compression takes a copy of
// the message, so the caller's words stay as they were
template <int R> DEV void pd_compress_in(uint64_t* st, uint64_t* w) {
  if constexpr (R != 0) rl_compress(st, w); else rs_compress(st, w);
}
template <int R> DEV void pd_rounds_in(uint64_t* st, const uint64_t* kw) {
  if constexpr (R != 0) rl_rounds(st, kw); else rs_rounds(st, kw);
}
template <int R> DEV void pd_sched_in(const uint64_t* m16, uint64_t* kw) {
  if constexpr (R != 0) rl_sched(m16, kw); else rs_sched(m16, kw);
}
template <int R> __device__ __noinline__ void pd_compress_call(uint64_t* st, const uint64_t* w) {
  uint64_t x[16];
#pragma unroll
  for (int i = 0; i < 16; i++) x[i] = w[i];
  pd_compress_in<R>(st, x);
}
template <int R> __device__ __noinline__ void pd_rounds_call(uint64_t* st, const uint64_t* kw) { pd_rounds_in<R>(st, kw); }
template <int R> __device__ __noinline__ void pd_sched_call(const uint64_t* m16, uint64_t* kw) { pd_sched_in<R>(m16, kw); }
template <int V> DEV void pd_cmp(uint64_t* st, uint64_t* w) {
  if constexpr ((V & PD_CALL) != 0) pd_compress_call<V & PD_ROLLED>(st, w); else pd_compress_in<V & PD_ROLLED>(st, w);
}
template <int V> DEV void pd_rnd(uint64_t* st, const uint64_t* kw) {
  if constexpr ((V & PD_CALL) != 0) pd_rounds_call<V & PD_ROLLED>(st, kw); else pd_rounds_in<V & PD_ROLLED>(st, kw);
}
template <int V> DEV void pd_sch(const uint64_t* m16, uint64_t* kw) {
  if constexpr ((V & PD_CALL) != 0) pd_sched_call<V & PD_ROLLED>(m16, kw); else pd_sched_in<V & PD_ROLLED>(m16, kw);
}
template <int V> DEV void pd_padc(uint64_t* st, uint64_t bytes) {   // rs_pad
  uint64_t w[16];
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = 0;
  w[0] = 0x8000000000000000ull;
  w[15] = 8 * bytes;
  pd_cmp<V>(st, w);
}
DEV void pd_words(uint64_t* w, const uint4* v) {   // a 128-byte block of little-endian u32 words as 16 message words
#pragma unroll
  for (int q = 0; q < 8; q++) {
    const uint4 x = v[q];
    w[2 * q] = rs_be64(x.x, x.y);
    w[2 * q + 1] = rs_be64(x.z, x.w);
  }
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
// rs_fold's digest state over s digests (dig, shared), by a whole warp, lane `lane`: with PD_ROUNDS its lanes compute the
// s / 2 + 1 blocks' round inputs into kw, 80 words each, and lane 0 runs the rounds alone; without it lane 0 compresses in
// the thread.  st out on lane 0
template <int V> DEV void pd_fold(const uint64_t* qst, const uint64_t* dig, int s, uint64_t* st, int lane, uint64_t* kw) {
  const int nb = s / 2 + 1;
  uint64_t w[16];
  if constexpr ((V & PD_ROUNDS) != 0) {
    for (int b = lane; b < nb; b += 32) {
      pd_fold_block(dig, s, b, w);
      pd_sch<V>(w, kw + 80 * b);
    }
    __syncwarp();
  }
  if (lane) return;
#pragma unroll
  for (int t = 0; t < 8; t++) st[t] = qst[t];
#pragma unroll 1
  for (int b = 0; b < nb; b++) {
    if constexpr ((V & PD_ROUNDS) != 0) {
      pd_rnd<V>(st, kw + 80 * b);
    } else {
      pd_fold_block(dig, s, b, w);
      pd_cmp<V>(st, w);
    }
  }
}
// the hm96 salt pieces of leaf i (hm_leaves' work before x), by one whole warp: lanes 0-2 the three ChaCha20 blocks of y
// (into shared ys), each lane 6 of the mask's 192 byte-table rows (4 q + r for y[q]'s byte r, an L2 or DRAM load each: one
// lane alone waits on 192 of them in turn), XORed across the warp, then lane 0 c's state.  Out on lane 0: the mask M·y (8
// little-endian limbs) and c's state
template <int V>
DEV void pd_hm_salt(uint64_t salt_key, uint64_t nonce, uint64_t i, uint64_t hm, uint32_t* ys, int lane, uint64_t* mk,
                    uint64_t* c) {
  const uint64_t* mid = reinterpret_cast<const uint64_t*>(hm);
  const uint64_t* tables = reinterpret_cast<const uint64_t*>(hm + 128);
  if (lane < 3) {
    uint32_t key[8];
    load_key(salt_key, key);
    chacha20_block(key, 3 * i + lane, nonce, ys + 16 * lane);
  }
  __syncwarp();
  uint64_t m[8];
#pragma unroll
  for (int k = 0; k < 8; k++) m[k] = 0;
#pragma unroll
  for (int u = 0; u < 6; u++) {
    const int j = lane + 32 * u;
    const uint4* e = reinterpret_cast<const uint4*>(tables + ((size_t)j * 256 + ((ys[j >> 2] >> (8 * (j & 3))) & 0xFF)) * 8);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const uint4 v = e[k];
      m[2 * k] ^= (uint64_t)v.y << 32 | v.x;
      m[2 * k + 1] ^= (uint64_t)v.w << 32 | v.z;
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1)
#pragma unroll
    for (int k = 0; k < 8; k++) m[k] ^= __shfl_xor_sync(0xffffffffu, m[k], o);
  if (lane) return;
#pragma unroll
  for (int k = 0; k < 8; k++) mk[k] = m[k];
  uint64_t st[8], w[16];
#pragma unroll
  for (int j = 0; j < 8; j++) st[j] = mid[j];
#pragma unroll
  for (int j = 0; j < 16; j++) w[j] = rs_be64(ys[2 * j], ys[2 * j + 1]);
  pd_cmp<V>(st, w);
#pragma unroll
  for (int j = 0; j < 16; j++) w[j] = j < 8 ? rs_be64(ys[32 + 2 * j], ys[33 + 2 * j]) : 0;
  w[8] = 0x8000000000000000ull;
  w[15] = 8 * (128 + 192);
  pd_cmp<V>(st, w);
#pragma unroll
  for (int j = 0; j < 8; j++) c[j] = st[j];
}
// H_seed-A(salt ‖ x ‖ root_B ‖ u64 LE iota) from the label's midstate, and its line key (s512_seeds' bytes); x as 8 words
template <int V> DEV void pd_seed(const uint32_t* unit, const uint64_t* x, uint64_t iota, uint32_t* seed, uint32_t* key) {
  uint64_t st[8], w[16];
#pragma unroll
  for (int t = 0; t < 8; t++) st[t] = PD_LABELS[PD_SEED_A][t];
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = i < 4 ? rs_be64(unit[2 * i], unit[2 * i + 1]) : i < 12 ? x[i - 4]
                                            : rs_be64(unit[8 + 2 * (i - 12)], unit[9 + 2 * (i - 12)]);
  pd_cmp<V>(st, w);
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = i < 4 ? rs_be64(unit[16 + 2 * i], unit[17 + 2 * i]) : 0;
  w[4] = rs_be64((uint32_t)iota, (uint32_t)(iota >> 32));
  w[5] = 0x8000000000000000ull;
  w[15] = 8 * (128 + 168);
  pd_cmp<V>(st, w);
  s512_out32(st, seed);
#pragma unroll
  for (int t = 0; t < 8; t++) st[t] = PD_LABELS[PD_NOISE_LINE][t];
#pragma unroll
  for (int i = 0; i < 16; i++) w[i] = i < 4 ? rs_be64(seed[2 * i], seed[2 * i + 1]) : 0;
  w[4] = 0x8000000000000000ull;
  w[15] = 8 * (128 + 32);
  pd_cmp<V>(st, w);
  s512_out32(st, key);
}

// ---------------------------------------------------------------- the tree's messages
constexpr int PD_MAX_ROWS = 256, PD_MAX_LEVELS = 8;   // width <= 256: every level and index below 256
constexpr int PD_KW_BLOCKS = 33;   // pd_commit's rounds-only fold: s / 2 + 1 <= 33 blocks (k <= 16384)
constexpr int PD_TM = 40;          // a level's node template, in words
constexpr int PD_NODES = 128 * 8 + 64 * 8;   // pd_levels' shared node states: level 1 (at most 128 nodes), then 64, 32, ...
// node (level, i)'s message (48 words: 3 blocks), its children zero
DEV void pd_node_msg(uint64_t dom, int level, int i, uint64_t* buf) {
  for (int t = 0; t < 48; t++) buf[t] = 0;
  WMsg m{buf, 0};
  wm_frame(m, "node", 4, dom);
  wm_uint(m, (uint64_t)level);
  wm_uint(m, (uint64_t)i);
  const int left = wm_slot64(m), right = wm_slot64(m);
  if (wm_finish(m) != 3 || left - 128 != PD_LEFT_AT || right - 128 != PD_RIGHT_AT) __trap();
}
// level l's node template (PD_TM words): the state after its first block (8 words), then blocks 1 and 2 at index 0 (32
// words).  Below 256 a node's first block holds the frame, dom_a, the level and its index's length prefix alone, and its
// index is byte 130: block 1's word 0, bits 40-47.  Both are checked here against the builder at index 1.  buf: 96 words
template <int V> DEV void pd_level_template(uint64_t dom, int level, uint64_t* tm, uint64_t* buf) {
  pd_node_msg(dom, level, 0, buf);
  pd_node_msg(dom, level, 1, buf + 48);
  for (int t = 0; t < 48; t++)
    if ((buf[t] ^ buf[48 + t]) != (t == 16 ? 1ull << 40 : 0ull)) __trap();
  for (int t = 0; t < 32; t++) tm[8 + t] = buf[16 + t];
  uint64_t st[8];
  rs_iv(st);
  pd_cmp<V>(st, buf);
#pragma unroll
  for (int t = 0; t < 8; t++) tm[t] = st[t];
}
template <int V> DEV void pd_pad(uint64_t dom, uint64_t rank, uint64_t tree, uint64_t* buf /*48 words*/) {   // as s512_pads
  for (int t = 0; t < 48; t++) buf[t] = 0;
  WMsg m{buf, 0};
  wm_frame(m, "pad", 3, dom);
  wm_uint(m, rank);
  const int blocks = wm_finish(m);
  uint64_t st[8];
  rs_iv(st);
  for (int b = 0; b < blocks; b++) pd_cmp<V>(st, buf + 16 * b);
  s512_store(tree + 64 * rank, st, 8);
}
// node i of a level from its children's states l and r and the level's template tm; out its state
template <int V> DEV void pd_node(const uint64_t* tm, int i, const uint64_t* l, const uint64_t* r, uint64_t* st) {
  uint64_t w[32];
#pragma unroll
  for (int t = 0; t < 32; t++) w[t] = tm[8 + t];
  w[0] |= (uint64_t)i << 40;
  wm_put64<PD_LEFT_AT>(w, l);
  wm_put64<PD_RIGHT_AT>(w, r);
#pragma unroll
  for (int t = 0; t < 8; t++) st[t] = tm[t];
  pd_cmp<V>(st, w);
  pd_cmp<V>(st, w + 16);
}
// the levels above level 0 of a tree of `width` leaves at `tree` (level 0 written, by any CTA), by the CTA's threads: level l
// from the templates tm (shared, PD_TM words a level), each level's states kept in shared `nodes` (PD_NODES words) for the
// next, and stored; out the root's state on thread 0
template <int V> DEV void pd_levels(const uint64_t* tm, uint64_t tree, int width, uint64_t* nodes, uint64_t* root) {
  uint64_t base = tree, l[8], r[8], st[8];
  const uint64_t* below = nullptr;
  uint64_t* out = nodes;
  for (int w = width, level = 0; w > 1; w >>= 1, level++) {
    const uint64_t above = base + 64ull * w;
    for (int i = threadIdx.x; i < w / 2; i += blockDim.x) {
      if (level == 0) {
        load_be64x8(base + 128ull * i, l);
        load_be64x8(base + 128ull * i + 64, r);
      } else {
#pragma unroll
        for (int t = 0; t < 8; t++) { l[t] = below[16 * i + t]; r[t] = below[16 * i + 8 + t]; }
      }
      pd_node<V>(tm + PD_TM * level, i, l, r, st);
      s512_store(above + 64ull * i, st, 8);
#pragma unroll
      for (int t = 0; t < 8; t++) out[8 * i + t] = st[t];
    }
    __syncthreads();
    below = out;
    out = out == nodes ? nodes + 128 * 8 : nodes;
    base = above;
  }
  if (threadIdx.x == 0) {
    if (width == 1) {
      load_be64x8(tree, root);
    } else {
#pragma unroll
      for (int t = 0; t < 8; t++) root[t] = below[t];
    }
  }
}
template <int V> DEV void pd_seed_a(uint64_t unit, const uint64_t* root, uint64_t seed_a, uint64_t line_a) {
  const uint32_t* u = reinterpret_cast<const uint32_t*>(unit);
  const uint64_t index = (uint64_t)u[24] | (uint64_t)u[25] << 32;
  uint32_t seed[8], key[8];
  pd_seed<V>(u, root, index, seed, key);
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

// pd_commit_V: a CTA a row r < i0, 32 (ceil(s / 32) + 2) threads (s = k / 256).  p0 A's rows (FP32, k words each), p1 row
// digests out (64 B a row), p2 A's tree (level 0, then each level above), p3 row seeds out (32 B a row), p4 their line keys
// out, p5 the unit block (salt ‖ root_B ‖ u64 LE index), p6 dom_a (64 B), p7 the prover's salt key, p8 the hm96 constants,
// p9 scratch (a u32 counter, zero before the first launch and left zero), p10 seed_A out, p11 line_a out; i0 rows (at most
// PD_MAX_ROWS), i1 k, i2 mode (0: the tree and seed_A too, 1: level 0 only)
template <int V> DEV void pd_commit_t(const Args a) {
  __shared__ uint64_t dig[RS_MAX_SEGS * 8], xr[8], mk[8], cc[8], ftm[8], ftw[32], hbuf[48], root[8];
  __shared__ uint64_t tm[PD_MAX_LEVELS * PD_TM];
  __shared__ uint64_t kw[PD_KW_BLOCKS * 80];   // the fold's schedules; before it the templates' builders, after it the levels
  __shared__ uint32_t ys[48];
  __shared__ int last;
  static_assert(PD_KW_BLOCKS * 80 >= PD_NODES && PD_KW_BLOCKS * 80 >= 96 * PD_MAX_LEVELS, "kw holds the builders and the levels");
  const int count = a.i[0], k = a.i[1], s = k / 256, cw = (s + 31) / 32;
  const int r = blockIdx.x, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int width = count <= 1 ? 1 : 1 << (32 - __clz(count - 1)), levels = 31 - __clz(width);
  if (count > PD_MAX_ROWS || ((V & PD_ROUNDS) && s / 2 + 1 > PD_KW_BLOCKS)) __trap();
  const uint64_t dom = a.p[6], scratch = a.p[9], tree = a.p[2];
  const uint32_t* unit = reinterpret_cast<const uint32_t*>(a.p[5]);
  const uint64_t index = (uint64_t)unit[24] | (uint64_t)unit[25] << 32;
  if (warp < cw) {                                  // the row's segments
    if (tid < s) {
      uint64_t st[8], w[16];
      rs_seg_head(st, tid, s);
      const uint4* seg = reinterpret_cast<const uint4*>(a.p[0] + 4ull * k * r + (uint64_t)RS_SEG * tid);
      if constexpr ((V & PD_PREFETCH) != 0) {
#pragma unroll
        for (int c = 0; c < RS_SEG / 128; c++) pd_prefetch(seg + 8 * c);
      }
#pragma unroll 1
      for (int c = 0; c < RS_SEG / 128; c++) {
        pd_words(w, seg + 8 * c);
        pd_cmp<V>(st, w);
      }
      pd_padc<V>(st, 128 + RS_SEG);
#pragma unroll
      for (int q = 0; q < 8; q++) dig[8 * tid + q] = st[q];
    }
  } else if (warp == cw) {                          // the row's hm96 salt
    pd_hm_salt<V>(a.p[7], index, (uint64_t)r, a.p[8], ys, lane, mk, cc);
  } else if (lane > 0) {                            // the levels' node templates, a lane a level
    if (a.i[2] == 0 && lane <= levels) pd_level_template<V>(dom, lane - 1, tm + PD_TM * (lane - 1), kw + 96 * (lane - 1));
  } else {                                          // the frame leaf's first block and this CTA's pads
    for (int t = 0; t < 32; t++) ftw[t] = 0;
    WMsg m{ftw, 0};
    wm_frame(m, "leaf", 4, dom);
    wm_uint(m, (uint64_t)r);
    wm_uint(m, (uint64_t)r);
    wm_be(m, 22, 8);
    for (int j = 0; j < 22; j++) wm_byte(m, HM96_SCHEMA[j]);
    const int at = wm_slot64(m);
    if (wm_finish(m) != 2 || at - 128 != PD_LEAF_AT) __trap();
    uint64_t st[8], w[16];
#pragma unroll
    for (int q = 0; q < 16; q++) w[q] = ftw[q];
    rs_iv(st);
    pd_cmp<V>(st, w);
#pragma unroll
    for (int q = 0; q < 8; q++) ftm[q] = st[q];
    for (int p = count + r; p < width; p += count) pd_pad<V>(dom, (uint64_t)p, tree, hbuf);
  }
  __syncthreads();
  if (warp == 0) {                                  // the row digest
    uint64_t st[8];
    pd_fold<V>(RS_Q_HEADS + 8 * s, dig, s, st, lane, kw);
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
    pd_cmp<V>(lt, w);
    pd_padc<V>(lt, 256);
#pragma unroll
    for (int q = 0; q < 16; q++) w[q] = ftw[16 + q];
    w[16] = 0;
    wm_put64<PD_LEAF_AT>(w, lt);
#pragma unroll
    for (int q = 0; q < 8; q++) lt[q] = ftm[q];
    pd_cmp<V>(lt, w);
    s512_store(tree + 64ull * r, lt, 8);
  } else if (tid == 32 * cw) {                      // the row's seed and line key
    uint64_t x[8];
    uint32_t seed[8], key[8];
#pragma unroll
    for (int q = 0; q < 8; q++) x[q] = xr[q];
    pd_seed<V>(unit, x, index << 32 | (uint64_t)r, seed, key);
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
  pd_levels<V>(tm, tree, width, kw, root);
  if (tid == 0) {
    uint64_t rt[8];
#pragma unroll
    for (int q = 0; q < 8; q++) rt[q] = root[q];
    pd_seed_a<V>(a.p[5], rt, a.p[10], a.p[11]);
    *reinterpret_cast<unsigned*>(scratch) = 0;
  }
}

// pd_tree_V: pd_commit's mode 0 tail after its mode 1, one CTA of 128 threads: the levels' templates (a thread a level), the
// levels above level 0, then seed_A and line_a.  p2 the tree, p5 the unit block, p6 dom_a, p10 seed_A out, p11 line_a out;
// i0 rows
template <int V> DEV void pd_tree_t(const Args a) {
  __shared__ uint64_t tm[PD_MAX_LEVELS * PD_TM], nodes[PD_NODES], root[8];
  const int count = a.i[0];
  const int width = count <= 1 ? 1 : 1 << (32 - __clz(count - 1)), levels = 31 - __clz(width);
  if (count > PD_MAX_ROWS) __trap();
  if ((int)threadIdx.x < levels) pd_level_template<V>(a.p[6], threadIdx.x, tm + PD_TM * threadIdx.x, nodes + 96 * threadIdx.x);
  __syncthreads();
  pd_levels<V>(tm, a.p[2], width, nodes, root);
  if (threadIdx.x == 0) {
    uint64_t rt[8];
#pragma unroll
    for (int q = 0; q < 8; q++) rt[q] = root[q];
    pd_seed_a<V>(a.p[5], rt, a.p[10], a.p[11]);
  }
}

// pd_tiles_V: the tile leaves (s512_tile_leaves' bytes), a CTA of 32 threads a tile (row-major; a last row block of R = 16,
// 32 or 48 rows partial), lane j < s = R / 2 its segment j from the header table, then pd_fold.  p0 C~ (rows x cols), p1 U,
// p2 out (64 B a tile); i0 rows (a multiple of 16), i1 cols (a multiple of 64)
template <int V> DEV void pd_tiles_t(const Args a) {
  __shared__ uint64_t dig[32 * 8], kw[17 * 80];
  const int cols = a.i[1], tn = cols / 64, t = blockIdx.x, j = threadIdx.x;
  const int ti = t / tn, tj = t % tn, R = min(64, a.i[0] - 64 * ti), s = R / 2;
  if (j < s) {
    uint64_t st[8], w[16];
    rs_seg_head(st, j, s);
    const auto blk = [&](int c) {
      const size_t row = 64 * (size_t)ti + 2 * j + c / 4;
      return reinterpret_cast<const uint4*>(a.p[c & 1]) + (row * cols + 64 * tj + 32 * (c >> 1 & 1)) / 4;
    };
    if constexpr ((V & PD_PREFETCH) != 0) {
#pragma unroll
      for (int c = 0; c < RS_SEG / 128; c++) pd_prefetch(blk(c));
    }
#pragma unroll 1
    for (int c = 0; c < RS_SEG / 128; c++) {
      pd_words(w, blk(c));
      pd_cmp<V>(st, w);
    }
    pd_padc<V>(st, 128 + RS_SEG);
#pragma unroll
    for (int q = 0; q < 8; q++) dig[8 * j + q] = st[q];
  }
  __syncthreads();
  uint64_t st[8];
  pd_fold<V>(RS_Q_HEADS + 8 * s, dig, s, st, j, kw);
  if (j == 0) s512_store(a.p[2] + 64ull * t, st, 8);
}

// pd_probe_V: a compression's cost on one warp, alone on its SM: i0 compressions chained, each fed the last one's state; i1
// 0 the compression (pd_cmp), 1 the rounds alone over a schedule computed beforehand (pd_rnd).  p0 out: the clock64 cycles
// (u64), then the final state (8 u64)
template <int V> DEV void pd_probe_t(const Args a) {
  __shared__ uint64_t kw[80];
  uint64_t st[8], w[16];
  rs_iv(st);
  for (int t = 0; t < 16; t++) w[t] = 0x0123456789abcdefull * (t + 1 + threadIdx.x);
  if (threadIdx.x == 0) pd_sch<V>(w, kw);
  __syncwarp();
  const int n = a.i[0], form = a.i[1];
  const long long t0 = clock64();
  if (form) {
#pragma unroll 1
    for (int i = 0; i < n; i++) pd_rnd<V>(st, kw);
  } else {
#pragma unroll 1
    for (int i = 0; i < n; i++) {
      uint64_t x[16];
#pragma unroll
      for (int t = 0; t < 16; t++) x[t] = w[t] ^ st[t & 7];
      pd_cmp<V>(st, x);
    }
  }
  const long long t1 = clock64();
  if (threadIdx.x == 0) {
    uint64_t* out = reinterpret_cast<uint64_t*>(a.p[0]);
    out[0] = (uint64_t)(t1 - t0);
    for (int t = 0; t < 8; t++) out[1 + t] = st[t];
  }
}

#define PD_INSTANCE(V)                                                                                                    \
  extern "C" __global__ void __launch_bounds__(192) pd_commit_##V(const Args a) { pd_commit_t<V>(a); }                  \
  extern "C" __global__ void __launch_bounds__(128) pd_tree_##V(const Args a) { pd_tree_t<V>(a); }                      \
  extern "C" __global__ void __launch_bounds__(32) pd_tiles_##V(const Args a) { pd_tiles_t<V>(a); }                     \
  extern "C" __global__ void __launch_bounds__(32) pd_probe_##V(const Args a) { pd_probe_t<V>(a); }
PD_VARIANT_LIST(PD_INSTANCE)

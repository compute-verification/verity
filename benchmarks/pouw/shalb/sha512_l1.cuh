// One leaf of L1 (hm96-sha512/subtree/v1): a frame-v3-sha512 leaf in the call's own domain,
// SHA-512(frame ‖ u32be 4 ‖ "leaf" ‖ (u64be len ‖ part) for part in [domain id, rank, position, schema, value]), with
// the rank and position one byte each, the 13-byte schema "u64be/1024/v1" and a value of 128 u64 words, each serialized
// big-endian. The value then starts word-aligned at byte 160 and its message words are its words as loaded: no byte
// permutes. Ten blocks in chain form (sha512_min.cuh):
// - block 1: the frame, the tag and the domain id (words 0–14, of which 6–14 are the call's) and the rank (word 15);
// - block 2: the position (word 0), the schema and the value's length (words 1–3) and value words 0–11;
// - blocks 3–9: value words 12–123;
// - block 10: value words 124–127 and the padding of a 9,472-bit message.
// What the call's words alone determine is computed once a call (call_pre: block 1's rounds 0–14, the per-call sums
// rounds 15–18 read, and the per-call part of each schedule word through W_36), as shalb.py's calculator counts it; a
// leaf (leaf_from) reads those PRE words and does the rest.
#pragma once
#include <cstdint>

#include "sha512_min.cuh"

namespace shal1 {

constexpr int VALUE_WORDS = 128;
constexpr uint64_t MESSAGE_BITS = 9472;

// Block 1's constant words 0–5: the frame, the tag's length and "leaf", and the domain id's length but its last byte.
SHAMIN_HD constexpr uint64_t B1(int i) {
  constexpr uint64_t W[6] = {0x76657269746f722full, 0x70726f746f636f6cull, 0x2f6d65726b6c652full,
                             0x6672616d652f7633ull, 0x00000000046c6561ull, 0x6600000000000000ull};
  return W[i];
}

// Block 1's schedule words a leaf computes: the rank (W_15) and every word that reads it. Words 0–5 are constant and
// 6–14, 16, 18 and 20 the call's.
SHAMIN_HD constexpr bool per_leaf(int i) { return i == 15 || (i >= 17 && i != 18 && i != 20); }

// The PRE words a call computes once. Entering block 1's round 15 the state is the call's (a..h); E15 = d + U and
// A15 = U + S0(a) + Maj(a, b, c) with U = h + K_15 + S1(e) + Ch(e, f, g), so round 15 is e = E15 + W_15, a = A15 + W_15;
// rounds 16–18 read a, b, c, e, f and the sums U16 = g + W_16 + K_16, H17 = f + K_17 and U18 = e + W_18 + K_18; P + t − 16
// is W_t's per-call part for t = 16..36 (all of it for 16, 18 and 20).
enum : int { E15, A15, RA, RB, RC, RE, RF, U16, H17, U18, P, PRE = P + 21 };

// Block 1's words 6–14, the call's: bytes 48–119, the domain id's length's last byte, the domain id and the first 7
// bytes of the rank's length.
SHAMIN_HD void call_words(const uint8_t* domain_id, uint64_t* call) {
  uint8_t b[72] = {0x40};
  for (int i = 0; i < 64; ++i) b[1 + i] = domain_id[i];
  for (int i = 0; i < 9; ++i) {
    uint64_t x = 0;
    for (int k = 0; k < 8; ++k) x = x << 8 | b[8 * i + k];
    call[i] = x;
  }
}
// Block 1's word 15 (the rank's length's last byte, the rank) and block 2's word 0 (the position's length's last two
// bytes, the position).
SHAMIN_HD uint64_t rank_word(uint8_t rank) { return 1ull << 56 | (uint64_t)rank << 48; }
SHAMIN_HD uint64_t pos_word(uint8_t pos) { return 1ull << 48 | (uint64_t)pos << 40; }

// The call's PRE words from its words 6–14.
SHAMIN_HD void call_pre(const uint64_t* call, uint64_t* pre) {
  using namespace shamin;
  uint64_t w[37];
#pragma unroll
  for (int i = 0; i < 6; ++i) w[i] = B1(i);
#pragma unroll
  for (int i = 0; i < 9; ++i) w[6 + i] = call[i];
  w[15] = 0;
#pragma unroll
  for (int t = 16; t <= 36; ++t) {
    uint64_t x = 0;
    if (!per_leaf(t - 2)) x += s1(w[t - 2]);
    if (!per_leaf(t - 7)) x += w[t - 7];
    if (!per_leaf(t - 15)) x += s0(w[t - 15]);
    if (!per_leaf(t - 16)) x += w[t - 16];
    w[t] = x;
    pre[P + t - 16] = x;
  }
  uint64_t a = IVt(0), b = IVt(1), c = IVt(2), d = IVt(3), e = IVt(4), f = IVt(5), g = IVt(6), h = IVt(7);
#pragma unroll
  for (int t = 0; t < 15; ++t) {
    const uint64_t t1 = h + w[t] + Kt(t) + S1(e) + ch(e, f, g), t2 = S0(a) + maj(a, b, c);
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  const uint64_t u = h + Kt(15) + S1(e) + ch(e, f, g);
  pre[E15] = d + u;
  pre[A15] = u + S0(a) + maj(a, b, c);
  pre[RA] = a; pre[RB] = b; pre[RC] = c; pre[RE] = e; pre[RF] = f;
  pre[U16] = g + w[16] + Kt(16);
  pre[H17] = f + Kt(17);
  pre[U18] = e + w[18] + Kt(18);
}

// Block 1 from round 15, CHAIN 1 from the IV (st and pend as compress<0, 0, 1> leaves them). PreT is any indexable
// source of the PRE words.
template <class PreT>
SHAMIN_HD void block1(PreT p, uint64_t rank_w, uint64_t* st, uint64_t* pend) {
  using namespace shamin;
  uint64_t w[16];
  w[15] = rank_w;
  const uint64_t e16 = p[E15] + rank_w, a16 = p[A15] + rank_w;
  uint64_t t1 = p[U16] + S1(e16) + ch(e16, p[RE], p[RF]);
  const uint64_t e17 = t1 + p[RC], a17 = t1 + S0(a16) + maj(a16, p[RA], p[RB]);
  w[1] = s1(rank_w) + p[P + 1];
  t1 = p[H17] + w[1] + S1(e17) + ch(e17, e16, p[RE]);
  const uint64_t e18 = t1 + p[RB], a18 = t1 + S0(a17) + maj(a17, a16, p[RA]);
  t1 = p[U18] + S1(e18) + ch(e18, e17, e16);
  uint64_t a = t1 + S0(a18) + maj(a18, a17, a16), b = a18, c = a17, d = a16;
  uint64_t e = t1 + p[RA], f = e18, g = e17, h = e16;
  uint64_t t2 = 0;
#pragma unroll
  for (int t = 19; t < 80; ++t) {
    if (t > 19) {
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    uint64_t wt = t <= 36 ? (uint64_t)p[P + t - 16] : 0;
    if (per_leaf(t)) {
      if (per_leaf(t - 2)) wt += s1(w[(t - 2) & 15]);
      if (per_leaf(t - 7)) wt += w[(t - 7) & 15];
      if (per_leaf(t - 15)) wt += s0(w[(t - 15) & 15]);
      if (per_leaf(t - 16)) wt += w[(t - 16) & 15];
    }
    w[t & 15] = wt;
    t1 = h + wt + Kt(t) + S1(e) + ch(e, f, g);
    t2 = S0(a) + maj(a, b, c);
  }
  st[0] = t1 + t2 + IVt(0); st[1] = IVt(1) + a; st[2] = IVt(2) + b; st[3] = IVt(3);
  st[4] = t1 + d + IVt(4); st[5] = IVt(5) + e; st[6] = IVt(6) + f; st[7] = IVt(7) + Kt(0);
  pend[0] = c;
  pend[1] = g;
}

// Blocks 3–9, B onward: value words 12 + 16·B onward, alternately CHAIN 1 and 2, unrolled by recursion.
template <int B>
SHAMIN_HD void middle(const uint64_t* v, uint64_t* st, uint64_t* pend) {
  if constexpr (B < 7) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = v[12 + 16 * B + i];
    shamin::compress<0, 0, B % 2 == 0 ? 1 : 2>(st, w, pend);
    middle<B + 1>(v, st, pend);
  }
}

// The leaf's digest into st from the call's PRE words, the rank's and position's words and the value's 128 words.
template <class PreT>
SHAMIN_HD void leaf_from(PreT pre, uint64_t rank_w, uint64_t pos_w, const uint64_t* v, uint64_t* st) {
  uint64_t w[16], pend[2];
  block1(pre, rank_w, st, pend);
  w[0] = pos_w;
  w[1] = 0x00000d7536346265ull;
  w[2] = 0x2f313032342f7631ull;
  w[3] = 8 * VALUE_WORDS;
#pragma unroll
  for (int i = 0; i < 12; ++i) w[4 + i] = v[i];
  shamin::compress<0, 0, 2>(st, w, pend);
  middle<0>(v, st, pend);
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = i < 4 ? v[124 + i] : i == 4 ? 1ull << 63 : i == 15 ? MESSAGE_BITS : 0;
  shamin::compress<0, 0, 2>(st, w, pend);
  st[7] -= shamin::Kt(0);
}

// The leaf's digest from the call's words 6–14 (call_pre, then leaf_from).
SHAMIN_HD void leaf(const uint64_t* call, uint64_t rank_w, uint64_t pos_w, const uint64_t* v, uint64_t* st) {
  uint64_t pre[PRE];
  call_pre(call, pre);
  leaf_from((const uint64_t*)pre, rank_w, pos_w, v, st);
}

}  // namespace shal1

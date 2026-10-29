// One segment of a sha512/row-seg/v1 row under prefix binding: SHA-512 of the segment's 128-byte first block,
// S(j, s, m) ‖ salt_w ‖ u64be f ‖ u64be j_call ‖ u64be row zero padded (113 bytes of it), then the segment's 1,024
// bytes (m = 8,192 bits), its message words as loaded. Ten blocks in chain form (sha512_min.cuh):
// - block 1, the prefix: S(j, s, m) (words 0–6 and the first byte of 7), the call's salt_w, f and j (the rest of word 7
//   through the first byte of word 13), the row (the rest of word 13 and the first byte of 14), zeros (word 15);
// - blocks 2–9: the segment's 128 words;
// - block 10: the padding of a 9,216-bit message, a constant block.
// What the segment index and the call alone determine is computed once a call (seg_pre: block 1's rounds 0–12, the
// per-call sums rounds 13–16 read, and the per-call part of each schedule word through W_35), as shalb.py's calculator
// counts it; a segment (segment_from) reads those PRE words and does the rest.
#pragma once
#include <cstdint>

#include "sha512_min.cuh"

namespace shaseg {

constexpr int SEGMENT_WORDS = 128;
constexpr uint64_t MESSAGE_BITS = 9216;

// Block 1's schedule words a segment computes: words 13 and 14, which hold the row, and every word that reads them.
// Words 0–12 and 15 are the call's (15 is zero), and so are 17 and 19.
SHAMIN_HD constexpr bool per_item(int i) { return i == 13 || i == 14 || i == 16 || (i >= 18 && i != 19); }

// Whether W_t (t ≥ 16) has a per-call part that can be nonzero: a source word that is the call's other than word 15.
SHAMIN_HD constexpr bool call_part(int t) {
  return (!per_item(t - 2) && t - 2 != 15) || (!per_item(t - 7) && t - 7 != 15) ||
         (!per_item(t - 15) && t - 15 != 15) || (!per_item(t - 16) && t - 16 != 15);
}

// The PRE words a call computes once. Entering block 1's round 13 the state is the call's (a..h); E13 = d + U and
// A13 = U + S0(a) + Maj(a, b, c) with U = h + K_13 + S1(e) + Ch(e, f, g), so round 13 is e = E13 + W_13,
// a = A13 + W_13; rounds 14–16 read a, b, c, e, f and the sums H14 = g + K_14, H15 = f + K_15 (W_15 is zero) and
// H16 = e + K_16; P + t − 16 is W_t's per-call part for t = 16..35 (all of it for 17 and 19, and zero for 29–31).
enum : int { E13, A13, RA, RB, RC, RE, RF, H14, H15, H16, P, PRE = P + 20 };

// Block 1's 16 words from its 128 bytes.
SHAMIN_HD void prefix_words(const uint8_t* b, uint64_t* w) {
  for (int i = 0; i < 16; ++i) {
    uint64_t x = 0;
    for (int k = 0; k < 8; ++k) x = x << 8 | b[8 * i + k];
    w[i] = x;
  }
}

// The call's PRE words from block 1's words 0–12.
SHAMIN_HD void seg_pre(const uint64_t* call, uint64_t* pre) {
  using namespace shamin;
  uint64_t w[36];
#pragma unroll
  for (int i = 0; i < 13; ++i) w[i] = call[i];
  w[13] = w[14] = w[15] = 0;
#pragma unroll
  for (int t = 16; t <= 35; ++t) {
    uint64_t x = 0;
    if (!per_item(t - 2)) x += s1(w[t - 2]);
    if (!per_item(t - 7)) x += w[t - 7];
    if (!per_item(t - 15)) x += s0(w[t - 15]);
    if (!per_item(t - 16)) x += w[t - 16];
    w[t] = x;
    pre[P + t - 16] = x;
  }
  uint64_t a = IVt(0), b = IVt(1), c = IVt(2), d = IVt(3), e = IVt(4), f = IVt(5), g = IVt(6), h = IVt(7);
#pragma unroll
  for (int t = 0; t < 13; ++t) {
    const uint64_t t1 = h + w[t] + Kt(t) + S1(e) + ch(e, f, g), t2 = S0(a) + maj(a, b, c);
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  const uint64_t u = h + Kt(13) + S1(e) + ch(e, f, g);
  pre[E13] = d + u;
  pre[A13] = u + S0(a) + maj(a, b, c);
  pre[RA] = a; pre[RB] = b; pre[RC] = c; pre[RE] = e; pre[RF] = f;
  pre[H14] = g + Kt(14);
  pre[H15] = f + Kt(15);
  pre[H16] = e + Kt(16);
}

// Block 1 from round 13, CHAIN 1 from the IV (st and pend as compress<0, 0, 1> leaves them). PreT is any indexable
// source of the PRE words.
template <class PreT>
SHAMIN_HD void block1(PreT p, uint64_t w13, uint64_t w14, uint64_t* st, uint64_t* pend) {
  using namespace shamin;
  uint64_t w[16];
  w[13] = w13;
  w[14] = w14;
  const uint64_t e14 = p[E13] + w13, a14 = p[A13] + w13;
  uint64_t t1 = p[H14] + w14 + S1(e14) + ch(e14, p[RE], p[RF]);
  const uint64_t e15 = t1 + p[RC], a15 = t1 + S0(a14) + maj(a14, p[RA], p[RB]);
  t1 = p[H15] + S1(e15) + ch(e15, e14, p[RE]);
  const uint64_t e16 = t1 + p[RB], a16 = t1 + S0(a15) + maj(a15, a14, p[RA]);
  w[0] = s1(w14) + p[P];
  t1 = p[H16] + w[0] + S1(e16) + ch(e16, e15, e14);
  uint64_t a = t1 + S0(a16) + maj(a16, a15, a14), b = a16, c = a15, d = a14;
  uint64_t e = t1 + p[RA], f = e16, g = e15, h = e14;
  uint64_t t2 = 0;
#pragma unroll
  for (int t = 17; t < 80; ++t) {
    if (t > 17) {
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    uint64_t wt = t <= 35 && call_part(t) ? (uint64_t)p[P + t - 16] : 0;
    if (per_item(t)) {
      if (per_item(t - 2)) wt += s1(w[(t - 2) & 15]);
      if (per_item(t - 7)) wt += w[(t - 7) & 15];
      if (per_item(t - 15)) wt += s0(w[(t - 15) & 15]);
      if (per_item(t - 16)) wt += w[(t - 16) & 15];
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

// Blocks 2–9, B + 2 onward: segment words 16·B onward, CHAIN 2 on even blocks and 1 on odd, unrolled by recursion.
template <int B>
SHAMIN_HD void data(const uint64_t* v, uint64_t* st, uint64_t* pend) {
  if constexpr (B < 8) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = v[16 * B + i];
    shamin::compress<0, 0, B % 2 == 0 ? 2 : 1>(st, w, pend);
    data<B + 1>(v, st, pend);
  }
}

// The segment's digest into st from the call's PRE words, block 1's words 13 and 14 and the segment's 128 words.
template <class PreT>
SHAMIN_HD void segment_from(PreT pre, uint64_t w13, uint64_t w14, const uint64_t* v, uint64_t* st) {
  uint64_t pend[2];
  block1(pre, w13, w14, st, pend);
  data<0>(v, st, pend);
  shamin::compress<0, MESSAGE_BITS, 2>(st, nullptr, pend);
  st[7] -= shamin::Kt(0);
}

// The segment's digest from block 1's 16 words (seg_pre, then segment_from).
SHAMIN_HD void segment(const uint64_t* w1, const uint64_t* v, uint64_t* st) {
  uint64_t pre[PRE];
  seg_pre(w1, pre);
  segment_from((const uint64_t*)pre, w1[13], w1[14], v, st);
}

}  // namespace shaseg

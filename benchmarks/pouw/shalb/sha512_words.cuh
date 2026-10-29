// The hiding module's words in the minimal kernel's forms: a 16 KiB sha512/row/v2 chunk (sha512_rowv2.cuh, from
// P(131072)'s midstate), its frame-v3-sha512 leaf of the forward's words domain D_{f,W} with schema
// "sha512/row/v2/l1-w", and a level-0 node of that domain's tree. Positions and node indices take three bytes (uint()
// of 2^16 ≤ x < 2^24), so every message's layout is fixed and the leaf's rank (= position) is block 1's word 15.
//
//   leaf = SHA-512(frame ‖ u32be 4 ‖ "leaf" ‖ (u64be len ‖ p) for p in [domain id, uint(pos), uint(pos), schema, digest])
//   node = SHA-512(frame ‖ u32be 4 ‖ "node" ‖ (u64be len ‖ p) for p in [domain id, uint(0), uint(index), left, right])
//
// What the domain alone determines is computed once a call (Pre): the leaf's block 1 through round 14 and its per-call
// schedule (sha512_l1.cuh's call_pre: block 1's words 0–14 are the frame, the tag and the domain id there too), and
// the node's block 1 with round 0 of its block 2 but W_0. A leaf is then block 1 from round 15 and block 2; a node is
// blocks 2 and 3, in chain form. The digest starts at byte 169 of the leaf and the children at bytes 141 and 213 of the
// node, so each of their message words is two funnel shifts of two digest words.
#pragma once
#include <cstdint>

#include "sha512_l1.cuh"
#include "sha512_rowv2.cuh"

namespace shaw {

constexpr int ROW_BYTES = 16384, ROW_WORDS = ROW_BYTES / 8, NB = ROW_BYTES / 128;
constexpr uint64_t ROW_BITS = 8ull * ROW_BYTES, PAD_BITS = 8ull * (128 + ROW_BYTES);
using Mid = shamin::RowV2Mid<ROW_BITS>;

struct Pre {
  uint64_t leaf[shal1::PRE];
  uint64_t node_h[8];   // the node's chaining value after block 1, h[7] as H7 + K_0 (chain form)
  uint64_t node_pe, node_pa;   // e_1 − W_0 and a_1 − W_0 of block 2
};

// The leaf of position pos (2^16 ≤ pos < 2^24) over the chunk digest d.
template <class PreT>
SHAMIN_HD void leaf(PreT pre, uint64_t pos, const uint64_t* d, uint64_t* st) {
  uint64_t w[16], pend[2];
  shal1::block1(pre, 3ull << 56 | pos << 32, st, pend);
  w[0] = 3ull << 32 | pos << 8;
  w[1] = 0x1273;
  w[2] = 0x68613531322f726full;
  w[3] = 0x772f76322f6c312dull;
  w[4] = 0x7700000000000000ull;
  w[5] = 0x4000000000000000ull | d[0] >> 8;
#pragma unroll
  for (int i = 1; i < 8; ++i) w[5 + i] = d[i - 1] << 56 | d[i] >> 8;
  w[13] = d[7] << 56 | 0x80ull << 48;
  w[14] = 0;
  w[15] = 8 * 233;
  shamin::compress<0, 0, 2>(st, w, pend);
  st[7] -= shamin::Kt(0);
}

// Block 2 of a node from the call's Pre, CHAIN 1 (st and pend as compress<0, 0, 1> leaves them).
SHAMIN_HD void node_block2(const Pre& p, uint64_t* w, uint64_t* st, uint64_t* pend) {
  using namespace shamin;
  const uint64_t* hv = p.node_h;
  uint64_t a = w[0] + p.node_pa, b = hv[0], c = hv[1], d = hv[2], e = w[0] + p.node_pe, f = hv[4], g = hv[5],
           h = hv[6], t1 = 0, t2 = 0;
#pragma unroll
  for (int t = 1; t < 80; ++t) {
    if (t > 1) {
      h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    uint64_t wt;
    if (t < 16) {
      wt = w[t];
    } else {
      wt = s1(w[(t - 2) & 15]) + w[(t - 7) & 15] + s0(w[(t - 15) & 15]) + w[t & 15];
      w[t & 15] = wt;
    }
    t1 = h + wt + Kt(t) + S1(e) + ch(e, f, g);
    t2 = S0(a) + maj(a, b, c);
  }
  st[0] = t1 + t2 + hv[0]; st[1] = hv[1] + a; st[2] = hv[2] + b; st[3] = hv[3];
  st[4] = t1 + d + hv[4]; st[5] = hv[5] + e; st[6] = hv[6] + f; st[7] = hv[7];
  pend[0] = c;
  pend[1] = g;
}

// The level-0 node of index idx (2^16 ≤ idx < 2^24) over leaves l and r.
SHAMIN_HD void node(const Pre& p, uint64_t idx, const uint64_t* l, const uint64_t* r, uint64_t* st) {
  uint64_t w[16], pend[2];
  w[0] = 3ull << 48 | idx << 24;
  w[1] = 0x40ull << 24 | l[0] >> 40;
#pragma unroll
  for (int i = 2; i < 9; ++i) w[i] = l[i - 2] << 24 | l[i - 1] >> 40;
  w[9] = l[7] << 24;
  w[10] = 0x40ull << 24 | r[0] >> 40;
#pragma unroll
  for (int i = 11; i < 16; ++i) w[i] = r[i - 11] << 24 | r[i - 10] >> 40;
  node_block2(p, w, st, pend);
  w[0] = r[5] << 24 | r[6] >> 40;
  w[1] = r[6] << 24 | r[7] >> 40;
  w[2] = r[7] << 24 | 0x80ull << 16;
#pragma unroll
  for (int i = 3; i < 15; ++i) w[i] = 0;
  w[15] = 8 * 277;
  shamin::compress<0, 0, 2>(st, w, pend);
  st[7] -= shamin::Kt(0);
}

// The call's Pre from the domain id (64 bytes), on the host.
inline void pre_for(const uint8_t* domain_id, Pre& p) {
  using namespace shamin;
  uint64_t call[9];
  shal1::call_words(domain_id, call);
  shal1::call_pre(call, p.leaf);
  uint8_t m[128] = {};
  const char* frame = "veritor/protocol/merkle/frame/v3";
  int n = 0;
  while (frame[n]) { m[n] = (uint8_t)frame[n]; ++n; }
  m[n++] = 0;
  m[n + 3] = 4; n += 4;
  m[n++] = 'n'; m[n++] = 'o'; m[n++] = 'd'; m[n++] = 'e';
  m[n + 7] = 64; n += 8;
  for (int i = 0; i < 64; ++i) m[n++] = domain_id[i];
  m[n + 7] = 1; n += 8;
  m[n++] = 0;
  uint64_t w[16], st[8];
  for (int i = 0; i < 16; ++i) {
    w[i] = 0;
    for (int k = 0; k < 8; ++k) w[i] = w[i] << 8 | m[8 * i + k];
  }
  for (int i = 0; i < 8; ++i) st[i] = IVt(i);
  compress<0>(st, w);
  const uint64_t u = st[7] + Kt(0) + S1(st[4]) + ch(st[4], st[5], st[6]);
  for (int i = 0; i < 8; ++i) p.node_h[i] = st[i];
  p.node_h[7] += Kt(0);
  p.node_pe = st[3] + u;
  p.node_pa = u + S0(st[0]) + maj(st[0], st[1], st[2]);
}

}  // namespace shaw

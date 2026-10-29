// One leaf of L1 (hm96-sha512/subtree/v1): a frame-v3-sha512 leaf in the call's own domain,
// SHA-512(frame ‖ u32be 4 ‖ "leaf" ‖ (u64be len ‖ part) for part in [domain id, rank, position, schema, value]), with
// the rank and position one byte each, the 13-byte schema "u64be/1024/v1" and a value of 128 u64 words, each serialized
// big-endian. The value then starts word-aligned at byte 160 and its message words are its words as loaded: no byte
// permutes. Ten blocks in chain form (sha512_min.cuh):
// - block 1: the frame, the tag and the domain id (words 0–14, of which 6–14 are the call's) and the rank (word 15);
// - block 2: the position (word 0), the schema and the value's length (words 1–3) and value words 0–11;
// - blocks 3–9: value words 12–123;
// - block 10: value words 124–127 and the padding of a 9,472-bit message.
#pragma once
#include <cstdint>

#include "sha512_min.cuh"

namespace shal1 {

constexpr int VALUE_WORDS = 128;
constexpr uint64_t MESSAGE_BITS = 9472;

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

// The leaf's digest into st from the call's words, the rank's and position's words and the value's 128 words.
SHAMIN_HD void leaf(const uint64_t* call, uint64_t rank_w, uint64_t pos_w, const uint64_t* v, uint64_t* st) {
  constexpr uint64_t B1[6] = {0x76657269746f722full, 0x70726f746f636f6cull, 0x2f6d65726b6c652full,
                              0x6672616d652f7633ull, 0x00000000046c6561ull, 0x6600000000000000ull};
  uint64_t w[16], pend[2];
#pragma unroll
  for (int i = 0; i < 8; ++i) st[i] = shamin::IVt(i);
  st[7] += shamin::Kt(0);
#pragma unroll
  for (int i = 0; i < 6; ++i) w[i] = B1[i];
#pragma unroll
  for (int i = 0; i < 9; ++i) w[6 + i] = call[i];
  w[15] = rank_w;
  shamin::compress<0, 0, 1>(st, w, pend);
  w[0] = pos_w;
  w[1] = 0x00000d7536346265ull;
  w[2] = 0x2f313032342f7631ull;
  w[3] = 8 * VALUE_WORDS;
#pragma unroll
  for (int i = 0; i < 12; ++i) w[4 + i] = v[i];
  shamin::compress<0, 0, 2>(st, w, pend);
#pragma unroll
  for (int b = 0; b < 7; ++b) {
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = v[12 + 16 * b + i];
    if (b % 2 == 0) {
      shamin::compress<0, 0, 1>(st, w, pend);
    } else {
      shamin::compress<0, 0, 2>(st, w, pend);
    }
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) w[i] = i < 4 ? v[124 + i] : i == 4 ? 1ull << 63 : i == 15 ? MESSAGE_BITS : 0;
  shamin::compress<0, 0, 2>(st, w, pend);
  st[7] -= shamin::Kt(0);
}

}  // namespace shal1

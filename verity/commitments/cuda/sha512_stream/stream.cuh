// The listed keyed stream's blocks on the device and the host (`verity.commitments.randomness`'s `Key.block`): block c at
// an index is SHA-512(head ‖ be8(c)), where head = the stream's tag ‖ the key ‖ the index's frames, built by the caller.
// A block is one compression from the IV, so head is at most HEAD_MAX bytes; integer operations only.
#pragma once
#include "../sha512.cuh"

namespace sxs {

using u64 = uint64_t;
using u32 = uint32_t;

// head ‖ 8-byte counter ‖ 0x80 ‖ 16-byte length fit one 128-byte chunk
constexpr u32 HEAD_MAX = 128 - 8 - 1 - 16;
constexpr u32 LEAF_BLOCKS = 3;
// the largest first leaf + leaves whose last counter is below 2^64
constexpr u64 LEAVES_MAX = ~0ull / LEAF_BLOCKS;
// the launch functions' refusals, below 0 so they never read as a CUDA error
constexpr int REFUSED_HEAD = -1;
constexpr int REFUSED_RANGE = -2;
constexpr int REFUSED_CAPTURE = -3;

// the padded chunk of every block at one head, with the counter's eight bytes zero
struct Head {
  u64 w[16];
  u32 len;
  u32 pad_;
};

// the chunk for `head` (`len` bytes), else REFUSED_HEAD: nothing is truncated
inline int head_of(const uint8_t* head, u32 len, Head& h) {
  if (len > HEAD_MAX) return REFUSED_HEAD;
  uint8_t b[128] = {};
  for (u32 i = 0; i < len; ++i) b[i] = head[i];
  b[len + 8] = 0x80;
  const u64 bits = 8ull * (len + 8);
  for (int k = 0; k < 8; ++k) b[120 + k] = (uint8_t)(bits >> (56 - 8 * k));
  for (int i = 0; i < 16; ++i) {
    u64 v = 0;
    for (int k = 0; k < 8; ++k) v = v << 8 | b[8 * i + k];
    h.w[i] = v;
  }
  h.len = len;
  h.pad_ = 0;
  return 0;
}

inline int range_ok(u64 first, u64 leaves) { return leaves <= LEAVES_MAX && first <= LEAVES_MAX - leaves; }

// block c's chaining value: its 64 bytes are st[0..7], each big-endian
template <class D>
SX_HD void block(const Head& h, u64 c, u64* st) {
  // the counter's word picked by compare, not by index: a runtime index would put w in local memory
  const int q = (int)(h.len >> 3);
  const u32 r = 8 * (h.len & 7);
  const u64 hi = c >> r, lo = r ? c << (64 - r) : 0;
  u64 w[16];
  SHA_UNROLL
  for (int i = 0; i < 16; ++i) w[i] = h.w[i] | (i == q ? hi : 0) | (i == q + 1 ? lo : 0);
  sx::iv(st);
  D::compress(st, w);
}

}  // namespace sxs

// A P2 unit's sketch-sha512 receipt (verity/core/protocols/remote_bandwidth/PROTOCOL.md §11): y_g = SHA512(Z(prefix) ||
// u64be g || sigma_g), sigma_g the unit's 16 int32 little-endian (64 bytes). The prefix is one 128-byte block, so y is one
// compression from its midstate over a block holding the 72 message bytes, 0x80 and the 1,600-bit length. On the device
// and the host, with sha512_rows' compression (every word a big-endian u64).
#pragma once
#include "../sha512_rows/sha512x.cuh"

namespace p2r {

using sx::u64;

// sigma: the unit's 64 bytes as 8 little-endian u64 loads; y: the digest's 8 words
template <int V, int ROLL>
SX_HD void receipt(const u64* mid, u64 g, const u64* sigma, u64* y) {
  u64 w[16];
  w[0] = g;
  for (int i = 0; i < 8; ++i) w[1 + i] = sx::bswap64(sigma[i]);
  w[9] = 0x8000000000000000ull;
  for (int i = 10; i < 15; ++i) w[i] = 0;
  w[15] = 8 * (128 + 72);
  for (int i = 0; i < 8; ++i) y[i] = mid[i];
  sx::compress<V, ROLL>(y, w);
}

// the chaining value after one 128-byte block from the IV
inline void midstate(const uint8_t* block, u64* out) {
  u64 w[16];
  for (int i = 0; i < 16; ++i) {
    u64 v = 0;
    for (int k = 0; k < 8; ++k) v = v << 8 | block[8 * i + k];
    w[i] = v;
  }
  sx::iv(out);
  sx::compress<0, 1>(out, w);
}

}  // namespace p2r

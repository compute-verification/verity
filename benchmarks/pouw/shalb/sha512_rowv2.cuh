// A sha512/row/v2 row's hashing after its prefix block, which is a constant and hoisted (computed once), in the
// minimal kernel's chain form (sha512_min.cuh): NB data blocks from a constant chaining value, then the constant padding
// block of a PAD_BITS-bit message. The first block reads the IV in place of P(n_bits)'s midstate, and a row/v2 padding
// block's length is 8·(128 + row bytes); neither changes the registers written, and with the IV the digest is
// SHA-512 of the row alone, so a host check compares it with hashlib. The blocks pair from the first, so the padding
// block is the chain's trailing unpaired block and takes K_0 back out of H7 in its feed-forward.
#pragma once
#include <cstdint>

#include "sha512_min.cuh"

namespace shav2 {

template <int B, int NB>
SHAMIN_HD void blocks(const uint64_t* v, uint64_t* st, uint64_t* pend) {
  if constexpr (B < NB) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = v[16 * B + i];
    shamin::compress<B == 0 ? 1 : 0, 0, B % 2 == 0 ? 1 : 2>(st, w, pend);
    blocks<B + 1, NB>(v, st, pend);
  }
}

template <int NB, uint64_t PAD_BITS>
SHAMIN_HD void chunk(const uint64_t* v, uint64_t* st) {
  static_assert(NB % 2 == 0, "the padding block is the chain's trailing unpaired block");
  uint64_t pend[2];
  blocks<0, NB>(v, st, pend);
  shamin::compress<0, PAD_BITS, 1>(st, nullptr, pend);
  st[3] += pend[0];
  st[7] += pend[1] - shamin::Kt(0);
}

}  // namespace shav2

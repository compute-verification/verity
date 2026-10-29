// A sha512/row/v2 row's hashing after its prefix block, which is a constant and hoisted (computed once), in the
// minimal kernel's chain form (sha512_min.cuh): NB data blocks from a constant chaining value, then the constant padding
// block of a PAD_BITS-bit message. CVS is the constant chaining value: shamin::RowV2Mid<row bits> for the row's own
// digest; the IV by default, which writes the same registers and makes the digest SHA-512 of the row alone (the host
// check compares that with hashlib). A row/v2 padding block's length is 8·(128 + row bytes). The blocks pair from the
// first, so the padding block is the chain's trailing unpaired block and takes K_0 back out of H7 in its feed-forward.
// chunk unrolls every block; chunk_loop unrolls the first pair and loops over the rest, for long rows.
#pragma once
#include <cstdint>

#include "sha512_min.cuh"

namespace shav2 {

template <int B, int NB, class CVS>
SHAMIN_HD void blocks(const uint64_t* v, uint64_t* st, uint64_t* pend) {
  if constexpr (B < NB) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = v[16 * B + i];
    shamin::compress<B == 0 ? 1 : 0, 0, B % 2 == 0 ? 1 : 2, CVS>(st, w, pend);
    blocks<B + 1, NB, CVS>(v, st, pend);
  }
}

template <uint64_t PAD_BITS>
SHAMIN_HD void finish(uint64_t* st, uint64_t* pend) {
  shamin::compress<0, PAD_BITS, 1>(st, nullptr, pend);
  st[3] += pend[0];
  st[7] += pend[1] - shamin::Kt(0);
}

template <int NB, uint64_t PAD_BITS, class CVS = shamin::IVConst>
SHAMIN_HD void chunk(const uint64_t* v, uint64_t* st) {
  static_assert(NB % 2 == 0, "the padding block is the chain's trailing unpaired block");
  uint64_t pend[2];
  blocks<0, NB, CVS>(v, st, pend);
  finish<PAD_BITS>(st, pend);
}

template <int NB, uint64_t PAD_BITS, class CVS = shamin::IVConst>
SHAMIN_HD void chunk_loop(const uint64_t* v, uint64_t* st) {
  static_assert(NB % 2 == 0, "the padding block is the chain's trailing unpaired block");
  uint64_t pend[2];
  blocks<0, 2, CVS>(v, st, pend);
#pragma unroll 1
  for (int b = 2; b < NB; b += 2) {
    uint64_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = v[16 * b + i];
    shamin::compress<0, 0, 1>(st, w, pend);
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = v[16 * b + 16 + i];
    shamin::compress<0, 0, 2>(st, w, pend);
  }
  finish<PAD_BITS>(st, pend);
}

}  // namespace shav2

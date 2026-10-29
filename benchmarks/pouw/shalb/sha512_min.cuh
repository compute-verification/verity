// SHA-512's compression (FIPS 180-4) with the fewest 32-bit registers written that its standard dataflow allows on
// sm_90a: 1,728 SHF, 896 LOP3 and 910 adds (IADD3 / IADD3.X, some issued as IMAD.X), 3,534 in all, for a chaining value
// and a block that are both variables.
//
// - K_t is an immediate of the round's first add (IADD3 and IADD3.X take one 32-bit immediate each): no constant loads.
// - Each round is four 64-bit adds of at most three terms: U = h + W_t + K_t, T1 = U + S1(e) + Ch(e,f,g), e' = d + T1,
//   a' = T1 + S0(a) + Maj(a,b,c). Each schedule word is one four-term add, two IADD3 pairs.
// - e_80 is read only by the feed-forward, so H4 = T1 + d + cv4 is one three-term add.
// - CVC = 1: the chaining value is a compile-time constant, CVS's (the IV, or the state after a block-aligned constant
//   prefix), so round 0 folds to two adds of W_0 to constants.
// - PAD_BITS: the block is a block-aligned message's padding block, a constant, so its schedule is K_t + W_t immediates
//   and a round is 28: 80 × 28 + 14 = 2,254. nvcc doesn't fold the funnel shifts of a schedule over constant words, so
//   a constant block is hashed this way, in a chain too.
// - CHAIN: a chain of compressions two blocks at a time (CHAIN 1, then 2), the chaining value in chain form: cv[7] holds
//   H7 + K_0, so round 0's first add is h + W_0. The second block reads the first's H3 only in e_1 = d + T1 and its
//   H7 + K_0 only in that first add, both with a free IADD3 slot, so the first block leaves each as two terms (cv[3] or
//   cv[7] and pend[]), and the second adds both terms to its own d_80 or h_80 in one three-term add of its
//   feed-forward: 2 × 3,534 − 4 registers a pair of blocks. A chain from a constant chaining value starts with CVC and
//   CHAIN 1: its round 0 adds K_0 as usual, and it leaves H3 and H7 + K_0 as constants beside the pending c and g.
#pragma once
#include <cstdint>

#define SHAMIN_HD __host__ __device__ __forceinline__

namespace shamin {

SHAMIN_HD constexpr uint64_t Kt(int t) {
  constexpr uint64_t K[80] = {
      0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull, 0x3956c25bf348b538ull,
      0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull, 0xd807aa98a3030242ull, 0x12835b0145706fbeull,
      0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull, 0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull,
      0xc19bf174cf692694ull, 0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
      0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull, 0x983e5152ee66dfabull,
      0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull, 0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull,
      0x06ca6351e003826full, 0x142929670a0e6e70ull, 0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull,
      0x53380d139d95b3dfull, 0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
      0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull, 0xd192e819d6ef5218ull,
      0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull, 0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull,
      0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull, 0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull,
      0x682e6ff3d6b2b8a3ull, 0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
      0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull, 0xca273eceea26619cull,
      0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull, 0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull,
      0x113f9804bef90daeull, 0x1b710b35131c471bull, 0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull,
      0x431d67c49c100d4cull, 0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull};
  return K[t];
}

SHAMIN_HD constexpr uint64_t IVt(int i) {
  constexpr uint64_t IV[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
                              0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                              0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  return IV[i];
}

SHAMIN_HD constexpr uint64_t r64(uint64_t x, int n) { return x >> n | x << (64 - n); }

// K_t + W_t for the padding block of a message whose BITS bits end on a block boundary (W_0 = 2^63, W_15 = BITS, the
// rest 0): a compile-time constant, so each round's K_t + W_t is its first add's immediate.
struct PadBlock {
  uint64_t kw[80];
  __host__ __device__ constexpr explicit PadBlock(uint64_t bits) : kw{} {
    uint64_t w[80] = {};
    w[0] = 1ull << 63;
    w[15] = bits;
    for (int t = 16; t < 80; ++t) {
      const uint64_t a = w[t - 2], b = w[t - 15];
      w[t] = (r64(a, 19) ^ r64(a, 61) ^ a >> 6) + w[t - 7] + (r64(b, 1) ^ r64(b, 8) ^ b >> 7) + w[t - 16];
    }
    for (int t = 0; t < 80; ++t) kw[t] = Kt(t) + w[t];
  }
};

// A compile-time chaining value for CVC: the IV, or (RowV2Mid) the state after a constant prefix block.
struct IVConst {
  uint64_t h[8];
  __host__ __device__ constexpr IVConst() : h{IVt(0), IVt(1), IVt(2), IVt(3), IVt(4), IVt(5), IVt(6), IVt(7)} {}
};

// The state after sha512/row/v2's prefix block P(BITS) = ("verity/sha512-row/v2\0" ‖ u64be BITS) zero-padded to
// 128 bytes, compressed from the IV at compile time.
template <uint64_t BITS>
struct RowV2Mid {
  uint64_t h[8];
  __host__ __device__ constexpr RowV2Mid() : h{} {
    uint64_t w[80] = {0x7665726974792f73ull, 0x68613531322d726full, 0x772f763200000000ull | BITS >> 40, BITS << 24};
    for (int t = 16; t < 80; ++t) {
      const uint64_t x = w[t - 2], y = w[t - 15];
      w[t] = (r64(x, 19) ^ r64(x, 61) ^ x >> 6) + w[t - 7] + (r64(y, 1) ^ r64(y, 8) ^ y >> 7) + w[t - 16];
    }
    uint64_t s[8] = {IVt(0), IVt(1), IVt(2), IVt(3), IVt(4), IVt(5), IVt(6), IVt(7)};
    for (int t = 0; t < 80; ++t) {
      const uint64_t e = s[4], a = s[0];
      const uint64_t t1 = s[7] + (r64(e, 14) ^ r64(e, 18) ^ r64(e, 41)) + ((e & s[5]) ^ (~e & s[6])) + Kt(t) + w[t];
      const uint64_t t2 = (r64(a, 28) ^ r64(a, 34) ^ r64(a, 39)) + ((a & s[1]) | (s[2] & (a | s[1])));
      s[7] = s[6]; s[6] = s[5]; s[5] = s[4]; s[4] = s[3] + t1; s[3] = s[2]; s[2] = s[1]; s[1] = s[0]; s[0] = t1 + t2;
    }
    for (int i = 0; i < 8; ++i) h[i] = IVt(i) + s[i];
  }
};

SHAMIN_HD uint64_t pack(uint32_t lo, uint32_t hi) { return (uint64_t)hi << 32 | lo; }

// The 32-bit window of hi:lo starting at bit m: one SHF.R.W.U32 on the device.
SHAMIN_HD uint32_t window(uint32_t lo, uint32_t hi, int m) {
#ifdef __CUDA_ARCH__
  return __funnelshift_r(lo, hi, m);
#else
  return (uint32_t)((((uint64_t)hi << 32) | lo) >> m);
#endif
}

template <int N>
SHAMIN_HD uint64_t rotr(uint64_t x) {
  uint32_t lo = (uint32_t)x, hi = (uint32_t)(x >> 32);
  if constexpr (N >= 32) {
    const uint32_t t = lo;
    lo = hi;
    hi = t;
  }
  constexpr int m = N & 31;
  if constexpr (m == 0) {
    return pack(lo, hi);
  } else {
    return pack(window(lo, hi, m), window(hi, lo, m));
  }
}
template <int N>
SHAMIN_HD uint64_t shr(uint64_t x) {
  const uint32_t lo = (uint32_t)x, hi = (uint32_t)(x >> 32);
  return pack(window(lo, hi, N), hi >> N);
}
SHAMIN_HD uint64_t S0(uint64_t a) { return rotr<28>(a) ^ rotr<34>(a) ^ rotr<39>(a); }
SHAMIN_HD uint64_t S1(uint64_t e) { return rotr<14>(e) ^ rotr<18>(e) ^ rotr<41>(e); }
SHAMIN_HD uint64_t s0(uint64_t w) { return rotr<1>(w) ^ rotr<8>(w) ^ shr<7>(w); }
SHAMIN_HD uint64_t s1(uint64_t w) { return rotr<19>(w) ^ rotr<61>(w) ^ shr<6>(w); }
SHAMIN_HD uint64_t ch(uint64_t e, uint64_t f, uint64_t g) { return (e & f) ^ (~e & g); }
SHAMIN_HD uint64_t maj(uint64_t a, uint64_t b, uint64_t c) { return (a & b) | (c & (a | b)); }

// One compression of the 16-word block w (big-endian u64 words; the schedule overwrites it) into cv. PAD_BITS != 0: the
// block is the padding block of a PAD_BITS-bit block-aligned message, w is not read, and there is no schedule. CHAIN 1
// and 2 compress a pair of blocks of a chain in chain form, pend[] carrying H3's and H7 + K_0's second terms between
// them.
template <int CVC, uint64_t PAD_BITS = 0, int CHAIN = 0, class CVS = IVConst>
SHAMIN_HD void compress(uint64_t* cv, uint64_t* w, uint64_t* pend = nullptr) {
  static_assert(CHAIN != 2 || !CVC, "only a chain's first block has a constant chaining value");
  constexpr PadBlock pad(PAD_BITS);
  constexpr CVS k{};
  uint64_t a, b, c, d, e, f, g, h;
  if constexpr (CVC) {
    a = k.h[0]; b = k.h[1]; c = k.h[2]; d = k.h[3]; e = k.h[4]; f = k.h[5]; g = k.h[6]; h = k.h[7];
  } else {
    a = cv[0]; b = cv[1]; c = cv[2]; d = cv[3]; e = cv[4]; f = cv[5]; g = cv[6]; h = cv[7];
  }
  uint64_t t1 = 0, t2 = 0;
#pragma unroll
  for (int t = 0; t < 80; ++t) {
    if (t > 0) {
      h = g; g = f; f = e;
      if constexpr (CHAIN == 2) {
        e = t == 1 ? d + pend[0] + t1 : d + t1;
      } else {
        e = d + t1;
      }
      d = c; c = b; b = a; a = t1 + t2;
    }
    uint64_t u;
    if constexpr (PAD_BITS != 0) {
      if (CHAIN && !CVC && t == 0) {
        u = CHAIN == 2 ? h + pend[1] + (pad.kw[0] - Kt(0)) : h + (pad.kw[0] - Kt(0));
      } else {
        u = h + pad.kw[t];
      }
    } else {
      uint64_t wt;
      if (t < 16) {
        wt = w[t];
      } else {
        wt = s1(w[(t - 2) & 15]) + w[(t - 7) & 15] + s0(w[(t - 15) & 15]) + w[t & 15];
        w[t & 15] = wt;
      }
      if (CHAIN && !CVC && t == 0) {
        u = CHAIN == 2 ? h + pend[1] + wt : h + wt;
      } else {
        u = h + wt + Kt(t);
      }
    }
    t1 = u + S1(e) + ch(e, f, g);
    t2 = S0(a) + maj(a, b, c);
  }
  const uint64_t cv0 = CVC ? k.h[0] : cv[0], cv1 = CVC ? k.h[1] : cv[1], cv2 = CVC ? k.h[2] : cv[2];
  const uint64_t cv3 = CVC ? k.h[3] : cv[3], cv4 = CVC ? k.h[4] : cv[4], cv5 = CVC ? k.h[5] : cv[5];
  const uint64_t cv6 = CVC ? k.h[6] : cv[6], cv7 = CVC ? k.h[7] : cv[7];
  cv[0] = t1 + t2 + cv0; cv[1] = cv1 + a; cv[2] = cv2 + b;
  cv[4] = t1 + d + cv4; cv[5] = cv5 + e; cv[6] = cv6 + f;
  if constexpr (CHAIN == 1) {
    pend[0] = c;
    pend[1] = g;
    if constexpr (CVC) {
      cv[3] = cv3;
      cv[7] = cv7 + Kt(0);
    }
  } else if constexpr (CHAIN == 2) {
    uint64_t p3 = pend[0], p7 = pend[1];
#ifdef __CUDA_ARCH__
    // Round 0 added cv3 + p3 and cv7 + p7 to other terms; this keeps the compiler from materializing those sums once
    // for both uses, which would write the two registers each that the chain form saves.
    asm volatile("" : "+l"(p3), "+l"(p7));
#endif
    cv[3] = cv3 + p3 + c;
    cv[7] = cv7 + p7 + g;
  } else {
    cv[3] = cv3 + c;
    cv[7] = cv7 + g;
  }
}

}  // namespace shamin

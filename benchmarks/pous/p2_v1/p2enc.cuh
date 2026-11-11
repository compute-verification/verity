// P2's encode at 16,448 bits on p2sq.cuh's squaring, without CGBN: c = rho(sigma_K(rho(m + t mod p))), one block per warp.
// For a payload and a tweak below p it gives p2v1.cu's encode_v1_kernel's bytes, and it raises that kernel's flags: bit 0
// for a tweak not below p, bit 1 for a root that isn't one.
//
// rho(y) is the parity-canonical root of +-y: with s = y^((p + 1) / 4), the even one of {s, p - s} when s^2 = y and the odd
// one when s^2 = p - y. The exponent is 2^a (2^n - 1) + r, as the host passes it to p2v1.cu's pow_e, and the chain is
// pow_e's: 2^n - 1 by an addition chain on the bits of n, then a squarings, then y^r left to right. Every squaring is sqr_k,
// and a multiplication is two of them: x b = ((x + b)^2 - (x - b)^2) / 4 mod p.
//
// The kernel holds sqr_k once. The host lays one root out as steps (chain_for), each a glue operation and a count of
// squarings after it, and a warp walks the steps twice a block. sqr_k leaves no register for a second value, so everything
// but the running value waits in shared memory past the area sqr_k uses: the base y, the saved power and a multiplication's
// other operand, each a row in the staging layout (stage), where a lane reads and writes its own limbs only.
#pragma once
#include "p2sq.cuh"

namespace p2sq {

constexpr int ENC_SQ_WORDS = KSNAP + 32 * KSNAP_STRIDE;   // what sqr_k and a row's staging use of a warp's shared memory
constexpr int ENC_WORDS = ENC_SQ_WORDS + 3 * STAGE_WORDS;  // then the rows: the base, the saved power, the scratch
static_assert(E_WORDS + STAGE_WORDS <= ENC_SQ_WORDS, "load_row's staging, at the snapshot area's start, is inside it");
static_assert(ENC_WORDS % 4 == 0, "every warp's area is 16-byte aligned");

// One root's steps: step = glue | squarings << 4, the glue first
constexpr uint32_t CHAIN_MAX = 255;
enum : uint32_t {
  G_NONE,       // nothing
  G_SAVE,       // saved = x
  G_BASE,       // x = y
  G_MUL_BASE,   // a multiplication by y begins: scratch = x - y, x = x + y; one squaring follows
  G_MUL_SAVED,  // the same by the saved power
  G_SWAP,       // x and the scratch change places; one squaring follows
  G_MUL_END,    // x = (scratch - x) / 4: the product
  G_ROOT,       // x = s^2 and saved = s: x = the root of +-y, or the step fails
};
struct Chain {
  uint32_t len;
  uint32_t step[CHAIN_MAX];
};

// The steps of rho for the exponent 2^a (2^n - 1) + r, n >= 1; false when they don't fit a Chain
inline bool chain_for(uint32_t a, uint32_t n, uint32_t r, Chain &ch) {
  ch.len = 0;
  if (n == 0) return false;
  bool fits = true;
  auto put = [&](uint32_t glue, uint32_t squarings) {
    if (ch.len == CHAIN_MAX || squarings >> 28) { fits = false; return; }
    ch.step[ch.len++] = glue | squarings << 4;
  };
  auto mul = [&](uint32_t by) { put(by, 1); put(G_SWAP, 1); put(G_MUL_END, 0); };
  uint32_t k = 1;                                              // x = y^(2^k - 1)
  for (int bit = 30 - __builtin_clz(n); bit >= 0; bit--) {
    put(G_SAVE, k);                                            // x^(2^k) ...
    mul(G_MUL_SAVED);                                          // ... times x: y^(2^(2k) - 1)
    k *= 2;
    if ((n >> bit) & 1) {
      put(G_NONE, 1);
      mul(G_MUL_BASE);                                         // x^2 y: y^(2^(k + 1) - 1)
      k += 1;
    }
  }
  put(G_NONE, a);
  if (r) {
    put(G_SAVE, 0);                                            // saved = y^(2^a (2^n - 1))
    put(G_BASE, 0);                                            // x = y^r, left to right
    for (int bit = 30 - __builtin_clz(r); bit >= 0; bit--) {
      put(G_NONE, 1);
      if ((r >> bit) & 1) mul(G_MUL_BASE);
    }
    mul(G_MUL_SAVED);
  }
  put(G_SAVE, 1);                                              // saved = s, x = s^2
  put(G_ROOT, 0);
  return fits;
}

// a lane's own limbs to and from a row
__device__ __forceinline__ void park(uint32_t *__restrict__ row, const Num &x, uint32_t lane) {
  #pragma unroll
  for (int q = 0; q < 16; q++) row[stage(16 * (int)lane + q)] = x.l[q];
  if (lane == 31) { row[stage(512)] = x.e0; row[stage(513)] = x.e1; }
}
__device__ __forceinline__ void unpark(const uint32_t *__restrict__ row, Num &x, uint32_t lane) {
  #pragma unroll
  for (int q = 0; q < 16; q++) x.l[q] = row[stage(16 * (int)lane + q)];
  x.e0 = lane == 31 ? row[stage(512)] : 0;
  x.e1 = lane == 31 ? row[stage(513)] : 0;
}
__device__ __forceinline__ bool same(const Num &a, const Num &b) {
  uint32_t v = (a.e0 ^ b.e0) | (a.e1 ^ b.e1);
  #pragma unroll
  for (int q = 0; q < 16; q++) v |= a.l[q] ^ b.l[q];
  return __all_sync(0xFFFFFFFFu, v == 0);
}
// x += v mod 2^W; returns the carry out of limb 513
__device__ __forceinline__ uint32_t add_small(Num &x, uint32_t v, uint32_t lane) {
  uint64_t c = lane == 0 ? v : 0;
  #pragma unroll
  for (int q = 0; q < 16; q++) { c += x.l[q]; x.l[q] = (uint32_t)c; c >>= 32; }
  return carry_across(x, (uint32_t)c, lane);                    // only lane 0 added anything
}
// x -= v mod 2^W; returns the borrow out of limb 513
__device__ __forceinline__ uint32_t sub_small(Num &x, uint32_t v, uint32_t lane) {
  uint32_t b = 0;
  if (lane == 0) {
    uint64_t d = (uint64_t)x.l[0] - v;
    x.l[0] = (uint32_t)d; b = (uint32_t)(d >> 63);
    #pragma unroll
    for (int q = 1; q < 16; q++) { d = (uint64_t)x.l[q] - b; x.l[q] = (uint32_t)d; b = (uint32_t)(d >> 63); }
  }
  return borrow_across(x, b, lane);
}
// x = x + y mod p as encode_v1_kernel reduces a sum: past 2^W by 2^W = c0, else by one p; below p when x and y are
__device__ __forceinline__ void add_p(Num &x, const Num &y, uint32_t c0, uint32_t lane) {
  uint64_t c = 0;
  #pragma unroll
  for (int q = 0; q < 16; q++) { c += (uint64_t)x.l[q] + y.l[q]; x.l[q] = (uint32_t)c; c >>= 32; }
  if (lane == 31) {
    c += (uint64_t)x.e0 + y.e0; x.e0 = (uint32_t)c; c >>= 32;
    c += (uint64_t)x.e1 + y.e1; x.e1 = (uint32_t)c; c >>= 32;
  }
  uint32_t g = (uint32_t)c;
  uint32_t top = __shfl_sync(0xFFFFFFFFu, g, 31);
  if (lane == 31) g = 0;
  top += carry_across(x, g, lane);
  if (top || ge_p(x, c0, lane)) add_small(x, c0, lane);
}
// x = x - y mod p, x and y below p
__device__ __forceinline__ void sub_p(Num &x, const Num &y, uint32_t c0, uint32_t lane) {
  uint32_t b = 0;
  {
    uint64_t v;
    #pragma unroll
    for (int q = 0; q < 16; q++) { v = (uint64_t)x.l[q] - y.l[q] - b; x.l[q] = (uint32_t)v; b = (uint32_t)(v >> 63); }
    if (lane == 31) {
      v = (uint64_t)x.e0 - y.e0 - b; x.e0 = (uint32_t)v; b = (uint32_t)(v >> 63);
      v = (uint64_t)x.e1 - y.e1 - b; x.e1 = (uint32_t)v; b = (uint32_t)(v >> 63);
    }
  }
  uint32_t topb = __shfl_sync(0xFFFFFFFFu, b, 31);
  if (lane == 31) b = 0;
  topb += borrow_across(x, b, lane);
  if (topb) sub_small(x, c0, lane);                            // x - y + 2^W, then - c0: x - y + p
}
// x = x / 2 mod p, x below p: an odd x is halved as x + p = 2^W + (x - c0)
__device__ __forceinline__ void half_p(Num &x, uint32_t c0, uint32_t lane) {
  uint32_t top = 0;
  if (odd_of(x)) top = 1 - sub_small(x, c0, lane);
  uint32_t up = __shfl_down_sync(0xFFFFFFFFu, x.l[0], 1);      // the next lane's lowest limb; lane 31 goes on to limb 512
  if (lane == 31) up = x.e0;
  #pragma unroll
  for (int q = 0; q < 15; q++) x.l[q] = x.l[q] >> 1 | x.l[q + 1] << 31;
  x.l[15] = x.l[15] >> 1 | up << 31;
  if (lane == 31) {
    x.e0 = x.e0 >> 1 | x.e1 << 31;
    x.e1 = x.e1 >> 1 | top << 31;
  }
}

// One step's glue on the running value x; false when G_ROOT finds no root
__device__ __forceinline__ bool glue(uint32_t g, uint32_t *__restrict__ base, uint32_t *__restrict__ saved,
                                     uint32_t *__restrict__ scratch, Num &x, uint32_t c0, uint32_t lane) {
  if (g == G_SAVE) {
    park(saved, x, lane);
  } else if (g == G_BASE) {
    unpark(base, x, lane);
  } else if (g == G_MUL_BASE || g == G_MUL_SAVED) {
    Num b, d = x;
    unpark(g == G_MUL_BASE ? base : saved, b, lane);
    sub_p(d, b, c0, lane);
    add_p(x, b, c0, lane);
    park(scratch, d, lane);
  } else if (g == G_SWAP) {
    Num d;
    unpark(scratch, d, lane);
    park(scratch, x, lane);
    x = d;
  } else if (g == G_MUL_END) {
    Num d;
    unpark(scratch, d, lane);
    sub_p(d, x, c0, lane);
    half_p(d, c0, lane);
    half_p(d, c0, lane);
    x = d;
  } else if (g == G_ROOT) {
    Num y;
    unpark(base, y, lane);
    const bool even = same(x, y);                              // s^2 = y: the even one of {s, p - s}
    finish(x, true, c0, lane);
    const bool odd = same(x, y);                               // s^2 = p - y: the odd one
    unpark(saved, x, lane);
    if (odd_of(x) == even) finish(x, true, c0, lane);
    return even || odd;
  }
  return true;
}

// c = rho(sigma_K(rho(m + t mod p))) of one block, K and t as rows; sm is the warp's ENC_WORDS
__device__ __forceinline__ void encode_one(uint32_t *__restrict__ sm, const Chain &chain, const uint32_t *__restrict__ mrow,
                                           uint32_t *__restrict__ crow, const uint32_t *__restrict__ krow,
                                           const uint32_t *__restrict__ trow, uint32_t c0, uint32_t *__restrict__ flags,
                                           uint32_t lane) {
  uint32_t *st = sm + E_WORDS, *base = sm + ENC_SQ_WORDS, *saved = base + STAGE_WORDS, *scratch = saved + STAGE_WORDS;
  Num x, k;
  load_row(st, mrow, x, lane);
  load_row(st, trow, k, lane);
  if (ge_p(k, c0, lane) && lane == 0) atomicOr(flags, 1u);
  add_p(x, k, c0, lane);                                       // y = m + t mod p
  bool ok = true;
  #pragma unroll 1
  for (int root = 0; root < 2; root++) {
    if (root) {                                                // sigma_K
      load_row(st, krow, k, lane);
      #pragma unroll
      for (int q = 0; q < 16; q++) k.l[q] ^= x.l[q];
      k.e0 ^= x.e0; k.e1 ^= x.e1;
      if (!ge_p(k, c0, lane)) x = k;
    }
    park(base, x, lane);
    #pragma unroll 1
    for (uint32_t pc = 0; pc < chain.len; pc++) {
      const uint32_t step = chain.step[pc];
      ok = glue(step & 15, base, saved, scratch, x, c0, lane) && ok;
      #pragma unroll 1
      for (uint32_t i = step >> 4; i; i--) sqr_k(sm, x, false, c0, lane);
    }
  }
  if (!ok && lane == 0) atomicOr(flags, 2u);
  store_row(st, crow, x, lane);
}

}  // namespace p2sq

// c = rho(sigma_K(rho(m + t mod p))), one block per warp, K and t as rows, chain being chain_for's
__global__ void __launch_bounds__(32 * P2SQ_WARPS, P2SQ_MINB) encode_sq_kernel(
    const __grid_constant__ p2sq::Chain chain, const uint32_t *__restrict__ min, uint32_t *__restrict__ cout,
    const uint32_t *__restrict__ kin, const uint32_t *__restrict__ tin, uint32_t n, uint32_t c0, uint32_t *__restrict__ flags) {
  using namespace p2sq;
  __shared__ __align__(16) uint32_t smem[P2SQ_WARPS][ENC_WORDS];
  const uint32_t lane = threadIdx.x % 32, wid = threadIdx.x / 32;
  const uint32_t stride = gridDim.x * P2SQ_WARPS;
  for (uint32_t j = blockIdx.x * P2SQ_WARPS + wid; j < n; j += stride)
    encode_one(smem[wid], chain, min + (size_t)j * N, cout + (size_t)j * N, kin + (size_t)j * N, tin + (size_t)j * N, c0,
               flags, lane);
}

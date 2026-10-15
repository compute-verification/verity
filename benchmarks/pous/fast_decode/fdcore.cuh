// What the fast-decode prototypes share (`tcsq.cu`, `nttsq.cu`): P2's values at width w = 32 (32 PER + 2) on one warp,
// the fold of a square's limb sums mod p = 2^w - c0, and the rest of the decode (sigma, the parity negation, m = a - t).
//
// A square arrives as int64 limb sums V_x = sum_t c_{4x+t} 2^(8t) (or c_{2x} + c_{2x+1} 2^16 from 16-bit limbs), x < 2 L,
// in shared memory at vpos(x) (one padding word every PER, so the lanes' loads fall in distinct banks). `fold` reduces
// them: W_x = lo32(V_x) + (V_{x-1} >> 32), U_L = W_L + c0 W_{L+L} (+ c0^2 W_{2L} at L = 0), each limb plus its share of
// the bias 2^30 p (b_L = 2^30 (2^32 - 1), b_0 = 2^30 (2^32 - c0), so the limbs are positive and below 2^63), then the
// carries (each lane its PER limbs, the lanes' carries passed up once and then resolved by a ballot lookahead, lane 31's
// carry out of 2^w folded back times c0), then the value is reduced below p and negated when the operand was odd.
// `algo.fold` is its twin.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#ifndef FD_PER
#define FD_PER 16
#endif

namespace fd {

constexpr int PER = FD_PER;
constexpr int L = 32 * PER + 2, LM = 32 * PER, NB = 4 * L;
constexpr int VLIMBS = 2 * L;
__host__ __device__ constexpr int vpos(int x) { return x + x / PER; }
constexpr int VWORDS = vpos(VLIMBS) + 1;
constexpr unsigned FULL = 0xFFFFFFFFu;
static_assert(PER % 2 == 0, "rows load in 8-byte pairs");

__device__ __forceinline__ uint32_t bswap(uint32_t x) { return __byte_perm(x, 0, 0x0123); }

// A value on a warp: lane t has limbs PER t .. PER t + PER - 1; lane 31 also limbs LM (e0) and LM + 1 (e1).
struct Num {
  uint32_t l[PER];
  uint32_t e0, e1;
};

__device__ __forceinline__ bool all_ones(const Num &x, int lane) {
  uint32_t v = FULL;
#pragma unroll
  for (int j = 0; j < PER; j++) v &= x.l[j];
  if (lane == 31) v &= x.e0 & x.e1;
  return v == FULL;
}
__device__ __forceinline__ bool all_zero(const Num &x, int lane) {
  uint32_t v = 0;
#pragma unroll
  for (int j = 0; j < PER; j++) v |= x.l[j];
  if (lane == 31) v |= x.e0 | x.e1;
  return v == 0;
}
__device__ __forceinline__ void add_bit(Num &x, uint32_t cin, int lane) {
  uint64_t c = cin;
#pragma unroll
  for (int j = 0; j < PER; j++) { c += x.l[j]; x.l[j] = (uint32_t)c; c >>= 32; }
  if (lane == 31) { c += x.e0; x.e0 = (uint32_t)c; c >>= 32; c += x.e1; x.e1 = (uint32_t)c; }
}
__device__ __forceinline__ void sub_bit(Num &x, uint32_t bin, int lane) {
  uint64_t b = bin;
#pragma unroll
  for (int j = 0; j < PER; j++) { const uint64_t v = (uint64_t)x.l[j] - b; x.l[j] = (uint32_t)v; b = v >> 63; }
  if (lane == 31) {
    uint64_t v = (uint64_t)x.e0 - b;
    x.e0 = (uint32_t)v;
    b = v >> 63;
    v = (uint64_t)x.e1 - b;
    x.e1 = (uint32_t)v;
  }
}
// each lane carries g (0/1) out of its top limb (lane 31's out of e1): resolved between lanes; returns the carry out of e1
__device__ __forceinline__ uint32_t carry_across(Num &x, uint32_t g, int lane) {
  const uint32_t G = __ballot_sync(FULL, g), A = G | __ballot_sync(FULL, all_ones(x, lane));
  const uint64_t S = (uint64_t)A + G;
  add_bit(x, (((uint32_t)S ^ A ^ G) >> lane) & 1, lane);
  return (uint32_t)(S >> 32);
}
__device__ __forceinline__ uint32_t borrow_across(Num &x, uint32_t b, int lane) {
  const uint32_t B = __ballot_sync(FULL, b), A = B | __ballot_sync(FULL, all_zero(x, lane));
  const uint64_t S = (uint64_t)A + B;
  sub_bit(x, (((uint32_t)S ^ A ^ B) >> lane) & 1, lane);
  return (uint32_t)(S >> 32);
}
// the lane's PER limbs += v (v < 2^47); returns the carry out of them
__device__ __forceinline__ uint32_t add_low(Num &x, uint64_t v) {
  uint64_t c = v;
#pragma unroll
  for (int j = 0; j < PER; j++) { c += x.l[j]; x.l[j] = (uint32_t)c; c >>= 32; }
  return (uint32_t)c;
}
// x >= p, i.e. x + c0 carries out of 2^W
__device__ __forceinline__ bool ge_p(const Num &x, uint32_t c0, int lane) {
  uint32_t gen = 0;
  if (lane == 0) {
    uint32_t rest = FULL;
#pragma unroll
    for (int j = 1; j < PER; j++) rest &= x.l[j];
    gen = x.l[0] >= 0u - c0 && rest == FULL;
  }
  const uint32_t G = __ballot_sync(FULL, gen), A = G | __ballot_sync(FULL, all_ones(x, lane));
  return (((uint64_t)A + G) >> 32) & 1;
}
// x (< 2^W) reduced below p, then negated when odd (p - x unless zero): the end of d
__device__ __forceinline__ void finish(Num &x, bool odd, uint32_t c0, int lane) {
  if (ge_p(x, c0, lane)) {   // x - p = x + c0 - 2^W
    const uint32_t g = lane == 0 ? add_low(x, c0) : 0;
    carry_across(x, g, lane);
  }
  if (odd && !__all_sync(FULL, all_zero(x, lane))) {   // p - x = (2^W - 1 - x) - (c0 - 1)
#pragma unroll
    for (int j = 0; j < PER; j++) x.l[j] = ~x.l[j];
    if (lane == 31) { x.e0 = ~x.e0; x.e1 = ~x.e1; }
    uint32_t b = 0;
    if (lane == 0) {
      uint64_t v = (uint64_t)x.l[0] - (c0 - 1);
      x.l[0] = (uint32_t)v;
      b = (uint32_t)(v >> 63);
#pragma unroll
      for (int j = 1; j < PER; j++) { v = (uint64_t)x.l[j] - b; x.l[j] = (uint32_t)v; b = (uint32_t)(v >> 63); }
    }
    borrow_across(x, b, lane);
  }
}

// big-endian row (L words) <-> Num
__device__ __forceinline__ void load_row(Num &x, const uint32_t *__restrict__ row, int lane) {
  const uint2 *r2 = reinterpret_cast<const uint2 *>(row + (L - PER * (lane + 1)));
#pragma unroll
  for (int k = 0; k < PER / 2; k++) {
    const uint2 w = r2[k];   // words L - PER (lane + 1) + 2k, + 1: limbs PER lane + PER - 1 - 2k, PER lane + PER - 2 - 2k
    x.l[PER - 1 - 2 * k] = bswap(w.x);
    x.l[PER - 2 - 2 * k] = bswap(w.y);
  }
  x.e0 = x.e1 = 0;
  if (lane == 31) {
    const uint2 w = reinterpret_cast<const uint2 *>(row)[0];
    x.e1 = bswap(w.x);
    x.e0 = bswap(w.y);
  }
}
__device__ __forceinline__ void store_row(uint32_t *__restrict__ row, const Num &x, int lane) {
  uint2 *r2 = reinterpret_cast<uint2 *>(row + (L - PER * (lane + 1)));
#pragma unroll
  for (int k = 0; k < PER / 2; k++) r2[k] = make_uint2(bswap(x.l[PER - 1 - 2 * k]), bswap(x.l[PER - 2 - 2 * k]));
  if (lane == 31) reinterpret_cast<uint2 *>(row)[0] = make_uint2(bswap(x.e1), bswap(x.e0));
}

__device__ __forceinline__ uint64_t lo32(int64_t v) { return (uint64_t)v & 0xFFFFFFFFull; }

// out = (sum V_x 2^(32x)) mod p, canonical, negated when odd. TOP adds the terms of z's top 8 bytes T to the limb sums:
// z = M + X T (X = 2^(32 LM)) and V covers M^2 only, so V'_x = V_x + 2 (lo(M_{x-LM} T0) + hi(M_{x-LM-1} T0) +
// lo(M_{x-LM-1} T1) + hi(M_{x-LM-2} T1)) + T^2's limb x - 2 LM (algo.top_terms), from the limbs of z this warp holds.
template <bool TOP>
__device__ __forceinline__ void fold(const int64_t *V, Num &out, const Num &z, bool odd, uint32_t c0, int lane) {
  uint32_t mp = 0, mn0 = 0, mn1 = 0, t0 = 0, t1 = 0, m00 = 0, m01 = 0, tsq[4] = {0, 0, 0, 0};
  if constexpr (TOP) {
    mp = __shfl_up_sync(FULL, z.l[PER - 1], 1);
    mn0 = __shfl_down_sync(FULL, z.l[0], 1);
    mn1 = __shfl_down_sync(FULL, z.l[1], 1);
    if (lane == 0) mp = 0;
    if (lane == 31) mn0 = mn1 = 0;
    t0 = __shfl_sync(FULL, z.e0, 31);
    t1 = __shfl_sync(FULL, z.e1, 31);
    m00 = __shfl_sync(FULL, z.l[0], 0);
    m01 = __shfl_sync(FULL, z.l[1], 0);
    const uint64_t tt = ((uint64_t)t1 << 32) | t0;
    const uint64_t lo = tt * tt, hi = __umul64hi(tt, tt);
    tsq[0] = (uint32_t)lo, tsq[1] = (uint32_t)(lo >> 32), tsq[2] = (uint32_t)hi, tsq[3] = (uint32_t)(hi >> 32);
  }
  auto M = [&](int r) -> uint32_t { return r < 0 ? mp : r < PER ? z.l[r] : r == PER ? mn0 : mn1; };
  auto mlo = [](uint32_t a, uint32_t b) -> int64_t { return (int64_t)(uint32_t)(a * b); };
  auto mhi = [](uint32_t a, uint32_t b) -> int64_t { return (int64_t)__umulhi(a, b); };
  // the top's terms of limb sum x = PER lane + r + L, r in [-1, PER + 2)
  auto E2 = [&](int r) -> int64_t {
    if constexpr (!TOP) {
      return 0;
    } else {
      const int64_t e = 2 * (mlo(M(r + 2), t0) + mhi(M(r + 1), t0) + mlo(M(r + 1), t1) + mhi(M(r), t1));
      const int k = PER * lane + r + L - 2 * LM;
      return e + (k >= 0 && k < 4 ? (int64_t)tsq[k] : 0);
    }
  };
  auto Vr = [&](int x) -> int64_t { return V[vpos(x)]; };

  int64_t v1p = lane == 0 ? 0 : Vr(PER * lane - 1);
  int64_t v2p = Vr(PER * lane + L - 1) + E2(-1);
  int64_t wtop = 0;
  if (lane == 0) wtop = (int64_t)c0 * c0 * (Vr(2 * L - 1) >> 32);   // with TOP, V'_{2L-1} = T^2's limb 3 < 2^32
  const uint64_t bias = (uint64_t)1 << 30, bb = bias * 0xFFFFFFFFull;
  uint64_t r = 0;
#pragma unroll
  for (int j = 0; j < PER; j++) {
    const int x = PER * lane + j;
    const int64_t v1 = Vr(x), v2 = Vr(x + L) + E2(j);
    int64_t u = (int64_t)lo32(v1) + (v1p >> 32) + (int64_t)c0 * ((int64_t)lo32(v2) + (v2p >> 32));
    uint64_t b = bb;
    if (x == 0) { u += wtop; b = bias * (0x100000000ull - c0); }
    const uint64_t s = (uint64_t)(u + (int64_t)b) + r;
    out.l[j] = (uint32_t)s;
    r = s >> 32;
    v1p = v1;
    v2p = v2;
  }
  out.e0 = out.e1 = 0;
  if (lane == 31) {
    // limbs LM and LM + 1, whose own sums carry the top's terms too: 2 lo(M_0 T0); 2 (lo(M_1 T0) + hi(M_0 T0) + lo(M_0 T1))
    const int64_t ea = TOP ? 2 * mlo(m00, t0) : 0;
    const int64_t eb = TOP ? 2 * (mlo(m01, t0) + mhi(m00, t0) + mlo(m00, t1)) : 0;
    const int64_t va = Vr(LM) + ea, vb = Vr(LM + 1) + eb;
    const int64_t a2 = Vr(LM + L) + E2(PER), b2 = Vr(LM + 1 + L) + E2(PER + 1);
    int64_t u = (int64_t)lo32(va) + (v1p >> 32) + (int64_t)c0 * ((int64_t)lo32(a2) + (v2p >> 32));
    uint64_t s = (uint64_t)(u + (int64_t)bb) + r;
    out.e0 = (uint32_t)s;
    r = s >> 32;
    u = (int64_t)lo32(vb) + (va >> 32) + (int64_t)c0 * ((int64_t)lo32(b2) + (a2 >> 32));
    s = (uint64_t)(u + (int64_t)bb) + r;
    out.e1 = (uint32_t)s;
    r = s >> 32;   // weight 2^W = c0
  }
  // lane t's carry (< 2^32) goes to lane t + 1; lane 31's, times c0, to lane 0
  const uint32_t rl = (uint32_t)r, rh = (uint32_t)(r >> 32);
  uint64_t in = ((uint64_t)__shfl_up_sync(FULL, rh, 1) << 32) | __shfl_up_sync(FULL, rl, 1);
  const uint64_t top = ((uint64_t)__shfl_sync(FULL, rh, 31) << 32) | __shfl_sync(FULL, rl, 31);
  if (lane == 0) in = top * c0;
  uint32_t g = add_low(out, in);
  if (lane == 31) {
    uint64_t c = (uint64_t)out.e0 + g;
    out.e0 = (uint32_t)c;
    c = (c >> 32) + out.e1;
    out.e1 = (uint32_t)c;
    g = (uint32_t)(c >> 32);
  }
  uint32_t topc = __shfl_sync(FULL, g, 31);
  if (lane == 31) g = 0;
  topc += carry_across(out, g, lane);
  while (topc) {
    const uint32_t gg = lane == 0 ? add_low(out, (uint64_t)topc * c0) : 0;
    topc = carry_across(out, gg, lane);
  }
  finish(out, odd, c0, lane);
}

// sigma_K(r) = r xor K if that is below p, else r: K's row read from global
__device__ __forceinline__ void sigma(Num &z, const Num &r, const uint32_t *__restrict__ krow, uint32_t c0, int lane) {
  Num k;
  load_row(k, krow, lane);
#pragma unroll
  for (int q = 0; q < PER; q++) k.l[q] ^= r.l[q];
  k.e0 ^= r.e0;
  k.e1 ^= r.e1;
  const bool ge = ge_p(k, c0, lane);
#pragma unroll
  for (int q = 0; q < PER; q++) z.l[q] = ge ? r.l[q] : k.l[q];
  z.e0 = ge ? r.e0 : k.e0;
  z.e1 = ge ? r.e1 : k.e1;
}

// out = out - t mod p (both canonical); flags bit 0 when t >= p
__device__ __forceinline__ void sub_t(Num &out, const uint32_t *__restrict__ trow, uint32_t c0, uint32_t *flags, int lane) {
  Num t;
  load_row(t, trow, lane);
  if (ge_p(t, c0, lane) && lane == 0) atomicOr(flags, 1u);
  uint64_t b = 0;
#pragma unroll
  for (int q = 0; q < PER; q++) { const uint64_t v = (uint64_t)out.l[q] - t.l[q] - b; out.l[q] = (uint32_t)v; b = v >> 63; }
  if (lane == 31) {
    uint64_t v = (uint64_t)out.e0 - t.e0 - b;
    out.e0 = (uint32_t)v;
    b = v >> 63;
    v = (uint64_t)out.e1 - t.e1 - b;
    out.e1 = (uint32_t)v;
    b = v >> 63;
  }
  uint32_t bt = __shfl_sync(FULL, (uint32_t)b, 31);
  if (lane == 31) b = 0;
  bt |= borrow_across(out, (uint32_t)b, lane);
  if (bt) {   // + p = - c0 mod 2^W
    uint32_t bb = 0;
    if (lane == 0) {
      uint64_t v = (uint64_t)out.l[0] - c0;
      out.l[0] = (uint32_t)v;
      bb = (uint32_t)(v >> 63);
#pragma unroll
      for (int q = 1; q < PER; q++) { v = (uint64_t)out.l[q] - bb; out.l[q] = (uint32_t)v; bb = (uint32_t)(v >> 63); }
    }
    borrow_across(out, bb, lane);
  }
}

}  // namespace fd

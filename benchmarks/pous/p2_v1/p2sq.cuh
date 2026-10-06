// P2's decode at 16,448 bits on its own symmetric squaring, without CGBN: one block per warp, every value 514 32-bit limbs.
//
// The square is computed column by column of the cyclic product mod p = 2^W - c0 (W = 32 * 514): column c (mod 514) sums
// a_i a_j over the unordered pairs i + j = c (unwrapped, weight 1) or i + j = c + 514 (wrapped, weight c0, as 2^W = c0 mod
// p). Writing c = 2h or 2h + 1, the pairs are (h + k, h - k), k = 1..256, and (h + k, h + 1 - k), k = 1..257: every
// column has 256 or 257 of them, so 32 lanes that own 8 consecutive h each (16 columns) do equal work; h = 256 (columns
// 512 and 513) is shared by all lanes and reduced. A pair is wrapped exactly when its second index is negative, i.e. k > h
// (even) or k > h + 1 (odd).
//
// Lane t (h0 = 8t) walks k from 257 down to 1 in chunks of 8 (k = 8Q + 1 .. 8Q + 8, Q = 31 .. 0), sliding a window of
// a_{h+k} up-operands and a_{h-k} down-operands through registers (16 shared-memory loads per 128 products). Chunks Q > t
// are wrapped, Q < t unwrapped, and Q = t is mixed in a pattern that does not depend on t. One 96-bit accumulator per column
// takes every product; before chunk t the lane stores it (its wrapped part so far), and the mixed chunk's wrapped products
// are recomputed after the walk, so each column is  2 (acc + (c0 - 1) Y) + diagonal  with Y its wrapped sum.
//
// Shared memory holds a twice over its upper half (index x in [-257, 514) at OFF + x, a_x = a_{x mod 514}), one padding
// word every 8 (phys), so the lanes' loads, 8 limbs apart, fall in distinct banks.
#pragma once
#include <cstdint>

namespace p2sq {

constexpr int N = 514;
constexpr int OFF = 264;
constexpr int E_WORDS = 880;           // phys(OFF + 513) + 1, rounded up
constexpr int SNAP_STRIDE = 52;        // words per lane: 48 used, 16-byte aligned, conflict-free for 128-bit accesses
constexpr int SNAP_WORDS = 32 * SNAP_STRIDE;
constexpr int STAGE_WORDS = 552;       // stage(x) = x + x / 16 for x < 514
constexpr int WARP_WORDS = E_WORDS + SNAP_WORDS;   // the staging area aliases the snapshot area
typedef unsigned __int128 u128;

__device__ __forceinline__ int phys(int e) { return e + (e >> 3); }
__device__ __forceinline__ int stage(int x) { return x + (x >> 4); }

__device__ __forceinline__ void mac(uint32_t *r, uint32_t a, uint32_t b) {
  asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\tmadc.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;"
      : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]) : "r"(a), "r"(b));
}

// A value held by a warp: lane t has limbs 16t .. 16t + 15; lane 31 also limbs 512 (e0) and 513 (e1), zero elsewhere.
struct Num {
  uint32_t l[16];
  uint32_t e0, e1;
};

__device__ __forceinline__ bool all_ones(const Num &x, uint32_t lane) {
  uint32_t v = 0xFFFFFFFFu;
  #pragma unroll
  for (int q = 0; q < 16; q++) v &= x.l[q];
  if (lane == 31) v &= x.e0 & x.e1;
  return v == 0xFFFFFFFFu;
}
__device__ __forceinline__ bool all_zero(const Num &x, uint32_t lane) {
  uint32_t v = 0;
  #pragma unroll
  for (int q = 0; q < 16; q++) v |= x.l[q];
  if (lane == 31) v |= x.e0 | x.e1;
  return v == 0;
}
// x += cin (0 or 1) through the lane's limbs, the carry out dropped (the caller resolved it)
__device__ __forceinline__ void add_bit(Num &x, uint32_t cin, uint32_t lane) {
  uint64_t c = cin;
  #pragma unroll
  for (int q = 0; q < 16; q++) { c += x.l[q]; x.l[q] = (uint32_t)c; c >>= 32; }
  if (lane == 31) { c += x.e0; x.e0 = (uint32_t)c; c >>= 32; c += x.e1; x.e1 = (uint32_t)c; }
}
__device__ __forceinline__ void sub_bit(Num &x, uint32_t bin, uint32_t lane) {
  uint64_t b = bin;
  #pragma unroll
  for (int q = 0; q < 16; q++) { uint64_t v = (uint64_t)x.l[q] - b; x.l[q] = (uint32_t)v; b = (v >> 63); }
  if (lane == 31) {
    uint64_t v = (uint64_t)x.e0 - b; x.e0 = (uint32_t)v; b = v >> 63;
    v = (uint64_t)x.e1 - b; x.e1 = (uint32_t)v;
  }
}
// Each lane has added locally and carries g (0/1) out of its top limb: resolve the carries between lanes (a lane whose limbs
// are all ones passes one on) and return the carry out of limb 513, the same on every lane.
__device__ __forceinline__ uint32_t carry_across(Num &x, uint32_t g, uint32_t lane) {
  uint32_t G = __ballot_sync(0xFFFFFFFFu, g), A = G | __ballot_sync(0xFFFFFFFFu, all_ones(x, lane));
  uint64_t S = (uint64_t)A + G;
  uint32_t cin = (((uint32_t)S ^ A ^ G) >> lane) & 1;
  add_bit(x, cin, lane);
  return (uint32_t)(S >> 32);
}
__device__ __forceinline__ uint32_t borrow_across(Num &x, uint32_t b, uint32_t lane) {
  uint32_t B = __ballot_sync(0xFFFFFFFFu, b), A = B | __ballot_sync(0xFFFFFFFFu, all_zero(x, lane));
  uint64_t S = (uint64_t)A + B;
  uint32_t bin = (((uint32_t)S ^ A ^ B) >> lane) & 1;
  sub_bit(x, bin, lane);
  return (uint32_t)(S >> 32);
}
// a^2 mod p, canonical, from a in shared memory (sm: this warp's E area; snap: its snapshot area); returns it negated when
// odd is set (p - a^2 unless zero): P2's d(a).
__device__ __forceinline__ void sqr_d(const uint32_t *__restrict__ sm, uint32_t *__restrict__ snap, Num &out, bool odd,
                                      uint32_t c0, uint32_t lane) {
#define E(x) sm[phys(OFF + (x))]
  const int h0 = 8 * (int)lane;
  uint32_t acc[16][3];
  #pragma unroll
  for (int q = 0; q < 16; q++) acc[q][0] = acc[q][1] = acc[q][2] = 0;
  #pragma unroll
  for (int r = 0; r < 8; r++) mac(acc[2 * r + 1], E(h0 + r + 257), E(h0 + r + 258));   // odd columns' k = 257 (wrapped)

  uint32_t U0[8], U1[8], D0[8], D1[8];
  #pragma unroll
  for (int m = 0; m < 8; m++) {
    U0[m] = E(h0 + 249 + m);           // chunk Q = 31: U[m] = a_{h0 + 8Q + 1 + m}, D[m] = a_{h0 - 8Q - 8 + m}
    U1[m] = m < 7 ? E(h0 + 257 + m) : 0;
    D0[m] = E(h0 - 256 + m);
    D1[m] = E(h0 - 248 + m);
  }
#define P2SQ_CHUNK(Ulo, Uhi, Dlo, Dhi)                                                     \
  _Pragma("unroll") for (int r = 0; r < 8; r++)                                           \
  _Pragma("unroll") for (int s = 0; s < 8; s++) {                                         \
    const int mu = r + s, de = 7 + r - s, dd = 8 + r - s;                                 \
    const uint32_t u = mu < 8 ? Ulo[mu & 7] : Uhi[mu & 7];                                \
    mac(acc[2 * r], u, de < 8 ? Dlo[de & 7] : Dhi[de & 7]);                              \
    mac(acc[2 * r + 1], u, dd < 8 ? Dlo[dd & 7] : Dhi[dd & 7]);                          \
  }
#define P2SQ_SNAP(Q)                                                                       \
  if ((Q) == (int)lane) {                                                                 \
    _Pragma("unroll") for (int w = 0; w < 12; w++)                                        \
      reinterpret_cast<uint4 *>(snap + lane * SNAP_STRIDE)[w] =                           \
        make_uint4(acc[(4 * w) / 3][(4 * w) % 3], acc[(4 * w + 1) / 3][(4 * w + 1) % 3],  \
                   acc[(4 * w + 2) / 3][(4 * w + 2) % 3], acc[(4 * w + 3) / 3][(4 * w + 3) % 3]); \
  }
  #pragma unroll 1
  for (int Q = 31; Q >= 0; Q -= 2) {
    P2SQ_SNAP(Q)
    P2SQ_CHUNK(U0, U1, D0, D1)
    #pragma unroll
    for (int m = 0; m < 8; m++) { U1[m] = E(h0 + 8 * Q - 7 + m); D0[m] = E(h0 - 8 * Q + 8 + m); }
    P2SQ_SNAP(Q - 1)
    P2SQ_CHUNK(U1, U0, D1, D0)
    #pragma unroll
    for (int m = 0; m < 8; m++) { U0[m] = E(h0 + 8 * Q - 15 + m); D1[m] = E(h0 - 8 * Q + 16 + m); }
  }
#undef P2SQ_CHUNK
#undef P2SQ_SNAP

  // the mixed chunk Q = lane: its wrapped products (even s >= r, odd s > r), with U[m] = a_{16t + 1 + m}, D[m] = a_{m - 8}
  uint32_t my[16][3];
  #pragma unroll
  for (int q = 0; q < 16; q++) my[q][0] = my[q][1] = my[q][2] = 0;
  {
    uint32_t U[15], D[16];
    #pragma unroll
    for (int m = 0; m < 15; m++) U[m] = E(16 * (int)lane + 1 + m);
    #pragma unroll
    for (int m = 0; m < 16; m++) D[m] = E(m - 8);
    #pragma unroll
    for (int r = 0; r < 8; r++)
      #pragma unroll
      for (int s = 0; s < 8; s++) {
        if (s >= r) mac(my[2 * r], U[r + s], D[7 + r - s]);
        if (s > r) mac(my[2 * r + 1], U[r + s], D[8 + r - s]);
      }
  }
  // columns 512 and 513 (h = 256), all unwrapped but the diagonal a_513^2: this lane's k = 8t + 1 .. 8t + 8, then reduced
  u128 se = 0, so = 0;
  {
    uint32_t e3[3] = {0, 0, 0}, o3[3] = {0, 0, 0};
    #pragma unroll
    for (int s = 0; s < 8; s++) {
      const int k = h0 + 1 + s;
      const uint32_t u = E(256 + k);
      mac(e3, u, E(256 - k));
      mac(o3, u, E(257 - k));
    }
    if (lane == 0) mac(o3, E(513), E(0));
    se = ((u128)e3[2] << 64) | ((u128)e3[1] << 32) | e3[0];
    so = ((u128)o3[2] << 64) | ((u128)o3[1] << 32) | o3[0];
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      uint64_t elo = (uint64_t)se, ehi = (uint64_t)(se >> 64), olo = (uint64_t)so, ohi = (uint64_t)(so >> 64);
      se += ((u128)__shfl_xor_sync(0xFFFFFFFFu, ehi, d) << 64) | __shfl_xor_sync(0xFFFFFFFFu, elo, d);
      so += ((u128)__shfl_xor_sync(0xFFFFFFFFu, ohi, d) << 64) | __shfl_xor_sync(0xFFFFFFFFu, olo, d);
    }
  }
  __syncwarp();
  // every column: 2 (acc + (c0 - 1) Y) + diagonal (even columns 2h: a_h^2 + c0 a_{h+257}^2), then carried into limbs
  u128 carry = 0;
  #pragma unroll
  for (int q = 0; q < 16; q++) {
    const uint32_t *sv = snap + lane * SNAP_STRIDE;
    u128 y = ((u128)(sv[3 * q + 2] + 0) << 64) + ((u128)sv[3 * q + 1] << 32) + sv[3 * q];
    y += ((u128)my[q][2] << 64) + ((u128)my[q][1] << 32) + my[q][0];
    u128 v = ((u128)acc[q][2] << 64) + ((u128)acc[q][1] << 32) + acc[q][0];
    v = 2 * (v + (u128)(c0 - 1) * y);
    if ((q & 1) == 0) {
      const uint32_t lo = E(h0 + q / 2), hi = E(h0 + q / 2 + 257);
      v += (u128)((uint64_t)lo * lo) + (u128)c0 * (uint64_t)hi * hi;
    }
    v += carry;
    out.l[q] = (uint32_t)v;
    carry = v >> 32;
  }
  out.e0 = out.e1 = 0;
  // lane 31 continues through limbs 512, 513; its carry out has weight 2^W = c0
  u128 top = 0;
  if (lane == 31) {
    const uint32_t lo = E(256), hi = E(513);
    u128 v = 2 * se + (uint64_t)lo * lo + (u128)c0 * ((uint64_t)hi * hi) + carry;
    out.e0 = (uint32_t)v;
    v = 2 * so + (v >> 32);
    out.e1 = (uint32_t)v;
    top = (v >> 32) * c0;
    carry = 0;
  }
#undef E
  // pass each lane's carry (< 2^61) up; lane 0 takes c0 times lane 31's
  uint64_t c64 = (uint64_t)carry, t_lo = (uint64_t)top, t_hi = (uint64_t)(top >> 64);
  uint64_t in = __shfl_up_sync(0xFFFFFFFFu, c64, 1);
  uint64_t in_lo = __shfl_sync(0xFFFFFFFFu, t_lo, 31), in_hi = __shfl_sync(0xFFFFFFFFu, t_hi, 31);
  u128 inc = lane == 0 ? (((u128)in_hi << 64) | in_lo) : (u128)in;
  #pragma unroll
  for (int q = 0; q < 16; q++) { inc += out.l[q]; out.l[q] = (uint32_t)inc; inc >>= 32; }
  if (lane == 31) { inc += out.e0; out.e0 = (uint32_t)inc; inc >>= 32; inc += out.e1; out.e1 = (uint32_t)inc; inc >>= 32; }
  uint32_t g = (uint32_t)inc;                                    // 0 or 1; lane 31's has weight 2^W
  uint32_t topc = __shfl_sync(0xFFFFFFFFu, g, 31);
  if (lane == 31) g = 0;
  topc += carry_across(out, g, lane);
  // fold the carries out of limb 513 and reduce below p
  while (topc) {
    uint32_t gg = 0;
    if (lane == 0) {
      uint64_t c = (uint64_t)topc * c0;
      #pragma unroll
      for (int q = 0; q < 16; q++) { c += out.l[q]; out.l[q] = (uint32_t)c; c >>= 32; }
      gg = (uint32_t)c;
    }
    topc = carry_across(out, gg, lane);
  }
  {
    uint32_t v = 0xFFFFFFFFu;
    #pragma unroll
    for (int q = 1; q < 16; q++) v &= out.l[q];
    if (lane != 0) v &= out.l[0];
    if (lane == 31) v &= out.e0 & out.e1;
    bool ge = (lane != 0 || out.l[0] >= 0u - c0) && v == 0xFFFFFFFFu;
    if (__all_sync(0xFFFFFFFFu, ge)) {
      uint32_t l0 = out.l[0] + c0;
      #pragma unroll
      for (int q = 0; q < 16; q++) out.l[q] = 0;
      out.e0 = out.e1 = 0;
      if (lane == 0) out.l[0] = l0;
    }
  }
  // d: p - a^2 when a is odd and a^2 != 0; p - x = (2^W - 1 - x) - (c0 - 1)
  if (odd && !__all_sync(0xFFFFFFFFu, all_zero(out, lane))) {
    #pragma unroll
    for (int q = 0; q < 16; q++) out.l[q] = ~out.l[q];
    if (lane == 31) { out.e0 = ~out.e0; out.e1 = ~out.e1; }
    uint32_t b = 0;
    if (lane == 0) {
      uint64_t v = (uint64_t)out.l[0] - (c0 - 1);
      out.l[0] = (uint32_t)v;
      b = (uint32_t)(v >> 63);
      #pragma unroll
      for (int q = 1; q < 16; q++) { v = (uint64_t)out.l[q] - b; out.l[q] = (uint32_t)v; b = (uint32_t)(v >> 63); }
    }
    borrow_across(out, b, lane);
  }
}

// rows of NL big-endian words <-> limbs: through the staging area, coalesced on the global side
__device__ __forceinline__ void load_row(uint32_t *__restrict__ st, const uint32_t *__restrict__ row, Num &x, uint32_t lane) {
  __syncwarp();
  for (int w = lane; w < N; w += 32) st[stage(N - 1 - w)] = __byte_perm(row[w], 0, 0x0123);
  __syncwarp();
  #pragma unroll
  for (int q = 0; q < 16; q++) x.l[q] = st[stage(16 * lane + q)];
  x.e0 = lane == 31 ? st[stage(512)] : 0;
  x.e1 = lane == 31 ? st[stage(513)] : 0;
  __syncwarp();
}
__device__ __forceinline__ void store_row(uint32_t *__restrict__ st, uint32_t *__restrict__ row, const Num &x, uint32_t lane) {
  __syncwarp();
  #pragma unroll
  for (int q = 0; q < 16; q++) st[stage(16 * lane + q)] = x.l[q];
  if (lane == 31) { st[stage(512)] = x.e0; st[stage(513)] = x.e1; }
  __syncwarp();
  for (int w = lane; w < N; w += 32) row[w] = __byte_perm(st[stage(N - 1 - w)], 0, 0x0123);
  __syncwarp();
}
// x into the E area (index x and, for x >= 257, x - 514)
__device__ __forceinline__ void put_e(uint32_t *__restrict__ sm, const Num &x, uint32_t lane) {
  __syncwarp();
  #pragma unroll
  for (int q = 0; q < 16; q++) {
    const int i = 16 * (int)lane + q;
    sm[phys(OFF + i)] = x.l[q];
    if (i >= 257) sm[phys(OFF + i - N)] = x.l[q];
  }
  if (lane == 31) {
    sm[phys(OFF + 512)] = x.e0; sm[phys(OFF + 513)] = x.e1;
    sm[phys(OFF - 2)] = x.e0; sm[phys(OFF - 1)] = x.e1;
  }
  __syncwarp();
}
__device__ __forceinline__ bool odd_of(const Num &x) { return __shfl_sync(0xFFFFFFFFu, x.l[0] & 1, 0) != 0; }
// x >= p
__device__ __forceinline__ bool ge_p(const Num &x, uint32_t c0, uint32_t lane) {
  uint32_t v = 0xFFFFFFFFu;
  #pragma unroll
  for (int q = 1; q < 16; q++) v &= x.l[q];
  if (lane != 0) v &= x.l[0];
  if (lane == 31) v &= x.e0 & x.e1;
  return __all_sync(0xFFFFFFFFu, (lane != 0 || x.l[0] >= 0u - c0) && v == 0xFFFFFFFFu);
}

}  // namespace p2sq

#ifndef P2SQ_WARPS
#define P2SQ_WARPS 2
#endif

// m = d(sigma_K(d(c))) - t mod p, one block per warp, K and t as rows
__global__ void __launch_bounds__(32 * P2SQ_WARPS) decode_sq_kernel(const uint32_t *__restrict__ cin, uint32_t *__restrict__ mout,
                                                                    const uint32_t *__restrict__ kin, const uint32_t *__restrict__ tin,
                                                                    uint32_t n, uint32_t c0, uint32_t *__restrict__ flags) {
  using namespace p2sq;
  __shared__ __align__(16) uint32_t smem[P2SQ_WARPS][WARP_WORDS];
  const uint32_t lane = threadIdx.x % 32, wid = threadIdx.x / 32;
  uint32_t *sm = smem[wid], *snap = smem[wid] + E_WORDS;
  const uint32_t stride = gridDim.x * P2SQ_WARPS;
  for (uint32_t j = blockIdx.x * P2SQ_WARPS + wid; j < n; j += stride) {
    Num x, k;
    load_row(snap, cin + (size_t)j * N, x, lane);
    put_e(sm, x, lane);
    sqr_d(sm, snap, x, odd_of(x), c0, lane);                     // x = d(c)
    __syncwarp();
    load_row(snap, kin + (size_t)j * N, k, lane);
    #pragma unroll
    for (int q = 0; q < 16; q++) k.l[q] ^= x.l[q];
    k.e0 ^= x.e0; k.e1 ^= x.e1;
    if (!ge_p(k, c0, lane)) x = k;                               // sigma_K
    put_e(sm, x, lane);
    sqr_d(sm, snap, x, odd_of(x), c0, lane);                     // x = d(sigma_K(d(c)))
    __syncwarp();
    load_row(snap, tin + (size_t)j * N, k, lane);
    if (ge_p(k, c0, lane) && lane == 0) atomicOr(flags, 1u);
    uint32_t b = 0;                                              // m = x - t mod p
    {
      uint64_t v;
      #pragma unroll
      for (int q = 0; q < 16; q++) { v = (uint64_t)x.l[q] - k.l[q] - b; x.l[q] = (uint32_t)v; b = (uint32_t)(v >> 63); }
      if (lane == 31) {
        v = (uint64_t)x.e0 - k.e0 - b; x.e0 = (uint32_t)v; b = (uint32_t)(v >> 63);
        v = (uint64_t)x.e1 - k.e1 - b; x.e1 = (uint32_t)v; b = (uint32_t)(v >> 63);
      }
    }
    uint32_t topb = __shfl_sync(0xFFFFFFFFu, b, 31);
    if (lane == 31) b = 0;
    topb += borrow_across(x, b, lane);
    if (topb) {                                                  // x - t + 2^W, then - c0: x - t + p
      uint32_t bb = 0;
      if (lane == 0) {
        uint64_t v = (uint64_t)x.l[0] - c0;
        x.l[0] = (uint32_t)v; bb = (uint32_t)(v >> 63);
        #pragma unroll
        for (int q = 1; q < 16; q++) { v = (uint64_t)x.l[q] - bb; x.l[q] = (uint32_t)v; bb = (uint32_t)(v >> 63); }
      }
      borrow_across(x, bb, lane);
    }
    store_row(snap, mout + (size_t)j * N, x, lane);
  }
}

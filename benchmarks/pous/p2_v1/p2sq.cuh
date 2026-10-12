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

__device__ __forceinline__ int phys(int e) { return e + (e >> 3); }
__device__ __forceinline__ int stage(int x) { return x + (x >> 4); }

__device__ __forceinline__ void mac(uint32_t *r, uint32_t a, uint32_t b) {
  asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\tmadc.hi.cc.u32 %1, %3, %4, %1;\n\taddc.u32 %2, %2, 0;"
      : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]) : "r"(a), "r"(b));
}

__device__ __forceinline__ void add3(uint32_t *r, uint32_t a, uint32_t b, uint32_t c) {
  asm("add.cc.u32 %0, %0, %3;\n\taddc.cc.u32 %1, %1, %4;\n\taddc.u32 %2, %2, %5;" : "+r"(r[0]), "+r"(r[1]), "+r"(r[2])
      : "r"(a), "r"(b), "r"(c));
}
// r (3 words) += lo^2 + c0 hi^2
__device__ __forceinline__ void add_diag(uint32_t *r, uint32_t lo, uint32_t hi, uint32_t c0) {
  const uint32_t h0 = hi * hi, h1 = __umulhi(hi, hi);
  asm("mad.lo.cc.u32 %0, %3, %3, %0;\n\tmadc.hi.cc.u32 %1, %3, %3, %1;\n\taddc.u32 %2, %2, 0;\n\t"
      "mad.lo.cc.u32 %0, %4, %6, %0;\n\tmadc.lo.cc.u32 %1, %5, %6, %1;\n\taddc.u32 %2, %2, 0;\n\t"
      "mad.hi.cc.u32 %1, %4, %6, %1;\n\tmadc.hi.u32 %2, %5, %6, %2;"
      : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]) : "r"(lo), "r"(h0), "r"(h1), "r"(c0));
}
// l (16 words) += a + b 2^32 + c 2^64; returns the carry out
__device__ __forceinline__ uint32_t add_in(uint32_t *l, uint32_t a, uint32_t b, uint32_t c) {
  uint32_t g;
  asm("add.cc.u32 %0, %0, %17;\n\taddc.cc.u32 %1, %1, %18;\n\taddc.cc.u32 %2, %2, %19;\n\t"
      "addc.cc.u32 %3, %3, 0;\n\taddc.cc.u32 %4, %4, 0;\n\taddc.cc.u32 %5, %5, 0;\n\taddc.cc.u32 %6, %6, 0;\n\t"
      "addc.cc.u32 %7, %7, 0;\n\taddc.cc.u32 %8, %8, 0;\n\taddc.cc.u32 %9, %9, 0;\n\taddc.cc.u32 %10, %10, 0;\n\t"
      "addc.cc.u32 %11, %11, 0;\n\taddc.cc.u32 %12, %12, 0;\n\taddc.cc.u32 %13, %13, 0;\n\taddc.cc.u32 %14, %14, 0;\n\t"
      "addc.cc.u32 %15, %15, 0;\n\taddc.u32 %16, 0, 0;"
      : "+r"(l[0]), "+r"(l[1]), "+r"(l[2]), "+r"(l[3]), "+r"(l[4]), "+r"(l[5]), "+r"(l[6]), "+r"(l[7]), "+r"(l[8]),
        "+r"(l[9]), "+r"(l[10]), "+r"(l[11]), "+r"(l[12]), "+r"(l[13]), "+r"(l[14]), "+r"(l[15]), "=r"(g)
      : "r"(a), "r"(b), "r"(c));
  return g;
}

// A value held by a warp: lane t has limbs 16t .. 16t + 15; lane 31 also limbs 512 (e0) and 513 (e1), zero elsewhere.
struct Num {
  uint32_t l[16];
  uint32_t e0, e1;
};

// lane 31's limbs 512, 513 += g; returns the carry out of limb 513
__device__ __forceinline__ uint32_t add_e(Num &x, uint32_t g) {
  uint32_t c;
  asm("add.cc.u32 %0, %0, %3;\n\taddc.cc.u32 %1, %1, 0;\n\taddc.u32 %2, 0, 0;" : "+r"(x.e0), "+r"(x.e1), "=r"(c) : "r"(g));
  return c;
}
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
// out (< 2^W) reduced below p, then negated when odd is set (p - out unless zero): the end of P2's d
__device__ __forceinline__ void finish(Num &out, bool odd, uint32_t c0, uint32_t lane) {
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
#ifndef P2SQ_ORDER
#define P2SQ_ORDER 0
#endif
#if P2SQ_ORDER == 1
#define P2SQ_LOOPS _Pragma("unroll") for (int s = 0; s < 8; s++) _Pragma("unroll") for (int r = 0; r < 8; r++)
#elif P2SQ_ORDER == 2
#define P2SQ_LOOPS _Pragma("unroll") for (int i = 0; i < 64; i++)
#else
#define P2SQ_LOOPS _Pragma("unroll") for (int r = 0; r < 8; r++) _Pragma("unroll") for (int s = 0; s < 8; s++)
#endif
#if P2SQ_ORDER == 2
#define P2SQ_RS const int s = 2 * ((i >> 4) & 3) + (i & 1), r = (i >> 1) & 7;  // two products of an accumulator, eight apart
#else
#define P2SQ_RS
#endif
#define P2SQ_CHUNK(Ulo, Uhi, Dlo, Dhi)                                                     \
  P2SQ_LOOPS {                                                                            \
    P2SQ_RS                                                                               \
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
  // E(h0 + 8Q - 7 + m) = pu[m + (m == 7)] and E(h0 - 8Q + 8 + m) = pd[m] (phys of 8B + 1 + m, of 8B + m): constant steps
  const uint32_t *pu = sm + 9 * (32 + (int)lane + 31) + 1, *pd = sm + 9 * (34 + (int)lane - 31);
  #pragma unroll 1
  for (int Q = 31; Q >= 0; Q -= 2) {
    P2SQ_SNAP(Q)
    P2SQ_CHUNK(U0, U1, D0, D1)
    #pragma unroll
    for (int m = 0; m < 8; m++) { U1[m] = pu[m + (m == 7)]; D0[m] = pd[m]; }
    P2SQ_SNAP(Q - 1)
    P2SQ_CHUNK(U1, U0, D1, D0)
    #pragma unroll
    for (int m = 0; m < 8; m++) { U0[m] = pu[m + (m == 7) - 9]; D1[m] = pd[m + 9]; }
    pu -= 18;
    pd += 18;
  }
#undef P2SQ_CHUNK
#undef P2SQ_LOOPS
#undef P2SQ_RS
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
  uint32_t se[3] = {0, 0, 0}, so[3] = {0, 0, 0};
  #pragma unroll
  for (int s = 0; s < 8; s++) {
    const int k = h0 + 1 + s;
    const uint32_t u = E(256 + k);
    mac(se, u, E(256 - k));
    mac(so, u, E(257 - k));
  }
  if (lane == 0) mac(so, E(513), E(0));
  #pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    add3(se, __shfl_xor_sync(0xFFFFFFFFu, se[0], d), __shfl_xor_sync(0xFFFFFFFFu, se[1], d), __shfl_xor_sync(0xFFFFFFFFu, se[2], d));
    add3(so, __shfl_xor_sync(0xFFFFFFFFu, so[0], d), __shfl_xor_sync(0xFFFFFFFFu, so[1], d), __shfl_xor_sync(0xFFFFFFFFu, so[2], d));
  }
  __syncwarp();
  // every column: 2 (acc + (c0 - 1) Y) + diagonal (even columns 2h: a_h^2 + c0 a_{h+257}^2), then carried into limbs. The
  // bounds: acc, Y < 2^73 (257 products), acc + (c0 - 1) Y < 2^88, a column with its carry in < 2^90, its carry out < 2^58.
  const uint32_t k = c0 - 1, *sv = snap + lane * SNAP_STRIDE;
  uint32_t cl = 0, ch = 0;
  #pragma unroll
  for (int q = 0; q < 16; q++) {
    uint32_t t0 = acc[q][0], t1 = acc[q][1], t2 = acc[q][2], y0 = sv[3 * q], y1 = sv[3 * q + 1], y2 = sv[3 * q + 2];
    asm("add.cc.u32 %3, %3, %7;\n\taddc.cc.u32 %4, %4, %8;\n\taddc.u32 %5, %5, %9;\n\t"
        "mad.lo.cc.u32 %0, %3, %6, %0;\n\tmadc.lo.cc.u32 %1, %4, %6, %1;\n\tmadc.lo.u32 %2, %5, %6, %2;\n\t"
        "mad.hi.cc.u32 %1, %3, %6, %1;\n\tmadc.hi.u32 %2, %4, %6, %2;"
        : "+r"(t0), "+r"(t1), "+r"(t2), "+r"(y0), "+r"(y1), "+r"(y2) : "r"(k), "r"(my[q][0]), "r"(my[q][1]), "r"(my[q][2]));
    uint32_t v[3] = {t0 << 1, __funnelshift_l(t0, t1, 1), __funnelshift_l(t1, t2, 1)};
    add3(v, cl, ch, 0);
    if ((q & 1) == 0) add_diag(v, E(h0 + q / 2), E(h0 + q / 2 + 257), c0);
    out.l[q] = v[0];
    cl = v[1];
    ch = v[2];
  }
  out.e0 = out.e1 = 0;
  // lane 31 continues through limbs 512, 513; its carry out has weight 2^W = c0
  uint32_t top[3] = {0, 0, 0};
  if (lane == 31) {
    uint32_t v[3] = {se[0] << 1, __funnelshift_l(se[0], se[1], 1), __funnelshift_l(se[1], se[2], 1)};
    add3(v, cl, ch, 0);
    add_diag(v, E(256), E(513), c0);
    out.e0 = v[0];
    uint32_t o[3] = {so[0] << 1, __funnelshift_l(so[0], so[1], 1), __funnelshift_l(so[1], so[2], 1)};
    add3(o, v[1], v[2], 0);
    out.e1 = o[0];
    const uint64_t c = ((uint64_t)o[2] << 32 | o[1]);
    const uint64_t lo = (uint64_t)(uint32_t)c * c0, hi = (c >> 32) * c0 + (lo >> 32);
    top[0] = (uint32_t)lo;
    top[1] = (uint32_t)hi;
    top[2] = (uint32_t)(hi >> 32);
    cl = ch = 0;
  }
#undef E
  // pass each lane's carry (< 2^58) up; lane 0 takes c0 times lane 31's (< 2^73)
  uint32_t in0 = __shfl_up_sync(0xFFFFFFFFu, cl, 1), in1 = __shfl_up_sync(0xFFFFFFFFu, ch, 1), in2 = 0;
  {
    const uint32_t t0 = __shfl_sync(0xFFFFFFFFu, top[0], 31), t1 = __shfl_sync(0xFFFFFFFFu, top[1], 31),
                   t2 = __shfl_sync(0xFFFFFFFFu, top[2], 31);
    if (lane == 0) { in0 = t0; in1 = t1; in2 = t2; }
  }
  uint32_t g = add_in(out.l, in0, in1, in2);                     // 0 or 1
  if (lane == 31) g = add_e(out, g);                             // lane 31's carry has weight 2^W
  uint32_t topc = __shfl_sync(0xFFFFFFFFu, g, 31);
  if (lane == 31) g = 0;
  topc += carry_across(out, g, lane);
  // fold the carries out of limb 513 and reduce below p
  while (topc) {
    uint32_t gg = 0;
    if (lane == 0) gg = add_in(out.l, topc * c0, 0, 0);
    topc = carry_across(out, gg, lane);
  }
  finish(out, odd, c0, lane);
}

// The same square by one Karatsuba step (P2SQ_KARA). With X = 2^(32 * 257) and a = A0 + A1 X (257 limbs each), X^2 = 2^W = c0
// mod p, and 2 A0 A1 = A0^2 + A1^2 - D^2 for D = |A0 - A1|, a^2 is assembled from three cyclic squares mod x^257 - 1. Square S's
// column d (0 .. 256) has T_S (all its products) and Y_S (those with i + j = d + 257, so Y_S(256) = 0); then
//   column d       = T0 + c0 T1 + (c0 - 1) Y0 - c0 YD,
//   column d + 257 = T0 + T1 - TD + (c0 - 1) Y1 + YD.
// Columns can be negative, so every one starts at 2^92 - 2^60 and column 0 at 2^92 - c0 2^60 (in all 2^60 p, 0 mod p); the
// columns then lie in [0, 2^93) and are summed mod 2^96.
//
// A cyclic square of 257 limbs is the 514-limb walk at half the size: column 2h (h = 0 .. 128) pairs (h + k, h - k), column
// 2h + 1 (h = 0 .. 127) pairs (h + k, h + 1 - k), k = 1 .. 128, wrapped exactly when the second index is negative (k > h,
// k > h + 1), plus the diagonal s_h^2 (even, unwrapped) or s_{h+129}^2 (odd, wrapped). Lane t owns h = 4t .. 4t + 3 (columns
// 8t .. 8t + 7) and walks k down in chunks of 8 (Q = 15 .. 0): Q > t / 2 wrapped, Q < t / 2 unwrapped, Q = t / 2 mixed (a
// triangle of 4 k on the boundary, one half of the chunk or the other by t's parity). Column 256 is shared and reduced. Lane t holds the result's limbs 8t .. 8t + 7 and 257 + 8t ..
// 264 + 8t (lane 31 also 256 and 513), a ring of 64 runs whose carries resolve over 64-bit ballots.
constexpr int KOFF = 128;              // index x in [-128, 257) at kphys(KOFF + x), s_x = s_{x mod 257}
constexpr int K_ARR = 484;             // kphys(384) + 1, rounded up
constexpr int KSNAP = 3 * K_ARR;       // 16-byte aligned
constexpr int KSNAP_STRIDE = 24;       // a lane's 8 columns: low words in pairs, then the top words
static_assert(KSNAP + 32 * KSNAP_STRIDE <= WARP_WORDS, "the Karatsuba arrays fit a warp's shared memory");
static_assert(STAGE_WORDS <= KSNAP, "the result's staging does not touch the snapshots");
__device__ __forceinline__ int kphys(int e) { return e + (e >> 2); }   // lanes' loads, 4 limbs apart, in distinct banks

// Snapshot stores under a predicate, which the compiler cannot turn into a divergent branch, and the loads that read them back
// after the walk: volatile, so they stay after the stores, and clobbering memory, so later stores stay after them.
__device__ __forceinline__ uint32_t saddr(const uint32_t *a) { return static_cast<uint32_t>(__cvta_generic_to_shared(a)); }
__device__ __forceinline__ void st2_if(bool p, uint32_t *a, uint32_t x, uint32_t y) {
  asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.u32 q, %0, 0;\n\t@q st.shared.v2.u32 [%1], {%2, %3};\n\t}"
               :: "r"((uint32_t)p), "r"(saddr(a)), "r"(x), "r"(y));
}
__device__ __forceinline__ void st1_if(bool p, uint32_t *a, uint32_t x) {
  asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.u32 q, %0, 0;\n\t@q st.shared.u32 [%1], %2;\n\t}" :: "r"((uint32_t)p), "r"(saddr(a)), "r"(x));
}
__device__ __forceinline__ void ld2_ord(const uint32_t *a, uint32_t &x, uint32_t &y) {
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(saddr(a)) : "memory");
}
__device__ __forceinline__ uint32_t ld1_ord(const uint32_t *a) {
  uint32_t x;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(saddr(a)) : "memory");
  return x;
}

__device__ __forceinline__ void add96(uint32_t *r, const uint32_t *x) { add3(r, x[0], x[1], x[2]); }
__device__ __forceinline__ void sub96(uint32_t *r, const uint32_t *x) {
  asm("sub.cc.u32 %0, %0, %3;\n\tsubc.cc.u32 %1, %1, %4;\n\tsubc.u32 %2, %2, %5;" : "+r"(r[0]), "+r"(r[1]), "+r"(r[2])
      : "r"(x[0]), "r"(x[1]), "r"(x[2]));
}
// r += x k mod 2^96
__device__ __forceinline__ void mad96(uint32_t *r, const uint32_t *x, uint32_t k) {
  asm("mad.lo.cc.u32 %0, %3, %6, %0;\n\tmadc.hi.cc.u32 %1, %3, %6, %1;\n\tmadc.hi.u32 %2, %4, %6, %2;\n\t"
      "mad.lo.cc.u32 %1, %4, %6, %1;\n\tmadc.lo.u32 %2, %5, %6, %2;"
      : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]) : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(k));
}
// r -= x k mod 2^96
__device__ __forceinline__ void msb96(uint32_t *r, const uint32_t *x, uint32_t k) {
  uint32_t p[3] = {0, 0, 0};
  mad96(p, x, k);
  sub96(r, p);
}
// r = 2 x + s^2
__device__ __forceinline__ void dbl_sq(uint32_t *r, const uint32_t *x, uint32_t s) {
  r[0] = x[0] << 1; r[1] = __funnelshift_l(x[0], x[1], 1); r[2] = __funnelshift_l(x[1], x[2], 1);
  mac(r, s, s);
}

// run (8 limbs, 9 on lane 31) += cin; returns the carry out
__device__ __forceinline__ uint32_t run_add(uint32_t *v, uint64_t cin, uint32_t lane) {
  #pragma unroll
  for (int m = 0; m < 8; m++) { cin += v[m]; v[m] = (uint32_t)cin; cin >>= 32; }
  if (lane == 31) { cin += v[8]; v[8] = (uint32_t)cin; cin >>= 32; }
  return (uint32_t)cin;
}
__device__ __forceinline__ bool run_ones(const uint32_t *v, uint32_t lane) {
  uint32_t o = 0xFFFFFFFFu;
  #pragma unroll
  for (int m = 0; m < 8; m++) o &= v[m];
  if (lane == 31) o &= v[8];
  return o == 0xFFFFFFFFu;
}

// a's halves A0, A1 and D = |A0 - A1| into this warp's three arrays (D: lane t's limbs 8t .. 8t + 7, lane 31's also 256)
__device__ __forceinline__ void put_k(uint32_t *__restrict__ sm, const Num &x, uint32_t lane) {
  uint32_t *SA = sm, *SB = sm + K_ARR, *SD = sm + 2 * K_ARR;
#define KS(a, i) (a)[kphys(KOFF + (i))]
  __syncwarp();
  #pragma unroll
  for (int q = 0; q < 16; q++) {
    const int i = 16 * (int)lane + q, j = i < 257 ? i : i - 257;
    uint32_t *S = i < 257 ? SA : SB;
    KS(S, j) = x.l[q];
    if (j >= 129) KS(S, j - 257) = x.l[q];
  }
  if (lane == 31) { KS(SB, 255) = x.e0; KS(SB, -2) = x.e0; KS(SB, 256) = x.e1; KS(SB, -1) = x.e1; }
  __syncwarp();
  const int i0 = 8 * (int)lane;
  uint32_t d[9], b = 0;
  #pragma unroll
  for (int m = 0; m < 8; m++) {
    const uint64_t v = (uint64_t)KS(SA, i0 + m) - KS(SB, i0 + m) - b;
    d[m] = (uint32_t)v;
    b = (uint32_t)(v >> 63);
  }
  d[8] = 0;
  if (lane == 31) {
    const uint64_t v = (uint64_t)KS(SA, 256) - KS(SB, 256) - b;
    d[8] = (uint32_t)v;
    b = (uint32_t)(v >> 63);
  }
  // borrows between lanes (a run of zeros passes one on); the one out of the top is A0 < A1
  uint32_t neg = __shfl_sync(0xFFFFFFFFu, b, 31);
  if (lane == 31) b = 0;
  {
    uint32_t z = 0;
    #pragma unroll
    for (int m = 0; m < 9; m++) z |= d[m];
    const uint32_t B = __ballot_sync(0xFFFFFFFFu, b), A = B | __ballot_sync(0xFFFFFFFFu, z == 0);
    const uint64_t S = (uint64_t)A + B;
    uint64_t bin = (((uint32_t)S ^ A ^ B) >> lane) & 1;
    neg += (uint32_t)(S >> 32);
    #pragma unroll
    for (int m = 0; m < 8; m++) { const uint64_t v = (uint64_t)d[m] - bin; d[m] = (uint32_t)v; bin = v >> 63; }
    if (lane == 31) d[8] -= (uint32_t)bin;
  }
  if (neg) {   // D = 2^(32 * 257) - (A0 - A1 mod 2^(32 * 257)): the complement plus one
    #pragma unroll
    for (int m = 0; m < 9; m++) d[m] = ~d[m];
    if (lane != 31) d[8] = 0;
    const uint32_t g = lane == 0 ? run_add(d, 1, lane) : 0;
    const uint32_t G = __ballot_sync(0xFFFFFFFFu, g), A = G | __ballot_sync(0xFFFFFFFFu, run_ones(d, lane));
    run_add(d, ((A + G) ^ A ^ G) >> lane & 1, lane);
  }
  #pragma unroll
  for (int m = 0; m < 8; m++) {
    const int i = i0 + m;
    KS(SD, i) = d[m];
    if (i >= 129) KS(SD, i - 257) = d[m];
  }
  if (lane == 31) { KS(SD, 256) = d[8]; KS(SD, -1) = d[8]; }
  __syncwarp();
}

// The cyclic square of the 257 limbs at S: T and Y of this lane's columns 8t + j (j = 2r: h = 4t + r even, j = 2r + 1 odd),
// and T of column 256 on every lane.
__device__ __forceinline__ void ksq(const uint32_t *__restrict__ S, uint32_t *__restrict__ snap, uint32_t (&T)[8][3],
                                    uint32_t (&Y)[8][3], uint32_t (&t256)[3], uint32_t lane) {
  const int h0 = 4 * (int)lane;
  uint32_t acc[8][3];
  #pragma unroll
  for (int j = 0; j < 8; j++) acc[j][0] = acc[j][1] = acc[j][2] = 0;
  // chunk Q: U[m] = s_{h0 + 8Q + 1 + m} (m < 11), D[m] = s_{h0 - 8Q - 8 + m} (m < 12); at kphys: pu[m + (m + 1) / 4] with
  // pu = S + 5 (32 + t + 2Q) + 1, pd[m + m / 4] with pd = S + 5 (30 + t - 2Q)
  const uint32_t *pu = S + 5 * (32 + (int)lane + 30) + 1, *pd = S + 5 * (30 + (int)lane - 30);
  uint32_t Ua[8], Ub[8], Da[8], Db[8];
  #pragma unroll
  for (int m = 0; m < 8; m++) {
    Ua[m] = pu[m + (m + 1) / 4];
    Ub[m] = m < 3 ? pu[m + 8 + (m + 9) / 4] : 0;
    Da[m] = pd[m + m / 4];
    Db[m] = m < 4 ? pd[m + 8 + (m + 8) / 4] : 0;
  }
#ifndef P2SQ_KORDER
#define P2SQ_KORDER 1
#endif
#if P2SQ_KORDER == 1
#define KLOOPS _Pragma("unroll") for (int s = 0; s < 8; s++) _Pragma("unroll") for (int r = 0; r < 4; r++)
#else
#define KLOOPS _Pragma("unroll") for (int r = 0; r < 4; r++) _Pragma("unroll") for (int s = 0; s < 8; s++)
#endif
// half H of a chunk: s = 4H .. 4H + 3 (k = 8Q + 1 + s)
#define KHALF(Ulo, Uhi, Dlo, Dhi, H)                                                       \
  KLOOPS if (s / 4 == (H)) {                                                              \
    const int mu = r + s, de = 7 + r - s, dd = 8 + r - s;                                 \
    const uint32_t u = mu < 8 ? Ulo[mu & 7] : Uhi[mu & 7];                                \
    mac(acc[2 * r], u, de < 8 ? Dlo[de & 7] : Dhi[de & 7]);                              \
    mac(acc[2 * r + 1], u, dd < 8 ? Dlo[dd & 7] : Dhi[dd & 7]);                          \
  }
#define KSNAPQ(Q)                                                                          \
  {                                                                                       \
    const bool p_ = (Q) == (int)(lane >> 1);                                              \
    _Pragma("unroll") for (int j = 0; j < 8; j++) {                                       \
      st2_if(p_, snap + lane * KSNAP_STRIDE + 2 * j, acc[j][0], acc[j][1]);               \
      st1_if(p_, snap + lane * KSNAP_STRIDE + 16 + j, acc[j][2]);                         \
    }                                                                                     \
  }
#define KCHUNK(Ulo, Uhi, Dlo, Dhi, Q)                                                      \
  KHALF(Ulo, Uhi, Dlo, Dhi, 1)                                                            \
  KSNAPQ(Q)                                                                               \
  KHALF(Ulo, Uhi, Dlo, Dhi, 0)
  #pragma unroll 1
  for (int Q = 15; Q >= 0; Q -= 2) {
    KCHUNK(Ua, Ub, Da, Db, Q)
    pu -= 10;
    pd += 10;
    #pragma unroll
    for (int m = 0; m < 8; m++) Ub[m] = pu[m + (m + 1) / 4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { Db[4 + i] = pd[4 + i + (4 + i) / 4]; Da[i] = pd[8 + i + (8 + i) / 4]; }
    KCHUNK(Ub, Ua, Db, Da, Q - 1)
    pu -= 10;
    pd += 10;
    #pragma unroll
    for (int m = 0; m < 8; m++) Ua[m] = pu[m + (m + 1) / 4];
    #pragma unroll
    for (int i = 0; i < 4; i++) { Da[4 + i] = pd[4 + i + (4 + i) / 4]; Db[i] = pd[8 + i + (8 + i) / 4]; }
  }
#undef KCHUNK
#undef KHALF
#undef KLOOPS
#undef KSNAPQ
  // The snapshot is taken halfway through chunk t / 2, which leaves the triangle k = 4t + 1 + s (s < 4), h = 4t + r on the
  // boundary: still to come on an even lane, which adds its wrapped products (even column s >= r, odd s > r), and already in on
  // an odd lane, which subtracts its unwrapped ones (even s < r, odd s <= r). U[m] = s_{8t + 1 + m}, D[m] = s_{m - 4}.
  uint32_t my[8][3];
  #pragma unroll
  for (int j = 0; j < 8; j++) my[j][0] = my[j][1] = my[j][2] = 0;
  const bool lodd = lane & 1;
  {
    uint32_t U[7], D[8];
    #pragma unroll
    for (int m = 0; m < 7; m++) U[m] = KS(S, 8 * (int)lane + 1 + m);
    #pragma unroll
    for (int m = 0; m < 8; m++) D[m] = KS(S, m - 4);
    #pragma unroll
    for (int r = 0; r < 4; r++)
      #pragma unroll
      for (int s = 0; s < 4; s++) {
        if (lodd ? s < r : s >= r) mac(my[2 * r], U[r + s], D[3 + r - s]);
        if (lodd ? s <= r : s > r) mac(my[2 * r + 1], U[r + s], D[4 + r - s]);
      }
  }
  const uint32_t *sv = snap + lane * KSNAP_STRIDE;
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    const int h = h0 + j / 2;
    uint32_t y[3];
    ld2_ord(sv + 2 * j, y[0], y[1]);
    y[2] = ld1_ord(sv + 16 + j);
    if (lodd) sub96(y, my[j]);
    else add96(y, my[j]);
    if (j & 1) {
      const uint32_t dg = KS(S, h + 129);
      dbl_sq(T[j], acc[j], dg);
      dbl_sq(Y[j], y, dg);
    } else {
      dbl_sq(T[j], acc[j], KS(S, h));
      Y[j][0] = y[0] << 1; Y[j][1] = __funnelshift_l(y[0], y[1], 1); Y[j][2] = __funnelshift_l(y[1], y[2], 1);
    }
  }
  // column 256: (128 + k, 128 - k), this lane's k = 4t + 1 .. 4t + 4, all unwrapped
  uint32_t c[3] = {0, 0, 0};
  #pragma unroll
  for (int s = 0; s < 4; s++) mac(c, KS(S, 128 + h0 + 1 + s), KS(S, 127 - h0 - s));
  #pragma unroll
  for (int d = 1; d < 32; d <<= 1)
    add3(c, __shfl_xor_sync(0xFFFFFFFFu, c[0], d), __shfl_xor_sync(0xFFFFFFFFu, c[1], d), __shfl_xor_sync(0xFFFFFFFFu, c[2], d));
  dbl_sq(t256, c, KS(S, 128));
}

// 64-bit ballot lookahead over the ring of runs (run L of lane t at bit t, run H at bit 32 + t): gl, gh are the runs' carries
// out (0/1), ol, oh whether they are all ones; returns the carry out of the top (bit 63) and sets the carries in.
__device__ __forceinline__ uint32_t ring_carry(uint32_t gl, uint32_t gh, bool ol, bool oh, uint32_t &cl, uint32_t &ch,
                                               uint32_t lane) {
  const uint64_t G = (uint64_t)__ballot_sync(0xFFFFFFFFu, gl) | (uint64_t)__ballot_sync(0xFFFFFFFFu, gh) << 32;
  const uint64_t A = G | (uint64_t)__ballot_sync(0xFFFFFFFFu, ol) | (uint64_t)__ballot_sync(0xFFFFFFFFu, oh) << 32;
  const uint64_t S = A + G, C = S ^ A ^ G;
  cl = (uint32_t)(C >> lane) & 1;
  ch = (uint32_t)(C >> (32 + lane)) & 1;
  return S < A;
}
// P2's d(a) as sqr_d, by the Karatsuba step; x is a, and becomes d(a)
__device__ __forceinline__ void sqr_k(uint32_t *__restrict__ sm, Num &x, bool odd, uint32_t c0, uint32_t lane) {
  put_k(sm, x, lane);
  uint32_t *snap = sm + KSNAP;
  uint32_t lo[9][3], hi[9][3];   // columns 8t + j and 257 + 8t + j; [8]: 256 and 513 (lane 31's)
  #pragma unroll
  for (int j = 0; j < 9; j++) { lo[j][0] = hi[j][0] = 0; lo[j][1] = hi[j][1] = 0xF0000000u; lo[j][2] = hi[j][2] = 0x0FFFFFFFu; }
  if (lane == 0) { const uint32_t b[3] = {0, (c0 - 1) << 28, (c0 - 1) >> 4}; sub96(lo[0], b); }
  const uint32_t k = c0 - 1;
  {
    uint32_t T[8][3], Y[8][3], t[3];
    ksq(sm, snap, T, Y, t, lane);                                   // A0
    #pragma unroll
    for (int j = 0; j < 8; j++) { add96(lo[j], T[j]); mad96(lo[j], Y[j], k); add96(hi[j], T[j]); }
    add96(lo[8], t); add96(hi[8], t);
    ksq(sm + K_ARR, snap, T, Y, t, lane);                           // A1
    #pragma unroll
    for (int j = 0; j < 8; j++) { mad96(lo[j], T[j], c0); add96(hi[j], T[j]); mad96(hi[j], Y[j], k); }
    mad96(lo[8], t, c0); add96(hi[8], t);
    ksq(sm + 2 * K_ARR, snap, T, Y, t, lane);                       // D
    #pragma unroll
    for (int j = 0; j < 8; j++) { msb96(lo[j], Y[j], c0); add96(hi[j], Y[j]); sub96(hi[j], T[j]); }
    sub96(hi[8], t);
  }
  // each run's columns into limbs; carries (< 2^62) go to the next run, the top's times c0 to run L of lane 0
  uint32_t vl[9], vh[9], ll0 = 0, ll1 = 0, hl0 = 0, hl1 = 0;
  const int n = lane == 31 ? 9 : 8;
  #pragma unroll
  for (int j = 0; j < 9; j++) {
    if (j >= n) { vl[j] = vh[j] = 0; continue; }
    add3(lo[j], ll0, ll1, 0); vl[j] = lo[j][0]; ll0 = lo[j][1]; ll1 = lo[j][2];
    add3(hi[j], hl0, hl1, 0); vh[j] = hi[j][0]; hl0 = hi[j][1]; hl1 = hi[j][2];
  }
  uint32_t il0 = __shfl_up_sync(0xFFFFFFFFu, ll0, 1), il1 = __shfl_up_sync(0xFFFFFFFFu, ll1, 1), il2 = 0;
  uint32_t ih0 = __shfl_up_sync(0xFFFFFFFFu, hl0, 1), ih1 = __shfl_up_sync(0xFFFFFFFFu, hl1, 1);
  {
    const uint32_t a0 = __shfl_sync(0xFFFFFFFFu, ll0, 31), a1 = __shfl_sync(0xFFFFFFFFu, ll1, 31);
    const uint32_t b0 = __shfl_sync(0xFFFFFFFFu, hl0, 31), b1 = __shfl_sync(0xFFFFFFFFu, hl1, 31);
    if (lane == 0) {
      ih0 = a0; ih1 = a1;
      const uint64_t p0 = (uint64_t)b0 * c0, p1 = (uint64_t)b1 * c0 + (p0 >> 32);
      il0 = (uint32_t)p0; il1 = (uint32_t)p1; il2 = (uint32_t)(p1 >> 32);
    }
  }
  uint32_t gl, gh;
  {
    uint32_t t3[3] = {il0, il1, il2};
    uint64_t c = 0;
    #pragma unroll
    for (int m = 0; m < 9; m++) {
      if (m >= n) break;
      c += (uint64_t)vl[m] + (m < 3 ? t3[m] : 0);
      vl[m] = (uint32_t)c; c >>= 32;
    }
    gl = (uint32_t)c;
    c = 0;
    #pragma unroll
    for (int m = 0; m < 9; m++) {
      if (m >= n) break;
      c += (uint64_t)vh[m] + (m == 0 ? ih0 : m == 1 ? ih1 : 0);
      vh[m] = (uint32_t)c; c >>= 32;
    }
    gh = (uint32_t)c;
  }
  uint32_t cl, ch;
  uint32_t topc = ring_carry(gl, gh, run_ones(vl, lane), run_ones(vh, lane), cl, ch, lane);
  run_add(vl, cl, lane);
  run_add(vh, ch, lane);
  while (__any_sync(0xFFFFFFFFu, topc)) {   // the carry out of limb 513 has weight 2^W = c0
    gl = lane == 0 ? run_add(vl, (uint64_t)topc * c0, lane) : 0;
    topc = ring_carry(gl, 0, run_ones(vl, lane), run_ones(vh, lane), cl, ch, lane);
    run_add(vl, cl, lane);
    run_add(vh, ch, lane);
  }
  // into the Num layout through the staging area
  __syncwarp();
  #pragma unroll
  for (int m = 0; m < 8; m++) { sm[stage(8 * lane + m)] = vl[m]; sm[stage(257 + 8 * lane + m)] = vh[m]; }
  if (lane == 31) { sm[stage(256)] = vl[8]; sm[stage(513)] = vh[8]; }
  __syncwarp();
  #pragma unroll
  for (int q = 0; q < 16; q++) x.l[q] = sm[stage(16 * lane + q)];
  x.e0 = lane == 31 ? sm[stage(512)] : 0;
  x.e1 = lane == 31 ? sm[stage(513)] : 0;
  __syncwarp();
#undef KS
  finish(x, odd, c0, lane);
}

// rows of NL big-endian words <-> limbs: through the staging area, coalesced on the global side
__device__ __forceinline__ void load_row(uint32_t *__restrict__ st, const uint32_t *__restrict__ row, Num &x, uint32_t lane) {
  uint32_t v[17];                                              // all of a lane's loads in flight at once
  #pragma unroll
  for (int i = 0; i < 17; i++) v[i] = i < 16 || lane < N - 512 ? row[lane + 32 * i] : 0;
  __syncwarp();
  #pragma unroll
  for (int i = 0; i < 17; i++)
    if (i < 16 || lane < N - 512) st[stage(N - 1 - (int)lane - 32 * i)] = __byte_perm(v[i], 0, 0x0123);
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

// m = d(sigma_K(d(c))) - t mod p of one block, K and t as rows
__device__ __forceinline__ void decode_one(uint32_t *__restrict__ sm, uint32_t *__restrict__ snap,
                                           const uint32_t *__restrict__ crow, uint32_t *__restrict__ mrow,
                                           const uint32_t *__restrict__ krow, const uint32_t *__restrict__ trow,
                                           uint32_t c0, uint32_t *__restrict__ flags, uint32_t lane) {
  Num x, k;
  load_row(snap, crow, x, lane);
#ifdef P2SQ_KARA
  sqr_k(sm, x, odd_of(x), c0, lane);                           // x = d(c)
#else
  put_e(sm, x, lane);
  sqr_d(sm, snap, x, odd_of(x), c0, lane);                     // x = d(c)
#endif
  __syncwarp();
  load_row(snap, krow, k, lane);
  #pragma unroll
  for (int q = 0; q < 16; q++) k.l[q] ^= x.l[q];
  k.e0 ^= x.e0; k.e1 ^= x.e1;
  if (!ge_p(k, c0, lane)) x = k;                               // sigma_K
#ifdef P2SQ_KARA
  sqr_k(sm, x, odd_of(x), c0, lane);                           // x = d(sigma_K(d(c)))
#else
  put_e(sm, x, lane);
  sqr_d(sm, snap, x, odd_of(x), c0, lane);                     // x = d(sigma_K(d(c)))
#endif
  __syncwarp();
  load_row(snap, trow, k, lane);
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
  store_row(snap, mrow, x, lane);
}
// a row's 2,056 bytes lie in 17 lines of 128, one per lane
__device__ __forceinline__ void prefetch_row(const uint32_t *row, uint32_t lane) {
  asm volatile("prefetch.global.L2 [%0];" ::"l"((const char *)row + (lane < 17 ? 128 * lane : 128 * 16)));
}

}  // namespace p2sq

#ifndef P2SQ_WARPS
#define P2SQ_WARPS 2
#endif
#ifndef P2SQ_MINB
#define P2SQ_MINB 1
#endif

// m = d(sigma_K(d(c))) - t mod p, one block per warp, K and t as rows
__global__ void __launch_bounds__(32 * P2SQ_WARPS, P2SQ_MINB) decode_sq_kernel(const uint32_t *__restrict__ cin, uint32_t *__restrict__ mout,
                                                                    const uint32_t *__restrict__ kin, const uint32_t *__restrict__ tin,
                                                                    uint32_t n, uint32_t c0, uint32_t *__restrict__ flags) {
  using namespace p2sq;
  __shared__ __align__(16) uint32_t smem[P2SQ_WARPS][WARP_WORDS];
  const uint32_t lane = threadIdx.x % 32, wid = threadIdx.x / 32;
  uint32_t *sm = smem[wid], *snap = smem[wid] + E_WORDS;
  const uint32_t stride = gridDim.x * P2SQ_WARPS;
  for (uint32_t j = blockIdx.x * P2SQ_WARPS + wid; j < n; j += stride) {
#ifndef P2SQ_NO_PREFETCH
    const size_t jn = j + stride < n ? j + stride : j;   // K and t of this block, c of the next, into L2
    prefetch_row(kin + (size_t)j * N, lane);
    prefetch_row(tin + (size_t)j * N, lane);
    prefetch_row(cin + jn * N, lane);
#endif
    decode_one(sm, snap, cin + (size_t)j * N, mout + (size_t)j * N, kin + (size_t)j * N, tin + (size_t)j * N, c0, flags, lane);
  }
}

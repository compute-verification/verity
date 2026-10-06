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
    Num x, k;
#ifndef P2SQ_NO_PREFETCH
    {  // K and t of this block, c of the next, into L2: a row's 2,056 bytes lie in 17 lines of 128, one per lane
      const uint32_t at = lane < 17 ? 128 * lane : 128 * 16;
      const size_t jn = j + stride < n ? j + stride : j;
      asm volatile("prefetch.global.L2 [%0];" ::"l"((const char *)(kin + (size_t)j * N) + at));
      asm volatile("prefetch.global.L2 [%0];" ::"l"((const char *)(tin + (size_t)j * N) + at));
      asm volatile("prefetch.global.L2 [%0];" ::"l"((const char *)(cin + jn * N) + at));
    }
#endif
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

// The filler reader with receipts, `verity/kv-filler/fill-sk16/v0` (`fill_ref.py` is the reference): a small fixed
// grid of CTAs that streams registered blocks at a bounded number of bytes in flight, so that beside a running step it
// takes the DRAM time the step leaves and little of the step's own, and needs no shared memory, so that it co-resides
// with the step's GEMM and attention CTAs.
//
// A block is 4,096 bytes, read as int8. Its sketch is 16 int32 sums s_r = Σ_i σ(r, i)·x_i, σ(r, i) = 1 − 2·bit, the bit
// at position 4096 r + i of the 8 KiB sign string (LSB first within each byte). Its leaf is BLAKE3's compression of the
// 16 sums (little-endian) under the chunk's key K (its seed's 8 little-endian words) with counter j (the block's index in
// the plan), length 64 and flags KEYED | CHUNK_START | CHUNK_END. A unit is 32 consecutive blocks (128 KiB), its root the
// binary tree over its leaves: the parent of nodes covering leaves [a, a + 2^(l-1)) and [a + 2^(l-1), a + 2^l) is the
// compression of left ‖ right under K, counter (l << 56) | a, length 64, flags KEYED | PARENT. A warp takes a unit at a
// time from an atomic counter and writes its 32-byte root; an opening recomputes the unit from its blocks.
//
// The MMA: mma.m16n8k32.s8 with the signs as A (16 rows) and 8 blocks as B's 8 columns. Lane (g, c) = (lane / 4,
// lane % 4) loads 16 bytes of block g at offset 64·o + 16c, which are B's k = 4c..4c+3 and 16 + 4c..16 + 4c + 3 of two
// MMAs (bytes 0–7, then 8–15), and expands its own 16 sign bits of rows g and g + 8 at the same bytes into A. A warp
// runs four such groups of 8 blocks at once, and keeps DEPTH offsets of loads in flight.
//
// Included by topup.cu (mode 4) after `../codec_tensor/sha512.cuh` (v2's SHA-512) and readers.cu (its TMA copies); its C
// ABI is there.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace fl {

constexpr uint32_t BLK = 4096, UNIT_BLOCKS = 32, UNIT_BYTES = BLK * UNIT_BLOCKS, ROWS = 16;
constexpr uint32_t SIGN_WORDS = ROWS * BLK / 32;  // 2,048 u32
constexpr uint32_t F_START = 1, F_END = 2, F_PARENT = 4, F_KEYED = 16;
constexpr int DEPTH = 2;

struct Key {
  uint32_t k[8];
};

// Which CTAs of a resident fill kernel take units: CTA b does while b < (busy > 0 ? busy_ctas : idle_ctas), and the
// rest sleep, so the host (tu_gate) or the step's own stream (tu_gate_step: busy += ±1 at a step's start and end) moves
// the filler's grid within a unit's time instead of a chunk's. Which CTA takes a unit doesn't enter its root. Closed
// both ways, the kernel returns, its chunk cut to the units taken (a prefix: they are taken in order), so that nothing
// waiting on the device waits on a closed gate.
struct Gate {
  int busy;
  uint32_t busy_ctas, idle_ctas;
  int prefill;  // the steps on the GPU that prefill (the PREFILL role reads while it is > 0)
  // fill_seq's units taken by role; at each step's start (`gate_step`) STEP's and PREFILL's counts, and the steps (and
  // prefill steps) that ended with their role's count where it started: no unit read beside them
  unsigned long long units[4], mark[2];
  unsigned int bare[2];
};
constexpr uint32_t ALL = 0xFFFFFFFFu;

struct Regions {
  static constexpr int MAX = 80;
  const uint8_t* base[MAX];
  uint64_t first_unit[MAX + 1];  // prefix sums of each region's whole units
  int n;
};

#define FL_G(a, b, c, d, x, y) \
  a += b + (x);                \
  d = rotr(d ^ a, 16);         \
  c += d;                      \
  b = rotr(b ^ c, 12);         \
  a += b + (y);                \
  d = rotr(d ^ a, 8);          \
  c += d;                      \
  b = rotr(b ^ c, 7)

__host__ __device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
#ifdef __CUDA_ARCH__
  return __funnelshift_r(x, x, n);
#else
  return (x >> n) | (x << (32 - n));
#endif
}

// BLAKE3's compression with a 64-bit counter; `out` the first 8 output words.
__host__ __device__ __forceinline__ void compress(const uint32_t cv[8], const uint32_t msg[16], uint64_t counter,
                                                  uint32_t flags, uint32_t out[8]) {
  uint32_t m[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) m[i] = msg[i];
  uint32_t v0 = cv[0], v1 = cv[1], v2 = cv[2], v3 = cv[3], v4 = cv[4], v5 = cv[5], v6 = cv[6], v7 = cv[7];
  uint32_t v8 = 0x6A09E667u, v9 = 0xBB67AE85u, v10 = 0x3C6EF372u, v11 = 0xA54FF53Au;
  uint32_t v12 = (uint32_t)counter, v13 = (uint32_t)(counter >> 32), v14 = 64, v15 = flags;
#pragma unroll
  for (int r = 0; r < 7; ++r) {
    FL_G(v0, v4, v8, v12, m[0], m[1]);
    FL_G(v1, v5, v9, v13, m[2], m[3]);
    FL_G(v2, v6, v10, v14, m[4], m[5]);
    FL_G(v3, v7, v11, v15, m[6], m[7]);
    FL_G(v0, v5, v10, v15, m[8], m[9]);
    FL_G(v1, v6, v11, v12, m[10], m[11]);
    FL_G(v2, v7, v8, v13, m[12], m[13]);
    FL_G(v3, v4, v9, v14, m[14], m[15]);
    if (r < 6) {
      const uint32_t t[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13],
                              m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
#pragma unroll
      for (int i = 0; i < 16; ++i) m[i] = t[i];
    }
  }
  out[0] = v0 ^ v8, out[1] = v1 ^ v9, out[2] = v2 ^ v10, out[3] = v3 ^ v11;
  out[4] = v4 ^ v12, out[5] = v5 ^ v13, out[6] = v6 ^ v14, out[7] = v7 ^ v15;
}

#ifdef __CUDACC__
__device__ __forceinline__ uint4 ld_stream(const uint8_t* p, uint64_t pol) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p), "l"(pol));
  return v;
}

// 4 sign bits (bit e = byte e) as 4 int8 of ±1: 0x01 for 0, 0xFF for 1.
__device__ __forceinline__ uint32_t expand(uint32_t nib) {
  const uint32_t spread = (nib * 0x204081u) & 0x01010101u;
  return 0x01010101u + spread * 0xFEu;
}

__device__ __forceinline__ void mma(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A timing probe's switches (fill_seq_kernel and fill_seq_bulk_kernel's PROBE; every receipt is wrong under one):
// NO_HASH skips the SHA-512, NO_MMA the sketch's MMAs (the bytes are XOR-folded instead).
constexpr int NO_HASH = 1, NO_MMA = 2;

// One offset o (64 bytes of each block) of a group's 8 blocks: lane (g, c)'s 16 bytes `cur` of block g at 64 o + 16 c,
// its first 8 bytes' MMA into `acc` and its last 8 bytes' into `acc2` (the same, or a second chain summed later).
template <int PROBE = 0>
__device__ __forceinline__ void sketch_step(int (&acc)[4], int (&acc2)[4], const uint32_t* sg, const uint32_t* sg8,
                                            int sh, int o, uint4 cur) {
  if (PROBE & NO_MMA) {
    acc[0] ^= (int)(cur.x ^ cur.y ^ cur.z ^ cur.w);
    return;
  }
  // plain loads, not __ldg: v2 rewrites a slot of signs while kernels run (see `acquire_signs`)
  const uint32_t r0 = (sg[2 * o] >> sh) & 0xFFFFu, r8 = (sg8[2 * o] >> sh) & 0xFFFFu;
  const uint32_t a0 = expand(r0 & 15), a2 = expand((r0 >> 4) & 15), a1 = expand(r8 & 15), a3 = expand((r8 >> 4) & 15);
  const uint32_t e0 = expand((r0 >> 8) & 15), e2 = expand(r0 >> 12), e1 = expand((r8 >> 8) & 15), e3 = expand(r8 >> 12);
  mma(acc, a0, a1, a2, a3, cur.x, cur.y);
  mma(acc2, e0, e1, e2, e3, cur.z, cur.w);
}

// Lane n's block n = 8q + col: D[r][col] of group q is in lane 4 (r % 8) + col / 2, register 2 (r / 8) + col % 2.
__device__ __forceinline__ void gather(const int (&acc)[4][4], uint32_t (&m)[16]) {
  const int lane = threadIdx.x & 31, q_me = lane >> 3, col = lane & 7;
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int hi = 0; hi < 2; ++hi)
#pragma unroll
      for (int par = 0; par < 2; ++par) {
        const int val = acc[q][2 * hi + par];
#pragma unroll
        for (int r8 = 0; r8 < 8; ++r8) {
          const int got = __shfl_sync(0xffffffffu, val, 4 * r8 + (col >> 1));
          if (q == q_me && par == (col & 1)) m[8 * hi + r8] = (uint32_t)got;
        }
      }
}

// The first D steps' loads of the 32 blocks at `p` (lane (g, c)'s 16 bytes of block 8q + g at 64 d + 16 c).
template <int D = DEPTH>
__device__ __forceinline__ void preload(uint4 (&v)[D][4], const uint8_t* __restrict__ p, uint64_t pol) {
  const int lane = threadIdx.x & 31;
  const uint8_t* src = p + (uint64_t)(lane >> 2) * BLK + 16 * (lane & 3);
#pragma unroll
  for (int d = 0; d < D; ++d)
#pragma unroll
    for (int q = 0; q < 4; ++q) v[d][q] = ld_stream(src + (uint64_t)q * 8 * BLK + 64 * d, pol);
}

// The sketches of 32 blocks at `p` (block 8q + g of group q), its first D steps already in v (`preload`, D divides
// the 64 steps): lane n's block n, its 16 sums into acc's gather. Offset o's sign fragments depend on the lane and o
// only, so the four groups share them.
template <int PROBE = 0, int D = DEPTH>
__device__ __forceinline__ void sketch_loaded(uint4 (&v)[D][4], const uint8_t* __restrict__ p,
                                              const uint32_t* signs, uint64_t pol, int (&acc)[4][4]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[q][i] = 0;
  const uint8_t* src = p + (uint64_t)g * BLK + 16 * c;
  const uint32_t* sg = signs + g * (BLK / 32) + (c >> 1);
  const uint32_t* sg8 = sg + 8 * (BLK / 32);
  const int sh = (c & 1) * 16;
  constexpr int STEPS = BLK / 64;
  static_assert(STEPS % D == 0, "a slot's index must be static");
#pragma unroll D
  for (int o = 0; o < STEPS; ++o) {
    const int slot = o % D;
    uint4 cur[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) cur[q] = v[slot][q];
    if (o + D < STEPS) {
#pragma unroll
      for (int q = 0; q < 4; ++q) v[slot][q] = ld_stream(src + (uint64_t)q * 8 * BLK + 64 * (o + D), pol);
    }
    if (PROBE & NO_MMA) {
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[q][0] ^= (int)(cur[q].x ^ cur[q].y ^ cur[q].z ^ cur[q].w);
      continue;
    }
    const uint32_t r0 = (sg[2 * o] >> sh) & 0xFFFFu, r8 = (sg8[2 * o] >> sh) & 0xFFFFu;
    const uint32_t a0 = expand(r0 & 15), a2 = expand((r0 >> 4) & 15), a1 = expand(r8 & 15), a3 = expand((r8 >> 4) & 15);
    const uint32_t e0 = expand((r0 >> 8) & 15), e2 = expand(r0 >> 12), e1 = expand((r8 >> 8) & 15), e3 = expand(r8 >> 12);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      mma(acc[q], a0, a1, a2, a3, cur[q].x, cur[q].y);
      mma(acc[q], e0, e1, e2, e3, cur[q].z, cur[q].w);
    }
  }
}

// The sketches of 32 blocks at `p` (block 8q + g of group q): lane n's block n, its 16 sums into m.
template <int PROBE = 0>
__device__ __forceinline__ void sketch(const uint8_t* __restrict__ p, const uint32_t* signs, uint64_t pol,
                                       uint32_t (&m)[16]) {
  uint4 v[DEPTH][4];
  int acc[4][4];
  preload(v, p, pol);
  sketch_loaded<PROBE>(v, p, signs, pol, acc);
  gather(acc, m);
}

// One v0 unit: 32 blocks at `p`, its root into out[0..7] (lane 0, unless out is null); returns the root's first word
// (lane 0).
__device__ __forceinline__ uint32_t unit(const uint8_t* __restrict__ p, uint64_t j0, const uint32_t* signs,
                                     const Key& K, uint64_t pol, uint32_t* __restrict__ out) {
  const int lane = threadIdx.x & 31;
  uint32_t m[16];
  sketch(p, signs, pol, m);
  uint32_t node[8];
  compress(K.k, m, j0 + lane, F_KEYED | F_START | F_END, node);
#pragma unroll
  for (int l = 1; l <= 5; ++l) {
    const int half = 1 << (l - 1);
    uint32_t mm[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      mm[i] = node[i];
      mm[8 + i] = __shfl_down_sync(0xffffffffu, node[i], half);
    }
    uint32_t up[8];
    compress(K.k, mm, ((uint64_t)l << 56) | (j0 + lane), F_KEYED | F_PARENT, up);
    if ((lane & ((1 << l) - 1)) == 0)
#pragma unroll
      for (int i = 0; i < 8; ++i) node[i] = up[i];
  }
  if (lane == 0 && out) {
    uint4* o4 = reinterpret_cast<uint4*>(out);
    o4[0] = make_uint4(node[0], node[1], node[2], node[3]);
    o4[1] = make_uint4(node[4], node[5], node[6], node[7]);
  }
  return node[0];
}

// Units [unit0, unit0 + units) of the plan over `R`, W warps a CTA, taken a warp at a time from `ctr` (zeroed before the launch); unit
// u's root at roots[8 (u − unit0)], and the XOR of the roots' first words into *xr (the chunk's check word); the units
// done are min(*ctr, units).
template <int W>
__global__ void __launch_bounds__(W * 32) fill_kernel(const Regions R, uint64_t unit0, uint64_t units,
                                                           const uint32_t* __restrict__ signs, Key K,
                                                           uint32_t* __restrict__ roots, unsigned long long* ctr,
                                                           uint32_t* xr, const Gate* gate) {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  const int lane = threadIdx.x & 31;
  for (;;) {
    unsigned long long u = 0;
    if (lane == 0) {
      const volatile Gate* g = gate;
      for (;;) {
        const uint32_t busy_ctas = g->busy_ctas, idle_ctas = g->idle_ctas;
        if (blockIdx.x < (g->busy > 0 ? busy_ctas : idle_ctas)) {
          u = atomicAdd(ctr, 1ull);
          break;
        }
        if ((busy_ctas == 0 && idle_ctas == 0) || *(volatile unsigned long long*)ctr >= units) {
          u = units;
          break;
        }
        __nanosleep(2000);
      }
    }
    u = __shfl_sync(0xffffffffu, u, 0);
    if (u >= units) return;
    const uint64_t U = unit0 + u;
    int r = 0;
    while (r + 1 < R.n && R.first_unit[r + 1] <= U) ++r;
    const uint8_t* p = R.base[r] + (U - R.first_unit[r]) * UNIT_BYTES;
    uint32_t* o = roots + 8 * u;
    unit(p, U * UNIT_BLOCKS, signs, K, pol, o);
    if (lane == 0) atomicXor(xr, o[0]);
  }
}

// fill-sk16/v2 (`fill_ref.epoch`), sketch-then-SHA-512: every kernel of a run takes units from one sequence `next`;
// unit s is plan unit s mod P in pass p = s div P, and lands in epoch s >> shift of a ring of `epochs` (a power of 2).
// Pass p's signs are its own: string byte 64 i + b is byte b of SHA-512(TAG_SGN ‖ run seed ‖ u64be p ‖ u64be i),
// i < 128 (one compression each). Block j's (its plan index) word is the first 4 bytes, big-endian, of
// SHA-512(TAG_BLK ‖ its 16 sums as i32be ‖ u64be j ‖ u64be p), 96 bytes and one compression; a unit's word is the XOR of
// its blocks', XORed into xr[e], cnt[e] counted, and the unit's completion stamps t0[e] (min) and t1[e] (max) with the
// GPU's globaltimer. A kernel takes at most `budget` units (its own counter `taken`), so nothing waiting on the device
// waits long, and returns as soon as its role's condition fails:
//   BASE  while the gate isn't closed both ways
//   STEP  while a step is open (gate busy > 0): launched beside a step, it ends with it
//   IDLE  while no step is open and blockIdx < idle_ctas: launched by the thread, it gives way to the next step
enum Role : int { BASE = 0, STEP = 1, IDLE = 2, PREFILL = 3 };
constexpr unsigned long long STEP_LINGER_NS = 50000;  // how long a STEP warp waits for the next step

// `limit` bounds `next` so the units taken never lap the ring before the host has logged their epoch; a CTA checks it
// before taking one unit, so `next` passes it by at most the CTAs in flight.
struct Seq {
  unsigned long long next, stop, limit;
};

struct Ring {
  uint32_t* xr;
  uint32_t* cnt;
  unsigned long long *t0, *t1;  // the globaltimer at the epoch's first and last unit done (min and max)
  uint32_t shift, mask;  // epoch = s >> shift, slot = epoch & mask
};

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

// fill-sk16/v2's domain tags, as SHA-512's big-endian words: "fill-sk16/v2/blk" and "fill-sk16/v2/sgn".
constexpr uint64_t TAG_HI = 0x66696c6c2d736b31ull, TAG_BLK = 0x362f76322f626c6bull, TAG_SGN = 0x362f76322f73676eull;

// The signs of the passes in flight: pass p's in slot p mod SIGN_SLOTS once pass[slot] == p. The warp that takes a
// pass's first unit derives the next pass's, after every unit of the pass that held its slot is done (done[slot] == P):
// those units were taken before it, so the wait ends, and no unit reads signs that are being replaced.
constexpr uint32_t SIGN_SLOTS = 4;
struct Signs {
  uint32_t* words;  // SIGN_SLOTS × SIGN_WORDS, each a sign string's bytes as little-endian words
  unsigned long long* pass;  // which pass each slot holds (~0: none)
  unsigned long long* done;  // the units done of the pass each slot holds
  uint64_t seed[4];  // the run seed as big-endian words
};

__device__ __forceinline__ uint32_t bswap32(uint32_t x) { return __byte_perm(x, 0, 0x0123); }

// Pass q's signs into its slot, by one warp (every lane).
__device__ void derive_signs(const Signs& sg, uint64_t q, uint64_t P) {
  const int lane = threadIdx.x & 31;
  const uint32_t slot = (uint32_t)(q % SIGN_SLOTS);
  if (lane == 0) {
    if (q >= SIGN_SLOTS)
      while (((volatile unsigned long long*)sg.done)[slot] != P) __nanosleep(500);
    sg.done[slot] = 0;
  }
  __syncwarp();
  uint32_t* out = sg.words + (uint64_t)slot * SIGN_WORDS;
#pragma unroll 1
  for (uint32_t i = lane; i < 128; i += 32) {
    uint64_t w[16] = {TAG_HI, TAG_SGN, sg.seed[0], sg.seed[1], sg.seed[2], sg.seed[3], q, i, 1ull << 63, 0, 0, 0, 0, 0, 0,
                      512};
    uint64_t s[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) s[k] = sx::IV_DEV[k];
    sha512_compress(s, w);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      out[16 * i + 2 * k] = bswap32((uint32_t)(s[k] >> 32));
      out[16 * i + 2 * k + 1] = bswap32((uint32_t)s[k]);
    }
  }
  __threadfence();
  __syncwarp();
  if (lane == 0) {
    ((volatile unsigned long long*)sg.pass)[slot] = q;
    __threadfence();
  }
}

// The v2 word of the unit whose lane n holds block j0 + n's sums in pass `pass` (every lane).
template <int PROBE = 0>
__device__ __forceinline__ uint32_t words_v2(const uint32_t (&m)[16], uint64_t j0, uint64_t pass) {
  const int lane = threadIdx.x & 31;
  if (PROBE & NO_HASH) return m[0] ^ m[15];
  uint64_t w[16];
  w[0] = TAG_HI, w[1] = TAG_BLK;
#pragma unroll
  for (int k = 0; k < 8; ++k) w[2 + k] = ((uint64_t)m[2 * k] << 32) | m[2 * k + 1];
  w[10] = j0 + lane, w[11] = pass, w[12] = 1ull << 63, w[13] = 0, w[14] = 0, w[15] = 768;
  uint64_t s[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) s[k] = sx::IV_DEV[k];
  sha512_compress(s, w);
  uint32_t x = (uint32_t)(s[0] >> 32);
#pragma unroll
  for (int o = 16; o; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}

template <int PROBE = 0, int D = DEPTH>
__device__ __forceinline__ uint32_t unit_v2(const uint8_t* __restrict__ p, uint64_t j0, uint64_t pass,
                                        const uint32_t* signs, uint64_t pol) {
  uint4 v[D][4];
  int acc[4][4];
  uint32_t m[16];
  preload<D>(v, p, pol);
  sketch_loaded<PROBE, D>(v, p, signs, pol, acc);
  gather(acc, m);
  return words_v2<PROBE>(m, j0, pass);
}

constexpr unsigned long long NONE = ~0ull;

// The next unit of the sequence for this warp's role, or NONE once the role's condition fails (every lane).
__device__ __forceinline__ unsigned long long take(const Gate* gate, Seq* seq, int role, unsigned long long* taken,
                                                   unsigned long long budget) {
  unsigned long long s = NONE;
  if ((threadIdx.x & 31) == 0) {
    const volatile Gate* g = gate;
    const volatile Seq* q = seq;
    const int busy = g->busy;
    const uint32_t busy_ctas = g->busy_ctas, idle_ctas = g->idle_ctas;
    const bool open = !q->stop && q->next < q->limit;
    const volatile int* count = role == PREFILL ? &g->prefill : &g->busy;
    bool go = open && (role == BASE                     ? !(busy_ctas == 0 && idle_ctas == 0)
                       : role == STEP || role == PREFILL ? *count > 0
                                                        : busy <= 0 && blockIdx.x < idle_ctas);
    if (open && !go && (role == STEP || role == PREFILL)) {
      // between pipelined steps the count is 0 for microseconds: a warp returning there would leave the kernel to
      // dwindle, never relaunched while its last warp reads on
      const unsigned long long until = globaltimer() + STEP_LINGER_NS;
      while (!(go = *count > 0) && !q->stop && globaltimer() < until) __nanosleep(1000);
    }
    if (go && atomicAdd(taken, 1ull) < budget) {
      s = atomicAdd(&seq->next, 1ull);
      atomicAdd(const_cast<unsigned long long*>(&gate->units[role]), 1ull);
    }
  }
  return __shfl_sync(0xffffffffu, s, 0);
}

// Waits for pass `pass`'s signs (every lane), unless this warp already has (`*have`). Each lane's acquire of the slot's
// tag orders its later plain loads of the signs after their derivation; a slot holds one pass until every unit of it
// is done, so one acquire a pass suffices.
__device__ __forceinline__ void acquire_signs(const Signs& sg, uint64_t pass, unsigned long long* have) {
  if (pass == *have) return;
  const uint32_t slot = (uint32_t)(pass % SIGN_SLOTS);
  if ((threadIdx.x & 31) == 0)
    while (((volatile unsigned long long*)sg.pass)[slot] != pass) __nanosleep(500);
  __syncwarp();
  unsigned long long t;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(t) : "l"(sg.pass + slot) : "memory");
  *have = pass;
}

__device__ __forceinline__ const uint8_t* unit_at(const Regions& R, uint64_t U) {
  int r = 0;
  while (r + 1 < R.n && R.first_unit[r + 1] <= U) ++r;
  return R.base[r] + (U - R.first_unit[r]) * UNIT_BYTES;
}

// Unit s's word into its epoch, and, if it is its pass's first, the next pass's signs.
__device__ __forceinline__ void finish(const Signs& sg, const Ring& ring, unsigned long long s, uint64_t P, uint32_t w) {
  const uint64_t U = s % P, pass = s / P;
  if ((threadIdx.x & 31) == 0) {
    const uint32_t e = (uint32_t)(s >> ring.shift) & ring.mask;
    const unsigned long long now = globaltimer();
    atomicMin(ring.t0 + e, now);
    atomicMax(ring.t1 + e, now);
    atomicXor(ring.xr + e, w);
    __threadfence();
    atomicAdd(ring.cnt + e, 1u);
    atomicAdd(sg.done + pass % SIGN_SLOTS, 1ull);
  }
  if (U == 0) derive_signs(sg, pass + 1, P);
}

// D: the offsets of loads in flight a warp (`sketch_loaded`)
template <int W, int PROBE = 0, int D = DEPTH>
__global__ void __launch_bounds__(W * 32) fill_seq_kernel(const Regions R, Signs sg, const Gate* gate, Seq* seq,
                                                          Ring ring, int role, unsigned long long* taken,
                                                          unsigned long long budget) {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  const uint64_t P = R.first_unit[R.n];
  unsigned long long have = NONE;
  for (;;) {
    const unsigned long long s = take(gate, seq, role, taken, budget);
    if (s == NONE) return;
    const uint64_t U = s % P, pass = s / P;
    acquire_signs(sg, pass, &have);
    const uint32_t* signs = sg.words + (pass % SIGN_SLOTS) * SIGN_WORDS;
    finish(sg, ring, s, P, unit_v2<PROBE, D>(unit_at(R, U), U * UNIT_BLOCKS, pass, signs, pol));
  }
}

// The same sequence read through the TMA (the PREFILL role's reader): one warp a CTA, each unit as 8 pieces, piece k
// the half k % 2 of group k / 2's 8 blocks, copied by cp.async.bulk into S stages of shared memory (each half-block at
// a stride of 2,112 bytes, 64 mod 128, so the sketch's 16-byte loads hit distinct banks) and sketched from there. The
// warp takes its next unit before it hashes this one, so copies stay in flight through the hash.
constexpr uint32_t TB_HALF = BLK / 2, TB_STRIDE = TB_HALF + 64, TB_STAGE = 8 * TB_STRIDE;

__device__ __forceinline__ void issue_piece(uint8_t* stage, uint64_t* bar, const uint8_t* p, int k, uint64_t pol) {
  const int lane = threadIdx.x & 31;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // the stage's generic reads before the async writes
  if (lane == 0) rd::mbar_expect(bar, 8 * TB_HALF);
  __syncwarp();
  if (lane < 8)
    rd::bulk_g2s(stage + lane * TB_STRIDE, p + (uint64_t)(8 * (k >> 1) + lane) * BLK + (k & 1) * TB_HALF, TB_HALF, bar,
                 pol);
}

template <int S, int PROBE = 0>
__global__ void __launch_bounds__(32) fill_seq_bulk_kernel(const Regions R, Signs sg, const Gate* gate, Seq* seq,
                                                           Ring ring, int role, unsigned long long* taken,
                                                           unsigned long long budget) {
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ uint64_t bar[S];
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3, sh = (c & 1) * 16;
  const uint64_t P = R.first_unit[R.n];
  if (lane == 0) {
    for (int k = 0; k < S; ++k) rd::mbar_init(bar + k, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncwarp();
  unsigned long long have = NONE, s = take(gate, seq, role, taken, budget);
  if (s == NONE) return;
  const uint8_t* p = unit_at(R, s % P);
  uint32_t issued = 0, used = 0;
#pragma unroll
  for (int k = 0; k < S; ++k, ++issued) issue_piece(sm + (issued % S) * TB_STAGE, bar + issued % S, p, k, pol);
  for (;;) {
    const uint64_t U = s % P, pass = s / P;
    acquire_signs(sg, pass, &have);
    const uint32_t* sgp = sg.words + (pass % SIGN_SLOTS) * SIGN_WORDS + g * (BLK / 32) + (c >> 1);
    const uint32_t* sgp8 = sgp + 8 * (BLK / 32);
    int acc[4][4];
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[q][i] = 0;
    unsigned long long nx = NONE;
    const uint8_t* np = nullptr;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const uint32_t st = used % S;
      rd::mbar_wait(bar + st, (used / S) & 1);
      const uint8_t* b = sm + st * TB_STAGE + g * TB_STRIDE + 16 * c;
      // a piece is one group's: four chains of MMAs instead of one, summed after it (int32 sums are exact)
      int part[4][4] = {};
#pragma unroll 4
      for (int o = 0; o < 32; o += 2) {
        sketch_step<PROBE>(part[0], part[1], sgp, sgp8, sh, 32 * (k & 1) + o, *reinterpret_cast<const uint4*>(b + 64 * o));
        sketch_step<PROBE>(part[2], part[3], sgp, sgp8, sh, 32 * (k & 1) + o + 1,
                           *reinterpret_cast<const uint4*>(b + 64 * (o + 1)));
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[k >> 1][i] += part[0][i] + part[1][i] + part[2][i] + part[3][i];
      ++used;
      __syncwarp();
      if (k + S < 8) {
        issue_piece(sm + (issued % S) * TB_STAGE, bar + issued % S, p, k + S, pol);
        ++issued;
      } else {
        if (k + S == 8) {
          nx = take(gate, seq, role, taken, budget);
          if (nx != NONE) np = unit_at(R, nx % P);
        }
        if (nx != NONE) {
          issue_piece(sm + (issued % S) * TB_STAGE, bar + issued % S, np, k + S - 8, pol);
          ++issued;
        }
      }
    }
    uint32_t m[16];
    gather(acc, m);
    finish(sg, ring, s, P, words_v2<PROBE>(m, U * UNIT_BLOCKS, pass));
    if (nx == NONE) return;
    s = nx;
    p = np;
  }
}

// Pass 0's signs, before the first kernel of a run (one warp).
__global__ void signs_first(Signs sg) { derive_signs(sg, 0, 0); }

__global__ void stamp(unsigned long long* t) { *t = globaltimer(); }

__global__ void gate_set(Gate* g, uint32_t busy_ctas, uint32_t idle_ctas, int reset) {
  if (reset) {
    g->busy = g->prefill = 0;
    for (int r = 0; r < 4; ++r) g->units[r] = 0;
    g->mark[0] = g->mark[1] = 0;
    g->bare[0] = g->bare[1] = 0;
  }
  g->busy_ctas = busy_ctas;
  g->idle_ctas = idle_ctas;
  __threadfence();
}

__global__ void gate_step(Gate* g, int delta, int prefill = 0) {
  const volatile unsigned long long* u = g->units;
  if (delta < 0) {
    if (u[STEP] == g->mark[0]) ++g->bare[0];
    if (prefill && u[PREFILL] == g->mark[1]) ++g->bare[1];
  }
  atomicAdd(&g->busy, delta);
  if (prefill) atomicAdd(&g->prefill, delta);
  if (delta > 0) {
    g->mark[0] = u[STEP];
    if (prefill) g->mark[1] = u[PREFILL];
  }
  __threadfence();
}
#endif

}  // namespace fl

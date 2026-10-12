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
// Included by topup.cu (mode 4); its C ABI is there.
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
  uint32_t busy_ctas, idle_ctas, pad;
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

// One unit: 32 blocks at `p` (block 8q + g of group q), its root into out[0..7] (lane 0, unless out is null); returns
// the root's first word (lane 0).
__device__ __forceinline__ uint32_t unit(const uint8_t* __restrict__ p, uint64_t j0, const uint32_t* __restrict__ signs,
                                     const Key& K, uint64_t pol, uint32_t* __restrict__ out) {
  const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  int acc[4][4];
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[q][i] = 0;
  const uint8_t* src = p + (uint64_t)g * BLK + 16 * c;
  const uint32_t* sg = signs + g * (BLK / 32) + (c >> 1);
  const uint32_t* sg8 = sg + 8 * (BLK / 32);
  const int sh = (c & 1) * 16;
  constexpr int STEPS = BLK / 64;
  uint4 v[DEPTH][4];
#pragma unroll
  for (int d = 0; d < DEPTH; ++d)
#pragma unroll
    for (int q = 0; q < 4; ++q) v[d][q] = ld_stream(src + (uint64_t)q * 8 * BLK + 64 * d, pol);
#pragma unroll 2
  for (int o = 0; o < STEPS; ++o) {
    const int slot = o % DEPTH;
    uint4 cur[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) cur[q] = v[slot][q];
    if (o + DEPTH < STEPS) {
#pragma unroll
      for (int q = 0; q < 4; ++q) v[slot][q] = ld_stream(src + (uint64_t)q * 8 * BLK + 64 * (o + DEPTH), pol);
    }
    const uint32_t r0 = (__ldg(sg + 2 * o) >> sh) & 0xFFFFu, r8 = (__ldg(sg8 + 2 * o) >> sh) & 0xFFFFu;
    const uint32_t a0 = expand(r0 & 15), a2 = expand((r0 >> 4) & 15), a1 = expand(r8 & 15), a3 = expand((r8 >> 4) & 15);
    const uint32_t e0 = expand((r0 >> 8) & 15), e2 = expand(r0 >> 12), e1 = expand((r8 >> 8) & 15), e3 = expand(r8 >> 12);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      mma(acc[q], a0, a1, a2, a3, cur[q].x, cur[q].y);
      mma(acc[q], e0, e1, e2, e3, cur[q].z, cur[q].w);
    }
  }
  // Lane n hashes block n = 8q + col: D[r][col] of group q is in lane 4 (r % 8) + col / 2, register 2 (r / 8) + col % 2.
  const int q_me = lane >> 3, col = lane & 7;
  uint32_t m[16];
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

// fill-sk16/v1 (`fill_ref.epoch`): every kernel of a run takes units from one sequence `next`; unit s is plan unit
// s mod P under pass s div P's key (K with its first two words XORed with the pass), and lands in epoch s >> shift of a
// ring of `epochs` (a power of 2): its root's first word XORed into xr[e], cnt[e] counted, and the epoch's last unit
// stamps t[e] with the GPU's globaltimer. A kernel takes at most `budget` units (its own counter `taken`), so nothing
// waiting on the device waits long, and returns as soon as its role's condition fails:
//   BASE  while the gate isn't closed both ways
//   STEP  while a step is open (gate busy > 0): launched beside a step, it ends with it
//   IDLE  while no step is open and blockIdx < idle_ctas: launched by the thread, it gives way to the next step
enum Role : int { BASE = 0, STEP = 1, IDLE = 2 };
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

template <int W>
__global__ void __launch_bounds__(W * 32) fill_seq_kernel(const Regions R, const uint32_t* __restrict__ signs, Key K,
                                                          const Gate* gate, Seq* seq, Ring ring, int role,
                                                          unsigned long long* taken, unsigned long long budget) {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  const int lane = threadIdx.x & 31;
  const uint64_t P = R.first_unit[R.n];
  const unsigned long long NONE = ~0ull;
  for (;;) {
    unsigned long long s = NONE;
    if (lane == 0) {
      const volatile Gate* g = gate;
      const volatile Seq* q = seq;
      const int busy = g->busy;
      const uint32_t busy_ctas = g->busy_ctas, idle_ctas = g->idle_ctas;
      const bool open = !q->stop && q->next < q->limit;
      bool go = open && (role == BASE   ? !(busy_ctas == 0 && idle_ctas == 0)
                         : role == STEP ? busy > 0
                                        : busy <= 0 && blockIdx.x < idle_ctas);
      if (open && !go && role == STEP) {
        // between pipelined steps the count is 0 for microseconds: a STEP warp returning there would leave the
        // kernel to dwindle, never relaunched while its last warp reads on
        const unsigned long long until = globaltimer() + STEP_LINGER_NS;
        while (!(go = g->busy > 0) && !q->stop && globaltimer() < until) __nanosleep(1000);
      }
      if (go && atomicAdd(taken, 1ull) < budget) s = atomicAdd(&seq->next, 1ull);
    }
    s = __shfl_sync(0xffffffffu, s, 0);
    if (s == NONE) return;
    const uint64_t U = s % P, pass = s / P;
    Key Kp = K;
    Kp.k[0] ^= (uint32_t)pass;
    Kp.k[1] ^= (uint32_t)(pass >> 32);
    int r = 0;
    while (r + 1 < R.n && R.first_unit[r + 1] <= U) ++r;
    const uint8_t* p = R.base[r] + (U - R.first_unit[r]) * UNIT_BYTES;
    const uint32_t w = unit(p, U * UNIT_BLOCKS, signs, Kp, pol, nullptr);
    if (lane == 0) {
      const uint32_t e = (uint32_t)(s >> ring.shift) & ring.mask;
      const unsigned long long now = globaltimer();
      atomicMin(ring.t0 + e, now);
      atomicMax(ring.t1 + e, now);
      atomicXor(ring.xr + e, w);
      __threadfence();
      atomicAdd(ring.cnt + e, 1u);
    }
  }
}

__global__ void stamp(unsigned long long* t) { *t = globaltimer(); }

__global__ void gate_set(Gate* g, uint32_t busy_ctas, uint32_t idle_ctas, int reset) {
  if (reset) g->busy = 0;
  g->busy_ctas = busy_ctas;
  g->idle_ctas = idle_ctas;
  __threadfence();
}

__global__ void gate_step(Gate* g, int delta) {
  atomicAdd(&g->busy, delta);
  __threadfence();
}
#endif

}  // namespace fl

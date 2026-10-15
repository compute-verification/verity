// Modular squaring at p = 2^1024 - 105 (the remote bandwidth audit's default codec, p1024) with the limb products on
// the int8 tensor cores of the RTX PRO 6000 (sm_120), beside `benchmarks/word_codec/kv.cu`'s CUDA-core squaring, which
// this file includes so the reference and its rate chain are kv.cu's own code in the same binary (build with
// -DKV_FIELD=1024, and -DKV_FOLD6=1 for ra-squaring-ceiling's FOLD 6 form).
//
// The tensor-core squaring is `benchmarks/codec_tensor/wt.cu`'s plan (`wide.tc_plan`) at U = 4, a warp a word: lane l
// holds limb l (bytes 4l .. 4l + 3), the square's 255 byte coefficients are 10 mma.m16n8k32.u8.u8.s32
// (A_w[r][v] = a_(32w + r - v), B[v][s] = a_(32(s - w) + v), w = 0 .. 4, rows r in two halves), they go through shared
// memory to the owner layout, are folded at byte 128 and carried across lanes. wt.cu's tensor path covers only fields with
// n = 8S - 1, where 2^(8S) = 2c; at the full-slot p1024, 2^(8S) = 2^1024 = c, so the fold weight is c = 105, the lazy value
// is below 2^1024 as kv.cu's is, and canonical is that value less p when it is at least p.
//
// Kernels (extern "C" launchers at the end): the gate (one squaring of each of n words, lazy and canonical out); the
// timed chain of W words a warp in three modes (0 the squaring; 1 the same with each MMA replaced by an ALU op on its
// operands, the CUDA-core path alone; 2 the operands and MMAs with their coefficients stored, no fold or carries);
// independent MMAs on register operands (the tensor pipe's int8 rate at this shape); and a time-boxed CTA of 8 warps,
// cw of them squaring on the CUDA cores with kv.cu's sqr (a word a thread) and the rest on the tensor cores.
#ifndef KV_FIELD
#define KV_FIELD 1024
#endif
#include "../word_codec/kv.cu"

namespace mc {

using u32 = uint32_t;
using u64 = uint64_t;

constexpr int NL = 32, S = 128, U = 4, DB = 32;
constexpr u32 CF = 105;
constexpr u32 FW = CF;                                   // 2^(8S) = 2^1024 = c (mod p)
constexpr u32 FULL = 0xFFFFFFFFu;
constexpr u64 MASK = 0xFFFFFFFFull;
constexpr int SHUF_ROUNDS = 4;
static_assert(kv::NBITS == 1024 && kv::CF == CF && kv::NL == NL, "built at KV_FIELD = 1024");

__device__ __forceinline__ void mma_u8(int32_t (&d)[4], const u32 (&a)[4], u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The reversed bytes G[x] = a_(GM - x), zero outside the word, in four copies G_c[y] = G[y + c] so a lane's A register
// is one aligned load; B registers come from G_0 byte-swapped.
constexpr int GM = 32 * U + 35;
constexpr int GLEN = 32 * U + 68;
constexpr int GSTRIDE = ((GLEN + 31) / 32) * 32 + 16;
constexpr int CSTRIDE = 36;                              // coefficient slots a column (conflict-free C stores)
constexpr int CO = CSTRIDE * 8;
constexpr int WORD_BYTES = 4 * GSTRIDE + 4 * CO;
static_assert(GM % 4 == 3 && GSTRIDE % 16 == 0, "B's loads and the copies are aligned");

struct Scratch {
  uint8_t *g;
  int32_t *co;
};

__device__ __forceinline__ Scratch scratch_at(uint8_t *base) {
  Scratch s;
  s.g = base;
  s.co = reinterpret_cast<int32_t *>(base + 4 * GSTRIDE);
  return s;
}

__device__ __forceinline__ void init(Scratch s, int lane) {
  u32 *g = reinterpret_cast<u32 *>(s.g);
  for (int e = lane; e < GSTRIDE; e += 32) g[e] = 0;
  for (int e = lane; e < CO; e += 32) s.co[e] = 0;
  __syncwarp();
}

// Lane l's limb (bytes 4l .. 4l + 3) into the copies: byte i goes to G[GM - i], copy c at GM - i - c.
__device__ __forceinline__ void put(Scratch s, u64 v, int lane) {
#pragma unroll
  for (int b = 0; b < U; ++b) {
    const int i = U * lane + b;
    const uint8_t byte = (uint8_t)(v >> (8 * b));
#pragma unroll
    for (int c = 0; c < 4; ++c) s.g[c * GSTRIDE + GM - i - c] = byte;
  }
  __syncwarp();
}

// The square's byte coefficients c_m (m = 32 col + r at slot CSTRIDE col + r) into s.co. MMA false replaces each MMA
// by an ALU op on its operands (mode 1).
template <bool MMA>
__device__ __forceinline__ void square_coeffs(Scratch s, int lane) {
  const int g = lane >> 2, q = lane & 3;
  const int sig = (GM - g) & 3;
  const u32 *ga = reinterpret_cast<const u32 *>(s.g + sig * GSTRIDE);
  const u32 *g0 = reinterpret_cast<const u32 *>(s.g);
  // A_w[h] reg e = a_base .. a_(base - 3), base = 16 j + g + 8 (e & 1) - 4q, j = 2w + h - (e >> 1) in [-1, 2U + 1]
  u32 areg[2 * U + 3][2];
#pragma unroll
  for (int j = -1; j <= 2 * U + 1; ++j)
#pragma unroll
    for (int par = 0; par < 2; ++par) {
      const int base = 16 * j + g + 8 * par - 4 * q;
      areg[j + 1][par] = ga[(GM - base - sig) >> 2];
    }
  // B for w in [0, U]: column s = g, k = 4q + b (reg 0), 16 + 4q + b (reg 1); a_x .. a_(x+3) is the byte swap of G_0's
  // aligned word at GM - x - 3.
  u32 breg[U + 1][2];
#pragma unroll
  for (int dd = 0; dd <= U; ++dd) {
    const int x0 = 32 * (g - dd) + 4 * q;
    const int y0 = GM - x0 - 3, y1 = GM - x0 - 19;
    breg[dd][0] = (y0 >= 0 && y0 < GLEN) ? __byte_perm(g0[y0 >> 2], 0, 0x0123) : 0u;
    breg[dd][1] = (y1 >= 0 && y1 < GLEN) ? __byte_perm(g0[y1 >> 2], 0, 0x0123) : 0u;
  }
  int32_t acc[2][4];
#pragma unroll
  for (int h = 0; h < 2; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0;
#pragma unroll
  for (int w = 0; w <= U; ++w) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      u32 a[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) a[e] = areg[2 * w + h - (e >> 1) + 1][e & 1];
      if constexpr (MMA) {
        mma_u8(acc[h], a, breg[w][0], breg[w][1]);
      } else {
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[h][e] += (int32_t)(a[e] ^ breg[w][e & 1]);
      }
    }
  }
  // C fragment: reg e at row g + 8 (e >> 1) (+ 16 h), column 2q + (e & 1).
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int col = 2 * q + (e & 1), r = 16 * h + g + 8 * (e >> 1);
      s.co[CSTRIDE * col + r] = acc[h][e];
    }
  __syncwarp();
}

__device__ __forceinline__ int64_t coeff(Scratch s, int m) {   // c_m, 0 <= m < 256
  return s.co[CSTRIDE * (m >> 5) + (m & 31)];
}

// Digits < 2^(DB + 1) to digits < 2^DB by a carry-lookahead over two ballots; `out` the carry out of lane 31.
__device__ __forceinline__ u64 lookahead(u64 v, int lane, u32 &out) {
  const u32 gen = __ballot_sync(FULL, v > MASK);
  v &= MASK;
  const u32 prop = __ballot_sync(FULL, v == MASK);
  const u64 a = (u64)(gen | prop);
  const u64 sm = a + gen;
  const u32 cin = (u32)((sm ^ a ^ gen) >> lane) & 1u;
  out = (u32)(sm >> 32);
  return (v + cin) & MASK;
}

__device__ __forceinline__ u64 add_low(u64 v, int lane, u32 k) { return lane == 0 ? v + k : v; }

// The square mod p, lazily: the coefficients folded at byte 128 (x c), carried; limbs < 2^32 (value < 2^1024),
// congruent to a^2. Any a < 2^1024.
template <bool MMA>
__device__ __forceinline__ u64 square(Scratch s, u64 v, int lane) {
  put(s, v, lane);
  square_coeffs<MMA>(s, lane);
  u64 acc = 0;
#pragma unroll
  for (int b = U - 1; b >= 0; --b) {
    const int k = U * lane + b;
    const int64_t d = coeff(s, k) + (int64_t)FW * coeff(s, k + S);
    acc = (acc << 8) + (u64)d;
  }
#pragma unroll
  for (int round = 0; round < SHUF_ROUNDS; ++round) {
    const u64 c = acc >> DB;
    acc &= MASK;
    u64 in = __shfl_up_sync(FULL, c, 1);
    const u64 top = __shfl_sync(FULL, c, 31);
    if (lane == 0) in = top * FW;
    acc += in;
  }
  u32 out;
  acc = lookahead(acc, lane, out);
  if (out) {
    acc = lookahead(add_low(acc, lane, FW), lane, out);
    if (out) acc = lookahead(add_low(acc, lane, FW), lane, out);
  }
  return acc;
}

// v < 2^1024 to [0, p): v - p = v + c - 2^1024 when that carries out of 2^1024.
__device__ __forceinline__ u64 canon(u64 v, int lane) {
  u32 out;
  const u64 w = lookahead(add_low(v, lane, CF), lane, out);
  return out ? w : v;
}

// The gate: one squaring of each word (NL limbs, any value below 2^1024), the lazy result and the canonical one.
__global__ void __launch_bounds__(32) gate_kernel(const u32 *__restrict__ in, u32 *__restrict__ lazy,
                                                  u32 *__restrict__ canonical, int n) {
  __shared__ __align__(16) uint8_t buf[WORD_BYTES];
  const int lane = threadIdx.x;
  Scratch s = scratch_at(buf);
  init(s, lane);
  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    const u64 v = in[(u64)i * NL + lane];
    const u64 y = square<true>(s, v, lane);
    lazy[(u64)i * NL + lane] = (u32)y;
    canonical[(u64)i * NL + lane] = (u32)canon(y, lane);
  }
}

constexpr int WARPS = 4;

__device__ __forceinline__ u64 seed_limb(u64 id, int lane) {
  u64 z = id * 0x9E3779B97F4A7C15ull + lane * 0xBF58476D1CE4E5B9ull;
  z ^= z >> 31;
  z *= 0x94D049BB133111EBull;
  z ^= z >> 29;
  return z & MASK;
}

// W independent words a warp, each squared `steps` times; mode 0 the squaring, 1 without MMAs, 2 without the fold and
// carries (each step's word is the previous one XOR a coefficient, so the operands change).
template <int W, int MODE>
__global__ void __launch_bounds__(WARPS * 32) chain_kernel(u64 steps, unsigned long long *xr) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  Scratch s[W];
  u64 v[W];
  const u64 id = (u64)blockIdx.x * WARPS + warp;
#pragma unroll
  for (int w = 0; w < W; ++w) {
    s[w] = scratch_at(smem + (warp * W + w) * WORD_BYTES);
    init(s[w], lane);
    v[w] = seed_limb(id * W + w, lane);
  }
#pragma unroll 1
  for (u64 st = 0; st < steps; ++st) {
#pragma unroll
    for (int w = 0; w < W; ++w) {
      if constexpr (MODE == 0) {
        v[w] = square<true>(s[w], v[w], lane);
      } else if constexpr (MODE == 1) {
        v[w] = square<false>(s[w], v[w], lane);
      } else {
        put(s[w], v[w], lane);
        square_coeffs<true>(s[w], lane);
        v[w] = (v[w] ^ (u32)s[w].co[lane]) & MASK;
      }
    }
  }
  u64 acc = 0;
#pragma unroll
  for (int w = 0; w < W; ++w) acc ^= v[w] * (2 * w + 1);
  atomicXor(xr, (unsigned long long)acc);
}

// The tensor pipe's int8 rate at m16n8k32: NACC independent accumulators a warp, `steps` rounds of NACC MMAs on
// register operands (the operands change every round, so nothing is hoisted).
template <int NACC>
__global__ void __launch_bounds__(WARPS * 32) mma_kernel(u64 steps, unsigned long long *xr) {
  const int lane = threadIdx.x & 31;
  u32 a[4], b0 = 0x01020304u * (lane + 1), b1 = 0x05060708u ^ lane;
#pragma unroll
  for (int e = 0; e < 4; ++e) a[e] = 0x11111111u * (e + 1) + lane;
  int32_t acc[NACC][4];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0;
#pragma unroll 1
  for (u64 st = 0; st < steps; ++st) {
#pragma unroll
    for (int i = 0; i < NACC; ++i) mma_u8(acc[i], a, b0, b1);
    b0 += 0x01010101u;
  }
  int32_t h = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) h ^= acc[i][0] ^ acc[i][1] ^ acc[i][2] ^ acc[i][3];
  atomicXor(xr, (unsigned long long)(u32)h);
}

// Both units at once, time-boxed: warps [0, cw) square on the CUDA cores (kv.cu's sqr, a word a thread), warps [cw, 8) on
// the tensor cores (a word a warp); every warp squares until its SM's clock64 passes t0 + cycles and adds what it
// finished to counts[0] (CUDA cores) or counts[1] (tensor cores).
constexpr int MIX_WARPS = 8;
__global__ void __launch_bounds__(MIX_WARPS * 32, 1) mix_kernel(int cw, long long cycles, unsigned long long *counts,
                                                                unsigned long long *xr) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const long long t0 = clock64();
  u64 done = 0, h = 0;
  if (warp < cw) {
    u32 x[NL];
    u64 z = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 0x9E3779B97F4A7C15ull;
#pragma unroll
    for (int k = 0; k < NL; ++k) {
      z = z * 6364136223846793005ull + 1442695040888963407ull;
      x[k] = (u32)(z >> 32);
    }
    x[NL - 1] &= 0x7fffffffu;
#pragma unroll 1
    while (clock64() - t0 < cycles) {
      u32 y[NL];
      kv::sqr(x, y);
      kv::copy(x, y);
      ++done;
    }
#pragma unroll
    for (int k = 0; k < NL; ++k) h ^= x[k];
    done = __reduce_add_sync(FULL, (u32)done);
    if (lane == 0) atomicAdd(&counts[0], (unsigned long long)done);
  } else {
    Scratch s = scratch_at(smem + (warp - cw) * WORD_BYTES);
    init(s, lane);
    u64 v = seed_limb((u64)blockIdx.x * MIX_WARPS + warp, lane);
#pragma unroll 1
    while (clock64() - t0 < cycles) {
      v = square<true>(s, v, lane);
      ++done;
    }
    h = v;
    if (lane == 0) atomicAdd(&counts[1], (unsigned long long)done);
  }
  atomicXor(xr, (unsigned long long)h);
}

template <int W, int MODE>
int launch_chain(u64 steps, int grid, unsigned long long *xr, cudaStream_t st) {
  const int smem = WARPS * W * WORD_BYTES;
  auto k = chain_kernel<W, MODE>;
  if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return 2;
  k<<<grid, WARPS * 32, smem, st>>>(steps, xr);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

template <int W, int MODE>
int occupancy() {
  int n = 0;
  const int smem = WARPS * W * WORD_BYTES;
  auto k = chain_kernel<W, MODE>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, WARPS * 32, smem) == cudaSuccess ? n : -1;
}

}  // namespace mc

extern "C" {

int mc_word_bytes() { return mc::WORD_BYTES; }
int mc_fold6() { return kv::FOLD6; }

int mc_gate(const void *in, void *lazy, void *canonical, int n, int grid, cudaStream_t s) {
  mc::gate_kernel<<<grid, 32, 0, s>>>((const uint32_t *)in, (uint32_t *)lazy, (uint32_t *)canonical, n);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

int mc_chain(int mode, int w, uint64_t steps, int grid, unsigned long long *xr, cudaStream_t s) {
#define MC_CHAIN(W, M) if (w == W && mode == M) return mc::launch_chain<W, M>(steps, grid, xr, s);
  MC_CHAIN(1, 0) MC_CHAIN(2, 0) MC_CHAIN(4, 0)
  MC_CHAIN(1, 1) MC_CHAIN(2, 1) MC_CHAIN(4, 1)
  MC_CHAIN(1, 2) MC_CHAIN(2, 2) MC_CHAIN(4, 2)
#undef MC_CHAIN
  return 3;
}

int mc_chain_occupancy(int mode, int w) {
#define MC_OCC(W, M) if (w == W && mode == M) return mc::occupancy<W, M>();
  MC_OCC(1, 0) MC_OCC(2, 0) MC_OCC(4, 0)
  MC_OCC(1, 1) MC_OCC(2, 1) MC_OCC(4, 1)
  MC_OCC(1, 2) MC_OCC(2, 2) MC_OCC(4, 2)
#undef MC_OCC
  return -3;
}

int mc_mma(int nacc, uint64_t steps, int grid, unsigned long long *xr, cudaStream_t s) {
  if (nacc == 1) mc::mma_kernel<1><<<grid, mc::WARPS * 32, 0, s>>>(steps, xr);
  else if (nacc == 2) mc::mma_kernel<2><<<grid, mc::WARPS * 32, 0, s>>>(steps, xr);
  else if (nacc == 4) mc::mma_kernel<4><<<grid, mc::WARPS * 32, 0, s>>>(steps, xr);
  else if (nacc == 8) mc::mma_kernel<8><<<grid, mc::WARPS * 32, 0, s>>>(steps, xr);
  else return 3;
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

int mc_mix(int cw, long long cycles, int grid, unsigned long long *counts, unsigned long long *xr, cudaStream_t s) {
  const int smem = (mc::MIX_WARPS - cw) * mc::WORD_BYTES;
  if (cudaFuncSetAttribute(mc::mix_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) return 2;
  mc::mix_kernel<<<grid, mc::MIX_WARPS * 32, smem, s>>>(cw, cycles, counts, xr);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

}  // extern "C"

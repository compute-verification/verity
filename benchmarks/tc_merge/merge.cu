// Exact tensor-core merge D = Z1 + Z2 by one `mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32` per tile.
//
// Z1 enters as the C input. Z2 enters as A·B in the block-diagonal base-255 radix of `construction.py`: with n digit
// groups and s = min(8, 32/n) merged columns, group g of column j sits at k-slot g*s + j, B[g*s+j][j] = w_g with
// w = (1, 255, 255, ...), and A[i][g*s+j] is digit g of Z2[i][j]. The host builds A, B and C; the device runs the one
// instruction; the host checks every merged word bit for bit against Z1 + Z2 (mod 2^32), and every other word against C.
// Signed words go in biased by +bias (D holds Z1 + Z2 + bias, an exact form with a public offset).
//
//   merge check N_DIGITS TILES SEED LO HI BIAS   one JSON line: words checked, mismatches, units a word
//   merge time  N_DIGITS ITERS BLOCKS_PER_SM     one JSON line: MMAs a SM a cycle on merge operands
//   merge f16 TILES SEED LO HI BIAS Z1SPAN       the FP16 merge (two 11-bit digits, weights 1 and 2^11), checked
//   merge timef16 ITERS BLOCKS_PER_SM            its rate, in units a word (one unit = 1/256 SMSP tensor cycle)
//   merge fused TILES SEED LO HI BIAS Z1SPAN     the FP16 merge with its CUDA-core digit split from s32 words, checked
//   merge timefused ITERS BLOCKS_PER_SM          its rate: split + merge, everything counted
//
// Fragment layout as benchmarks/tc_probe/mma_tiles.cu (PTX "mma.m16n8k32" 8-bit figure), g = lane>>2, t = lane&3:
// A reg i holds row g (+8 for odd i), k = 4t..4t+3 (+16 for i >= 2); B reg i holds column g, the same k range;
// C/D reg 0..3 hold (g,2t), (g,2t+1), (g+8,2t), (g+8,2t+1).
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));         \
      exit(3);                                                                                   \
    }                                                                                            \
  } while (0)

namespace {

constexpr int M = 16, N = 8, K = 32;
constexpr int UNITS_PER_MMA = M * N * K;

__device__ __forceinline__ void mma_u8(uint32_t d[4], const uint32_t a[4], const uint32_t b[2], const uint32_t c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__global__ void merge_tiles(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bt,
                            const uint32_t* __restrict__ C, uint32_t* __restrict__ D, int ntiles) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= ntiles) return;
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const uint8_t* a = A + (size_t)warp * M * K;
  const uint8_t* b = Bt + (size_t)warp * N * K;
  const uint32_t* c = C + (size_t)warp * M * N;
  uint32_t* d = D + (size_t)warp * M * N;
  uint32_t af[4], bf[2], cf[4], df[4];
  af[0] = *reinterpret_cast<const uint32_t*>(a + g * K + 4 * t);
  af[1] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * K + 4 * t);
  af[2] = *reinterpret_cast<const uint32_t*>(a + g * K + 4 * t + 16);
  af[3] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * K + 4 * t + 16);
  bf[0] = *reinterpret_cast<const uint32_t*>(b + g * K + 4 * t);
  bf[1] = *reinterpret_cast<const uint32_t*>(b + g * K + 4 * t + 16);
  cf[0] = c[g * N + 2 * t];
  cf[1] = c[g * N + 2 * t + 1];
  cf[2] = c[(g + 8) * N + 2 * t];
  cf[3] = c[(g + 8) * N + 2 * t + 1];
  mma_u8(df, af, bf, cf);
  d[g * N + 2 * t] = df[0];
  d[g * N + 2 * t + 1] = df[1];
  d[(g + 8) * N + 2 * t] = df[2];
  d[(g + 8) * N + 2 * t + 1] = df[3];
}

// Throughput of the merge instruction on merge-shaped operands: CH independent accumulator chains a warp, so the pipe,
// not one chain's latency, sets the rate. A is fixed digit bytes, B the radix weights; C carries the chain.
template <int CH>
__global__ void time_merge(uint64_t iters, uint32_t wreg0, uint32_t wreg1, unsigned long long* cycles,
                           uint32_t* sink) {
  const int lane = threadIdx.x & 31;
  uint32_t a[4] = {0x01020304u ^ lane, 0x11223344u, 0x55667788u ^ lane, 0x0a0b0c0du};
  uint32_t b[2] = {wreg0, wreg1};
  uint32_t c[CH][4];
#pragma unroll
  for (int k = 0; k < CH; ++k)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[k][i] = k * 4 + i;
  __syncthreads();
  const unsigned long long t0 = clock64();
  for (uint64_t it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < CH; ++k) mma_u8(c[k], a, b, c[k]);
  }
  __syncthreads();
  const unsigned long long t1 = clock64();
  uint32_t acc = 0;
#pragma unroll
  for (int k = 0; k < CH; ++k) acc ^= c[k][0] ^ c[k][1] ^ c[k][2] ^ c[k][3];
  if (threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    cycles[3 * blockIdx.x] = t0;
    cycles[3 * blockIdx.x + 1] = t1;
    cycles[3 * blockIdx.x + 2] = smid;
  }
  if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;
}

// FP16 merge: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. FP16 holds power-of-two weights, so two 11-bit
// digits (lo weight 1, hi weight 2048 = 2^11) cover a 22-bit word where int8's 255 cap needs nine. Group g of column j
// sits at k-slot g*8 + j (s = 8, all 8 columns). C/D are FP32 holding the integer words; exact while |D| < 2^24.
// Layout as tc_probe's m16n8k16 (E = 2 halves a register, K = 16): A reg i holds row g (+8 odd i), k = 2t, 2t+1 (+8 for
// i >= 2); B reg i holds column g, the same k range.
constexpr int KH = 16;
constexpr int UNITS_PER_MMA_F16 = 2 * M * N * KH;  // FP16 at half int8's MAC rate: 2 units a MAC

__device__ __forceinline__ void mma_f16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__global__ void merge_tiles_f16(const uint16_t* __restrict__ A, const uint16_t* __restrict__ Bt,
                                const float* __restrict__ C, float* __restrict__ D, int ntiles) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= ntiles) return;
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const uint16_t* a = A + (size_t)warp * M * KH;
  const uint16_t* b = Bt + (size_t)warp * N * KH;
  const float* c = C + (size_t)warp * M * N;
  float* d = D + (size_t)warp * M * N;
  uint32_t af[4], bf[2];
  af[0] = *reinterpret_cast<const uint32_t*>(a + g * KH + 2 * t);
  af[1] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * KH + 2 * t);
  af[2] = *reinterpret_cast<const uint32_t*>(a + g * KH + 2 * t + 8);
  af[3] = *reinterpret_cast<const uint32_t*>(a + (g + 8) * KH + 2 * t + 8);
  bf[0] = *reinterpret_cast<const uint32_t*>(b + g * KH + 2 * t);
  bf[1] = *reinterpret_cast<const uint32_t*>(b + g * KH + 2 * t + 8);
  float cf[4] = {c[g * N + 2 * t], c[g * N + 2 * t + 1], c[(g + 8) * N + 2 * t], c[(g + 8) * N + 2 * t + 1]};
  float df[4];
  mma_f16(df, af, bf, cf);
  d[g * N + 2 * t] = df[0];
  d[g * N + 2 * t + 1] = df[1];
  d[(g + 8) * N + 2 * t] = df[2];
  d[(g + 8) * N + 2 * t + 1] = df[3];
}

template <int CH>
__global__ void time_merge_f16(uint64_t iters, uint32_t wreg0, uint32_t wreg1, unsigned long long* cycles,
                               uint32_t* sink) {
  const int lane = threadIdx.x & 31;
  // FP16 digit pairs (small exact integers); 0x3C00 = 1.0, 0x6800 = 2048.0
  uint32_t a[4] = {0x3C003C00u ^ (lane & 1), 0x40004000u, 0x3C004000u, 0x42004200u};
  uint32_t b[2] = {wreg0, wreg1};
  float c[CH][4];
#pragma unroll
  for (int k = 0; k < CH; ++k)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[k][i] = (float)(k * 4 + i);
  __syncthreads();
  const unsigned long long t0 = clock64();
  for (uint64_t it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < CH; ++k) mma_f16(c[k], a, b, c[k]);
  }
  __syncthreads();
  const unsigned long long t1 = clock64();
  float acc = 0;
#pragma unroll
  for (int k = 0; k < CH; ++k) acc += c[k][0] + c[k][1] + c[k][2] + c[k][3];
  if (threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    cycles[3 * blockIdx.x] = t0;
    cycles[3 * blockIdx.x + 1] = t1;
    cycles[3 * blockIdx.x + 2] = smid;
  }
  if (acc == 1.2345f) sink[blockIdx.x] = 1;
}

// The split on the CUDA cores, from four s32 words u in the C/D fragment layout ((g,2t), (g,2t+1), (g+8,2t), (g+8,2t+1))
// straight into the merge's A fragment: lo = u & 0x3FF and hi = (u >> 10) & 0x3FF, each as the FP16 1024 + d (bits
// 0x6400 | d), two words a register. Integer instructions only (4 PRMT, 2 SHF, 4 LOP3 a lane); the 1024s add the public
// constant SPLIT_OFFSET to D. Exact for u in [0, 2^20).
constexpr int64_t SPLIT_OFFSET = 1024 + 1024 * 1024;

__device__ __forceinline__ void split_f16(const uint32_t u[4], uint32_t af[4]) {
  af[0] = (__byte_perm(u[0], u[1], 0x5410) & 0x03FF03FFu) | 0x64006400u;
  af[1] = (__byte_perm(u[2], u[3], 0x5410) & 0x03FF03FFu) | 0x64006400u;
  af[2] = ((__byte_perm(u[0], u[1], 0x6521) >> 2) & 0x03FF03FFu) | 0x64006400u;
  af[3] = ((__byte_perm(u[2], u[3], 0x6521) >> 2) & 0x03FF03FFu) | 0x64006400u;
}

// Column g's radix weights at the lane's k-slots: 1.0 at slot g (lo), 1024.0 at slot 8 + g (hi).
__device__ __forceinline__ void split_weights(int g, int t, uint32_t bf[2]) {
  bf[0] = (2 * t == g ? 0x3C00u : 0u) | (2 * t + 1 == g ? 0x3C000000u : 0u);
  bf[1] = (2 * t == g ? 0x6400u : 0u) | (2 * t + 1 == g ? 0x64000000u : 0u);
}

__global__ void fused_tiles(const uint32_t* __restrict__ Z2, const float* __restrict__ C, float* __restrict__ D,
                            int ntiles) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= ntiles) return;
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const size_t o = (size_t)warp * M * N;
  const int at[4] = {g * N + 2 * t, g * N + 2 * t + 1, (g + 8) * N + 2 * t, (g + 8) * N + 2 * t + 1};
  uint32_t u[4], af[4], bf[2];
  float cf[4], df[4];
  for (int i = 0; i < 4; ++i) {
    u[i] = Z2[o + at[i]];
    cf[i] = C[o + at[i]];
  }
  split_f16(u, af);
  split_weights(g, t, bf);
  mma_f16(df, af, bf, cf);
  for (int i = 0; i < 4; ++i) D[o + at[i]] = df[i];
}

// Split + merge throughput: each chain's next Z2 words are another chain's accumulator bits, so the split is redone
// every MMA and nothing is hoisted.
template <int CH>
__global__ void time_fused(uint64_t iters, unsigned long long* cycles, uint32_t* sink) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  uint32_t bf[2];
  split_weights(g, t, bf);
  float c[CH][4];
#pragma unroll
  for (int k = 0; k < CH; ++k)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[k][i] = (float)(k * 4 + i + lane);
  __syncthreads();
  const unsigned long long t0 = clock64();
  for (uint64_t it = 0; it < iters; ++it) {
#pragma unroll
    for (int k = 0; k < CH; ++k) {
      const float* src = c[(k + 1) % CH];
      const uint32_t u[4] = {__float_as_uint(src[0]), __float_as_uint(src[1]), __float_as_uint(src[2]),
                             __float_as_uint(src[3])};
      uint32_t af[4];
      split_f16(u, af);
      mma_f16(c[k], af, bf, c[k]);
    }
  }
  __syncthreads();
  const unsigned long long t1 = clock64();
  float acc = 0;
#pragma unroll
  for (int k = 0; k < CH; ++k) acc += c[k][0] + c[k][1] + c[k][2] + c[k][3];
  if (threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    cycles[3 * blockIdx.x] = t0;
    cycles[3 * blockIdx.x + 1] = t1;
    cycles[3 * blockIdx.x + 2] = smid;
  }
  if (acc == 1.2345f) sink[blockIdx.x] = 1;
}

int64_t vmax(int n) { return 254 + (int64_t)(n - 1) * 255 * 255; }
int cols(int n) { return n > 32 ? 1 : (N < K / n ? N : K / n); }

void digits(int64_t v, int n, int* d) {
  d[0] = (int)(v % 255);
  int64_t q = v / 255;
  for (int g = 1; g < n; ++g) {
    const int64_t take = q < 255 ? q : 255;
    d[g] = (int)take;
    q -= take;
  }
  if (q != 0) {
    fprintf(stderr, "digit overflow v=%lld n=%d\n", (long long)v, n);
    exit(5);
  }
}

uint32_t wrap(int64_t x) { return (uint32_t)(int32_t)(uint32_t)(uint64_t)x; }

int run_check(int n, int ntiles, uint32_t seed, int64_t lo, int64_t hi, int64_t bias, int perturb) {
  if (n < 1 || n > 32) {
    fprintf(stderr, "n is 1..32\n");
    return 2;
  }
  const int s = cols(n);
  if (hi + bias > vmax(n) || lo + bias < 0) {
    fprintf(stderr, "domain [%lld,%lld]+%lld outside [0,%lld]\n", (long long)lo, (long long)hi, (long long)bias,
            (long long)vmax(n));
    return 2;
  }
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> z2d(lo, hi);
  std::uniform_int_distribution<int64_t> z1d(-(1LL << 30), (1LL << 30));
  std::vector<uint8_t> A((size_t)ntiles * M * K, 0), Bt((size_t)ntiles * N * K, 0);
  std::vector<uint32_t> C((size_t)ntiles * M * N), D((size_t)ntiles * M * N);
  std::vector<int64_t> Z2((size_t)ntiles * M * N, 0);
  int w[32];
  w[0] = 1;
  for (int g = 1; g < 32; ++g) w[g] = 255;
  if (perturb && n > 1) w[1] = 254;
  int dg[32];
  for (int tile = 0; tile < ntiles; ++tile) {
    uint8_t* a = &A[(size_t)tile * M * K];
    uint8_t* bt = &Bt[(size_t)tile * N * K];
    for (int g = 0; g < n; ++g)
      for (int j = 0; j < s; ++j) bt[j * K + g * s + j] = (uint8_t)w[g];
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        const size_t o = (size_t)tile * M * N + i * N + j;
        C[o] = wrap(z1d(rng));
        if (j >= s) continue;
        int64_t v = z2d(rng);
        if (tile == 0) v = (i + j) & 1 ? hi : lo;  // both domain edges in the first tile
        Z2[o] = v;
        digits(v + bias, n, dg);
        for (int g = 0; g < n; ++g) a[i * K + g * s + j] = (uint8_t)dg[g];
      }
  }
  uint8_t *dA, *dB;
  uint32_t *dC, *dD;
  CK(cudaMalloc(&dA, A.size()));
  CK(cudaMalloc(&dB, Bt.size()));
  CK(cudaMalloc(&dC, C.size() * 4));
  CK(cudaMalloc(&dD, D.size() * 4));
  CK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, Bt.data(), Bt.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dD, 0xA5, D.size() * 4));
  constexpr int WPB = 4;
  merge_tiles<<<(ntiles + WPB - 1) / WPB, 32 * WPB>>>(dA, dB, dC, dD, ntiles);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost));
  long long merged = 0, bad = 0, passthru = 0, badpass = 0;
  for (int tile = 0; tile < ntiles; ++tile)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        const size_t o = (size_t)tile * M * N + i * N + j;
        if (j < s) {
          ++merged;
          const uint32_t want = wrap((int64_t)(int32_t)C[o] + Z2[o] + bias);
          if (D[o] != want) {
            if (bad < 5)
              fprintf(stderr, "mismatch tile %d (%d,%d): got %u want %u\n", tile, i, j, D[o], want);
            ++bad;
          }
        } else {
          ++passthru;
          badpass += D[o] != C[o];
        }
      }
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
  CK(cudaFree(dD));
  printf(
      "{\"mode\":\"check\",\"n_digits\":%d,\"cols\":%d,\"words_per_mma\":%d,\"units_per_word\":%.4f,\"vmax\":%lld,"
      "\"lo\":%lld,\"hi\":%lld,\"bias\":%lld,\"perturb\":%d,\"tiles\":%d,\"seed\":%u,\"merged_words\":%lld,"
      "\"mismatches\":%lld,\"passthrough_words\":%lld,\"passthrough_mismatches\":%lld}\n",
      n, s, M * s, (double)UNITS_PER_MMA / (M * s), (long long)vmax(n), (long long)lo, (long long)hi,
      (long long)bias, perturb, ntiles, seed, merged, bad, passthru, badpass);
  return bad || badpass ? 1 : 0;
}

// MMAs an SM-cycle from each block's (t0, t1, smid): each SM's MMAs over its own busy window, first start to last end,
// which stays right when an SM doesn't hold all its blocks at once. Also the mean span of one block.
double sm_rate(const std::vector<unsigned long long>& cyc, int blocks, double mma_per_block, double* mean_block) {
  constexpr int MAXSM = 1024;
  std::vector<unsigned long long> first(MAXSM, ~0ULL), last(MAXSM, 0);
  std::vector<int> nb(MAXSM, 0);
  double sum = 0;
  for (int b = 0; b < blocks; ++b) {
    const unsigned long long t0 = cyc[3 * b], t1 = cyc[3 * b + 1];
    const int sm = (int)cyc[3 * b + 2];
    first[sm] = std::min(first[sm], t0);
    last[sm] = std::max(last[sm], t1);
    ++nb[sm];
    sum += (double)(t1 - t0);
  }
  *mean_block = sum / blocks;
  double mmas = 0, span = 0;
  for (int sm = 0; sm < MAXSM; ++sm)
    if (nb[sm]) {
      mmas += nb[sm] * mma_per_block;
      span += (double)(last[sm] - first[sm]);
    }
  return mmas / span;
}

int run_time(int n, uint64_t iters, int bpsm) {
  const int s = cols(n);
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  // IMMA's rate doesn't depend on operand values; radix-weight bytes (1 and 255) keep the operands merge-shaped.
  const uint32_t w0 = 0x01FFFFFFu, w1 = 0xFFFFFFFFu;
  constexpr int CH = 8, WARPS = 4;
  const int blocks = p.multiProcessorCount * bpsm;
  unsigned long long* dcyc;
  uint32_t* dsink;
  CK(cudaMalloc(&dcyc, 3 * blocks * sizeof(unsigned long long)));
  CK(cudaMalloc(&dsink, blocks * 4));
  time_merge<CH><<<blocks, 32 * WARPS>>>(iters / 8 + 1, w0, w1, dcyc, dsink);
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  time_merge<CH><<<blocks, 32 * WARPS>>>(iters, w0, w1, dcyc, dsink);
  CK(cudaEventRecord(e1));
  CK(cudaDeviceSynchronize());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  std::vector<unsigned long long> cyc(3 * blocks);
  CK(cudaMemcpy(cyc.data(), dcyc, 3 * blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  const double mma_per_block = (double)iters * CH * WARPS;
  double mean_cyc;
  const double mma_per_sm_cycle = sm_rate(cyc, blocks, mma_per_block, &mean_cyc);
  const double macs_per_sm_clk = mma_per_sm_cycle * UNITS_PER_MMA;
  const double cyc_per_mma_smsp = 4.0 / mma_per_sm_cycle;
  const double units_per_word = (cyc_per_mma_smsp / 16.0) * UNITS_PER_MMA / (M * s);
  printf(
      "{\"mode\":\"time\",\"n_digits\":%d,\"cols\":%d,\"iters\":%llu,\"blocks\":%d,\"bpsm\":%d,\"warps\":%d,"
      "\"chains\":%d,\"mean_block_cycles\":%.0f,\"ms\":%.4f,\"mma_per_sm_cycle\":%.5f,\"macs_per_sm_clk\":%.2f,"
      "\"cycles_per_mma_smsp\":%.4f,\"units_per_word_measured\":%.4f,\"units_per_word_predicted\":%.4f,"
      "\"sms\":%d}\n",
      n, s, (unsigned long long)iters, blocks, bpsm, WARPS, CH, mean_cyc, ms, mma_per_sm_cycle, macs_per_sm_clk,
      cyc_per_mma_smsp, units_per_word, (double)UNITS_PER_MMA / (M * s), p.multiProcessorCount);
  CK(cudaFree(dcyc));
  CK(cudaFree(dsink));
  return 0;
}

// binary16 bits of a non-negative integer exactly representable in FP16 (every value here is <= 2048).
uint16_t half_bits(int v) {
  if (v == 0) return 0;
  const float f = (float)v;
  uint32_t x;
  memcpy(&x, &f, 4);
  const int e = (int)((x >> 23) & 0xFF) - 127 + 15;
  const uint32_t mant = x & 0x7FFFFF;
  if (e <= 0 || e >= 31 || (mant & 0x1FFF)) {
    fprintf(stderr, "%d not exact in fp16\n", v);
    exit(5);
  }
  return (uint16_t)((e << 10) | (mant >> 13));
}

uint32_t fbits(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  return x;
}

uint16_t half_bits_signed(int v) { return v < 0 ? (uint16_t)(0x8000 | half_bits(-v)) : half_bits(v); }

// u = Z2 + bias splits as u = hi * 2^11 + lo with lo in [0, 2047] and hi = floor(u / 2^11) signed, so any
// u in [-2^22, 2^22) is two exact FP16 digits.
int run_check_f16(int ntiles, uint32_t seed, int64_t lo, int64_t hi, int64_t bias, int64_t z1span, int perturb) {
  constexpr int NB = 2, BITS = 11, S = 8;
  if (lo + bias < -(1LL << (NB * BITS)) || hi + bias >= (1LL << (NB * BITS))) {
    fprintf(stderr, "domain [%lld,%lld]+%lld outside [-2^22,2^22)\n", (long long)lo, (long long)hi, (long long)bias);
    return 2;
  }
  if (z1span + hi + bias >= (1LL << 24) || z1span - lo - bias >= (1LL << 24)) {
    fprintf(stderr, "|Z1 + Z2| may reach 2^24, past FP32's exact integers\n");
    return 2;
  }
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> z2d(lo, hi), z1d(-z1span, z1span);
  std::vector<uint16_t> A((size_t)ntiles * M * KH, 0), Bt((size_t)ntiles * N * KH, 0);
  std::vector<float> C((size_t)ntiles * M * N), D((size_t)ntiles * M * N);
  std::vector<int64_t> want((size_t)ntiles * M * N);
  const int w[NB] = {1, perturb ? 2047 : 1 << BITS};
  int64_t maxabs = 0;
  for (int tile = 0; tile < ntiles; ++tile) {
    uint16_t* a = &A[(size_t)tile * M * KH];
    uint16_t* bt = &Bt[(size_t)tile * N * KH];
    for (int gi = 0; gi < NB; ++gi)
      for (int j = 0; j < S; ++j) bt[j * KH + gi * S + j] = half_bits(w[gi]);
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        const size_t o = (size_t)tile * M * N + i * N + j;
        int64_t z1 = z1d(rng), v = z2d(rng);
        if (tile == 0) {  // the four corners of (Z1, Z2) in the first tile
          v = (i + j) & 1 ? hi : lo;
          z1 = (i >> 1) & 1 ? z1span : -z1span;
        }
        C[o] = (float)z1;
        const int64_t u = v + bias;
        a[i * KH + 0 * S + j] = half_bits((int)(u & ((1 << BITS) - 1)));
        a[i * KH + 1 * S + j] = half_bits_signed((int)(u >> BITS));
        want[o] = z1 + u;
        maxabs = std::max<int64_t>(maxabs, std::llabs(want[o]));
      }
  }
  uint16_t *dA, *dB;
  float *dC, *dD;
  CK(cudaMalloc(&dA, A.size() * 2));
  CK(cudaMalloc(&dB, Bt.size() * 2));
  CK(cudaMalloc(&dC, C.size() * 4));
  CK(cudaMalloc(&dD, D.size() * 4));
  CK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, Bt.data(), Bt.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dD, 0xA5, D.size() * 4));
  constexpr int WPB = 4;
  merge_tiles_f16<<<(ntiles + WPB - 1) / WPB, 32 * WPB>>>(dA, dB, dC, dD, ntiles);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost));
  long long merged = 0, bad = 0;
  for (size_t o = 0; o < D.size(); ++o) {
    ++merged;
    if (fbits(D[o]) != fbits((float)want[o])) {
      if (bad < 5) fprintf(stderr, "mismatch word %zu: got %.1f want %lld\n", o, (double)D[o], (long long)want[o]);
      ++bad;
    }
  }
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
  CK(cudaFree(dD));
  printf(
      "{\"mode\":\"check_f16\",\"n_digits\":%d,\"digit_bits\":%d,\"cols\":%d,\"words_per_mma\":%d,"
      "\"units_per_word\":%.4f,\"lo\":%lld,\"hi\":%lld,\"bias\":%lld,\"z1span\":%lld,\"max_abs_result\":%lld,"
      "\"perturb\":%d,\"tiles\":%d,\"seed\":%u,\"merged_words\":%lld,\"mismatches\":%lld}\n",
      NB, BITS, S, M * S, (double)UNITS_PER_MMA_F16 / (M * S), (long long)lo, (long long)hi, (long long)bias,
      (long long)z1span, (long long)maxabs, perturb, ntiles, seed, merged, bad);
  return bad ? 1 : 0;
}

int run_time_f16(uint64_t iters, int bpsm) {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  const uint32_t w = 0x68003C00u;  // halves (1.0, 2048.0): the radix weights
  constexpr int CH = 8, WARPS = 4;
  const int blocks = p.multiProcessorCount * bpsm;
  unsigned long long* dcyc;
  uint32_t* dsink;
  CK(cudaMalloc(&dcyc, 3 * blocks * sizeof(unsigned long long)));
  CK(cudaMalloc(&dsink, blocks * 4));
  time_merge_f16<CH><<<blocks, 32 * WARPS>>>(iters / 8 + 1, w, w, dcyc, dsink);
  CK(cudaDeviceSynchronize());
  time_merge_f16<CH><<<blocks, 32 * WARPS>>>(iters, w, w, dcyc, dsink);
  CK(cudaDeviceSynchronize());
  std::vector<unsigned long long> cyc(3 * blocks);
  CK(cudaMemcpy(cyc.data(), dcyc, 3 * blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  double mean_cyc;
  const double mma_per_sm_cycle = sm_rate(cyc, blocks, (double)iters * CH * WARPS, &mean_cyc);
  const double cyc_per_mma_smsp = 4.0 / mma_per_sm_cycle;
  // one unit = one int8 MAC slot = 1/256 of an SMSP tensor cycle (m16n8k32 u8 holds the pipe 16 cycles)
  const double units_per_word = cyc_per_mma_smsp * 256.0 / (M * N);
  printf(
      "{\"mode\":\"time_f16\",\"iters\":%llu,\"blocks\":%d,\"bpsm\":%d,\"warps\":%d,\"chains\":%d,"
      "\"mean_block_cycles\":%.0f,\"mma_per_sm_cycle\":%.5f,\"f16_macs_per_sm_clk\":%.2f,"
      "\"cycles_per_mma_smsp\":%.4f,\"units_per_word_measured\":%.4f,\"units_per_word_predicted\":%.4f,\"sms\":%d}\n",
      (unsigned long long)iters, blocks, bpsm, WARPS, CH, mean_cyc, mma_per_sm_cycle,
      mma_per_sm_cycle * M * N * KH, cyc_per_mma_smsp, units_per_word, (double)UNITS_PER_MMA_F16 / (M * N),
      p.multiProcessorCount);
  CK(cudaFree(dcyc));
  CK(cudaFree(dsink));
  return 0;
}

// The fused split + merge on s32 words: Z2 + bias in [0, 2^20) (a bias the producing GEMM can start its accumulator at),
// Z1 an FP32 integer; D must be Z1 + Z2 + bias + SPLIT_OFFSET bit for bit.
int run_check_fused(int ntiles, uint32_t seed, int64_t lo, int64_t hi, int64_t bias, int64_t z1span) {
  if (lo + bias < 0 || hi + bias >= (1LL << 20)) {
    fprintf(stderr, "domain [%lld,%lld]+%lld outside [0,2^20)\n", (long long)lo, (long long)hi, (long long)bias);
    return 2;
  }
  if (z1span + hi + bias + SPLIT_OFFSET >= (1LL << 24)) {
    fprintf(stderr, "|D| may reach 2^24\n");
    return 2;
  }
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> z2d(lo, hi), z1d(-z1span, z1span);
  const size_t words = (size_t)ntiles * M * N;
  std::vector<uint32_t> Z2(words);
  std::vector<float> C(words), D(words);
  std::vector<int64_t> want(words);
  int64_t maxabs = 0;
  for (size_t o = 0; o < words; ++o) {
    int64_t v = z2d(rng), z1 = z1d(rng);
    if (o < M * N) {  // the four corners of (Z1, Z2) in the first tile
      v = o & 1 ? hi : lo;
      z1 = o & 2 ? z1span : -z1span;
    }
    Z2[o] = (uint32_t)(v + bias);
    C[o] = (float)z1;
    want[o] = z1 + v + bias + SPLIT_OFFSET;
    maxabs = std::max<int64_t>(maxabs, std::llabs(want[o]));
  }
  uint32_t* dZ;
  float *dC, *dD;
  CK(cudaMalloc(&dZ, words * 4));
  CK(cudaMalloc(&dC, words * 4));
  CK(cudaMalloc(&dD, words * 4));
  CK(cudaMemcpy(dZ, Z2.data(), words * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dC, C.data(), words * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dD, 0xA5, words * 4));
  constexpr int WPB = 4;
  fused_tiles<<<(ntiles + WPB - 1) / WPB, 32 * WPB>>>(dZ, dC, dD, ntiles);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(D.data(), dD, words * 4, cudaMemcpyDeviceToHost));
  long long bad = 0;
  for (size_t o = 0; o < words; ++o)
    if (fbits(D[o]) != fbits((float)want[o])) {
      if (bad < 5) fprintf(stderr, "fused mismatch word %zu: got %.1f want %lld\n", o, (double)D[o], (long long)want[o]);
      ++bad;
    }
  CK(cudaFree(dZ));
  CK(cudaFree(dC));
  CK(cudaFree(dD));
  printf(
      "{\"mode\":\"check_fused\",\"digit_bits\":10,\"lo\":%lld,\"hi\":%lld,\"bias\":%lld,\"offset\":%lld,"
      "\"z1span\":%lld,\"max_abs_result\":%lld,\"tiles\":%d,\"seed\":%u,\"merged_words\":%zu,\"mismatches\":%lld}\n",
      (long long)lo, (long long)hi, (long long)bias, (long long)SPLIT_OFFSET, (long long)z1span, (long long)maxabs,
      ntiles, seed, words, bad);
  return bad ? 1 : 0;
}

int run_time_fused(uint64_t iters, int bpsm) {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  constexpr int CH = 8, WARPS = 4;
  const int blocks = p.multiProcessorCount * bpsm;
  unsigned long long* dcyc;
  uint32_t* dsink;
  CK(cudaMalloc(&dcyc, 3 * blocks * sizeof(unsigned long long)));
  CK(cudaMalloc(&dsink, blocks * 4));
  time_fused<CH><<<blocks, 32 * WARPS>>>(iters / 8 + 1, dcyc, dsink);
  CK(cudaDeviceSynchronize());
  time_fused<CH><<<blocks, 32 * WARPS>>>(iters, dcyc, dsink);
  CK(cudaDeviceSynchronize());
  std::vector<unsigned long long> cyc(3 * blocks);
  CK(cudaMemcpy(cyc.data(), dcyc, 3 * blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  double mean_cyc;
  const double mma_per_sm_cycle = sm_rate(cyc, blocks, (double)iters * CH * WARPS, &mean_cyc);
  const double cyc_per_mma_smsp = 4.0 / mma_per_sm_cycle;
  printf(
      "{\"mode\":\"time_fused\",\"iters\":%llu,\"blocks\":%d,\"bpsm\":%d,\"warps\":%d,\"chains\":%d,"
      "\"mean_block_cycles\":%.0f,\"mma_per_sm_cycle\":%.5f,\"cycles_per_mma_smsp\":%.4f,"
      "\"units_per_word_measured\":%.4f,\"sms\":%d}\n",
      (unsigned long long)iters, blocks, bpsm, WARPS, CH, mean_cyc, mma_per_sm_cycle, cyc_per_mma_smsp,
      cyc_per_mma_smsp * 256.0 / (M * N), p.multiProcessorCount);
  CK(cudaFree(dcyc));
  CK(cudaFree(dsink));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 8 && !strcmp(argv[1], "check"))
    return run_check(atoi(argv[2]), atoi(argv[3]), (uint32_t)strtoul(argv[4], 0, 0), atoll(argv[5]), atoll(argv[6]),
                     atoll(argv[7]), argc >= 9 ? atoi(argv[8]) : 0);
  if (argc >= 5 && !strcmp(argv[1], "time")) return run_time(atoi(argv[2]), strtoull(argv[3], 0, 0), atoi(argv[4]));
  if (argc >= 8 && !strcmp(argv[1], "f16"))
    return run_check_f16(atoi(argv[2]), (uint32_t)strtoul(argv[3], 0, 0), atoll(argv[4]), atoll(argv[5]),
                         atoll(argv[6]), atoll(argv[7]), argc >= 9 ? atoi(argv[8]) : 0);
  if (argc >= 4 && !strcmp(argv[1], "timef16")) return run_time_f16(strtoull(argv[2], 0, 0), atoi(argv[3]));
  if (argc >= 8 && !strcmp(argv[1], "fused"))
    return run_check_fused(atoi(argv[2]), (uint32_t)strtoul(argv[3], 0, 0), atoll(argv[4]), atoll(argv[5]),
                           atoll(argv[6]), atoll(argv[7]));
  if (argc >= 4 && !strcmp(argv[1], "timefused")) return run_time_fused(strtoull(argv[2], 0, 0), atoi(argv[3]));
  fprintf(stderr,
          "usage: merge check N TILES SEED LO HI BIAS [PERTURB] | merge time N ITERS BLOCKS_PER_SM\n"
          "       merge f16 TILES SEED LO HI BIAS Z1SPAN [PERTURB] | merge timef16 ITERS BLOCKS_PER_SM\n"
          "       merge fused TILES SEED LO HI BIAS Z1SPAN | merge timefused ITERS BLOCKS_PER_SM\n");
  return 2;
}

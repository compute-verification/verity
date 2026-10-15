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
//
// Fragment layout as benchmarks/tc_probe/mma_tiles.cu (PTX "mma.m16n8k32" 8-bit figure), g = lane>>2, t = lane&3:
// A reg i holds row g (+8 for odd i), k = 4t..4t+3 (+16 for i >= 2); B reg i holds column g, the same k range;
// C/D reg 0..3 hold (g,2t), (g,2t+1), (g+8,2t), (g+8,2t+1).
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
  if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;
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
  CK(cudaMalloc(&dcyc, blocks * sizeof(unsigned long long)));
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
  std::vector<unsigned long long> cyc(blocks);
  CK(cudaMemcpy(cyc.data(), dcyc, blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  double mean_cyc = 0;
  for (auto c : cyc) mean_cyc += (double)c;
  mean_cyc /= blocks;
  const double mma_per_block = (double)iters * CH * WARPS;
  const double mma_per_sm_cycle = mma_per_block * bpsm / mean_cyc;
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

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 8 && !strcmp(argv[1], "check"))
    return run_check(atoi(argv[2]), atoi(argv[3]), (uint32_t)strtoul(argv[4], 0, 0), atoll(argv[5]), atoll(argv[6]),
                     atoll(argv[7]), argc >= 9 ? atoi(argv[8]) : 0);
  if (argc >= 5 && !strcmp(argv[1], "time")) return run_time(atoi(argv[2]), strtoull(argv[3], 0, 0), atoi(argv[4]));
  fprintf(stderr, "usage: merge check N TILES SEED LO HI BIAS [PERTURB] | merge time N ITERS BLOCKS_PER_SM\n");
  return 2;
}

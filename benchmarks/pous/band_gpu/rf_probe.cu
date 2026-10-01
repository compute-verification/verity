// A standalone sm_120 probe (no torch): does the integer pipe's issue rate depend on how many distinct registers an
// instruction reads, how far back its source was written, how many registers are live, or how many warps share a
// scheduler?  The band chain's Keccak-f, whose LOP3s read three registers, issues at about 0.49 per scheduler clock and
// gains 2 % from a second warp per scheduler.
//
//   nvcc -O3 -arch=sm_120 -o rf_probe rf_probe.cu && ./rf_probe > rf.json
//
// Each thread runs 16 chains x[0..15] of one pattern, unrolled; y, z are loop-invariant.  x[k+a] is x[(k+a) % 16], written
// 16 - a instructions before it is read, and no two consecutive instructions read a common register unless the pattern says
// so.  Reported as warp instructions per SM clock (clock64 inside the kernel) at 1, 2, 4 and 12 warps per scheduler, null
// where that many blocks do not fit on an SM.  Under ncu, smsp__inst_executed over smsp__cycles_elapsed and
// sm__pipe_aluheavy_cycles_active check these rates.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));        \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)

__device__ __forceinline__ uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ uint32_t shf(uint32_t a, uint32_t b) {
  uint32_t d;
  asm volatile("shf.l.wrap.b32 %0, %1, %2, 7;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ uint32_t keep(uint32_t a) {
  uint32_t d;
  asm volatile("mov.b32 %0, %1;" : "=r"(d) : "r"(a));
  return d;
}

constexpr int NPAT = 13;
const char* PATTERNS[NPAT] = {
    "lop3_1: x[k] ^ y ^ z (one changing source)",
    "lop3_2: x[k] ^ x[k+3] ^ z (two changing)",
    "lop3_3: x[k] ^ x[k+3] ^ x[k+6] (three changing, none shared with the previous instruction)",
    "lop3_3g: x[k] ^ u ^ v, u and v two other chains fixed for 4 consecutive instructions (theta's column update)",
    "shf_1: shf(x[k], x[k]) (one source)",
    "shf_2: shf(x[k+3], x[k]) (two)",
    "lop3_3+shf_1: alternating",
    "lop3_3x2+shf_1: two lop3_3 per shf_1 (Keccak's 122 LOP3 : 52 SHF is 2.3 : 1)",
    "dep1: x[k] ^ x[k-1] ^ z (each LOP3 reads the previous one's result)",
    "dep2: x[k] ^ x[k-2] ^ z (reads the result of two instructions back)",
    "dep4: x[k] ^ x[k-4] ^ z (four back)",
    "lop3_3 with 64 more registers live across the loop",
    "lop3_3 with 128 more registers live across the loop",
};
__host__ __device__ constexpr int extra_live(int p) { return p == 11 ? 64 : p == 12 ? 128 : 0; }

constexpr int UNROLL = 8;
template <int P>
__global__ void rf_kernel(uint32_t* out, int iters, long long* cycles) {
  constexpr int R = extra_live(P);
  uint32_t x[16], e[R > 0 ? R : 1];
  const uint32_t y = threadIdx.x * 0x9E3779B9u, z = blockIdx.x * 0x85EBCA6Bu;
#pragma unroll
  for (int k = 0; k < 16; ++k) x[k] = threadIdx.x + k * 0x1234567u;
#pragma unroll
  for (int k = 0; k < R; ++k) e[k] = keep(threadIdx.x * 0x2545F491u + k);
  __syncthreads();
  const long long c0 = clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const int g = k >> 2;
        if (P == 0) x[k] = lop3(x[k], y, z);
        if (P == 1) x[k] = lop3(x[k], x[(k + 3) & 15], z);
        if (P == 2 || R > 0) x[k] = lop3(x[k], x[(k + 3) & 15], x[(k + 6) & 15]);
        if (P == 3) x[k] = lop3(x[k], x[((g + 1) & 3) * 4], x[((g + 2) & 3) * 4 + 1]);
        if (P == 4) x[k] = shf(x[k], x[k]);
        if (P == 5) x[k] = shf(x[(k + 3) & 15], x[k]);
        if (P == 6) x[k] = (k & 1) ? shf(x[k], x[k]) : lop3(x[k], x[(k + 3) & 15], x[(k + 6) & 15]);
        if (P == 7) x[k] = (k % 3 == 2) ? shf(x[k], x[k]) : lop3(x[k], x[(k + 3) & 15], x[(k + 6) & 15]);
        if (P == 8) x[k] = lop3(x[k], x[(k + 15) & 15], z);
        if (P == 9) x[k] = lop3(x[k], x[(k + 14) & 15], z);
        if (P == 10) x[k] = lop3(x[k], x[(k + 12) & 15], z);
      }
    }
  }
  __syncthreads();
  const long long c1 = clock64();
  uint32_t acc = 0;
#pragma unroll
  for (int k = 0; k < 16; ++k) acc ^= x[k];
#pragma unroll
  for (int k = 0; k < R; ++k) acc ^= keep(e[k]);
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
  if (threadIdx.x == 0) {
    uint32_t sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    cycles[3 * blockIdx.x] = c0;
    cycles[3 * blockIdx.x + 1] = c1;
    cycles[3 * blockIdx.x + 2] = sm;
  }
}

template <int P>
static double rate(int sms, int wps, int iters, uint32_t* dout, long long* dcyc) {
  const int threads = 128, blocks = sms * wps;  // four warps a block, one per scheduler
  int resident = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, rf_kernel<P>, threads, 0));
  if (resident < wps) return -1;  // the blocks would not all be resident, and per-block clocks would hide it
  rf_kernel<P><<<blocks, threads>>>(dout, 4, dcyc);
  CK(cudaDeviceSynchronize());
  rf_kernel<P><<<blocks, threads>>>(dout, iters, dcyc);
  CK(cudaDeviceSynchronize());
  std::vector<long long> cyc(3 * blocks);
  CK(cudaMemcpy(cyc.data(), dcyc, cyc.size() * sizeof(long long), cudaMemcpyDeviceToHost));
  // Blocks that share an SM need not overlap for their whole run, and one that runs alone for a while finishes early, so a
  // mean of per-block durations can overstate the rate; time each SM from its first block's start to its last block's end
  // on that SM's clock instead.
  std::map<long long, std::array<long long, 3>> span;  // smid -> first start, last end, blocks
  for (int b = 0; b < blocks; ++b) {
    auto it = span.find(cyc[3 * b + 2]);
    if (it == span.end()) span[cyc[3 * b + 2]] = {cyc[3 * b], cyc[3 * b + 1], 1};
    else it->second = {std::min(it->second[0], cyc[3 * b]), std::max(it->second[1], cyc[3 * b + 1]), it->second[2] + 1};
  }
  double sum = 0;
  for (auto& [sm, s] : span) sum += (double)iters * UNROLL * 16 * 4 * s[2] / (double)(s[1] - s[0]);
  return sum / span.size();  // warp instructions per SM clock
}

typedef double (*Rate)(int, int, int, uint32_t*, long long*);
static Rate RATES[NPAT] = {rate<0>, rate<1>, rate<2>, rate<3>, rate<4>,  rate<5>, rate<6>,
                           rate<7>, rate<8>, rate<9>, rate<10>, rate<11>, rate<12>};

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 4000;
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  const int sms = p.multiProcessorCount;
  const int wpss[] = {1, 2, 4, 12};
  uint32_t* dout;
  long long* dcyc;
  CK(cudaMalloc(&dout, (size_t)sms * 12 * 128 * sizeof(uint32_t)));
  CK(cudaMalloc(&dcyc, (size_t)sms * 12 * 3 * sizeof(long long)));
  printf("{\"device\": \"%s\", \"sms\": %d, \"cc\": \"%d.%d\", \"iters\": %d, \"unit\": \"warp instructions per SM clock\", "
         "\"warps_per_scheduler\": [1, 2, 4, 12], \"patterns\": [",
         p.name, sms, p.major, p.minor, iters);
  for (int i = 0; i < NPAT; ++i) {
    printf("%s\n {\"pattern\": \"%s\", \"rate\": [", i ? "," : "", PATTERNS[i]);
    for (int w = 0; w < 4; ++w) {
      const double r = RATES[i](sms, wpss[w], iters, dout, dcyc);
      if (r < 0) printf("%snull", w ? ", " : "");
      else printf("%s%.4f", w ? ", " : "", r);
    }
    printf("]}");
  }
  printf("\n]}\n");
  return 0;
}

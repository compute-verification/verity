// The H200's off-register and off-SM word routes, each checked before it is read: TMA bulk reductions
// (cp.reduce.async.bulk .add.u32, shared -> global, the L2 computing the word), LDSM.x4 (ldmatrix, 4 registers a lane)
// beside LDS.128, the copy engine (device-to-device memcpy) and PCIe in each direction (the host route's link).
// Prints one JSON object. A price is MH100's: 4,096 over the 32-bit words one SM writes a clock (SM routes, clock64),
// or for a chip-wide route 540,672 x f_SM x seconds over words (f_SM = 1.98 GHz, the clock nvidia-smi reports).
//   nvcc -O3 -arch=sm_90a -std=c++17 engines.cu -o engines
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

constexpr double UNITS_PER_S = 540672.0 * 1.98e9;

// Each issuing warp's lane 0 reduce-adds its own TILE bytes of shared memory into NT tiles of its own global region,
// REPS times, keeping every bulk group in flight until one wait at the end.
template <int TILE>
__global__ void bulk_red(unsigned* g, int nt, int reps) {
  extern __shared__ __align__(128) unsigned s[];
  const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int i = threadIdx.x; i < nw * TILE / 4; i += blockDim.x) s[i] = 1u + (i & 1);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if ((threadIdx.x & 31) == 0) {
    const unsigned src = (unsigned)__cvta_generic_to_shared(s + w * TILE / 4);
    unsigned* base = g + ((size_t)blockIdx.x * nw + w) * nt * (TILE / 4);
    for (int r = 0; r < reps; r++)
      for (int t = 0; t < nt; t++) {
        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;" ::"l"(
                         base + (size_t)t * (TILE / 4)),
                     "r"(src), "n"(TILE)
                     : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  }
}

// LDSM.x4: lane l gives row l's address (16 B rows, 8 rows a matrix fill the 32 banks once), 8 an iteration.
__global__ void ldsm_x4(unsigned* sink, long long* cyc, int iters) {
  __shared__ __align__(128) unsigned s[8 * 128];
  for (int i = threadIdx.x; i < 8 * 128; i += blockDim.x) s[i] = i * 2654435761u;
  __syncthreads();
  const unsigned lane = threadIdx.x & 31, base = (unsigned)__cvta_generic_to_shared(s) + lane * 16;
  unsigned acc = threadIdx.x;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      unsigned r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                   : "r"(base + ((u + it) & 7) * 512));
      acc ^= r0 ^ r1 ^ r2 ^ r3;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) cyc[blockIdx.x] = clock64() - t0;
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

__global__ void lds128(unsigned* sink, long long* cyc, int iters) {
  __shared__ __align__(128) uint4 s[8 * 32];
  for (int i = threadIdx.x; i < 8 * 32; i += blockDim.x) s[i] = make_uint4(i, i * 3, i * 5, i * 7);
  __syncthreads();
  const unsigned lane = threadIdx.x & 31, base = (unsigned)__cvta_generic_to_shared(s) + lane * 16;
  unsigned acc = threadIdx.x;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      unsigned r0, r1, r2, r3;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                   : "r"(base + ((u + it) & 7) * 512));
      acc ^= r0 ^ r1 ^ r2 ^ r3;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) cyc[blockIdx.x] = clock64() - t0;
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

static float ev_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms;
  CK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

template <int TILE>
static void run_bulk(const char* name, int sms, int warps, int nt, int reps, bool* first) {
  size_t words = (size_t)sms * warps * nt * (TILE / 4);
  unsigned* g;
  CK(cudaMalloc(&g, words * 4));
  CK(cudaMemset(g, 0, words * 4));
  CK(cudaFuncSetAttribute(bulk_red<TILE>, cudaFuncAttributeMaxDynamicSharedMemorySize, warps * TILE));
  bulk_red<TILE><<<sms, warps * 32, warps * TILE>>>(g, nt, 1);  // warm-up and check: every word 1 or 2
  CK(cudaDeviceSynchronize());
  std::vector<unsigned> h(words);
  CK(cudaMemcpy(h.data(), g, words * 4, cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < words; i++) bad += h[i] != 1u + (unsigned)((i % (TILE / 4)) & 1);
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  bulk_red<TILE><<<sms, warps * 32, warps * TILE>>>(g, nt, reps);
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  double s = ev_ms(a, b) * 1e-3;
  CK(cudaMemcpy(h.data(), g, words * 4, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < words; i++) bad += h[i] != (1u + reps) * (1u + (unsigned)((i % (TILE / 4)) & 1));
  double wps = (double)words * reps / s;
  printf("%s\"%s\": {\"tile\": %d, \"sms\": %d, \"warps\": %d, \"region_mb\": %.1f, \"words_per_s\": %.6g, "
         "\"units_per_word\": %.2f, \"bad\": %zu}",
         *first ? "" : ", ", name, TILE, sms, warps, words * 4 / 1e6, wps, UNITS_PER_S / wps, bad);
  *first = false;
  CK(cudaFree(g));
}

template <typename K>
static void run_sm(const char* name, K kern, int sms, int tpb, int iters, int regs_per_inst, bool* first) {
  unsigned* sink;
  long long* cyc;
  CK(cudaMalloc(&sink, (size_t)sms * tpb * 4));
  CK(cudaMalloc(&cyc, sms * 8));
  kern<<<sms, tpb>>>(sink, cyc, 64);
  kern<<<sms, tpb>>>(sink, cyc, iters);
  CK(cudaDeviceSynchronize());
  std::vector<long long> c(sms);
  CK(cudaMemcpy(c.data(), cyc, sms * 8, cudaMemcpyDeviceToHost));
  double mean = 0;
  for (int i = 0; i < sms; i++) mean += c[i] / (double)sms;
  double words_per_clk = (double)(tpb / 32) * iters * 8 * 32 * regs_per_inst / mean;
  printf("%s\"%s\": {\"tpb\": %d, \"iters\": %d, \"cycles\": %.0f, \"words_per_sm_clk\": %.4f, "
         "\"units_per_word\": %.3f}",
         *first ? "" : ", ", name, tpb, iters, mean, words_per_clk, 4096.0 / words_per_clk);
  *first = false;
  CK(cudaFree(sink));
  CK(cudaFree(cyc));
}

int main() {
  int sms;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  bool first = true;
  printf("{");
  run_bulk<16384>("bulk_red_l2_16k", sms, 4, 2, 64, &first);     // 34.6 MB: L2-resident
  run_bulk<4096>("bulk_red_l2_4k", sms, 8, 4, 64, &first);       // 17.3 MB
  run_bulk<16384>("bulk_red_hbm_16k", sms, 4, 64, 4, &first);    // 553 MB: HBM
  run_sm("ldsm_x4", ldsm_x4, sms, 1024, 4096, 4, &first);
  run_sm("lds128", lds128, sms, 1024, 4096, 4, &first);
  {  // the copy engine and PCIe
    size_t n = 1ull << 30;
    void *h, *d0, *d1;
    CK(cudaMallocHost(&h, n));
    CK(cudaMalloc(&d0, n));
    CK(cudaMalloc(&d1, n));
    memset(h, 1, n);
    cudaStream_t s0, s1;
    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    cudaEvent_t a, b, c2, d2;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaEventCreate(&c2));
    CK(cudaEventCreate(&d2));
    CK(cudaMemcpy(d0, h, n, cudaMemcpyHostToDevice));
    const int R = 4;
    CK(cudaEventRecord(a, s0));
    for (int r = 0; r < R; r++) CK(cudaMemcpyAsync(d0, h, n, cudaMemcpyHostToDevice, s0));
    CK(cudaEventRecord(b, s0));
    CK(cudaEventSynchronize(b));
    double h2d = R * n / (ev_ms(a, b) * 1e-3);
    CK(cudaEventRecord(a, s0));
    for (int r = 0; r < R; r++) CK(cudaMemcpyAsync(h, d0, n, cudaMemcpyDeviceToHost, s0));
    CK(cudaEventRecord(b, s0));
    CK(cudaEventSynchronize(b));
    double d2h = R * n / (ev_ms(a, b) * 1e-3);
    CK(cudaEventRecord(a, s0));
    CK(cudaEventRecord(c2, s1));
    for (int r = 0; r < R; r++) {
      CK(cudaMemcpyAsync(d1, h, n / 2, cudaMemcpyHostToDevice, s0));
      CK(cudaMemcpyAsync((char*)h + n / 2, d0, n / 2, cudaMemcpyDeviceToHost, s1));
    }
    CK(cudaEventRecord(b, s0));
    CK(cudaEventRecord(d2, s1));
    CK(cudaEventSynchronize(b));
    CK(cudaEventSynchronize(d2));
    double bidir = R * n / (std::max(ev_ms(a, b), ev_ms(c2, d2)) * 1e-3);
    CK(cudaEventRecord(a, s0));
    for (int r = 0; r < R; r++) CK(cudaMemcpyAsync(d1, d0, n, cudaMemcpyDeviceToDevice, s0));
    CK(cudaEventRecord(b, s0));
    CK(cudaEventSynchronize(b));
    double d2d = R * n / (ev_ms(a, b) * 1e-3);
    unsigned char chk[16];
    CK(cudaMemcpy(chk, (char*)d1 + 12345, 16, cudaMemcpyDeviceToHost));
    int ok = 1;
    for (int i = 0; i < 16; i++) ok &= chk[i] == 1;
    printf(", \"pcie_h2d_bytes_per_s\": %.6g, \"pcie_d2h_bytes_per_s\": %.6g, \"pcie_bidir_bytes_per_s\": %.6g, "
           "\"d2d_copy_bytes_per_s\": %.6g, \"d2d_ok\": %d",
           h2d, d2h, bidir, d2d, ok);
  }
  printf("}\n");
  return 0;
}

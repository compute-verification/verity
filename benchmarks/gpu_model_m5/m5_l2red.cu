// Axiom probe: how many FP32 additions a second can the memory side (L2 reduction units, the TMA's bulk reduce) do with
// no CUDA-core add per element? Three routes into an L2-resident 32 MiB buffer:
//   red-f32:   one `red.global.add.f32` per element (atomicAdd, result unused), one issue slot per element;
//   red-v4:    `atomicAdd(float4*)` (sm_90+), one issue slot per 4 elements;
//   bulk-f32:  `cp.reduce.async.bulk...add.f32` of a 16 KiB shared-memory chunk, one issue slot per 4,096 elements.
// Prints elements added per second and per SM per clock (clock from clock64 over the kernel), best of 5.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

constexpr int NBUF = 8 << 20;  // floats: 32 MiB, inside the 128 MB L2
constexpr int CHUNK = 4096;    // floats: 16 KiB per bulk reduce

__global__ void k_red(float* buf, int iters, long long* cyc) {
  long long t0 = clock64();
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x, nth = gridDim.x * blockDim.x;
  for (int i = 0; i < iters; i++) atomicAdd(&buf[(tid + (unsigned)i * nth) & (NBUF - 1)], 1.0f);
  if (threadIdx.x == 0) atomicMax((unsigned long long*)cyc, (unsigned long long)(clock64() - t0));
}

__global__ void k_red4(float4* buf, int iters, long long* cyc) {
  long long t0 = clock64();
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x, nth = gridDim.x * blockDim.x;
  for (int i = 0; i < iters; i++)
    atomicAdd(&buf[(tid + (unsigned)i * nth) & (NBUF / 4 - 1)], make_float4(1.f, 1.f, 1.f, 1.f));
  if (threadIdx.x == 0) atomicMax((unsigned long long*)cyc, (unsigned long long)(clock64() - t0));
}

__global__ void k_bulk(float* buf, int iters, long long* cyc) {
  __shared__ __align__(128) float s[CHUNK];
  for (int i = threadIdx.x; i < CHUNK; i += blockDim.x) s[i] = 1.0f;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  long long t0 = clock64();
  if (threadIdx.x == 0) {
    unsigned saddr = (unsigned)__cvta_generic_to_shared(s);
    unsigned nchunks = NBUF / CHUNK;
    for (int i = 0; i < iters; i++) {
      float* dst = buf + (size_t)((blockIdx.x + (unsigned)i * gridDim.x) % nchunks) * CHUNK;
      asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;"
                   :: "l"(dst), "r"(saddr), "r"(CHUNK * 4) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      if ((i & 7) == 7) asm volatile("cp.async.bulk.wait_group.read 4;" ::: "memory");
    }
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    atomicMax((unsigned long long*)cyc, (unsigned long long)(clock64() - t0));
  }
}

int main() {
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  float* buf;
  long long* cyc;
  CK(cudaMalloc(&buf, (size_t)NBUF * 4));
  CK(cudaMalloc(&cyc, 8));
  CK(cudaMemset(buf, 0, (size_t)NBUF * 4));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int route = 0; route < 3; route++) {
    double best_s = 1e30, best_cyc = 0, elems = 0;
    for (int rep = 0; rep < 6; rep++) {
      CK(cudaMemset(cyc, 0, 8));
      int blocks, threads, iters;
      CK(cudaEventRecord(e0));
      if (route == 0) { blocks = sms * 8; threads = 256; iters = 256; k_red<<<blocks, threads>>>(buf, iters, cyc);
                        elems = (double)blocks * threads * iters; }
      else if (route == 1) { blocks = sms * 8; threads = 256; iters = 128; k_red4<<<blocks, threads>>>((float4*)buf, iters, cyc);
                             elems = 4.0 * blocks * threads * iters; }
      else { blocks = sms * 4; threads = 128; iters = 256; k_bulk<<<blocks, threads>>>(buf, iters, cyc);
             elems = (double)blocks * iters * CHUNK; }
      CK(cudaEventRecord(e1));
      CK(cudaGetLastError()); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      long long c; CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
      if (rep > 0 && ms / 1e3 < best_s) { best_s = ms / 1e3; best_cyc = (double)c; }
    }
    const char* name = route == 0 ? "red-f32" : route == 1 ? "red-v4-f32" : "bulk-reduce-f32";
    double clk_hz = best_cyc / best_s;  // the slowest CTA's cycles over the event time: a lower bound on the clock
    printf("{\"probe\":\"l2-reduce\",\"route\":\"%s\",\"elements\":%.0f,\"seconds\":%.6e,\"adds_per_s\":%.4e,"
           "\"kernel_cycles\":%.0f,\"adds_per_sm_per_clk\":%.3f,\"sms\":%d}\n",
           name, elems, best_s, elems / best_s, best_cyc, elems / best_cyc / sms, sms);
    (void)clk_hz;
  }
  return 0;
}

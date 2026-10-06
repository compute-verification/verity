// Latency and throughput probes that bound a dependent 16,448-bit squaring chain on one GPU: the device's SM count and
// clocks, the largest thread-block cluster it launches, IMAD.WIDE throughput on one SM, a cluster barrier's latency, a
// DSMEM ping-pong between two CTAs of a cluster, and a ping-pong through L2 between two CTAs on different SMs.
// Times are device clocks (clock64, SM cycles) and %globaltimer (ns). Output: one JSON object per line.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ uint64_t gtime() { uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// IMAD.WIDE throughput: each thread runs 8 independent 64-bit accumulators.
__global__ void imad_wide(uint64_t *out, uint32_t iters, uint64_t *cyc) {
  uint32_t a = threadIdx.x * 2654435761u + 1, b = a ^ 0x9e3779b9u;
  uint64_t acc[8] = {0};
  __syncthreads();
  uint64_t t0 = clock64();
  for (uint32_t i = 0; i < iters; i++) {
    #pragma unroll
    for (int k = 0; k < 8; k++) acc[k] += (uint64_t)(a + k) * (b ^ i);
  }
  __syncthreads();
  uint64_t t1 = clock64();
  uint64_t s = 0;
  for (int k = 0; k < 8; k++) s ^= acc[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

// 32-bit IMAD (mul.lo) throughput, same shape.
__global__ void imad_lo(uint32_t *out, uint32_t iters, uint64_t *cyc) {
  uint32_t a = threadIdx.x * 2654435761u + 1, b = a ^ 0x9e3779b9u;
  uint32_t acc[8] = {0};
  __syncthreads();
  uint64_t t0 = clock64();
  for (uint32_t i = 0; i < iters; i++) {
    #pragma unroll
    for (int k = 0; k < 8; k++) acc[k] += (a + k) * (b ^ i);
  }
  __syncthreads();
  uint64_t t1 = clock64();
  uint32_t s = 0;
  for (int k = 0; k < 8; k++) s ^= acc[k];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

// cluster barrier latency: iters back-to-back barrier.cluster.arrive + wait
__global__ void cluster_bar(uint32_t iters, uint64_t *cyc, uint64_t *ns) {
  cg::cluster_group cl = cg::this_cluster();
  cl.sync();
  uint64_t t0 = clock64(), g0 = gtime();
  for (uint32_t i = 0; i < iters; i++) cl.sync();
  uint64_t t1 = clock64(), g1 = gtime();
  if (threadIdx.x == 0 && cl.block_rank() == 0) { cyc[0] = t1 - t0; ns[0] = g1 - g0; }
}

// DSMEM ping-pong: rank 0 writes a counter into rank 1's smem, rank 1 echoes it back into rank 0's.
__global__ void dsmem_pingpong(uint32_t iters, uint64_t *cyc, uint64_t *ns) {
  __shared__ volatile uint32_t box;
  cg::cluster_group cl = cg::this_cluster();
  box = 0;
  cl.sync();
  uint32_t me = cl.block_rank();
  volatile uint32_t *peer = cl.map_shared_rank((uint32_t *)&box, me ^ 1);
  uint64_t t0 = clock64(), g0 = gtime();
  if (threadIdx.x == 0 && me < 2) {
    for (uint32_t i = 1; i <= iters; i++) {
      if (me == 0) {
        *peer = i;
        while (box != i) {}
      } else {
        while (box != i) {}
        *peer = i;
      }
    }
  }
  uint64_t t1 = clock64(), g1 = gtime();
  cl.sync();
  if (threadIdx.x == 0 && me == 0) { cyc[0] = t1 - t0; ns[0] = g1 - g0; }
}

// L2 ping-pong between block 0 and block 1 (on different SMs) through a global word each.
__global__ void l2_pingpong(volatile uint32_t *flags, uint32_t iters, uint64_t *cyc, uint64_t *ns, uint32_t *smid) {
  uint32_t s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  if (threadIdx.x) return;
  smid[blockIdx.x] = s;
  uint32_t me = blockIdx.x;
  volatile uint32_t *mine = flags + 32 * me, *theirs = flags + 32 * (me ^ 1);
  uint64_t t0 = clock64(), g0 = gtime();
  for (uint32_t i = 1; i <= iters; i++) {
    if (me == 0) {
      *theirs = i; __threadfence();
      while (*mine != i) {}
    } else {
      while (*mine != i) {}
      *theirs = i; __threadfence();
    }
  }
  uint64_t t1 = clock64(), g1 = gtime();
  if (me == 0) { cyc[0] = t1 - t0; ns[0] = g1 - g0; }
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  int clk = 0, mclk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
  CK(cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0));
  printf("{\"probe\":\"device\",\"name\":\"%s\",\"cc\":\"%d.%d\",\"sms\":%d,\"clock_khz\":%d,\"smem_per_block_optin\":%zu,"
         "\"smem_per_sm\":%zu,\"regs_per_sm\":%d,\"l2_bytes\":%d}\n",
         p.name, p.major, p.minor, p.multiProcessorCount, clk, p.sharedMemPerBlockOptin, p.sharedMemPerMultiprocessor,
         p.regsPerMultiprocessor, p.l2CacheSize);

  uint64_t *dcyc, *dns, *dout; uint32_t *dout32, *dsm;
  CK(cudaMalloc(&dcyc, 4096 * 8)); CK(cudaMalloc(&dns, 64)); CK(cudaMalloc(&dout, 1 << 24)); CK(cudaMalloc(&dout32, 1 << 22));
  CK(cudaMalloc(&dsm, 64));
  uint64_t cyc, ns;

  for (int tpb : {128, 256, 512, 1024}) {
    const uint32_t it = 4096;
    imad_wide<<<1, tpb>>>(dout, it, dcyc); CK(cudaDeviceSynchronize());
    imad_wide<<<1, tpb>>>(dout, it, dcyc); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&cyc, dcyc, 8, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"imad_wide\",\"tpb\":%d,\"per_sm_per_clk\":%.2f}\n", tpb, (double)tpb * it * 8 / cyc);
    imad_lo<<<1, tpb>>>(dout32, it, dcyc); CK(cudaDeviceSynchronize());
    imad_lo<<<1, tpb>>>(dout32, it, dcyc); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&cyc, dcyc, 8, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"imad_lo\",\"tpb\":%d,\"per_sm_per_clk\":%.2f}\n", tpb, (double)tpb * it * 8 / cyc);
  }

  for (int cs : {2, 4, 8, 16}) {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = cs; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(cs); cfg.blockDim = dim3(256); cfg.attrs = at; cfg.numAttrs = 1;
    if (cs > 8) CK(cudaFuncSetAttribute(cluster_bar, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    if (cs > 8) CK(cudaFuncSetAttribute(dsmem_pingpong, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    int maxc = 0;
    cudaError_t e = cudaOccupancyMaxPotentialClusterSize(&maxc, cluster_bar, &cfg);
    int nclusters = 0;
    cudaError_t e2 = cudaOccupancyMaxActiveClusters(&nclusters, cluster_bar, &cfg);
    const uint32_t it = 20000;
    e = cudaLaunchKernelEx(&cfg, cluster_bar, it, dcyc, dns);
    if (e != cudaSuccess) { printf("{\"probe\":\"cluster_bar\",\"cluster\":%d,\"error\":\"%s\",\"max_potential\":%d}\n", cs, cudaGetErrorString(e), maxc); cudaGetLastError(); continue; }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&cyc, dcyc, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&ns, dns, 8, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"cluster_bar\",\"cluster\":%d,\"max_potential\":%d,\"max_active_clusters\":%d,\"cyc_per\":%.1f,\"ns_per\":%.2f%s}\n",
           cs, maxc, nclusters, (double)cyc / it, (double)ns / it, e2 == cudaSuccess ? "" : ",\"occ_err\":1");
    CK(cudaLaunchKernelEx(&cfg, dsmem_pingpong, it, dcyc, dns)); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&cyc, dcyc, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&ns, dns, 8, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"dsmem_pingpong\",\"cluster\":%d,\"roundtrip_cyc\":%.1f,\"roundtrip_ns\":%.2f}\n", cs, (double)cyc / it, (double)ns / it);
  }

  uint32_t *dflags; CK(cudaMalloc(&dflags, 4096)); CK(cudaMemset(dflags, 0, 4096));
  for (int rep = 0; rep < 2; rep++) {
    CK(cudaMemset(dflags, 0, 4096));
    const uint32_t it = 20000;
    l2_pingpong<<<2, 32>>>(dflags, it, dcyc, dns, dsm); CK(cudaDeviceSynchronize());
    uint32_t sm[2]; CK(cudaMemcpy(sm, dsm, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&cyc, dcyc, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&ns, dns, 8, cudaMemcpyDeviceToHost));
    printf("{\"probe\":\"l2_pingpong\",\"sms\":[%u,%u],\"roundtrip_cyc\":%.1f,\"roundtrip_ns\":%.2f}\n", sm[0], sm[1], (double)cyc / it, (double)ns / it);
  }
  return 0;
}

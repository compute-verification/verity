// The collector's gather (integrations/vllm/verity_vllm/commit/committer/native_gather.cu) on an idle GPU, over one Qwen3-8B prefill
// layer's staged tensors (2,560 tokens; out/layouts.jsonl of a served_commit run), against its variants: the descriptors read from
// host-mapped memory by every warp (the committer's), read once a block into shared memory, or copied to device memory first; and
// cudaMemcpyAsync a tensor.  Measurement only.
//
//   nvcc -O3 -arch=native -o /tmp/gather_probe benchmarks/hash/loop/gather_probe.cu && /tmp/gather_probe [blocks_per_desc]
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct GatherDesc {
  const char* src;
  char* dst;
  int64_t nbytes, width, pitch, skip;
};

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e = (x);                                                              \
    if (e != cudaSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));  \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

__device__ __forceinline__ void copy_aligned(const GatherDesc& g) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x, nthr = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const uint4* s = reinterpret_cast<const uint4*>(g.src);
  uint4* t = reinterpret_cast<uint4*>(g.dst);
  for (int64_t i = tid; i < g.nbytes / 16; i += nthr) t[i] = s[i];
}

__global__ void per_warp(const GatherDesc* __restrict__ d) {
  const GatherDesc g = d[blockIdx.y];
  copy_aligned(g);
}

__global__ void per_block(const GatherDesc* __restrict__ d) {
  __shared__ GatherDesc g;
  if (threadIdx.x == 0) g = d[blockIdx.y];
  __syncthreads();
  copy_aligned(g);
}

int main(int argc, char** argv) {
  const int bpd = argc > 1 ? std::atoi(argv[1]) : 1024;
  const std::vector<int64_t> sizes = {20971520, 20971520, 125829120, 62914560, 20971520, 20971520, 20971520, 31457280, 20971520, 5242880, 20971520};
  int64_t total = 0;
  for (int64_t s : sizes) total += s;
  char *src, *dst;
  CK(cudaMalloc(&src, total));
  CK(cudaMalloc(&dst, total));
  CK(cudaMemset(src, 7, total));
  const int n = static_cast<int>(sizes.size());
  GatherDesc* host;
  CK(cudaHostAlloc(&host, n * sizeof(GatherDesc), cudaHostAllocMapped | cudaHostAllocPortable));
  GatherDesc* mapped;
  CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped), host, 0));
  int64_t off = 0;
  for (int i = 0; i < n; ++i) {
    host[i] = {src + off, dst + off, sizes[i], sizes[i], sizes[i], 0};
    off += sizes[i];
  }
  GatherDesc* dev;
  CK(cudaMalloc(&dev, n * sizeof(GatherDesc)));
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  const dim3 grid(bpd, n);
  auto timeit = [&](const char* name, auto&& fn) {
    for (int w = 0; w < 3; ++w) fn();
    CK(cudaStreamSynchronize(st));
    const int reps = 20;
    CK(cudaEventRecord(a, st));
    for (int r = 0; r < reps; ++r) fn();
    CK(cudaEventRecord(b, st));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    ms /= reps;
    std::printf("%-44s %8.3f ms  %7.0f GB/s copied\n", name, ms, total / (ms * 1e6));
  };
  std::printf("%d tensors, %.1f MB, %d blocks a descriptor\n", n, total / 1e6, bpd);
  timeit("descriptors host-mapped, read by every warp", [&] { per_warp<<<grid, 256, 0, st>>>(mapped); });
  timeit("descriptors host-mapped, read once a block", [&] { per_block<<<grid, 256, 0, st>>>(mapped); });
  timeit("descriptors copied to device, every warp", [&] {
    cudaMemcpyAsync(dev, host, n * sizeof(GatherDesc), cudaMemcpyHostToDevice, st);
    per_warp<<<grid, 256, 0, st>>>(dev);
  });
  timeit("cudaMemcpyAsync a tensor", [&] {
    for (int i = 0; i < n; ++i) cudaMemcpyAsync(host[i].dst, host[i].src, sizes[i], cudaMemcpyDeviceToDevice, st);
  });
  CK(cudaGetLastError());
  return 0;
}

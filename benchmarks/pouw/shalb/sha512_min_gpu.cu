// The minimal compression on the GPU against the host: every thread chains NBLOCKS compressions of its own blocks from
// the IV through shamin::compress<0>, and also compresses its first block from the IV with compress<1> and its state
// through the padding form compress<0, 2048>; the committer's sha::compress chains the same blocks, and so does the
// chain form (compress<0, 0, 1> and <0, 0, 2>, two blocks a trip). Every result must
// equal the host's (sha512_min.cuh compiled for the host, which sha512_min_check holds to hashlib). Then each chain
// kernel is timed (CUDA events, median of REPS launches), giving ns a compression across the GPU.
//
//   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_90a -I<committer> sha512_min_gpu.cu -o gpu && ./gpu
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sha512.cuh"
#include "sha512_min.cuh"

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e = (x);                                                              \
    if (e != cudaSuccess) {                                                           \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));       \
      exit(2);                                                                        \
    }                                                                                 \
  } while (0)

constexpr int NBLOCKS = 32, REPS = 9;

template <int KIND>
__global__ void chain(const uint64_t* __restrict__ in, uint64_t* __restrict__ out, int nthreads) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nthreads) return;
  uint64_t st[8];
  for (int i = 0; i < 8; ++i) st[i] = shamin::IVt(i);
  const uint64_t* p = in + (size_t)t * 16 * NBLOCKS;
  if constexpr (KIND == 2) {
    st[7] += shamin::Kt(0);
#pragma unroll 1
    for (int b = 0; b < NBLOCKS; b += 2) {
      uint64_t w0[16], w1[16], pend[2];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        w0[i] = p[b * 16 + i];
        w1[i] = p[b * 16 + 16 + i];
      }
      shamin::compress<0, 0, 1>(st, w0, pend);
      shamin::compress<0, 0, 2>(st, w1, pend);
    }
    st[7] -= shamin::Kt(0);
  } else {
#pragma unroll 1
    for (int b = 0; b < NBLOCKS; ++b) {
      uint64_t w[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) w[i] = p[b * 16 + i];
      if constexpr (KIND == 0) {
        shamin::compress<0>(st, w);
      } else {
        sha::compress(st, w);
      }
    }
  }
  for (int i = 0; i < 8; ++i) out[(size_t)t * 8 + i] = st[i];
}

static const char* const KINDS[] = {"minimal", "committer", "minimal chain form"};

static void launch(int kind, int grid, int threads, const uint64_t* in, uint64_t* out, int nthreads) {
  if (kind == 0) chain<0><<<grid, threads>>>(in, out, nthreads);
  else if (kind == 1) chain<1><<<grid, threads>>>(in, out, nthreads);
  else chain<2><<<grid, threads>>>(in, out, nthreads);
}

__global__ void forms(const uint64_t* __restrict__ in, uint64_t* __restrict__ out, int nthreads) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nthreads) return;
  uint64_t st[8], w[16];
  for (int i = 0; i < 16; ++i) w[i] = in[(size_t)t * 16 * NBLOCKS + i];
  shamin::compress<1>(st, w);
  for (int i = 0; i < 8; ++i) out[(size_t)t * 16 + i] = st[i];
  shamin::compress<0, 2048>(st, nullptr);
  for (int i = 0; i < 8; ++i) out[(size_t)t * 16 + 8 + i] = st[i];
}

int main() {
  int dev = 0, sms = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  const int threads = 256, nthreads = sms * 8 * threads;
  const size_t words = (size_t)nthreads * 16 * NBLOCKS;
  std::vector<uint64_t> h_in(words);
  uint64_t s = 0x9e3779b97f4a7c15ull;
  for (auto& v : h_in) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    v = s;
  }
  uint64_t *d_in, *d_out;
  CK(cudaMalloc(&d_in, words * 8));
  CK(cudaMalloc(&d_out, (size_t)nthreads * 16 * 8));
  CK(cudaMemcpy(d_in, h_in.data(), words * 8, cudaMemcpyHostToDevice));
  std::vector<uint64_t> got((size_t)nthreads * 16);
  const int grid = (nthreads + threads - 1) / threads;
  int bad_all = 0;
  const int checked = std::min(nthreads, 4096);
  for (int kind = 0; kind < 3; ++kind) {
    int bad = 0;
    launch(kind, grid, threads, d_in, d_out, nthreads);
    CK(cudaGetLastError());
    CK(cudaMemcpy(got.data(), d_out, (size_t)nthreads * 8 * 8, cudaMemcpyDeviceToHost));
    for (int t = 0; t < checked; ++t) {
      uint64_t st[8], w[16];
      for (int i = 0; i < 8; ++i) st[i] = shamin::IVt(i);
      for (int b = 0; b < NBLOCKS; ++b) {
        for (int i = 0; i < 16; ++i) w[i] = h_in[(size_t)t * 16 * NBLOCKS + b * 16 + i];
        shamin::compress<0>(st, w);
      }
      for (int i = 0; i < 8; ++i) bad += got[(size_t)t * 8 + i] != st[i];
    }
    printf("chain %s: %d threads checked against the host, %d words differ\n", KINDS[kind], checked, bad);
    bad_all += bad;
  }
  forms<<<grid, threads>>>(d_in, d_out, nthreads);
  CK(cudaGetLastError());
  CK(cudaMemcpy(got.data(), d_out, (size_t)nthreads * 16 * 8, cudaMemcpyDeviceToHost));
  int bad_forms = 0;
  for (int t = 0; t < checked; ++t) {
    uint64_t st[8], w[16];
    for (int i = 0; i < 16; ++i) w[i] = h_in[(size_t)t * 16 * NBLOCKS + i];
    shamin::compress<1>(st, w);
    for (int i = 0; i < 8; ++i) bad_forms += got[(size_t)t * 16 + i] != st[i];
    shamin::compress<0, 2048>(st, nullptr);
    for (int i = 0; i < 8; ++i) bad_forms += got[(size_t)t * 16 + 8 + i] != st[i];
  }
  printf("forms (constant chaining value, padding block): %d threads checked, %d words differ\n", checked, bad_forms);
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  for (int kind = 0; kind < 3; ++kind) {
    std::vector<float> ms;
    for (int r = 0; r < REPS + 1; ++r) {
      CK(cudaEventRecord(a));
      launch(kind, grid, threads, d_in, d_out, nthreads);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float m;
      CK(cudaEventElapsedTime(&m, a, b));
      if (r) ms.push_back(m);
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2], comp = (double)nthreads * NBLOCKS;
    printf("time %s: median %.3f ms over %d launches, %.0f compressions, %.4f ns a compression, %d SMs\n",
           KINDS[kind], med, REPS, comp, med * 1e6 / comp, sms);
  }
  return bad_all || bad_forms ? 1 : 0;
}

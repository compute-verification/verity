// RT4-SHEET on the H200 and B200: li's multiply-add as IMAD.WIDE. li's half step is y = (y ^ p0 ^ k)·a + p1 mod 2^32,
// one LOP3 and one IMAD (cpatom.cuh on cursor/ca-rt3-dc-ea24 @ 0c9b5adfac). That y is the low word of
// `mad.wide.u32 d, u, a, {p1, h}` for any high word h, since h·2^32 leaves the low word alone. `loads.class_of`
// classes IMAD.WIDE as IMADW, which neither DC census vector has a line for, and DC has no catch-all, so W = 1 prices
// li's IMADs at 0 there. Each chain keeps its high word live (the next step's addend high, stored at the end to a
// sink), so ptxas can't narrow the wide multiply-add back to IMAD. The checkpoint is red_generic.cu's (CP_RED2
// `.global`, the DC menu's route; CP_RED8 generic, RT4-SHEET-7's).
//
// `main` runs li_exact<CP, GEN, W> (one warp, U units, every checkpoint and final state against the host's li
// reference). li_loop<CP, GEN, W> is the timed loop red_price.py prices: opaque words, 4 units an iteration.
#include <stdint.h>
#include <stdio.h>
#include <vector>

__host__ __device__ constexpr uint32_t li_mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  return x ^ (x >> 16);
}
__host__ __device__ constexpr uint32_t li_a(int c, int h) { return li_mix32(0xA0000000u + 2u * (32 + 2 * c + h)) | 1u; }
__host__ __device__ constexpr uint32_t li_k(int c, int h) { return li_mix32(0xB0000000u + 2u * (32 + 2 * c + h) + 1u); }
__host__ __device__ constexpr int li_entry(int c, int i) { return 4 * (c / 4 + 2 * i) + c % 4; }

// the host's reference: cpatom.cuh's li set, narrow
inline void li_set_ref(uint32_t (&y)[8], const uint32_t* p) {
  for (int c = 0; c < 8; c++)
    for (int h = 0; h < 2; h++)
      y[c] = (y[c] ^ p[li_entry(c, 2 * h)] ^ li_k(c, h)) * li_a(c, h) + p[li_entry(c, 2 * h + 1)];
}

// li's set on the device: W = 0 the narrow IMAD, W = 1 the low word of a wide multiply-add whose high word chains in hw
template <int W>
__device__ __forceinline__ void li_set(uint32_t (&y)[8], uint32_t (&hw)[8], const uint32_t* p) {
#pragma unroll
  for (int c = 0; c < 8; c++) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const uint32_t u = y[c] ^ p[li_entry(c, 2 * h)] ^ li_k(c, h);
      const uint32_t p1 = p[li_entry(c, 2 * h + 1)];
      if constexpr (W) {
        uint64_t cc, d;
        asm("mov.b64 %0, {%1, %2};" : "=l"(cc) : "r"(p1), "r"(hw[c]));
        asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(d) : "r"(u), "r"(li_a(c, h)), "l"(cc));
        asm("mov.b64 {%0, %1}, %2;" : "=r"(y[c]), "=r"(hw[c]) : "l"(d));
      } else {
        y[c] = u * li_a(c, h) + p1;
      }
    }
  }
}

template <int GEN>
__device__ __forceinline__ void red_xor(uint32_t* p, uint32_t v) {
  if constexpr (GEN)
    asm volatile("red.relaxed.gpu.xor.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
  else
    asm volatile("red.relaxed.gpu.global.xor.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t lop3x(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

enum { CP_RED2 = 1, CP_RED8 = 2 };

// one segment's checkpoint into gw (which carries lane & zm, zm 0 at run time, so ptxas can't prove it warp-uniform)
template <int CP, int GEN>
__device__ __forceinline__ void cp_seg(const uint32_t (&y)[8], uint32_t* gw) {
  if constexpr (CP == CP_RED2) {
    red_xor<GEN>(gw, lop3x(lop3x(lop3x(y[0], y[1], y[2]), y[3], y[4]), y[5], y[6]));
    red_xor<GEN>(gw, y[7]);
  }
  if constexpr (CP == CP_RED8) {
#pragma unroll
    for (int c = 0; c < 8; c++) red_xor<GEN>(gw, y[c]);
  }
}

// one warp: words U units x 32 lanes x 128 (lane-major in a unit), st 32 lanes x 8 states; out (zeroed) U checkpoints,
// then each lane's final 8 states, then the 8 high words to a sink the reference doesn't read
template <int CP, int GEN, int W>
__global__ void __launch_bounds__(32, 1) li_exact(const uint4* __restrict__ words, const uint32_t* st, uint32_t* out,
                                                 int U, uint32_t zm) {
  const int lane = threadIdx.x;
  uint32_t* const gw0 = out + (lane & zm);
  uint32_t y[8], hw[8];
#pragma unroll
  for (int c = 0; c < 8; c++) y[c] = st[8 * lane + c], hw[c] = 0u;
#pragma unroll 1
  for (int u = 0; u < U; u++) {
    const uint4* src = words + ((size_t)u * 32 + lane) * 32;
#pragma unroll
    for (int s = 0; s < 4; s++) {
      uint32_t p[32];
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const uint4 v = src[8 * s + q];
        p[4 * q] = v.x, p[4 * q + 1] = v.y, p[4 * q + 2] = v.z, p[4 * q + 3] = v.w;
      }
      li_set<W>(y, hw, p);
    }
    cp_seg<CP, GEN>(y, gw0 + u);
  }
#pragma unroll
  for (int c = 0; c < 8; c++) out[U + 8 * lane + c] = y[c], out[U + 256 + 8 * lane + c] = hw[c];
}

// the timed loop: a set's 32 words opaque registers each set (the tensor's accumulators), 4 units an iteration
template <int CP, int GEN, int W>
__global__ void __launch_bounds__(128) li_loop(const uint32_t* st, uint32_t* out, uint32_t* cpw, int iters, uint32_t zm) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31;
  uint32_t y[8], hw[8], p[32];
#pragma unroll
  for (int c = 0; c < 8; c++) y[c] = st[(8 * t + c) & 255], hw[c] = st[(8 * t + c + 1) & 255];
#pragma unroll
  for (int i = 0; i < 32; i++) p[i] = st[(t + 7 * i) & 255];
  uint32_t* gw = cpw + 4 * (t >> 5) * iters + (lane & zm);
#pragma unroll 1
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
      for (int s = 0; s < 4; s++) {
#pragma unroll
        for (int i = 0; i < 32; i++) asm volatile("" : "+r"(p[i]));
        li_set<W>(y, hw, p);
      }
      cp_seg<CP, GEN>(y, gw + j);
    }
    gw += 4;
  }
#pragma unroll
  for (int c = 0; c < 8; c++) out[8 * t + c] = y[c], out[2048 + 8 * t + c] = hw[c];
}

#define INST(CP, GEN, W)                                                                                              \
  template __global__ void li_exact<CP, GEN, W>(const uint4*, const uint32_t*, uint32_t*, int, uint32_t);             \
  template __global__ void li_loop<CP, GEN, W>(const uint32_t*, uint32_t*, uint32_t*, int, uint32_t);
INST(CP_RED2, 0, 0)
INST(CP_RED2, 0, 1)
INST(CP_RED8, 1, 0)
INST(CP_RED8, 1, 1)

#ifndef __CUDA_ARCH__
static uint32_t lcg(uint64_t& s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(s >> 32);
}

typedef void (*Exact)(const uint4*, const uint32_t*, uint32_t*, int, uint32_t);

int main() {
  int dev = 0, maj = 0, mnr = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev);
  printf("device cc=%d.%d\n", maj, mnr);
  const int U = 64;
  const size_t nw = (size_t)U * 32 * 128;
  std::vector<uint32_t> w(nw), st(256);
  uint64_t seed = 0x5EED2026ull;
  for (auto& x : w) x = lcg(seed);
  for (auto& x : st) x = lcg(seed);
  std::vector<uint32_t> cp(U), y(st);
  for (int u = 0; u < U; u++) {
    uint32_t x = 0;
    for (int lane = 0; lane < 32; lane++) {
      uint32_t yy[8];
      for (int c = 0; c < 8; c++) yy[c] = y[8 * lane + c];
      for (int s = 0; s < 4; s++) li_set_ref(yy, &w[((size_t)u * 32 + lane) * 128 + 32 * s]);
      for (int c = 0; c < 8; c++) y[8 * lane + c] = yy[c], x ^= yy[c];
    }
    cp[u] = x;
  }
  uint32_t *dw, *dst, *dout;
  cudaMalloc(&dw, nw * 4);
  cudaMalloc(&dst, 256 * 4);
  cudaMalloc(&dout, (U + 512) * 4);
  cudaMemcpy(dw, w.data(), nw * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dst, st.data(), 256 * 4, cudaMemcpyHostToDevice);
  const struct { const char* name; Exact k; } ks[] = {
      {"red2 global narrow", li_exact<CP_RED2, 0, 0>}, {"red2 global wide", li_exact<CP_RED2, 0, 1>},
      {"red8 generic narrow", li_exact<CP_RED8, 1, 0>}, {"red8 generic wide", li_exact<CP_RED8, 1, 1>}};
  int bad = 0;
  for (auto& e : ks) {
    cudaMemset(dout, 0, (U + 512) * 4);
    e.k<<<1, 32>>>((const uint4*)dw, dst, dout, U, 0u);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    std::vector<uint32_t> got(U + 512);
    cudaMemcpy(got.data(), dout, (U + 512) * 4, cudaMemcpyDeviceToHost);
    int mism = err != cudaSuccess;
    for (int u = 0; u < U; u++) mism += got[u] != cp[u];
    for (int i = 0; i < 256; i++) mism += got[U + i] != y[i];
    printf("li_exact<%s> units=%d words_checked=%d mismatches=%d err=%s\n", e.name, U, U + 256, mism,
           cudaGetErrorString(err));
    bad += mism != 0;
  }
  printf("all_bitexact=%d\n", bad == 0);
  return bad ? 3 : 0;
}
#endif

// What a squaring step's phase 1 costs in isolation: one CTA of W warps, each warp one column pair's 315 products over a
// 633-digit vector in shared memory (as sqchain4), then a shared store and __syncthreads, repeated N times. Variants:
//   0  barrier only (the store and __syncthreads)
//   1  interleaved slices, runtime trip counts (sqchain4's run_pair_il)
//   2  interleaved slices, a fixed 10 iterations, fully unrolled, guarded by j < T
//   3  as 2, with the vector re-read through a volatile pointer each step (no hoisting of loads across steps)
// Also the throughput of warp-wide conflict-free shared loads of 4, 8 and 16 bytes per lane (ldsbw).
// Device cycles (clock64) per step. Output: JSON lines.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

static const int ND = 633;
__device__ __forceinline__ uint64_t madw(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r; asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(c)); return r;
}
__host__ __device__ __forceinline__ int floordiv2(int x) { return x >= 0 ? x / 2 : -((1 - x) / 2); }
__host__ __device__ __forceinline__ int len_lo(int c) { return floordiv2(c - 1) + 1; }
__host__ __device__ __forceinline__ int me_hi(int k) { return floordiv2(k + ND - 1); }

template <int V>
__global__ void k(const uint32_t *xin, int nsteps, int k0, unsigned long long *out) {
  __shared__ uint32_t D[64 + 640 + 96];
  __shared__ uint64_t R[1024];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  for (int q = tid; q < 800; q += blockDim.x) { int x = q - 64; D[q] = (x >= 0 && x < ND) ? xin[x] : 0; }
  const uint32_t *cur = D + 64;
  const int kk = (k0 + 2 * w) % ND, L0 = len_lo(kk), T = L0 + max(0, me_hi(kk) - (kk + 1));
  const int tA = L0 >> 5, tB = (L0 + 31) >> 5, tE = (T + 31) >> 5, hoff = kk + 2 - L0, kh = kk + ND;
  __syncthreads();
  uint64_t acc = 0;
  unsigned long long c0 = clock64();
  #pragma unroll 1
  for (int s = 0; s < nsteps; s++) {
    uint64_t l0 = 0, l1 = 0, h0 = 0, h1 = 0;
    if (V == 1) {
      #pragma unroll 2
      for (int t = 0; t < tA; t++) {
        const int i = lane + 32 * t;
        const uint32_t di = cur[i], w0 = cur[kk - i], w1 = cur[kk + 1 - i];
        l0 = madw(di, w0, l0); l1 = madw(di, w1, l1);
      }
      if (tA < tB) {
        const int j = lane + 32 * tA;
        if (j < L0) { const uint32_t di = cur[j]; l0 = madw(di, cur[kk - j], l0); l1 = madw(di, cur[kk + 1 - j], l1); }
        else if (j < T) { const int i = j + hoff; const uint32_t di = cur[i]; h0 = madw(di, cur[kh - i], h0); h1 = madw(di, cur[kh + 1 - i], h1); }
      }
      #pragma unroll 2
      for (int t = tB; t < tE; t++) {
        const int j = lane + 32 * t, i = j + hoff;
        if (j < T) { const uint32_t di = cur[i], w0 = cur[kh - i], w1 = cur[kh + 1 - i]; h0 = madw(di, w0, h0); h1 = madw(di, w1, h1); }
      }
    } else if (V == 2 || V == 3) {
      const volatile uint32_t *vc = cur;
      #pragma unroll
      for (int t = 0; t < 10; t++) {
        const int j = lane + 32 * t;
        const bool lo = j < L0;
        const int i = lo ? j : j + hoff, b = lo ? kk : kh;
        uint32_t di, w0, w1;
        if (V == 3) { di = vc[i]; w0 = vc[b - i]; w1 = vc[b + 1 - i]; } else { di = cur[i]; w0 = cur[b - i]; w1 = cur[b + 1 - i]; }
        di = j < T ? di : 0;
        if (lo) { l0 = madw(di, w0, l0); l1 = madw(di, w1, l1); } else { h0 = madw(di, w0, h0); h1 = madw(di, w1, h1); }
      }
    }
    R[tid] = l0 + l1 + h0 + h1 + acc;
    __syncthreads();
    acc += R[(tid + 1) & (blockDim.x - 1)];
    __syncthreads();
  }
  unsigned long long c1 = clock64();
  if (tid == 0) { out[0] = c1 - c0; out[1] = acc; }
}

// Shared-load throughput: every warp issues R independent conflict-free loads of B bytes per lane (consecutive lanes,
// consecutive B-byte words), summed so none is dead.
template <int B>
__global__ void ldsbw(int nsteps, unsigned long long *out) {
  __shared__ __align__(16) uint32_t S[4 * 1024 + 64];
  const int tid = threadIdx.x, lane = tid & 31;
  for (int q = tid; q < 4 * 1024 + 64; q += blockDim.x) S[q] = q * 7;
  __syncthreads();
  uint32_t acc = 0;
  unsigned long long c0 = clock64();
  #pragma unroll 1
  for (int s = 0; s < nsteps; s++) {
    const int base = (s & 7) * 4;
    #pragma unroll
    for (int r = 0; r < 16; r++) {
      const int o = base + r * 128 * (B / 4) / 4;
      if (B == 4) acc += S[o + lane];
      else if (B == 8) { uint2 v = ((const uint2 *)(S + o))[lane]; acc += v.x ^ v.y; }
      else { uint4 v = ((const uint4 *)(S + o))[lane]; acc += v.x ^ v.y ^ v.z ^ v.w; }
    }
    acc = acc * 3 + 1;
  }
  unsigned long long c1 = clock64();
  if (tid == 0) out[0] = c1 - c0;
  if (acc == 0x12345678u) out[1] = acc;
}

template <int B>
int bw(unsigned long long *dout, int warps) {
  const int N = 20000;
  ldsbw<B><<<1, 32 * warps>>>(N, dout); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  ldsbw<B><<<1, 32 * warps>>>(N, dout); CK(cudaDeviceSynchronize());
  unsigned long long h; CK(cudaMemcpy(&h, dout, 8, cudaMemcpyDeviceToHost));
  printf("{\"probe\":\"lds_throughput\",\"bytes_per_lane\":%d,\"warps\":%d,\"cyc_per_warp_load\":%.2f}\n", B, warps,
         (double)h / N / 16 / warps);
  return 0;
}

template <int V>
int one(const uint32_t *dx, unsigned long long *dout, int warps, int k0) {
  const int N = 20000;
  k<V><<<1, 32 * warps>>>(dx, N, k0, dout);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  k<V><<<1, 32 * warps>>>(dx, N, k0, dout);
  CK(cudaDeviceSynchronize());
  unsigned long long h[2]; CK(cudaMemcpy(h, dout, 16, cudaMemcpyDeviceToHost));
  printf("{\"probe\":\"phase1\",\"variant\":%d,\"warps\":%d,\"k0\":%d,\"cyc_per_step\":%.1f}\n", V, warps, k0, (double)h[0] / N);
  return 0;
}

int main() {
  uint32_t hx[ND]; for (int i = 0; i < ND; i++) hx[i] = (uint32_t)(i * 2654435761u) & ((1u << 26) - 1);
  uint32_t *dx; unsigned long long *dout;
  CK(cudaMalloc(&dx, ND * 4)); CK(cudaMalloc(&dout, 16)); CK(cudaMemcpy(dx, hx, ND * 4, cudaMemcpyHostToDevice));
  for (int warps : {1, 4, 8, 16, 32})
    if (bw<4>(dout, warps) || bw<8>(dout, warps) || bw<16>(dout, warps)) return 1;
  for (int warps : {1, 2, 4, 8, 16})
    for (int k0 : {56, 300}) {
      if (one<0>(dx, dout, warps, k0) || one<1>(dx, dout, warps, k0) || one<2>(dx, dout, warps, k0) || one<3>(dx, dout, warps, k0)) return 1;
    }
  return 0;
}
